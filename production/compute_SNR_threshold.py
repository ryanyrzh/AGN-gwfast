#!/usr/local/bin/python3
# import os
# os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
from time import time
import argparse
from pathlib import Path
from functools import partial
from multiprocessing import Pool, set_start_method

import numpy as onp
from jax import config
import jax.numpy as np
from jax.lax import integer_pow
config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
from matplotlib import colors

from gwfast.gwfastGlobals import detectors as det_dict, detPath
import gwfast.waveforms as waveforms
from gwfast.detector import Detector
import gwfast.network as network
from gwfast.signals import AGNLensedGWSignal, GeneralLensedGWSignal
from gwfast.lensing_utils import (
    compute_lensed_angles_approx,
    convert_y_from_Einstein_to_Rorbit
)
from gwfast.fisherTools import (
    reduce_Fisher_matrix,
    compute_covariance_matrix,
    covariance_change_variable
)

parser = argparse.ArgumentParser(description='Input control.')
parser.add_argument('--nR', type=int, required=True,
                    help='Number of cells in orbital radius (R_orbit).')
parser.add_argument('--ny', type=int, required=True,
                    help='Number of cells in source position (y).')
parser.add_argument('--cores', type=int, default=4,
                    help='Number of cores to use.')
parser.add_argument('--model', type=str, default='agn',
                    choices=['agn', 'generic'],
                    help='Which model to use for covariance calculation.')

# Set up detectors
H1 = Detector('H1', **det_dict['H1'],
              noise_curve_path=Path(detPath)/'observing_scenarios_paper/AplusDesign.txt')
L1 = Detector('L1', **det_dict['L1'],
              noise_curve_path=Path(detPath)/'observing_scenarios_paper/AplusDesign.txt')
V1 = Detector('V1', **det_dict['Virgo'],
              noise_curve_path=Path(detPath)/'observing_scenarios_paper/avirgo_O5low_NEW.txt')

wf_model = waveforms.IMRPhenomD()

H1_AGN = AGNLensedGWSignal(wf_model=wf_model, detector=H1, fmin=10)
L1_AGN = AGNLensedGWSignal(wf_model=wf_model, detector=L1, fmin=10)
V1_AGN = AGNLensedGWSignal(wf_model=wf_model, detector=V1, fmin=10)
HLV_AGN = network.DetNet({'H1': H1_AGN, 'L1': L1_AGN, 'V1': V1_AGN})

H1_Lensed = GeneralLensedGWSignal(wf_model=wf_model, detector=H1, fmin=10)
L1_Lensed = GeneralLensedGWSignal(wf_model=wf_model, detector=L1, fmin=10)
V1_Lensed = GeneralLensedGWSignal(wf_model=wf_model, detector=V1, fmin=10)
HLV_Lensed = network.DetNet({'H1': H1_Lensed, 'L1': L1_Lensed, 'V1': V1_Lensed})

reference_parameters = {
    'Mc': 30, 'eta': 0.24, 'iota': 0.99*np.pi/2, 'phase': 2,
    'chi1z': 0.3, 'chi2z': 0.5, 'tcoal': 0,
    'R_orbit': 100, 'M_lz': 1e4, 'src_pos': 0.5,
    'dL': 1, 'psi': 4, 'theta': 1.87, 'phi': 2.66,
}
# reference_parameters['M_lz'] = 1e6
# reference_parameters['iota'] = 0.999 * np.pi / 2
reference_parameters['Mc'] = 50


def Jacobian_covariance(lensing_parameters):
    # 4.b Compute the Fisher
    fisher_matrix = HLV_AGN.FisherMatr(lensing_parameters, res=100)
    # 4.c Reduce and compute covar
    covar_matrix, _ = compute_covariance_matrix(fisher_matrix, cores=1)

    from_params = ['iota', 'R_orbit', 'src_pos', 'M_lz', 'dL']
    transformed_cov_mat, transformed_parameters, transformed_keys = covariance_change_variable(
        covar_matrix, lensing_parameters, lensing_transform, from_params
    )
    return transformed_cov_mat, transformed_parameters, transformed_keys


def direct_covariance(lensing_parameters):
    model_parameters = L1_AGN.convert_to_general_lensed_parameters(lensing_parameters)
    keys = list(model_parameters.keys()).copy()

    lensed_HLV_fisher = HLV_Lensed.FisherMatr(model_parameters, res=200)

    # If some of the events is nan, then the reduce matrix won't work
    cleaned_lensed_HLV_fisher = np.nan_to_num(lensed_HLV_fisher, nan=0.0)

    lensed_fisher_mat, rm_keys = reduce_Fisher_matrix(cleaned_lensed_HLV_fisher, keys=keys)
    print(rm_keys)
    lensed_fisher_mat[lensed_fisher_mat == 0.0] = np.nan
    lensed_cov_mats, _ = compute_covariance_matrix(lensed_fisher_mat, cores=1)
    return lensed_cov_mats, model_parameters, keys


def lensing_transform(lensing_parameters):
    # A fiducial phase which does not affect the Jacobian results
    lensing_parameters['phase'] = 0.0
    lensing_parameters['psi'] = 0.0
    outputs = compute_lensed_angles_approx(lensing_parameters)
    phenom_changes = {}
    phenom_changes['delta_iota'] = outputs['iota_m'] - outputs['iota_p']
    phenom_changes['delta_phi'] = outputs['phase_m'] - outputs['phase_p']

    # (Radial gravitational potential is cancelled)
    phenom_changes['relative_mass'] = (1 + outputs['z_rel_m']) / (1 + outputs['z_rel_p'])
    relative_magification = outputs['sqrt_mu_p'] / outputs['sqrt_mu_m']
    phenom_changes['relative_distance'] = \
        relative_magification * integer_pow((1 + outputs['z_rel_m']) / (1 + outputs['z_rel_p']), 2)
    phenom_changes['delta_time'] = outputs['delta_time']
    return phenom_changes


def worker(Ry_tuple_sublist, model='agn', loop=2, actual_snr=False):
    '''
    Performance comment (2025/08/21)
    * Looping is recommended, but one loop is sufficient to bring the
        fractional difference between SNR and target SNR to below 1e-3
    * Computing the actual SNR at the end is not needed. The typical fractional
        error between the actual SNR and the one obtained from scaling is
        of order 1e-5 (or less).
    '''
    input_len = Ry_tuple_sublist.shape[0]
    shape = (input_len)
    lensing_parameters = {key: np.full(shape, val).astype(np.float64) for key, val in reference_parameters.items()}
    lensing_parameters['R_orbit'] = Ry_tuple_sublist[:, 0]
    lensing_parameters['src_pos'] = Ry_tuple_sublist[:, 1]

    if model == 'agn':
        covar_func = Jacobian_covariance
        network = HLV_AGN
    elif model == 'generic':
        covar_func = direct_covariance
        network = HLV_Lensed
    covariance_mat, params_dict, keys = covar_func(lensing_parameters)

    key_variable = 'relative_mass'
    key_idx = keys.index(key_variable)
    std = onp.sqrt(covariance_mat[key_idx, key_idx], dtype='float64')
    mean = params_dict['relative_mass']
    ln_mean = np.log(mean)
    std_ln = std / mean
    target_std = np.abs((ln_mean - 0))/ 3
    scale = std_ln / target_std

    if model == 'agn':
        orig_snr = network.SNR(lensing_parameters, res=1000)
    elif model == 'generic':
        orig_snr = network.SNR(params_dict, res=1000)
    result_snr = orig_snr / scale

    if loop:
        def snr_loop(old_parameters, scale, target_std):
            new_parameters = old_parameters.copy()
            new_parameters['dL'] /= scale
            new_covariance_mat, _, _ = covar_func(new_parameters)
            new_std = onp.sqrt(new_covariance_mat[key_idx, key_idx], dtype='float64')
            new_std /= mean
            new_scale = new_std / target_std
            return new_scale, new_parameters

        looped = 0
        while looped < loop:
            scale, lensing_parameters = snr_loop(lensing_parameters, scale, target_std)
            frac = 1 - scale
            looped += 1
            frac = frac[~np.isnan(frac)]
            print(f'Loop {looped:d} v.s. target: {np.mean(frac):.6f}, std: {np.std(frac):.8f}')

        _result_snr = orig_snr / lensing_parameters['dL'] * reference_parameters['dL']
        print('(After - Before) loop', _result_snr - result_snr)

        if actual_snr:
            if model == 'agn':
                computed_snr = network.SNR(lensing_parameters, res=1000)
            elif model == 'generic':
                computed_snr = network.SNR(params_dict, res=1000)
            print('Actual v.s. Scaling (1 - Scaling/Actual):', 1 - computed_snr / _result_snr)
            return computed_snr
        else:
            return _result_snr


if __name__ == '__main__':
    set_start_method('spawn', force=True)
    args = parser.parse_args()
    n_y = args.ny
    n_R = args.nR
    cores = args.cores
    model = args.model
    label = 'Mc50'

    tic = time()
    # 1. Prepare matrix of (y, R)
    y_Eins_array = np.linspace(0.01, 1, n_y)  # in Einstein radii
    R_orbit_array = np.geomspace(10, 5000, n_R)
    R_orbit_mesh, y_Eins_mesh = np.meshgrid(R_orbit_array, y_Eins_array, indexing='xy')
    y_Rorbit_mesh = convert_y_from_Einstein_to_Rorbit(y_Eins_mesh, R_orbit_mesh)
    Ry_tuple_list = np.vstack([R_orbit_mesh.flatten(), y_Rorbit_mesh.flatten()]).T

    # Custom settings go here
    the_worker = partial(worker, model=model, loop=1, actual_snr=False)

    with Pool(cores) as p:
        results = list(p.map(the_worker, np.array_split(Ry_tuple_list, cores)))

    print('ny, nR, cores', n_y, n_R, cores)
    print('Elapsed Time (min):', (time() - tic) / 60)

    concat_result = np.concatenate(results, axis=0)
    snr_grid = concat_result.reshape((n_y, n_R))

    # Saving result for reproducibility
    print('Saving results')
    np.savez(f'output/result_y{n_y:d}_R{n_R:d}_{model}_{label}',
             y_Eins=y_Eins_mesh.flatten(),
             y_Rorb=y_Rorbit_mesh.flatten(),
             R_orbit=R_orbit_mesh.flatten(),
             snr=snr_grid.flatten()
             )

    print('Start plotting')
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4), constrained_layout=True)
    cmap = plt.cm.plasma_r
    cmap.set_bad(color='lightgrey')

    log10_snr = np.log10(snr_grid)
    nans = np.isnan(log10_snr)
    centre = np.mean(log10_snr[~nans])
    # Make the whole thing 5% larger
    _min, _max = np.min(log10_snr[~nans]), np.max(log10_snr[~nans])
    midpoint = (_max + _min) / 2
    half_range = (_max - _min) / 2 * 1.05
    norm = colors.TwoSlopeNorm(vmin=midpoint - half_range, vcenter=centre, vmax=midpoint + half_range)
    im = ax.pcolormesh(R_orbit_array, y_Eins_array, log10_snr, cmap=cmap, norm=norm, shading='gouraud')
    cont_snrs = [8, 15, 30, 50, 100]
    cont = ax.contour(R_orbit_array, y_Eins_array, log10_snr, colors=['white'], levels=onp.log10(cont_snrs))
    labels = {lvl: f'{snr:d}' for lvl, snr in zip(cont.levels, cont_snrs)}
    ax.clabel(cont, fmt=labels, fontsize=10)
    ax.tick_params(which='both', direction='out')
    ax.set_xscale('log')
    ax.set_xlabel(r'$R_{\rm orbit}\,/\,R_S$')
    ax.set_ylabel(r'$y\,\equiv\,\beta\,/\,\theta_{\rm E}$')
    ax.set_title(r'$\rho$ required for 0 to lie outside the $3\sigma$ region of $p(\ln({\cal M}_1/{\cal M}_2))$')
    fig.colorbar(im, ax=ax, label=r'$\log_{10}(\rho_{\rm opt})$')
    # fig.savefig('plots/test_contour.pdf')
    fig.savefig(f'plots/snr_threshold_{args.model}_{label}_Ry_plot.pdf')
