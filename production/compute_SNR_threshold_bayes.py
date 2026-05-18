#!/usr/local/bin/python3
# import os
# os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
from time import time
import argparse
from pathlib import Path
from functools import partial
from multiprocessing import Pool, set_start_method

import numpy as onp
from jax import config, jacfwd
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
    covariance_change_variable,
    covariance_change_variable_1
)

parser = argparse.ArgumentParser(description='Input control.')
parser.add_argument('--nR', type=int, required=True,
                    help='Number of cells in orbital radius (R_orbit).')
parser.add_argument('--ny', type=int, required=True,
                    help='Number of cells in source position (y).')
parser.add_argument('--cores', type=int, default=4,
                    help='Number of cores to use.')
parser.add_argument('--model', type=str, default='agn',
                    choices=['agn', 'generic', 'agn_intrinsic'],
                    help='Which model to use for covariance calculation.')
parser.add_argument('--jacobian-diagnostics', action='store_true',
                    help='Print Jacobian singular values/condition numbers and exit.')
parser.add_argument('--log10-mlz-prior-sigma', type=float, default=1.0,
                    help='Gaussian prior sigma on log10_M_lz (dex); set <=0 to disable.')

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
    'Mc': 30.0, 'eta': 0.24, 'iota': 0.99*np.pi/2, 'phase': 2.0,
    'chi1z': 0.3, 'chi2z': 0.5, 'tcoal': 0,
    'R_orbit': 50, 'log10_M_lz': 4.0, 'src_pos': 0.5,
    'dL': 1.0, 'psi': 1.0, 'theta': 1.87, 'phi': 2.66,
}
# reference_parameters['M_lz'] = 1e6
# reference_parameters['iota'] = 0.999 * np.pi / 2
reference_parameters_1 = reference_parameters.copy()
reference_parameters_2 = reference_parameters.copy()
reference_parameters_1['Mc'] = 30.0
reference_parameters_2['Mc'] = 80.0


# def Jacobian_covariance(lensing_parameters):
#     # 4.b Compute the Fisher
#     fisher_matrix = HLV_AGN.FisherMatr(lensing_parameters, res=100)
#     # 4.c Reduce and compute covar
#     covar_matrix, _ = compute_covariance_matrix(fisher_matrix, cores=1)

#     # lensing_transform is a 6-to-6 transform
#     from_params = ['iota', 'R_orbit', 'src_pos', 'M_lz', 'dL']
#     transformed_cov_mat, transformed_parameters, transformed_keys = covariance_change_variable(
#         covar_matrix, lensing_parameters, lensing_transform, from_params
#     )
#     return transformed_cov_mat, transformed_parameters, transformed_keys

def _apply_log10_mlz_prior(fisher_matrix, keys, log10_sigma, mlz_values=None):
    if log10_sigma is None or log10_sigma <= 0:
        return fisher_matrix

    if 'log10_M_lz' in keys:
        idx = keys.index('log10_M_lz')
        prior_val = 1.0 / (log10_sigma ** 2)
        fisher_matrix = onp.array(fisher_matrix, copy=True)
        if fisher_matrix.ndim == 2:
            fisher_matrix[idx, idx] += prior_val
        else:
            fisher_matrix[idx, idx, ...] += prior_val
        return fisher_matrix

    if 'M_lz' in keys and mlz_values is not None:
        idx = keys.index('M_lz')
        mlz_vals = onp.asarray(mlz_values, dtype=onp.float64)
        if fisher_matrix.ndim == 2:
            mlz_vals = float(mlz_vals)
        sigma_mlz = onp.log(10.0) * mlz_vals * log10_sigma
        fisher_matrix = onp.array(fisher_matrix, copy=True)
        if fisher_matrix.ndim == 2:
            fisher_matrix[idx, idx] += 1.0 / (sigma_mlz ** 2)
        else:
            fisher_matrix[idx, idx, ...] += 1.0 / (sigma_mlz ** 2)
    return fisher_matrix


def Jacobian_covariance(lensing_parameters, log10_mlz_prior_sigma=None):
    # 4.b Compute the Fisher
    fisher_matrix = HLV_AGN.FisherMatr(lensing_parameters, res=100)
    fisher_keys = list(lensing_parameters.keys())
    fisher_matrix = _apply_log10_mlz_prior(
        fisher_matrix, fisher_keys, log10_mlz_prior_sigma
    )
    
    # Diagnostics: original determinants
    flat_fisher_mat = onp.asarray(
        fisher_matrix.reshape(fisher_matrix.shape[0], fisher_matrix.shape[1], -1),
        dtype=onp.float64,
    )
    fisher_evt_first = onp.moveaxis(flat_fisher_mat, -1, 0)
    fisher_sign, fisher_logdet = onp.linalg.slogdet(fisher_evt_first)
    print('fisher slogdet sign:', fisher_sign)
    print('fisher slogdet logdet:', fisher_logdet)
    
    # 4.c Reduce and compute covar
    covar_matrix, _ = compute_covariance_matrix(fisher_matrix, cores=1)
    
    flat_covar_mat = onp.asarray(
        covar_matrix.reshape(covar_matrix.shape[0], covar_matrix.shape[1], -1),
        dtype=onp.float64,
    )
    covar_evt_first = onp.moveaxis(flat_covar_mat, -1, 0)
    covar_sign, covar_logdet = onp.linalg.slogdet(covar_evt_first)
    print('covar slogdet sign:', covar_sign)
    print('covar slogdet logdet:', covar_logdet)

    # original 5-to-5 transform
    from_params = ['R_orbit', 'src_pos', 'log10_M_lz', 'iota', 'dL']
    transformed_cov_mat, transformed_parameters, transformed_keys = covariance_change_variable(
        covar_matrix, lensing_parameters, lensing_transform, from_params
    )

    # Diagnostics: transformed determinants
    flat_trans_cov = onp.asarray(
        transformed_cov_mat.reshape(transformed_cov_mat.shape[0], transformed_cov_mat.shape[1], -1),
        dtype=onp.float64,
    )
    trans_evt_first = onp.moveaxis(flat_trans_cov, -1, 0)
    trans_sign, trans_logdet = onp.linalg.slogdet(trans_evt_first)
    print('transformed cov slogdet sign:', trans_sign)
    print('transformed cov slogdet logdet:', trans_logdet)
    # Force symmetrize
    trans_sym = 0.5 * (trans_evt_first + onp.swapaxes(trans_evt_first, -1, -2))
    trans_min_eig = onp.linalg.eigvalsh(trans_sym)[..., 0]
    print('transformed cov min eig:', trans_min_eig)
    return transformed_cov_mat, transformed_parameters, transformed_keys


def direct_covariance(lensing_parameters, log10_mlz_prior_sigma=None):
    model_parameters = L1_AGN.convert_to_general_lensed_parameters(lensing_parameters)
    keys = list(model_parameters.keys()).copy()

    lensed_HLV_fisher = HLV_Lensed.FisherMatr(model_parameters, res=200)
    lensed_HLV_fisher = _apply_log10_mlz_prior(
        lensed_HLV_fisher, keys, log10_mlz_prior_sigma, model_parameters.get('M_lz')
    )

    # If some of the events is nan, then the reduce matrix won't work
    cleaned_lensed_HLV_fisher = np.nan_to_num(lensed_HLV_fisher, nan=0.0)

    lensed_fisher_mat, rm_keys = reduce_Fisher_matrix(cleaned_lensed_HLV_fisher, keys=keys)
    lensed_fisher_mat[lensed_fisher_mat == 0.0] = np.nan
    lensed_cov_mats, _ = compute_covariance_matrix(lensed_fisher_mat, cores=1)
    return lensed_cov_mats, model_parameters, keys

def simple_lensing_covariance(lensing_parameters, log10_mlz_prior_sigma=None):
    model_parameters = L1_AGN.convert_to_general_lensed_parameters(lensing_parameters)
    keys = list(model_parameters.keys()).copy()

    shape = model_parameters['Mc'].shape

    model_parameters['delta_iota'] = np.full(shape, 0.0)
    model_parameters['delta_phase'] = np.full(shape, 0.0)
    model_parameters['delta_psi'] = np.full(shape, 0.0)
    model_parameters['relative_mass'] = np.full(shape, 1.0)

    simple_HLV_fisher = HLV_Lensed.FisherMatr(model_parameters, res=200)
    simple_HLV_fisher = _apply_log10_mlz_prior(
        simple_HLV_fisher, keys, log10_mlz_prior_sigma, model_parameters.get('M_lz')
    )
    # If some of the events is nan, then the reduce matrix won't work
    cleaned_simple_HLV_fisher = np.nan_to_num(simple_HLV_fisher, nan=0.0)

    simple_fisher_mat, rm_keys = reduce_Fisher_matrix(cleaned_simple_HLV_fisher, keys=keys)
    print('Fisher removed keys:', rm_keys)
    simple_fisher_mat[simple_fisher_mat == 0.0] = np.nan
    simple_cov_mats, _ = compute_covariance_matrix(simple_fisher_mat, cores=1)
    return simple_cov_mats, model_parameters, keys

def lensing_transform(lensing_parameters):
    # A fiducial phase which does not affect the Jacobian results
    # lensing_parameters['iota'] = reference_parameters['iota'] # temporary fix
    lensing_parameters['phase'] = 0.0
    lensing_parameters['psi'] = 0.0
    # lensing_parameters['dL'] = reference_parameters['dL'] # temporary fix
    if 'log10_M_lz' in lensing_parameters:
        lensing_parameters['M_lz'] = np.power(10.0, lensing_parameters.pop('log10_M_lz'))
    outputs = compute_lensed_angles_approx(lensing_parameters)
    phenom_changes = {}
    phenom_changes['delta_iota'] = outputs['iota_m'] - outputs['iota_p']
    phenom_changes['delta_phase'] = outputs['phase_m'] - outputs['phase_p']
    # phenom_changes['delta_psi'] = outputs['psi_m'] - outputs['psi_p'] # comment out in original transform

    # (Radial gravitational potential is cancelled)
    relative_magification = outputs['sqrt_mu_p'] / outputs['sqrt_mu_m']
    phenom_changes['relative_distance'] = relative_magification * integer_pow((1 + outputs['z_rel_m']) / (1 + outputs['z_rel_p']), 2)
    phenom_changes['relative_mass'] = (1 + outputs['z_rel_m']) / (1 + outputs['z_rel_p'])
    phenom_changes['delta_time'] = outputs['delta_time']

    return phenom_changes


def run_jacobian_diagnostics(x0=None, phase=0.3, psi=0.7):
    if x0 is None:
        x0 = np.array([
            reference_parameters_1['R_orbit'],
            reference_parameters_1['src_pos'],
            reference_parameters_1['log10_M_lz'],
            reference_parameters_1['iota'],
            reference_parameters_1['dL'],
        ], dtype=np.float64)

    input_names = ['R_orbit', 'src_pos', 'log10_M_lz', 'iota', 'dL']
    output_names_map = {
        'raw': [
            'iota_p', 'iota_m',
            'phase_p', 'phase_m',
            'z_rel_p', 'z_rel_m',
            'sqrt_mu_p', 'sqrt_mu_m',
            'delta_time',
        ],
        'full': [
            'delta_iota', 'delta_phase',
            'relative_distance', 'relative_mass',
            'delta_time',
        ],
    }

    def pack(params_vec):
        return {
            'R_orbit': params_vec[0],
            'src_pos': params_vec[1],
            'log10_M_lz': params_vec[2],
            'iota': params_vec[3],
            'dL': params_vec[4],
            'phase': phase,
            'psi': psi,
        }

    def full_transform_vec(params_vec):
        outputs = lensing_transform(pack(params_vec))
        return np.array([
            outputs['delta_iota'],
            outputs['delta_phase'],
            outputs['relative_distance'],
            outputs['relative_mass'],
            outputs['delta_time'],
        ], dtype=np.float64)

    def raw_outputs_vec(params_vec):
        params = pack(params_vec)
        params['M_lz'] = np.power(10.0, params.pop('log10_M_lz'))
        outputs = compute_lensed_angles_approx(params)
        return np.array([
            outputs['iota_p'], outputs['iota_m'],
            outputs['phase_p'], outputs['phase_m'],
            outputs['z_rel_p'], outputs['z_rel_m'],
            outputs['sqrt_mu_p'], outputs['sqrt_mu_m'],
            outputs['delta_time'],
        ], dtype=np.float64)

    for name, func in [('raw', raw_outputs_vec), ('full', full_transform_vec)]:
        jac = jacfwd(func)(x0)
        jac_np = onp.asarray(jac, dtype=onp.float64)
        u_mat, sing_vals_np, vh_mat = onp.linalg.svd(jac_np, full_matrices=False)
        cond = sing_vals_np[0] / sing_vals_np[-1] if sing_vals_np[-1] != 0 else onp.inf
        print(f'{name} singular values:', sing_vals_np)
        print(f'{name} condition number:', cond)

        smallest_right = vh_mat[-1, :]
        print(f'{name} smallest right singular vector (inputs):')
        for key, val in zip(input_names, smallest_right):
            print(f'  {key}: {val:+.6e}')

        output_names = output_names_map[name]
        smallest_left = u_mat[:, -1]
        print(f'{name} smallest left singular vector (outputs):')
        for key, val in zip(output_names, smallest_left):
            print(f'  {key}: {val:+.6e}')

def reorder_covariance(cov, keys, desired_order):
    idx = [keys.index(k) for k in desired_order]
    idx_arr = np.array(idx, dtype=np.int32)
    # reorder rows then columns using jax.numpy.take
    assert cov.shape[0] == len(keys) == cov.shape[1]
    cov_reordered = np.take(np.take(cov.astype(np.float64), idx_arr, axis=0), idx_arr, axis=1)
    new_keys = [keys[i] for i in idx]
    return cov_reordered, new_keys

def reorder_params_dict(params_dict, desired_order):
    return {k: np.array(params_dict[k]) for k in desired_order}

def get_bayes_factor(full_cov, full_params_dict, simple_cov, simple_params_dict, 
                     orig_keys, simple_keys, params_order, simple_params_order, 
                     prior_widths):
    full_cov, _ = reorder_covariance(full_cov, orig_keys, params_order)
    simple_cov, _ = reorder_covariance(simple_cov, simple_keys, simple_params_order)
    full_params_dict = reorder_params_dict(full_params_dict, params_order)
    simple_params_dict = reorder_params_dict(simple_params_dict, simple_params_order)

    n_extra_params = prior_widths.shape[0] # delta_iota, delta_phase, (delta_psi), relative_mass
    n_simple_params = full_cov.shape[0] - n_extra_params # 13 simple lensing parameters
    simple_cov = simple_cov[:n_simple_params, :n_simple_params]
    cross_block = full_cov[:n_simple_params, n_simple_params:]

    relative_mass_offset = full_params_dict['relative_mass'] - 1.0
    delta_iota_offset = full_params_dict['delta_iota']
    delta_phase_offset = full_params_dict['delta_phase']
    # Use for old transform:
    offset = np.array([relative_mass_offset, delta_iota_offset, delta_phase_offset])
    # Use for new transform:
    # delta_psi_offset = full_params_dict['delta_psi']
    # offset = np.array([relative_mass_offset, delta_iota_offset, delta_phase_offset, delta_psi_offset])

    # make axis order compatible with linalg operations
    full_cov = np.moveaxis(full_cov, -1, 0) # (n, 17, 17)
    simple_cov = np.moveaxis(simple_cov, -1, 0) # (n, 13, 13)
    cross_block = np.moveaxis(cross_block, -1, 0) # (n, 13, 4)
    offset = np.moveaxis(offset, -1, 0) # (n, 4)

    correction = np.einsum('...ik,...k->...i', -np.linalg.inv(simple_cov), 
                         np.einsum('...ij,...j->...i', cross_block, offset)
                         ) # (n, 13)

    exp_term = np.einsum('...i,...i->...', correction,
                         np.einsum('...ij,...j->...i',
                                    simple_cov,
                                    correction)
                         ) # (n,)
    print('exp_term:', exp_term)

    prior_product = np.prod(prior_widths, axis=0)

    full_cov_det = np.linalg.det(full_cov)
    simple_cov_det = np.linalg.det(simple_cov)
    print('full_cov_det:', full_cov_det)
    print('simple_cov_det:', simple_cov_det)    
    det_ratio = np.sqrt(full_cov_det / simple_cov_det)
    print('det_ratio:', det_ratio)

    B = (2*np.pi)**(-n_extra_params/2) * det_ratio * exp_term * prior_product
    # This definition in Heavens (2016) is the inverse of the standard defintion,
    # where positive logB favours the more complicated model
    return 1 / B



def worker(Ry_tuple_sublist, model='agn', loop=2, actual_snr=False,
           log10_mlz_prior_sigma=None):
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
    lensing_parameters_1 = {key: np.full(shape, val).astype(np.float64) for key, val in reference_parameters_1.items()}
    lensing_parameters_1['R_orbit'] = Ry_tuple_sublist[:, 0]
    lensing_parameters_1['src_pos'] = Ry_tuple_sublist[:, 1]

    lensing_parameters_2 = {key: np.full(shape, val).astype(np.float64) for key, val in reference_parameters_2.items()}
    lensing_parameters_2['R_orbit'] = Ry_tuple_sublist[:, 0]
    lensing_parameters_2['src_pos'] = Ry_tuple_sublist[:, 1]

    if model == 'agn':
        covar_func = partial(Jacobian_covariance, log10_mlz_prior_sigma=log10_mlz_prior_sigma)
        network = HLV_AGN
    elif model == 'generic':
        covar_func = partial(direct_covariance, log10_mlz_prior_sigma=log10_mlz_prior_sigma)
        network = HLV_Lensed
    covariance_mat_1, params_dict_1, full_keys = covar_func(lensing_parameters_1)
    covariance_mat_2, params_dict_2, _ = covar_func(lensing_parameters_2)
    try:
        simple_cov_mats_1, simple_params_dict_1, simple_keys = simple_lensing_covariance(
            lensing_parameters_1, log10_mlz_prior_sigma=log10_mlz_prior_sigma
        )
        simple_cov_mats_2, simple_params_dict_2, _ = simple_lensing_covariance(
            lensing_parameters_2, log10_mlz_prior_sigma=log10_mlz_prior_sigma
        )
    except ValueError:
        print('simple_lensing_covariance failed, returning nans')
        return np.full(shape, np.nan), np.full(shape, np.nan)

    # # Use this for generic model
    # order = [
    #     'Mc', 'eta', 'iota', 'phase',
    #     'chi1z', 'chi2z', 'tcoal',
    #     'dL', 'psi', 'theta', 'phi',
    #     'delta_time', 'relative_distance', 'relative_mass', 'delta_iota', 'delta_phase', 'delta_psi'
    # ]
    
    # Use this for old AGN model
    order = [
        'Mc', 'eta', 'phase',
        'chi1z', 'chi2z', 'tcoal',
        'psi', 'theta', 'phi',
        'delta_time', 'relative_distance', 'relative_mass', 'delta_iota', 'delta_phase'
    ]
    simple_order = [
    'Mc', 'eta', 'phase',
    'chi1z', 'chi2z', 'tcoal',
    'psi', 'theta', 'phi',
    'delta_time', 'relative_distance'
    ]

    # # Use this for new AGN model
    # order = [
    #     'Mc', 'eta', 'iota', 'phase',
    #     'chi1z', 'chi2z', 'tcoal',
    #     'dL', 'psi', 'theta', 'phi',
    #     'delta_time', 'relative_distance', 'relative_mass', 'delta_iota', 'delta_phase', 'delta_psi'
    # ]

    # # Prior widths for generic model
    # relative_mass_prior_width = 1.0
    # delta_iota_prior_width = np.pi
    # delta_phase_prior_width = np.pi
    # delta_psi_prior_width = np.pi
    # prior_widths = np.array([relative_mass_prior_width, delta_iota_prior_width, delta_phase_prior_width, delta_psi_prior_width])

    # Prior widths for old agn model
    R_orbit_prior_width = 50.
    src_pos_prior_width = 0.1
    log10_M_lz_prior_width = 4
    prior_widths = np.array([R_orbit_prior_width, src_pos_prior_width, log10_M_lz_prior_width])

    B_1 = get_bayes_factor(covariance_mat_1, params_dict_1, simple_cov_mats_1, simple_params_dict_1, 
                           full_keys, simple_keys, order, simple_order, 
                           prior_widths)
    B_2 = get_bayes_factor(covariance_mat_2, params_dict_2, simple_cov_mats_2, simple_params_dict_2, 
                            full_keys, simple_keys, order, simple_order, 
                            prior_widths)
    logB_1, logB_2 = np.log10(B_1), np.log10(B_2)

    target_logB = 3
    scale_1 = logB_1 - target_logB
    scale_2 = logB_2 - target_logB
    print('initial scale:', scale_1, scale_2)

    if model == 'agn':
        orig_snr_1 = network.SNR(lensing_parameters_1, res=1000)
        orig_snr_2 = network.SNR(lensing_parameters_2, res=1000)
    elif model == 'generic':
        orig_snr_1 = network.SNR(params_dict_1, res=1000)
        orig_snr_2 = network.SNR(params_dict_2, res=1000)
    result_snr_1 = orig_snr_1 + scale_1
    result_snr_2 = orig_snr_2 + scale_2

    if loop:
        def snr_loop(old_parameters, scale, target_logB):
            new_parameters = old_parameters.copy()
            new_parameters['dL'] += scale
            try:
                new_covariance_mat, new_params_dict, keys = covar_func(new_parameters)
                new_simple_cov_mats, new_simple_params_dict, _ = simple_lensing_covariance(
                    new_parameters, log10_mlz_prior_sigma=log10_mlz_prior_sigma
                )
                new_B = get_bayes_factor(new_covariance_mat, new_params_dict,
                                        new_simple_cov_mats, new_simple_params_dict,
                                        full_keys, simple_keys, order, simple_order, prior_widths)
                new_logB = np.log10(new_B)
                new_scale = new_logB - target_logB
            except ValueError:
                print('simple_lensing_covariance failed, returning nans')
                new_scale = np.full_like(scale, np.nan)
            return new_scale, new_parameters

        looped = 0
        while looped < loop:
            scale_1, lensing_parameters_1 = snr_loop(lensing_parameters_1, scale_1, target_logB)
            scale_2, lensing_parameters_2 = snr_loop(lensing_parameters_2, scale_2, target_logB)
            # frac_1 = 1 - scale_1
            # frac_2 = 1 - scale_2
            # looped += 1
            # frac_1 = frac_1[~np.isnan(frac_1)]
            # frac_2 = frac_2[~np.isnan(frac_2)]
            # print(f'Loop {looped:d} v.s. target: {np.mean(frac_1):.6f} and {np.mean(frac_2):.6f}, std: {np.std(frac_1):.8f} and {np.std(frac_2):.8f}')
            frac_1 = scale_1
            frac_2 = scale_2
            looped += 1
            frac_1 = frac_1[~np.isnan(frac_1)]
            frac_2 = frac_2[~np.isnan(frac_2)]
            print(f'Loop {looped:d} v.s. target: {np.mean(frac_1):.6f} and {np.mean(frac_2):.6f}, std: {np.std(frac_1):.8f} and {np.std(frac_2):.8f}')

        _result_snr_1 = orig_snr_1 / lensing_parameters_1['dL'] * reference_parameters_1['dL']
        _result_snr_2 = orig_snr_2 / lensing_parameters_2['dL'] * reference_parameters_2['dL']
        print('(After - Before) loop', _result_snr_1 - result_snr_1, _result_snr_2 - result_snr_2)

        if actual_snr:
            if model == 'agn':
                computed_snr_1 = network.SNR(lensing_parameters_1, res=1000)
                computed_snr_2 = network.SNR(lensing_parameters_2, res=1000)
            elif model == 'generic':
                computed_snr_1 = network.SNR(params_dict_1, res=1000)
                computed_snr_2 = network.SNR(params_dict_2, res=1000)
            print('Actual v.s. Scaling (1 - Scaling/Actual):', 1 - computed_snr_1 / _result_snr_1, 1 - computed_snr_2 / _result_snr_2)
            return computed_snr_1, computed_snr_2
        else:
            return _result_snr_1, _result_snr_2


if __name__ == '__main__':
    set_start_method('spawn', force=True) # Multiprocessing
    args = parser.parse_args() # Get args
    n_y = args.ny # Get n_y
    n_R = args.nR # Get n_R
    cores = args.cores # Get cores
    model = args.model # Get model
    label = f'tdays-logMlz'
    log10_mlz_prior_sigma = args.log10_mlz_prior_sigma

    if args.jacobian_diagnostics:
        run_jacobian_diagnostics()
        raise SystemExit(0)

    tic = time()
    # 1. Prepare matrix of (y, R)
    y_Eins_array = np.linspace(0.4, 2, n_y)  # in Einstein radii
    R_orbit_array = np.geomspace(10, 500, n_R)
    R_orbit_mesh, y_Eins_mesh = np.meshgrid(R_orbit_array, y_Eins_array, indexing='xy')
    y_Rorbit_mesh = convert_y_from_Einstein_to_Rorbit(y_Eins_mesh, R_orbit_mesh)
    Ry_tuple_list = np.vstack([R_orbit_mesh.flatten(), y_Rorbit_mesh.flatten()]).T

    # Custom settings go here
    the_worker = partial(
        worker, model=model, loop=3, actual_snr=False,
        log10_mlz_prior_sigma=log10_mlz_prior_sigma
    )

    with Pool(cores) as p:
        results = p.map(the_worker, np.array_split(Ry_tuple_list, cores))
        results_1, results_2 = zip(*results)
        results_1 = list(results_1)
        results_2 = list(results_2)

    print('ny, nR, cores', n_y, n_R, cores)
    print('Elapsed Time (min):', (time() - tic) / 60)

    concat_result_1 = np.concatenate(results_1, axis=0)
    concat_result_2 = np.concatenate(results_2, axis=0)
    snr_grid_1 = concat_result_1.reshape((n_y, n_R))
    snr_grid_2 = concat_result_2.reshape((n_y, n_R))

    # Saving result for reproducibility
    print('Saving results')
    np.savez(f'output/result_y{n_y:d}_R{n_R:d}_{model}_{label}',
             y_Eins=y_Eins_mesh.flatten(),
             y_Rorb=y_Rorbit_mesh.flatten(),
             R_orbit=R_orbit_mesh.flatten(),
             snr_1=snr_grid_1.flatten(),
             snr_2=snr_grid_2.flatten()
             )

    print('Start plotting')
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4), constrained_layout=True)
    # Color plot for 1 of 2 datasets
    log10_snr = np.log10(snr_grid_1)
    cmap = plt.cm.plasma
    cmap.set_bad(color='lightgrey')
    nans = np.isnan(log10_snr)
    centre = np.mean(log10_snr[~nans])
    # Make the whole thing 5% larger
    _min, _max = np.min(log10_snr[~nans]), np.max(log10_snr[~nans])
    midpoint = (_max + _min) / 2
    half_range = (_max - _min) / 2 * 1.05
    norm = colors.TwoSlopeNorm(vmin=midpoint - half_range, vcenter=centre, vmax=midpoint + half_range)
    im = ax.pcolormesh(R_orbit_array, y_Eins_array, log10_snr, cmap=cmap, norm=norm, shading='gouraud')
    # Contour lines for both datasets
    for snr_grid, color in zip([snr_grid_1, snr_grid_2], ['black', 'white']):
        log10_snr = np.log10(snr_grid)
        cont_snrs = [8, 15, 30, 50, 100]
        cont = ax.contour(R_orbit_array, y_Eins_array, log10_snr, colors=[color], levels=onp.log10(cont_snrs))
        labels = {lvl: f'{snr:d}' for lvl, snr in zip(cont.levels, cont_snrs)}
        ax.clabel(cont, fmt=labels, fontsize=10)
    ax.tick_params(which='both', direction='out')
    ax.set_xscale('log')
    ax.set_xlabel(r'$R_{\rm orbit}\,/\,R_S$')
    ax.set_ylabel(r'$y\,\equiv\,\beta\,/\,\theta_{\rm E}$')
    ax.set_title(r'$\rho$ required for $\log B>3$')
    # ax.set_title(r'$\rho$ required for 0 to lie outside the $5\sigma$ region of $p(M_{Lz})$')
    fig.colorbar(im, ax=ax, label=r'$\log_{10}(\rho_{\rm opt})$')
    # fig.savefig('plots/test_contour.pdf')
    fig.savefig(f'plots/snr_threshold_bayes_{args.model}_{label}_Ry_plot.pdf')


    # # Now do a simpler example with no parallelisation and no grids. Just single value of (R, y) to test the code and understand the results. This is also useful for debugging and for understanding the behaviour of the Bayes factor as a function of SNR and lensing parameters. # Do NOT use worker function. Do NOT use the loop to adjust SNR. Just compute the Bayes factor for a single set of lensing parameters and a single SNR, and see how it compares to the target logB of 3. This will help us understand if the code is working as expected and if the Bayes factor is sensitive to the lensing parameters in the way we expect.
    # y_Eins_test = np.array([0.5])
    # R_orbit_test = np.array([50])
    # lensing_parameters_test = reference_parameters_1.copy()
    # lensing_parameters_test['R_orbit'] = R_orbit_test
    # lensing_parameters_test['src_pos'] = convert_y_from_Einstein_to_Rorbit(y_Eins_test, R_orbit_test)
    # # Convert all the lensing parameters to arrays (if not yet already) for compatibility with the covariance functions
    # for key in lensing_parameters_test.keys():
    #     if not isinstance(lensing_parameters_test[key], np.ndarray):
    #         lensing_parameters_test[key] = np.array([lensing_parameters_test[key]], dtype=np.float64)
    #     else:
    #         lensing_parameters_test[key] = lensing_parameters_test[key].astype(np.float64)
    # # Conv
    # covar_mat_test, params_dict_test, keys_test = Jacobian_covariance(lensing_parameters_test)
    # simple_cov_mats_test, simple_params_dict_test, simple_keys_test = simple_lensing_covariance(lensing_parameters_test)
    # B_test = get_bayes_factor(covar_mat_test, params_dict_test, simple_cov_mats_test, simple_params_dict_test, 
    #                           keys_test, simple_keys_test, order, simple_order, 
    #                           prior_widths)
    # logB_test = np.log10(B_test)
    # print('Test Bayes factor:', B_test)
    # print('Test log10(Bayes factor):', logB_test)
