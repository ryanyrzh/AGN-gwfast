#!/usr/local/bin/python3
# import os
# os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
from time import time
import argparse
from pathlib import Path
from functools import partial
from multiprocessing import Pool, set_start_method
from collections import OrderedDict

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
    'R_orbit': 50, 'log10_M_lz': 4.0, 'src_pos': 0.5,
    'dL': 1.0, 'psi': 1, 'theta': 1.87, 'phi': 2.66,
}
# reference_parameters['log10_M_lz'] = 6.0
# reference_parameters['iota'] = 0.999 * np.pi / 2
reference_parameters_1 = reference_parameters.copy()
reference_parameters_2 = reference_parameters.copy()
reference_parameters_1['Mc'] = 30
reference_parameters_2['Mc'] = 80


def Jacobian_covariance(lensing_parameters):
    # 4.b Compute the Fisher
    fisher_matrix = HLV_AGN.FisherMatr(lensing_parameters, res=100)
    # 4.c Reduce and compute covar
    covar_matrix, _ = compute_covariance_matrix(fisher_matrix, cores=1)

    from_params = ['R_orbit', 'src_pos', 'log10_M_lz', 'iota', 'dL']
    transformed_cov_mat, transformed_parameters, transformed_keys = covariance_change_variable(
        covar_matrix, lensing_parameters, lensing_transform, from_params
    )
    # Sanity checks: key order must match matrix indexing
    assert transformed_cov_mat.shape[0] == transformed_cov_mat.shape[1] == len(transformed_keys)
    assert list(transformed_parameters.keys()) == list(transformed_keys)
    assert len(set(transformed_keys)) == len(transformed_keys), (
        f"Duplicate keys produced by transform: {transformed_keys}"
    )
    return transformed_cov_mat, transformed_parameters, transformed_keys


def direct_covariance(lensing_parameters):
    model_parameters = L1_AGN.convert_to_general_lensed_parameters(lensing_parameters)
    keys = list(model_parameters.keys()).copy()
    original_keys = keys.copy()

    lensed_HLV_fisher = HLV_Lensed.FisherMatr(model_parameters, res=200)

    # If some of the events is nan, then the reduce matrix won't work
    cleaned_lensed_HLV_fisher = np.nan_to_num(lensed_HLV_fisher, nan=0.0)

    lensed_fisher_mat, rm_keys = reduce_Fisher_matrix(cleaned_lensed_HLV_fisher, keys=keys)
    print("Fisher removed keys:", rm_keys)

    if lensed_fisher_mat.shape[0] == 0:
        n_params = len(original_keys)
        event_shape = lensed_HLV_fisher.shape[2:]
        nan_cov = np.full((n_params, n_params) + event_shape, np.nan)
        return nan_cov, model_parameters, original_keys

    lensed_fisher_mat[lensed_fisher_mat == 0.0] = np.nan
    lensed_cov_mats, _ = compute_covariance_matrix(lensed_fisher_mat, cores=1)
    return lensed_cov_mats, model_parameters, keys

def simple_lensing_covariance(lensing_parameters):
    model_parameters = L1_AGN.convert_to_general_lensed_parameters(lensing_parameters)
    keys = list(model_parameters.keys()).copy()
    original_keys = keys.copy()

    shape = model_parameters['Mc'].shape

    model_parameters['delta_iota'] = np.full(shape, 0.0)
    model_parameters['delta_phase'] = np.full(shape, 0.0)
    model_parameters['delta_psi'] = np.full(shape, 0.0)
    model_parameters['relative_mass'] = np.full(shape, 1.0)
    print("simple model parameters:", model_parameters)

    simple_HLV_fisher = HLV_Lensed.FisherMatr(model_parameters, res=200)
    # If some of the events is nan, then the reduce matrix won't work
    cleaned_simple_HLV_fisher = np.nan_to_num(simple_HLV_fisher, nan=0.0)

    simple_fisher_mat, rm_keys = reduce_Fisher_matrix(cleaned_simple_HLV_fisher, keys=keys)
    print("Fisher removed keys:", rm_keys)

    if simple_fisher_mat.shape[0] == 0:
        n_params = len(original_keys)
        event_shape = simple_HLV_fisher.shape[2:]
        nan_cov = np.full((n_params, n_params) + event_shape, np.nan)
        return nan_cov, model_parameters, original_keys

    simple_fisher_mat[simple_fisher_mat == 0.0] = np.nan
    simple_cov_mats, _ = compute_covariance_matrix(simple_fisher_mat, cores=1)
    return simple_cov_mats, model_parameters, keys

def lensing_transform(lensing_parameters):
    # A fiducial phase which does not affect the Jacobian results
    lensing_parameters = lensing_parameters.copy()
    lensing_parameters['phase'] = 0.0
    lensing_parameters['psi'] = 0.0
    if 'log10_M_lz' in lensing_parameters:
        lensing_parameters['M_lz'] = np.power(10.0, lensing_parameters.pop('log10_M_lz'))
    outputs = compute_lensed_angles_approx(lensing_parameters)
    # NOTE: `covariance_change_variable()` assumes a deterministic output order.
    # Use an OrderedDict with an explicit key order to avoid accidental reordering.
    relative_magification = outputs['sqrt_mu_p'] / outputs['sqrt_mu_m']
    relative_distance = relative_magification * integer_pow(
        (1 + outputs['z_rel_m']) / (1 + outputs['z_rel_p']), 2
    )
    relative_mass = (1 + outputs['z_rel_m']) / (1 + outputs['z_rel_p'])
    delta_iota = outputs['iota_m'] - outputs['iota_p']
    delta_phase = outputs['phase_m'] - outputs['phase_p']
    delta_time = outputs['delta_time']

    return OrderedDict([
        ('delta_time', delta_time),
        ('relative_distance', relative_distance),
        ('relative_mass', relative_mass),
        ('delta_iota', delta_iota),
        ('delta_phase', delta_phase),
        # Remove delta_psi bc it's very small and has been causing problems
        # ('delta_psi', outputs['psi_m'] - outputs['psi_p']),
    ])


def _batched_slogdet_stats(square_mat):
    """Return (first_sign, first_logdet, mean_logdet, min_logdet, max_logdet).

    Accepts a square matrix with optional batch dims (n, n, ...).
    """
    a = onp.asarray(square_mat)
    if a.ndim == 2:
        mats = a[None, :, :]
    else:
        # (n, n, ...batch...) -> (...batch..., n, n) -> (N, n, n)
        mats = onp.moveaxis(a, (0, 1), (-2, -1))
        mats = mats.reshape(-1, mats.shape[-2], mats.shape[-1])

    sign, logdet = onp.linalg.slogdet(mats)

    if onp.all(onp.isnan(logdet)):
        mean_ld = min_ld = max_ld = onp.nan
    else:
        mean_ld = onp.nanmean(logdet)
        min_ld = onp.nanmin(logdet)
        max_ld = onp.nanmax(logdet)
    return (
        float(sign[0]),
        float(logdet[0]),
        float(mean_ld),
        float(min_ld),
        float(max_ld),
    )

def reorder_covariance(cov, keys, desired_order):
    idx = [keys.index(k) for k in desired_order]
    idx_arr = np.array(idx, dtype=np.int32)
    # reorder rows then columns using jax.numpy.take
    assert cov.shape[0] == len(keys) == cov.shape[1]
    cov_reordered = np.take(np.take(cov.astype(np.float64), idx_arr, axis=0), idx_arr, axis=1)
    new_keys = [keys[i] for i in idx]
    return cov_reordered, new_keys


def _covariance_matrix_table(cov, labels, float_fmt='{: .6g}'):
    """Pretty-print a (batched) covariance with row/column parameter labels."""
    c = onp.asarray(cov)
    labels = list(labels)
    if c.ndim > 2:
        c = c[(slice(None), slice(None)) + (0,) * (c.ndim - 2)]
    if c.shape != (len(labels), len(labels)):
        return (
            f'(cannot tabulate: cov shape {onp.asarray(cov).shape}, '
            f'{len(labels)} labels)\n{onp.asarray(cov)}'
        )
    cells = [[float_fmt.format(float(c[i, j])) for j in range(c.shape[1])]
             for i in range(c.shape[0])]
    w_lab = max(len(s) for s in labels)
    w_col = [
        max(len(labels[j]), max(len(cells[i][j]) for i in range(c.shape[0])))
        for j in range(c.shape[1])
    ]
    pad = 2
    top = ' ' * (w_lab + pad) + ''.join(
        labels[j].ljust(w_col[j] + pad) for j in range(c.shape[1])
    )
    rows = [top]
    for i in range(c.shape[0]):
        row = labels[i].ljust(w_lab + pad) + ''.join(
            cells[i][j].rjust(w_col[j]).ljust(w_col[j] + pad)
            for j in range(c.shape[1])
        )
        rows.append(row)
    return '\n'.join(rows)


def _covariance_trailing_block_table(cov, keys, k=4, float_fmt='{: .6g}'):
    """Pretty-print the trailing k×k block (first batch slice if cov is batched)."""
    labels = list(keys)
    if len(labels) < k:
        return f'(need at least {k} parameters, have {len(labels)})'
    c = onp.asarray(cov)
    if c.ndim > 2:
        c = c[(slice(None), slice(None)) + (0,) * (c.ndim - 2)]
    if c.shape[0] < k or c.shape[1] < k:
        return f'(cov too small for {k}×{k} block: shape {c.shape})'
    return _covariance_matrix_table(c[-k:, -k:], labels[-k:], float_fmt=float_fmt)


def reorder_params_dict(params_dict, desired_order):
    return {k: np.array(params_dict[k]) for k in desired_order}

def get_bayes_factor(full_cov, full_params_dict, orig_keys, params_order, prior_widths):
    """Savage-Dickey density ratio: B_10 = p(psi_0|M1) / p(psi_0|d,M1)."""
    full_cov, _ = reorder_covariance(full_cov, orig_keys, params_order)
    full_params_dict = reorder_params_dict(full_params_dict, params_order)

    n_extra = prior_widths.shape[0]

    # Offset of fiducial values from null hypothesis (rm=1, di=dp=dps=0)
    offset = np.array([
        full_params_dict['relative_mass'] - 1.0,
        full_params_dict['delta_iota'],
        full_params_dict['delta_phase'],
        full_params_dict['delta_psi'],
    ])

    full_cov = np.moveaxis(full_cov, -1, 0)   # (n, D, D)
    offset = np.moveaxis(offset, -1, 0)        # (n, k)

    # Marginal posterior covariance of the extra parameters
    extra_cov = full_cov[:, -n_extra:, -n_extra:]  # (n, k, k)

    extra_cov_inv = np.linalg.inv(extra_cov)
    chi2 = np.einsum('...i,...ij,...j->...', offset, extra_cov_inv, offset)
    log_det_extra = np.linalg.slogdet(extra_cov)[1]

    log_posterior_at_null = (
        -0.5 * n_extra * np.log(2 * np.pi)
        - 0.5 * log_det_extra
        - 0.5 * chi2
    )
    log_prior_at_null = -np.sum(np.log(prior_widths))

    # B_10 = prior / posterior  (full model favoured when B_10 > 1)
    log_B = log_prior_at_null - log_posterior_at_null
    return log_B, chi2


def get_bayes_factor_capped(full_cov, full_params_dict, orig_keys, params_order, prior_widths):
    """SDDR Bayes factor with the log-determinant capped to prevent the
    non-physical upturn at large dL.

    Two caps are applied and whichever is stronger wins:
      1. Marginal cap: each marginal sigma is capped at the prior width.
         Prevents the Gaussian posterior from exceeding the prior bounds.
      2. Turning-point cap: the log-det is frozen once chi2 drops below k.
         Prevents the Occam factor from growing past the point where the
         posterior volume growth outpaces the chi2 decay.
    """
    full_cov, _ = reorder_covariance(full_cov, orig_keys, params_order)
    full_params_dict = reorder_params_dict(full_params_dict, params_order)

    n_extra = prior_widths.shape[0]

    offset = np.array([
        full_params_dict['relative_mass'] - 1.0,
        full_params_dict['delta_iota'],
        full_params_dict['delta_phase'],
        # full_params_dict['delta_psi'],
    ])

    full_cov = np.moveaxis(full_cov, -1, 0)   # (n, D, D)
    offset = np.moveaxis(offset, -1, 0)        # (n, k)

    extra_cov = full_cov[:, -n_extra:, -n_extra:]  # (n, k, k)

    extra_cov_inv = np.linalg.inv(extra_cov)
    chi2 = np.einsum('...i,...ij,...j->...', offset, extra_cov_inv, offset)
    log_det_extra = np.linalg.slogdet(extra_cov)[1]

    # Cap 1: marginal sigmas vs prior widths (per-direction)
    marginal_sigmas = np.sqrt(np.diagonal(extra_cov, axis1=-2, axis2=-1))
    marginal_correction = 2.0 * np.sum(
        np.minimum(0.0, np.log(prior_widths) - np.log(marginal_sigmas)),
        axis=-1,
    )

    # Cap 2: turning-point cap (freeze log-det once chi2 < k)
    # In the toy model, log_det at the turning point equals
    # log_det_current + k * ln(chi2/k).  When chi2 < k this is negative,
    # pulling log_det back to its turning-point value.
    turning_correction = n_extra * np.minimum(0.0, np.log(chi2 / n_extra))

    log_det_capped = log_det_extra + np.minimum(marginal_correction,
                                                 turning_correction)
    print(f'===== log_det_capped =====: {log_det_capped}')

    log_posterior_at_null = (
        -0.5 * n_extra * np.log(2 * np.pi)
        - 0.5 * log_det_capped
        - 0.5 * chi2
    )
    log_prior_at_null = -np.sum(np.log(prior_widths))

    log_B = log_prior_at_null - log_posterior_at_null
    return log_B, chi2


def worker(Ry_tuple_sublist, model='agn', n_newton=5):
    input_len = Ry_tuple_sublist.shape[0]
    shape = (input_len)
    lensing_parameters_1 = {key: np.full(shape, val).astype(np.float64) for key, val in reference_parameters_1.items()}
    lensing_parameters_1['R_orbit'] = Ry_tuple_sublist[:, 0]
    lensing_parameters_1['src_pos'] = Ry_tuple_sublist[:, 1]

    lensing_parameters_2 = {key: np.full(shape, val).astype(np.float64) for key, val in reference_parameters_2.items()}
    lensing_parameters_2['R_orbit'] = Ry_tuple_sublist[:, 0]
    lensing_parameters_2['src_pos'] = Ry_tuple_sublist[:, 1]

    if model == 'agn':
        covar_func = Jacobian_covariance
        net = HLV_AGN
    elif model == 'generic':
        covar_func = direct_covariance
        net = HLV_Lensed

    order = [
        'Mc', 'eta', 'phase',
        'chi1z', 'chi2z', 'tcoal',
        'psi', 'theta', 'phi',
        'delta_time', 'relative_distance', 'relative_mass', 'delta_iota', 'delta_phase', 
        # 'delta_psi'
    ]

    relative_mass_prior_width = 1.0
    delta_iota_prior_width = 0.5
    delta_phase_prior_width = 0.5
    # delta_psi_prior_width = 0.5
    prior_widths = np.array([relative_mass_prior_width, delta_iota_prior_width, delta_phase_prior_width, 
                            # delta_psi_prior_width
                            ])

    target_B = 1e2
    log_target_B = np.log(target_B)

    cov_1, pd_1, keys = covar_func(lensing_parameters_1)
    cov_2, pd_2, _ = covar_func(lensing_parameters_2)

    # Print the covariance block used by SDDR: the trailing (n_extra x n_extra)
    # block after reordering to `order`.
    n_extra = int(prior_widths.shape[0])
    cov_1_reordered, keys_reordered = reorder_covariance(cov_1, keys, order)
    c1 = onp.asarray(cov_1_reordered)
    batch_note = ''
    if c1.ndim > 2:
        batch_note = f' (first slice of {c1.shape[2:]} batch)'
    print(f'===== cov_1 SDDR extra block{batch_note} =====')
    print(_covariance_trailing_block_table(cov_1_reordered, keys_reordered, k=n_extra))

    extra_block = cov_1_reordered[-n_extra:, -n_extra:]
    first_sign, first_logdet, mean_logdet, min_logdet, max_logdet = _batched_slogdet_stats(extra_block)
    first_det = first_sign * onp.exp(first_logdet)
    print(
        '===== cov_1 extra block det/logdet =====: '
        f'det(first)={first_det:+.6e}, logdet(first)={first_logdet:+.6g}, sign(first)={first_sign:+.0f}; '
        f'logdet(mean/min/max)={mean_logdet:+.6g}/{min_logdet:+.6g}/{max_logdet:+.6g}'
    )

    if model == 'agn':
        orig_snr_1 = net.SNR(lensing_parameters_1, res=1000)
        orig_snr_2 = net.SNR(lensing_parameters_2, res=1000)
    elif model == 'generic':
        orig_snr_1 = net.SNR(pd_1, res=1000)
        orig_snr_2 = net.SNR(pd_2, res=1000)

    ref_dL_1 = lensing_parameters_1['dL'].copy()
    ref_dL_2 = lensing_parameters_2['dL'].copy()

    # --- Newton method with capped marginals ---
    bayes_func = get_bayes_factor_capped

    lnB_1, chi2_1 = bayes_func(cov_1, pd_1, keys, order, prior_widths)
    lnB_2, chi2_2 = bayes_func(cov_2, pd_2, keys, order, prior_widths)
    logB_1 = lnB_1 / np.log(10)
    logB_2 = lnB_2 / np.log(10)
    target_logB = np.log10(target_B)

    scale_1 = 10 ** ((logB_1 - target_logB) / (chi2_1 - prior_widths.shape[0]))
    scale_2 = 10 ** ((logB_2 - target_logB) / (chi2_2 - prior_widths.shape[0]))
    print(f'initial logB: mean {np.nanmean(logB_1):.4f} / {np.nanmean(logB_2):.4f}')
    print(f'initial chi2: mean {np.nanmean(chi2_1):.4f} / {np.nanmean(chi2_2):.4f},  '
          f'min {np.nanmin(chi2_1):.4f} / {np.nanmin(chi2_2):.4f},  '
          f'max {np.nanmax(chi2_1):.4f} / {np.nanmax(chi2_2):.4f}')
    print(f'initial scale: mean {np.nanmean(scale_1):.6f} / {np.nanmean(scale_2):.6f}')

    n_loops = n_newton
    def snr_loop(old_parameters, scale, target_logB, label=''):
        new_parameters = old_parameters.copy()
        new_parameters['dL'] *= scale
        try:
            new_cov, new_pd, ks = covar_func(new_parameters)
            new_lnB, new_chi2 = bayes_func(new_cov, new_pd, ks, order, prior_widths)
            new_logB = new_lnB / np.log(10)
            new_scale = 10 ** ((new_logB - target_logB) / (new_chi2 - prior_widths.shape[0]))
            mask = ~np.isnan(new_chi2)
            print(f'  [{label}] logB:  mean {np.nanmean(new_logB):.4f},  '
                  f'min {np.nanmin(new_logB):.4f},  max {np.nanmax(new_logB):.4f}')
            print(f'  [{label}] chi2:  mean {np.mean(new_chi2[mask]):.4f},  '
                  f'min {np.min(new_chi2[mask]):.4f},  max {np.max(new_chi2[mask]):.4f}')
            print(f'  [{label}] dL:    mean {np.nanmean(new_parameters["dL"]):.4f},  '
                  f'min {np.nanmin(new_parameters["dL"]):.4f},  max {np.nanmax(new_parameters["dL"]):.4f}')
        except ValueError:
            print(f'  [{label}] covar_func failed, returning nans')
            new_scale = np.full_like(scale, np.nan)
        return new_scale, new_parameters

    looped = 0
    while looped < n_loops:
        scale_1, lensing_parameters_1 = snr_loop(lensing_parameters_1, scale_1, target_logB, label=f'Mc30 step{looped+1}')
        scale_2, lensing_parameters_2 = snr_loop(lensing_parameters_2, scale_2, target_logB, label=f'Mc80 step{looped+1}')
        frac_1 = scale_1
        frac_2 = scale_2
        looped += 1
        frac_1 = frac_1[~np.isnan(frac_1)]
        frac_2 = frac_2[~np.isnan(frac_2)]
        print(f'Loop {looped:d} scale v.s. 1: {np.mean(frac_1):.6f} and {np.mean(frac_2):.6f}, std: {np.std(frac_1):.8f} and {np.std(frac_2):.8f}')

    result_snr_1 = orig_snr_1 / lensing_parameters_1['dL'] * ref_dL_1
    result_snr_2 = orig_snr_2 / lensing_parameters_2['dL'] * ref_dL_2

    return result_snr_1, result_snr_2

if __name__ == '__main__':
    set_start_method('spawn', force=True) # Multiprocessing
    args = parser.parse_args() # Get args
    n_y = args.ny # Get n_y
    n_R = args.nR # Get n_R
    cores = args.cores # Get cores
    model = args.model # Get model
    n_newton = 20
    label = f'tdays-logMlz'

    tic = time()
    # 1. Prepare matrix of (y, R)
    y_Eins_array = np.linspace(0.1, 1, n_y)  # in Einstein radii
    R_orbit_array = np.geomspace(10, 2e3, n_R)
    R_orbit_mesh, y_Eins_mesh = np.meshgrid(R_orbit_array, y_Eins_array, indexing='xy')
    y_Rorbit_mesh = convert_y_from_Einstein_to_Rorbit(y_Eins_mesh, R_orbit_mesh)
    Ry_tuple_list = np.vstack([R_orbit_mesh.flatten(), y_Rorbit_mesh.flatten()]).T

    # Custom settings go here
    the_worker = partial(worker, model=model, n_newton=n_newton)

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
    log10_snr = np.log10(snr_grid_1)
    ax.set_facecolor('lightgrey')
    cmap = plt.cm.plasma.copy()
    cmap.set_bad(color='lightgrey')
    nans = np.isnan(log10_snr)
    centre = np.mean(log10_snr[~nans])
    _min, _max = np.min(log10_snr[~nans]), np.max(log10_snr[~nans])
    midpoint = (_max + _min) / 2
    half_range = (_max - _min) / 2 * 1.05
    norm = colors.TwoSlopeNorm(vmin=midpoint - half_range, vcenter=centre, vmax=midpoint + half_range)

    im = ax.pcolormesh(R_orbit_array, y_Eins_array, log10_snr, cmap=cmap, norm=norm, shading='nearest')
    for snr_grid, color in zip([snr_grid_1, snr_grid_2], ['black', 'white']):
        log10_snr = np.log10(snr_grid)
        cont_snrs = [10, 50, 100]
        cont = ax.contour(R_orbit_array, y_Eins_array, log10_snr, colors=[color], levels=cont_snrs)
        labels = {lvl: f'{snr:d}' for lvl, snr in zip(cont.levels, cont_snrs)}
        ax.clabel(cont, fmt=labels, fontsize=10)
    ax.tick_params(which='both', direction='out')

    ax.set_xscale('log')
    ax.set_xlabel(r'$R_{\rm orbit}\,/\,R_S$')
    ax.set_ylabel(r'$y\,\equiv\,\beta\,/\,\theta_{\rm E}$')
    ax.set_title(r'$\rho$ required for $B > 100$')
    fig.colorbar(im, ax=ax, label=r'$\log_{10}(\rho_{\rm opt})$')
    fig.savefig(f'plots/snr_threshold_sddr_{args.model}_{label}_Ry_plot.pdf')
