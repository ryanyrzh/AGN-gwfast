#!/usr/local/bin/python3
# import os
# os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
import os
os.environ['JAX_COMPILATION_CACHE_DIR'] = '/tmp/jax_cache_snr_threshold'
# Also set this so JAX doesn't refuse to cache unless it saves enough time:
os.environ['JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS'] = '0'

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
from matplotlib.lines import Line2D

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
parser.add_argument('--nMc', type=int, required=True,
                    help='Number of cells in chirp mass (Mc).')
parser.add_argument('--cores', type=int, default=4,
                    help='Number of cores to use.')
parser.add_argument('--model', type=str, default='agn',
                    choices=['agn', 'generic', 'agn_intrinsic'],
                    help='Which model to use for covariance calculation.')
parser.add_argument('--logB', type=float, required=True,
                    help='Threshold log10(B) value for the SDDR Bayes factor.')
parser.add_argument('--steps', type=int, required=True,
                    help='Number of Newton steps for the dL search.')
parser.add_argument('--label', type=str, default=None,
                    help='Optional label appended to output filenames.')

args = parser.parse_args()
n_Mc = args.nMc
n_R = args.nR
cores = args.cores
model = args.model
n_newton = args.steps
target_logB = args.logB
label = args.label

print('nMc, nR:', n_Mc, n_R)
print('Cores:', cores)
print('Model:', model)
print('Target logB:', target_logB)
print('Newton steps:', n_newton)
print('Label:', label)

reference_parameters = {
    'Mc': 30, 'eta': 0.24, 'iota': 0.99*np.pi/2, 'phase': 2,
    'chi1z': 0.3, 'chi2z': 0.5, 'tcoal': 0,
    'R_orbit': 50, 'log10_M_lz': 4.0, 'src_pos': 0.5,
    'dL': 1.0, 'psi': 1, 'theta': 1.87, 'phi': 2.66,
}
print('Reference parameters:', reference_parameters)
y_Eins_1 = 0.7
y_Eins_2 = 0.2


# Set up detectors
H1 = Detector('H1', **det_dict['H1'],
              noise_curve_path=Path(detPath)/'observing_scenarios_paper/AplusDesign.txt')
L1 = Detector('L1', **det_dict['L1'],
              noise_curve_path=Path(detPath)/'observing_scenarios_paper/AplusDesign.txt')
V1 = Detector('V1', **det_dict['Virgo'],
              noise_curve_path=Path(detPath)/'observing_scenarios_paper/avirgo_O5low_NEW.txt')
# # Try lower sensitivities
# H1 = Detector('H1', **det_dict['H1'],
#               noise_curve_path=Path(detPath)/'LVC_O1O2O3/O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt')
# L1 = Detector('L1', **det_dict['L1'],
#               noise_curve_path=Path(detPath)/'LVC_O1O2O3/O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt')
# V1 = Detector('V1', **det_dict['Virgo'],
#               noise_curve_path=Path(detPath)/'LVC_O1O2O3/O3-V1_sensitivity_strain_asd.txt')

wf_model = waveforms.IMRPhenomD()

H1_AGN = AGNLensedGWSignal(wf_model=wf_model, detector=H1, fmin=10)
L1_AGN = AGNLensedGWSignal(wf_model=wf_model, detector=L1, fmin=10)
V1_AGN = AGNLensedGWSignal(wf_model=wf_model, detector=V1, fmin=10)
HLV_AGN = network.DetNet({'H1': H1_AGN, 'L1': L1_AGN, 'V1': V1_AGN})

H1_Lensed = GeneralLensedGWSignal(wf_model=wf_model, detector=H1, fmin=10)
L1_Lensed = GeneralLensedGWSignal(wf_model=wf_model, detector=L1, fmin=10)
V1_Lensed = GeneralLensedGWSignal(wf_model=wf_model, detector=V1, fmin=10)
HLV_Lensed = network.DetNet({'H1': H1_Lensed, 'L1': L1_Lensed, 'V1': V1_Lensed})

reference_parameters_1 = reference_parameters.copy()
reference_parameters_2 = reference_parameters.copy()


def Jacobian_covariance(lensing_parameters):
    # 4.b Compute the Fisher
    fisher_matrix = HLV_AGN.FisherMatr(lensing_parameters, res=100)
    # 4.c Reduce and compute covar
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


def _newton_step_converged(scale, rtol):
    """True when max |scale - 1| over finite events is within rtol."""
    s = onp.asarray(scale).ravel()
    finite = onp.isfinite(s)
    if not onp.any(finite):
        return False
    max_err = onp.max(onp.abs(s[finite] - 1.0))
    return bool(max_err <= rtol)


def _newton_dL_scale(logB, chi2, target_logB, n_extra, damping=0.5):
    """Multiplicative Newton step in dL, damped to suppress limit cycles."""
    scale = 10 ** ((logB - target_logB) / (chi2 - n_extra))
    scale = np.clip(scale, 1e-3, 1e3)
    return 1.0 + damping * (scale - 1.0)


def _sigma_from_cov(cov, keys, key):
    idx = keys.index(key)
    c = onp.asarray(cov)
    if c.ndim > 2:
        val = onp.nanmean(c[idx, idx, ...])
    else:
        val = float(c[idx, idx])
    if val < 0 or onp.isnan(val):
        return onp.nan
    return float(onp.sqrt(val))

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


def worker(RMc_tuple_sublist, model='agn', n_newton=5, target_logB=2.0):
    input_len = RMc_tuple_sublist.shape[0]
    shape = (input_len)
    R_orbit_batch = RMc_tuple_sublist[:, 0]
    Mc_batch = RMc_tuple_sublist[:, 1]

    lensing_parameters_1 = {key: np.full(shape, val).astype(np.float64) for key, val in reference_parameters_1.items()}
    lensing_parameters_1['R_orbit'] = R_orbit_batch
    lensing_parameters_1['Mc'] = Mc_batch
    lensing_parameters_1['src_pos'] = convert_y_from_Einstein_to_Rorbit(y_Eins_1, R_orbit_batch)

    lensing_parameters_2 = {key: np.full(shape, val).astype(np.float64) for key, val in reference_parameters_2.items()}
    lensing_parameters_2['R_orbit'] = R_orbit_batch
    lensing_parameters_2['Mc'] = Mc_batch
    lensing_parameters_2['src_pos'] = convert_y_from_Einstein_to_Rorbit(y_Eins_2, R_orbit_batch)

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
    delta_iota_prior_width = np.pi/2
    delta_phase_prior_width = np.pi/2
    # delta_psi_prior_width = 0.5
    prior_widths = np.array([relative_mass_prior_width, delta_iota_prior_width, delta_phase_prior_width, 
                            # delta_psi_prior_width
                            ])

    target_B = 10 ** target_logB
    log_target_B = target_logB

    cov_1, pd_1, keys = covar_func(lensing_parameters_1)
    cov_2, pd_2, _ = covar_func(lensing_parameters_2)

    # Diagnostic: compare Jacobian vs direct sigmas at fiducial point
    if model == 'agn':
        direct_cov_1, direct_pd_1, direct_keys_1 = direct_covariance(lensing_parameters_1)
        sig_jac_dp = _sigma_from_cov(cov_1, keys, 'delta_phase')
        sig_dir_dp = _sigma_from_cov(direct_cov_1, direct_keys_1, 'delta_phase')
        sig_jac_di = _sigma_from_cov(cov_1, keys, 'delta_iota')
        sig_dir_di = _sigma_from_cov(direct_cov_1, direct_keys_1, 'delta_iota')
        sig_jac_rm = _sigma_from_cov(cov_1, keys, 'relative_mass')
        sig_dir_rm = _sigma_from_cov(direct_cov_1, direct_keys_1, 'relative_mass')
        print(
            '===== sigma comparison (Jacobian vs direct) =====\n'
            f'delta_phase: {sig_jac_dp:.6g} vs {sig_dir_dp:.6g}, ratio={sig_jac_dp/sig_dir_dp:.6g}\n'
            f'delta_iota:  {sig_jac_di:.6g} vs {sig_dir_di:.6g}, ratio={sig_jac_di/sig_dir_di:.6g}\n'
            f'rel_mass:    {sig_jac_rm:.6g} vs {sig_dir_rm:.6g}, ratio={sig_jac_rm/sig_dir_rm:.6g}'
        )

    # Print the extra covariance block
    n_extra = int(prior_widths.shape[0])
    cov_1_reordered, keys_reordered = reorder_covariance(cov_1, keys, order)
    c1 = onp.asarray(cov_1_reordered)
    batch_note = ''
    if c1.ndim > 2:
        batch_note = f' (first slice of {c1.shape[2:]} batch)'
    print(f'===== cov_1 extra block{batch_note} =====')
    print(_covariance_trailing_block_table(cov_1_reordered, keys_reordered, k=n_extra))

    extra_block = cov_1_reordered[-n_extra:, -n_extra:]
    first_sign, first_logdet, mean_logdet, min_logdet, max_logdet = _batched_slogdet_stats(extra_block)
    print(
        '===== cov_1 extra block det/logdet =====: '
        f'logdet(first)={first_logdet:+.6g}, sign(first)={first_sign:+.0f}; '
        f'logdet(mean/min/max)={mean_logdet:+.6g}/{min_logdet:+.6g}/{max_logdet:+.6g}'
    )

    # Print minimum eigenvalue of the trailing block and the injected R_orbit
    # sampled across several batch slices (R_orbit varies along the batch)
    eb = onp.asarray(extra_block)
    R_orbit_all = onp.asarray(lensing_parameters_1['R_orbit']).ravel()
    if eb.ndim == 2:
        mats = eb[None, :, :]
    else:
        # (n, n, ...batch...) -> (N, n, n)
        mats = onp.moveaxis(eb, (0, 1), (-2, -1)).reshape(-1, eb.shape[0], eb.shape[1])
    n_slices = min(4, mats.shape[0])
    sample_idx = onp.linspace(0, mats.shape[0] - 1, n_slices).round().astype(int)
    print('===== cov_1 extra block min eigenvalue / injected R_orbit =====')
    for i in sample_idx:
        m = mats[i]
        if not onp.all(onp.isfinite(m)):
            min_eig = onp.nan
        else:
            try:
                min_eig = float(onp.min(onp.linalg.eigvalsh(m)))
            except onp.linalg.LinAlgError:
                min_eig = onp.nan
        R_orbit_inj = float(R_orbit_all[i]) if i < R_orbit_all.size else onp.nan
        print(f'  slice {i}: min_eig={min_eig:+.6g}, R_orbit={R_orbit_inj:.6g}')

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

    newton_damping = 0.5
    scale_1 = _newton_dL_scale(logB_1, chi2_1, target_logB, n_extra, newton_damping)
    scale_2 = _newton_dL_scale(logB_2, chi2_2, target_logB, n_extra, newton_damping)
    print(f'initial logB: mean {np.nanmean(logB_1):.4f} / {np.nanmean(logB_2):.4f}')
    print(f'initial chi2: mean {np.nanmean(chi2_1):.4f} / {np.nanmean(chi2_2):.4f},  '
          f'min {np.nanmin(chi2_1):.4f} / {np.nanmin(chi2_2):.4f},  '
          f'max {np.nanmax(chi2_1):.4f} / {np.nanmax(chi2_2):.4f}')
    print(f'initial scale: mean {np.nanmean(scale_1):.6f} / {np.nanmean(scale_2):.6f}')

    n_loops = n_newton
    newton_converge_rtol = 0.03
    newton_converge_steps = 3
    def snr_loop(old_parameters, scale, target_logB, label=''):
        new_parameters = old_parameters.copy()
        new_parameters['dL'] *= scale
        try:
            new_cov, new_pd, ks = covar_func(new_parameters)
            new_lnB, new_chi2 = bayes_func(new_cov, new_pd, ks, order, prior_widths)
            new_logB = new_lnB / np.log(10)
            new_scale = _newton_dL_scale(
                new_logB, new_chi2, target_logB, n_extra, newton_damping,
            )
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
    consecutive_converged = 0
    while looped < n_loops:
        scale_1, lensing_parameters_1 = snr_loop(lensing_parameters_1, scale_1, target_logB, label=f'y{y_Eins_1:g} step{looped+1}')
        scale_2, lensing_parameters_2 = snr_loop(lensing_parameters_2, scale_2, target_logB, label=f'y{y_Eins_2:g} step{looped+1}')
        frac_1 = scale_1
        frac_2 = scale_2
        looped += 1
        frac_1 = frac_1[~np.isnan(frac_1)]
        frac_2 = frac_2[~np.isnan(frac_2)]
        print(f'Loop {looped:d} scale v.s. 1: {np.mean(frac_1):.6f} and {np.mean(frac_2):.6f}, std: {np.std(frac_1):.8f} and {np.std(frac_2):.8f}')

        if (_newton_step_converged(scale_1, newton_converge_rtol)
                and _newton_step_converged(scale_2, newton_converge_rtol)):
            consecutive_converged += 1
            if consecutive_converged >= newton_converge_steps:
                print(
                    f'Newton early stop after {looped} steps '
                    f'(3 consecutive with max |scale-1| <= {newton_converge_rtol})'
                )
                break
        else:
            consecutive_converged = 0

    result_snr_1 = orig_snr_1 / lensing_parameters_1['dL'] * ref_dL_1
    result_snr_2 = orig_snr_2 / lensing_parameters_2['dL'] * ref_dL_2

    return result_snr_1, result_snr_2

if __name__ == '__main__':
    set_start_method('spawn', force=True)

    tic = time()
    # 1. Prepare matrix of (Mc, R)
    Mc_array = np.linspace(10, 100, n_Mc)
    R_orbit_array = np.geomspace(10, 2e3, n_R)
    R_orbit_mesh, Mc_mesh = np.meshgrid(R_orbit_array, Mc_array, indexing='xy')
    RMc_tuple_list = np.vstack([R_orbit_mesh.flatten(), Mc_mesh.flatten()]).T

    # Custom settings go here
    the_worker = partial(worker, model=model, n_newton=n_newton, target_logB=target_logB)

    with Pool(cores) as p:
        results = p.map(the_worker, np.array_split(RMc_tuple_list, cores))
        results_1, results_2 = zip(*results)
        results_1 = list(results_1)
        results_2 = list(results_2)

    print('Elapsed Time (min):', (time() - tic) / 60)

    concat_result_1 = np.concatenate(results_1, axis=0)
    concat_result_2 = np.concatenate(results_2, axis=0)
    snr_grid_1 = concat_result_1.reshape((n_Mc, n_R))
    snr_grid_2 = concat_result_2.reshape((n_Mc, n_R))

    # Saving result for reproducibility
    print('Saving results')
    _label_suffix = f'_{label}' if label is not None else ''
    _logB_str = f'{target_logB:g}'
    _job_id = os.environ.get('SLURM_JOB_ID', '')
    _job_suffix = f'_{_job_id}' if _job_id else ''
    np.savez(f'output/result_Mc{n_Mc:d}_R{n_R:d}_sddr_{model}_logB{_logB_str}_{n_newton}steps{_label_suffix}{_job_suffix}',
             Mc=Mc_mesh.flatten(),
             R_orbit=R_orbit_mesh.flatten(),
             y_Eins_1=y_Eins_1,
             y_Eins_2=y_Eins_2,
             snr_1=snr_grid_1.flatten(),
             snr_2=snr_grid_2.flatten()
             )

    print('Start plotting')
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4), constrained_layout=True)
    log10_snr = np.log10(snr_grid_1)
    ax.set_facecolor('lightgrey')
    cmap = plt.cm.plasma.copy()
    cmap.set_bad(color='lightgrey')
    centre = np.nanmean(log10_snr)
    _min, _max = np.nanmin(log10_snr), np.nanmax(log10_snr)
    midpoint = (_max + _min) / 2
    half_range = (_max - _min) / 2 * 1.05
    norm = colors.TwoSlopeNorm(vmin=midpoint - half_range, vcenter=midpoint, vmax=midpoint + half_range)

    im = ax.pcolormesh(R_orbit_array, Mc_array, log10_snr, cmap=cmap, norm=norm, shading='nearest')
    legend_handles = []
    for snr_grid, color, contour_label in zip(
            [snr_grid_1, snr_grid_2],
            ['black', 'white'],
            [rf'$y = {y_Eins_1:g}$', rf'$y = {y_Eins_2:g}$']):
        log10_snr = np.log10(snr_grid)
        cont_snrs = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        cont = ax.contour(R_orbit_array, Mc_array, log10_snr, colors=[color], levels=cont_snrs)
        labels = {lvl: f'{snr}' for lvl, snr in zip(cont.levels, cont_snrs)}
        ax.clabel(cont, fmt=labels, fontsize=10)
        legend_handles.append(Line2D([0], [0], color=color, linewidth=1.5, label=contour_label))
    ax.legend(handles=legend_handles, loc='lower left', fontsize=10)
    ax.tick_params(which='both', direction='in', colors='white', labelcolor='black')

    ax.set_xscale('log')
    ax.set_xlabel(r'$R_{\rm orbit}\,/\,R_S$')
    ax.set_ylabel(r'$\mathcal{M}_c\,/\,M_\odot$')
    # ax.set_title(f'$\\rho$ required for $\\log B > {target_logB}$')
    fig.colorbar(im, ax=ax, label=r'$\log_{10}(\rho_{\rm req})$')
    fig.savefig(f'plots/sddr_{model}_logB{_logB_str}_{n_newton}steps{_label_suffix}{_job_suffix}_RMcplot.pdf')
