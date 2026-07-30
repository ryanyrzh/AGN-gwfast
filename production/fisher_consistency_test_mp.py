#!/usr/bin/env python3
"""
Fisher consistency test from Vallisneri (2008) https://arxiv.org/abs/gr-qc/0703086
- Evaluate the mismatch |log r| between the exact waveform and its linear expansion
along randomly sampled directions on the 1σ-surface.
- Find SNR where 90% of the 1σ-surface has |log r| < 0.1
"""
import os
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', '/tmp/jax_cache_fisher_consistency')
os.environ.setdefault('JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS', '0')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl')

import argparse
from collections import OrderedDict
from contextlib import nullcontext
from multiprocessing import Pool, set_start_method
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as onp
from jax import config
import jax.numpy as np
from jax.lax import integer_pow

config.update('jax_enable_x64', True)

from gwfast.gwfastGlobals import detectors as det_dict, detPath
from gwfast.gwfastUtils import get_mass_parameters, noise_weighted_inner_product
import gwfast.waveforms as waveforms
from gwfast.detector import Detector
import gwfast.network as network
from gwfast.signals import AGNLensedGWSignal, GeneralLensedGWSignal
from gwfast.lensing_utils import compute_lensed_angles_approx, convert_y_from_Einstein_to_Rorbit
from gwfast.fisherTools import reduce_Fisher_matrix, compute_covariance_matrix, covariance_change_variable


REFERENCE_PARAMETERS = {
    'Mc': 30, 'eta': 0.24, 'iota': 0.99 * onp.pi / 2, 'phase': 2,
    'chi1z': 0.3, 'chi2z': 0.5, 'tcoal': 0,
    'R_orbit': 50, 'log10_M_lz': 4.0, 'src_pos': 0.5,
    'dL': 1.0, 'psi': 1, 'theta': 1.87, 'phi': 2.66,
}

PARAM_ORDER = [
    'Mc', 'eta', 'phase',
    'chi1z', 'chi2z', 'tcoal',
    'psi', 'theta', 'phi',
    'delta_time', 'relative_distance', 'relative_mass', 'delta_iota', 'delta_phase',
]

EXTRA_PARAMS = ['relative_mass', 'delta_iota', 'delta_phase']

_HLV_LENSED = None


def lensing_transform(lensing_parameters):
    lensing_parameters = lensing_parameters.copy()
    lensing_parameters['phase'] = 0.0
    lensing_parameters['psi'] = 0.0
    if 'log10_M_lz' in lensing_parameters:
        lensing_parameters['M_lz'] = np.power(10.0, lensing_parameters.pop('log10_M_lz'))
    outputs = compute_lensed_angles_approx(lensing_parameters)
    relative_magification = outputs['sqrt_mu_p'] / outputs['sqrt_mu_m']
    relative_distance = relative_magification * integer_pow(
        (1 + outputs['z_rel_m']) / (1 + outputs['z_rel_p']), 2
    )
    relative_mass = (1 + outputs['z_rel_m']) / (1 + outputs['z_rel_p'])
    return OrderedDict([
        ('delta_time', outputs['delta_time']),
        ('relative_distance', relative_distance),
        ('relative_mass', relative_mass),
        ('delta_iota', outputs['iota_m'] - outputs['iota_p']),
        ('delta_phase', outputs['phase_m'] - outputs['phase_p']),
    ])


def build_networks():
    h1 = Detector('H1', **det_dict['H1'],
                  noise_curve_path=Path(detPath) / 'observing_scenarios_paper/AplusDesign.txt')
    l1 = Detector('L1', **det_dict['L1'],
                  noise_curve_path=Path(detPath) / 'observing_scenarios_paper/AplusDesign.txt')
    v1 = Detector('V1', **det_dict['Virgo'],
                  noise_curve_path=Path(detPath) / 'observing_scenarios_paper/avirgo_O5low_NEW.txt')
    wf_model = waveforms.IMRPhenomD()
    h1_agn = AGNLensedGWSignal(wf_model=wf_model, detector=h1, fmin=10, verbose=False)
    l1_agn = AGNLensedGWSignal(wf_model=wf_model, detector=l1, fmin=10, verbose=False)
    v1_agn = AGNLensedGWSignal(wf_model=wf_model, detector=v1, fmin=10, verbose=False)
    hlv_agn = network.DetNet({'H1': h1_agn, 'L1': l1_agn, 'V1': v1_agn}, verbose=False)
    h1_lensed = GeneralLensedGWSignal(wf_model=wf_model, detector=h1, fmin=10, verbose=False)
    l1_lensed = GeneralLensedGWSignal(wf_model=wf_model, detector=l1, fmin=10, verbose=False)
    v1_lensed = GeneralLensedGWSignal(wf_model=wf_model, detector=v1, fmin=10, verbose=False)
    hlv_lensed = network.DetNet({'H1': h1_lensed, 'L1': l1_lensed, 'V1': v1_lensed}, verbose=False)
    return l1_agn, hlv_agn, hlv_lensed


def _init_worker_pool():
    global _HLV_LENSED
    _, _, _HLV_LENSED = build_networks()


def squeeze_covariance(cov):
    c = onp.asarray(cov, dtype=onp.float64)
    c = onp.squeeze(c)
    if c.ndim != 2:
        raise ValueError(f'Expected 2D covariance after squeeze, got shape {c.shape}')
    c = 0.5 * (c + c.T)
    return c


def covariance_block(cov, keys, block_keys):
    c = squeeze_covariance(cov) if onp.asarray(cov).ndim != 2 else onp.asarray(cov, dtype=onp.float64)
    c = 0.5 * (c + c.T)
    idx = [keys.index(k) for k in block_keys]
    return c[onp.ix_(idx, idx)]


def reorder_covariance(cov, keys, desired_order):
    present = [k for k in desired_order if k in keys]
    c = squeeze_covariance(onp.asarray(cov))
    idx = [keys.index(k) for k in present]
    cov_reordered = c[onp.ix_(idx, idx)]
    return cov_reordered, present


def jacobian_covariance(lensing_parameters, hlv_agn):
    fisher_matrix = hlv_agn.FisherMatr(lensing_parameters, res=100)
    covar_matrix, _ = compute_covariance_matrix(fisher_matrix, cores=1)
    from_params = ['R_orbit', 'src_pos', 'log10_M_lz', 'iota', 'dL']
    transformed_cov, transformed_parameters, transformed_keys = covariance_change_variable(
        covar_matrix, lensing_parameters, lensing_transform, from_params
    )
    return transformed_cov, transformed_parameters, transformed_keys


def direct_covariance(lensing_parameters, l1_agn, hlv_lensed, res):
    model_parameters = l1_agn.convert_to_general_lensed_parameters(lensing_parameters)
    fisher_keys = list(model_parameters.keys())
    keys_work = fisher_keys.copy()
    fisher = hlv_lensed.FisherMatr(model_parameters, res=res)
    fisher = np.nan_to_num(fisher, nan=0.0)
    fisher, rm_keys = reduce_Fisher_matrix(fisher, keys=keys_work)
    if fisher.shape[0] == 0:
        raise ValueError(f'Fisher matrix empty after reduction; removed keys: {rm_keys}')
    fisher[fisher == 0.0] = np.nan
    cov, _ = compute_covariance_matrix(fisher, cores=1)
    return cov, model_parameters, keys_work


def get_covariance(model, lensing_parameters, l1_agn, hlv_agn, hlv_lensed, res):
    if model == 'agn':
        return jacobian_covariance(lensing_parameters, hlv_agn)
    if model == 'generic':
        return direct_covariance(lensing_parameters, l1_agn, hlv_lensed, res)
    raise ValueError(f'Unknown model: {model}')


def extract_waveform_params(l1_lensed, model_parameters):
    valid = set(l1_lensed.strain_model_keys)
    out = {}
    for key in valid:
        if key not in model_parameters:
            continue
        val = model_parameters[key]
        if val is None:
            continue
        out[key] = float(np.asarray(val).reshape(-1)[0])
    return out


def build_injection(R_orbit, y_eins, Mc, dL):
    lp = {k: np.array([v], dtype=np.float64) for k, v in REFERENCE_PARAMETERS.items()}
    lp['Mc'] = np.array([Mc], dtype=np.float64)
    lp['dL'] = np.array([dL], dtype=np.float64)
    lp['R_orbit'] = np.array([R_orbit], dtype=np.float64)
    lp['src_pos'] = np.array([float(convert_y_from_Einstein_to_Rorbit(y_eins, R_orbit))], dtype=np.float64)
    return lp


def frequency_grid(signal, params, res):
    with_mass = get_mass_parameters(params)
    fcut = signal.wf_model.fcut(**with_mass)
    if signal.fmax is not None:
        fcut = np.where(fcut > signal.fmax, signal.fmax, fcut)
    fmin = onp.full(fcut.shape, signal.fmin)
    return np.geomspace(fmin, fcut, num=int(res))


def _strain_on_detector(signal, freqs, params, rot=0.0):
    return np.asarray(signal.GWstrain(freqs, params, rot=rot))


def _detector_inner_product(signal, freqs, strain_a, strain_b):
    psd = np.asarray(signal.detector.psd_interp(freqs))
    ip = noise_weighted_inner_product(
        onp.asarray(freqs), onp.asarray(strain_a), onp.asarray(strain_b), onp.asarray(psd),
    )
    return float(onp.asarray(ip).reshape(-1)[0])


def to_event_params(params):
    return {k: np.array([float(v)]) for k, v in params.items()}


def network_inner_product(hlv_lensed, freqs, strain_a_by_det, strain_b_by_det):
    total = 0.0
    for name, signal in hlv_lensed.signals.items():
        if signal.detector.shape == 'L':
            total += _detector_inner_product(
                signal, freqs,
                strain_a_by_det[name], strain_b_by_det[name],
            )
        else:
            for rot in (0.0, 60.0, 120.0):
                total += _detector_inner_product(
                    signal, freqs,
                    strain_a_by_det[(name, rot)], strain_b_by_det[(name, rot)],
                )
    return total


def collect_network_strains(hlv_lensed, freqs, params):
    event_params = to_event_params(params)
    strains = {}
    for name, signal in hlv_lensed.signals.items():
        if signal.detector.shape == 'L':
            strains[name] = _strain_on_detector(signal, freqs, event_params)
        else:
            for rot in (0.0, 60.0, 120.0):
                strains[(name, rot)] = _strain_on_detector(signal, freqs, event_params, rot=rot)
    return strains


def collect_param_derivatives(hlv_lensed, freqs, params, param_keys):
    derivs = {k: {} for k in param_keys}
    ordered = OrderedDict(to_event_params(params))
    for name, signal in hlv_lensed.signals.items():
        rots = (0.0,) if signal.detector.shape == 'L' else (0.0, 60.0, 120.0)
        for rot in rots:
            key = name if signal.detector.shape == 'L' else (name, rot)
            jdict = signal.signal_derivatives(
                freq_grid=freqs,
                parameters=ordered,
                rot=rot,
                computeAnalyticalDeriv=True,
            )
            for pk in param_keys:
                if pk not in jdict:
                    raise KeyError(f'Parameter {pk!r} missing from waveform derivatives')
                arr = onp.asarray(jdict[pk])
                # _jax_derivative returns (1, N_freq) for single-event (N_freq, 1) grids
                if arr.ndim == 2 and arr.shape[0] == 1:
                    col = arr[0, :]
                elif arr.ndim == 2 and arr.shape[1] == 1:
                    col = arr[:, 0]
                else:
                    col = onp.squeeze(arr)
                # Match GWstrain shape (N_freq, 1); (N_freq,) would broadcast to (N_freq, N_freq)
                derivs[pk][key] = col[:, onp.newaxis] if col.ndim == 1 else col
    return derivs


def _numpy_strain_dict(strains):
    return {k: onp.asarray(v) for k, v in strains.items()}


def _numpy_deriv_dict(derivs):
    return {pk: {k: onp.asarray(v) for k, v in det.items()} for pk, det in derivs.items()}


def linear_strain_from_derivatives(h0, derivs, delta_by_param):
    h_lin = {k: onp.array(v, copy=True) for k, v in h0.items()}
    for pk, dval in delta_by_param.items():
        for det_key in h_lin:
            h_lin[det_key] = h_lin[det_key] + dval * derivs[pk][det_key]
    return h_lin


def log_r_mismatch(hlv_lensed, freqs, h0, derivs, wf_params_perturbed, delta_by_param):
    h_exact = collect_network_strains(hlv_lensed, freqs, wf_params_perturbed)
    if any(not onp.isfinite(onp.asarray(s)).all() for s in h_exact.values()):
        return onp.nan
    h_lin = linear_strain_from_derivatives(h0, derivs, delta_by_param)
    v = {k: h_lin[k] - onp.asarray(h_exact[k]) for k in h0}
    return 0.5 * network_inner_product(hlv_lensed, freqs, v, v)


def sample_unit_directions(rng, n_dirs, ndim):
    dirs = rng.normal(size=(n_dirs, ndim))
    dirs /= onp.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs


def directions_on_sigma_surface(cov_block, n_dirs, rng):
    """Return (ndim, n_dirs) displacements with δθ^T cov^{-1} δθ = 1.
    """
    cov = onp.asarray(cov_block, dtype=onp.float64)
    lam, V = onp.linalg.eigh(cov)
    lam = onp.clip(lam, 0.0, None)
    if onp.min(lam) <= 0:
        raise ValueError(
            f'Non-positive eigenvalues in covariance block: min={onp.min(lam):.3e}'
        )
    u = sample_unit_directions(rng, n_dirs, cov.shape[0])
    sqrt_lam = onp.sqrt(lam)
    return (V * sqrt_lam) @ u.T


def apply_param_delta(base_params, block_keys, delta_vector):
    perturbed = {k: float(v) for k, v in base_params.items()}
    for key, dval in zip(block_keys, delta_vector):
        perturbed[key] = perturbed[key] + float(dval)
    return perturbed


def snr_at_dL(hlv_lensed, params, res):
    return float(np.asarray(hlv_lensed.SNR(to_event_params(params), res=res)).reshape(-1)[0])


def trust_snr_from_cdf(snr_values, cdf, threshold=0.9, log_r_cut=0.1):
    snr_values = onp.asarray(snr_values, dtype=onp.float64)
    cdf = onp.asarray(cdf, dtype=onp.float64)
    for snr, frac in zip(snr_values, cdf):
        if frac >= threshold:
            return float(snr)
    return onp.nan


def _worker_direction(task):
    delta_vec, wf_s, block_keys, h0, derivs, freqs = task
    delta_by_param = {key: float(delta_vec[j]) for j, key in enumerate(block_keys)}
    wf_perturbed = apply_param_delta(wf_s, block_keys, delta_vec)
    return log_r_mismatch(
        _HLV_LENSED, freqs, h0, derivs, wf_perturbed, delta_by_param,
    )


def _eval_log_r_values(pool, directions_s, wf_s, block_keys, h0, derivs, freqs, cores):
    h0_np = _numpy_strain_dict(h0)
    derivs_np = _numpy_deriv_dict(derivs)
    freqs_np = onp.asarray(freqs)
    tasks = [
        (directions_s[:, i_dir], wf_s, block_keys, h0_np, derivs_np, freqs_np)
        for i_dir in range(directions_s.shape[1])
    ]
    if cores <= 1:
        return onp.array([_worker_direction(task) for task in tasks])
    return onp.array(pool.map(_worker_direction, tasks))


def main():
    parser = argparse.ArgumentParser(
        description='Vallisneri Fisher consistency (maximum-mismatch) test.',
    )
    parser.add_argument('--model', choices=['generic', 'agn'], default='generic')
    parser.add_argument('--block', choices=['extra', 'full'], default='extra',
                        help='Parameter block for the 1σ surface (default: SDDR extra params).')
    parser.add_argument('--R-orbit', type=float, default=100)
    parser.add_argument('--y-eins', type=float, default=0.5)
    parser.add_argument('--Mc', type=float, default=30.0)
    parser.add_argument('--res', type=int, default=200, help='Frequency grid resolution.')
    parser.add_argument('--n-directions', type=int, default=256)
    parser.add_argument('--snr-min', type=float, default=3.0)
    parser.add_argument('--snr-max', type=float, default=80.0)
    parser.add_argument('--n-snr', type=int, default=20)
    parser.add_argument('--log-r-cut', type=float, default=0.1)
    parser.add_argument('--cdf-fraction', type=float, default=0.9,
                        help='Target fraction of 1σ directions below log-r-cut.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cores', type=int, default=4,
                        help='Worker processes for direction-level |log r| evaluations.')
    parser.add_argument('--outdir', type=Path,
                        default=Path(__file__).resolve().parent / 'plots')
    parser.add_argument('--label', type=str, default=None)
    args = parser.parse_args()

    cores = args.cores
    print(f'Cores: {cores}')

    l1_agn, hlv_agn, hlv_lensed = build_networks()
    rng = onp.random.default_rng(args.seed)

    lensing_params = build_injection(args.R_orbit, args.y_eins, args.Mc, dL=1.0)
    cov, model_parameters, keys = get_covariance(
        args.model, lensing_params, l1_agn, hlv_agn, hlv_lensed, args.res,
    )
    cov, keys = reorder_covariance(cov, keys, PARAM_ORDER)
    wf_params = extract_waveform_params(
        hlv_lensed.signals['H1'],
        l1_agn.convert_to_general_lensed_parameters(lensing_params),
    )

    if args.block == 'extra':
        block_keys = [k for k in EXTRA_PARAMS if k in keys]
        cov_block = covariance_block(cov, keys, block_keys)
    else:
        block_keys = list(keys)
        cov_block = squeeze_covariance(cov)

    snr_ref = snr_at_dL(hlv_lensed, wf_params, args.res)
    print(f'Model: {args.model}, block: {args.block} ({len(block_keys)} params)')
    print(f'Injection: R_orbit={args.R_orbit}, y_Eins={args.y_eins}, Mc={args.Mc}')
    print(f'Network SNR at dL=1 (reference): {snr_ref:.4g}')
    print(f'Block keys: {block_keys}')

    lam = onp.linalg.eigvalsh(cov_block)
    print(f'Covariance block eigenvalues: {lam}')
    print(f'Condition number: {lam.max() / lam.min():.4g}')

    snr_grid = onp.geomspace(args.snr_min, args.snr_max, args.n_snr)
    cdf_fractions = onp.zeros_like(snr_grid)
    median_log_r = onp.zeros_like(snr_grid)

    ref_signal = hlv_lensed.signals['H1']
    freqs = frequency_grid(ref_signal, to_event_params(wf_params), args.res)

    pool_ctx = (
        Pool(cores, initializer=_init_worker_pool)
        if cores > 1 else
        nullcontext()
    )

    with pool_ctx as pool:
        for i_snr, snr_target in enumerate(snr_grid):
            dL = snr_ref / snr_target
            lp_scaled = lensing_params.copy()
            lp_scaled['dL'] = np.array([dL], dtype=np.float64)
            cov_s, _, keys_s = get_covariance(
                args.model, lp_scaled, l1_agn, hlv_agn, hlv_lensed, args.res,
            )
            cov_s, keys_s = reorder_covariance(cov_s, keys_s, PARAM_ORDER)
            wf_s = extract_waveform_params(
                hlv_lensed.signals['H1'],
                l1_agn.convert_to_general_lensed_parameters(lp_scaled),
            )
            if args.block == 'extra':
                cov_block_s = covariance_block(cov_s, keys_s, block_keys)
            else:
                cov_block_s = squeeze_covariance(cov_s)
            directions_s = directions_on_sigma_surface(cov_block_s, args.n_directions, rng)

            h0 = collect_network_strains(hlv_lensed, freqs, wf_s)
            derivs = collect_param_derivatives(hlv_lensed, freqs, wf_s, block_keys)

            log_r_vals = _eval_log_r_values(
                pool, directions_s, wf_s, block_keys, h0, derivs, freqs, cores,
            )

            finite = onp.isfinite(log_r_vals)
            cdf_fractions[i_snr] = onp.mean(log_r_vals[finite] < args.log_r_cut) if finite.any() else 0.0
            median_log_r[i_snr] = onp.nanmedian(log_r_vals)
            print(
                f'SNR={snr_target:6.3g}  dL={dL:8.4g}  '
                f'frac(|log r|<{args.log_r_cut})={cdf_fractions[i_snr]:.3f}  '
                f'median |log r|={median_log_r[i_snr]:.4f}'
            )

    snr_trust = trust_snr_from_cdf(
        snr_grid, cdf_fractions, threshold=args.cdf_fraction, log_r_cut=args.log_r_cut,
    )
    print(
        f'\nFisher self-consistent SNR '
        f'({args.cdf_fraction:.0%} of 1σ directions with |log r| < {args.log_r_cut}): '
        f'{snr_trust:.4g}' if onp.isfinite(snr_trust) else
        f'\nFisher not self-consistent up to SNR={args.snr_max} '
        f'(<{args.cdf_fraction:.0%} directions below |log r|={args.log_r_cut}).'
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    label = args.label or f'{args.model}_{args.block}_R{args.R_orbit:g}_y{args.y_eins:g}'
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    ax = axes[0]
    ax.plot(snr_grid, cdf_fractions, 'o-', lw=1.5)
    ax.axhline(args.cdf_fraction, color='k', ls='--', lw=0.8, alpha=0.6)
    ax.axvline(snr_trust, color='C3', ls='--', lw=0.8,
               label=f'trust SNR ≈ {snr_trust:.3g}' if onp.isfinite(snr_trust) else 'no trust SNR in range')
    ax.set_xscale('log')
    ax.set_xlabel('Network SNR')
    ax.set_ylabel(f'Fraction with $|\\log r| < {args.log_r_cut}$')
    ax.set_title('Vallisneri mismatch CDF')
    ax.set_ylim(0, 1.02)
    ax.legend(loc='lower right', fontsize=8)

    ax = axes[1]
    ax.plot(snr_grid, median_log_r, 's-', lw=1.5, color='C1')
    ax.axhline(args.log_r_cut, color='k', ls='--', lw=0.8, alpha=0.6)
    ax.set_xscale('log')
    if onp.any(onp.isfinite(median_log_r) & (median_log_r > 0)):
        ax.set_yscale('log')
    ax.set_xlabel('Network SNR')
    ax.set_ylabel('Median $|\\log r|$')
    ax.set_title('Typical mismatch on 1σ surface')

    fig.suptitle(
        f'Fisher consistency — {args.model} / {args.block}  '
        f'($R_{{\\rm orb}}$={args.R_orbit:g}, $y$={args.y_eins:g}, $M_c$={args.Mc:g})',
        fontsize=11,
    )

    out_pdf = args.outdir / f'fisher_consistency_{label}.pdf'
    out_npz = args.outdir / f'fisher_consistency_{label}.npz'
    fig.savefig(out_pdf)
    onp.savez(
        out_npz,
        snr_grid=snr_grid,
        cdf_fractions=cdf_fractions,
        median_log_r=median_log_r,
        snr_trust=snr_trust,
        snr_ref=snr_ref,
        block_keys=onp.array(block_keys),
        cov_block_eigenvalues=lam,
        R_orbit=args.R_orbit,
        y_eins=args.y_eins,
        Mc=args.Mc,
        model=args.model,
        block=args.block,
        cores=cores,
    )
    print(f'Saved {out_pdf}')
    print(f'Saved {out_npz}')


if __name__ == '__main__':
    set_start_method('spawn', force=True)
    main()
