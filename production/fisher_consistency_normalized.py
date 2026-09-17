#!/usr/bin/env python3
"""
This is a replacement for `fisher_consistency_test_mp.py`, whose goal is to perform 
the test from Vallisneri (2008) that determines the minimum SNR at which 
the Fisher formalism is self-consistent. 

Different than the original script, this one uses dimensionless covariance
to avoid unit artifacts in the statistics.
I.e., sampling of the 1-sigma surface is done in normalized coordinates theta_i / sigma_i,
degenerate directions are identified and removed based on the normalized eigenvalues,
and the remaining displacement is then converted back to dimensional to be applied to the waveform.

In the perturbative regime, δθ falls roughly as SNR^-1, 
so the residual δh ~ ∂^2h/∂θ^2 (δθ)^2 ~ SNR^-1.
Then |log r| = 1/2 <δh|δh> falls roughly as SNR^-2.
A plot of |log r| vs SNR is produced to track this inverse quadratic relationship.

Any deviation from this relationship signals a δθ that does not fall as SNR^-1,
which then points to parameter degeneracy.
Vallisneri's test requires that all such degeneracy have been removed.
Therefore by default, `{phase, chi1z, chi2z, tcoal}` are held fixed 
(the aligned-spin/phase/time degeneracy) and the conditional 1-sigma
surface of the rest is sampled.  
This can be overridden by explicitly defining fixed parameters with `--condition-on`
or by passing `--include-unconstrained` to skip conditioning altogether.
"""
import argparse
from contextlib import nullcontext
from multiprocessing import Pool, set_start_method
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as onp
import jax.numpy as np

import fisher_consistency_test_mp as base

DEFAULT_CONDITION_ON = ('phase', 'chi1z', 'chi2z', 'tcoal')


def condition_covariance(cov, keys, keep_keys, fix_keys=None):
    """Covariance of `keep_keys` with `fix_keys` held fixed.
    Parameters in neither list are marginalized (the keep+fix sub-block of
    `cov` is formed first).
    """
    keys = list(keys)
    keep_keys = list(keep_keys)
    fix_keys = [k for k in (fix_keys or []) if k not in keep_keys]
    missing = [k for k in keep_keys + fix_keys if k not in keys]
    if missing:
        raise KeyError(f'Parameters not in covariance: {missing}')
    cov = onp.asarray(cov, dtype=onp.float64)
    cov = 0.5 * (cov + cov.T)
    active = keep_keys + fix_keys
    idx_active = [keys.index(k) for k in active]
    c_active = cov[onp.ix_(idx_active, idx_active)]
    idx_keep = [active.index(k) for k in keep_keys]
    if not fix_keys:
        return c_active[onp.ix_(idx_keep, idx_keep)]
    diag = onp.diag(c_active).copy()
    if onp.any(~onp.isfinite(diag)) or onp.any(diag <= 0):
        raise ValueError(f'Non-positive variances while conditioning: {diag}')
    sigma = onp.sqrt(diag)
    corr = c_active / onp.outer(sigma, sigma)
    corr = 0.5 * (corr + corr.T)
    precision = onp.linalg.pinv(corr, rcond=1e-12)
    p_keep = precision[onp.ix_(idx_keep, idx_keep)]
    corr_cond = onp.linalg.pinv(p_keep, rcond=1e-12)
    corr_cond = 0.5 * (corr_cond + corr_cond.T)
    lam, vec = onp.linalg.eigh(corr_cond)
    lam = onp.clip(lam, 0.0, None)
    corr_cond = (vec * lam) @ vec.T
    sigma_keep = sigma[idx_keep]
    return corr_cond * onp.outer(sigma_keep, sigma_keep)


def apply_conditioning(cov, keys, block_keys, fix_keys):
    """Hold `fix_keys` fixed; return remaining keys, fix list, conditional cov, sigmas."""
    fix_keys = [k for k in fix_keys if k in keys]
    keep = [k for k in block_keys if k not in fix_keys]
    if not keep:
        raise ValueError('No parameters left to sample after conditioning.')
    c_cond = condition_covariance(cov, keys, keep, fix_keys)
    _, _, sigma = normalized_spectrum(c_cond)
    return keep, fix_keys, c_cond, sigma


def normalized_spectrum(cov_block):
    """Eigen-decomposition of the covariance block in units of its own sigmas.

    :param array cov_block: symmetric covariance block, shape (N, N).
    :return: normalized eigenvalues, eigenvectors, sigmas
    """
    cov = onp.asarray(cov_block, dtype=onp.float64)
    cov = 0.5 * (cov + cov.T)
    diag = onp.diag(cov)
    if onp.any(~onp.isfinite(diag)) or onp.any(diag <= 0):
        raise ValueError(f'Non-positive variances on the diagonal: {diag}')
    sigma = onp.sqrt(diag)
    corr = cov / onp.outer(sigma, sigma)
    corr = 0.5 * (corr + corr.T)
    lam_hat, V_hat = onp.linalg.eigh(corr)
    return lam_hat, V_hat, sigma


def normalized_direction_mask(lam_hat, max_inflation=4.0, min_eig=1e-10):
    """Keep the eigen-directions that are usable, judged in unit-free terms.
    Discard the following:
    - If `lam_hat > max_inflation`, the 1-sigma extent along this
    combination is inflated well beyond the marginal errors due to a degeneracy.
    - If `lam_hat < min_eig`, the value is a numerical residue rather than a measurement.
      This happens in the AGN path due to the rank-deficient Jacobian.

    :return: boolean mask (True = keep), same order as `lam_hat`.
    """
    lam_hat = onp.asarray(lam_hat, dtype=onp.float64)
    keep = onp.isfinite(lam_hat) & (lam_hat > min_eig) & (lam_hat <= max_inflation)
    if not keep.any():
        raise ValueError(
            f'No usable directions: normalized eigenvalues {lam_hat} all fall '
            f'outside ({min_eig:.3g}, {max_inflation:.3g}]'
        )
    return keep


def directions_on_sigma_surface_normalized(cov_block, n_dirs, rng,
                                           max_inflation=4.0, min_eig=1e-10,
                                           verbose=True):
    """Return (ndim, n_dirs) displacements on the 1-sigma surface of `cov_block`,
    restricted to its non-degenerate subspace in normalized coordinates.
    """
    lam_hat, V_hat, sigma = normalized_spectrum(cov_block)
    keep = normalized_direction_mask(lam_hat, max_inflation, min_eig)
    n_dropped = int((~keep).sum())
    if n_dropped and verbose:
        degenerate = onp.sort(lam_hat[lam_hat > max_inflation])
        singular = onp.sort(lam_hat[lam_hat <= min_eig])
        print(
            f'Removed {n_dropped}/{len(lam_hat)} direction(s): '
            f'{len(degenerate)} degenerate (normalized eig > '
            f'{max_inflation:.3g}: {degenerate}), '
            f'{len(singular)} numerically singular (normalized eig <= '
            f'{min_eig:.3g}: {singular})'
        )
    u = base.sample_unit_directions(rng, n_dirs, int(keep.sum()))
    unit_delta = (V_hat[:, keep] * onp.sqrt(lam_hat[keep])) @ u.T
    return sigma[:, onp.newaxis] * unit_delta


def scaling_report(snr_grid, median_log_r):
    """Power-law slope of median |log r|(SNR)."""
    snr = onp.asarray(snr_grid, dtype=onp.float64)
    med = onp.asarray(median_log_r, dtype=onp.float64)
    good = onp.isfinite(med) & (med > 0)
    slope = onp.nan
    if good.sum() >= 2:
        slope = onp.polyfit(onp.log(snr[good]), onp.log(med[good]), 1)[0]
    return slope


def main():
    parser = argparse.ArgumentParser(
        description='Vallisneri Fisher consistency test with a unit-free '
                    'degeneracy cut (normalised covariance spectrum).',
    )
    parser.add_argument('--model', choices=['generic', 'agn'], default='generic')
    parser.add_argument('--block', choices=['extra', 'full'], default='full')
    parser.add_argument('--R-orbit', type=float, default=100)
    parser.add_argument('--y-eins', type=float, default=0.5)
    parser.add_argument('--Mc', type=float, default=30.0)
    parser.add_argument('--res', type=int, default=200)
    parser.add_argument('--n-directions', type=int, default=256)
    parser.add_argument('--snr-min', type=float, default=3.0)
    parser.add_argument('--snr-max', type=float, default=80.0)
    parser.add_argument('--n-snr', type=int, default=20)
    parser.add_argument('--log-r-cut', type=float, default=0.1)
    parser.add_argument('--max-inflation', type=float, default=4.0,
                        help='Drop eigen-directions whose normalised '
                             'variance exceeds this; 1.0 is the '
                             'uncorrelated reference, so this bounds how much '
                             'degeneracy is allowed to inflate an excursion.')
    parser.add_argument('--min-normalized-eig', type=float, default=1e-10,
                        help='Drop eigen-directions whose normalised variance '
                             'falls below this, i.e. numerically singular '
                             'directions ')
    parser.add_argument('--include-unconstrained', action='store_true',
                        help='Skip conditioning on DEFAULT_CONDITION_ON '
                             '(phase, chi1z, chi2z, tcoal). ')
    parser.add_argument('--condition-on', nargs='*', default=None,
                        help='Parameter names to hold fixed instead of '
                             'DEFAULT_CONDITION_ON (ignored with '
                             '--include-unconstrained).')
    parser.add_argument('--cdf-fraction', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cores', type=int, default=4)
    parser.add_argument('--outdir', type=Path,
                        default=Path(__file__).resolve().parent / 'plots')
    parser.add_argument('--label', type=str, default=None)
    args = parser.parse_args()

    cores = args.cores
    print(f'Cores: {cores}')

    l1_agn, hlv_agn, hlv_lensed = base.build_networks()
    rng = onp.random.default_rng(args.seed)

    lensing_params = base.build_injection(args.R_orbit, args.y_eins, args.Mc, dL=1.0)
    cov, _, keys = base.get_covariance(
        args.model, lensing_params, l1_agn, hlv_agn, hlv_lensed, args.res,
    )
    cov, keys = base.reorder_covariance(cov, keys, base.PARAM_ORDER)
    wf_params = base.extract_waveform_params(
        hlv_lensed.signals['H1'],
        l1_agn.convert_to_general_lensed_parameters(lensing_params),
    )

    if args.block == 'extra':
        block_keys = [k for k in base.EXTRA_PARAMS if k in keys]
        cov_block = base.covariance_block(cov, keys, block_keys)
    else:
        block_keys = list(keys)
        cov_block = base.squeeze_covariance(cov)

    snr_ref = base.snr_at_dL(hlv_lensed, wf_params, args.res)
    print(f'Model: {args.model}, block: {args.block} ({len(block_keys)} params)')
    print(f'Injection: R_orbit={args.R_orbit}, y_Eins={args.y_eins}, Mc={args.Mc}')
    print(f'Network SNR at dL=1 (reference): {snr_ref:.4g}')
    print(f'Block keys: {block_keys}')

    lam_raw = onp.linalg.eigvalsh(cov_block)
    _, _, sigma = normalized_spectrum(cov_block)
    print(f'Raw eigenvalues (mixed units): {lam_raw}')
    print(f'Raw condition number (unit-contaminated): '
          f'{lam_raw.max() / lam_raw.min():.4g}')
    print('Marginal sigmas:')
    for key, sig in zip(block_keys, sigma):
        print(f'    {key:>18s}  {sig:.6e}')

    fixed_keys = []
    if args.include_unconstrained:
        print('Including unconstrained directions, not a valid Vallisneri test.')
    else:
        requested = list(args.condition_on) if args.condition_on else list(DEFAULT_CONDITION_ON)
        missing = [k for k in requested if k not in keys]
        if missing:
            print(f'Note: not in covariance, skipped: {missing}')
        print(f'Conditioning on {requested} (holding fixed, not marginalising).')
        block_keys, fixed_keys, cov_block, sigma = apply_conditioning(
            cov, keys, block_keys, requested,
        )
        lam_raw = onp.linalg.eigvalsh(cov_block)
        print(f'Conditioned block ({len(block_keys)} params): {block_keys}')
        print(f'Held fixed: {fixed_keys}')
        print('Conditional sigmas:')
        for key, sig in zip(block_keys, sigma):
            print(f'    {key:>18s}  {sig:.6e}')

    lam_hat, _, sigma = normalized_spectrum(cov_block)
    print(f'Normalised eigenvalues: {lam_hat}')
    print(f'Normalised condition number: '
          f'{lam_hat.max() / lam_hat[lam_hat > 0].min():.4g}')
    keep_ref = normalized_direction_mask(
        lam_hat, args.max_inflation, args.min_normalized_eig,
    )
    print(f'Keeping {int(keep_ref.sum())}/{len(lam_hat)} direction(s) '
          f'(max-inflation={args.max_inflation:.3g}, '
          f'min-normalised-eig={args.min_normalized_eig:.3g}).')

    snr_grid = onp.geomspace(args.snr_min, args.snr_max, args.n_snr)
    cdf_fractions = onp.zeros_like(snr_grid)
    median_log_r = onp.zeros_like(snr_grid)

    freqs = base.frequency_grid(
        hlv_lensed.signals['H1'], base.to_event_params(wf_params), args.res,
    )

    if cores > 1:
        pool_ctx = Pool(cores, initializer=base._init_worker_pool)
    else:
        base._HLV_LENSED = hlv_lensed
        pool_ctx = nullcontext()

    with pool_ctx as pool:
        for i_snr, snr_target in enumerate(snr_grid):
            dL = snr_ref / snr_target
            lp_scaled = lensing_params.copy()
            lp_scaled['dL'] = np.array([dL], dtype=np.float64)
            cov_s, _, keys_s = base.get_covariance(
                args.model, lp_scaled, l1_agn, hlv_agn, hlv_lensed, args.res,
            )
            cov_s, keys_s = base.reorder_covariance(cov_s, keys_s, base.PARAM_ORDER)
            wf_s = base.extract_waveform_params(
                hlv_lensed.signals['H1'],
                l1_agn.convert_to_general_lensed_parameters(lp_scaled),
            )
            if fixed_keys:
                cov_block_s = condition_covariance(
                    cov_s, keys_s, block_keys, fixed_keys,
                )
            elif args.block == 'extra':
                cov_block_s = base.covariance_block(cov_s, keys_s, block_keys)
            else:
                cov_block_s = base.squeeze_covariance(cov_s)
            directions_s = directions_on_sigma_surface_normalized(
                cov_block_s, args.n_directions, rng,
                max_inflation=args.max_inflation,
                min_eig=args.min_normalized_eig,
            )

            h0 = base.collect_network_strains(hlv_lensed, freqs, wf_s)
            derivs = base.collect_param_derivatives(hlv_lensed, freqs, wf_s, block_keys)

            log_r_vals = base._eval_log_r_values(
                pool, directions_s, wf_s, block_keys, h0, derivs, freqs, cores,
            )

            finite = onp.isfinite(log_r_vals)
            cdf_fractions[i_snr] = (
                onp.mean(log_r_vals[finite] < args.log_r_cut) if finite.any() else 0.0
            )
            median_log_r[i_snr] = onp.nanmedian(log_r_vals)
            print(
                f'SNR={snr_target:6.3g}  dL={dL:8.4g}  '
                f'frac(|log r|<{args.log_r_cut})={cdf_fractions[i_snr]:.3f}  '
                f'median |log r|={median_log_r[i_snr]:.4g}'
            )

    snr_trust = base.trust_snr_from_cdf(
        snr_grid, cdf_fractions, threshold=args.cdf_fraction, log_r_cut=args.log_r_cut,
    )
    slope = scaling_report(snr_grid, median_log_r)
    print(
        f'\nFisher self-consistent SNR '
        f'({args.cdf_fraction:.0%} of 1σ directions with |log r| < {args.log_r_cut}): '
        f'{snr_trust:.4g}' if onp.isfinite(snr_trust) else
        f'\nFisher not self-consistent up to SNR={args.snr_max} '
        f'(<{args.cdf_fraction:.0%} directions below |log r|={args.log_r_cut}).'
    )
    print(f'Fitted power law: median |log r| ~ SNR^{slope:+.2f} '
          f'(expected -2 in the perturbative regime)')
    if slope > -1.0:
        print('WARNING: |log r| does not fall like SNR^-2. The sampled '
              'excursions are not shrinking with SNR, so the retained subspace '
              'is still degenerate (or numerically ill-defined) and the trust '
              'SNR reported above is meaningless.')

    args.outdir.mkdir(parents=True, exist_ok=True)
    label = args.label or (
        f'{args.model}_{args.block}_R{args.R_orbit:g}_y{args.y_eins:g}_norm'
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

    ax = axes[0]
    ax.plot(snr_grid, cdf_fractions, 'o-', lw=1.5)
    ax.axhline(args.cdf_fraction, color='k', ls='--', lw=0.8, alpha=0.6)
    ax.axvline(snr_trust, color='C3', ls='--', lw=0.8,
               label=f'trust SNR = {snr_trust:.3g}' if onp.isfinite(snr_trust)
               else 'no trust SNR in range')
    ax.set_xscale('log')
    ax.set_xlabel('Network SNR')
    ax.set_ylabel(f'Fraction with $|\\log r| < {args.log_r_cut}$')
    ax.set_title('Vallisneri mismatch CDF')
    ax.set_ylim(0, 1.02)
    ax.legend(loc='lower right', fontsize=8)

    ax = axes[1]
    ax.plot(snr_grid, median_log_r, 's-', lw=1.5, color='C1',
            label=f'fit: $\\propto$ SNR$^{{{slope:+.2f}}}$')
    good = onp.isfinite(median_log_r) & (median_log_r > 0)
    if good.any():
        ref = median_log_r[good][0] * (snr_grid / snr_grid[good][0])**-2
        ax.plot(snr_grid, ref, ':', color='k', lw=1.0, label='expected SNR$^{-2}$')
    ax.axhline(args.log_r_cut, color='k', ls='--', lw=0.8, alpha=0.6)
    ax.set_xscale('log')
    if good.any():
        ax.set_yscale('log')
    ax.set_xlabel('Network SNR')
    ax.set_ylabel('Median $|\\log r|$')
    ax.set_title('Typical mismatch on $1\\sigma$ surface')
    ax.legend(fontsize=8)

    fig.suptitle(
        f'Fisher consistency (unit-free cut): {args.model} / {args.block}  '
        f'($R_{{\\rm orb}}$={args.R_orbit:g}, $y$={args.y_eins:g}, '
        f'$M_c$={args.Mc:g}, max inflation={args.max_inflation:g})',
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
        power_law_slope=slope,
        snr_trust=snr_trust,
        snr_ref=snr_ref,
        block_keys=onp.array(block_keys),
        cov_block_eigenvalues=lam_raw,
        cov_block_normalized_eigenvalues=lam_hat,
        marginal_sigmas=sigma,
        max_inflation=args.max_inflation,
        min_normalized_eig=args.min_normalized_eig,
        restricted_to_constrained=not args.include_unconstrained,

        fixed_keys=onp.array(fixed_keys),
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
