#!/usr/bin/env python3
"""
Compute a Fisher covariance for one AGN-lensed injection and save a corner plot.

Injection is always specified in AGN parameters. With ``--model agn`` the Fisher
is evaluated on the AGN-native network. With ``--model generic``, AGN parameters
are converted via ``convert_to_general_lensed_parameters`` and injected into the
generic lensed network.
"""
import os
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', '/tmp/jax_cache_fisher_corner')
os.environ.setdefault('JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS', '0')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl')

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as onp
from jax import config
import jax.numpy as np

config.update('jax_enable_x64', True)

from gwfast.fisherTools import (
    reduce_Fisher_matrix,
    compute_covariance_matrix,
    plot_corners,
)
from gwfast.lensing_utils import convert_y_from_Einstein_to_Rorbit
import fisher_consistency_test_mp as base

LABELS = {
    'Mc': r'${\cal M}\,/\,M_\odot$',
    'eta': r'$\eta$',
    'dL': r'$D_{\rm L}\,/\,{\rm Gpc}$',
    'theta': r'$\theta\,/\,{\rm rad}$',
    'phi': r'$\phi\,/\,{\rm rad}$',
    'iota': r'$\iota\,/\,{\rm rad}$',
    'psi': r'$\psi\,/\,{\rm rad}$',
    'tcoal': r'$t_{\rm c}\,/\,{\rm day}$',
    'phase': r'$\phi_{\rm c}\,/\,{\rm rad}$',
    'chi1z': r'$\chi_{1z}$',
    'chi2z': r'$\chi_{2z}$',
    'R_orbit': r'$R_{\rm orb}\,/\,R_{\rm S}$',
    'M_lz': r'$M_{\rm Lz}\,/\,M_\odot$',
    'log10_M_lz': r'$\log_{10}(M_{\rm Lz}\,/\,M_\odot)$',
    'src_pos': r'$y$',
    'delta_iota': r'$\Delta\iota\,/\,{\rm rad}$',
    'delta_phase': r'$\Delta\phi\,/\,{\rm rad}$',
    'delta_psi': r'$\Delta\psi\,/\,{\rm rad}$',
    'delta_time': r'$\Delta t\,/\,{\rm day}$',
    'relative_mass': r'${\cal M}_2\,/\,{\cal M}_1$',
    'relative_distance': r'$D_{L,2}\,/\,D_{L,1}$',
}

BLOCK_KEYS = {
    'full': None,
    'agn_extra': ['R_orbit', 'src_pos', 'log10_M_lz'],
    'generic_extra_4': ['relative_mass', 'delta_iota', 'delta_phase', 'delta_psi'],
    'generic_extra_6': [
        'relative_distance', 'delta_time',
        'relative_mass', 'delta_iota', 'delta_phase', 'delta_psi',
    ],
}

BLOCK_MODEL = {
    'full': None,
    'agn_extra': 'agn',
    'generic_extra_4': 'generic',
    'generic_extra_6': 'generic',
}


def build_injection(Mc, dL, R_orbit, M_lz, y_eins):
    # Source position is specified in Einstein radii; injection needs R_orbit units.
    src_pos = float(convert_y_from_Einstein_to_Rorbit(y_eins, R_orbit))
    lp = {k: np.array([v], dtype=np.float64) for k, v in base.REFERENCE_PARAMETERS.items()}
    lp['Mc'] = np.array([Mc], dtype=np.float64)
    lp['dL'] = np.array([dL], dtype=np.float64)
    lp['R_orbit'] = np.array([R_orbit], dtype=np.float64)
    lp['src_pos'] = np.array([src_pos], dtype=np.float64)
    lp['log10_M_lz'] = np.array([onp.log10(M_lz)], dtype=np.float64)
    return lp, src_pos


def squeeze_cov_2d(cov):
    c = onp.asarray(cov, dtype=onp.float64)
    while c.ndim > 2:
        c = onp.squeeze(c, axis=-1)
    if c.ndim != 2:
        raise ValueError(f'Expected 2D covariance, got shape {c.shape}')
    return 0.5 * (c + c.T)


def scalar_event(params):
    return {k: float(onp.asarray(v).reshape(-1)[0]) for k, v in params.items()}


def select_block(keys, cov2d, event, block):
    """Return (plot_keys, cov_sub) for the requested corner block."""
    wanted = BLOCK_KEYS[block]
    if wanted is None:
        plot_keys = [k for k in keys if k in event]
    else:
        missing = [k for k in wanted if k not in keys]
        if missing:
            raise ValueError(
                f'--block {block} needs keys {wanted}, but Fisher is missing {missing}. '
                f'Available keys: {keys}'
            )
        plot_keys = list(wanted)
    idx = [keys.index(k) for k in plot_keys]
    return plot_keys, cov2d[onp.ix_(idx, idx)]


def main():
    parser = argparse.ArgumentParser(
        description='Fisher corner plot for one AGN-lensed injection.',
    )
    parser.add_argument('--Mc', type=float, default=30.0, help='Chirp mass [Msun].')
    parser.add_argument('--dL', type=float, default=1.0, help='Luminosity distance [Gpc].')
    parser.add_argument('--R-orbit', type=float, default=100.0,
                        help='Orbital radius [R_Sch].')
    parser.add_argument('--M-lz', type=float, default=1e4,
                        help='Lens mass [Msun] (stored as log10_M_lz for AGN).')
    parser.add_argument('--y-eins', type=float, default=0.5,
                        help='Source position Einstein radii.')
    parser.add_argument('--model', choices=['agn', 'generic'], default='agn',
                        help='agn: native AGN Fisher; generic: convert then '
                             'inject into the generic lensed model.')
    parser.add_argument(
        '--block',
        choices=list(BLOCK_KEYS),
        default='full',
        help='Corner subset: full Fisher; agn_extra (R_orbit, y_eins, '
             'log10_M_lz); generic_extra_4 (relative_mass, delta_iota, '
             'delta_phase, delta_psi); generic_extra_6 (+ relative_distance, '
             'delta_time).',
    )
    parser.add_argument('--res', type=int, default=200,
                        help='Frequency-grid resolution for FisherMatr.')
    parser.add_argument('--outdir', type=Path,
                        default=Path(__file__).resolve().parent / 'plots')
    parser.add_argument('--label', type=str, default=None)
    args = parser.parse_args()

    required_model = BLOCK_MODEL[args.block]
    if required_model is not None and args.model != required_model:
        raise SystemExit(
            f'--block {args.block} requires --model {required_model}, '
            f'got --model {args.model}'
        )

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=12)

    l1_agn, hlv_agn, hlv_lensed = base.build_networks()
    lensing_params, src_pos_rorb = build_injection(
        args.Mc, args.dL, args.R_orbit, args.M_lz, args.y_eins,
    )

    if args.model == 'agn':
        cov, event_params, keys = base.intrinsic_covariance(
            lensing_params, hlv_agn, args.res,
        )
        snr = float(onp.asarray(hlv_agn.SNR(lensing_params, res=1000)).reshape(-1)[0])
        color = 'C0'
    else:
        cov, event_params, keys = base.direct_covariance(
            lensing_params, l1_agn, hlv_lensed, args.res,
        )
        snr = float(onp.asarray(hlv_lensed.SNR(event_params, res=1000)).reshape(-1)[0])
        color = 'C3'

    cov2d = squeeze_cov_2d(cov)
    event = scalar_event(event_params)
    # keys was mutated in-place by reduce_Fisher_matrix to match cov rows.
    plot_keys, cov_plot = select_block(keys, cov2d, event, args.block)
    indices = list(range(len(plot_keys)))

    print(f'Model: {args.model}, block: {args.block}')
    print(f'Injection: Mc={args.Mc}, dL={args.dL}, R_orbit={args.R_orbit}, '
          f'M_lz={args.M_lz}, y_eins={args.y_eins}, src_pos(R_orb)={src_pos_rorb:.6g}')
    print(f'Network SNR: {snr:.4g}')
    print(f'Plot keys ({len(plot_keys)}): {plot_keys}')
    print(f'Event values: { {k: event[k] for k in plot_keys} }')

    fig = plot_corners(
        cov_plot, plot_keys, indices,
        event=event, labels=LABELS, color=color,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    tag = args.label or (
        f'fisher_corner_{args.model}_{args.block}'
        f'_Mc{args.Mc:g}_dL{args.dL:g}'
        f'_R{args.R_orbit:g}_Mlz{args.M_lz:g}_yEins{args.y_eins:g}'
    )
    outfile = args.outdir / f'{tag}.pdf'
    fig.savefig(outfile, bbox_inches='tight')
    print(f'Saved {outfile}')


if __name__ == '__main__':
    main()
