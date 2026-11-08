#!/usr/bin/env python3
"""
Compute a Fisher covariance for one AGN-lensed injection and save a corner plot.

Injection is specified in AGN parameters, converted via
``convert_to_general_lensed_parameters``, and injected into the generic lensed
network. The corner plot shows relative_mass, delta_iota, delta_phase, delta_psi.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl')

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as onp
from jax import config
import jax.numpy as np

config.update('jax_enable_x64', True)

from gwfast.gwfastGlobals import detectors as det_dict, detPath
import gwfast.waveforms as waveforms
from gwfast.detector import Detector
import gwfast.network as network
from gwfast.signals import AGNLensedGWSignal, GeneralLensedGWSignal
from gwfast.fisherTools import (
    reduce_Fisher_matrix,
    compute_covariance_matrix,
    plot_corners,
)
from gwfast.lensing_utils import convert_y_from_Einstein_to_Rorbit


y_eins = 0.5
R_orbit = 100.0
src_pos = float(convert_y_from_Einstein_to_Rorbit(y_eins, R_orbit))
REFERENCE_PARAMETERS = {
    'Mc': 30, 'eta': 0.24, 'iota': 0.999 * onp.pi / 2, 'phase': 2,
    'chi1z': 0.3, 'chi2z': 0.5, 'tcoal': 0,
    'R_orbit': R_orbit, 'log10_M_lz': 4.0, 'src_pos': src_pos,
    'dL': 1.0, 'psi': 1, 'theta': 1.87, 'phi': 2.66,
}

LABELS = {
    'delta_iota': r'$\Delta\iota\,/\,{\rm rad}$',
    'delta_phase': r'$\Delta\phi\,/\,{\rm rad}$',
    'delta_psi': r'$\Delta\psi\,/\,{\rm rad}$',
    'relative_mass': r'${\cal M}_2\,/\,{\cal M}_1$',
}

PLOT_KEYS = ['relative_mass', 'delta_iota', 'delta_phase', 'delta_psi']


def build_networks():
    h1 = Detector('H1', **det_dict['H1'],
                  noise_curve_path=Path(detPath)/'observing_scenarios_paper/AplusDesign.txt')
    l1 = Detector('L1', **det_dict['L1'],
                  noise_curve_path=Path(detPath)/'observing_scenarios_paper/AplusDesign.txt')
    v1 = Detector('V1', **det_dict['Virgo'],
                  noise_curve_path=Path(detPath)/'observing_scenarios_paper/avirgo_O5low_NEW.txt')
    wf_model = waveforms.IMRPhenomHM()
    # AGN signal object is only used to convert injection parameters
    l1_agn = AGNLensedGWSignal(wf_model=wf_model, detector=l1, fmin=10, verbose=False)
    h1_lensed = GeneralLensedGWSignal(wf_model=wf_model, detector=h1, fmin=10, verbose=False)
    l1_lensed = GeneralLensedGWSignal(wf_model=wf_model, detector=l1, fmin=10, verbose=False)
    v1_lensed = GeneralLensedGWSignal(wf_model=wf_model, detector=v1, fmin=10, verbose=False)
    hlv_lensed = network.DetNet({'H1': h1_lensed, 'L1': l1_lensed, 'V1': v1_lensed}, verbose=False)
    return l1_agn, hlv_lensed


def direct_covariance(lensing_parameters, l1_agn, hlv_lensed, res=200):
    model_parameters = l1_agn.convert_to_general_lensed_parameters(lensing_parameters)
    keys = list(model_parameters.keys())
    fisher = hlv_lensed.FisherMatr(model_parameters, res=res)
    fisher = np.nan_to_num(fisher, nan=0.0)
    fisher, _ = reduce_Fisher_matrix(fisher, keys=keys)
    fisher[fisher == 0.0] = np.nan
    cov, _ = compute_covariance_matrix(fisher, cores=1)
    return cov[:, :, 0], model_parameters, keys


def main():
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif', size=12)

    l1_agn, hlv_lensed = build_networks()
    # gwfast expects length-1 arrays even for a single event
    lensing_params = {key: np.array([value]) for key, value in REFERENCE_PARAMETERS.items()}

    cov, event_params, keys = direct_covariance(lensing_params, l1_agn, hlv_lensed)
    event = {k: float(onp.asarray(v).ravel()[0]) for k, v in event_params.items()}
    idx = [keys.index(k) for k in PLOT_KEYS]
    cov_plot = cov[onp.ix_(idx, idx)]

    fig = plot_corners(
        cov_plot, PLOT_KEYS, list(range(len(PLOT_KEYS))),
        event=event, labels=LABELS, color='C3', fontsize=20,
    )
    fig.savefig('figure_1_fisher_corner.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
