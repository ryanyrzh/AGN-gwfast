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
    'R_orbit': 50, 'M_lz': 1e4, 'src_pos': 0.5,
    'dL': 0.5, 'psi': 1, 'theta': 1.87, 'phi': 2.66,
}
# reference_parameters['M_lz'] = 1e6
# reference_parameters['iota'] = 0.999 * np.pi / 2
reference_parameters_1 = reference_parameters.copy()
reference_parameters_2 = reference_parameters.copy()
reference_parameters_1['Mc'] = 30
reference_parameters_2['Mc'] = 80


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

def Jacobian_covariance(lensing_parameters):
    # 4.b Compute the Fisher
    fisher_matrix = HLV_AGN.FisherMatr(lensing_parameters, res=100)
    # 4.c Reduce and compute covar
    covar_matrix, _ = compute_covariance_matrix(fisher_matrix, cores=1)

    # # lensing_transform is a 3-to-6 transform
    # from_params = ['R_orbit', 'src_pos', 'M_lz']
    # to_params = ['delta_iota', 'delta_phase', 'delta_psi', 'relative_distance', 'relative_mass', 'delta_time']
    # fixed_params = [key for key in lensing_parameters.keys() if key not in from_params]
    # target_keys = fixed_params + to_params
    # transformed_cov_mat, transformed_parameters, transformed_keys = covariance_change_variable_1(
    #     covar_matrix, lensing_parameters, target_keys, lensing_transform, from_params, to_params
    # )

    # original 5-to-5 transform
    from_params = ['R_orbit', 'src_pos', 'M_lz', 'iota', 'dL']
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
    lensed_fisher_mat[lensed_fisher_mat == 0.0] = np.nan
    lensed_cov_mats, _ = compute_covariance_matrix(lensed_fisher_mat, cores=1)
    return lensed_cov_mats, model_parameters, keys

def simple_lensing_covariance(lensing_parameters):
    model_parameters = L1_AGN.convert_to_general_lensed_parameters(lensing_parameters)
    keys = list(model_parameters.keys()).copy()

    shape = model_parameters['Mc'].shape

    model_parameters['delta_iota'] = np.full(shape, 0.0)
    model_parameters['delta_phase'] = np.full(shape, 0.0)
    model_parameters['delta_psi'] = np.full(shape, 0.0)
    model_parameters['relative_mass'] = np.full(shape, 1.0)

    simple_HLV_fisher = HLV_Lensed.FisherMatr(model_parameters, res=200)
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
    ''' Compute the Bayes factor B = P(D|M_simple) / P(D|M_full) using the Savage-Dickey density ratio, where M_simple is the simpler model with fixed lensing parameters (delta_iota=0, delta_phase=0, delta_psi=0, relative_mass=1), and M_full is the more complicated model with free lensing parameters. The Bayes factor is computed as the ratio of the likelihoods of the data under the two models, which can be approximated using the covariance matrices of the parameters under each model and the prior widths of the extra parameters in the full model. The function takes in the covariance matrices and parameter dictionaries for both models, as well as the original keys and desired order of parameters for both models, and returns the Bayes factor.

    :param full_cov: covariance matrix of the full model parameters (including lensing parameters)
    :param full_params_dict: dictionary of the full model parameters and their values
    :param simple_cov: covariance matrix of the simple model parameters (with fixed lensing parameters)
    :param simple_params_dict: dictionary of the simple model parameters and their values
    :param orig_keys: original keys of the parameters in the covariance matrices
    :param simple_keys: original keys of the parameters in the simple model covariance matrix
    :param params_order: desired order of parameters for the full model (must include all keys in orig_keys)
    :param simple_params_order: desired order of parameters for the simple model (must include all keys in simple_keys)
    :param prior_widths: array of prior widths for the extra parameters in the full model (delta_iota, delta_phase, delta_psi, relative_mass)

    :return: Bayes factor B = P(D|M_simple) / P(D|M_full)
    '''
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

if __name__ == '__main__':
    # Now do a simple example with no parallelisation and no grids. Just single value of (R, y) to test the code and understand the results. This is also useful for debugging and for understanding the behaviour of the Bayes factor as a function of SNR and lensing parameters. # Do NOT use worker function. Do NOT use the loop to adjust SNR. Just compute the Bayes factor for a single set of lensing parameters and a single SNR, and see how it compares to the target logB of 3. This will help us understand if the code is working as expected and if the Bayes factor is sensitive to the lensing parameters in the way we expect.
    y_Eins_test = np.array([0.5])
    R_orbit_test = np.array([50])
    lensing_parameters_test = reference_parameters_1.copy()
    lensing_parameters_test['R_orbit'] = R_orbit_test
    lensing_parameters_test['src_pos'] = convert_y_from_Einstein_to_Rorbit(y_Eins_test, R_orbit_test)
    # Convert all the lensing parameters to arrays (if not yet already) for compatibility with the covariance functions
    for key in lensing_parameters_test.keys():
        if not isinstance(lensing_parameters_test[key], np.ndarray):
            lensing_parameters_test[key] = np.array([lensing_parameters_test[key]], dtype=np.float64)
        else:
            lensing_parameters_test[key] = lensing_parameters_test[key].astype(np.float64)
    # 4.b Compute the Fisher
    fisher_matrix_array = HLV_AGN.FisherMatr(lensing_parameters_test, res=100)
    # Take the first entry:
    fisher_matrix = fisher_matrix_array[:,:,0]
    # Get the keys associated with the Fisher matrix:
    keys = lensing_parameters_test.keys()
    # Print the diagonals
    print('Fisher matrix keys:', keys)
    print("Fisher matrix diagonal entries:", np.diag(fisher_matrix))
    # Print the condition number in scientific notation
    print("Fisher matrix condition number: %.2e" % np.linalg.cond(fisher_matrix))
    # Conv
    covar_mat_test, params_dict_test, keys_test = Jacobian_covariance(lensing_parameters_test)
    simple_cov_mats_test, simple_params_dict_test, simple_keys_test = simple_lensing_covariance(lensing_parameters_test)
    # Test not finished - to be continued below (feel free to modify)

