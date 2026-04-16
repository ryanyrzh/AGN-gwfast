#!/usr/local/bin/python3
"""Single-event AGN lensing diagnostic for the SDDR ingredients.

This is a focused diagnostic script for one AGN-lensed event, not an SNR
threshold scan. It evaluates the main building blocks one by one so it is easy
to see what each stage is doing:

1. Build one lensing parameter set.
2. Compute the raw lensing outputs from ``compute_lensed_angles_approx``.
3. Convert those outputs into the phenomenological parameters used downstream.
4. Compute AGN-network SNR, Fisher matrix, and covariance.
5. Convert to the general lensed parameterisation and compare covariances.
6. Break the SDDR Bayes factor into its individual terms.
"""

from pathlib import Path

import numpy as onp
from jax import config
import jax.numpy as np
from jax.lax import integer_pow

config.update("jax_enable_x64", True)

from gwfast.gwfastGlobals import detectors as det_dict, detPath
import gwfast.waveforms as waveforms
from gwfast.detector import Detector
import gwfast.network as network
from gwfast.signals import AGNLensedGWSignal, GeneralLensedGWSignal
from gwfast.lensing_utils import (
    compute_lensed_angles_approx,
    convert_y_from_Einstein_to_Rorbit,
)
from gwfast.fisherTools import (
    reduce_Fisher_matrix,
    compute_covariance_matrix,
    covariance_change_variable,
)


AGN_PARAMETER_ORDER = [
    "Mc",
    "eta",
    "iota",
    "phase",
    "chi1z",
    "chi2z",
    "tcoal",
    "dL",
    "psi",
    "theta",
    "phi",
    "delta_time",
    "relative_distance",
    "relative_mass",
    "delta_iota",
    "delta_phase",
    "delta_psi",
]

EXTRA_PARAMETER_KEYS = [
    "relative_mass",
    "delta_iota",
    "delta_phase",
    "delta_psi",
]

PRIOR_WIDTHS = np.array([1.0, 0.5, 0.5, 0.5], dtype=np.float64)

REFERENCE_PARAMETERS = {
    "Mc": 30.0,
    "eta": 0.24,
    "iota": 0.99 * np.pi / 2,
    "phase": 2.0,
    "chi1z": 0.3,
    "chi2z": 0.5,
    "tcoal": 0.0,
    "R_orbit": 100.0,
    "M_lz": 1e4,
    "src_pos": 0.5,
    "dL": 1.0,
    "psi": 1.0,
    "theta": 1.87,
    "phi": 2.66,
}

DEFAULT_Y_EINSTEIN = 0.5


H1 = Detector(
    "H1",
    **det_dict["H1"],
    noise_curve_path=Path(detPath) / "observing_scenarios_paper/AplusDesign.txt",
)
L1 = Detector(
    "L1",
    **det_dict["L1"],
    noise_curve_path=Path(detPath) / "observing_scenarios_paper/AplusDesign.txt",
)
V1 = Detector(
    "V1",
    **det_dict["Virgo"],
    noise_curve_path=Path(detPath) / "observing_scenarios_paper/avirgo_O5low_NEW.txt",
)

WAVEFORM_MODEL = waveforms.IMRPhenomD()

H1_AGN = AGNLensedGWSignal(wf_model=WAVEFORM_MODEL, detector=H1, fmin=10)
L1_AGN = AGNLensedGWSignal(wf_model=WAVEFORM_MODEL, detector=L1, fmin=10)
V1_AGN = AGNLensedGWSignal(wf_model=WAVEFORM_MODEL, detector=V1, fmin=10)
HLV_AGN = network.DetNet({"H1": H1_AGN, "L1": L1_AGN, "V1": V1_AGN})

H1_LENSED = GeneralLensedGWSignal(wf_model=WAVEFORM_MODEL, detector=H1, fmin=10)
L1_LENSED = GeneralLensedGWSignal(wf_model=WAVEFORM_MODEL, detector=L1, fmin=10)
V1_LENSED = GeneralLensedGWSignal(wf_model=WAVEFORM_MODEL, detector=V1, fmin=10)
HLV_LENSED = network.DetNet({"H1": H1_LENSED, "L1": L1_LENSED, "V1": V1_LENSED})


def _to_scalar(value):
    array = onp.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected a single value, got shape {array.shape}.")
    return float(array.reshape(-1)[0])


def _as_single_event_matrix(array):
    matrix = onp.asarray(array)
    while matrix.ndim > 2:
        matrix = matrix[..., 0]
    return matrix


def _format_scalar_dict(values):
    lines = []
    for key, value in values.items():
        if onp.asarray(value).size == 1:
            lines.append(f"  {key:>18} = {_to_scalar(value): .6g}")
        else:
            lines.append(f"  {key:>18} = {onp.asarray(value)}")
    return "\n".join(lines)


def _covariance_matrix_table(covariance, labels, float_fmt="{: .6g}"):
    matrix = _as_single_event_matrix(covariance)
    labels = list(labels)
    if matrix.shape != (len(labels), len(labels)):
        return (
            f"(cannot tabulate: cov shape {onp.asarray(covariance).shape}, "
            f"{len(labels)} labels)\n{onp.asarray(covariance)}"
        )

    cells = [
        [float_fmt.format(float(matrix[i, j])) for j in range(matrix.shape[1])]
        for i in range(matrix.shape[0])
    ]
    label_width = max(len(label) for label in labels)
    column_widths = [
        max(len(labels[j]), max(len(cells[i][j]) for i in range(matrix.shape[0])))
        for j in range(matrix.shape[1])
    ]
    padding = 2

    header = " " * (label_width + padding) + "".join(
        labels[j].ljust(column_widths[j] + padding)
        for j in range(matrix.shape[1])
    )
    rows = [header]
    for i in range(matrix.shape[0]):
        row = labels[i].ljust(label_width + padding) + "".join(
            cells[i][j].rjust(column_widths[j]).ljust(column_widths[j] + padding)
            for j in range(matrix.shape[1])
        )
        rows.append(row)
    return "\n".join(rows)


def build_single_event_parameters(
    reference_parameters=REFERENCE_PARAMETERS,
    y_einstein=DEFAULT_Y_EINSTEIN,
):
    parameters = {
        key: np.array([value], dtype=np.float64)
        for key, value in reference_parameters.items()
    }
    y_rorbit = convert_y_from_Einstein_to_Rorbit(
        np.array([y_einstein], dtype=np.float64),
        parameters["R_orbit"],
    )[0]
    parameters["src_pos"] = np.array([y_rorbit], dtype=np.float64)
    return parameters, y_rorbit


def get_raw_lensing_outputs(lensing_parameters):
    transform_input = dict(lensing_parameters)
    transform_input["phase"] = 0.0
    transform_input["psi"] = 0.0
    return compute_lensed_angles_approx(transform_input)


def lensing_transform(lensing_parameters):
    outputs = get_raw_lensing_outputs(lensing_parameters)
    relative_magnification = outputs["sqrt_mu_p"] / outputs["sqrt_mu_m"]
    return {
        "delta_iota": outputs["iota_m"] - outputs["iota_p"],
        "delta_phase": outputs["phase_m"] - outputs["phase_p"],
        "delta_psi": outputs["psi_m"] - outputs["psi_p"],
        "relative_distance": relative_magnification
        * integer_pow((1 + outputs["z_rel_m"]) / (1 + outputs["z_rel_p"]), 2),
        "relative_mass": (1 + outputs["z_rel_m"]) / (1 + outputs["z_rel_p"]),
        "delta_time": outputs["delta_time"],
    }


def compute_fisher_and_covariance(detector_network, parameters, res):
    fisher_matrix = detector_network.FisherMatr(parameters, res=res)
    covariance_matrix, _ = compute_covariance_matrix(fisher_matrix, cores=1)
    return fisher_matrix, covariance_matrix


def jacobian_covariance(lensing_parameters):
    fisher_matrix, covariance_matrix = compute_fisher_and_covariance(
        HLV_AGN,
        lensing_parameters,
        res=100,
    )
    transformed_covariance, transformed_parameters, transformed_keys = (
        covariance_change_variable(
            covariance_matrix,
            lensing_parameters,
            lensing_transform,
            ["R_orbit", "src_pos", "M_lz"],
        )
    )
    return fisher_matrix, covariance_matrix, transformed_covariance, transformed_parameters, transformed_keys


def direct_covariance(lensing_parameters):
    general_lensed_parameters = L1_AGN.convert_to_general_lensed_parameters(lensing_parameters)
    keys = list(general_lensed_parameters.keys()).copy()
    original_keys = keys.copy()

    fisher_matrix = HLV_LENSED.FisherMatr(general_lensed_parameters, res=200)
    cleaned_fisher_matrix = np.nan_to_num(fisher_matrix, nan=0.0)
    reduced_fisher_matrix, removed_keys = reduce_Fisher_matrix(
        cleaned_fisher_matrix,
        keys=keys,
    )
    reduced_keys = [key for key in original_keys if key not in removed_keys]

    if reduced_fisher_matrix.shape[0] == 0:
        n_parameters = len(original_keys)
        event_shape = fisher_matrix.shape[2:]
        nan_covariance = np.full((n_parameters, n_parameters) + event_shape, np.nan)
        return (
            general_lensed_parameters,
            fisher_matrix,
            nan_covariance,
            original_keys,
            removed_keys,
        )

    reduced_fisher_matrix[reduced_fisher_matrix == 0.0] = np.nan
    covariance_matrix, _ = compute_covariance_matrix(reduced_fisher_matrix, cores=1)
    return general_lensed_parameters, fisher_matrix, covariance_matrix, reduced_keys, removed_keys


def reorder_covariance(covariance, keys, desired_order):
    indices = [keys.index(key) for key in desired_order]
    index_array = np.array(indices, dtype=np.int32)
    reordered = np.take(
        np.take(covariance.astype(np.float64), index_array, axis=0),
        index_array,
        axis=1,
    )
    return reordered, [keys[index] for index in indices]


def reorder_params_dict(params_dict, desired_order):
    return {key: np.array(params_dict[key]) for key in desired_order}


def decompose_bayes_factor(full_covariance, full_params_dict, original_keys):
    ordered_covariance, _ = reorder_covariance(
        full_covariance,
        original_keys,
        AGN_PARAMETER_ORDER,
    )
    ordered_params = reorder_params_dict(full_params_dict, AGN_PARAMETER_ORDER)

    offset = np.array(
        [
            ordered_params["relative_mass"] - 1.0,
            ordered_params["delta_iota"],
            ordered_params["delta_phase"],
            ordered_params["delta_psi"],
        ]
    )
    offset = np.moveaxis(offset, -1, 0)
    ordered_covariance = np.moveaxis(ordered_covariance, -1, 0)

    n_extra = PRIOR_WIDTHS.shape[0]
    extra_covariance = ordered_covariance[:, -n_extra:, -n_extra:]
    extra_covariance_inverse = np.linalg.inv(extra_covariance)
    chi2 = np.einsum("...i,...ij,...j->...", offset, extra_covariance_inverse, offset)
    log_det_extra = np.linalg.slogdet(extra_covariance)[1]

    log_prior_at_null = -np.sum(np.log(PRIOR_WIDTHS))
    log_posterior_at_null = (
        -0.5 * n_extra * np.log(2 * np.pi)
        - 0.5 * log_det_extra
        - 0.5 * chi2
    )
    log_bayes_factor = log_prior_at_null - log_posterior_at_null

    marginal_sigmas = np.sqrt(np.diagonal(extra_covariance, axis1=-2, axis2=-1))
    marginal_correction = 2.0 * np.sum(
        np.minimum(0.0, np.log(PRIOR_WIDTHS) - np.log(marginal_sigmas)),
        axis=-1,
    )
    turning_point_correction = n_extra * np.minimum(0.0, np.log(chi2 / n_extra))
    log_det_capped = log_det_extra + np.minimum(
        marginal_correction,
        turning_point_correction,
    )
    log_posterior_at_null_capped = (
        -0.5 * n_extra * np.log(2 * np.pi)
        - 0.5 * log_det_capped
        - 0.5 * chi2
    )
    log_bayes_factor_capped = log_prior_at_null - log_posterior_at_null_capped

    return {
        "offset": offset,
        "extra_covariance": extra_covariance,
        "chi2": chi2,
        "log_det_extra": log_det_extra,
        "log_prior_at_null": log_prior_at_null,
        "log_posterior_at_null": log_posterior_at_null,
        "log_bayes_factor": log_bayes_factor,
        "marginal_sigmas": marginal_sigmas,
        "marginal_correction": marginal_correction,
        "turning_point_correction": turning_point_correction,
        "log_det_capped": log_det_capped,
        "log_posterior_at_null_capped": log_posterior_at_null_capped,
        "log_bayes_factor_capped": log_bayes_factor_capped,
    }


def run_component_checks(diagnostic):
    assert onp.isfinite(onp.asarray(diagnostic["agn_snr"])).all()
    assert onp.isfinite(onp.asarray(diagnostic["agn_fisher_diagonal"])).all()
    assert onp.isfinite(onp.asarray(diagnostic["jacobian_covariance_diagonal"])).all()
    assert onp.isfinite(onp.asarray(diagnostic["direct_covariance_diagonal"])).all()

    for key in EXTRA_PARAMETER_KEYS:
        assert onp.isfinite(onp.asarray(diagnostic["phenomenological_parameters"][key])).all()

    for key in [
        "chi2",
        "log_det_extra",
        "log_bayes_factor",
        "log_det_capped",
        "log_bayes_factor_capped",
    ]:
        assert onp.isfinite(onp.asarray(diagnostic["bayes_terms"][key])).all()


def diagnose_single_agn_event():
    lensing_parameters, y_rorbit = build_single_event_parameters()
    raw_lensing_outputs = get_raw_lensing_outputs(lensing_parameters)
    phenomenological_parameters = lensing_transform(lensing_parameters)

    agn_snr = HLV_AGN.SNR(lensing_parameters, res=1000)
    agn_fisher_matrix, agn_covariance, jacobian_covariance_matrix, transformed_parameters, transformed_keys = (
        jacobian_covariance(lensing_parameters)
    )

    direct_parameters, direct_fisher_matrix, direct_covariance_matrix, direct_keys, removed_keys = (
        direct_covariance(lensing_parameters)
    )
    general_lensed_snr = HLV_LENSED.SNR(direct_parameters, res=1000)

    agn_fisher_single = _as_single_event_matrix(agn_fisher_matrix)
    jacobian_covariance_single = _as_single_event_matrix(jacobian_covariance_matrix)
    direct_covariance_single = _as_single_event_matrix(direct_covariance_matrix)

    bayes_terms = decompose_bayes_factor(
        jacobian_covariance_matrix,
        transformed_parameters,
        transformed_keys,
    )

    diagnostic = {
        "lensing_parameters": lensing_parameters,
        "y_rorbit": y_rorbit,
        "raw_lensing_outputs": raw_lensing_outputs,
        "phenomenological_parameters": phenomenological_parameters,
        "agn_snr": agn_snr,
        "general_lensed_snr": general_lensed_snr,
        "agn_fisher_diagonal": np.diag(agn_fisher_single),
        "agn_fisher_condition_number": np.linalg.cond(agn_fisher_single),
        "agn_covariance_diagonal": np.diag(_as_single_event_matrix(agn_covariance)),
        "jacobian_covariance_diagonal": np.diag(jacobian_covariance_single),
        "jacobian_covariance_table": _covariance_matrix_table(
            jacobian_covariance_single,
            transformed_keys,
        ),
        "general_lensed_parameters": direct_parameters,
        "direct_fisher_shape": onp.asarray(direct_fisher_matrix).shape,
        "direct_covariance_diagonal": np.diag(direct_covariance_single),
        "direct_covariance_table": _covariance_matrix_table(
            direct_covariance_single,
            direct_keys,
        ),
        "direct_removed_keys": removed_keys,
        "bayes_terms": bayes_terms,
    }

    run_component_checks(diagnostic)
    return diagnostic


#def print_single_event_report(diagnostic):


if __name__ == "__main__":
    diagnostic = diagnose_single_agn_event()
    print("Single-event AGN lens diagnostic")
    print("\nInput lensing parameters")
    print(_format_scalar_dict(diagnostic["lensing_parameters"]))
    print(f"          y_Einstein = {DEFAULT_Y_EINSTEIN: .6g}")
    print(f"            y_Rorbit = {_to_scalar(diagnostic['y_rorbit']): .6g}")

    print("\nRaw lensing outputs")
    print(_format_scalar_dict(diagnostic["raw_lensing_outputs"]))

    print("\nPhenomenological lensing parameters")
    print(_format_scalar_dict(diagnostic["phenomenological_parameters"]))

    print("\nNetwork SNRs")
    print(f"               AGN SNR = {_to_scalar(diagnostic['agn_snr']): .6g}")
    print(f"    General-lensed SNR = {_to_scalar(diagnostic['general_lensed_snr']): .6g}")

    print("\nAGN Fisher / covariance summary")
    print(f"  Fisher condition number = {_to_scalar(diagnostic['agn_fisher_condition_number']): .6g}")
    print(f"        Fisher diagonal    = {onp.asarray(diagnostic['agn_fisher_diagonal'])}")
    print(f"        Covariance diagonal= {onp.asarray(diagnostic['agn_covariance_diagonal'])}")

    print("\nJacobian covariance (AGN -> phenomenological)")
    print(f"  diagonal = {onp.asarray(diagnostic['jacobian_covariance_diagonal'])}")
    print(diagnostic["jacobian_covariance_table"])

    print("\nDirect covariance (general lensed model)")
    if diagnostic["direct_removed_keys"]:
        print(f"  removed Fisher directions = {diagnostic['direct_removed_keys']}")
    print(f"  diagonal = {onp.asarray(diagnostic['direct_covariance_diagonal'])}")
    print(diagnostic["direct_covariance_table"])

    print("\nSDDR Bayes-factor terms")
    bayes_terms = diagnostic["bayes_terms"]
    print(f"  offset = {onp.asarray(bayes_terms['offset'])}")
    print(f"  extra covariance = {onp.asarray(bayes_terms['extra_covariance'])}")
    print(f"  chi2 = {onp.asarray(bayes_terms['chi2'])}")
    print(f"  log_det_extra = {onp.asarray(bayes_terms['log_det_extra'])}")
    print(f"  log_prior_at_null = {float(bayes_terms['log_prior_at_null']): .6g}")
    print(f"  log_posterior_at_null = {onp.asarray(bayes_terms['log_posterior_at_null'])}")
    print(f"  log_bayes_factor = {onp.asarray(bayes_terms['log_bayes_factor'])}")
    print(f"  marginal_sigmas = {onp.asarray(bayes_terms['marginal_sigmas'])}")
    print(f"  marginal_correction = {onp.asarray(bayes_terms['marginal_correction'])}")
    print(f"  turning_point_correction = {onp.asarray(bayes_terms['turning_point_correction'])}")
    print(f"  log_det_capped = {onp.asarray(bayes_terms['log_det_capped'])}")
    print(
        f"  log_posterior_at_null_capped = "
        f"{onp.asarray(bayes_terms['log_posterior_at_null_capped'])}"
    )
    print(f"  log_bayes_factor_capped = {onp.asarray(bayes_terms['log_bayes_factor_capped'])}")

