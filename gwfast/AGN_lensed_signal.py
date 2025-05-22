#
#    Copyright (c) 2025 Samson Leong <samson.leong@link.cuhk.edu.hk>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.

import os

from jax import config
import jax.numpy as np

# Enable 64bit on JAX, fundamental
config.update("jax_enable_x64", True)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import copy

from gwfast.gwfastGlobals import TWOPI, DAY_TO_SEC
from gwfast.gwfastUtils import (
    noise_weighted_inner_product,
    optimal_snr,
    get_model_parameters,
)
from gwfast.lensing_utils import (
    get_lensed_parameter_sets,
    get_lensing_time_delay,
    get_mag_factors,
)
from gwfast.new_signal import NewGWSignal


class AGNLensedGWSignal(NewGWSignal):
    """
    Class to compute the GW signal emitted by a coalescing binary system as seen by a detector on Earth.

    The functions defined within this class allow to get e.g. the amplitude of the signal, its phase, SNR and Fisher matrix elements.

    :param WaveFormModel wf_model: Object containing the waveform model.
    :param str psd_path: Full path to the file containing the detector's *Power Spectral Density*, PSD, or *Amplitude Spectral Density*, ASD, including the file extension. The file is assumed to have two columns, the first containing the frequencies (in :math:`\\rm Hz`) and the second containing the detector's PSD/ASD at each frequency.
    :param str optional detector_shape: The shape of the detector, to be chosen among ``'L'`` for an L-shaped detector (90°-arms) and ``'T'`` for a triangular detector (3 nested detectors with 60°-arms).
    :param float optional det_lat: Latitude of the detector, in degrees.
    :param float optional det_long: Longitude of the detector, in degrees.
    :param float optional det_xax: Angle between the bisector of the detector's arms (the first detector in the case of a triangle) and local East, in degrees.
    :param bool, optional verbose: Boolean specifying if the code has to print additional details during execution.
    :param bool, optional is_ASD: Boolean specifying if the provided file is a PSD or an ASD.
    :param bool, optional useEarthMotion: Boolean specifying if the effect of the Earth rotation has to be included in the analysis.
    :param bool, optional noMotion: Boolean specifying if the Earth should be considered fixed at ``tcoal=0``. In the case ``useEarthMotion=False`` the system is rotated depending on ``tcoal`` and then left fixed. This was needed for checks and is not to be used.
    :param float fmin: Minimum frequency to use for the grid in the analysis, in :math:`\\rm Hz`.
    :param float fmax: Maximum frequency to use for the grid in the analysis, in :math:`\\rm Hz`. The cut frequency of the waveform (which depends on the events parameters) will be used as maximum frequency if ``fmax=None`` or if it is smaller than ``fmax``.
    :param Detector optional detector: A detector object, once specified, it overrides the specified lat, long, and xax above.
    :param float detector.duty_cycle: Duty factor of the detector, between 0 and 1, representing the percentage of time the detector (each detector independently in the case of a triangular detector) is supposed to be operational.
    :param bool, optional compute2arms: Boolean specifying if, in the case of a triangular detector, the computation can be performed only in two of the instruments, using the null-stream to get the signal in the third instrument, speeding up the computation by 1/3.
    :param bool, optional jitCompileDerivs: Boolean specifying if the derivatives function has to be jit compiled. NOTE: This only works with JAX derivatives.

    """

    """
    Inputs are an object containing the waveform model, the coordinates of the detector (latitude and longitude in deg),
    its shape (L or T), the angle with respect to East of the bisector of the arms (deg)
    and its ASD or PSD (given in a .txt file containing two columns: one with the frequencies and one with the ASD or PSD values,
    remember ASD=sqrt(PSD))

    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.strain_model_keys = list(self.wf_model.ParNums.keys()) + [
            "R_orbit",
            "M_lz",
            "src_pos",
        ]

    def GWAmplitudes(self, evParams, f, rot=0.0):
        raise NotImplementedError("Yeah, someone should work on this.")

    def GWPhase(self, evParams, f):
        raise NotImplementedError("Yeah, someone should work on this.")

    def GWstrain(self, f, parameters, rot=0.0, return_single_comp=None):
        """
        Compute the full GW strain (complex) as a function of the parameters, at given frequencies.

        :param array or float f: The frequency(ies) at which to perform the calculation, in :math:`\\rm Hz`.
        :param dict parameters: The parameters dictionary to evaluate the strain at.
        :param float rot: Further rotation of the interferometer with respect to the :py:data:`self.xax` orientation, in degrees, needed for the triangular geometry.
        :param str return_single_comp: String specifying if a single component of the signal should be returned, to be chosen among ``Ap`` and ``Ac``, to return the plus and cross amplitude, :math:`A_+` and :math:`A_{\\times}`, respectively, and ``Psip`` and ``Psic``, to return the plus and cross phase, :math:`\Phi_+` and :math:`\Phi_{\\times}`, respectively.
        :return: Complete signal strain (complex), evaluated at the given parameters and frequency(ies).
        :rtype: array or float

        """
        # Full GW strain expression (complex)
        # Here we have the decompressed parameters and we put them back in a dictionary just to have an easier
        # implementation of the JAX module for derivatives

        omega = TWOPI * f * DAY_TO_SEC

        model_params = get_model_parameters(parameters, self.strain_model_keys)
        eval_params_1, eval_params_2 = get_lensed_parameter_sets(model_params)
        # Time delay and magnification
        # TODO: Check ordering of 1, 2.
        time_delay = get_lensing_time_delay(model_params)
        time_delay_phase_shift = np.exp(2j * np.pi * f * time_delay)
        mag_1, mag_2 = get_mag_factors(model_params)
        sqrt_mu_1 = np.sqrt(np.abs(mag_1))
        sqrt_mu_2 = np.sqrt(np.abs(mag_2))

        # Not sure what does this do, but it was set to zero in both cases
        # (with or without useEarthMotion)
        phiD = np.zeros_like(model_params["Mc"])

        # Moving on to combining the strain with the antenna patterns
        is_lal = self.wf_model.is_LAL

        # 22 mode waveforms
        if not (self.need_HM or is_lal):
            _, deltaT_1 = self.shifted_time(eval_params_1, f)
            phiL1 = omega * deltaT_1
            _, deltaT_2 = self.shifted_time(eval_params_2, f)
            phiL2 = omega * deltaT_2
            # Return with the simplest things
            # A hacky way to access the old GWSignal Amplitude method
            # One should just implement it in the NewSignal class
            Ap1, Ac1 = super().GWAmplitudes(eval_params_1, f, rot=rot)
            Psi1 = super().GWPhase(eval_params_1, f)
            Psi1 += phiD + phiL1

            Ap2, Ac2 = super().GWAmplitudes(eval_params_2, f, rot=rot)
            Psi2 = super().GWPhase(eval_params_2, f)
            Psi2 += phiD + phiL2

            # TODO: Check whether h = hp - i hc.
            hp1, hc1 = Ap1 * np.exp(Psi1 * 1j), 1j * Ac1 * np.exp(Psi1 * 1j)
            hp2, hc2 = Ap2 * np.exp(Psi2 * 1j), 1j * Ac2 * np.exp(Psi2 * 1j)

        else:
            phase_shift_factor = np.exp(1j * (phiD + omega * model_params["tcoal"]))

            hpc_12 = []
            for params in (eval_params_1, eval_params_2):
                iota = params["iota"]
                psi = params["psi"]
                phase = params["Phicoal"]
                theta = params["theta"]
                phi = params["phi"]

                time, deltaT = self.shifted_time(params, f)
                phiL = omega * deltaT

                Fpc = self.detector.compute_antenna_pattern(theta, phi, time, psi, rot=rot)
                hpc = self.wf_model.hphc(f, **params)
                phase_factor = phase_shift_factor * np.exp(1j * (phiL - phase))
                hp = hpc[0] * Fpc[0] * phase_factor
                hc = hpc[1] * Fpc[1] * phase_factor

                if is_lal:
                    hp *= 0.5 * (1.0 + np.cos(iota) ** 2)
                    hc *= np.cos(iota)

                hpc_12.append((hp, hc))

            ((hp1, hc1), (hp2, hc2)) = hpc_12

        hp = sqrt_mu_1 * hp1 + sqrt_mu_2 * time_delay_phase_shift * hp2
        hc = sqrt_mu_1 * hc1 + sqrt_mu_2 * time_delay_phase_shift * hc2

        if return_single_comp is not None:
            if return_single_comp == "Ap":
                return np.abs(hp)
            elif return_single_comp == "Ac":
                return np.abs(hc)
            elif return_single_comp == "Psip":
                return np.unwrap(np.angle(hp), axis=0)
            elif return_single_comp == "Psic":
                return np.unwrap(np.angle(hc), axis=0)
            elif return_single_comp == "At":
                return np.abs(hp + hc)
            elif return_single_comp == "Psit":
                return np.unwrap(np.angle(hp + hc), axis=0)
            elif return_single_comp == 'images':
                h1 = sqrt_mu_1 * (hp1 + hc1)
                h2 = sqrt_mu_2 * (hp2 + hc2) * time_delay_phase_shift
                return h1, h2
            else:
                raise ValueError(
                    "Single component to return has to be among Ap, Ac, Psip, Psic"
                )
        else:
            return hp + hc

    def _analytical_derivatives(self):
        raise NotImplementedError('Lensed waveforms have no well-defined analytical derivatives (yet)')

    def WFOverlap(
        self, WF1, WF2, evParams1, evParams2, res=1000, return_separate=False, **kwargs
    ):
        """
        Compute the *overlap* of two waveforms in a single detector on two sets of parameters, for one or multiple events.

        :param WaveFormModel WF1: Object containing the first waveform model to analyse.
        :param WaveFormModel WF2: Object containing the second waveform model to analyse.
        :param dict(array, array, ...) evParams1: Dictionary containing the parameters of the event(s) for the first waveform model, as in :py:data:`events`.
        :param dict(array, array, ...) evParams2: Dictionary containing the parameters of the event(s) for the second waveform model, as in :py:data:`events`.
        :param int res: The resolution of the frequency grid to use.
        :param bool, optional return_all: Boolean specifying if, instead of returning the overlap, the function has to return separately product at the numerator of the definition, :math:`(h_1|h_2)`, and the SNRs at the denominator. This is needed to compute the overlap for a detector network. In this case the return type is *tuple(array, array, array)*.
        :param unused kwargs: Optional arguments.

        :return: Overlap(s) of the two waveforms. The shape is :math:`(N_{\\rm events})`.
        :rtype: 1-D array

        """
        wfm_1_keys = list(WF1.ParNums.keys()) + ["R_orbit", "M_lz", "src_pos"]
        wfm_2_keys = list(WF2.ParNums.keys()) + ["R_orbit", "M_lz", "src_pos"]

        model_params_1 = get_model_parameters(evParams1, wfm_1_keys)
        model_params_2 = get_model_parameters(evParams2, wfm_2_keys)

        # The frequency cut is chosen to be the highest among the two
        fcut1 = WF1.fcut(**model_params_1)
        fcut2 = WF2.fcut(**model_params_2)

        fcutUse = np.where(fcut1 > fcut2, fcut1, fcut2)

        if self.fmax is not None:
            fcutUse = np.where(fcutUse > self.fmax, self.fmax, fcut1)
        fminarr = np.full(fcutUse.shape, self.fmin)

        fgrids = np.geomspace(fminarr, fcutUse, num=int(res))
        # Out of the provided PSD range, we use a constant value of 1, which results in completely negligible conntributions
        psd_strain_grids = self.detector.psd_interp(fgrids)

        # This is a horrible way of changing the waveform, but the fastest to implement
        WFor = copy.deepcopy(self.wf_model)

        strains = []
        SNRhs = []
        if self.detector.shape == "L":
            for model, params in zip((WF1, WF2), (model_params_1, model_params_2)):
                self.wf_model = model
                strain = self.GWstrain(fgrids, params)

                strains.append(strain)
                SNRhs.append(optimal_snr(fgrids, strain, psd_strain_grids))

            overlap_int = noise_weighted_inner_product(
                fgrids, *strains, psd_strain_grids
            )

        elif self.detector.shape == "T":
            for model, params in zip((WF1, WF2), (model_params_1, model_params_2)):
                self.wf_model = model
                h_1 = self.GWstrain(fgrids, params, rot=0.0)
                h_2 = self.GWstrain(fgrids, params, rot=60.0)
                h_3 = -(h_1 + h_2)

                strains.append((h_1, h_2, h_3))

                SNRh_1_sq = optimal_snr(fgrids, h_1, psd_strain_grids) ** 2
                SNRh_2_sq = optimal_snr(fgrids, h_2, psd_strain_grids) ** 2
                SNRh_3_sq = optimal_snr(fgrids, h_3, psd_strain_grids) ** 2
                SNRhs.append(np.sqrt(SNRh_1_sq + SNRh_2_sq + SNRh_3_sq))

            overlap_int = 0.0
            for h1_i, h2_i in zip(strains[0], strains[1]):
                overlap_int += noise_weighted_inner_product(
                    fgrids, h1_i, h2_i, psd_strain_grids
                )

        # Restore the waveform
        self.wf_model = WFor

        if return_separate:
            return overlap_int, *SNRhs
        else:
            return overlap_int / (SNRhs[0] * SNRhs[1])
