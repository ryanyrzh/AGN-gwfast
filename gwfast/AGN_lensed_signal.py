#
#    Copyright (c) 2025 Samson Leong <samson.leong@link.cuhk.edu.hk>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.

import os

from jax import config, vmap, jacrev, tree
import jax.numpy as np

# Enable 64bit on JAX, fundamental
config.update("jax_enable_x64", True)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import numpy as onp
import copy
from collections import OrderedDict

from gwfast.gwfastGlobals import TWOPI, DAY_TO_SEC
from gwfast.gwfastUtils import (
    noise_weighted_inner_product,
    optimal_snr,
    get_model_parameters,
    check_evparams,
)
from gwfast.lensing_utils import (
    get_lensed_parameter_sets,
    get_lensing_time_delay,
    get_mag_factors,
)
from gwfast.signal import GWSignal


class AGNLensedGWSignal(GWSignal):
    """
    Class to compute the GW signal emitted by a coalescing binary system as seen by a detector on Earth.

    The functions defined within this class allow to get e.g. the amplitude of the signal, its phase, SNR and Fisher matrix elements.

    :param WaveFormModel wf_model: Object containing the waveform model.
    :param str psd_path: Full path to the file containing the detector's *Power Spectral Density*, PSD, or *Amplitude Spectral Density*, ASD, including the file extension. The file is assumed to have two columns, the first containing the frequencies (in :math:`\\rm Hz`) and the second containing the detector's PSD/ASD at each frequency.
    :param str detector_shape: The shape of the detector, to be chosen among ``'L'`` for an L-shaped detector (90°-arms) and ``'T'`` for a triangular detector (3 nested detectors with 60°-arms).
    :param float det_lat: Latitude of the detector, in degrees.
    :param float det_long: Longitude of the detector, in degrees.
    :param float det_xax: Angle between the bisector of the detector's arms (the first detector in the case of a triangle) and local East, in degrees.
    :param bool, optional verbose: Boolean specifying if the code has to print additional details during execution.
    :param bool, optional is_ASD: Boolean specifying if the provided file is a PSD or an ASD.
    :param bool, optional useEarthMotion: Boolean specifying if the effect of the Earth rotation has to be included in the analysis.
    :param bool, optional noMotion: Boolean specifying if the Earth should be considered fixed at ``tcoal=0``. In the case ``useEarthMotion=False`` the system is rotated depending on ``tcoal`` and then left fixed. This was needed for checks and is not to be used.
    :param float fmin: Minimum frequency to use for the grid in the analysis, in :math:`\\rm Hz`.
    :param float fmax: Maximum frequency to use for the grid in the analysis, in :math:`\\rm Hz`. The cut frequency of the waveform (which depends on the events parameters) will be used as maximum frequency if ``fmax=None`` or if it is smaller than ``fmax``.
    :param str IntTablePath: Deprecated, not used.
    :param float detector.duty_cycle: Duty factor of the detector, between 0 and 1, representing the percentage of time the detector (each detector independently in the case of a triangular detector) is supposed to be operational.
    :param bool, optional compute2arms: Boolean specifying if, in the case of a triangular detector, the computation can be performed only in two of the instruments, using the null-stream to get the signal in the third instrument, speeding up the computation by 1/3.
    :param bool, optional jitCompileDerivs: Boolean specifying if the derivatives function has to be jit compiled.

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

    def GWstrain(
        self,
        f,
        parameters,
        rot=0.0,
        return_single_comp=None,
    ):
        """
        Compute the full GW strain (complex) as a function of the parameters, at given frequencies.

        :param array or float f: The frequency(ies) at which to perform the calculation, in :math:`\\rm Hz`.
        :param array or float Mc: The chirp mass(es), :math:`{\cal M}_c`, in units of :math:`\\rm M_{\odot}`. If ``is_m1m2=True`` this is interpreted as the primary mass, :math:`m_1`, in units of :math:`\\rm M_{\odot}`.
        :param array or float eta:  The symmetric mass ratio(s), :math:`\eta`. If ``is_m1m2=True`` this is interpreted as the secondary mass, :math:`m_2`, in units of :math:`\\rm M_{\odot}`.
        :param array or float dL: The luminosity distance(s), :math:`d_L`, in :math:`\\rm Gpc`.
        :param array or float theta: The :math:`\\theta` sky position angle(s), in :math:`\\rm rad`.
        :param array or float phi: The :math:`\phi` sky position angle(s), in :math:`\\rm rad`.
        :param array or float iota: The inclination angle(s), with respect to orbital angular momentum, :math:`\iota`, in :math:`\\rm rad`. If ``is_prec_ang=True`` this is interpreted as the inclination angle(s) with respect to total angular momentum, :math:`\\theta_{JN}`, in :math:`\\rm rad`.
        :param array or float psi: The polarisation angle(s), :math:`\psi`, in :math:`\\rm rad`.
        :param array or float tcoal: The time(s) of coalescence, :math:`t_{\\rm coal}`, as a GMST.
        :param array or float Phicoal: The phase(s) at coalescence, :math:`\Phi_{\\rm coal}`, in :math:`\\rm rad`.
        :param array or float chiS: The symmetric spin component(s), :math:`\chi_s`. If :py:class:`self.wf_model` is precessing or ``is_chi1chi2=True`` this is interpreted as the spin component(s) of the primary object(s) along the axis :math:`z`, :math:`\chi_{1,z}`. If ``is_prec_ang=True`` this is interpreted as the spin magnitude(s) of the primary object(s), :math:`\chi_1`.
        :param array or float chiA: The antisymmetric spin component(s) :math:`\chi_a`. If :py:class:`self.wf_model` is precessing or ``is_chi1chi2=True`` this is interpreted as the spin component(s) of the secondary object(s) along the axis :math:`z`, :math:`\chi_{2,z}`. If ``is_prec_ang=True`` this is interpreted as the spin magnitude(s) of the secondary object(s), :math:`\chi_2`.
        :param array or float chi1x: The spin component(s) of the primary object(s) along the axis :math:`x`, :math:`\chi_{1,x}`. If ``is_prec_ang=True`` this is interpreted as the spin tilt angle(s) of the primary object(s), :math:`\\theta_{s,1}`, in :math:`\\rm rad`.
        :param array or float chi2x: The spin component(s) of the secondary object(s) along the axis :math:`x`, :math:`\chi_{2,x}`. If ``is_prec_ang=True`` this is interpreted as the spin tilt angle(s) of the secondary object(s), :math:`\\theta_{s,2}`, in :math:`\\rm rad`.
        :param array or float chi1y: spin component(s) of the primary object(s) along the axis :math:`y`, :math:`\chi_{1,y}`. If ``is_prec_ang=True`` this is interpreted as the azimuthal angle(s) of orbital angular momentum relative to total angular momentum, :math:`\phi_{JL}`, in :math:`\\rm rad`.
        :param array or float chi2y: spin component(s) of the secondary object(s) along the axis :math:`y`, :math:`\chi_{2,y}`. If ``is_prec_ang=True`` this is interpreted as the difference(s) in azimuthal angle between spin vectors, :math:`\phi_{1,2}`, in :math:`\\rm rad`.
        :param array or float LambdaTilde: The adimensional tidal deformability(ies) of combination :math:`\\tilde{\Lambda}`.
        :param array or float deltaLambda: The adimensional tidal deformability(ies) of combination :math:`\delta\\tilde{\Lambda}`.
        :param array or float ecc: The orbital eccentricity(ies), :math:`e_0`.
        :param array or float R_orbit: The orbital radius of the BBH about the AGN lens, in Schwarszchild radii.
        :param array or float M_lz: The redshifted lens mass, in solar masses.
        :param array or float src_pos: The dimensionless source position, in Einstein radii.
        :param float rot: Further rotation of the interferometer with respect to the :py:data:`self.xax` orientation, in degrees, needed for the triangular geometry.
        :param bool, optional is_m1m2: Boolean specifying if the ``Mc`` and ``eta`` inputs should be interpreted as the primary and secondary mass(es).
        :param bool, optional is_chi1chi2: Boolean specifying if the ``chiS`` and ``chiA`` inputs should be interpreted as the primary and secondary spin components along the axis :math:`z`.
        :param bool, optional is_prec_ang: Boolean specifying if the ``iota`` input should be interpreted as the inclination angle with respect to total angular momentum, ``chiS`` and ``chiA`` as the primary and secondary spin magnitudes, ``chi1x`` and ``chi2x`` as the primary and secondary spin tilts, ``chi1y`` as the azimuthal angle of orbital angular momentum relative to total angular momentum and ``chi2y`` as the difference in azimuthal angle between spin vectors.
        :param str return_single_comp: String specifying if a single component of the signal should be returned, to be chosen among ``Ap`` and ``Ac``, to return the plus and cross amplitude, :math:`A_+` and :math:`A_{\\times}`, respectively, and ``Psip`` and ``Psic``, to return the plus and cross phase, :math:`\Phi_+` and :math:`\Phi_{\\times}`, respectively.
        :param bool, optional use_lensing: Boolean specifying if lensing-related transformations and orbital parameters should be used.
        :return: Complete signal strain (complex), evaluated at the given parameters and frequency(ies).
        :rtype: array or float

        """
        # Full GW strain expression (complex)
        # Here we have the decompressed parameters and we put them back in a dictionary just to have an easier
        # implementation of the JAX module for derivatives

        omega = TWOPI * f * DAY_TO_SEC
        ZEROS = np.zeros_like(parameters["Mc"])

        check_evparams(parameters)
        model_params = get_model_parameters(parameters, self.strain_model_keys)
        # Modifications from lensing goes the end
        eval_params_1, eval_params_2 = get_lensed_parameter_sets(model_params)
        # Time delay and magnification
        # TODO: Check ordering of 1, 2.
        time_delay = get_lensing_time_delay(model_params)
        time_delay_phase_shift = np.exp(2j * np.pi * f * time_delay)
        mag_1, mag_2 = get_mag_factors(model_params)

        # Not sure what does this do, but it was set to zero in both cases
        # (with or without useEarthMotion)
        phiD = ZEROS

        # Moving on to combining the strain with the antenna patterns
        need_HM = (self.wf_model.is_HigherModes) or (self.wf_model.is_Precessing)
        is_lal = self.wf_model.is_LAL

        if not (need_HM or is_lal):
            t1, deltaT_1 = self.shifted_time(eval_params_1, f)
            phiL1 = omega * deltaT_1
            t2, deltaT_2 = self.shifted_time(eval_params_2, f)
            phiL2 = omega * deltaT_2
            # Return with the simplest things
            Ap1, Ac1 = super().GWAmplitudes(eval_params_1, f, rot=rot)
            Psi1 = super().GWPhase(eval_params_1, f)
            Psi1 += phiD + phiL1

            Ap2, Ac2 = super().GWAmplitudes(eval_params_2, f, rot=rot)
            Psi2 = super().GWPhase(eval_params_2, f)
            Psi2 += phiD + phiL2

            # TODO: Check whether h = hp - i hc.
            hp1, hc1 = Ap1 * np.exp(Psi1 * 1j), 1j * Ac1 * np.exp(Psi1 * 1j)
            hp2, hc2 = Ap2 * np.exp(Psi2 * 1j), 1j * Ac2 * np.exp(Psi2 * 1j)

            hp = (
                np.sqrt(np.abs(mag_1)) * hp1
                + np.sqrt(np.abs(mag_2)) * time_delay_phase_shift * hp2
            )
            hc = (
                np.sqrt(np.abs(mag_1)) * hc1
                + np.sqrt(np.abs(mag_2)) * time_delay_phase_shift * hc2
            )
            Ap, Ac = np.abs(hp), np.abs(hc)

            Psi = np.unwrap(np.angle(hp + hc), axis=0)

            if return_single_comp is not None:
                if return_single_comp == "Ap":
                    return Ap
                elif return_single_comp == "Ac":
                    return Ac
                elif return_single_comp == "Psip":
                    return Psi  # np.unwrap(Psi)
                elif return_single_comp == "Psic":
                    return Psi + np.pi * 0.5  # np.unwrap(Psi + np.pi*0.5)
                elif return_single_comp == "At":
                    return np.abs(Ap + 1j * Ac)
                elif return_single_comp == "Psit":
                    return Psi + np.arctan2(np.real(Ac), np.real(Ap))
                else:
                    raise ValueError(
                        "Single component to return has to be among Ap, Ac, Psip, Psic"
                    )
            else:
                return (Ap + 1j * Ac) * np.exp(Psi * 1j)
            # return np.sqrt(Ap*Ap + Ac*Ac)*np.exp((Psi+phiP)*1j)

        phase_shift_factor = np.exp(1j * (phiD + omega * model_params["tcoal"]))

        hpc_12 = []
        for params in (eval_params_1, eval_params_2):
            iota = params["iota"]
            psi = params["psi"]
            phase = params["phase"]
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

        hp = np.sqrt(np.abs(mag_1)) * hpc_12[0][0] + \
            np.sqrt(np.abs(mag_2)) * time_delay_phase_shift * hpc_12[1][0]
        hc = np.sqrt(np.abs(mag_1)) * hpc_12[0][1] + \
            np.sqrt(np.abs(mag_2)) * time_delay_phase_shift * hpc_12[1][1]

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
            else:
                raise ValueError(
                    "Single component to return has to be among Ap, Ac, Psip, Psic"
                )
        else:
            return hp + hc

    def SNRInteg(self, parameters, res=1000, return_all=False):
        """
        Compute the *signal-to-noise-ratio*, SNR, as a function of the parameters of the event(s).

        :param dict(array, array, ...) evParams: Dictionary containing the parameters of the event(s), as in :py:data:`events`.
        :param int res: The resolution of the frequency grid to use.
        :param bool, optional return_all: Boolean specifying if, in the case of a triangular detector, the SNRs of the individual instruments have to be returned separately. In this case the return type is *list(array, array, array)*.

        :return: SNR(s) as a function of the parameters of the event(s). The shape is :math:`(N_{\\rm events})`.
        :rtype: 1-D array

        """
        # SNR calculation performing the frequency integral for each signal
        # This is computationally more expensive, but needed for complex waveform models
        if self.detector.duty_cycle is not None:
            onp.random.seed(self.seedUse)

        # TODO: Deprecate check_evaparams
        check_evparams(parameters)
        model_params = get_model_parameters(parameters, self.strain_model_keys)
        params_shape = model_params["Mc"].shape

        fcut = self.wf_model.fcut(**model_params)
        if self.fmax is not None:
            fcut = np.where(fcut > self.fmax, self.fmax, fcut)
        fminarr = np.full(fcut.shape, self.fmin)
        fgrids = np.geomspace(fminarr, fcut, num=int(res))

        allSNRsq = []
        # Out of the provided PSD range, we use a constant value of 1, which results in completely negligible conntributions
        psd_strain_grids = self.detector.psd_interp(fgrids)

        if self.detector.shape == "L":
            Atot = self.GWstrain(fgrids, parameters, return_single_comp="At") ** 2
            SNRsq = np.trapezoid(Atot / psd_strain_grids, fgrids, axis=0)
            if self.detector.duty_cycle is not None:
                SNRsq *= self.duty_cycle_mask(params_shape)
            allSNRsq.append(SNRsq)
        elif self.detector.shape == "T":
            if not self.compute2arms:
                for i in range(3):
                    Atot = self.GWstrain(
                        fgrids, parameters, rot=i * 60.0,
                        return_single_comp="At",
                    ) ** 2
                    tmpSNRsq = np.trapezoid(Atot / psd_strain_grids, fgrids, axis=0)
                    if self.detector.duty_cycle is not None:
                        tmpSNRsq = tmpSNRsq * self.duty_cycle_mask(params_shape)
                    allSNRsq.append(tmpSNRsq)
            else:
                # The signal in 3 arms sums to zero for geometrical reasons, so we can use this to skip some calculations
                h1 = self.GWstrain(fgrids, parameters)
                h2 = self.GWstrain(fgrids, parameters, rot=60.0)
                Atot1 = abs(h1) ** 2
                Atot2 = abs(h2) ** 2
                Atot3 = abs(h1 + h2) ** 2

                for amplitude in (Atot1, Atot2, Atot3):
                    snr_sq = np.trapezoid(amplitude / psd_strain_grids, fgrids, axis=0)
                    if self.detector.duty_cycle is not None:
                        snr_sq *= self.duty_cycle_mask(params_shape)
                    allSNRsq.append(snr_sq)

        allSNRsq = np.array(allSNRsq)

        # The factor of two arises by cutting the integral from 0 to infinity
        if self.detector.shape == "T":
            return (
                2 * np.sqrt(allSNRsq)
                if return_all
                else 2 * np.sqrt(allSNRsq.sum(axis=0))
            )
        return np.squeeze(2 * np.sqrt(allSNRsq), axis=0)

    def FisherMatr(
        self,
        evParams,
        res=1000,
        df=None,
        spacing="geom",
        computeDerivFinDiff=False,
        computeAnalyticalDeriv=False,
        return_all=False,
        **kwargs,
    ):
        """
        Compute the *Fisher information matrix*, FIM, as a function of the parameters of the event(s).

        :param dict(array, array, ...) evParams: Dictionary containing the parameters of the event(s), as in :py:data:`events`.
        :param int res: The resolution of the frequency grid to use.
        :param float df: The spacing of the frequency grid to use, in :math:`\\rm Hz`. Alternative to ``res``.
        :param str spacing: The kind of spacing of the frequency grid to use. If ``'geom'`` the grid will be spaced evenly on a log scale (geometric progression), if ``'lin'`` it will be spaced evenly on a linear scale.
        :param bool, optional use_m1m2: Boolean specifying if the FIM has to be computed with respect to the individual masses ``m1`` and ``m2`` rather than ``Mc`` and ``eta``.
        :param bool, optional use_chi1chi2: Boolean specifying if, in the non-precessing case, the FIM has to be computed with respect to the individual spins ``chi1z`` and ``chi2z`` rather than ``chiS`` and ``chiA``.
        :param bool, optional use_prec_ang: Boolean specifying if, in the precessing case, the FIM has to be computed with respect to the spin angular variables rather than the spin cartesian components.
        :param bool, optional computeDerivFinDiff: Boolean specifying if the derivatives have to be computed using numerical differentiation (finite differences) through the `numdifftools <https://github.com/pbrod/numdifftools>`_ package.
        :param bool, optional computeAnalyticalDeriv: Boolean specifying if the derivatives with respect to ``dL``, ``theta``, ``phi``, ``psi``, ``tcoal``, ``Phicoal`` and ``iota`` (the latter only for the fundamental mode in the non-precessing case) have to be computed analytically. This considerably speeds up the calculation and provides better accuracy.
        :param bool, optional return_all: Boolean specifying if, in the case of a triangular detector, the FIMs of the individual instruments have to be returned separately. In this case the return type is *list(array, array, array)*.
        :param kwargs: Optional arguments to be passed to :py:class:`gwfast.signal.GWSignal._SignalDerivatives`, such as ``methodNDT``.
        :return: FIM(s) as a function of the parameters of the event(s). The shape is :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters}`, :math:`N_{\\rm events})`.
        :rtype: 3-D array

        """
        # If use_m1m2=True the Fisher is computed w.r.t. m1 and m2, not Mc and eta
        # If use_chi1chi2=True the Fisher is computed w.r.t. chi1z and chi2z, not chiS and chiA
        if self.detector.duty_cycle is not None:
            onp.random.seed(self.seedUse)

        fcut = self.wf_model.fcut(**evParams)

        if self.fmax is not None:
            fcut = np.where(fcut > self.fmax, self.fmax, fcut)

        fminarr = np.full(fcut.shape, self.fmin)
        if res is None and df is not None:
            res = np.floor(np.real((1 + (fcut - fminarr) / df)))
            res = np.amax(res)
        elif res is None and df is None:
            raise ValueError("Provide either resolution in frequency or step size.")
        if spacing == "lin":
            fgrids = np.linspace(fminarr, fcut, num=int(res))
        elif spacing == "geom":
            fgrids = np.geomspace(fminarr, fcut, num=int(res))

        if (self.wf_model.is_LAL) and (not computeDerivFinDiff):
            computeDerivFinDiff = True
            if self.verbose:
                print(
                    "Using LAL or TEOBResumS waveforms it is not possible to compute the derivatives using JAX automatic differentiation routines, being the functions written in C. Proceeding using numdifftools for numerical differentiation (finite differences)"
                )

        allFishers = []

        # Convert to OrderDict to preserve order
        evParams = OrderedDict(evParams)

        if self.detector.shape == "L":
            # Compute derivatives
            jacobian_dict = self._jax_derivative(fgrids, evParams)
            # Change the units of the tcoal derivative from days to seconds (this improves conditioning)
            jacobian_dict["tcoal"] /= DAY_TO_SEC
            fisher_mat = self.convert_Jacobian_to_Fisher(jacobian_dict, fgrids)

            if self.detector.duty_cycle is not None:
                fisher_mat *= self.duty_cycle_mask(fisher_mat.shape[2])
            allFishers.append(fisher_mat)
        else:
            # Fisher = onp.zeros((nParams,nParams,len(Mc)))
            if not self.compute2arms:
                for i in range(3):
                    # Change rot and compute derivatives
                    jacobian_dict = self._jax_derivative(fgrids, evParams, rot=i * 60.0)
                    # Change the units of the tcoal derivative from days to seconds (this improves conditioning)
                    jacobian_dict["tcoal"] /= DAY_TO_SEC
                    fisher_mat = self.convert_Jacobian_to_Fisher(jacobian_dict, fgrids)
                    if self.detector.duty_cycle is not None:
                        fisher_mat *= self.duty_cycle_mask(fisher_mat.shape[2])
                    allFishers.append(fisher_mat)
                    # Fisher += tmpFisher
            else:
                # The signal in 3 arms sums to zero for geometrical reasons, so we can use this to skip some calculations
                jacobian_dict_1 = self._jax_derivative(fgrids, evParams, rot=0.0)
                jacobian_dict_1["tcoal"] /= DAY_TO_SEC
                fisher_mat_1 = self.convert_Jacobian_to_Fisher(jacobian_dict_1, fgrids)
                if self.detector.duty_cycle is not None:
                    fisher_mat_1 *= self.duty_cycle_mask(fisher_mat_1.shape[2])
                allFishers.append(fisher_mat_1)

                jacobian_dict_2 = self._jax_derivative(fgrids, evParams, rot=60.0)
                jacobian_dict_2["tcoal"] /= DAY_TO_SEC
                fisher_mat_2 = self.convert_Jacobian_to_Fisher(jacobian_dict_2, fgrids)
                if self.detector.duty_cycle is not None:
                    fisher_mat_1 *= self.duty_cycle_mask(fisher_mat_2.shape[2])
                allFishers.append(fisher_mat_2)

                jacobian_dict_3 = {
                    key: -(jacobian_dict_1[key] + jacobian_dict_2[key])
                    for key in jacobian_dict_1.keys()
                }
                fisher_mat_3 = self.convert_Jacobian_to_Fisher(jacobian_dict_3, fgrids)
                if self.detector.duty_cycle is not None:
                    fisher_mat_3 *= self.duty_cycle_mask(fisher_mat_3.shape[2])
                allFishers.append(fisher_mat_3)

        if return_all:
            return allFishers
        elif self.detector.shape == "T":
            return onp.array(allFishers).sum(axis=0)
        else:
            return allFishers[0]

    def _jax_derivative(self, freq_grid, parameters, rot=0.0):
        """
        Forget about analytic derivatives or finite differencing, just use JAX.

        Assuming shape of freq_grid is (N_freq, N_params).
        """
        print(parameters.keys())
        if self.wf_model.is_holomorphic:
            return vmap(jacrev(self.GWstrain, argnums=1, holomorphic=True))(
                    freq_grid.T, parameters, rot)

        def real_strain(freqs, params):
            return self.GWstrain(freqs, params, rot).real

        def imag_strain(freqs, params):
            return self.GWstrain(freqs, params, rot).imag

        real_deriv = vmap(jacrev(real_strain, argnums=1))(freq_grid.T, parameters)
        imag_deriv = vmap(jacrev(imag_strain, argnums=1))(freq_grid.T, parameters)
        return OrderedDict(
            {key: real_deriv[key] + 1j * imag_deriv[key] for key in parameters.keys()}
        )

    def convert_Jacobian_to_Fisher(self, jacobian_dict, freqs_grid):
        # The matrix has shape: (N_params, param_len, N_freq)
        jacobian_mat = np.array(tree.leaves(jacobian_dict))

        pre_fisher_mat = jacobian_mat[:, :, None, :].conj() * \
            jacobian_mat.transpose(1, 0, 2)
        pre_fisher_mat = np.swapaxes(pre_fisher_mat, 1, 2)

        fisher_shape = pre_fisher_mat.shape[:-1]
        fisher_mat = onp.zeros(fisher_shape)

        freqs_grid_T = freqs_grid.T
        psd_grids = self.detector.psd_interp(freqs_grid_T)
        for row, col in zip(*np.triu_indices(fisher_shape[0])):
            fisher_mat[row, col] = \
                4 * np.trapezoid(
                    pre_fisher_mat[row, col] / psd_grids,
                    freqs_grid_T, axis=1).real

            if row != col:
                fisher_mat[col, row] = fisher_mat[row, col]

        return fisher_mat

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
        # Checks on imput parameters for waveforms
        check_evparams(evParams1)
        check_evparams(evParams2)

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
