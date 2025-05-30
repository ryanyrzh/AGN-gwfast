#
#    Copyright (c) 2025 Samson Leong <samson.leong@link.cuhk.edu.hk>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.

import os
from functools import partial

from jax import config, vmap, jacrev, tree, jit, device_count, local_device_count
import jax.numpy as np

# Enable 64bit on JAX, fundamental
config.update("jax_enable_x64", True)

import copy
from collections import OrderedDict
import numpy as onp
from scipy.integrate import cumulative_trapezoid
import numdifftools as ndt
from numdifftools.step_generators import MaxStepGenerator

from gwfast.gwfastGlobals import TWOPI, DAY_TO_SEC, DEG_TO_RAD
from gwfast.gwfastUtils import (
    noise_weighted_inner_product,
    optimal_snr,
    get_model_parameters,
    apply_psi_rotation,
    ra_dec_from_th_phi_rad,
)
from gwfast.detector import Detector


class NewGWSignal(object):
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

    def __init__(
        self,
        wf_model,
        psd_path=None,
        detector_shape="T",
        det_lat=40.44,
        det_long=9.45,
        det_xax=0.0,
        verbose=True,
        useEarthMotion=False,
        noMotion=False,  # use only for checks
        fmin=2.0,
        fmax=None,
        detector=None,
        init_params = {},
        DutyFactor=None,
        compute2arms=True,
        jitCompileDerivs=False,
    ):

        if (useEarthMotion) and (wf_model.objType == "BBH") and (verbose):
            print(
                "WARNING: the Earth's motion gives a negligible contribution for BBH signals, consider switching it off to make the code run faster"
            )
        if (not useEarthMotion) and (wf_model.objType == "BNS") and (verbose):
            print(
                "WARNING: the motion of Earth gives a relevant contribution for BNS signals, consider switching it on"
            )
        if (not useEarthMotion) and (wf_model.objType == "NSBH") and (verbose):
            print(
                "WARNING: the motion of Earth gives a relevant contribution for NSBH signals, consider switching it on"
            )

        self.verbose = verbose
        self.wf_model = wf_model
        self.strain_model_keys = list(self.wf_model.ParNums.keys())
        self.fmin = fmin  # Hz
        self.fmax = fmax  # Hz or None

        self.useEarthMotion = useEarthMotion
        self.noMotion = noMotion
        if self.noMotion and self.useEarthMotion:
            print("noMotion and useEarthMotion are True. switching off useEarthMotion ")
            self.useEarthMotion = False

        self.compute2arms = compute2arms

        if detector is None:
            self.detector = Detector(
                "ifo",
                det_lat,
                det_long,
                det_xax,
                detector_shape,
                DutyFactor,
                psd_path,
                verbose=verbose,
            )
        else:
            self.detector = detector

        mask = self.detector.psd_frequencies >= self.fmin
        if self.fmax is not None:
            mask *= self.detector.psd_frequencies <= self.fmax

        masked_freqs = self.detector.psd_frequencies[mask]
        self.strainInteg = cumulative_trapezoid(
            masked_freqs ** (-7.0 / 3.0) / self.detector.psd_array[mask],
            masked_freqs,
            initial=0,
        )

        onp.random.seed(None)
        self.seedUse = onp.random.randint(2**32 - 1, size=1)
        self.jitCompileDerivs = jitCompileDerivs

        # These are initial parameters that are for general use,
        # so they can be over-complete. 
        self.additional_params = {}
        self.init_params = {
            "Mc": 77.23905294, "eta": 0.20586622,
            "chi1x": 0.1, "chi1y": 0.1, "chi1z": 0.2018924,
            "chi2x": 0.05, "chi2y": -0.01, "chi2z": -0.68859213,
            "chis": 0.2018924, "chia": -0.68859213,
            "dL": 22.68426174, "psi": 3.11843169,
            "iota": 4.48411048, "Phicoal": 3.28297867,
            "theta": 3.00702251, "phi": 0.90252645,
            "Lambda1": 300.0, "Lambda2": 300.0,
            "tcoal": 0.0, "ecc": 0.0,
        }
        self.init_params.update(init_params)

        if self.wf_model.is_LAL:
            self.signal_derivatives = self._signal_derivatives
        else:
            self._init_jax()

    def _init_jax(self):
        """
        JAX initialisation method
        """
        if self.verbose:
            print("Initializing jax...")
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
        # os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count=8'
        if self.verbose:
            print(f"Jax local device count: {local_device_count():d}")
            print(f"Jax device count: {device_count():d}")

        if self.jitCompileDerivs:
            self.signal_derivatives = jit(
                self._signal_derivatives,
                static_argnames=[
                    "computeAnalyticalDeriv",
                    "computeDerivFinDiff",
                ],
            )
        else:
            self.signal_derivatives = self._signal_derivatives

        inj_params_init = {key: np.array([val]) for key, val in self.init_params.items()}

        _verbose = self.verbose
        self.verbose = False
        _detector_shape = self.detector.shape
        self.detector.shape = "L"  # Get a faster Initialization with an L
        _strain_model_keys = self.strain_model_keys
        self.strain_model_keys = list(inj_params_init.keys())
        _ = self.SNRInteg(inj_params_init, res=10)
        _ = self.FisherMatr(inj_params_init, res=10)

        # Restore the original values
        self.verbose = _verbose
        self.detector.shape = _detector_shape
        self.strain_model_keys = _strain_model_keys

    @property
    def need_HM(self):
        """
        Dynamically evaluate whether the waveform model needs
        higher harmonics
        """
        return (self.wf_model.is_HigherModes) or (self.wf_model.is_Precessing)

    def shifted_time(self, parameters, frequencies):
        theta = parameters["theta"]
        phi = parameters["phi"]
        tcoal = parameters["tcoal"]

        if self.noMotion:
            time = 0.0
        elif self.useEarthMotion:
            time = (
                tcoal - self.wf_model.tau_star(frequencies, **parameters) / DAY_TO_SEC
            )
        else:
            time = tcoal
        delta_t = self.detector.compute_geocent_deltat(theta, phi, time)
        return time + delta_t, delta_t

    def duty_cycle_mask(self, shape):
        """
        Generate a (new) duty-cycle mask for the given shape.
        """
        return onp.random.random(shape) > self.detector.duty_cycle

    def GWAmplitudes(self, parameters, freqs, rot=0.0):
        """
        Compute the amplitude of the signal(s) as seen by the detector, as a function of the parameters, at given frequencies.

        :param dict(array, array, ...) parameters: Dictionary containing the parameters of the event(s), as in :py:data:`events`.
        :param array or float freqs: The frequency(ies) at which to perform the calculation, in :math:`\\rm Hz`.
        :param float rot: Further rotation of the interferometer with respect to the :py:data:`self.xax` orientation, in degrees, needed for the triangular geometry.
        :return: Plus and cross amplitudes at the detector, evaluated at the given parameters and frequency(ies).
        :rtype: tuple(array, array) or tuple(float, float)

        """
        # evParams are all the parameters characterizing the event(s) under exam. It has to be a dictionary containing the entries:
        # Mc -> chirp mass (Msun), dL -> luminosity distance (Gpc), theta & phi -> sky position (rad), iota -> inclination angle of orbital angular momentum to l.o.s toward the detector,
        # psi -> polarisation angle, tcoal -> time of coalescence as GMST (fraction of days), eta -> symmetric mass ratio, Phicoal -> GW frequency at coalescence.
        # chi1z, chi2z -> dimensionless spin components aligned to orbital angular momentum [-1;1], Lambda1,2 -> tidal parameters of the objects,
        # f is the frequency (Hz)

        theta = parameters["theta"]
        phi = parameters["phi"]
        iota = parameters["iota"]
        psi = parameters["psi"]

        time, _ = self.shifted_time(parameters, freqs)
        Fp, Fc = self.detector.compute_antenna_pattern(theta, phi, time, psi, rot=rot)

        if self.need_HM:
            # If the waveform includes higher modes or precessing spins,
            # it is not possible to compute amplitude and phase separately, make all together
            hp, hc = self.wf_model.hphc(freqs, **parameters)
            Ap, Ac = abs(hp) * Fp, abs(hc) * Fc
        else:
            wfAmpl = self.wf_model.Ampl(freqs, **parameters)
            Ap = wfAmpl * Fp * 0.5 * (1.0 + (np.cos(iota)) ** 2)
            Ac = wfAmpl * Fc * np.cos(iota)

        return Ap, Ac

    def GWPhase(self, parameters, freqs):
        """
        Compute the complete phase of the signal(s), as a function of the parameters, at given frequencies.

        :param dict(array, array, ...) parameters: Dictionary containing the parameters of the event(s), as in :py:data:`events`.
        :param array or float freqs: The frequency(ies) at which to perform the calculation, in :math:`\\rm Hz`.

        :return: Complete signal phase, evaluated at the given parameters and frequency(ies).
        :rtype: array or float

        """
        # Phase of the GW signal
        tcoal, Phicoal = parameters["tcoal"], parameters["Phicoal"]
        PhiGw = self.wf_model.Phi(freqs, **parameters)

        return TWOPI * freqs * (tcoal * DAY_TO_SEC) - Phicoal - PhiGw

    def GWstrain(self, f, parameters, rot=0.0, return_single_comp=None):
        """
        Compute the full GW strain (complex) as a function of the parameters, at given frequencies.

        :param array or float f: The frequency(ies) at which to perform the calculation, in :math:`\\rm Hz`.
        :param dict parameters: The parameters dictionary to evaluate the strain at, could be dict of array or float.
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
        time, deltaT = self.shifted_time(model_params, f)
        phiL = omega * deltaT

        # Not sure what does this do, but it was set to zero in both cases
        # (with or without useEarthMotion)
        phiD = np.zeros_like(model_params["Mc"])

        # Moving on to combining the strain with the antenna patterns
        is_lal = self.wf_model.is_LAL

        if not (self.need_HM or is_lal):
            # Return with the simplest things
            Ap, Ac = self.GWAmplitudes(model_params, f, rot=rot)
            Psi = self.GWPhase(model_params, f)
            Psi += phiD + phiL

            # TODO: Check whether h = hp - i hc.
            hp, hc = Ap * np.exp(Psi * 1j), 1j * Ac * np.exp(Psi * 1j)

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
                # TODO: Check conventions
                return (Ap + 1j * Ac) * np.exp(Psi * 1j)

        phase_shift_factor = np.exp(1j * (phiD + omega * model_params["tcoal"]))

        iota = model_params["iota"]
        psi = model_params["psi"]
        phase = model_params["phase"]
        theta = model_params["theta"]
        phi = model_params["phi"]

        Fpc = self.detector.compute_antenna_pattern(theta, phi, time, psi, rot=rot)
        hpc = self.wf_model.hphc(f, **model_params)
        phase_factor = phase_shift_factor * np.exp(1j * (phiL - phase))
        hp = hpc[0] * Fpc[0] * phase_factor
        hc = hpc[1] * Fpc[1] * phase_factor

        if is_lal:
            hp *= 0.5 * (1.0 + np.cos(iota) ** 2)
            hc *= np.cos(iota)

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

        :param dict(array, array, ...) parameters: Dictionary containing the parameters of the event(s), as in :py:data:`events`.
        :param int res: The resolution of the frequency grid to use.
        :param bool, optional return_all: Boolean specifying if, in the case of a triangular detector, the SNRs of the individual instruments have to be returned separately. In this case the return type is *list(array, array, array)*.

        :return: SNR(s) as a function of the parameters of the event(s). The shape is :math:`(N_{\\rm events})`.
        :rtype: 1-D array

        """
        # SNR calculation performing the frequency integral for each signal
        # This is computationally more expensive, but needed for complex waveform models
        if self.detector.duty_cycle is not None:
            onp.random.seed(self.seedUse)

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
                        fgrids, parameters, rot=i * 60.0, return_single_comp="At"
                    )
                    Atot = Atot**2
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

    def _signal_derivatives(
        self,
        freq_grid,
        parameters,
        rot=0.0,
        computeDerivFinDiff=False,
        computeAnalyticalDeriv=False,
    ):
        if not computeDerivFinDiff:
            jacobian_dict = self._jax_derivative(freq_grid, parameters, rot=rot)

        else:
            finite_diff_jacobian = self._finite_difference(
                freq_grid, parameters, rot=rot
            )

            if computeAnalyticalDeriv:
                analytic_jacobian = self._analytical_derivatives(
                    freq_grid, parameters, rot=rot
                )
                if analytic_jacobian["iota"] is None:
                    # This is when the waveform has HM (or precessing)
                    analytic_jacobian.pop("iota")
                finite_diff_jacobian.update(analytic_jacobian)

            jacobian_dict = finite_diff_jacobian

        if "tcoal" in jacobian_dict.keys():
            # Change the units of the tcoal derivative from days to seconds (this improves conditioning)
            # Not sure if this matches with description tho.
            jacobian_dict["tcoal"] /= DAY_TO_SEC

        return jacobian_dict

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

        # Need to compute the model_params once first
        model_params = get_model_parameters(evParams, self.strain_model_keys)
        fcut = self.wf_model.fcut(**model_params)

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

        deriv_kwargs = dict(
            freq_grid=fgrids,
            parameters=OrderedDict(evParams),  # Convert to OrderDict to preserve order
            computeDerivFinDiff=computeDerivFinDiff,
            computeAnalyticalDeriv=computeAnalyticalDeriv,
        )

        if self.detector.shape == "L":
            # Compute derivatives
            jacobian_dict = self.signal_derivatives(**deriv_kwargs)
            fisher_mat = self.convert_Jacobian_to_Fisher(jacobian_dict, fgrids)

            if self.detector.duty_cycle is not None:
                fisher_mat *= self.duty_cycle_mask(fisher_mat.shape[2])
            allFishers.append(fisher_mat)
        else:
            # Fisher = onp.zeros((nParams,nParams,len(Mc)))
            if not self.compute2arms:
                for i in range(3):
                    # Change rot and compute derivatives
                    jacobian_dict = self.signal_derivatives(
                        **deriv_kwargs, rot=i * 60.0
                    )
                    fisher_mat = self.convert_Jacobian_to_Fisher(jacobian_dict, fgrids)
                    if self.detector.duty_cycle is not None:
                        fisher_mat *= self.duty_cycle_mask(fisher_mat.shape[2])
                    allFishers.append(fisher_mat)
            else:
                # The signal in 3 arms sums to zero for geometrical reasons,
                # so we can use this to skip some calculations
                jacobian_dict_1 = self.signal_derivatives(**deriv_kwargs, rot=0.0)
                fisher_mat_1 = self.convert_Jacobian_to_Fisher(jacobian_dict_1, fgrids)
                if self.detector.duty_cycle is not None:
                    fisher_mat_1 *= self.duty_cycle_mask(fisher_mat_1.shape[2])
                allFishers.append(fisher_mat_1)

                jacobian_dict_2 = self.signal_derivatives(**deriv_kwargs, rot=60.0)
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
        if self.wf_model.is_holomorphic:
            cplx_freq_grid = freq_grid.astype("complex128")
            cplx_parameters = {
                key: val.astype("complex128") for key, val in parameters.items()
            }
            return vmap(
                jacrev(partial(self.GWstrain, rot=rot), argnums=1, holomorphic=True)
            )(cplx_freq_grid.T, cplx_parameters)

        def real_strain(freqs, params):
            return self.GWstrain(freqs, params, rot).real

        def imag_strain(freqs, params):
            return self.GWstrain(freqs, params, rot).imag

        real_deriv = vmap(jacrev(real_strain, argnums=1))(freq_grid.T, parameters)
        imag_deriv = vmap(jacrev(imag_strain, argnums=1))(freq_grid.T, parameters)
        return OrderedDict(
            {key: real_deriv[key] + 1j * imag_deriv[key] for key in parameters.keys()}
        )

    def _GWstrain_wrapper(self, param_values, param_keys, freqs, rot=0.0):
        parameters = dict(zip(param_keys, param_values))
        return self.GWstrain(freqs, parameters, rot)

    def _finite_difference(
        self,
        freq_grid,
        parameters,
        rot=0.0,
        step=MaxStepGenerator(base_step=1e-5),
        method="central",
    ):

        jacobian_obj = ndt.Jacobian(
            self._GWstrain_wrapper, step=step, method=method, order=4, n=1
        )
        jacobian = np.asarray(
            jacobian_obj(
                list(parameters.values()), list(parameters.keys()), freq_grid, rot
            )
        )
        if len(jacobian.shape) == 2:  # len(Mc) == 1:
            jacobian = jacobian[:, :, None]
        jacobian = jacobian.transpose(1, 2, 0)

        return OrderedDict(dict(zip(parameters.keys(), jacobian)))

    def convert_Jacobian_to_Fisher(self, jacobian_dict, freqs_grid):
        # The matrix has shape: (N_params, param_len, N_freq)
        jacobian_mat = np.array(tree.leaves(jacobian_dict))

        pre_fisher_mat = jacobian_mat[:, :, None, :].conj() * jacobian_mat.transpose(
            1, 0, 2
        )
        pre_fisher_mat = np.swapaxes(pre_fisher_mat, 1, 2)

        fisher_shape = pre_fisher_mat.shape[:-1]
        fisher_mat = onp.zeros(fisher_shape)

        freqs_grid_T = freqs_grid.T
        psd_grids = self.detector.psd_interp(freqs_grid_T)
        for row, col in zip(*np.triu_indices(fisher_shape[0])):
            fisher_mat[row, col] = np.trapezoid(
                pre_fisher_mat[row, col] / psd_grids, freqs_grid_T, axis=1
            ).real
            fisher_mat[row, col] *= 4.0

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
        wfm_1_keys = list(WF1.ParNums.keys() | self.additional_params.keys())
        wfm_2_keys = list(WF2.ParNums.keys() | self.additional_params.keys())

        # This step is needed only for computation of `fcut`
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

    def _analytical_derivatives(self, freqs, parameters, rot=0.0):
        """
        Compute analytical derivatives with respect to ``dL``, ``theta``, ``phi``, ``psi``, ``tcoal``, ``Phicoal`` and ``iota`` (the latter only for the fundamental mode in the non-precessing case).

        :param array or float freqs: The frequency(ies) at which to perform the calculation, in :math:`\\rm Hz`.
        :param float rot: Further rotation of the interferometer with respect to the :py:data:`self.xax` orientation, in degrees, needed for the triangular geometry.
        :return: Analytical derivatives with respect to ``dL``, ``theta``, ``phi``, ``iota``, ``psi``, ``tcoal`` and ``Phicoal``. If the :py:class:`self.wf_model` is precessing or includes higher order modes the derivative with respect to ``iota`` will be ``None``
        :rtype: tuple(array, array, array, array, array, array, array)

        """
        omega = TWOPI * freqs * DAY_TO_SEC

        model_params = get_model_parameters(parameters, self.strain_model_keys)

        iota = parameters.get("iota", None)
        theta = parameters.get("theta", None)
        phi = parameters.get("phi", None)
        psi = parameters.get("psi", None)
        tcoal = parameters.get("tcoal", None)
        Phicoal = parameters.get("Phicoal", None)
        dL = parameters.get("dL", None)

        if (not self.wf_model.is_HigherModes) and (not self.wf_model.is_Precessing):
            wfPhiGw = self.wf_model.Phi(freqs, **model_params)
            wfAmpl = self.wf_model.Ampl(freqs, **model_params)
            wfhpc = wfAmpl * np.exp(-1j * wfPhiGw)
            wfhp = wfhpc * 0.5 * (1.0 + np.cos(iota) ** 2)
            wfhc = 1j * wfhpc * np.cos(iota)
        else:
            # If the waveform includes higher modes, it is not possible to compute amplitude and phase separately, make all together
            wfhp, wfhc = self.wf_model.hphc(freqs, **model_params)

        phiD = np.zeros_like(parameters["Mc"])
        t, tmpDeltLoc = self.shifted_time(model_params, freqs)
        phiL = (TWOPI * freqs) * tmpDeltLoc

        rot_rad = rot * DEG_TO_RAD
        sin_angbtwArms = np.sin(self.angbtwArms)

        ras, decs = ra_dec_from_th_phi_rad(theta, phi)
        Fpc = self.detector.compute_antenna_pattern(theta, phi, t, psi, rot)

        phase = 1j * (omega * tcoal - Phicoal + phiD + phiL)
        _hp = wfhp * np.exp(phase)
        _hc = wfhc * np.exp(phase)

        hp, hc = Fpc[0] * _hp, Fpc[1] * _hc

        def psi_par_deriv():
            cos_2psi = np.cos(2 * psi)
            sin_2psi = np.sin(2 * psi)
            dpsi_rotation = -2 * np.array([[sin_2psi, -cos_2psi], [cos_2psi, sin_2psi]])
            ab_factors, _ = self.detector._compute_ab_factors(ras, decs, t, rot_rad)
            Fpc_dpsi = (
                np.einsum("ij...,j...->i...", dpsi_rotation, ab_factors)
                * sin_angbtwArms
            )
            return Fpc_dpsi[0] * _hp, Fpc_dpsi[1] * _hc

        def phi_par_deriv():
            afac_dphi, bfac_dphi, deltat_dphi = self.detector._compute_ab_factors(
                ras, decs, t, rot_rad, dphi=True
            )

            Fpc = apply_psi_rotation(psi, afac_dphi, bfac_dphi) * sin_angbtwArms
            Ap_dphi = Fpc[0] * _hp
            Ac_dphi = Fpc[1] * _hc

            phiD_dphi = 0.0
            phiL_dphi = omega * deltat_dphi

            return (
                Ap_dphi
                + 1j * (phiD_dphi + phiL_dphi) * hp
                + Ac_dphi
                + 1j * (phiD_dphi + phiL_dphi) * hc
            )

        def theta_par_deriv():
            afac_dtheta, bfac_dtheta, deltat_dtheta = self.detector._compute_ab_factors(
                ras, decs, t, rot_rad, dtheta=True
            )

            Fpc = apply_psi_rotation(psi, afac_dtheta, bfac_dtheta) * sin_angbtwArms
            Ap_dtheta = Fpc[0] * _hp
            Ac_dtheta = Fpc[1] * _hc

            phiD_dtheta = 0.0
            phiL_dtheta = omega * deltat_dtheta

            return (
                Ap_dtheta
                + 1j * (phiD_dtheta + phiL_dtheta) * hp
                + Ac_dtheta
                + 1j * (phiD_dtheta + phiL_dtheta) * hc
            )

        def tcoal_par_deriv():
            afac_dtime, bfac_dtime, deltat_dtime = self.detector._compute_ab_factors(
                ras, decs, t, rot_rad, dtheta=True
            )

            Fpc = apply_psi_rotation(psi, afac_dtime, bfac_dtime) * sin_angbtwArms
            Ap_dtime = Fpc[0] * _hp
            Ac_dtime = Fpc[1] * _hc

            phiD_dtime = 0.0
            phiL_dtime = omega * deltat_dtime

            return (
                Ap_dtime
                + 1j * (phiD_dtime + phiL_dtime + omega) * hp
                + Ac_dtime
                + 1j * (phiD_dtime + phiL_dtime + omega) * hc
            )

        def iota_par_deriv():

            if (not self.wf_model.is_HigherModes) and (not self.wf_model.is_Precessing):
                wfhp_diota = wfhpc * (-0.5 * np.sin(2 * iota))
                wfhc_diota = -1j * wfhpc * np.sin(iota)
                return (Fpc[0] * wfhp_diota + Fpc[1] * wfhc_diota) * np.exp(phase)
            else:
                # This derivative is computed numerically if the waveform contains higher modes
                return None

        return {
            "dL": -(hp + hc) / dL,
            "theta": theta_par_deriv(),
            "phi": phi_par_deriv(),
            "iota": iota_par_deriv(),
            "psi": psi_par_deriv(),
            "tcoal": tcoal_par_deriv(),
            "Phicoal": -1j * (hp + hc),
        }
