#
#    Copyright (c) 2022 Francesco Iacovelli <francesco.iacovelli@unige.ch>, Michele Mancarella <michele.mancarella@unige.ch>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.

import os

from jax import config, vmap, jacrev, jit, device_count, local_device_count
import jax.numpy as np

# Enable 64bit on JAX, fundamental
config.update("jax_enable_x64", True)
# config.update("TF_CPP_MIN_LOG_LEVEL", 0)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# We use both the original numpy, denoted as onp, and the JAX implementation of numpy, denoted as np
import numpy as onp
from scipy.optimize import minimize
from scipy.integrate import cumulative_trapezoid
import time
import h5py
import numdifftools as ndt
import copy
from numdifftools.step_generators import MaxStepGenerator

from gwfast import gwfastUtils as utils
from gwfast import gwfastGlobals as glob
from gwfast.gwfastGlobals import TWOPI, DAY_TO_SEC, DEG_TO_RAD
from gwfast.gwfastUtils import (
    ra_dec_from_th_phi_rad,
    check_evparams,
    spin_angle_keys,
    spin_comps_keys,
    apply_psi_rotation,
    CosineIntegrand,
    SineIntegrand,
    noise_weighted_inner_product,
    optimal_snr,
)
from gwfast.lensing_utils import (
    get_lensed_parameter_sets,
    get_lensing_time_delay,
    get_mag_factors,
)
from gwfast.detector import Detector, FpFcsqInt


class GWSignal(object):
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
        IntTablePath=None,
        DutyFactor=None,
        compute2arms=True,
        jitCompileDerivs=False,
    ):
        """
        Constructor method
        """
        if (useEarthMotion) and (wf_model.objType == "BBH") and (verbose):
            print(
                "WARNING: the motion of Earth gives a negligible contribution for BBH signals, consider switching it off to make the code run faster"
            )
        if (not useEarthMotion) and (wf_model.objType == "BNS") and (verbose):
            print(
                "WARNING: the motion of Earth gives a relevant contribution for BNS signals, consider switching it on"
            )
        if (not useEarthMotion) and (wf_model.objType == "NSBH") and (verbose):
            print(
                "WARNING: the motion of Earth gives a relevant contribution for NSBH signals, consider switching it on"
            )

        self.wf_model = wf_model
        self.strain_model_keys = list(self.wf_model.ParNums.keys())

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

        self.verbose = verbose
        self.IntTablePath = IntTablePath

        self.fmin = fmin  # Hz
        self.fmax = fmax  # Hz or None

        mask = self.detector.psd_frequencies >= self.fmin
        if self.fmax is not None:
            mask *= self.detector.psd_frequencies <= self.fmax

        masked_freqs = self.detector.psd_frequencies[mask]
        self.strainInteg = cumulative_trapezoid(
            masked_freqs ** (-7.0 / 3.0) / self.detector.psd_array[mask],
            masked_freqs,
            initial=0,
        )

        self.useEarthMotion = useEarthMotion
        self.noMotion = noMotion
        if self.noMotion and self.useEarthMotion:
            print("noMotion and useEarthMotion are True. switching off useEarthMotion ")
            self.useEarthMotion = False

        self.IntegInterpArr = None
        self.compute2arms = compute2arms

        onp.random.seed(None)
        self.seedUse = onp.random.randint(2**32 - 1, size=1)
        self.jitCompileDerivs = jitCompileDerivs

        if not self.wf_model.is_LAL:
            self._init_jax()
        else:
            self._SignalDerivatives_use = self._SignalDerivatives

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
            self._SignalDerivatives_use = jit(
                self._SignalDerivatives,
                static_argnames=[
                    "use_chi1chi2",
                    "use_m1m2",
                    "computeAnalyticalDeriv",
                    "use_prec_ang",
                    "computeDerivFinDiff",
                    "stepNDT",
                    "methodNDT",
                ],
            )
        else:
            self._SignalDerivatives_use = self._SignalDerivatives

        inj_params_init = {
            "Mc": np.array([77.23905294]),
            "phase": np.array([3.28297867]),
            "chi1z": np.array([0.2018924]),
            "chi2z": np.array([-0.68859213]),
            "chis": np.array([0.2018924]),
            "chia": np.array([-0.68859213]),
            "dL": np.array([22.68426174]),
            "eta": np.array([0.20586622]),
            "iota": np.array([4.48411048]),
            "phi": np.array([0.90252645]),
            "psi": np.array([3.11843169]),
            "Lambda1": np.array([300.0]),
            "Lambda2": np.array([300.0]),
            #'snr': np.array([21.20295982]),
            #'tGPS': np.array([1.78168705e+09]),
            "tcoal": np.array([0.0]),
            "theta": np.array([3.00702251]),
            "chi1x": np.array([0.1]),
            "chi2x": np.array([0.05]),
            "chi1y": np.array([0.1]),
            "chi2y": np.array([-0.01]),
            "ecc": np.array([0.0]),
            "R_orbit": np.array([100.0]),
            "M_lz": np.array([1e6]),
            "src_pos": np.array([0.1]),
        }
        _verbose = self.verbose
        self.verbose = False
        _detector_shape = self.detector.shape
        self.detector.shape = "L"  # Get a faster Initialization with an L
        _strain_model_keys = self.strain_model_keys
        self.strain_model_keys = list(inj_params_init.keys())
        _ = self.SNRInteg(inj_params_init, res=10)
        _ = self.FisherMatr(inj_params_init, res=10)

        if self.verbose:
            print("Done.")
        # Restore the original values
        self.verbose = _verbose
        self.detector.shape = _detector_shape
        self.strain_model_keys = _strain_model_keys

    def _clear_cache(self):
        if self.jitCompileDerivs:
            print("Clearing cache...")
            self._SignalDerivatives_use = jit(
                self._SignalDerivatives, static_argnums=(15, 16, 17, 18, 19)
            )

    def _update_seed(self, seed=None):
        """
        Update the seed for the duty cycle with a random value or a user input value.

        :param int, optional seed: User input value for the seed.

        """
        onp.random.seed(None)
        if seed is None:
            self.seedUse = onp.random.randint(2**32 - 1, size=1)
        else:
            self.seedUse = seed

    def _tabulateIntegrals(self, res=200, store=True, Mcmin=0.9, Mcmax=9.0, etamin=0.1):
        """
        Compute the table of integrals to use :py:class:`GWSignal.SNRFastInsp`.

        .. deprecated:: 1.0.0

        """
        Mcgrid = onp.linspace(Mcmin, Mcmax, res)
        etagrid = onp.linspace(etamin, 0.25, res)
        tcgrid = onp.linspace(0.0, TWOPI, res)

        Igrid = onp.zeros((res, res, res, 9))

        if self.verbose:
            print("Computing table of integrals...\n")

        in_time = time.time()

        for i, Mc in enumerate(Mcgrid):
            for j, eta in enumerate(etagrid):
                tmpev = {
                    "Mc": np.array([Mc]),
                    "eta": np.array([eta]),
                }
                fcut = self.wf_model.fcut(**tmpev)
                mask = self.detector.psd_frequencies >= self.fmin
                mask *= self.detector.psd_frequencies <= fcut
                masked_freqs = self.detector.psd_frequencies[mask]
                masked_psd = self.detector.psd_array[mask]
                # for k,tc in enumerate(tcgrid):
                # TODO: Use some kind of expand axis
                fgrids = (
                    np.ones((res, len(masked_freqs))) * masked_freqs
                )
                noisegrids = (
                    np.ones((res, len(masked_psd))) * masked_psd
                )
                masked_freqs = masked_freqs[:, onp.newaxis]
                for m in range(4):
                    tmpIntegrandC = CosineIntegrand(masked_freqs, Mc, tcgrid, m + 1.0)
                    tmpIntegrandS = SineIntegrand(masked_freqs, Mc, tcgrid, m + 1.0)
                    Igrid[i, j, :, m] = onp.trapz(
                        tmpIntegrandC / noisegrids.T, fgrids.T, axis=0
                    )
                    Igrid[i, j, :, m + 4] = onp.trapz(
                        tmpIntegrandS / noisegrids.T, fgrids.T, axis=0
                    )

                tmpIntegrand = CosineIntegrand(masked_freqs, Mc, tcgrid, 0.0)
                Igrid[i, j, :, 8] = onp.trapz(
                    tmpIntegrand / noisegrids.T, fgrids.T, axis=0
                )

        if self.verbose:
            print("Done in %.2fs \n" % (time.time() - in_time))

        if store:
            print("Saving result...")
            if not os.path.isdir(os.path.join(self.psd_base_path, "Integral_Tables")):
                os.mkdir(os.path.join(self.psd_base_path, "Integral_Tables"))

            with h5py.File(
                os.path.join(
                    self.psd_base_path,
                    "Integral_Tables",
                    type(self.wf_model).__name__ + str(res) + ".h5",
                ),
                "w",
            ) as out:
                out.create_dataset("Mc", data=Mcgrid, compression="gzip", shuffle=True)
                out.create_dataset(
                    "eta", data=etagrid, compression="gzip", shuffle=True
                )
                out.create_dataset("tc", data=tcgrid, compression="gzip", shuffle=True)
                out.create_dataset(
                    "Integs", data=Igrid, compression="gzip", shuffle=True
                )
                out.attrs["npoints"] = res
                out.attrs["etamin"] = etamin
                out.attrs["Mcmin"] = Mcmin
                out.attrs["Mcmax"] = Mcmax

        return Igrid, Mcgrid, etagrid, tcgrid

    def _make_SNRig_interpolator(
        self,
    ):
        """
        Make interpolator of the table of integrals to use :py:class:`GWSignal.SNRFastInsp`.

        .. deprecated:: 1.0.0

        """
        from scipy.interpolate import RegularGridInterpolator

        if self.IntTablePath is not None:
            if os.path.exists(self.IntTablePath):
                if self.verbose:
                    print(
                        "Pre-computed optimal integrals grid is present for this waveform. Loading..."
                    )
                with h5py.File(self.IntTablePath, "r") as inp:
                    Mcs = np.array(inp["Mc"])
                    etas = np.array(inp["eta"])
                    tcs = np.array(inp["tc"])
                    Igrid = np.array(inp["Integs"])
                    if self.verbose:
                        print("Attributes of pre-computed integrals: ")
                        print([(k, inp.attrs[k]) for k in inp.attrs.keys()])

            else:
                print("Tabulating integrals...")
                Igrid, Mcs, etas, tcs = self._tabulateIntegrals()

        else:
            print("Tabulating integrals...")
            Igrid, Mcs, etas, tcs = self._tabulateIntegrals()

        self.IntegInterpArr = onp.array([])
        for i in range(9):
            # The interpolator contains in the elements from i=0 to 3 the integrals of cos((i+1) Om t)f^{-7/3}
            # in the elements from 4 to 7 the integrals of the sine (from lower to higher i), and in element 8 the
            # integral of f^-7/3 alone

            self.IntegInterpArr = onp.append(
                self.IntegInterpArr,
                RegularGridInterpolator((Mcs, etas, tcs), Igrid[:, :, :, i]),
            )

    def _phiPhase(self, theta, phi, t, iota, psi, Fp=None, Fc=None):
        # The polarization phase contribution (the change in F+ and Fx with time influences also the phase)
        if (Fp is None) or (Fc is None):
            Fp, Fc = self.detector.compute_antenna_pattern(theta, phi, t, psi)

        phiP = -np.arctan2(np.cos(iota) * Fc, 0.5 * (1.0 + np.cos(iota) ** 2) * Fp)

        # The contriution to the amplitude is negligible, so we do not compute it
        return phiP

    def shifted_time(self, parameters, frequencies):
        theta, phi, tcoal = (
            parameters["theta"],
            parameters["phi"],
            parameters["tcoal"],
        )
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

    def GWAmplitudes(self, evParams, f, rot=0.0):
        """
        Compute the amplitude of the signal(s) as seen by the detector, as a function of the parameters, at given frequencies.

        :param dict(array, array, ...) evParams: Dictionary containing the parameters of the event(s), as in :py:data:`events`.
        :param array or float f: The frequency(ies) at which to perform the calculation, in :math:`\\rm Hz`.
        :param float rot: Further rotation of the interferometer with respect to the :py:data:`self.xax` orientation, in degrees, needed for the triangular geometry.
        :return: Plus and cross amplitudes at the detector, evaluated at the given parameters and frequency(ies).
        :rtype: tuple(array, array) or tuple(float, float)

        """
        # evParams are all the parameters characterizing the event(s) under exam. It has to be a dictionary containing the entries:
        # Mc -> chirp mass (Msun), dL -> luminosity distance (Gpc), theta & phi -> sky position (rad), iota -> inclination angle of orbital angular momentum to l.o.s toward the detector,
        # psi -> polarisation angle, tcoal -> time of coalescence as GMST (fraction of days), eta -> symmetric mass ratio, phase -> GW frequency at coalescence.
        # chi1z, chi2z -> dimensionless spin components aligned to orbital angular momentum [-1;1], Lambda1,2 -> tidal parameters of the objects,
        # f is the frequency (Hz)

        theta, phi, iota, psi = (
            evParams["theta"],
            evParams["phi"],
            evParams["iota"],
            evParams["psi"],
        )

        time, _ = self.shifted_time(evParams, f)
        Fp, Fc = self.detector.compute_antenna_pattern(theta, phi, time, psi, rot=rot)

        if (self.wf_model.is_HigherModes) or (self.wf_model.is_Precessing):
            # If the waveform includes higher modes or precessing spins,
            # it is not possible to compute amplitude and phase separately, make all together
            hp, hc = self.wf_model.hphc(f, **evParams)
            Ap, Ac = abs(hp) * Fp, abs(hc) * Fc
        else:
            wfAmpl = self.wf_model.Ampl(f, **evParams)
            Ap = wfAmpl * Fp * 0.5 * (1.0 + (np.cos(iota)) ** 2)
            Ac = wfAmpl * Fc * np.cos(iota)

        return Ap, Ac

    def GWPhase(self, evParams, f):
        """
        Compute the complete phase of the signal(s), as a function of the parameters, at given frequencies.

        :param dict(array, array, ...) evParams: Dictionary containing the parameters of the event(s), as in :py:data:`events`.
        :param array or float f: The frequency(ies) at which to perform the calculation, in :math:`\\rm Hz`.

        :return: Complete signal phase, evaluated at the given parameters and frequency(ies).
        :rtype: array or float

        """
        # Phase of the GW signal
        tcoal, phase = evParams["tcoal"], evParams["phase"]
        PhiGw = self.wf_model.Phi(f, **evParams)

        return TWOPI * f * (tcoal * DAY_TO_SEC) - phase - PhiGw

    def GWstrain(
        self,
        f,
        Mc,
        eta,
        dL,
        theta,
        phi,
        iota,
        psi,
        tcoal,
        phase,
        chiS,
        chiA,
        chi1x,
        chi2x,
        chi1y,
        chi2y,
        LambdaTilde,
        deltaLambda,
        ecc,
        R_orbit=None,
        M_lz=None,
        src_pos=None,
        rot=0.0,
        is_m1m2=False,
        is_chi1chi2=False,
        is_prec_ang=False,
        return_single_comp=None,
        use_lensing=False,
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
        :param array or float phase: The phase(s) at coalescence, :math:`\Phi_{\\rm coal}`, in :math:`\\rm rad`.
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
        if is_m1m2:
            # Interpret Mc as m1 and eta as m2
            McUse, etaUse = utils.Mceta_from_m1m2(Mc, eta)
        else:
            McUse = Mc
            etaUse = eta

        ZEROS = np.zeros_like(McUse)

        if not self.wf_model.is_Precessing:
            if is_chi1chi2:
                # Interpret chiS as chi1z and chiA as chi2z
                chi1z = chiS
                chi2z = chiA
            else:
                chi1z = chiS + chiA
                chi2z = chiS - chiA
            chi1xUse, chi2xUse, chi1yUse, chi2yUse = ZEROS, ZEROS, ZEROS, ZEROS
        else:
            if not is_prec_ang:
                chi1z = chiS
                chi2z = chiA
                chi1xUse = chi1x
                chi2xUse = chi2x
                chi1yUse = chi1y
                chi2yUse = chi2y
            else:
                # convert angles and iota
                iota, chi1xUse, chi1yUse, chi1z, chi2xUse, chi2yUse, chi2z = (
                    utils.TransformPrecessing_angles2comp(
                        thetaJN=iota,
                        phiJL=chi1y,
                        tilt1=chi1x,
                        tilt2=chi2x,
                        phi12=chi2y,
                        chi1=chiS,
                        chi2=chiA,
                        Mc=McUse,
                        eta=etaUse,
                        fRef=self.fmin,
                        phiRef=0.0,
                    )
                )

        evParams = {
            "Mc": McUse,
            "eta": etaUse,
            "iota": iota,
            "phase": phase,
            "chi1x": chi1xUse,
            "chi1y": chi1yUse,
            "chi1z": chi1z,
            "chi2x": chi2xUse,
            "chi2y": chi2yUse,
            "chi2z": chi2z,
            "dL": dL,
            "theta": theta,
            "phi": phi,
            "psi": psi,
            "tcoal": tcoal,
        }

        if self.wf_model.is_tidal:
            evParams["Lambda1"], evParams["Lambda2"] = utils.Lam12_from_Lamt_delLam(
                LambdaTilde, deltaLambda, etaUse
            )

        if self.wf_model.is_eccentric:
            evParams["ecc"] = ecc

        # Modifications from lensing goes the end
        if use_lensing:
            evParams1, evParams2 = get_lensed_parameter_sets(
                evParams, R_orbit=R_orbit, M_lz=M_lz, src_pos=src_pos
            )

        # Not sure what does this do, but it was set to zero in both cases
        # (with or without useEarthMotion)
        phiD = ZEROS

        omega = TWOPI * f * DAY_TO_SEC
        if use_lensing:
            t1, deltaT_1 = self.shifted_time(evParams1, f)
            phiL1 = omega * deltaT_1
            t2, deltaT_2 = self.shifted_time(evParams2, f)
            phiL2 = omega * deltaT_2
        else:
            t, deltaT = self.shifted_time(evParams, f)
            phiL = omega * deltaT

        # Moving on to combining the strain with the antenna patterns
        need_HM = (self.wf_model.is_HigherModes) or (self.wf_model.is_Precessing)
        is_lal = self.wf_model.is_LAL

        if not (need_HM or is_lal):
            # Return with the simplest things
            if not use_lensing:
                Ap, Ac = self.GWAmplitudes(evParams, f, rot=rot)
                Psi = self.GWPhase(evParams, f)
                Psi += phiD + phiL
            else:
                Ap1, Ac1 = self.GWAmplitudes(evParams1, f, rot=rot)
                Psi1 = self.GWPhase(evParams1, f)
                Psi1 += phiD + phiL1

                Ap2, Ac2 = self.GWAmplitudes(evParams2, f, rot=rot)
                Psi2 = self.GWPhase(evParams2, f)
                Psi2 += phiD + phiL2

                # TODO: Check whether h = hp - i hc.
                hp1, hc1 = Ap1 * np.exp(Psi1 * 1j), 1j * Ac1 * np.exp(Psi1 * 1j)
                hp2, hc2 = Ap2 * np.exp(Psi2 * 1j), 1j * Ac2 * np.exp(Psi2 * 1j)

                # Time delay and magnification
                time_delay = get_lensing_time_delay(
                    evParams1, M_lz=M_lz, src_pos=src_pos
                )
                time_delay_phase_shift = np.exp(2j * np.pi * f * time_delay)
                mag_1, mag_2 = get_mag_factors(evParams1, src_pos=src_pos)

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

        phase_shift_factor = np.exp(1j * (phiD + omega * tcoal))
        # It appears that using LAL or not only matters in the antenna pattern, combining both cases.
        if not use_lensing:
            # If the waveform includes higher modes or precessing spins, it is not possible to compute amplitude and phase separately, make all together
            Fp, Fc = self.detector.compute_antenna_pattern(theta, phi, t, psi, rot=rot)
            hp, hc = self.wf_model.hphc(f, **evParams)

            hp *= Fp * phase_shift_factor * np.exp(1j * (phiL - phase))
            hc *= Fc * phase_shift_factor * np.exp(1j * (phiL - phase))

            if is_lal:
                hp *= 0.5 * (1.0 + np.cos(iota) ** 2)
                hc *= np.cos(iota)

        else:
            iota1 = evParams1["iota"]
            psi1 = evParams1["psi"]
            phase1 = evParams1["phase"]
            Fp1, Fc1 = self.detector.compute_antenna_pattern(
                theta, phi, t1, psi1, rot=rot
            )
            hp1, hc1 = self.wf_model.hphc(f, **evParams1)
            hp1 = hp1 * Fp1 * phase_shift_factor * np.exp(1j * (phiL1 - phase1))
            hc1 = hc1 * Fc1 * phase_shift_factor * np.exp(1j * (phiL1 - phase1))

            if is_lal:
                hp1 *= 0.5 * (1.0 + np.cos(iota1) ** 2)
                hc1 *= np.cos(iota1)

            iota2 = evParams2["iota"]
            psi2 = evParams2["psi"]
            phase2 = evParams2["phase"]
            Fp2, Fc2 = self.detector.compute_antenna_pattern(
                theta, phi, t2, psi2, rot=rot
            )
            hp2, hc2 = self.wf_model.hphc(f, **evParams2)
            hp2 = hp2 * Fp2 * phase_shift_factor * np.exp(1j * (phiL2 - phase2))
            hc2 = hc2 * Fc2 * phase_shift_factor * np.exp(1j * (phiL2 - phase2))

            if is_lal:
                hp2 *= 0.5 * (1.0 + np.cos(iota2) ** 2)
                hc2 *= np.cos(iota2)

            # Time delay and magnification
            time_delay = get_lensing_time_delay(evParams, M_lz, src_pos)
            time_delay_phase_shift = np.exp(2j * np.pi * f * time_delay)
            mag_1, mag_2 = get_mag_factors(evParams, src_pos)

            hp = (
                np.sqrt(np.abs(mag_1)) * hp1
                + np.sqrt(np.abs(mag_2)) * time_delay_phase_shift * hp2
            )
            hc = (
                np.sqrt(np.abs(mag_1)) * hc1
                + np.sqrt(np.abs(mag_2)) * time_delay_phase_shift * hc2
            )

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

    def SNRInteg(self, evParams, res=1000, return_all=False, use_lensing=False):
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

        utils.check_evparams(evParams)
        all_params_keys = list(evParams.keys())

        # if not np.isscalar(evParams['Mc']):
        #    SNR = np.zeros(len(np.asarray(evParams['Mc'])))
        # else:
        #    SNR = 0.

        allSNRsq = []

        if self.wf_model.is_Precessing:
            # Check if cartesian spins are provided
            if not all([(key in all_params_keys) for key in spin_comps_keys]):
                # Check if spin angles are provided instead
                if all([(key in all_params_keys) for key in spin_angle_keys]):
                    if self.verbose:
                        print(
                            "Adding cartesian components of the spins from angular variables"
                        )
                    (
                        evParams["iota"],
                        evParams["chi1x"],
                        evParams["chi1y"],
                        evParams["chi1z"],
                        evParams["chi2x"],
                        evParams["chi2y"],
                        evParams["chi2z"],
                    ) = utils.TransformPrecessing_angles2comp(
                        thetaJN=evParams["thetaJN"],
                        phiJL=evParams["phiJL"],
                        tilt1=evParams["tilt1"],
                        tilt2=evParams["tilt2"],
                        phi12=evParams["phi12"],
                        chi1=evParams["chi1"],
                        chi2=evParams["chi2"],
                        Mc=evParams["Mc"],
                        eta=evParams["eta"],
                        fRef=self.fmin,
                        phiRef=0.0,
                    )
                else:
                    raise ValueError(
                        "Either the cartesian components of the precessing spins (iota, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z) or their modulus and orientations (thetaJN, chi1, chi2, tilt1, tilt2, phiJL, phi12) have to be provided."
                    )

        else:
            if ("chi1z" in all_params_keys) and ("chi2z" in all_params_keys):
                # If both of them are present, we do nothing.
                pass
            elif ("chiS" in all_params_keys) and ("chiA" in all_params_keys):
                # We compute chi1z and chi2z from chiS and chiA
                if self.verbose:
                    print("Adding chi1z, chi2z from chiS, chiA")
                evParams["chi1z"] = evParams["chiS"] + evParams["chiA"]
                evParams["chi2z"] = evParams["chiS"] - evParams["chiA"]
            else:
                raise ValueError(
                    "One pair among (chi1z, chi2z) and (chiS, chiA) have to be provided."
                )

        if self.wf_model.is_tidal:
            if ("Lambda1" in all_params_keys) and ("Lambda2" in all_params_keys):
                pass
            elif ("LambdaTilde" in all_params_keys) and (
                "deltaLambda" in all_params_keys
            ):
                evParams["Lambda1"], evParams["Lambda2"] = utils.Lam12_from_Lamt_delLam(
                    evParams["LambdaTilde"], evParams["deltaLambda"], evParams["eta"]
                )
            else:
                raise ValueError(
                    "One pair among (Lambda1, Lambda2) and (LambdaTilde and deltaLambda) have to be provided."
                )

        if use_lensing:
            evParams1, evParams2 = get_lensed_parameter_sets(evParams)

        fcut = self.wf_model.fcut(**evParams)

        if self.fmax is not None:
            fcut = np.where(fcut > self.fmax, self.fmax, fcut)

        fminarr = np.full(fcut.shape, self.fmin)
        fgrids = np.geomspace(fminarr, fcut, num=int(res))
        strainGrids = self.detector.psd_interp(fgrids)

        if self.detector.shape == "L":
            if not use_lensing:
                Aps, Acs = self.GWAmplitudes(evParams, fgrids)
                Atot = Aps * Aps + Acs * Acs
            else:
                htot1 = self.GWstrain(
                    fgrids,
                    evParams1["Mc"],
                    evParams1["eta"],
                    evParams1["dL"],
                    evParams1["theta"],
                    evParams1["phi"],
                    evParams1["iota"],
                    evParams1["psi"],
                    evParams1["tcoal"],
                    evParams1["phase"],
                    evParams1["chi1z"],
                    evParams1["chi2z"],
                    evParams1["chi1x"],
                    evParams1["chi2x"],
                    evParams1["chi1y"],
                    evParams1["chi2y"],
                    evParams1["LambdaTilde"],
                    evParams1["deltaLambda"],
                    evParams1["ecc"],
                    R_orbit=evParams1["R_orbit"],
                    M_lz=evParams1["M_lz"],
                    src_pos=evParams1["src_pos"],
                    rot=0.0,
                    is_m1m2=False,
                    is_chi1chi2=True,
                    is_prec_ang=False,
                    return_single_comp=None,
                    use_lensing=False,
                )
                htot2 = self.GWstrain(
                    fgrids,
                    evParams2["Mc"],
                    evParams2["eta"],
                    evParams2["dL"],
                    evParams2["theta"],
                    evParams2["phi"],
                    evParams2["iota"],
                    evParams2["psi"],
                    evParams2["tcoal"],
                    evParams2["phase"],
                    evParams2["chi1z"],
                    evParams2["chi2z"],
                    evParams2["chi1x"],
                    evParams2["chi2x"],
                    evParams2["chi1y"],
                    evParams2["chi2y"],
                    evParams2["LambdaTilde"],
                    evParams2["deltaLambda"],
                    evParams2["ecc"],
                    R_orbit=evParams2["R_orbit"],
                    M_lz=evParams2["M_lz"],
                    src_pos=evParams2["src_pos"],
                    rot=0.0,
                    is_m1m2=False,
                    is_chi1chi2=True,
                    is_prec_ang=False,
                    return_single_comp=None,
                    use_lensing=False,
                )
                Atot = abs(htot1 + htot2) ** 2
                # print(htot1)

            SNRsq = np.trapezoid(Atot / strainGrids, fgrids, axis=0)
            if self.detector.duty_cycle is not None:
                excl = onp.random.random(len(evParams["Mc"])) > self.detector.duty_cycle
                SNRsq = SNRsq * excl
            allSNRsq.append(SNRsq)
        elif self.detector.shape == "T":
            if not self.compute2arms:
                for i in range(3):
                    if not use_lensing:
                        Aps, Acs = self.GWAmplitudes(evParams, fgrids, rot=i * 60.0)
                        Atot = Aps * Aps + Acs * Acs
                    else:
                        htot1 = self.GWstrain(
                            fgrids,
                            evParams1["Mc"],
                            evParams1["eta"],
                            evParams1["dL"],
                            evParams1["theta"],
                            evParams1["phi"],
                            evParams1["iota"],
                            evParams1["psi"],
                            evParams1["tcoal"],
                            evParams1["phase"],
                            evParams1["chi1z"],
                            evParams1["chi2z"],
                            evParams1["chi1x"],
                            evParams1["chi2x"],
                            evParams1["chi1y"],
                            evParams1["chi2y"],
                            evParams1["LambdaTilde"],
                            evParams1["deltaLambda"],
                            evParams1["ecc"],
                            R_orbit=evParams1["R_orbit"],
                            M_lz=evParams1["M_lz"],
                            src_pos=evParams1["src_pos"],
                            rot=i * 60.0,
                            is_m1m2=False,
                            is_chi1chi2=True,
                            is_prec_ang=False,
                            return_single_comp=None,
                            use_lensing=False,
                        )
                        htot2 = self.GWstrain(
                            fgrids,
                            evParams2["Mc"],
                            evParams2["eta"],
                            evParams2["dL"],
                            evParams2["theta"],
                            evParams2["phi"],
                            evParams2["iota"],
                            evParams2["psi"],
                            evParams2["tcoal"],
                            evParams2["phase"],
                            evParams2["chi1z"],
                            evParams2["chi2z"],
                            evParams2["chi1x"],
                            evParams2["chi2x"],
                            evParams2["chi1y"],
                            evParams2["chi2y"],
                            evParams2["LambdaTilde"],
                            evParams2["deltaLambda"],
                            evParams2["ecc"],
                            R_orbit=evParams2["R_orbit"],
                            M_lz=evParams2["M_lz"],
                            src_pos=evParams2["src_pos"],
                            rot=i * 60.0,
                            is_m1m2=False,
                            is_chi1chi2=True,
                            is_prec_ang=False,
                            return_single_comp=None,
                            use_lensing=False,
                        )
                        Atot = abs(htot1 + htot2) ** 2
                    tmpSNRsq = np.trapezoid(Atot / strainGrids, fgrids, axis=0)
                    if self.detector.duty_cycle is not None:
                        excl = (
                            onp.random.random(len(evParams["Mc"]))
                            > self.detector.duty_cycle
                        )
                        tmpSNRsq = tmpSNRsq * excl
                    allSNRsq.append(tmpSNRsq)
                    # SNR = SNR + tmpSNRsq
                # SNR = np.sqrt(SNR)
            else:
                # The signal in 3 arms sums to zero for geometrical reasons, so we can use this to skip some calculations
                if not use_lensing:
                    Aps1, Acs1 = self.GWAmplitudes(evParams, fgrids, rot=0.0)
                    Atot1 = Aps1 * Aps1 + Acs1 * Acs1
                    Aps2, Acs2 = self.GWAmplitudes(evParams, fgrids, rot=60.0)
                    Atot2 = Aps2 * Aps2 + Acs2 * Acs2
                    Aps3, Acs3 = -(Aps1 + Aps2), -(Acs1 + Acs2)
                    Atot3 = Aps3 * Aps3 + Acs3 * Acs3
                else:
                    htot1_1 = self.GWstrain(
                        fgrids,
                        evParams1["Mc"],
                        evParams1["eta"],
                        evParams1["dL"],
                        evParams1["theta"],
                        evParams1["phi"],
                        evParams1["iota"],
                        evParams1["psi"],
                        evParams1["tcoal"],
                        evParams1["phase"],
                        evParams1["chi1z"],
                        evParams1["chi2z"],
                        evParams1["chi1x"],
                        evParams1["chi2x"],
                        evParams1["chi1y"],
                        evParams1["chi2y"],
                        evParams1["LambdaTilde"],
                        evParams1["deltaLambda"],
                        evParams1["ecc"],
                        R_orbit=evParams1["R_orbit"],
                        M_lz=evParams1["M_lz"],
                        src_pos=evParams1["src_pos"],
                        rot=0.0,
                        is_m1m2=False,
                        is_chi1chi2=True,
                        is_prec_ang=False,
                        return_single_comp=None,
                        use_lensing=False,
                    )
                    htot2_1 = self.GWstrain(
                        fgrids,
                        evParams2["Mc"],
                        evParams2["eta"],
                        evParams2["dL"],
                        evParams2["theta"],
                        evParams2["phi"],
                        evParams2["iota"],
                        evParams2["psi"],
                        evParams2["tcoal"],
                        evParams2["phase"],
                        evParams2["chi1z"],
                        evParams2["chi2z"],
                        evParams2["chi1x"],
                        evParams2["chi2x"],
                        evParams2["chi1y"],
                        evParams2["chi2y"],
                        evParams2["LambdaTilde"],
                        evParams2["deltaLambda"],
                        evParams2["ecc"],
                        R_orbit=evParams2["R_orbit"],
                        M_lz=evParams2["M_lz"],
                        src_pos=evParams2["src_pos"],
                        rot=0.0,
                        is_m1m2=False,
                        is_chi1chi2=True,
                        is_prec_ang=False,
                        return_single_comp=None,
                        use_lensing=False,
                    )
                    Atot1 = abs(htot1_1 + htot2_1) ** 2
                    htot1_2 = self.GWstrain(
                        fgrids,
                        evParams1["Mc"],
                        evParams1["eta"],
                        evParams1["dL"],
                        evParams1["theta"],
                        evParams1["phi"],
                        evParams1["iota"],
                        evParams1["psi"],
                        evParams1["tcoal"],
                        evParams1["phase"],
                        evParams1["chi1z"],
                        evParams1["chi2z"],
                        evParams1["chi1x"],
                        evParams1["chi2x"],
                        evParams1["chi1y"],
                        evParams1["chi2y"],
                        evParams1["LambdaTilde"],
                        evParams1["deltaLambda"],
                        evParams1["ecc"],
                        R_orbit=evParams1["R_orbit"],
                        M_lz=evParams1["M_lz"],
                        src_pos=evParams1["src_pos"],
                        rot=60.0,
                        is_m1m2=False,
                        is_chi1chi2=True,
                        is_prec_ang=False,
                        return_single_comp=None,
                        use_lensing=False,
                    )
                    htot2_2 = self.GWstrain(
                        fgrids,
                        evParams2["Mc"],
                        evParams2["eta"],
                        evParams2["dL"],
                        evParams2["theta"],
                        evParams2["phi"],
                        evParams2["iota"],
                        evParams2["psi"],
                        evParams2["tcoal"],
                        evParams2["phase"],
                        evParams2["chi1z"],
                        evParams2["chi2z"],
                        evParams2["chi1x"],
                        evParams2["chi2x"],
                        evParams2["chi1y"],
                        evParams2["chi2y"],
                        evParams2["LambdaTilde"],
                        evParams2["deltaLambda"],
                        evParams2["ecc"],
                        R_orbit=evParams2["R_orbit"],
                        M_lz=evParams2["M_lz"],
                        src_pos=evParams2["src_pos"],
                        rot=60.0,
                        is_m1m2=False,
                        is_chi1chi2=True,
                        is_prec_ang=False,
                        return_single_comp=None,
                        use_lensing=False,
                    )
                    Atot2 = abs(htot1_2 + htot2_2) ** 2
                    Atot3 = abs(htot1_1 + htot1_2 + htot2_1 + htot2_2) ** 2

                tmpSNRsq1 = np.trapezoid(Atot1 / strainGrids, fgrids, axis=0)
                tmpSNRsq2 = np.trapezoid(Atot2 / strainGrids, fgrids, axis=0)
                tmpSNRsq3 = np.trapezoid(Atot3 / strainGrids, fgrids, axis=0)
                if self.detector.duty_cycle is not None:
                    excl = (
                        onp.random.random(len(evParams["Mc"]))
                        > self.detector.duty_cycle
                    )
                    tmpSNRsq1 = tmpSNRsq1 * excl
                    excl = (
                        onp.random.random(len(evParams["Mc"]))
                        > self.detector.duty_cycle
                    )
                    tmpSNRsq2 = tmpSNRsq2 * excl
                    excl = (
                        onp.random.random(len(evParams["Mc"]))
                        > self.detector.duty_cycle
                    )
                    tmpSNRsq3 = tmpSNRsq3 * excl
                allSNRsq.append(tmpSNRsq1)
                allSNRsq.append(tmpSNRsq2)
                allSNRsq.append(tmpSNRsq3)
                # SNR = np.sqrt(tmpSNRsq1 + tmpSNRsq2 + tmpSNRsq3)
        allSNRsq = np.array(allSNRsq)

        if self.detector.shape == "T":
            return (
                2 * np.sqrt(allSNRsq)
                if return_all
                else 2 * np.sqrt(allSNRsq.sum(axis=0))
            )

        return np.squeeze(2 * np.sqrt(allSNRsq), axis=0)

    # The factor of two arises by cutting the integral from 0 to infinity
    def FisherMatr(
        self,
        evParams,
        res=1000,
        df=None,
        spacing="geom",
        use_m1m2=False,
        use_chi1chi2=True,
        use_prec_ang=True,
        computeDerivFinDiff=False,
        computeAnalyticalDeriv=False,
        return_all=False,
        use_lensing=False,
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
        :param bool, optional computeAnalyticalDeriv: Boolean specifying if the derivatives with respect to ``dL``, ``theta``, ``phi``, ``psi``, ``tcoal``, ``phase`` and ``iota`` (the latter only for the fundamental mode in the non-precessing case) have to be computed analytically. This considerably speeds up the calculation and provides better accuracy.
        :param bool, optional return_all: Boolean specifying if, in the case of a triangular detector, the FIMs of the individual instruments have to be returned separately. In this case the return type is *list(array, array, array)*.
        :param kwargs: Optional arguments to be passed to :py:class:`gwfast.signal.GWSignal._SignalDerivatives`, such as ``methodNDT``.
        :return: FIM(s) as a function of the parameters of the event(s). The shape is :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters}`, :math:`N_{\\rm events})`.
        :rtype: 3-D array

        """
        # If use_m1m2=True the Fisher is computed w.r.t. m1 and m2, not Mc and eta
        # If use_chi1chi2=True the Fisher is computed w.r.t. chi1z and chi2z, not chiS and chiA
        if self.detector.duty_cycle is not None:
            onp.random.seed(self.seedUse)

        utils.check_evparams(evParams)
        all_params_keys = list(evParams.keys())

        McOr, dL, theta, phi = (
            evParams["Mc"].astype("complex128"),
            evParams["dL"].astype("complex128"),
            evParams["theta"].astype("complex128"),
            evParams["phi"].astype("complex128"),
        )
        iota, psi, tcoal, etaOr, phase = (
            evParams["iota"].astype("complex128"),
            evParams["psi"].astype("complex128"),
            evParams["tcoal"].astype("complex128"),
            evParams["eta"].astype("complex128"),
            evParams["phase"].astype("complex128"),
        )

        ZEROS = np.zeros_like(McOr)

        if use_m1m2:
            # In this case Mc represents m1 and eta represents m2
            Mc, eta = utils.m1m2_from_Mceta(McOr, etaOr)
        else:
            Mc, eta = McOr, etaOr

        if not self.wf_model.is_Precessing:
            # For the aligned-spin models:

            # We first guarantee the existence of chi1z and chi2z.
            if ("chi1z" in all_params_keys) and ("chi2z" in all_params_keys):
                # If both of them are present, we do nothing.
                pass
            elif ("chiS" in all_params_keys) and ("chiA" in all_params_keys):
                # We compute chi1z and chi2z from chiS and chiA
                if self.verbose:
                    print("Adding chi1z, chi2z from chiS, chiA")
                evParams["chi1z"] = evParams["chiS"] + evParams["chiA"]
                evParams["chi2z"] = evParams["chiS"] - evParams["chiA"]
            else:
                raise ValueError(
                    "One pair among (chi1z, chi2z) and (chiS, chiA) have to be provided."
                )

            chi1z = evParams["chi1z"].astype("complex128")
            chi2z = evParams["chi2z"].astype("complex128")

            # Get sym and asym spin components
            if use_chi1chi2:
                # In this case chiS represents chi1z and chiA represents chi2z
                chiS, chiA = chi1z, chi2z
            else:
                chiS = 0.5 * (chi1z + chi2z)
                chiA = 0.5 * (chi1z - chi2z)

            # In any case, we set all in-plane components to zeroes.
            chi1x, chi2x, chi1y, chi2y = ZEROS, ZEROS, ZEROS, ZEROS

        else:
            # Check if cartesian spins are provided
            if not all([(key in all_params_keys) for key in spin_comps_keys]):
                # Check if spin angles are provided instead
                if all([(key in all_params_keys) for key in spin_angle_keys]):

                    if self.verbose:
                        print(
                            "Adding cartesian components of the spins from angular variables"
                        )
                    (
                        evParams["iota"],
                        evParams["chi1x"],
                        evParams["chi1y"],
                        evParams["chi1z"],
                        evParams["chi2x"],
                        evParams["chi2y"],
                        evParams["chi2z"],
                    ) = utils.TransformPrecessing_angles2comp(
                        thetaJN=evParams["thetaJN"],
                        phiJL=evParams["phiJL"],
                        tilt1=evParams["tilt1"],
                        tilt2=evParams["tilt2"],
                        phi12=evParams["phi12"],
                        chi1=evParams["chi1"],
                        chi2=evParams["chi2"],
                        Mc=evParams["Mc"],
                        eta=evParams["eta"],
                        fRef=self.fmin,
                        phiRef=0.0,
                    )
                else:
                    raise ValueError(
                        "Either the cartesian components of the precessing spins (iota, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z) or their modulus and orientations (thetaJN, chi1, chi2, tilt1, tilt2, phiJL, phi12) have to be provided."
                    )

            chi1x = evParams["chi1x"].astype("complex128")
            chi1y = evParams["chi1y"].astype("complex128")
            _chi1z = evParams["chi1z"].astype("complex128")
            chi2x = evParams["chi2x"].astype("complex128")
            chi2y = evParams["chi2y"].astype("complex128")
            _chi2z = evParams["chi2z"].astype("complex128")

            if not use_prec_ang:
                chiS = _chi1z
                chiA = _chi2z
            else:
                # In this case iota=thetaJN, chi1y=phiJL, chi1x=tilt1, chi2x=tilt2, chi2y=phi12, chiS=chi1, chiA=chi2
                iota, chi1y, chi1x, chi2x, chi2y, chiS, chiA = (
                    utils.TransformPrecessing_comp2angles(
                        evParams["iota"].astype("complex128"),
                        chi1x,
                        chi1y,
                        _chi1z,
                        chi2x,
                        chi2y,
                        _chi2z,
                        McOr,
                        etaOr,
                        fRef=self.fmin,
                        phiRef=0.0,
                    )
                )

        if self.wf_model.is_tidal:
            if ("Lambda1" in all_params_keys) and ("Lambda2" in all_params_keys):
                Lambda1 = evParams["Lambda1"].astype("complex128")
                Lambda2 = evParams["Lambda2"].astype("complex128")
                LambdaTilde, deltaLambda = utils.Lamt_delLam_from_Lam12(
                    Lambda1, Lambda2, etaOr
                )
            elif ("LambdaTilde" in all_params_keys) and (
                "deltaLambda" in all_params_keys
            ):
                LambdaTilde = evParams["LambdaTilde"].astype("complex128")
                deltaLambda = evParams["deltaLambda"].astype("complex128")
                Lambda1, Lambda2 = utils.Lam12_from_Lamt_delLam(
                    LambdaTilde, deltaLambda, etaOr
                )
            else:
                raise ValueError(
                    "One pair among (Lambda1, Lambda2) and (LambdaTilde and deltaLambda) have to be provided."
                )
        else:
            Lambda1, Lambda2, LambdaTilde, deltaLambda = ZEROS, ZEROS, ZEROS, ZEROS

        if self.wf_model.is_eccentric:
            try:
                ecc = evParams["ecc"].astype("complex128")
            except KeyError:
                raise ValueError(
                    "Eccentricity has to be provided for an eccentric model."
                )
        else:
            ecc = ZEROS

        if use_lensing:
            try:
                R_orbit = evParams["R_orbit"].astype("complex128")
                M_lz = evParams["M_lz"].astype("complex128")
                src_pos = evParams["src_pos"].astype("complex128")
            except KeyError:
                raise IOError("Lensing parameters are needed for `use_lensing=True`!")
        else:
            R_orbit, M_lz, src_pos = ZEROS, ZEROS, ZEROS

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

        # Out of the provided PSD range, we use a constant value of 1, which results in completely negligible conntributions
        strainGrids = self.detector.psd_interp(fgrids)

        nParams = self.wf_model.nParams
        if use_lensing:
            nParams = self.wf_model.nParams + 3
        tcelem = self.wf_model.ParNums["tcoal"]

        if (self.wf_model.is_LAL) and (not computeDerivFinDiff):
            computeDerivFinDiff = True
            if self.verbose:
                print(
                    "Using LAL or TEOBResumS waveforms it is not possible to compute the derivatives using JAX automatic differentiation routines, being the functions written in C. Proceeding using numdifftools for numerical differentiation (finite differences)"
                )

        allFishers = []

        if self.detector.shape == "L":
            # Compute derivatives
            FisherDerivs = self._SignalDerivatives_use(
                fgrids,
                Mc,
                eta,
                dL,
                theta,
                phi,
                iota,
                psi,
                tcoal,
                phase,
                chiS,
                chiA,
                chi1x,
                chi2x,
                chi1y,
                chi2y,
                LambdaTilde,
                deltaLambda,
                ecc,
                R_orbit,
                M_lz,
                src_pos,
                rot=0.0,
                use_m1m2=use_m1m2,
                use_chi1chi2=use_chi1chi2,
                use_prec_ang=use_prec_ang,
                computeAnalyticalDeriv=computeAnalyticalDeriv,
                computeDerivFinDiff=computeDerivFinDiff,
                use_lensing=use_lensing,
                **kwargs,
            )
            # Change the units of the tcoal derivative from days to seconds (this improves conditioning)
            FisherDerivs = onp.array(FisherDerivs)
            FisherDerivs[tcelem, :, :] /= DAY_TO_SEC

            FisherIntegrands = onp.conjugate(
                FisherDerivs[:, :, onp.newaxis, :]
            ) * FisherDerivs.transpose(1, 0, 2)

            Fisher = onp.zeros((nParams, nParams, len(Mc)))
            # This for is unavoidable
            for alpha in range(nParams):
                for beta in range(alpha, nParams):
                    tmpElem = FisherIntegrands[alpha, :, beta, :].T
                    Fisher[alpha, beta, :] = (
                        onp.trapz(tmpElem.real / strainGrids.real, fgrids.real, axis=0)
                        * 4.0
                    )

                    Fisher[beta, alpha, :] = Fisher[alpha, beta, :]
            if self.detector.duty_cycle is not None:
                excl = onp.random.random(len(evParams["Mc"])) > self.detector.duty_cycle
                Fisher = Fisher * excl
            allFishers.append(Fisher)
        else:
            # Fisher = onp.zeros((nParams,nParams,len(Mc)))
            if not self.compute2arms:
                for i in range(3):
                    # Change rot and compute derivatives
                    FisherDerivs = self._SignalDerivatives_use(
                        fgrids,
                        Mc,
                        eta,
                        dL,
                        theta,
                        phi,
                        iota,
                        psi,
                        tcoal,
                        phase,
                        chiS,
                        chiA,
                        chi1x,
                        chi2x,
                        chi1y,
                        chi2y,
                        LambdaTilde,
                        deltaLambda,
                        ecc,
                        R_orbit,
                        M_lz,
                        src_pos,
                        rot=i * 60.0,
                        use_m1m2=use_m1m2,
                        use_chi1chi2=use_chi1chi2,
                        use_prec_ang=use_prec_ang,
                        computeAnalyticalDeriv=computeAnalyticalDeriv,
                        computeDerivFinDiff=computeDerivFinDiff,
                        use_lensing=use_lensing,
                        **kwargs,
                    )
                    # Change the units of the tcoal derivative from days to seconds (this improves conditioning)
                    FisherDerivs = onp.array(FisherDerivs)
                    FisherDerivs[tcelem, :, :] /= DAY_TO_SEC
                    FisherIntegrands = onp.conjugate(
                        FisherDerivs[:, :, onp.newaxis, :]
                    ) * FisherDerivs.transpose(1, 0, 2)

                    tmpFisher = onp.zeros((nParams, nParams, len(Mc)))
                    # This for is unavoidable
                    if self.verbose:
                        print("Filling matrix for arm %s..." % (i + 1))

                    for alpha in range(nParams):
                        for beta in range(alpha, nParams):
                            tmpElem = FisherIntegrands[alpha, :, beta, :].T
                            tmpFisher[alpha, beta, :] = (
                                onp.trapz(
                                    tmpElem.real / strainGrids.real, fgrids.real, axis=0
                                )
                                * 4.0
                            )

                            tmpFisher[beta, alpha, :] = tmpFisher[alpha, beta, :]
                    if self.detector.duty_cycle is not None:
                        excl = (
                            onp.random.random(len(evParams["Mc"]))
                            > self.detector.duty_cycle
                        )
                        tmpFisher = tmpFisher * excl
                    allFishers.append(tmpFisher)
                    # Fisher += tmpFisher
            else:
                # The signal in 3 arms sums to zero for geometrical reasons, so we can use this to skip some calculations

                # Compute derivatives
                FisherDerivs1 = self._SignalDerivatives_use(
                    fgrids,
                    Mc,
                    eta,
                    dL,
                    theta,
                    phi,
                    iota,
                    psi,
                    tcoal,
                    phase,
                    chiS,
                    chiA,
                    chi1x,
                    chi2x,
                    chi1y,
                    chi2y,
                    LambdaTilde,
                    deltaLambda,
                    ecc,
                    R_orbit,
                    M_lz,
                    src_pos,
                    rot=0.0,
                    use_m1m2=use_m1m2,
                    use_chi1chi2=use_chi1chi2,
                    use_prec_ang=use_prec_ang,
                    computeAnalyticalDeriv=computeAnalyticalDeriv,
                    computeDerivFinDiff=computeDerivFinDiff,
                    use_lensing=use_lensing,
                    **kwargs,
                )
                # Change the units of the tcoal derivative from days to seconds (this improves conditioning)
                FisherDerivs1 = onp.array(FisherDerivs1)
                FisherDerivs1[tcelem, :, :] /= DAY_TO_SEC

                FisherIntegrands = onp.conjugate(
                    FisherDerivs1[:, :, onp.newaxis, :]
                ) * FisherDerivs1.transpose(1, 0, 2)

                tmpFisher = onp.zeros((nParams, nParams, len(Mc)))
                if self.verbose:
                    print("Filling matrix for arm 1...")
                # This for is unavoidable
                for alpha in range(nParams):
                    for beta in range(alpha, nParams):
                        tmpElem = FisherIntegrands[alpha, :, beta, :].T
                        tmpFisher[alpha, beta, :] = (
                            onp.trapz(
                                tmpElem.real / strainGrids.real, fgrids.real, axis=0
                            )
                            * 4.0
                        )

                        tmpFisher[beta, alpha, :] = tmpFisher[alpha, beta, :]
                if self.detector.duty_cycle is not None:
                    excl = (
                        onp.random.random(len(evParams["Mc"]))
                        > self.detector.duty_cycle
                    )
                    tmpFisher = tmpFisher * excl
                # Fisher += tmpFisher
                allFishers.append(tmpFisher)

                FisherDerivs2 = self._SignalDerivatives_use(
                    fgrids,
                    Mc,
                    eta,
                    dL,
                    theta,
                    phi,
                    iota,
                    psi,
                    tcoal,
                    phase,
                    chiS,
                    chiA,
                    chi1x,
                    chi2x,
                    chi1y,
                    chi2y,
                    LambdaTilde,
                    deltaLambda,
                    ecc,
                    R_orbit,
                    M_lz,
                    src_pos,
                    rot=60.0,
                    use_m1m2=use_m1m2,
                    use_chi1chi2=use_chi1chi2,
                    use_prec_ang=use_prec_ang,
                    computeAnalyticalDeriv=computeAnalyticalDeriv,
                    computeDerivFinDiff=computeDerivFinDiff,
                    use_lensing=use_lensing,
                    **kwargs,
                )
                FisherDerivs2 = onp.array(FisherDerivs2)
                FisherDerivs2[tcelem, :, :] /= DAY_TO_SEC
                FisherIntegrands = onp.conjugate(
                    FisherDerivs2[:, :, onp.newaxis, :]
                ) * FisherDerivs2.transpose(1, 0, 2)

                tmpFisher = onp.zeros((nParams, nParams, len(Mc)))
                # This for is unavoidable
                if self.verbose:
                    print("Filling matrix for arm 2...")
                for alpha in range(nParams):
                    for beta in range(alpha, nParams):
                        tmpElem = FisherIntegrands[alpha, :, beta, :].T
                        tmpFisher[alpha, beta, :] = (
                            onp.trapz(
                                tmpElem.real / strainGrids.real, fgrids.real, axis=0
                            )
                            * 4.0
                        )

                        tmpFisher[beta, alpha, :] = tmpFisher[alpha, beta, :]
                if self.detector.duty_cycle is not None:
                    excl = (
                        onp.random.random(len(evParams["Mc"]))
                        > self.detector.duty_cycle
                    )
                    tmpFisher = tmpFisher * excl
                # Fisher += tmpFisher
                allFishers.append(tmpFisher)

                FisherDerivs3 = -(FisherDerivs1 + FisherDerivs2)

                FisherIntegrands = onp.conjugate(
                    FisherDerivs3[:, :, onp.newaxis, :]
                ) * FisherDerivs3.transpose(1, 0, 2)

                tmpFisher = onp.zeros((nParams, nParams, len(Mc)))
                # This for is unavoidable
                if self.verbose:
                    print("Filling matrix for arm 3...")
                for alpha in range(nParams):
                    for beta in range(alpha, nParams):
                        tmpElem = FisherIntegrands[alpha, :, beta, :].T
                        tmpFisher[alpha, beta, :] = (
                            onp.trapz(
                                tmpElem.real / strainGrids.real, fgrids.real, axis=0
                            )
                            * 4.0
                        )

                        tmpFisher[beta, alpha, :] = tmpFisher[alpha, beta, :]
                if self.detector.duty_cycle is not None:
                    excl = (
                        onp.random.random(len(evParams["Mc"]))
                        > self.detector.duty_cycle
                    )
                    tmpFisher = tmpFisher * excl
                # Fisher += tmpFisher
                allFishers.append(tmpFisher)

        if return_all:
            return allFishers
        elif self.detector.shape == "T":
            return onp.array(allFishers).sum(axis=0)
        else:
            return allFishers[0]

    def _SignalDerivatives(
        self,
        fgrids,
        Mc,
        eta,
        dL,
        theta,
        phi,
        iota,
        psi,
        tcoal,
        phase,
        chiS,
        chiA,
        chi1x,
        chi2x,
        chi1y,
        chi2y,
        LambdaTilde,
        deltaLambda,
        ecc,
        R_orbit=None,
        M_lz=None,
        src_pos=None,
        rot=0.0,
        use_m1m2=False,
        use_chi1chi2=True,
        use_prec_ang=True,
        computeDerivFinDiff=False,
        computeAnalyticalDeriv=True,
        stepNDT=MaxStepGenerator(base_step=1e-5),
        methodNDT="central",
        use_lensing=False,
        **kwargs,
    ):
        """
        Compute the derivatives of the GW strain with respect to the parameters of the event(s) at given frequencies (in :math:`\\rm Hz`).

        :param array or float fgrids: The frequency(ies) at which to perform the calculation, in :math:`\\rm Hz`.
        :param array or float Mc: The chirp mass(es), :math:`{\cal M}_c`, in units of :math:`\\rm M_{\odot}`. If ``use_m1m2=True`` this is interpreted as the primary mass, :math:`m_1`, in units of :math:`\\rm M_{\odot}`.
        :param array or float eta:  The symmetric mass ratio(s), :math:`\eta`. If ``use_m1m2=True`` this is interpreted as the secondary mass, :math:`m_2`, in units of :math:`\\rm M_{\odot}`.
        :param array or float dL: The luminosity distance(s), :math:`d_L`, in :math:`\\rm Gpc`.
        :param array or float theta: The :math:`\\theta` sky position angle(s), in :math:`\\rm rad`.
        :param array or float phi: The :math:`\phi` sky position angle(s), in :math:`\\rm rad`.
        :param array or float iota: The inclination angle(s), with respect to orbital angular momentum, :math:`\iota`, in :math:`\\rm rad`. If ``is_prec_ang=True`` this is interpreted as the inclination angle(s) with respect to total angular momentum, :math:`\\theta_{JN}`, in :math:`\\rm rad`.
        :param array or float psi: The polarisation angle(s), :math:`\psi`, in :math:`\\rm rad`.
        :param array or float tcoal: The time(s) of coalescence, :math:`t_{\\rm coal}`, as a GMST.
        :param array or float phase: The phase(s) at coalescence, :math:`\Phi_{\\rm coal}`, in :math:`\\rm rad`.
        :param array or float chiS: The symmetric spin component(s), :math:`\chi_s`. If :py:class:`self.wf_model` is precessing or ``use_chi1chi2=True`` this is interpreted as the spin component(s) of the primary object(s) along the axis :math:`z`, :math:`\chi_{1,z}`. If ``use_prec_ang=True`` this is interpreted as the spin magnitude(s) of the primary object(s), :math:`\chi_1`.
        :param array or float chiA: The antisymmetric spin component(s) :math:`\chi_a`. If :py:class:`self.wf_model` is precessing or ``use_chi1chi2=True`` this is interpreted as the spin component(s) of the secondary object(s) along the axis :math:`z`, :math:`\chi_{2,z}`. If ``use_prec_ang=True`` this is interpreted as the spin magnitude(s) of the secondary object(s), :math:`\chi_2`.
        :param array or float chi1x: The spin component(s) of the primary object(s) along the axis :math:`x`, :math:`\chi_{1,x}`. If ``use_prec_ang=True`` this is interpreted as the spin tilt angle(s) of the primary object(s), :math:`\\theta_{s,1}`, in :math:`\\rm rad`.
        :param array or float chi2x: The spin component(s) of the secondary object(s) along the axis :math:`x`, :math:`\chi_{2,x}`. If ``use_prec_ang=True`` this is interpreted as the spin tilt angle(s) of the secondary object(s), :math:`\\theta_{s,2}`, in :math:`\\rm rad`.
        :param array or float chi1y: spin component(s) of the primary object(s) along the axis :math:`y`, :math:`\chi_{1,y}`. If ``use_prec_ang=True`` this is interpreted as the azimuthal angle(s) of orbital angular momentum relative to total angular momentum, :math:`\phi_{JL}`, in :math:`\\rm rad`.
        :param array or float chi2y: spin component(s) of the secondary object(s) along the axis :math:`y`, :math:`\chi_{2,y}`. If ``use_prec_ang=True`` this is interpreted as the difference(s) in azimuthal angle between spin vectors, :math:`\phi_{1,2}`, in :math:`\\rm rad`.
        :param array or float LambdaTilde: The adimensional tidal deformability(ies) of combination :math:`\\tilde{\Lambda}`.
        :param array or float deltaLambda: The adimensional tidal deformability(ies) of combination :math:`\delta\\tilde{\Lambda}`.
        :param array or float ecc: The orbital eccentricity(ies), :math:`e_0`.
        :param float rot: Further rotation of the interferometer with respect to the :py:data:`self.xax` orientation, in degrees, needed for the triangular geometry.
        :param bool, optional use_m1m2: Boolean specifying if the ``Mc`` and ``eta`` inputs should be interpreted as the primary and secondary mass(es). In this case the derivatives are then taken with respect to ``m1`` and ``m2``.
        :param bool, optional use_chi1chi2: Boolean specifying if the ``chiS`` and ``chiA`` inputs should be interpreted as the primary and secondary spin components along the axis :math:`z`. In this case the derivatives are then taken with respect to ``chi1z`` and ``chi2z``.
        :param bool, optional use_prec_ang: Boolean specifying if the ``iota`` input should be interpreted as the inclination angle with respect to total angular momentum, ``chiS`` and ``chiA`` as the primary and secondary spin magnitudes, ``chi1x`` and ``chi2x`` as the primary and secondary spin tilts, ``chi1y`` as the azimuthal angle of orbital angular momentum relative to total angular momentum and ``chi2y`` as the difference in azimuthal angle between spin vectors. In this case the derivatives are then taken with respect to ``thetaJN``, ``chi1``, ``chi2``, ``tilt1``, ``tilt2``, ``phiJL`` and ``phi12``.
        :param bool, optional computeDerivFinDiff: Boolean specifying if the derivatives have to be computed using numerical differentiation (finite differences) through the `numdifftools <https://github.com/pbrod/numdifftools>`_ package.
        :param bool, optional computeAnalyticalDeriv: Boolean specifying if the derivatives with respect to ``dL``, ``theta``, ``phi``, ``psi``, ``tcoal``, ``phase`` and ``iota`` (the latter only for the fundamental mode in the non-precessing case) have to be computed analytically. This considerably speeds up the calculation and provides better accuracy.
        :param stepNDT: The step size to use in the computation with numerical differentiation (finite differences).
        :type stepNDT: float or numdifftools.step_generators.MaxStepGenerator
        :param str methodNDT: The method to use in the computation with numerical differentiation (finite differences). This can be ``'central'``, ``'complex'``, ``'multicomplex'``, ``'forward'`` or ``'backward'``.
        :return: Complete signal strain derivatives (complex), evaluated at the given parameters and frequency(ies).
        :rtype: array

        """
        # `numdifftools.step_generators <https://numdifftools.readthedocs.io/en/latest/reference/numdifftools.html#module-numdifftools.step_generators>`_
        if self.verbose:
            print("Computing derivatives...")
        # Function to compute the derivatives of a GW signal, both with JAX (automatic differentiation) and NumDiffTools (finite differences). It offers the possibility to compute directly the derivative of the complex signal. It is also possible to compute analytically the derivatives w.r.t. dL, theta, phi, psi, tcoal and phase, and also iota in absence of HM or precessing spins.

        if self.wf_model.is_newtonian:
            if self.verbose:
                print(
                    "WARNING: In the Newtonian inspiral case the mass ratio and spins do not enter the waveform, and the corresponding Fisher matrix elements vanish, we then discard them.\n"
                )

            if computeAnalyticalDeriv:
                derivargs = 1
                inputNumdL, inputNumiota = 1, 2
            else:
                derivargs = (1, 3, 4, 5, 6, 7, 8, 9)
        else:
            if computeAnalyticalDeriv:
                if (not self.wf_model.is_HigherModes) and (
                    not self.wf_model.is_Precessing
                ):
                    derivargs = (1, 2, 10, 11, 16, 17)
                elif self.wf_model.is_Precessing:
                    derivargs = (1, 2, 6, 10, 11, 12, 13, 14, 15, 16, 17)
                elif (not self.wf_model.is_Precessing) and self.wf_model.is_HigherModes:
                    derivargs = (1, 2, 6, 10, 11, 16, 17)
                inputNumdL, inputNumiota = 2, 3
            else:
                if not self.wf_model.is_Precessing:
                    derivargs = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 17)
                else:
                    # All 17 parameters are used
                    derivargs = tuple(range(17))
        if not self.wf_model.is_tidal:
            derivargs = derivargs[:-2]

        if self.wf_model.is_eccentric:
            derivargs = derivargs + (18,)

        nParams = self.wf_model.nParams

        if use_lensing:
            derivargs = derivargs + (
                19,
                20,
                21,
            )
            nParams = self.wf_model.nParams + 3

        if not computeDerivFinDiff:
            if self.wf_model.is_holomorphic:
                GWstrainUse = lambda f, Mc, eta, dL, theta, phi, iota, psi, tcoal, phase, chiS, chiA, chi1x, chi2x, chi1y, chi2y, LambdaTilde, deltaLambda, ecc, R_orbit, M_lz, src_pos: self.GWstrain(
                    f,
                    Mc,
                    eta,
                    dL,
                    theta,
                    phi,
                    iota,
                    psi,
                    tcoal,
                    phase,
                    chiS,
                    chiA,
                    chi1x,
                    chi2x,
                    chi1y,
                    chi2y,
                    LambdaTilde,
                    deltaLambda,
                    ecc,
                    R_orbit,
                    M_lz,
                    src_pos,
                    rot=rot,
                    is_m1m2=use_m1m2,
                    is_chi1chi2=use_chi1chi2,
                    is_prec_ang=use_prec_ang,
                    use_lensing=use_lensing,
                )

                FisherDerivs = np.asarray(
                    vmap(jacrev(GWstrainUse, argnums=derivargs, holomorphic=True))(
                        fgrids.T,
                        Mc,
                        eta,
                        dL,
                        theta,
                        phi,
                        iota,
                        psi,
                        tcoal,
                        phase,
                        chiS,
                        chiA,
                        chi1x,
                        chi2x,
                        chi1y,
                        chi2y,
                        LambdaTilde,
                        deltaLambda,
                        ecc,
                        R_orbit,
                        M_lz,
                        src_pos,
                    )
                )
            else:
                # In the non holomorphic case, to improve the accuracy, we compute separately the derivatives of the real and imaginary part of the strain as real functions
                (
                    fgrids,
                    Mc,
                    eta,
                    dL,
                    theta,
                    phi,
                    iota,
                    psi,
                    tcoal,
                    phase,
                    chiS,
                    chiA,
                    chi1x,
                    chi2x,
                    chi1y,
                    chi2y,
                    LambdaTilde,
                    deltaLambda,
                    ecc,
                    R_orbit,
                    M_lz,
                    src_pos,
                ) = (
                    np.real(fgrids),
                    np.real(Mc),
                    np.real(eta),
                    np.real(dL),
                    np.real(theta),
                    np.real(phi),
                    np.real(iota),
                    np.real(psi),
                    np.real(tcoal),
                    np.real(phase),
                    np.real(chiS),
                    np.real(chiA),
                    np.real(chi1x),
                    np.real(chi2x),
                    np.real(chi1y),
                    np.real(chi2y),
                    np.real(LambdaTilde),
                    np.real(deltaLambda),
                    np.real(ecc),
                    np.real(R_orbit),
                    np.real(M_lz),
                    np.real(src_pos),
                )

                GWstrainUse_real = lambda f, Mc, eta, dL, theta, phi, iota, psi, tcoal, phase, chiS, chiA, chi1x, chi2x, chi1y, chi2y, LambdaTilde, deltaLambda, ecc, R_orbit, M_lz, src_pos: np.real(
                    self.GWstrain(
                        f,
                        Mc,
                        eta,
                        dL,
                        theta,
                        phi,
                        iota,
                        psi,
                        tcoal,
                        phase,
                        chiS,
                        chiA,
                        chi1x,
                        chi2x,
                        chi1y,
                        chi2y,
                        LambdaTilde,
                        deltaLambda,
                        ecc,
                        R_orbit,
                        M_lz,
                        src_pos,
                        rot=rot,
                        is_m1m2=use_m1m2,
                        is_chi1chi2=use_chi1chi2,
                        is_prec_ang=use_prec_ang,
                        use_lensing=use_lensing,
                    )
                )
                GWstrainUse_imag = lambda f, Mc, eta, dL, theta, phi, iota, psi, tcoal, phase, chiS, chiA, chi1x, chi2x, chi1y, chi2y, LambdaTilde, deltaLambda, ecc, R_orbit, M_lz, src_pos: np.imag(
                    self.GWstrain(
                        f,
                        Mc,
                        eta,
                        dL,
                        theta,
                        phi,
                        iota,
                        psi,
                        tcoal,
                        phase,
                        chiS,
                        chiA,
                        chi1x,
                        chi2x,
                        chi1y,
                        chi2y,
                        LambdaTilde,
                        deltaLambda,
                        ecc,
                        R_orbit,
                        M_lz,
                        src_pos,
                        rot=rot,
                        is_m1m2=use_m1m2,
                        is_chi1chi2=use_chi1chi2,
                        is_prec_ang=use_prec_ang,
                        use_lensing=use_lensing,
                    )
                )

                realDerivs = np.asarray(
                    vmap(jacrev(GWstrainUse_real, argnums=derivargs))(
                        fgrids.T,
                        Mc,
                        eta,
                        dL,
                        theta,
                        phi,
                        iota,
                        psi,
                        tcoal,
                        phase,
                        chiS,
                        chiA,
                        chi1x,
                        chi2x,
                        chi1y,
                        chi2y,
                        LambdaTilde,
                        deltaLambda,
                        ecc,
                        R_orbit,
                        M_lz,
                        src_pos,
                    )
                )
                imagDerivs = np.asarray(
                    vmap(jacrev(GWstrainUse_imag, argnums=derivargs))(
                        fgrids.T,
                        Mc,
                        eta,
                        dL,
                        theta,
                        phi,
                        iota,
                        psi,
                        tcoal,
                        phase,
                        chiS,
                        chiA,
                        chi1x,
                        chi2x,
                        chi1y,
                        chi2y,
                        LambdaTilde,
                        deltaLambda,
                        ecc,
                        R_orbit,
                        M_lz,
                        src_pos,
                    )
                )

                FisherDerivs = realDerivs + 1j * imagDerivs
        else:
            # Lensing edits incomplete for the case of computeDerivFinDiff
            if self.wf_model.is_newtonian:
                if computeAnalyticalDeriv:
                    GWstrainUse = lambda pars: self.GWstrain(
                        fgrids,
                        pars[0],
                        eta,
                        dL,
                        theta,
                        phi,
                        iota,
                        psi,
                        tcoal,
                        phase,
                        chiS,
                        chiA,
                        chi1x,
                        chi2x,
                        chi1y,
                        chi2y,
                        LambdaTilde,
                        deltaLambda,
                        ecc,
                        R_orbit,
                        rot=rot,
                        is_m1m2=use_m1m2,
                        is_chi1chi2=use_chi1chi2,
                    )
                    evpars = [Mc]
                else:
                    GWstrainUse = lambda pars: self.GWstrain(
                        fgrids,
                        pars[0],
                        eta,
                        pars[1],
                        pars[2],
                        pars[3],
                        pars[4],
                        pars[5],
                        pars[6],
                        pars[7],
                        chiS,
                        chiA,
                        chi1x,
                        chi2x,
                        chi1y,
                        chi2y,
                        LambdaTilde,
                        deltaLambda,
                        ecc,
                        R_orbit,
                        rot=rot,
                        is_m1m2=use_m1m2,
                        is_chi1chi2=use_chi1chi2,
                    )
                    evpars = [Mc, dL, theta, phi, iota, psi, tcoal, phase]
            elif self.wf_model.is_tidal:
                if self.wf_model.is_Precessing:
                    if not self.wf_model.is_eccentric:
                        if not use_lensing:
                            if not computeAnalyticalDeriv:
                                GWstrainUse = lambda pars: self.GWstrain(
                                    fgrids,
                                    *pars[:17],
                                    ecc,
                                    R_orbit,
                                    rot=rot,
                                    is_m1m2=use_m1m2,
                                    is_chi1chi2=use_chi1chi2,
                                    is_prec_ang=use_prec_ang,
                                    use_lensing=use_lensing,
                                )
                                evpars = [
                                    Mc,
                                    eta,
                                    dL,
                                    theta,
                                    phi,
                                    iota,
                                    psi,
                                    tcoal,
                                    phase,
                                    chiS,
                                    chiA,
                                    chi1x,
                                    chi2x,
                                    chi1y,
                                    chi2y,
                                    LambdaTilde,
                                    deltaLambda,
                                ]
                            else:
                                GWstrainUse = lambda pars: self.GWstrain(
                                    fgrids,
                                    pars[0],
                                    pars[1],
                                    dL,
                                    theta,
                                    phi,
                                    pars[2],
                                    psi,
                                    tcoal,
                                    phase,
                                    *pars[3:11],
                                    ecc,
                                    R_orbit,
                                    rot=rot,
                                    is_m1m2=use_m1m2,
                                    is_chi1chi2=use_chi1chi2,
                                    is_prec_ang=use_prec_ang,
                                    use_lensing=use_lensing,
                                )
                                evpars = [
                                    Mc,
                                    eta,
                                    iota,
                                    chiS,
                                    chiA,
                                    chi1x,
                                    chi2x,
                                    chi1y,
                                    chi2y,
                                    LambdaTilde,
                                    deltaLambda,
                                ]
                        else:
                            if not computeAnalyticalDeriv:
                                GWstrainUse = lambda pars: self.GWstrain(
                                    fgrids,
                                    *pars[:17],
                                    ecc,
                                    pars[17],
                                    pars[18],
                                    rot=rot,
                                    is_m1m2=use_m1m2,
                                    is_chi1chi2=use_chi1chi2,
                                    is_prec_ang=use_prec_ang,
                                    use_lensing=use_lensing,
                                )
                                evpars = [
                                    Mc,
                                    eta,
                                    dL,
                                    theta,
                                    phi,
                                    iota,
                                    psi,
                                    tcoal,
                                    phase,
                                    chiS,
                                    chiA,
                                    chi1x,
                                    chi2x,
                                    chi1y,
                                    chi2y,
                                    LambdaTilde,
                                    deltaLambda,
                                    R_orbit,
                                ]
                            else:
                                GWstrainUse = lambda pars: self.GWstrain(
                                    fgrids,
                                    pars[0],
                                    pars[1],
                                    dL,
                                    theta,
                                    phi,
                                    pars[2],
                                    psi,
                                    tcoal,
                                    phase,
                                    *pars[3:11],
                                    ecc,
                                    pars[11],
                                    pars[12],
                                    rot=rot,
                                    is_m1m2=use_m1m2,
                                    is_chi1chi2=use_chi1chi2,
                                    is_prec_ang=use_prec_ang,
                                    use_lensing=use_lensing,
                                )
                                evpars = [
                                    Mc,
                                    eta,
                                    iota,
                                    chiS,
                                    chiA,
                                    chi1x,
                                    chi2x,
                                    chi1y,
                                    chi2y,
                                    LambdaTilde,
                                    deltaLambda,
                                    R_orbit,
                                ]
                    else:
                        if not use_lensing:
                            if not computeAnalyticalDeriv:
                                GWstrainUse = lambda pars: self.GWstrain(
                                    fgrids,
                                    *pars[:18],
                                    R_orbit,
                                    rot=rot,
                                    is_m1m2=use_m1m2,
                                    is_chi1chi2=use_chi1chi2,
                                    is_prec_ang=use_prec_ang,
                                    use_lensing=use_lensing,
                                )
                                evpars = [
                                    Mc,
                                    eta,
                                    dL,
                                    theta,
                                    phi,
                                    iota,
                                    psi,
                                    tcoal,
                                    phase,
                                    chiS,
                                    chiA,
                                    chi1x,
                                    chi2x,
                                    chi1y,
                                    chi2y,
                                    LambdaTilde,
                                    deltaLambda,
                                    ecc,
                                ]
                            else:
                                GWstrainUse = lambda pars: self.GWstrain(
                                    fgrids,
                                    pars[0],
                                    pars[1],
                                    dL,
                                    theta,
                                    phi,
                                    pars[2],
                                    psi,
                                    tcoal,
                                    phase,
                                    *pars[3:12],
                                    R_orbit,
                                    rot=rot,
                                    is_m1m2=use_m1m2,
                                    is_chi1chi2=use_chi1chi2,
                                    is_prec_ang=use_prec_ang,
                                    use_lensing=use_lensing,
                                )
                                evpars = [
                                    Mc,
                                    eta,
                                    iota,
                                    chiS,
                                    chiA,
                                    chi1x,
                                    chi2x,
                                    chi1y,
                                    chi2y,
                                    LambdaTilde,
                                    deltaLambda,
                                    ecc,
                                ]
                        else:
                            if not computeAnalyticalDeriv:
                                GWstrainUse = lambda pars: self.GWstrain(
                                    fgrids,
                                    *pars[:20],
                                    rot=rot,
                                    is_m1m2=use_m1m2,
                                    is_chi1chi2=use_chi1chi2,
                                    is_prec_ang=use_prec_ang,
                                    use_lensing=use_lensing,
                                )
                                evpars = [
                                    Mc,
                                    eta,
                                    dL,
                                    theta,
                                    phi,
                                    iota,
                                    psi,
                                    tcoal,
                                    phase,
                                    chiS,
                                    chiA,
                                    chi1x,
                                    chi2x,
                                    chi1y,
                                    chi2y,
                                    LambdaTilde,
                                    deltaLambda,
                                    ecc,
                                    R_orbit,
                                ]
                            else:
                                GWstrainUse = lambda pars: self.GWstrain(
                                    fgrids,
                                    pars[0],
                                    pars[1],
                                    dL,
                                    theta,
                                    phi,
                                    pars[2],
                                    psi,
                                    tcoal,
                                    phase,
                                    *pars[3:14],
                                    rot=rot,
                                    is_m1m2=use_m1m2,
                                    is_chi1chi2=use_chi1chi2,
                                    is_prec_ang=use_prec_ang,
                                    use_lensing=use_lensing,
                                )
                                evpars = [
                                    Mc,
                                    eta,
                                    iota,
                                    chiS,
                                    chiA,
                                    chi1x,
                                    chi2x,
                                    chi1y,
                                    chi2y,
                                    LambdaTilde,
                                    deltaLambda,
                                    ecc,
                                    R_orbit,
                                ]
                else:
                    if not self.wf_model.is_eccentric:
                        if not use_lensing:
                            if not computeAnalyticalDeriv:
                                GWstrainUse = lambda pars: self.GWstrain(
                                    fgrids,
                                    *pars[:11],
                                    chi1x,
                                    chi2x,
                                    chi1y,
                                    chi2y,
                                    pars[11],
                                    pars[12],
                                    ecc,
                                    R_orbit,
                                    rot=rot,
                                    is_m1m2=use_m1m2,
                                    is_chi1chi2=use_chi1chi2,
                                    use_lensing=use_lensing,
                                )
                                evpars = [
                                    Mc,
                                    eta,
                                    dL,
                                    theta,
                                    phi,
                                    iota,
                                    psi,
                                    tcoal,
                                    phase,
                                    chiS,
                                    chiA,
                                    LambdaTilde,
                                    deltaLambda,
                                ]
                            else:
                                if not self.wf_model.is_HigherModes:
                                    GWstrainUse = lambda pars: self.GWstrain(
                                        fgrids,
                                        pars[0],
                                        pars[1],
                                        dL,
                                        theta,
                                        phi,
                                        iota,
                                        psi,
                                        tcoal,
                                        phase,
                                        pars[2],
                                        pars[3],
                                        chi1x,
                                        chi2x,
                                        chi1y,
                                        chi2y,
                                        pars[4],
                                        pars[5],
                                        ecc,
                                        R_orbit,
                                        rot=rot,
                                        is_m1m2=use_m1m2,
                                        is_chi1chi2=use_chi1chi2,
                                        use_lensing=use_lensing,
                                    )
                                    evpars = [
                                        Mc,
                                        eta,
                                        chiS,
                                        chiA,
                                        LambdaTilde,
                                        deltaLambda,
                                    ]
                                else:
                                    GWstrainUse = lambda pars: self.GWstrain(
                                        fgrids,
                                        pars[0],
                                        pars[1],
                                        dL,
                                        theta,
                                        phi,
                                        pars[2],
                                        psi,
                                        tcoal,
                                        phase,
                                        pars[3],
                                        pars[4],
                                        chi1x,
                                        chi2x,
                                        chi1y,
                                        chi2y,
                                        pars[5],
                                        pars[6],
                                        ecc,
                                        R_orbit,
                                        rot=rot,
                                        is_m1m2=use_m1m2,
                                        is_chi1chi2=use_chi1chi2,
                                        use_lensing=use_lensing,
                                    )
                                    evpars = [
                                        Mc,
                                        eta,
                                        iota,
                                        chiS,
                                        chiA,
                                        LambdaTilde,
                                        deltaLambda,
                                    ]
                        else:
                            if not computeAnalyticalDeriv:
                                GWstrainUse = lambda pars: self.GWstrain(
                                    fgrids,
                                    *pars[:11],
                                    chi1x,
                                    chi2x,
                                    chi1y,
                                    chi2y,
                                    pars[11],
                                    pars[12],
                                    ecc,
                                    pars[13],
                                    pars[14],
                                    R_orbit,
                                    rot=rot,
                                    is_m1m2=use_m1m2,
                                    is_chi1chi2=use_chi1chi2,
                                    use_lensing=use_lensing,
                                )
                                evpars = [
                                    Mc,
                                    eta,
                                    dL,
                                    theta,
                                    phi,
                                    iota,
                                    psi,
                                    tcoal,
                                    phase,
                                    chiS,
                                    chiA,
                                    LambdaTilde,
                                    deltaLambda,
                                    R_orbit,
                                ]
                            else:
                                if not self.wf_model.is_HigherModes:
                                    GWstrainUse = lambda pars: self.GWstrain(
                                        fgrids,
                                        pars[0],
                                        pars[1],
                                        dL,
                                        theta,
                                        phi,
                                        iota,
                                        psi,
                                        tcoal,
                                        phase,
                                        pars[2],
                                        pars[3],
                                        chi1x,
                                        chi2x,
                                        chi1y,
                                        chi2y,
                                        pars[4],
                                        pars[5],
                                        ecc,
                                        pars[6],
                                        pars[7],
                                        rot=rot,
                                        is_m1m2=use_m1m2,
                                        is_chi1chi2=use_chi1chi2,
                                        use_lensing=use_lensing,
                                    )
                                    evpars = [
                                        Mc,
                                        eta,
                                        chiS,
                                        chiA,
                                        LambdaTilde,
                                        deltaLambda,
                                        R_orbit,
                                    ]
                                else:
                                    GWstrainUse = lambda pars: self.GWstrain(
                                        fgrids,
                                        pars[0],
                                        pars[1],
                                        dL,
                                        theta,
                                        phi,
                                        pars[2],
                                        psi,
                                        tcoal,
                                        phase,
                                        pars[3],
                                        pars[4],
                                        chi1x,
                                        chi2x,
                                        chi1y,
                                        chi2y,
                                        pars[5],
                                        pars[6],
                                        ecc,
                                        pars[7],
                                        pars[8],
                                        rot=rot,
                                        is_m1m2=use_m1m2,
                                        is_chi1chi2=use_chi1chi2,
                                        use_lensing=use_lensing,
                                    )
                                    evpars = [
                                        Mc,
                                        eta,
                                        iota,
                                        chiS,
                                        chiA,
                                        LambdaTilde,
                                        deltaLambda,
                                        R_orbit,
                                    ]
                    else:
                        if not use_lensing:
                            if not computeAnalyticalDeriv:
                                GWstrainUse = lambda pars: self.GWstrain(
                                    fgrids,
                                    *pars[:11],
                                    chi1x,
                                    chi2x,
                                    chi1y,
                                    chi2y,
                                    pars[11],
                                    pars[12],
                                    pars[13],
                                    R_orbit,
                                    rot=rot,
                                    is_m1m2=use_m1m2,
                                    is_chi1chi2=use_chi1chi2,
                                    use_lensing=use_lensing,
                                )
                                evpars = [
                                    Mc,
                                    eta,
                                    dL,
                                    theta,
                                    phi,
                                    iota,
                                    psi,
                                    tcoal,
                                    phase,
                                    chiS,
                                    chiA,
                                    LambdaTilde,
                                    deltaLambda,
                                    ecc,
                                ]
                            else:
                                if not self.wf_model.is_HigherModes:
                                    GWstrainUse = lambda pars: self.GWstrain(
                                        fgrids,
                                        pars[0],
                                        pars[1],
                                        dL,
                                        theta,
                                        phi,
                                        iota,
                                        psi,
                                        tcoal,
                                        phase,
                                        pars[2],
                                        pars[3],
                                        chi1x,
                                        chi2x,
                                        chi1y,
                                        chi2y,
                                        pars[4],
                                        pars[5],
                                        pars[6],
                                        R_orbit,
                                        rot=rot,
                                        is_m1m2=use_m1m2,
                                        is_chi1chi2=use_chi1chi2,
                                        use_lensing=use_lensing,
                                    )
                                    evpars = [
                                        Mc,
                                        eta,
                                        chiS,
                                        chiA,
                                        LambdaTilde,
                                        deltaLambda,
                                        ecc,
                                    ]
                                else:
                                    GWstrainUse = lambda pars: self.GWstrain(
                                        fgrids,
                                        pars[0],
                                        pars[1],
                                        dL,
                                        theta,
                                        phi,
                                        pars[2],
                                        psi,
                                        tcoal,
                                        phase,
                                        pars[3],
                                        pars[4],
                                        chi1x,
                                        chi2x,
                                        chi1y,
                                        chi2y,
                                        pars[5],
                                        pars[6],
                                        pars[7],
                                        R_orbit,
                                        rot=rot,
                                        is_m1m2=use_m1m2,
                                        is_chi1chi2=use_chi1chi2,
                                        use_lensing=use_lensing,
                                    )
                                    evpars = [
                                        Mc,
                                        eta,
                                        iota,
                                        chiS,
                                        chiA,
                                        LambdaTilde,
                                        deltaLambda,
                                        ecc,
                                    ]
                        else:
                            if not computeAnalyticalDeriv:
                                GWstrainUse = lambda pars: self.GWstrain(
                                    fgrids,
                                    *pars[:11],
                                    chi1x,
                                    chi2x,
                                    chi1y,
                                    chi2y,
                                    *pars[11:16],
                                    rot=rot,
                                    is_m1m2=use_m1m2,
                                    is_chi1chi2=use_chi1chi2,
                                    use_lensing=use_lensing,
                                )
                                evpars = [
                                    Mc,
                                    eta,
                                    dL,
                                    theta,
                                    phi,
                                    iota,
                                    psi,
                                    tcoal,
                                    phase,
                                    chiS,
                                    chiA,
                                    LambdaTilde,
                                    deltaLambda,
                                    ecc,
                                    R_orbit,
                                ]
                            else:
                                if not self.wf_model.is_HigherModes:
                                    GWstrainUse = lambda pars: self.GWstrain(
                                        fgrids,
                                        pars[0],
                                        pars[1],
                                        dL,
                                        theta,
                                        phi,
                                        iota,
                                        psi,
                                        tcoal,
                                        phase,
                                        pars[2],
                                        pars[3],
                                        chi1x,
                                        chi2x,
                                        chi1y,
                                        chi2y,
                                        *pars[4:9],
                                        rot=rot,
                                        is_m1m2=use_m1m2,
                                        is_chi1chi2=use_chi1chi2,
                                        use_lensing=use_lensing,
                                    )
                                    evpars = [
                                        Mc,
                                        eta,
                                        chiS,
                                        chiA,
                                        LambdaTilde,
                                        deltaLambda,
                                        ecc,
                                        R_orbit,
                                    ]
                                else:
                                    GWstrainUse = lambda pars: self.GWstrain(
                                        fgrids,
                                        pars[0],
                                        pars[1],
                                        dL,
                                        theta,
                                        phi,
                                        pars[2],
                                        psi,
                                        tcoal,
                                        phase,
                                        pars[3],
                                        pars[4],
                                        chi1x,
                                        chi2x,
                                        chi1y,
                                        chi2y,
                                        pars[5],
                                        pars[6],
                                        pars[8],
                                        pars[9],
                                        rot=rot,
                                        is_m1m2=use_m1m2,
                                        is_chi1chi2=use_chi1chi2,
                                        use_lensing=use_lensing,
                                    )
                                    evpars = [
                                        Mc,
                                        eta,
                                        iota,
                                        chiS,
                                        chiA,
                                        LambdaTilde,
                                        deltaLambda,
                                        ecc,
                                        R_orbit,
                                    ]
            else:
                if self.wf_model.is_Precessing:
                    if not self.wf_model.is_eccentric:
                        if not use_lensing:
                            if not computeAnalyticalDeriv:
                                GWstrainUse = lambda pars: self.GWstrain(
                                    fgrids,
                                    *pars[:15],
                                    LambdaTilde,
                                    deltaLambda,
                                    ecc,
                                    R_orbit,
                                    rot=rot,
                                    is_m1m2=use_m1m2,
                                    is_chi1chi2=use_chi1chi2,
                                    is_prec_ang=use_prec_ang,
                                    use_lensing=use_lensing,
                                )
                                evpars = [
                                    Mc,
                                    eta,
                                    dL,
                                    theta,
                                    phi,
                                    iota,
                                    psi,
                                    tcoal,
                                    phase,
                                    chiS,
                                    chiA,
                                    chi1x,
                                    chi2x,
                                    chi1y,
                                    chi2y,
                                ]
                            else:
                                GWstrainUse = lambda pars: self.GWstrain(
                                    fgrids,
                                    pars[0],
                                    pars[1],
                                    dL,
                                    theta,
                                    phi,
                                    pars[2],
                                    psi,
                                    tcoal,
                                    phase,
                                    *pars[3:9],
                                    LambdaTilde,
                                    deltaLambda,
                                    ecc,
                                    R_orbit,
                                    rot=rot,
                                    is_m1m2=use_m1m2,
                                    is_chi1chi2=use_chi1chi2,
                                    is_prec_ang=use_prec_ang,
                                    use_lensing=use_lensing,
                                )
                                evpars = [
                                    Mc,
                                    eta,
                                    iota,
                                    chiS,
                                    chiA,
                                    chi1x,
                                    chi2x,
                                    chi1y,
                                    chi2y,
                                ]
                        else:
                            if not computeAnalyticalDeriv:
                                GWstrainUse = lambda pars: self.GWstrain(
                                    fgrids,
                                    *pars[:15],
                                    LambdaTilde,
                                    deltaLambda,
                                    ecc,
                                    pars[15],
                                    pars[16],
                                    rot=rot,
                                    is_m1m2=use_m1m2,
                                    is_chi1chi2=use_chi1chi2,
                                    is_prec_ang=use_prec_ang,
                                    use_lensing=use_lensing,
                                )
                                evpars = [
                                    Mc,
                                    eta,
                                    dL,
                                    theta,
                                    phi,
                                    iota,
                                    psi,
                                    tcoal,
                                    phase,
                                    chiS,
                                    chiA,
                                    chi1x,
                                    chi2x,
                                    chi1y,
                                    chi2y,
                                    R_orbit,
                                ]
                            else:
                                GWstrainUse = lambda pars: self.GWstrain(
                                    fgrids,
                                    pars[0],
                                    pars[1],
                                    dL,
                                    theta,
                                    phi,
                                    pars[2],
                                    psi,
                                    tcoal,
                                    phase,
                                    *pars[3:9],
                                    LambdaTilde,
                                    deltaLambda,
                                    ecc,
                                    pars[9],
                                    pars[10],
                                    rot=rot,
                                    is_m1m2=use_m1m2,
                                    is_chi1chi2=use_chi1chi2,
                                    is_prec_ang=use_prec_ang,
                                    use_lensing=use_lensing,
                                )
                                evpars = [
                                    Mc,
                                    eta,
                                    iota,
                                    chiS,
                                    chiA,
                                    chi1x,
                                    chi2x,
                                    chi1y,
                                    chi2y,
                                    R_orbit,
                                ]
                    else:
                        if not use_lensing:
                            if not computeAnalyticalDeriv:
                                GWstrainUse = lambda pars: self.GWstrain(
                                    fgrids,
                                    *pars[:15],
                                    LambdaTilde,
                                    deltaLambda,
                                    pars[15],
                                    R_orbit,
                                    rot=rot,
                                    is_m1m2=use_m1m2,
                                    is_chi1chi2=use_chi1chi2,
                                    is_prec_ang=use_prec_ang,
                                    use_lensing=use_lensing,
                                )
                                evpars = [
                                    Mc,
                                    eta,
                                    dL,
                                    theta,
                                    phi,
                                    iota,
                                    psi,
                                    tcoal,
                                    phase,
                                    chiS,
                                    chiA,
                                    chi1x,
                                    chi2x,
                                    chi1y,
                                    chi2y,
                                    ecc,
                                ]
                            else:
                                GWstrainUse = lambda pars: self.GWstrain(
                                    fgrids,
                                    pars[0],
                                    pars[1],
                                    dL,
                                    theta,
                                    phi,
                                    pars[2],
                                    psi,
                                    tcoal,
                                    phase,
                                    *pars[3:9],
                                    LambdaTilde,
                                    deltaLambda,
                                    pars[9],
                                    R_orbit,
                                    rot=rot,
                                    is_m1m2=use_m1m2,
                                    is_chi1chi2=use_chi1chi2,
                                    is_prec_ang=use_prec_ang,
                                    use_lensing=use_lensing,
                                )
                                evpars = [
                                    Mc,
                                    eta,
                                    iota,
                                    chiS,
                                    chiA,
                                    chi1x,
                                    chi2x,
                                    chi1y,
                                    chi2y,
                                    ecc,
                                ]
                        else:
                            if not computeAnalyticalDeriv:
                                GWstrainUse = lambda pars: self.GWstrain(
                                    fgrids,
                                    *pars[:15],
                                    LambdaTilde,
                                    deltaLambda,
                                    pars[15],
                                    pars[16],
                                    pars[17],
                                    rot=rot,
                                    is_m1m2=use_m1m2,
                                    is_chi1chi2=use_chi1chi2,
                                    is_prec_ang=use_prec_ang,
                                    use_lensing=use_lensing,
                                )
                                evpars = [
                                    Mc,
                                    eta,
                                    dL,
                                    theta,
                                    phi,
                                    iota,
                                    psi,
                                    tcoal,
                                    phase,
                                    chiS,
                                    chiA,
                                    chi1x,
                                    chi2x,
                                    chi1y,
                                    chi2y,
                                    ecc,
                                    R_orbit,
                                ]
                            else:
                                GWstrainUse = lambda pars: self.GWstrain(
                                    fgrids,
                                    pars[0],
                                    pars[1],
                                    dL,
                                    theta,
                                    phi,
                                    pars[2],
                                    psi,
                                    tcoal,
                                    phase,
                                    *pars[3:9],
                                    LambdaTilde,
                                    deltaLambda,
                                    pars[9],
                                    pars[10],
                                    pars[11],
                                    rot=rot,
                                    is_m1m2=use_m1m2,
                                    is_chi1chi2=use_chi1chi2,
                                    is_prec_ang=use_prec_ang,
                                    use_lensing=use_lensing,
                                )
                                evpars = [
                                    Mc,
                                    eta,
                                    iota,
                                    chiS,
                                    chiA,
                                    chi1x,
                                    chi2x,
                                    chi1y,
                                    chi2y,
                                    ecc,
                                    R_orbit,
                                ]
                else:
                    if not self.wf_model.is_eccentric:
                        if not use_lensing:
                            if not computeAnalyticalDeriv:
                                GWstrainUse = lambda pars: self.GWstrain(
                                    fgrids,
                                    *pars[:11],
                                    chi1x,
                                    chi2x,
                                    chi1y,
                                    chi2y,
                                    LambdaTilde,
                                    deltaLambda,
                                    ecc,
                                    R_orbit,
                                    rot=rot,
                                    is_m1m2=use_m1m2,
                                    is_chi1chi2=use_chi1chi2,
                                    use_lensing=use_lensing,
                                )
                                evpars = [
                                    Mc,
                                    eta,
                                    dL,
                                    theta,
                                    phi,
                                    iota,
                                    psi,
                                    tcoal,
                                    phase,
                                    chiS,
                                    chiA,
                                ]
                            else:
                                if not self.wf_model.is_HigherModes:
                                    GWstrainUse = lambda pars: self.GWstrain(
                                        fgrids,
                                        pars[0],
                                        pars[1],
                                        dL,
                                        theta,
                                        phi,
                                        iota,
                                        psi,
                                        tcoal,
                                        phase,
                                        pars[2],
                                        pars[3],
                                        chi1x,
                                        chi2x,
                                        chi1y,
                                        chi2y,
                                        LambdaTilde,
                                        deltaLambda,
                                        ecc,
                                        R_orbit,
                                        rot=rot,
                                        is_m1m2=use_m1m2,
                                        is_chi1chi2=use_chi1chi2,
                                        use_lensing=use_lensing,
                                    )
                                    evpars = [Mc, eta, chiS, chiA]
                                else:
                                    GWstrainUse = lambda pars: self.GWstrain(
                                        fgrids,
                                        pars[0],
                                        pars[1],
                                        dL,
                                        theta,
                                        phi,
                                        pars[2],
                                        psi,
                                        tcoal,
                                        phase,
                                        pars[3],
                                        pars[4],
                                        chi1x,
                                        chi2x,
                                        chi1y,
                                        chi2y,
                                        LambdaTilde,
                                        deltaLambda,
                                        ecc,
                                        R_orbit,
                                        rot=rot,
                                        is_m1m2=use_m1m2,
                                        is_chi1chi2=use_chi1chi2,
                                        use_lensing=use_lensing,
                                    )
                                    evpars = [Mc, eta, iota, chiS, chiA]
                        else:
                            if not computeAnalyticalDeriv:
                                GWstrainUse = lambda pars: self.GWstrain(
                                    fgrids,
                                    *pars[:11],
                                    chi1x,
                                    chi2x,
                                    chi1y,
                                    chi2y,
                                    LambdaTilde,
                                    deltaLambda,
                                    ecc,
                                    pars[11],
                                    pars[12],
                                    rot=rot,
                                    is_m1m2=use_m1m2,
                                    is_chi1chi2=use_chi1chi2,
                                    use_lensing=use_lensing,
                                )
                                evpars = [
                                    Mc,
                                    eta,
                                    dL,
                                    theta,
                                    phi,
                                    iota,
                                    psi,
                                    tcoal,
                                    phase,
                                    chiS,
                                    chiA,
                                    R_orbit,
                                ]
                            else:
                                if not self.wf_model.is_HigherModes:
                                    GWstrainUse = lambda pars: self.GWstrain(
                                        fgrids,
                                        pars[0],
                                        pars[1],
                                        dL,
                                        theta,
                                        phi,
                                        iota,
                                        psi,
                                        tcoal,
                                        phase,
                                        pars[2],
                                        pars[3],
                                        chi1x,
                                        chi2x,
                                        chi1y,
                                        chi2y,
                                        LambdaTilde,
                                        deltaLambda,
                                        ecc,
                                        pars[4],
                                        pars[5],
                                        rot=rot,
                                        is_m1m2=use_m1m2,
                                        is_chi1chi2=use_chi1chi2,
                                        use_lensing=use_lensing,
                                    )
                                    evpars = [Mc, eta, chiS, chiA, R_orbit]
                                else:
                                    GWstrainUse = lambda pars: self.GWstrain(
                                        fgrids,
                                        pars[0],
                                        pars[1],
                                        dL,
                                        theta,
                                        phi,
                                        pars[2],
                                        psi,
                                        tcoal,
                                        phase,
                                        pars[3],
                                        pars[4],
                                        chi1x,
                                        chi2x,
                                        chi1y,
                                        chi2y,
                                        LambdaTilde,
                                        deltaLambda,
                                        ecc,
                                        pars[5],
                                        pars[6],
                                        rot=rot,
                                        is_m1m2=use_m1m2,
                                        is_chi1chi2=use_chi1chi2,
                                        use_lensing=use_lensing,
                                    )
                                    evpars = [Mc, eta, iota, chiS, chiA, R_orbit]
                    else:
                        if not use_lensing:
                            if not computeAnalyticalDeriv:
                                GWstrainUse = lambda pars: self.GWstrain(
                                    fgrids,
                                    *pars[:11],
                                    chi1x,
                                    chi2x,
                                    chi1y,
                                    chi2y,
                                    LambdaTilde,
                                    deltaLambda,
                                    pars[11],
                                    R_orbit,
                                    rot=rot,
                                    is_m1m2=use_m1m2,
                                    is_chi1chi2=use_chi1chi2,
                                    use_lensing=use_lensing,
                                )
                                evpars = [
                                    Mc,
                                    eta,
                                    dL,
                                    theta,
                                    phi,
                                    iota,
                                    psi,
                                    tcoal,
                                    phase,
                                    chiS,
                                    chiA,
                                    ecc,
                                ]
                            else:
                                if not self.wf_model.is_HigherModes:
                                    GWstrainUse = lambda pars: self.GWstrain(
                                        fgrids,
                                        pars[0],
                                        pars[1],
                                        dL,
                                        theta,
                                        phi,
                                        iota,
                                        psi,
                                        tcoal,
                                        phase,
                                        pars[2],
                                        pars[3],
                                        chi1x,
                                        chi2x,
                                        chi1y,
                                        chi2y,
                                        LambdaTilde,
                                        deltaLambda,
                                        pars[4],
                                        R_orbit,
                                        rot=rot,
                                        is_m1m2=use_m1m2,
                                        is_chi1chi2=use_chi1chi2,
                                        use_lensing=use_lensing,
                                    )
                                    evpars = [Mc, eta, chiS, chiA, ecc]
                                else:
                                    GWstrainUse = lambda pars: self.GWstrain(
                                        fgrids,
                                        pars[0],
                                        pars[1],
                                        dL,
                                        theta,
                                        phi,
                                        pars[2],
                                        psi,
                                        tcoal,
                                        phase,
                                        pars[3],
                                        pars[4],
                                        chi1x,
                                        chi2x,
                                        chi1y,
                                        chi2y,
                                        LambdaTilde,
                                        deltaLambda,
                                        pars[5],
                                        R_orbit,
                                        rot=rot,
                                        is_m1m2=use_m1m2,
                                        is_chi1chi2=use_chi1chi2,
                                        use_lensing=use_lensing,
                                    )
                                    evpars = [Mc, eta, iota, chiS, chiA, ecc]
                        else:
                            if not computeAnalyticalDeriv:
                                GWstrainUse = lambda pars: self.GWstrain(
                                    fgrids,
                                    *pars[:11],
                                    chi1x,
                                    chi2x,
                                    chi1y,
                                    chi2y,
                                    LambdaTilde,
                                    deltaLambda,
                                    pars[11],
                                    pars[12],
                                    pars[13],
                                    rot=rot,
                                    is_m1m2=use_m1m2,
                                    is_chi1chi2=use_chi1chi2,
                                    use_lensing=use_lensing,
                                )
                                evpars = [
                                    Mc,
                                    eta,
                                    dL,
                                    theta,
                                    phi,
                                    iota,
                                    psi,
                                    tcoal,
                                    phase,
                                    chiS,
                                    chiA,
                                    ecc,
                                    R_orbit,
                                ]
                            else:
                                if not self.wf_model.is_HigherModes:
                                    GWstrainUse = lambda pars: self.GWstrain(
                                        fgrids,
                                        pars[0],
                                        pars[1],
                                        dL,
                                        theta,
                                        phi,
                                        iota,
                                        psi,
                                        tcoal,
                                        phase,
                                        pars[2],
                                        pars[3],
                                        chi1x,
                                        chi2x,
                                        chi1y,
                                        chi2y,
                                        LambdaTilde,
                                        deltaLambda,
                                        pars[4],
                                        pars[5],
                                        pars[6],
                                        rot=rot,
                                        is_m1m2=use_m1m2,
                                        is_chi1chi2=use_chi1chi2,
                                        use_lensing=use_lensing,
                                    )
                                    evpars = [Mc, eta, chiS, chiA, ecc, R_orbit]
                                else:
                                    GWstrainUse = lambda pars: self.GWstrain(
                                        fgrids,
                                        pars[0],
                                        pars[1],
                                        dL,
                                        theta,
                                        phi,
                                        pars[2],
                                        psi,
                                        tcoal,
                                        phase,
                                        pars[3],
                                        pars[4],
                                        chi1x,
                                        chi2x,
                                        chi1y,
                                        chi2y,
                                        LambdaTilde,
                                        deltaLambda,
                                        pars[5],
                                        pars[6],
                                        pars[7],
                                        rot=rot,
                                        is_m1m2=use_m1m2,
                                        is_chi1chi2=use_chi1chi2,
                                        use_lensing=use_lensing,
                                    )
                                    evpars = [Mc, eta, iota, chiS, chiA, ecc, R_orbit]

            dh = ndt.Jacobian(GWstrainUse, step=stepNDT, method=methodNDT, order=2, n=1)
            FisherDerivs = np.asarray(dh(evpars))
            if len(FisherDerivs.shape) == 2:  # len(Mc) == 1:
                FisherDerivs = FisherDerivs[:, :, np.newaxis]
            FisherDerivs = FisherDerivs.transpose(1, 2, 0)

        if computeAnalyticalDeriv:
            # We compute the derivative w.r.t. dL, theta, phi, iota, psi, tcoal and phase analytically, so have to split the matrix and insert them
            if (not self.wf_model.is_HigherModes) and (not self.wf_model.is_Precessing):
                NAnalyticalDerivs = 7
            else:
                NAnalyticalDerivs = 6

            (
                dL_deriv,
                theta_deriv,
                phi_deriv,
                iota_deriv,
                psi_deriv,
                tc_deriv,
                phase_deriv,
            ) = self._AnalyticalDerivatives(
                fgrids,
                Mc,
                eta,
                dL,
                theta,
                phi,
                iota,
                psi,
                tcoal,
                phase,
                chiS,
                chiA,
                chi1x,
                chi2x,
                chi1y,
                chi2y,
                LambdaTilde,
                deltaLambda,
                ecc,
                R_orbit,
                M_lz,
                src_pos,
                rot=rot,
                use_m1m2=use_m1m2,
                use_chi1chi2=use_chi1chi2,
                use_prec_ang=use_prec_ang,
                use_lensing=use_lensing,
            )
            if (not self.wf_model.is_HigherModes) and (not self.wf_model.is_Precessing):
                if not self.wf_model.is_newtonian:
                    tmpsplit1, tmpsplit2, _ = onp.vsplit(
                        FisherDerivs,
                        onp.array([inputNumdL, nParams - NAnalyticalDerivs]),
                    )
                    FisherDerivs = np.vstack(
                        (
                            tmpsplit1,
                            np.asarray(dL_deriv).T[np.newaxis, :],
                            np.asarray(theta_deriv).T[np.newaxis, :],
                            np.asarray(phi_deriv).T[np.newaxis, :],
                            np.asarray(iota_deriv).T[np.newaxis, :],
                            np.asarray(psi_deriv).T[np.newaxis, :],
                            np.asarray(tc_deriv).T[np.newaxis, :],
                            np.asarray(phase_deriv).T[np.newaxis, :],
                            tmpsplit2,
                        )
                    )
                else:
                    FisherDerivs = np.vstack(
                        (
                            FisherDerivs[np.newaxis, :],
                            np.asarray(dL_deriv).T[np.newaxis, :],
                            np.asarray(theta_deriv).T[np.newaxis, :],
                            np.asarray(phi_deriv).T[np.newaxis, :],
                            np.asarray(iota_deriv).T[np.newaxis, :],
                            np.asarray(psi_deriv).T[np.newaxis, :],
                            np.asarray(tc_deriv).T[np.newaxis, :],
                            np.asarray(phase_deriv).T[np.newaxis, :],
                        )
                    )
            else:
                tmpsplit1, tmpsplit2, tmpsplit3, _ = onp.vsplit(
                    FisherDerivs,
                    onp.array([inputNumdL, inputNumiota, nParams - NAnalyticalDerivs]),
                )
                FisherDerivs = np.vstack(
                    (
                        tmpsplit1,
                        np.asarray(dL_deriv).T[np.newaxis, :],
                        np.asarray(theta_deriv).T[np.newaxis, :],
                        np.asarray(phi_deriv).T[np.newaxis, :],
                        tmpsplit2,
                        np.asarray(psi_deriv).T[np.newaxis, :],
                        np.asarray(tc_deriv).T[np.newaxis, :],
                        np.asarray(phase_deriv).T[np.newaxis, :],
                        tmpsplit3,
                    )
                )

        return FisherDerivs

    def _AnalyticalDerivatives(
        self,
        f,
        Mc,
        eta,
        dL,
        theta,
        phi,
        iota,
        psi,
        tcoal,
        phase,
        chiS,
        chiA,
        chi1x,
        chi2x,
        chi1y,
        chi2y,
        LambdaTilde,
        deltaLambda,
        ecc,
        phi_L=None,
        R_orbit=None,
        M_lz=None,
        src_pos=None,
        rot=0.0,
        use_m1m2=False,
        use_chi1chi2=False,
        use_prec_ang=False,
        use_lensing=False,
    ):
        """
        Compute analytical derivatives with respect to ``dL``, ``theta``, ``phi``, ``psi``, ``tcoal``, ``phase`` and ``iota`` (the latter only for the fundamental mode in the non-precessing case).

        :param array or float f: The frequency(ies) at which to perform the calculation, in :math:`\\rm Hz`.
        :param array or float Mc: The chirp mass(es), :math:`{\cal M}_c`, in units of :math:`\\rm M_{\odot}`. If ``use_m1m2=True`` this is interpreted as the primary mass, :math:`m_1`, in units of :math:`\\rm M_{\odot}`.
        :param array or float eta:  The symmetric mass ratio(s), :math:`\eta`. If ``use_m1m2=True`` this is interpreted as the secondary mass, :math:`m_2`, in units of :math:`\\rm M_{\odot}`.
        :param array or float dL: The luminosity distance(s), :math:`d_L`, in :math:`\\rm Gpc`.
        :param array or float theta: The :math:`\\theta` sky position angle(s), in :math:`\\rm rad`.
        :param array or float phi: The :math:`\phi` sky position angle(s), in :math:`\\rm rad`.
        :param array or float iota: The inclination angle(s), with respect to orbital angular momentum, :math:`\iota`, in :math:`\\rm rad`. If ``is_prec_ang=True`` this is interpreted as the inclination angle(s) with respect to total angular momentum, :math:`\\theta_{JN}`, in :math:`\\rm rad`.
        :param array or float psi: The polarisation angle(s), :math:`\psi`, in :math:`\\rm rad`.
        :param array or float tcoal: The time(s) of coalescence, :math:`t_{\\rm coal}`, as a GMST.
        :param array or float phase: The phase(s) at coalescence, :math:`\Phi_{\\rm coal}`, in :math:`\\rm rad`.
        :param array or float chiS: The symmetric spin component(s), :math:`\chi_s`. If :py:class:`self.wf_model` is precessing or ``use_chi1chi2=True`` this is interpreted as the spin component(s) of the primary object(s) along the axis :math:`z`, :math:`\chi_{1,z}`. If ``use_prec_ang=True`` this is interpreted as the spin magnitude(s) of the primary object(s), :math:`\chi_1`.
        :param array or float chiA: The antisymmetric spin component(s) :math:`\chi_a`. If :py:class:`self.wf_model` is precessing or ``use_chi1chi2=True`` this is interpreted as the spin component(s) of the secondary object(s) along the axis :math:`z`, :math:`\chi_{2,z}`. If ``use_prec_ang=True`` this is interpreted as the spin magnitude(s) of the secondary object(s), :math:`\chi_2`.
        :param array or float chi1x: The spin component(s) of the primary object(s) along the axis :math:`x`, :math:`\chi_{1,x}`. If ``use_prec_ang=True`` this is interpreted as the spin tilt angle(s) of the primary object(s), :math:`\\theta_{s,1}`, in :math:`\\rm rad`.
        :param array or float chi2x: The spin component(s) of the secondary object(s) along the axis :math:`x`, :math:`\chi_{2,x}`. If ``use_prec_ang=True`` this is interpreted as the spin tilt angle(s) of the secondary object(s), :math:`\\theta_{s,2}`, in :math:`\\rm rad`.
        :param array or float chi1y: spin component(s) of the primary object(s) along the axis :math:`y`, :math:`\chi_{1,y}`. If ``use_prec_ang=True`` this is interpreted as the azimuthal angle(s) of orbital angular momentum relative to total angular momentum, :math:`\phi_{JL}`, in :math:`\\rm rad`.
        :param array or float chi2y: spin component(s) of the secondary object(s) along the axis :math:`y`, :math:`\chi_{2,y}`. If ``use_prec_ang=True`` this is interpreted as the difference(s) in azimuthal angle between spin vectors, :math:`\phi_{1,2}`, in :math:`\\rm rad`.
        :param array or float LambdaTilde: The adimensional tidal deformability(ies) of combination :math:`\\tilde{\Lambda}`.
        :param array or float deltaLambda: The adimensional tidal deformability(ies) of combination :math:`\delta\\tilde{\Lambda}`.
        :param array or float ecc: The orbital eccentricity(ies), :math:`e_0`.
        :param float rot: Further rotation of the interferometer with respect to the :py:data:`self.xax` orientation, in degrees, needed for the triangular geometry.
        :param bool, optional use_m1m2: Boolean specifying if the ``Mc`` and ``eta`` inputs should be interpreted as the primary and secondary mass(es).
        :param bool, optional use_chi1chi2: Boolean specifying if the ``chiS`` and ``chiA`` inputs should be interpreted as the primary and secondary spin components along the axis :math:`z`.
        :param bool, optional use_prec_ang: Boolean specifying if the ``iota`` input should be interpreted as the inclination angle with respect to total angular momentum, ``chiS`` and ``chiA`` as the primary and secondary spin magnitudes, ``chi1x`` and ``chi2x`` as the primary and secondary spin tilts, ``chi1y`` as the azimuthal angle of orbital angular momentum relative to total angular momentum and ``chi2y`` as the difference in azimuthal angle between spin vectors.
        :return: Analytical derivatives with respect to ``dL``, ``theta``, ``phi``, ``iota``, ``psi``, ``tcoal`` and ``phase``. If the :py:class:`self.wf_model` is precessing or includes higher order modes the derivative with respect to ``iota`` will be ``None``
        :rtype: tuple(array, array, array, array, array, array, array)

        """
        ZEROS = np.zeros_like(Mc)
        # Module to compute analytically the derivatives w.r.t. dL, theta, phi, psi, tcoal, phase and also iota in absence of HM or precessing spins. Each derivative is inserted into its own function with representative name, for ease of check.
        if use_m1m2:
            # Interpret Mc as m1 and eta as m2
            McUse, etaUse = utils.Mceta_from_m1m2(Mc, eta)
        else:
            McUse = Mc
            etaUse = eta

        if not self.wf_model.is_Precessing:
            if use_chi1chi2:
                # Interpret chiS as chi1z and chiA as chi2z
                chi1z = chiS
                chi2z = chiA
            else:
                chi1z = chiS + chiA
                chi2z = chiS - chiA
            chi1xUse = ZEROS
            chi2xUse = ZEROS
            chi1yUse = ZEROS
            chi2yUse = ZEROS
        else:
            if not use_prec_ang:
                chi1z = chiS
                chi2z = chiA
                chi1xUse = chi1x
                chi2xUse = chi2x
                chi1yUse = chi1y
                chi2yUse = chi2y
            else:
                # convert angles and iota
                iota, chi1xUse, chi1yUse, chi1z, chi2xUse, chi2yUse, chi2z = (
                    utils.TransformPrecessing_angles2comp(
                        thetaJN=iota,
                        phiJL=chi1y,
                        tilt1=chi1x,
                        tilt2=chi2x,
                        phi12=chi2y,
                        chi1=chiS,
                        chi2=chiA,
                        Mc=McUse,
                        eta=etaUse,
                        fRef=self.fmin,
                        phiRef=0.0,
                    )
                )

        # if use_lensing:
        #     # computeAnalyticalDeriv is always left as False by default in lensing tests
        #     # Lensing modification ignored for now
        #     alpha_hat = _get_alpha_hat(R_orbit)
        #     iotaUse = get_image_iota(iota, phi_L, alpha_hat)[0]
        #     phase = get_image_phase(iota, phi_L, phase, alpha_hat)[0]
        #     psi = get_image_psi(iota, phi_L, psi, alpha_hat)[0]
        #     cos_phi_proj = get_cos_phi_proj(iota, phi_L)

        #     # doppler effect is treated as a change in the effective chirp mass
        #     z = np.interp(
        #         np.real(dL).astype("float64"), utils.dLGridGlob, utils.zGridGlob
        #     )  # z_at_value(Planck18.luminosity_distance, dL * u.Mpc)
        #     delta_z = get_delta_z(R_orbit, cos_phi_proj)[0]
        #     # need to double check if this formula is correct
        #     McUse *= ((1 + z) / (1 + z + delta_z)) ** (8 / 5)
        #     # eta will also change!!!
        #     iota = iotaUse

        evParams = {
            "Mc": McUse,
            "dL": dL,
            "theta": theta,
            "phi": phi,
            "iota": iota,
            "psi": psi,
            "tcoal": tcoal,
            "eta": etaUse,
            "phase": phase,
            "chi1z": chi1z,
            "chi2z": chi2z,
            "chi1x": chi1xUse,
            "chi2x": chi2xUse,
            "chi1y": chi1yUse,
            "chi2y": chi2yUse,
        }

        if self.wf_model.is_tidal:
            evParams["Lambda1"], evParams["Lambda2"] = utils.Lam12_from_Lamt_delLam(
                LambdaTilde, deltaLambda, etaUse
            )

        if self.wf_model.is_eccentric:
            evParams["ecc"] = ecc

        if (not self.wf_model.is_HigherModes) and (not self.wf_model.is_Precessing):
            wfPhiGw = self.wf_model.Phi(f, **evParams)
            wfAmpl = self.wf_model.Ampl(f, **evParams)
            wfhpc = wfAmpl * np.exp(-1j * wfPhiGw)
            wfhp = wfhpc * 0.5 * (1.0 + np.cos(iota) ** 2)
            wfhc = 1j * wfhpc * np.cos(iota)
        else:
            # If the waveform includes higher modes, it is not possible to compute amplitude and phase separately, make all together
            wfhp, wfhc = self.wf_model.hphc(f, **evParams)

        phiD = np.zeros_like(Mc)
        t, tmpDeltLoc = self.shifted_time(evParams, f)
        phiL = (TWOPI * f) * tmpDeltLoc

        rot_rad = rot * DEG_TO_RAD
        sin_angbtwArms = np.sin(self.angbtwArms)

        ras, decs = ra_dec_from_th_phi_rad(theta, phi)
        Fpc = self.detector.compute_antenna_pattern(theta, phi, t, psi, rot)

        omega = TWOPI * f * DAY_TO_SEC
        phase = 1j * (omega * tcoal - phase + phiD + phiL)
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

        return (
            -(hp + hc) / dL,
            theta_par_deriv(),
            phi_par_deriv(),
            iota_par_deriv(),
            psi_par_deriv(),
            tcoal_par_deriv(),
            -1j * (hp + hc),
        )

    def optimal_location(self, tcoal, is_tGPS=False):
        """
        Compute the optimal sky position for a signal to be seen by the detector at a given time.

        The computation assumes :math:`\psi = 0`.

        :param float tcoal: The time at which to compute the optimal location, as GMST in days.
        :param bool, optional is_tGPS: Boolean specifying if the provided time is a GPS time (in seconds) rather than a GMST.

        :return: Optimal :math:`\\theta` and :math:`\phi` sky coordinates, in :math:`\\rm rad`.
        :rtype: array(float, float)

        """
        # Function to compute the optimal theta and phi for a signal to be seen by the detector at a given GMST. The boolean is_tGPS can be used to specify whether the provided time is a GPS time rather than a GMST, so that it will be converted.
        # For a triangle the best location is the same of an L in the same place, as can be shown by explicit geometrical computation.
        # Even if considering Earth rotation, the highest SNR will still be obtained if the source is in the optimal location close to the merger.

        if is_tGPS:
            tc = utils.GPSt_to_LMST(tcoal, lat=0.0, long=0.0)
        else:
            tc = tcoal

        def pattern_fixedtpsi(pars, tc=tc):
            Fp, Fc = self.detector.compute_antenna_pattern(*pars, t=tc, psi=0)
            return -np.sqrt(Fp**2 + Fc**2)

        # we actually minimize the pattern function times -1, which is the same as maximizing it
        return minimize(
            pattern_fixedtpsi, [1.0, 1.0], bounds=((0.0, onp.pi), (0.0, 2.0 * onp.pi))
        ).x

    def SNRFastInsp(self, evParams, checkInterp=False):
        """
        Compute the inspiral SNR taking into account Earth rotation, without the need of performing an integral for each event

        .. deprecated:: 1.0.0
            Use the standard function :py:class:`GWSignal.SNRInteg`.
        """
        # This module allows to compute the inspiral SNR taking into account Earth rotation, without the need
        # of performing an integral for each event

        Mc, dL, theta, phi, iota, psi, tcoal, eta = (
            evParams["Mc"],
            evParams["dL"],
            evParams["theta"],
            evParams["phi"],
            evParams["iota"],
            evParams["psi"],
            evParams["tcoal"],
            evParams["eta"],
        )

        ras, decs = ra_dec_from_th_phi_rad(theta, phi)

        if not np.isscalar(Mc):
            SNR = np.zeros(Mc.shape)
        else:
            SNR = 0

        # Factor in front of the integral in the inspiral only case
        fac = (
            np.sqrt(5.0 / 6.0)
            / np.pi ** (2.0 / 3.0)
            * (glob.GMsun_over_c3 * Mc) ** (5.0 / 6.0)
            * glob.clightGpc
            / dL
        )  # *np.exp(-logdL)

        fcut = self.wf_model.fcut(**evParams)
        if self.fmax is not None:
            fcut = np.where(fcut > self.fmax, self.fmax, fcut)
        mask = self.detector.psd_frequencies >= self.fmin
        masked_freqs = self.detector.psd_frequencies[mask]

        if not self.useEarthMotion:
            t = tcoal - self.wf_model.tau_star(self.fmin, **evParams) / DAY_TO_SEC
            if self.detector.shape == "L":
                Fp, Fc = self.detector.compute_antenna_pattern(
                    theta, phi, t, psi, rot=0.0
                )
                Qsq = (Fp * 0.5 * (1.0 + (np.cos(iota)) ** 2)) ** 2 + (
                    Fc * np.cos(iota)
                ) ** 2
                SNR = fac * np.sqrt(
                    Qsq
                    * onp.interp(
                        fcut,
                        masked_freqs,
                        self.strainInteg,
                        left=1.0,
                        right=1.0,
                    )
                )
            elif self.detector.shape == "T":
                for i in range(3):
                    Fp, Fc = self.detector.compute_antenna_pattern(
                        theta, phi, t, psi, rot=60.0 * i
                    )
                    Qsq = (Fp * 0.5 * (1.0 + np.cos(iota) ** 2)) ** 2 + (
                        Fc * np.cos(iota)
                    ) ** 2
                    tmpSNR = fac * np.sqrt(
                        Qsq
                        * onp.interp(
                            fcut,
                            masked_freqs,
                            self.strainInteg,
                            left=1.0,
                            right=1.0,
                        )
                    )
                    SNR = SNR + tmpSNR * tmpSNR
                SNR = np.sqrt(SNR)
            return SNR

        if self.IntegInterpArr is None:
            self._make_SNRig_interpolator()
        Igs = onp.zeros((9, len(Mc)))
        if not checkInterp:
            for i in range(9):
                Igs[i, :] = self.IntegInterpArr[i](onp.array([Mc, eta, tcoal]).T)
        else:
            fminarr = np.full(fcut.shape, self.fmin)
            fgrids = np.geomspace(fminarr, fcut, num=int(5000))
            strainGrids = self.detector.psd_interp(fgrids)

            for m in range(4):
                tmpIntegrandC = CosineIntegrand(fgrids, Mc, tcoal, m + 1.0)
                tmpIntegrandS = SineIntegrand(fgrids, Mc, tcoal, m + 1.0)
                Igs[m, :] = onp.trapz(tmpIntegrandC / strainGrids, fgrids, axis=0)
                Igs[m + 4, :] = onp.trapz(tmpIntegrandS / strainGrids, fgrids, axis=0)
            tmpIntegrand = CosineIntegrand(fgrids, Mc, tcoal, 0.0)
            Igs[8, :] = onp.trapz(tmpIntegrand / strainGrids, fgrids, axis=0)

        if self.detector.shape == "L":
            C2s, S2s, C1s, S1s, C0s = self.detector.CoeffsRot(ras, decs, psi, rot=0.0)
            FpsqInt, FcsqInt = FpFcsqInt(C2s, S2s, C1s, S1s, C0s, Igs, iota)
            QsqInt = FpsqInt + FcsqInt
            SNR = fac * np.sqrt(QsqInt)
        elif self.detector.shape == "T":
            snr_sq = 0.0
            for i in range(3):
                C2s, S2s, C1s, S1s, C0s = self.detector.CoeffsRot(
                    ras, decs, psi, rot=i * 60.0
                )
                FpsqInt, FcsqInt = FpFcsqInt(C2s, S2s, C1s, S1s, C0s, Igs, iota)
                QsqInt = FpsqInt + FcsqInt
                snr_sq += fac * fac * QsqInt
            SNR = np.sqrt(snr_sq)
        return SNR

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

        if WF1.is_Precessing:
            try:
                _ = evParams1["chi1x"]
            except KeyError:
                try:
                    print(
                        "Adding cartesian components of the spins from angular variables"
                    )
                    (
                        evParams1["iota"],
                        evParams1["chi1x"],
                        evParams1["chi1y"],
                        evParams1["chi1z"],
                        evParams1["chi2x"],
                        evParams1["chi2y"],
                        evParams1["chi2z"],
                    ) = utils.TransformPrecessing_angles2comp(
                        thetaJN=evParams1["thetaJN"],
                        phiJL=evParams1["phiJL"],
                        tilt1=evParams1["tilt1"],
                        tilt2=evParams1["tilt2"],
                        phi12=evParams1["phi12"],
                        chi1=evParams1["chi1"],
                        chi2=evParams1["chi2"],
                        Mc=evParams1["Mc"],
                        eta=evParams1["eta"],
                        fRef=self.fmin,
                        phiRef=0.0,
                    )
                except KeyError:
                    raise ValueError(
                        "Either the cartesian components of the precessing spins (iota, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z) or their modulus and orientations (thetaJN, chi1, chi2, tilt1, tilt2, phiJL, phi12) have to be provided."
                    )
        else:
            try:
                _ = evParams1["chi1z"]
                (
                    evParams1["chi1x"],
                    evParams1["chi1y"],
                    evParams1["chi2x"],
                    evParams1["chi2y"],
                ) = (
                    np.zeros_like(evParams1["Mc"]),
                    np.zeros_like(evParams1["Mc"]),
                    np.zeros_like(evParams1["Mc"]),
                    np.zeros_like(evParams1["Mc"]),
                )
            except KeyError:
                try:
                    print("Adding chi1z, chi2z from chiS, chiA")
                    evParams1["chi1z"] = evParams1["chiS"] + evParams1["chiA"]
                    evParams1["chi2z"] = evParams1["chiS"] - evParams1["chiA"]
                except KeyError:
                    raise ValueError(
                        "Two among chi1z, chi2z and chiS, chiA have to be provided."
                    )

        if WF1.is_tidal:
            try:
                _ = evParams1["LambdaTilde"]
            except KeyError:
                try:
                    evParams1["LambdaTilde"], evParams1["deltaLambda"] = (
                        utils.Lamt_delLam_from_Lam12(
                            evParams1["Lambda1"], evParams1["Lambda2"], evParams1["eta"]
                        )
                    )
                except KeyError:
                    raise ValueError(
                        "Two among Lambda1, Lambda2 and LambdaTilde and deltaLambda have to be provided."
                    )
        else:
            evParams1["LambdaTilde"], evParams1["deltaLambda"] = np.zeros_like(
                evParams1["Mc"]
            ), np.zeros_like(evParams1["Mc"])

        if not WF1.is_eccentric:
            evParams1["ecc"] = np.zeros_like(evParams1["Mc"])

        # Checks on imput parameters for waveform 2

        if WF2.is_Precessing:
            try:
                _ = evParams2["chi1x"]
            except KeyError:
                try:
                    print(
                        "Adding cartesian components of the spins from angular variables"
                    )
                    (
                        evParams2["iota"],
                        evParams2["chi1x"],
                        evParams2["chi1y"],
                        evParams2["chi1z"],
                        evParams2["chi2x"],
                        evParams2["chi2y"],
                        evParams2["chi2z"],
                    ) = utils.TransformPrecessing_angles2comp(
                        thetaJN=evParams2["thetaJN"],
                        phiJL=evParams2["phiJL"],
                        tilt1=evParams2["tilt1"],
                        tilt2=evParams2["tilt2"],
                        phi12=evParams2["phi12"],
                        chi1=evParams2["chi1"],
                        chi2=evParams2["chi2"],
                        Mc=evParams2["Mc"],
                        eta=evParams2["eta"],
                        fRef=self.fmin,
                        phiRef=0.0,
                    )
                except KeyError:
                    raise ValueError(
                        "Either the cartesian components of the precessing spins (iota, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z) or their modulus and orientations (thetaJN, chi1, chi2, tilt1, tilt2, phiJL, phi12) have to be provided."
                    )
        else:
            try:
                _ = evParams2["chi1z"]
                (
                    evParams2["chi1x"],
                    evParams2["chi1y"],
                    evParams2["chi2x"],
                    evParams2["chi2y"],
                ) = (
                    np.zeros_like(evParams2["Mc"]),
                    np.zeros_like(evParams2["Mc"]),
                    np.zeros_like(evParams2["Mc"]),
                    np.zeros_like(evParams2["Mc"]),
                )
            except KeyError:
                try:
                    print("Adding chi1z, chi2z from chiS, chiA")
                    evParams2["chi1z"] = evParams2["chiS"] + evParams2["chiA"]
                    evParams2["chi2z"] = evParams2["chiS"] - evParams2["chiA"]
                except KeyError:
                    raise ValueError(
                        "Two among chi1z, chi2z and chiS, chiA have to be provided."
                    )

        if WF2.is_tidal:
            try:
                _ = evParams2["LambdaTilde"]
            except KeyError:
                try:
                    evParams2["LambdaTilde"], evParams2["deltaLambda"] = (
                        utils.Lamt_delLam_from_Lam12(
                            evParams2["Lambda1"], evParams2["Lambda2"], evParams2["eta"]
                        )
                    )
                except KeyError:
                    raise ValueError(
                        "Two among Lambda1, Lambda2 and LambdaTilde and deltaLambda have to be provided."
                    )
        else:
            evParams2["LambdaTilde"], evParams2["deltaLambda"] = np.zeros_like(
                evParams2["Mc"]
            ), np.zeros_like(evParams2["Mc"])

        if not WF2.is_eccentric:
            evParams2["ecc"] = np.zeros_like(evParams2["Mc"])

        # The frequency cut is chosen to be the highest among the two
        fcut1 = WF1.fcut(**evParams1)
        fcut2 = WF2.fcut(**evParams2)

        fcutUse = np.where(fcut1 > fcut2, fcut1, fcut2)

        if self.fmax is not None:
            fcutUse = np.where(fcutUse > self.fmax, self.fmax, fcut1)
        fminarr = np.full(fcutUse.shape, self.fmin)

        fgrids = np.geomspace(fminarr, fcutUse, num=int(res))
        psd_strain_grids = self.detector.psd_interp(fgrids)

        # This is a horrible way of changing the waveform, but the fastest to implement
        WFor = copy.deepcopy(self.wf_model)

        strains = []
        SNRhs = []
        if self.detector.shape == "L":
            for model, params in zip((WF1, WF2), (evParams1, evParams2)):
                self.wf_model = model
                strain = self.GWstrain(
                    fgrids,
                    params["Mc"],
                    params["eta"],
                    params["dL"],
                    params["theta"],
                    params["phi"],
                    params["iota"],
                    params["psi"],
                    params["tcoal"],
                    params["phase"],
                    params["chi1z"],
                    params["chi2z"],
                    params["chi1x"],
                    params["chi2x"],
                    params["chi1y"],
                    params["chi2y"],
                    params["LambdaTilde"],
                    params["deltaLambda"],
                    params["ecc"],
                    is_chi1chi2=True,
                )

                strains.append(strain)
                SNRhs.append(optimal_snr(fgrids, strain, psd_strain_grids))

            overlap_int = noise_weighted_inner_product(
                fgrids, *strains, psd_strain_grids
            )

        elif self.detector.shape == "T":
            for model, params in zip((WF1, WF2), (evParams1, evParams2)):
                self.wf_model = model
                h_1 = self.GWstrain(
                    fgrids,
                    params["Mc"],
                    params["eta"],
                    params["dL"],
                    params["theta"],
                    params["phi"],
                    params["iota"],
                    params["psi"],
                    params["tcoal"],
                    params["phase"],
                    params["chi1z"],
                    params["chi2z"],
                    params["chi1x"],
                    params["chi2x"],
                    params["chi1y"],
                    params["chi2y"],
                    params["LambdaTilde"],
                    params["deltaLambda"],
                    params["ecc"],
                    is_chi1chi2=True,
                    rot=0.0,
                )
                h_2 = self.GWstrain(
                    fgrids,
                    params["Mc"],
                    params["eta"],
                    params["dL"],
                    params["theta"],
                    params["phi"],
                    params["iota"],
                    params["psi"],
                    params["tcoal"],
                    params["phase"],
                    params["chi1z"],
                    params["chi2z"],
                    params["chi1x"],
                    params["chi2x"],
                    params["chi1y"],
                    params["chi2y"],
                    params["LambdaTilde"],
                    params["deltaLambda"],
                    params["ecc"],
                    is_chi1chi2=True,
                    rot=60.0,
                )
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
