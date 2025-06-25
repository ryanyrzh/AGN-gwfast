#
#    Copyright (c) 2025 Samson Leong <samson.leong@link.cuhk.edu.hk>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.

from jax import config
import jax.numpy as np
# Enable 64bit on JAX, fundamental
config.update("jax_enable_x64", True)

import logging

from gwfast.gwfastGlobals import TWOPI, DAY_TO_SEC
from gwfast.gwfastUtils import get_model_parameters, chirp_time_bound
from gwfast.signals import BasicGWSignal


class GeneralLensedGWSignal(BasicGWSignal):
    """
    Class to compute the lensed GW signal emitted by a coalescing binary system as seen by a detector on Earth.
    This assumes the point-mass lens model, splitting the GW signal into two, each with a phenomenological change
    in (effective) luminosity distance, coalescese time, inclination, phase and polarisation angle.
    On top of that, a phenomenological redshift is also allowed to apply to the signal (in particular, the chirp mass).
    This could due to the orbital motion of the source around some massive object.

    There are 5 additional parameters that are added to the usual BBH waveform:
    * ``delta_iota``: change in inclination angle, in radians
    * ``delta_phase``: change in phase, in radians
    * ``delta_time``: change in coalescence time, in seconds
    * ``relative_distance``: change in luminosity distance, dimensionless
    * ``relative_mass``: change in chirp mass, dimensionless

    Alternatively, the parameters can be given in the form of:
    * ``iota_1``, ``iota_2``: two different inclination angles, in radians
    * ``phase_1``, ``phase_2``: two different phases, in radians
    * ``tGPS_1``, ``tGPS_2``: two different GPS times, in seconds
    * ``tcoal_1``, ``tcoal_2``: two different coalescence times, in seconds
    * ``dL_1``, ``dL_2``: two different luminosity distances, in meters
    * ``Mc_1``, ``Mc_2``: two different chirp masses, in solar masses

    Other than the additional parameters, the rest of this class behaves like a standard GW signal class:
    """

    __doc__ += BasicGWSignal.__doc__

    def __init__(self, **kwargs):

        additional_params = {
            'delta_iota': 0.5,   # radian
            'delta_phase': 0.5,  # radian
            'delta_time': 1.0,   # seconds
            'relative_distance': 1.0,   # dimensionless
            'relative_mass': 1.0,   # dimensionless
        }
        super().__init__(**kwargs, init_params=additional_params)
        self.additional_params = additional_params
        self.strain_model_keys = list(
            self.wf_model.ParNums.keys() | self.additional_params.keys()
        )

    def GWAmplitudes(self, evParams, f, rot=0.0):
        raise NotImplementedError("Yeah, someone should work on this.")

    def GWPhase(self, evParams, f):
        raise NotImplementedError("Yeah, someone should work on this.")

    def GWstrain(self, freqs, parameters, rot=0.0, return_single_comp=None):
        omega = TWOPI * freqs * DAY_TO_SEC

        # We need `chi1,2` for duration checks
        request_model_keys = self.strain_model_keys + ['chi1', 'chi2']
        ref_freqs = self.get_reference_frequency(freqs)
        signal_1_params, signal_2_params = self.get_parameter_sets(parameters)
        signal_1_params = get_model_parameters(signal_1_params, request_model_keys, ref_freqs)
        signal_2_params = get_model_parameters(signal_2_params, request_model_keys, ref_freqs)

        self.check_total_duration(freqs, signal_1_params, signal_2_params)

        # Not sure what does this do, but it was set to zero in both cases
        # (with or without useEarthMotion)
        phiD = np.zeros_like(signal_1_params["Mc"])

        # Moving on to combining the strain with the antenna patterns
        is_lal = self.wf_model.is_LAL

        hpc_12 = []
        for params in (signal_1_params, signal_2_params):
            # 22 mode waveforms
            if not (self.need_HM or is_lal):
                _, deltaT = self.shifted_time(params, freqs)
                phiL = omega * deltaT

                Ap, Ac = super().GWAmplitudes(params, freqs, rot=rot)
                Psi = super().GWPhase(params, freqs)
                Psi += phiD + phiL

                hp, hc = Ap * np.exp(Psi * 1j), 1j * Ac * np.exp(Psi * 1j)
                hpc_12.append((hp, hc))
            else:
                iota = params["iota"]
                psi = params["psi"]
                phase = params["phase"]
                theta = params["theta"]
                phi = params["phi"]

                phase_shift_factor = np.exp(1j * (phiD + omega * params["tcoal"]))
                time, deltaT = self.shifted_time(params, freqs)
                phiL = omega * deltaT

                Fpc = self.detector.compute_antenna_pattern(theta, phi, time, psi, rot=rot)
                hpc = self.wf_model.hphc(freqs, **params)
                phase_factor = phase_shift_factor * np.exp(1j * (phiL - phase))
                hp = hpc[0] * Fpc[0] * phase_factor
                hc = hpc[1] * Fpc[1] * phase_factor

                if is_lal:
                    hp *= 0.5 * (1.0 + np.cos(iota) ** 2)
                    hc *= np.cos(iota)

                hpc_12.append((hp, hc))

        ((hp1, hc1), (hp2, hc2)) = hpc_12

        hp = hp1 + hp2
        hc = hc1 + hc2

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
                return hp1 + hc1, hp2 + hc2
            else:
                raise ValueError(
                    "Single component to return has to be among Ap, Ac, Psip, Psic"
                )
        else:
            return hp + hc

    @staticmethod
    def check_total_duration(frequencies, params_1, params_2):
        """Check the rough total duration of signal is within the frequency resolution.
        """
        f_min = frequencies[0]
        print(frequencies)
        T_max = 1 / np.min(np.diff(frequencies, axis=0), axis=0)

        chirp_time_1 = chirp_time_bound(
                f_min, params_1['Mc'], params_1['eta'], params_1['chi1'], params_1['chi2'])
        chirp_time_2 = chirp_time_bound(
                f_min, params_2['Mc'], params_2['eta'], params_2['chi1'], params_2['chi2'])
        long_chirp = np.maximum(chirp_time_1, chirp_time_2)

        delta_t = np.abs(params_1['tcoal'] - params_2['tcoal']) * DAY_TO_SEC

        if np.any((long_chirp + delta_t) > T_max):
            logging.warning(
                    'Some of the input parameters will likely yield waveforms with signal length longer than the maximum duration resolved by the frequencies.'
                    )

    def _analytical_derivatives(self):
        raise NotImplementedError('Lensed waveforms have no well-defined analytical derivatives (yet)')

    @staticmethod
    def _get_parameter_pairs(key, reference_parameters, mode):
        alternative_keys = {'dL': 'distance', 'Mc': 'mass', 'tGPS': 'time', 'tcoal': 'time'}

        param_1 = reference_parameters.pop(f"{key}_1", None)
        if param_1 is None:
            param_1 = reference_parameters.pop(key, None)
        if param_1 is None:
            return None, None

        param_2 = reference_parameters.pop(f"{key}_2", None)
        if param_2 is None:
            key = alternative_keys.get(key, key)
            variation = reference_parameters.pop(f"{mode}_{key}", None)
            if variation is None:
                return param_1, None
            if mode == 'relative':
                param_2 = param_1 * variation
            elif mode == 'delta':
                param_2 = param_1 + variation

        return param_1, param_2

    def get_parameter_sets(self, parameters_dict):
        """
        Generate two sets of parameters from the provided dictionary.
        The two sets of parameters differ in:
            * iota, phase, time, luminosity distance and chirp mass

        It is assumed that they are always given in either of the two forms:
            * param_1, param_2
            * param, delta_param / relative_param

        `delta` or `relative` depends on the parameter itself,
            * iota, phase, and time are `delta`
            * distance and chirp mass are `relative`
        """
        ref_parameters_dict = parameters_dict.copy()
        iota_1, iota_2 = self._get_parameter_pairs('iota', ref_parameters_dict, 'delta')
        phase_1, phase_2 = self._get_parameter_pairs('phase', ref_parameters_dict, 'delta')
        # Caution: tGPS is in unit of seconds, tcoal is days
        tGPS_1, tGPS_2 = self._get_parameter_pairs('tGPS', ref_parameters_dict, 'delta')
        tcoal_1, tcoal_2 = self._get_parameter_pairs('tcoal', ref_parameters_dict, 'delta')
        distance_1, distance_2 = self._get_parameter_pairs('dL', ref_parameters_dict, 'relative')
        mass_1, mass_2 = self._get_parameter_pairs('Mc', ref_parameters_dict, 'relative')

        signal_1_params = ref_parameters_dict.copy()
        signal_2_params = ref_parameters_dict.copy()

        signal_1_params.update({
            "iota": iota_1,
            "phase": phase_1,
            "dL": distance_1,
            "Mc": mass_1,
        })
        signal_2_params.update({
            "iota": iota_2,
            "phase": phase_2,
            "dL": distance_2,
            "Mc": mass_2,
        })

        if not (tcoal_1 is None or tcoal_2 is None):
            signal_1_params["tcoal"] = tcoal_1
            signal_2_params["tcoal"] = tcoal_2
        elif not (tGPS_1 is None or tGPS_2 is None):
            signal_1_params["tGPS"] = tGPS_1
            signal_2_params["tGPS"] = tGPS_2

        return signal_1_params, signal_2_params
