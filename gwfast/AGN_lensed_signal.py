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

from gwfast.gwfastGlobals import DAY_TO_SEC
from gwfast.gwfastUtils import get_model_parameters
from gwfast.lensing_utils_alt import get_agn_lensed_parameters
from gwfast.flexible_lensed_signal import FlexibleLensedGWSignal


class AGNLensedGWSignal(FlexibleLensedGWSignal):
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

        additional_params = {
            'R_orbit': 100,
            'M_lz': 1e6,
            'src_pos': 0.1
        }
        # Use the base class constructor
        super(FlexibleLensedGWSignal, self).__init__(**kwargs, init_params=additional_params)
        self.additional_params = additional_params
        self.strain_model_keys = list(
            self.wf_model.ParNums.keys() | self.additional_params.keys()
        )

    def GWAmplitudes(self, evParams, f, rot=0.0):
        raise NotImplementedError("Yeah, someone should work on this.")

    def GWPhase(self, evParams, f):
        raise NotImplementedError("Yeah, someone should work on this.")

    def GWstrain(self, freqs, parameters, rot=0.0, return_single_comp=None):
        """
        Compute the full GW strain (complex) as a function of the parameters, at given frequencies.

        :param array or float f: The frequency(ies) at which to perform the calculation, in :math:`\\rm Hz`.
        :param dict parameters: The parameters dictionary to evaluate the strain at.
        :param float rot: Further rotation of the interferometer with respect to the :py:data:`self.xax` orientation, in degrees, needed for the triangular geometry.
        :param str return_single_comp: String specifying if a single component of the signal should be returned, to be chosen among ``Ap`` and ``Ac``, to return the plus and cross amplitude, :math:`A_+` and :math:`A_{\\times}`, respectively, and ``Psip`` and ``Psic``, to return the plus and cross phase, :math:`\Phi_+` and :math:`\Phi_{\\times}`, respectively.
        :return: Complete signal strain (complex), evaluated at the given parameters and frequency(ies).
        :rtype: array or float

        """
        model_parameters = self.convert_to_flexible_model_parameters(parameters)

        return super().GWstrain(
            freqs, model_parameters, rot=rot, return_single_comp=return_single_comp)

    def _analytical_derivatives(self):
        raise NotImplementedError('Lensed waveforms have no well-defined analytical derivatives (yet)')

    def convert_to_flexible_model_parameters(self, agn_lensed_params):
        """
        Convert the AGN lensed parameters to parameters of the flexible model.

        Parameters
        ----------
        agn_lensed_params : dict
            Dictionary containing the AGN lensed parameters.
        """
        model_params = get_model_parameters(agn_lensed_params, self.strain_model_keys)
        params_1, params_2 = get_agn_lensed_parameters(model_params)

        output_params = params_1.copy()
        output_params.update({
            'iota': params_1['iota'],
            'delta_iota': params_2['iota'] - params_1['iota'],
            'phase': params_1['phase'],
            'delta_phase': params_2['phase'] - params_1['phase'],
            'dL': params_1['dL'],
            'relative_distance': params_2['dL'] / params_1['dL'],
            'Mc': params_1['Mc'],
            'relative_mass': params_2['Mc'] / params_1['Mc'],
        })

        tcoal = params_1.get('tcoal', None)
        if tcoal is not None:
            output_params['tcoal'] = tcoal
            output_params['delta_time'] = params_2['tcoal'] - tcoal
        else:
            tGPS = params_1.get('tGPS', None)
            output_params['tGPS'] = tGPS
            output_params['delta_time'] = params_2['tGPS'] - tGPS

        return output_params