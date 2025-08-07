#
#    Copyright (c) 2025 Samson Leong <samson.leong@link.cuhk.edu.hk>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.

from jax import config
# Enable 64bit on JAX, fundamental
config.update("jax_enable_x64", True)

from gwfast.gwfastUtils import get_model_parameters
from gwfast.lensing_utils import get_agn_lensed_parameters
from gwfast.signals import BasicGWSignal, GeneralLensedGWSignal


class AGNLensedGWSignal(GeneralLensedGWSignal):
    """
    This class is built on top of the :py:class:`GeneralLensedGWSignal` class, and is specifically designed for
    lensed GW signals from BBH that are embeded within the AGN accretion disk, with the assumption that the binary's orbital angular momentum is aligned with that of the disk.

    Other than the standard binary parameters, this class adds the following parameters:
    * ``R_orbit``: orbital radius of the binary, in :math:`\\rm R_{\\odot}`
    * ``M_lz``: mass of the AGN, in :math:`\\rm M_{\\odot}`
    * ``src_pos``: position of the source in the accretion disk, in :math:`\\rm R_{\\odot}`

    Other than the additional parameters, the rest of this class behaves like a standard GW signal class:

    """

    __doc__ += BasicGWSignal.__doc__

    def __init__(self, **kwargs):

        additional_params = {
            'R_orbit': 100,
            'M_lz': 1e6,
            'src_pos': 0.1
        }
        # Use the base class constructor
        super(GeneralLensedGWSignal, self).__init__(**kwargs, init_params=additional_params)
        self.additional_params = additional_params
        self.strain_model_keys = list(
            self.wf_model.ParNums.keys() | self.additional_params.keys()
        )

    def GWAmplitudes(self, evParams, f, rot=0.0):
        raise NotImplementedError("Yeah, someone should work on this.")

    def GWPhase(self, evParams, f):
        raise NotImplementedError("Yeah, someone should work on this.")

    def GWstrain(self, freqs, parameters, rot=0.0, return_single_comp=None):
        ref_freqs = self.get_reference_frequency(freqs)
        model_parameters = self.convert_to_general_lensed_parameters(parameters, ref_freqs)

        return super().GWstrain(
            freqs, model_parameters, rot=rot, return_single_comp=return_single_comp)

    def _analytical_derivatives(self):
        raise NotImplementedError('Lensed waveforms have no well-defined analytical derivatives (yet)')

    def convert_to_general_lensed_parameters(self, agn_lensed_params, reference_frequency=None):
        """
        Convert the AGN lensed parameters to parameters of the flexible model.

        Parameters
        ----------
        agn_lensed_params : dict
            Dictionary containing the AGN lensed parameters.
        """
        model_params = get_model_parameters(agn_lensed_params, self.strain_model_keys, reference_frequency)
        params_1, params_2 = get_agn_lensed_parameters(model_params)

        output_params = params_1.copy()
        output_params.update({
            'iota': params_1['iota'],
            'delta_iota': params_2['iota'] - params_1['iota'],
            'phase': params_1['phase'],
            'delta_phase': params_2['phase'] - params_1['phase'],
            'psi': params_1['psi'],
            'delta_psi': params_2['psi'] - params_1['psi'],
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
