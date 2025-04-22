from pathlib import Path

from jax import config
import jax.numpy as np

# Enable 64bit on JAX, fundamental
config.update("jax_enable_x64", True)

import numpy as onp
from gwfast.gwfastGlobals import TWOPI, DAY_TO_SEC, DEG_TO_RAD, clight, REarth
from gwfast.gwfastUtils import (
    ra_dec_from_th_phi_rad,
    apply_psi_rotation,
)


class Detector(object):
    def __init__(self, name, lat, lon, xax, shape, duty_cycle=None, noise_curve_path=None, verbose=False):
        self.name = name
        self.verbose = verbose

        if shape == "L":
            self.ang_btw_arms = 0.5 * np.pi
        elif shape == "T":
            self.ang_btw_arms = np.pi / 3.0
        else:
            raise ValueError("Enter valid detector configuration")
        self.shape = shape
        # This is the percentage of time each arm of the detector (or the whole detector for an L) 
        # is supposed to be operational, between 0 and 1, default is None, that is always online
        self.duty_cycle = duty_cycle

        self.lat = lat
        self.lon = lon
        self.xax = xax
        self.lat_rad = lat * DEG_TO_RAD
        self.lon_rad = lon * DEG_TO_RAD
        self.xax_rad = xax * DEG_TO_RAD

        self.load_noise_curve(noise_curve_path, self.verbose)

    def load_noise_curve(self, file_path, verbose=False):
        if file_path is None:
            raise ValueError("Enter a valid PSD or ASD path")
        
        _noise_curve_path = Path(file_path)
        self.noise_basedir = _noise_curve_path.parent
        self.noise_filename = _noise_curve_path.name

        self.psd_frequencies, spectral_density = \
            onp.loadtxt(file_path, usecols=(0, 1), unpack=True)
        
        # Load PSD or ASD based on the amplitudes.
        if min(spectral_density) > 1e-30:
            if verbose:
                print("Using ASD from file %s " % file_path)
            self.asd_array = spectral_density
            self.psd_array = spectral_density ** 2
        else:
            if verbose:
                print("Using PSD from file %s " % file_path)
            self.asd_array = np.sqrt(spectral_density)
            self.psd_array = spectral_density

        return 0

    def compute_antenna_pattern(self, theta, phi, t, psi, rot=0.0):
        """
        Compute the value of the so-called pattern functions of the detector for a set of sky coordinates, GW polarisation(s) and time(s).

        For the definition of the pattern functions see `arXiv:gr-qc/9804014 <https://arxiv.org/abs/gr-qc/9804014>`_ eq. (10)--(13).

        :param array or float theta: The :math:`\\theta` sky position angle(s), in :math:`\\rm rad`.
        :param array or float phi: The :math:`\phi` sky position angle(s), in :math:`\\rm rad`.
        :param array or float t: The time(s) given as GMST.
        :param array or float psi: The GW polarisation angle(s) :math:`\psi`, in :math:`\\rm rad`.
        :param float rot: Further rotation of the interferometer with respect to the :py:data:`self.xax` orientation, in degrees, needed for the triangular geometry. In this case, the three arms will have orientations 1 --> :py:data:`self.xax`, 2 --> :py:data:`self.xax` + 60°, 3 --> :py:data:`self.xax` + 120°.
        :return: Plus and cross pattern functions of the detector evaluated at the given parameters.
        :rtype: tuple(array, array) or tuple(float, float)

        """

        rot_rad = rot * DEG_TO_RAD
        ras, decs = ra_dec_from_th_phi_rad(theta, phi)
        ab_factors = self.compute_ab_factors(
            ras, decs, t, rot_rad
        )

        sin_angbtwArms = np.sin(self.angbtwArms)
        Fp, Fc = apply_psi_rotation(psi, *ab_factors) * sin_angbtwArms
        return Fp, Fc

    def compute_geocent_deltat(self, theta, phi, t):
        """
        Compute the time needed to go from Earth center to detector location for a set of sky coordinates and time(s). The result is given in seconds.

        :param array or float theta: The :math:`\\theta` sky position angle(s), in :math:`\\rm rad`.
        :param array or float phi: The :math:`\phi` sky position angle(s), in :math:`\\rm rad`.
        :param array or float t: The time(s) given as GMST.

        :return: Time shift(s) to go from Earth center to detector location.
        :rtype: array or float

        """
        # Time needed to go from Earth center to detector location
        ras, decs = self._ra_dec_from_th_phi(theta, phi)

        # Note the change on 2025/04/21,
        # Output from second to days, as all subsequent usages are in seconds.
        return self._geocentric_deltat(ras, decs, t, self.lat_rad, self.long_rad)

    ##############################################################################
    # Helper functions for the Antenna Pattern
    ##############################################################################
    def _compute_ab_factors(
        self,
        ra,
        dec,
        time,
        rot,
        dphi=False,
        dtheta=False,
        dtime=False,
    ):
        """
        See P. Jaranowski, A. Krolak, B. F. Schutz, PRD 58, 063001, eq. (10)--(13)
        """
        sin_lat = np.sin(self.lat_rad)
        cos_lat = np.cos(self.lat_rad)
        sin_2lat = np.sin(2.0 * self.lat_rad)
        m3_cos_2lat = 3 - np.cos(2.0 * self.lat_rad)
        sin_2xax = np.sin(2.0 * (self.xax_rad + rot))
        cos_2xax = np.cos(2.0 * (self.xax_rad + rot))
        m3_cos_2dec = 3 - np.cos(2.0 * dec)
        cos_2dec = np.cos(2.0 * dec)
        sin_2dec = np.sin(2.0 * dec)

        ang = ra - self.lon_rad - TWOPI * time
        cos_2ang = np.cos(2.0 * ang)
        sin_2ang = np.sin(2.0 * ang)
        cos_ang = np.cos(ang)
        sin_ang = np.sin(ang)

        deltat_deriv = 0.0

        a1 = 0.0625 * sin_2xax * m3_cos_2lat
        a2 = 0.25 * cos_2xax * sin_lat
        a3 = 0.25 * sin_2xax * sin_2lat
        a4 = 0.5 * cos_2xax * cos_lat
        a5 = 3.0 * 0.25 * sin_2xax * cos_lat**2

        b1 = cos_2xax * sin_lat
        b2 = 0.25 * sin_2xax * m3_cos_2lat
        b3 = cos_2xax * cos_lat
        b4 = 0.5 * sin_2xax * sin_2lat

        if dphi or dtime:
            cos_2ang = -2 * np.sin(2.0 * ang)
            sin_2ang = +2 * np.cos(2.0 * ang)
            cos_ang = -1 * np.sin(ang)
            sin_ang = +1 * np.cos(ang)

        if dtheta:
            deltat_deriv = self._geocentric_deltat(ra, dec, time, dtheta=True)
            pi2_deltat = TWOPI * deltat_deriv
            a1 *= -2.0 * sin_2dec * cos_2ang + m3_cos_2dec * sin_2ang * (2.0 * pi2_deltat)
            a2 *= -2.0 * sin_2dec * sin_2ang - m3_cos_2dec * cos_2ang * (2.0 * pi2_deltat)
            a3 *= -2.0 * cos_2dec * cos_ang + sin_2dec * sin_ang * pi2_deltat
            a4 *= -1 * (2.0 * cos_2dec * sin_ang + sin_2dec * cos_ang * pi2_deltat)
            a5 *= sin_2dec

            b1 *= -np.cos(dec) * cos_2ang + np.sin(dec) * sin_2ang * (2.0 * pi2_deltat)
            b2 *= -np.cos(dec) * sin_2ang - np.sin(dec) * cos_2ang * (2.0 * pi2_deltat)
            b3 *= np.sin(dec) * cos_ang + np.cos(dec) * sin_ang * pi2_deltat
            b4 *= np.sin(dec) * sin_ang - np.cos(dec) * cos_ang * pi2_deltat

        else:
            a1 *= m3_cos_2dec * cos_2ang
            a2 *= m3_cos_2dec * sin_2ang
            a3 *= sin_2dec * cos_ang
            a4 *= sin_2dec * sin_ang
            a5 *= np.cos(dec) ** 2.0
            b1 *= np.sin(dec) * cos_2ang
            b2 *= np.sin(dec) * sin_2ang
            b3 *= np.cos(dec) * cos_ang
            b4 *= np.cos(dec) * sin_ang

        a_factor = a1 - a2 + a3 - a4 + a5
        b_factor = b1 + b2 + b3 + b4

        if dphi:
            a_factor = a1 - a2 + a3 - a4
            deltat_deriv = self._geocentric_deltat(ra, dec, time, dphi=True)
            a_factor *= 1.0 - TWOPI * deltat_deriv
            b_factor *= 1.0 - TWOPI * deltat_deriv
        elif dtime:
            a_factor = a1 - a2 + a3 - a4
            deltat_deriv = self._geocentric_deltat(ra, dec, time, dtime=True)
            a_factor *= -TWOPI * (1.0 + deltat_deriv)
            b_factor *= -TWOPI * (1.0 + deltat_deriv)

        return a_factor, b_factor, deltat_deriv

    def _geocentric_deltat(
        self, ra, dec, time, dphi=False, dtheta=False, dtime=False
    ):
        """
        Compute the time needed to go from Earth center to detector location
        for a set of sky coordinates and time(s). The result is given in days.

        Also the derivatives, simplified from:
        * `Delt_loc_phider`
        * `Delt_loc_thder`
        * `Delt_loc_tcder`

        :param array or float ra: The :math:`\\theta` sky position angle(s), in :math:`\\rm rad`.
        :param array or float dec: The :math:`\phi` sky position angle(s), in :math:`\\rm rad`.
        :param array or float time: The time(s) given as GMST.

        :return: Time shift (days) to go from Earth center to detector location.
        :rtype: array or float
        """
        cos_lat = np.cos(self.lat_rad)
        sin_lat = np.sin(self.lat_rad)
        sin_ra = np.sin(ra)
        cos_ra = np.cos(ra)
        sin_dec = np.sin(dec)
        cos_dec = np.cos(dec)

        deltat = self.lon_rad + TWOPI * time
        cos_deltat = np.cos(deltat)
        sin_deltat = np.sin(deltat)

        _comp1 = cos_ra * cos_lat
        _comp2 = sin_ra * cos_lat

        # This is to maintain the function being jit-able.
        deriv_case = 1 * dphi + 2 * dtheta + 4 * dtime
        comp1, comp2, comp3 = {
            0: (cos_dec * cos_deltat, cos_dec * sin_deltat, sin_dec * sin_lat),
            1: (-cos_dec * sin_deltat, +cos_dec * cos_deltat, 0.0),
            2: (sin_dec * cos_deltat, sin_dec * sin_deltat, -cos_dec * sin_lat),
            4: (-cos_dec * sin_deltat * TWOPI, +cos_dec * cos_deltat * TWOPI, 0.0),
        }[deriv_case]

        sum_comp = comp1 * _comp1 + comp2 * _comp2 + comp3
        # The minus sign arises from the definition of the unit vector pointing to the source
        earth_traverse_time = -REarth / clight / DAY_TO_SEC

        return sum_comp * earth_traverse_time