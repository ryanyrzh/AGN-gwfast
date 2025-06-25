from pathlib import Path

from jax import config
import jax.numpy as np
config.update("jax_enable_x64", True)
from scipy.interpolate import interp1d

import numpy as onp
from gwfast.gwfastGlobals import TWOPI, DAY_TO_SEC, DEG_TO_RAD, clight, REarth
from gwfast.gwfastUtils import (
    ra_dec_from_th_phi_rad,
    apply_psi_rotation,
)


class Detector(object):
    def __init__(
        self,
        name,
        lat,
        lon,
        xax,
        shape,
        duty_cycle=None,
        noise_curve_path=None,
        verbose=False,
    ):
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

        self.psd_frequencies, spectral_density = onp.loadtxt(
            file_path, usecols=(0, 1), unpack=True
        )

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

        # Out of the provided PSD range, we use a constant value of 1,
        # which results in completely negligible conntributions
        self.psd_interp = interp1d(
            self.psd_frequencies, self.psd_array, bounds_error=False, fill_value=1.0
        )

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
        *ab_factors, _ = self._compute_ab_factors(ras, decs, t, rot_rad)

        sin_angbtwarms = np.sin(self.ang_btw_arms)
        Fp, Fc = apply_psi_rotation(psi, *ab_factors) * sin_angbtwarms
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
        ras, decs = ra_dec_from_th_phi_rad(theta, phi)

        # Note the change on 2025/04/21,
        # Output from second to days, as all subsequent usages are in days.
        return self._geocentric_deltat(ras, decs, t)

    def CoeffsRot(self, ra, dec, psi, rot=0.0):
        rot = rot * DEG_TO_RAD
        rasDet = ra - self.lon_rad
        # Referring to overleaf, I now call VC2 the last vector appearing in the C2 expression, VS2 the one in the S2 expression and so on
        # e1 is the first element and e2 the second

        sin_angbtwarms = np.sin(self.ang_btw_arms)

        sin_lat = np.sin(self.det_lat_rad)
        cos_lat = np.cos(self.det_lat_rad)
        sin_2lat = np.sin(2.0 * self.det_lat_rad)
        m3_cos_2lat = 3 - np.cos(2.0 * self.det_lat_rad)
        sin_2xax = np.sin(2.0 * (self.det_xax_rad + rot))
        cos_2xax = np.cos(2.0 * (self.det_xax_rad + rot))
        cos_2ra = np.cos(2.0 * rasDet)
        sin_2ra = np.sin(2.0 * rasDet)
        m3_cos_2dec = 3 - np.cos(2.0 * dec)
        sin_2dec = np.sin(2.0 * dec)

        # TODO: Why 0.0675?
        VC2e1 = (
            0.0675 * cos_2ra * sin_2xax * m3_cos_2dec * m3_cos_2lat
            - 0.25 * sin_2ra * cos_2xax * m3_cos_2dec * sin_lat
        )
        VC2e2 = (
            0.25 * sin_2ra * sin_2xax * np.sin(dec) * m3_cos_2lat
            + cos_2ra * cos_2xax * np.sin(dec) * sin_lat
        )
        C2p, C2c = sin_angbtwarms * apply_psi_rotation(psi, VC2e1, VC2e2)

        VS2e1 = (
            0.0675 * sin_2ra * sin_2xax * m3_cos_2dec * m3_cos_2lat
            + 0.25 * cos_2ra * cos_2xax * m3_cos_2dec * sin_lat
        )
        VS2e2 = (
            -0.25 * cos_2ra * sin_2xax * np.sin(dec) * m3_cos_2lat
            + sin_2ra * cos_2xax * np.sin(dec) * sin_lat
        )
        S2p, S2c = sin_angbtwarms * apply_psi_rotation(psi, VS2e1, VS2e2)

        VC1e1 = 0.25 * (
            np.cos(rasDet) * sin_2xax * sin_2dec * sin_2lat
            - 2 * np.sin(rasDet) * cos_2xax * sin_2dec * cos_lat
        )
        VC1e2 = (
            np.cos(rasDet) * cos_2xax * np.cos(dec) * cos_lat
            + 0.5 * np.sin(rasDet) * sin_2xax * np.cos(dec) * sin_2lat
        )
        C1p, C1c = sin_angbtwarms * apply_psi_rotation(psi, VC1e1, VC1e2)

        VS1e1 = 0.25 * (
            np.sin(rasDet) * sin_2xax * sin_2dec * sin_2lat
            + 2 * np.cos(rasDet) * cos_2xax * sin_2dec * cos_lat
        )
        VS1e2 = (
            np.sin(rasDet) * cos_2xax * np.cos(dec) * cos_lat
            - 0.5 * np.cos(rasDet) * sin_2xax * np.cos(dec) * sin_2lat
        )
        S1p, S1c = sin_angbtwarms * apply_psi_rotation(psi, VS1e1, VS1e2)

        _C0 = 0.75 * sin_2xax * ((np.cos(dec) * cos_lat) ** 2) * sin_angbtwarms
        C0p = _C0 * np.cos(2.0 * psi)
        C0c = -_C0 * np.sin(2.0 * psi)

        return (
            np.array([C2p, C2c]),
            np.array([S2p, S2c]),
            np.array([C1p, C1c]),
            np.array([S1p, S1c]),
            np.array([C0p, C0c]),
        )

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
        a5 = 3.0 * 0.25 * sin_2xax * cos_lat ** 2

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
            a1 *= -2.0 * sin_2dec * cos_2ang + m3_cos_2dec * sin_2ang * (
                2.0 * pi2_deltat
            )
            a2 *= -2.0 * sin_2dec * sin_2ang - m3_cos_2dec * cos_2ang * (
                2.0 * pi2_deltat
            )
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

    def _geocentric_deltat(self, ra, dec, time, dphi=False, dtheta=False, dtime=False):
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

    def _phiPhase(self, iota, theta, phi, t, psi, Fp=None, Fc=None):
        # The polarization phase contribution (the change in F+ and Fx with time influences also the phase)
        if (Fp is None) or (Fc is None):
            Fp, Fc = self.compute_antenna_pattern(theta, phi, t, psi)

        phiP = -np.arctan2(np.cos(iota) * Fc, 0.5 * (1.0 + np.cos(iota) ** 2) * Fp)

        # The contriution to the amplitude is negligible, so we do not compute it
        return phiP


def FpFcsqInt(C2s, S2s, C1s, S1s, C0s, Igs, iota):
    # Is he out of his mind??
    Fp4 = 0.5 * (C2s[0] ** 2 - S2s[0] ** 2) * Igs[3] + C2s[0] * S2s[0] * Igs[7]

    Fp3 = (C2s[0] * C1s[0] - S2s[0] * S1s[0]) * Igs[2] + (
        C2s[0] * S1s[0] + S2s[0] * C1s[0]
    ) * Igs[6]

    Fp2 = (0.5 * (C1s[0] ** 2 - S1s[0] ** 2) + 2.0 * C2s[0] * C0s[0]) * Igs[1] + (
        2.0 * C0s[0] * S2s[0] + C1s[0] * S1s[0]
    ) * Igs[5]

    Fp1 = (2.0 * C0s[0] * C1s[0] + C1s[0] * C2s[0] + S2s[0] * S1s[0]) * Igs[0] + (
        2.0 * C0s[0] * S1s[0] + C1s[0] * S2s[0] - S1s[0] * C2s[0]
    ) * Igs[4]

    Fp0 = (
        C0s[0] ** 2 + 0.5 * (C1s[0] ** 2 + C2s[0] ** 2 + S1s[0] ** 2 + S2s[0] ** 2)
    ) * Igs[8]

    FpsqInt = Fp4 + Fp3 + Fp2 + Fp1 + Fp0

    Fc4 = 0.5 * (C2s[1] ** 2 - S2s[1] ** 2) * Igs[3] + C2s[1] * S2s[1] * Igs[7]

    Fc3 = (C2s[1] * C1s[1] - S2s[1] * S1s[1]) * Igs[2] + (
        C2s[1] * S1s[1] + S2s[1] * C1s[1]
    ) * Igs[6]

    Fc2 = (0.5 * (C1s[1] ** 2 - S1s[1] ** 2) + 2.0 * C2s[1] * C0s[1]) * Igs[1] + (
        2.0 * C0s[1] * S2s[1] + C1s[1] * S1s[1]
    ) * Igs[5]

    Fc1 = (2.0 * C0s[1] * C1s[1] + C1s[1] * C2s[1] + S2s[1] * S1s[1]) * Igs[0] + (
        2.0 * C0s[1] * S1s[1] + C1s[1] * S2s[1] - S1s[1] * C2s[1]
    ) * Igs[4]

    Fc0 = (
        C0s[1] ** 2 + 0.5 * (C1s[1] ** 2 + C2s[1] ** 2 + S1s[1] ** 2 + S2s[1] ** 2)
    ) * Igs[8]

    FcsqInt = Fc4 + Fc3 + Fc2 + Fc1 + Fc0

    return (
        FpsqInt * (0.5 * (1.0 + (np.cos(iota)) ** 2)) ** 2,
        FcsqInt * (np.cos(iota)) ** 2,
    )
