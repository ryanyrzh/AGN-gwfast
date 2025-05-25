from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as np
from jax.lax import integer_pow

from astropy.cosmology import Planck18 as cosmo
from gwfast.gwfastGlobals import MRSUN_SI, MTSUN_SI, uGpc
from gwfast.lensing_utils import _get_alpha_hat, get_im_pos

zGridGlob = np.logspace(start=-6, stop=5, base=10, num=7000)
dLGridGlob = cosmo.luminosity_distance(zGridGlob) / 1000.0  # Gpc


def einstein_angle(lens_mass_source, angular_D_L, D_LS):
    '''
    Compute the Einstein radius (θ_E) from the given distances.

    We assume D_S = D_L + D_LS, and we assume D_LS is sufficiently
    small such that the (1 + z)^2 correction is unnecessary.

    Making use of the distance hierarchsies, we write:
        θ_E^2 = 2 * (R_S / D_L) * [d_ls / (1 + d_ls)]
        where d_ls = D_LS / D_L

    Parameters
    ----------
    lens_mass_source: float / array-like
        The mass of the lens in the source frame, in solar masses.
    angular_D_L: float / array-like
        The angular distance between the lens and the observer, in Gpc.
    D_LS: float / array-like
        The luminosity distance between the lens and the source, in R_Sch.

    Returns
    -------
    theta_E: float / array-like
        The Einstein radius in radian.
    '''
    RSch = 2 * lens_mass_source * MRSUN_SI  # m
    RSch_2_Gpc = RSch / uGpc

    RSch_DL = RSch_2_Gpc / angular_D_L
    d_ls = D_LS * RSch_DL

    return np.sqrt(2 * RSch_DL * d_ls / (1 + d_ls))


def get_phi_L(iota, y_src_pos, r_orbit, phi_N):
    '''
    Computes the phi_L from the given observer and source positions,
        such that lensing could happen.

    This implies that:
        |δφ| = |φN - φL| < 90º

    In order to recover the sign of δφ, we allow input of
    negative y_src_pos to indicate that.

    Parameters
    ----------
    iota: float / array-like
        Inclination of the observer w.r.t. to the source, radian.
    phi_N: float / array-like
        Azimuthal angle of the observer w.r.t. to the source, radian.
    y_src_pos: float / array-like
        The source position, from (-1, +1), units of r_orbit.
    r_orbit: float / array-like
        The orbital radius of the source around the lens, R_Sch.

    Returns
    -------
    phi_L: float / array-like
        The azimuthal angle of the lens around the source,
        such that lensing could occur.
    '''
    arg = np.sqrt(1 - y_src_pos**2) / np.sin(iota)
    return phi_N + np.sign(y_src_pos) * np.arccos(arg)


def line_of_sight_unit_vec(iota, phase):
    phi = np.pi / 2 - phase
    return np.array([
        np.sin(iota) * np.cos(phi),
        np.sin(iota) * np.sin(phi),
        np.cos(iota)
    ])


def compute_exact_lensed_angles_SourceFrame(agn_bbh_system_params):
    '''
    In the following, all vectors will take shape (3, N),
    where N is the number of samples.

    This follows the convention in the paper draft at the moment.

    We abbreviate the frames as follows:
    - Source frame: `_src`
    - Lens plane frame: `_lens`
    '''
    iota = agn_bbh_system_params["iota"]
    phase = agn_bbh_system_params["Phicoal"]
    # phi_L = agn_bbh_system_params["phi_L"]
    r_orbit = agn_bbh_system_params["R_orbit"]  # R_Sch
    luminosity_distance = agn_bbh_system_params["dL"]  # Gpc
    lens_mass = agn_bbh_system_params["M_lz"]  # Gpc
    y_src = agn_bbh_system_params["src_pos"]  # R_orbit
    zeros = np.zeros_like(iota)
    L_hat_src = np.array([zeros, zeros, zeros + 1])

    # Useful constructs
    phi_N = np.pi / 2 - phase
    d_LS = r_orbit * np.sqrt(1 - y_src**2)  # R_Sch

    # Schwarschild radius
    # A small, but necessary assumption, that the lens is at dL
    z = np.interp(luminosity_distance, dLGridGlob, zGridGlob)
    lens_mass_src = lens_mass / (1 + z)
    R_Sch = 2 * lens_mass_src * MRSUN_SI  # m
    delta = R_Sch / uGpc
    sq_1pz = integer_pow(1 + z, 2)

    # Angular distances
    # For most practical purposes, ang_lum_dist = ang_D_S
    ang_lum_dist = luminosity_distance / sq_1pz  # Gpc
    _y_src = y_src * r_orbit  # R_Sch
    beta = np.arcsin(_y_src / ang_lum_dist * delta)  # Radian
    ang_D_S = ang_lum_dist * np.cos(beta)   # Gpc
    ang_D_L = ang_D_S / (1 + d_LS * delta)  # Gpc
    theta_E = einstein_angle(lens_mass_src, ang_D_L, d_LS)  # Radian

    # Source and image positions
    beta = _y_src / ang_D_S * delta  # Radian
    beta_thetaE = beta / theta_E

    _img_pos_1, _img_pos_2 = get_im_pos(beta_thetaE)  # in units of Einstein angle
    img_pos_1 = _img_pos_1 * theta_E
    img_pos_2 = _img_pos_2 * theta_E

    # The opening angles
    alpha_hat = _get_alpha_hat(r_orbit)  # rad
    theta_bar_p = alpha_hat - img_pos_1 + beta
    theta_bar_m = alpha_hat - img_pos_2 - beta

    phi_L = agn_bbh_system_params.get(
            'phi_L', get_phi_L(iota, y_src, r_orbit, phi_N))

    obs_pos = line_of_sight_unit_vec(iota, phase)
    lens_pos = np.array([np.cos(phi_L), np.sin(phi_L), zeros])

    lens_pln_x = obs_pos
    lens_pln_z = np.cross(lens_pos, obs_pos, axis=0)
    lens_pln_z /= np.linalg.norm(lens_pln_z, axis=0)
    lens_pln_y = np.cross(lens_pln_z, lens_pln_x, axis=0)
    lens_pln_y /= np.linalg.norm(lens_pln_y, axis=0)
    lens_pln_frame = np.array([lens_pln_x, lens_pln_y, lens_pln_z])

    # Compute phi_L from vectors:
    optical_axis = luminosity_distance * obs_pos - r_orbit * lens_pos
    optical_axis /= np.linalg.norm(optical_axis, axis=0)

    # Image positions in the lensing plane
    img_p_hat_lens = np.array([
        np.cos(theta_bar_p), np.sin(theta_bar_p), zeros
    ])
    img_m_hat_lens = np.array([
        np.cos(theta_bar_m), -np.sin(theta_bar_m), zeros
    ])
    img_p_hat_src = np.einsum('ik,ijk->jk', img_p_hat_lens, lens_pln_frame)
    img_m_hat_src = np.einsum('ik,ijk->jk', img_m_hat_lens, lens_pln_frame)

    iota_p = np.arccos(img_p_hat_src[2])
    iota_m = np.arccos(img_m_hat_src[2])

    # Get the change in phase

    # Define the line-of-separation vector
    _phi_p = np.arctan2(img_p_hat_src[1], img_p_hat_src[0])
    _phi_m = np.arctan2(img_m_hat_src[1], img_m_hat_src[0])

    phase_p = np.pi / 2 - _phi_p
    phase_m = np.pi / 2 - _phi_m

    # Not implementing the polarisation angle shift

    # These velocities are in unit of c
    v_orbit_mag = 1 / np.sqrt(2 * r_orbit)
    v_orbit_hat = np.cross(L_hat_src, lens_pos, axis=0)
    v_orbit_hat /= np.linalg.norm(v_orbit_hat, axis=0)
    v_orbit_vec_src = v_orbit_mag * r_orbit * v_orbit_hat
    v_proj_p = np.einsum('ij,ij->j', v_orbit_vec_src, img_p_hat_src)
    v_proj_m = np.einsum('ij,ij->j', v_orbit_vec_src, img_m_hat_src)

    return {
        'iota_p': iota_p,
        'iota_m': iota_m,
        'phase_p': phase_p,
        'phase_m': phase_m,
        'v_proj_p': v_proj_p,
        'v_proj_m': v_proj_m,
    }


