from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as np
from jax.lax import integer_pow

from astropy.cosmology import Planck18 as cosmo
from gwfast.gwfastGlobals import MRSUN_SI, MTSUN_SI, uGpc, DAY_TO_SEC
from gwfast.old_lensing_utils import _get_alpha_hat

zGridGlob = np.logspace(start=-6, stop=5, base=10, num=7000)
dLGridGlob = cosmo.luminosity_distance(zGridGlob) / 1000.0  # Gpc


def einstein_angle(lens_mass_source, angular_D_L, D_LS):
    '''
    Compute the Einstein radius (θ_E) from the given distances.

    We assume D_S = D_L + D_LS, and we assume D_LS is sufficiently
    small such that its (1 + z)^2 correction is unnecessary.

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


def get_phi_L(iota, y_src_pos, phi_N):
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
    return phi_N - np.sign(y_src_pos) * np.arccos(arg)


def Keplerian_speed(r_orbit):
    '''
    From arXiv:2310.16025, Eq.(2)
        v = (2r - 1)^(-1/2)

    Parameters:
    ----------
    r_orbit: float / array-like
        The orbital radius of the source around the SMBH, R_Sch.

    Returns:
    ----------
    float / array-like
        The orbital speed from Kepler's law, light speed.
    '''

    # TODO: Check whether this is true
    return (2 * r_orbit - 1)**(-0.5)


def gravitational_redshift(r_orbit):
    '''
    From arXiv:2310.16025, Eq.(3)
        z_grav = (1 - 1/r_orbit)^1/2 - 1

    Parameters:
    ----------
    r_orbit: float / array-like
        The orbital radius of the source around the SMBH, R_Sch.

    Returns:
    ----------
    float / array-like
        The gravitational redshift from a Schwarzschild BH.
    '''

    # TODO: Check whether this is true
    return (1 - 1 / r_orbit)**0.5 - 1


def Lorentz_factor(beta):
    return (1 - beta**2)**(-0.5)


def PML_image_position(beta_src, theta_E=1):
    _beta = beta_src / theta_E
    sqrt_term = np.sqrt(4 + _beta**2)
    img_p = (_beta + sqrt_term) / 2 * theta_E
    img_m = (_beta - sqrt_term) / 2 * theta_E
    return img_p, img_m


def PML_time_delay_magnification(beta_src, theta_E=1):
    '''
    Time delay is simplified from https://inspirehep.net/literature/1862768,
    Eq. (3.13), with θ_± = (β ± √ (β² + 4)) / 2, for β measured in units of θ_E.

    t(+) - t(-) = -β √(4 +  β²) + 2 ln(θ(-) / θ(+))

    The magnifications are taken from Eq. (3.8) directly.

    Parameters:
    ----------
    beta_src: float / array-like
        The source position angle.
        If theta_E is not given, it is assumed to be in unit of theta_E.
        Otherwise, it should have the same unit as theta_E (radian, R_Sch, etc).
    theta_E: float / array-like
        The Einstein radius (angle), it acts as the scale of for beta_src.
        It can have any units, as long as it being consistent with beta_src.

    Returns:
    ----------
    delta_t: float / array-like
        The time delay between the "+" and "-" images, in geometric time.
        (Need to multiply by GM/c^3 to get SI unit.)
    mag_p, mag_m: float / array-like
        The magnification of the "+" and "-" images respectively.
    '''

    img_p, img_m = PML_image_position(beta_src, theta_E)

    _beta = beta_src / theta_E
    sqrt_term = np.sqrt(4 + _beta**2)
    delta_t_geom = - _beta * sqrt_term
    delta_t_Shap = 2 * np.log(np.abs(img_m / img_p))

    # The factor of 2 is to account for the later multiplication by GM/c^3
    delta_t = (delta_t_geom + delta_t_Shap) * 2

    common_term = _beta / sqrt_term + sqrt_term / _beta
    mag_p = 0.25 * (common_term + 2)
    mag_m = 0.25 * (common_term - 2)

    return delta_t, mag_p, mag_m


def line_of_sight_unit_vec(iota, phase):
    phi = np.pi / 2 - phase
    return np.array([
        np.sin(iota) * np.cos(phi),
        np.sin(iota) * np.sin(phi),
        np.cos(iota)
    ])


def convert_y_from_Einstein_to_Rorbit(y_Eins, r_orbit):
    kappa = y_Eins*y_Eins / r_orbit
    return np.sign(y_Eins) * np.sqrt(2 * kappa * (np.sqrt(1 + kappa*kappa) - kappa))


def convert_y_from_Einstein_to_Rorbit_first_order(y_Eins, r_orbit, delta):
    kappa = y_Eins*y_Eins / r_orbit
    delta_r = r_orbit * delta
    surd = np.sqrt(1 + (kappa - delta_r)**2)
    t2 = kappa + delta_r
    return np.sign(y_Eins) * np.sqrt(2 * kappa / (surd + t2))


def get_agn_lens_angles(redshifted_lens_mass, r_orbit, source_position,
                        luminosity_distance, angular_distances=False):
    '''
    Computes the Einstein angle, the source position angle, and the lens mass.

    Parameters
    ----------
    redshifted_lens_mass: float / array-like
        The redshifted lens mass in the detector frame, in solar masses.
    r_orbit: float / array-like
        The orbital radius of the binary BHs around the lens, in R_Sch.
    source_position: float / array-like
        The source position angle, in units of r_orbit.
        It should be in the range [-1, 1].
    luminosity_distance: float / array-like
        The unperturbed luminosity distance to the source, in Gpc.
    angular_distances: bool, optional
        When True, convert to angular distances when computing theta_E;
        otherwise, luminosity distances are used instead.
        Default is False.

    Returns
    -------
    theta_E: float / array-like
        The Einstein angle in radian.
    beta: float / array-like
        The source position angle in radian.
    lens_mass_source: float / array-like
        The lens mass in the source frame, in solar masses.
    '''
    d_LS = r_orbit * np.sqrt(1 - source_position**2)  # R_Sch
    _source_position = source_position * r_orbit  # R_Sch

    # Schwarschild radius
    # A small, but necessary assumption, that the lens is at dL
    z = np.interp(luminosity_distance, dLGridGlob, zGridGlob)
    lens_mass_source = redshifted_lens_mass / (1 + z)
    R_Sch = 2 * lens_mass_source * MRSUN_SI  # m
    delta = R_Sch / uGpc

    if angular_distances:
        # For most practical purposes, ang_lum_dist = ang_D_S
        ang_lum_dist = luminosity_distance / integer_pow(1 + z, 2)  # Gpc
        beta = np.arcsin(_source_position / ang_lum_dist * delta)  # Radian
        ang_D_S = ang_lum_dist * np.cos(beta)   # Gpc
    else:
        beta = _source_position / luminosity_distance * delta  # Radian
        ang_D_S = luminosity_distance

    ang_D_L = ang_D_S / (1 + d_LS * delta)  # Gpc
    theta_E = einstein_angle(lens_mass_source, ang_D_L, d_LS)  # Radian

    return theta_E, beta, lens_mass_source


def get_agn_lensed_parameters(unlensed_parameters):
    plus_image_params = unlensed_parameters.copy()
    minus_image_params = unlensed_parameters.copy()

    lensed_params = compute_lensed_angles_approx(unlensed_parameters)

    # Environemental effects (orbit-induced redshift and gravitational redshift) can be modeled as
    # changes in effective chirp mass and effective luminosity distance
    # https://arxiv.org/abs/2310.16025 Eqs. 4&5
    plus_redshift_factor = (1 + lensed_params['z_rel_p']) * (1 + lensed_params['z_grav'])
    minus_redshift_factor = (1 + lensed_params['z_rel_m']) * (1 + lensed_params['z_grav'])

    plus_image_params['iota'] = lensed_params['iota_p']
    plus_image_params['phase'] = lensed_params['phase_p']
    plus_image_params['Mc'] *= plus_redshift_factor
    plus_image_params['dL'] /= lensed_params['sqrt_mu_p']
    plus_image_params['dL'] *= (1 + lensed_params['z_rel_p']) * plus_redshift_factor
    minus_image_params['iota'] = lensed_params['iota_m']
    minus_image_params['phase'] = lensed_params['phase_m']
    minus_image_params['Mc'] *= minus_redshift_factor
    minus_image_params['dL'] /= lensed_params['sqrt_mu_m']
    minus_image_params['dL'] *= (1 + lensed_params['z_rel_m']) * minus_redshift_factor
    minus_image_params['tcoal'] += lensed_params['delta_time']  # days

    return plus_image_params, minus_image_params


def convert_simple_PML_to_general_lensed_parameters(parameters):
    output_params = parameters.copy()
    luminosity_distance = output_params.pop("dL")

    theta_E, beta, lens_mass_src = get_agn_lens_angles(
        output_params['M_lz'], output_params['R_orbit'],
        output_params['src_pos'], luminosity_distance
    )
    time_delay, mag_1, mag_2 = PML_time_delay_magnification(beta_src=beta, theta_E=theta_E)

    output_params['delta_time'] = time_delay * lens_mass_src * MTSUN_SI / DAY_TO_SEC  # Days (tcoal)
    output_params['dL_1'] = luminosity_distance / np.sqrt(np.abs(mag_1))
    output_params['dL_2'] = luminosity_distance / np.sqrt(np.abs(mag_2))
    output_params['delta_iota'] = np.zeros_like(mag_1)  # No change in iota
    output_params['delta_phase'] = np.zeros_like(mag_1)  # No change in phi
    output_params['relative_mass'] = np.ones_like(mag_1)  # No change in phi
    return output_params


def compute_lensed_angles_approx(
        agn_bbh_system_params, angular_distances=False):
    parameters = agn_bbh_system_params.copy()
    iota = parameters["iota"]
    phase = parameters["phase"]
    psi = parameters["psi"]
    r_orbit = parameters["R_orbit"]  # R_Sch
    y_src = parameters["src_pos"]  # R_orbit

    # Useful constructs
    phi_N = np.pi / 2 - phase

    theta_E, beta, lens_mass_src = get_agn_lens_angles(
        parameters["M_lz"], r_orbit, y_src, parameters["dL"],
        angular_distances=angular_distances)

    # Image positions
    img_pos_1, img_pos_2 = PML_image_position(beta, theta_E)  # Radian

    # The opening angles
    alpha_hat = _get_alpha_hat(r_orbit)  # rad
    theta_bar_p = alpha_hat - (img_pos_1 - beta)
    theta_bar_m = alpha_hat - (img_pos_2 + beta)

    # Setting phi_N = 0 gives - delta_phi
    delta_phi = - get_phi_L(iota, y_src, 0)

    inv_Delta = (np.cos(iota)**2 + np.sin(iota)**2 * np.sin(delta_phi)**2)**-0.5
    iota_term = np.cos(iota) * np.cos(delta_phi) * inv_Delta
    phi_term = np.sin(delta_phi) / np.sin(iota) * inv_Delta
    psi_term = np.sin(iota) * np.sqrt(np.tan(iota)**2 + 1 / np.sin(delta_phi)**2)
    speed_term = np.sin(iota) * np.cos(delta_phi) * inv_Delta

    iota_p = iota - theta_bar_p * iota_term
    iota_m = iota + theta_bar_m * iota_term
    phi_p = phi_N + theta_bar_p * phi_term
    phi_m = phi_N - theta_bar_m * phi_term
    psi_p = psi + theta_bar_p / psi_term
    psi_m = psi + theta_bar_m / psi_term

    v_orb = Keplerian_speed(r_orbit)
    gamma = Lorentz_factor(v_orb)
    v_proj = - v_orb * np.sin(iota) * np.sin(delta_phi)
    v_orb_p = v_proj * (1 + theta_bar_p * speed_term)
    v_orb_m = v_proj * (1 - theta_bar_m * speed_term)

    z_rel_p = gamma * (1 + v_orb_p) - 1
    z_rel_m = gamma * (1 + v_orb_m) - 1
    z_grav = gravitational_redshift(r_orbit)

    delta_time, mu_p, mu_m = PML_time_delay_magnification(beta, theta_E)
    delta_time *= lens_mass_src * MTSUN_SI / DAY_TO_SEC  # days
    sqrt_mu_p = np.sqrt(np.abs(mu_p))
    sqrt_mu_m = np.sqrt(np.abs(mu_m))

    return {
        'iota_p': iota_p,
        'iota_m': iota_m,
        'phase_p': np.pi/2 - phi_p,
        'phase_m': np.pi/2 - phi_m,
        'psi_p': psi_p,
        'psi_m': psi_m,
        'v_proj_p': v_orb_p,
        'v_proj_m': v_orb_m,
        'z_rel_p': z_rel_p,
        'z_rel_m': z_rel_m,
        'z_grav': z_grav,
        'delta_time': delta_time,
        'sqrt_mu_p': sqrt_mu_p,
        'sqrt_mu_m': sqrt_mu_m,
    }


def compute_exact_lensed_angles_SourceFrame(agn_bbh_system_params):
    '''
    In the following, all vectors will take shape (3, N),
    where N is the number of samples.

    This follows the convention in the paper draft at the moment.

    We abbreviate the frames as follows:
    - Source frame: `_src`
    - Lens plane frame: `_lens`
    '''
    parameters = agn_bbh_system_params.copy()
    iota = parameters["iota"]
    phase = parameters["phase"]
    r_orbit = parameters["R_orbit"]  # R_Sch
    luminosity_distance = parameters["dL"]  # Gpc
    y_src = parameters["src_pos"]  # R_orbit
    zeros = np.zeros_like(iota)
    L_hat_src = np.array([zeros, zeros, zeros + 1])

    # Useful constructs
    phi_N = np.pi / 2 - phase
    theta_E, beta, lens_mass_src = get_agn_lens_angles(
        parameters["M_lz"], r_orbit, y_src, parameters["dL"])

    # Image positions
    img_pos_1, img_pos_2 = PML_image_position(beta, theta_E)  # Radian

    # The opening angles
    alpha_hat = _get_alpha_hat(r_orbit)  # rad
    theta_bar_p = alpha_hat - (img_pos_1 - beta)
    theta_bar_m = alpha_hat - (img_pos_2 + beta)

    phi_L = agn_bbh_system_params.get(
            'phi_L', get_phi_L(iota, y_src, phi_N))

    obs_pos = line_of_sight_unit_vec(iota, phase)
    lens_pos = np.array([np.cos(phi_L), np.sin(phi_L), zeros])

    lens_pln_x = obs_pos
    lens_pln_z = np.cross(lens_pos, obs_pos, axis=0)
    lens_pln_z /= np.linalg.norm(lens_pln_z, axis=0)
    lens_pln_y = np.cross(lens_pln_z, lens_pln_x, axis=0)
    lens_pln_y /= np.linalg.norm(lens_pln_y, axis=0)
    lens_pln_frame = np.array([lens_pln_x, lens_pln_y, lens_pln_z])

    # Compute phi_L from vectors:
    R_Sch = 2 * lens_mass_src * MRSUN_SI  # m
    delta = R_Sch / uGpc
    optical_axis = obs_pos - r_orbit / luminosity_distance * delta * lens_pos
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
    phi_p = np.arctan2(img_p_hat_src[1], img_p_hat_src[0])
    phi_m = np.arctan2(img_m_hat_src[1], img_m_hat_src[0])

    # These velocities are in unit of c
    v_orbit_mag = Keplerian_speed(r_orbit)
    v_orbit_hat = np.cross(L_hat_src, lens_pos, axis=0)
    v_orbit_hat /= np.linalg.norm(v_orbit_hat, axis=0)
    v_orbit_vec_src = v_orbit_mag * r_orbit * v_orbit_hat
    v_proj_p = np.einsum('ij,ij->j', v_orbit_vec_src, img_p_hat_src)
    v_proj_m = np.einsum('ij,ij->j', v_orbit_vec_src, img_m_hat_src)

    gamma = Lorentz_factor(v_orbit_mag)
    z_rel_p = gamma * (1 + v_proj_p) - 1
    z_rel_m = gamma * (1 + v_proj_m) - 1

    return {
        'iota_p': iota_p,
        'iota_m': iota_m,
        'phase_p': np.pi / 2 - phi_p,
        'phase_m': np.pi / 2 - phi_m,
        'v_proj_p': v_proj_p,
        'v_proj_m': v_proj_m,
        'z_rel_p': z_rel_p,
        'z_rel_m': z_rel_m,
    }


