from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as np

from astropy.cosmology import Planck18 as cosmo
from gwfast.gwfastGlobals import MRSUN_SI, MTSUN_SI, uGpc, DEG_TO_RAD

zGridGlob = np.logspace(start=-6, stop=5, base=10, num=7000)
dLGridGlob = cosmo.luminosity_distance(zGridGlob) / 1000.0  # Gpc

# Constants in SI
# TODO: Use the constants defined in gwfastGlobals
G = 6.6743 * 1e-11
c = 2.979246 * 1e8


##############################################################################
# LENSING
##############################################################################
def theta_in_terms_of_x_0(x_0, D_l):
    """
    x_0: Minimal approach distance [R_Sch]
    D_l: Lens distance [R_Sch]
    """
    return x_0 / (D_l * np.sqrt(1 - 1 / x_0))


def thin_lens_equation(x_0, beta, D_ls, D_l):
    """
    x_0:    Minimal approach distance [R_Sch]
    beta:   Angular source position [radian]
    D_ls:   Lens-source plane distance [R_Sch]
    D_l:    Lens distance [R_Sch]
    """
    D_s = D_l + D_ls
    theta = theta_in_terms_of_x_0(x_0, D_l)
    return beta - theta + D_ls / D_s * alpha(x_0)


def alpha(x_0):
    """
    x_0: Minimal approach distance [R_Sch]
    """
    # Alpha approximations
    x2_coef = (15 / 16) * np.pi - 1
    x2_coef = 0
    return 2 / x_0 + x2_coef / (x_0**2)


# def einstein_radius(D_ls, D_l):
#     '''
#     The Einstein radius
#     D_ls:   Lens-source plane distance [R_Sch]
#     D_l:    Lens distance [R_Sch]
#     '''
#     D_ratio = (1 + D_ls / D_l) * D_l**2
#     return np.sqrt(2 * D_ratio)

def einstein_radius(M_lens, dL, R_orbit):
    """
    Calculate Einstein radius [rad], using the effectve formula given in https://en.wikipedia.org/wiki/Einstein_radius
    With approximations D_s=D_l, D_ls=R_orbit
    M_lens: Lens mass [M_sun]
    dL: Source luminosity distance [Gpc]
    R_orbit: Orbital radius of BBH around AGN [RSch]
    """
    M_term = M_lens / 10**11.09
    D_ls_in_Gpc = get_Gpc_from_R_Sch(
        R_orbit, M_lens
    )  # Lens-source distance approximated as robital radius
    z = np.interp(dL, dLGridGlob, zGridGlob)
    D_l_in_Gpc = dL / (1 + z) ** 2
    D_term = D_ls_in_Gpc / D_l_in_Gpc**2  # Approximating D_s=D_s
    einstein_radius_in_arcsec = np.sqrt(M_term * D_term)
    einstein_radius_in_rad = einstein_radius_in_arcsec * (1 / 3600) * DEG_TO_RAD
    return einstein_radius_in_rad


def get_Gpc_from_R_Sch(qty, M):
    """
    Convert from units of Schwarzschild radius to Gpc for a given mass.
    qty: Distance to be converted [R_Sch]
    M: Mass corresponding to the Schwarzschild radius [M_sun]
    """
    one_R_Sch = 2 * MRSUN_SI * M  # m
    qty_in_m = qty * one_R_Sch
    qty_in_Gpc = qty_in_m / uGpc
    return qty_in_Gpc


def get_R_Sch_from_Gpc(qty, M):
    """
    Convert from Gpc to units of Schwarzschild radius of a given mass
    qty: Distance to be converted [Gpc]
    M: Mass for which to calculate R_Sch [M_sun]
    """
    R_Sch_in_m = 2 * MRSUN_SI * M  # m
    R_Sch_in_Gpc = R_Sch_in_m / uGpc
    return qty / R_Sch_in_Gpc


def get_im_pos(src_pos):
    """
    Image positions, normalized by Einstein radius.
    src_pos: Source position [Einstein radius]
    """
    sqrt_term = np.sqrt(src_pos**2 / 4 + 1)
    im_pos_1 = src_pos / 2 + sqrt_term
    im_pos_2 = src_pos / 2 - sqrt_term
    return im_pos_1, im_pos_2


def _get_alpha_hat(R_orbit, approx=1):
    """
    Compute deflection angle from the orbital radius
    between the BBH and the SMBH.

    This assumes β = 0.

    R_orbit -- Unit: Schwarschild radius
    """
    approx_simp = np.sqrt(2 / R_orbit)
    match approx:
        ## Approx 1: The simplest approximation
        ## assuming α(x) to the first order
        case 1:
            return approx_simp
        ## Approx 2: Fit with log(r) vs log(err)
        ## still assuming α(x) to the first order
        case 2:
            idx = -0.5415779752686682
            y0 = -0.6327303836364937
            return (1 + 10 ** (y0) * R_orbit ** (idx)) * approx_simp
        ## Approx 3: Fit with log(r) vs log(err)
        ## assuming α(x) to the second order
        case 3:
            idx = -0.5042733754506686
            y0 = -0.2727560615461613
            return (1 + 10 ** (y0) * R_orbit ** (idx)) * approx_simp


def find_quadratic_roots(a, b, c):
    delta = np.sqrt(b**2 - 4 * a * c)
    return (-b + delta) / (2 * a), (-b - delta) / (2 * a)


def get_phi_L(iota, R_orbit, src_pos, theta_E, D_l, M_lens):
    """
    Lens position in the source frame (origin is at source),
    defined as π minus the angle between lens position and observer position.
    iota: Inclination angle [rad]
    R_orbit: Orbital radius of BBH about AGN [R_Sch]
    src_pos: Dimensionless source position [Einstein radius]
    theta_E: Einstein radius [rad]
    D_l: Source distance [Gpc]
    M_lens: Lens mass [M_sun]
    """
    # Convert source position into units of R_Sch
    src_pos_in_rad = src_pos * theta_E
    src_pos_in_Gpc = src_pos_in_rad * D_l
    src_pos_in_R_Sch = get_R_Sch_from_Gpc(src_pos_in_Gpc, M_lens)

    cos_phi_L = -np.sqrt(R_orbit**2 - src_pos_in_R_Sch**2) / (R_orbit * np.sin(iota))
    return np.arccos(cos_phi_L)


def _sqrt_term(iota, phi_L):
    angle_sq = np.cos(iota) ** 2 + np.sin(iota) ** 2 * np.sin(phi_L) ** 2
    return np.sqrt(angle_sq)


# def _get_cos_phi_proj(iota, phi_L):
#     '''
#     Compute projection from orbital plane onto lensing plane

#     iota -- Inclination, Unit: radian
#     phi_L -- Azimuthal angle of the lens?, unit: radian
#     '''
#     return np.sin(iota) * np.sin(phi_L) / _sqrt_term(iota, phi_L)


# def get_image_iota(iota, phi_L, alpha_hat, theta_1, theta_2, beta): # angle between total angular momentum and observer position
#     # the formula used here was derived for inclination angle, not theta_jn! need to fix this!!
#     common_term = 1 / _sqrt_term(iota, phi_L)
#     correction = common_term * (np.sin(iota) * np.cos(iota) * np.cos(phi_L))
#     return np.arccos(np.cos(iota) - (alpha_hat - theta_1 + beta) * correction), np.arccos(np.cos(iota) + (alpha_hat - theta_2 - beta) * correction)


# def get_image_Phicoal(iota, phi_L, Phicoal, alpha_hat, theta_1, theta_2, beta): # coalescence phase
#     common_term = 1 / _sqrt_term(iota, phi_L)
#     correction = common_term * (np.sin(Phicoal) * (1 / np.sin(iota)) * np.sin(phi_L))
#     return np.arccos(np.cos(Phicoal) + (alpha_hat - theta_1 + beta) * correction), np.arccos(np.cos(Phicoal) - (alpha_hat - theta_2 - beta) * correction)


# def get_image_psi(iota, phi_L, psi, alpha_hat, theta_1, theta_2, beta): # polarization angle
#     common_term = 1 / _sqrt_term(iota, phi_L)
#     correction = common_term * ((1 / np.tan(iota)) * np.sin(phi_L) * np.sin(psi))
#     return np.arccos(np.cos(psi) - (alpha_hat - theta_1 + beta) * correction), np.arccos(np.cos(psi) + (alpha_hat - theta_2 - beta) * correction)


def get_lensing_induced_cosine_shifts(iota, phi_L, R_orbit, phi_coal, psi):
    """
    Absolute shift = angular factor * cosine factor.
    This function calculates cosine factor, which solely depends on which angle we're shifting (iota, Phicoal, or psi),
    while the angular factor differentiates between the two images.
    Redshifts induced by environmental effects are also calculated, including orbit-induced redshift and gravitational redshift.
    """
    sqrt_term = _sqrt_term(iota, phi_L)
    common_term = 1 / _sqrt_term(iota, phi_L)

    delta_cos_iota = common_term * (np.sin(iota) * np.cos(iota) * np.cos(phi_L))
    delta_cos_phi = common_term * (
        np.sin(phi_coal) * (1 / np.sin(iota)) * np.sin(phi_L)
    )
    delta_cos_psi = common_term * ((1 / np.tan(iota)) * np.sin(phi_L) * np.sin(psi))

    cos_phi_proj = np.sin(iota) * np.sin(phi_L) / sqrt_term
    z_orbit = 2 * cos_phi_proj / R_orbit

    z_grav = np.sqrt(1 - 1 / R_orbit) - 1

    return delta_cos_iota, delta_cos_phi, delta_cos_psi, z_orbit, z_grav


def _new_angle_from_lensing_shift(
    angle, delta_cos, alpha_hat, theta_E, src_pos, im_pos_1, im_pos_2
):
    """
    Alpha_hat and theta_E in rad, source and image positions in units of Einstein radius.
    """
    # convert source and image positions into rad
    src_pos_in_rad = src_pos * theta_E
    im_pos_1_in_rad = im_pos_1 * theta_E
    im_pos_2_in_rad = im_pos_2 * theta_E

    # angular factors for the two images
    gamma_1 = alpha_hat - im_pos_1_in_rad + src_pos_in_rad
    gamma_2 = alpha_hat - im_pos_2_in_rad - src_pos_in_rad

    return np.arccos(np.cos(angle) + gamma_1 * delta_cos), np.arccos(
        np.cos(angle) - gamma_2 * delta_cos
    )


def get_lensed_parameter_sets(
    unlensed_bbh_params, R_orbit=None, M_lz=None, src_pos=None
):
    # First make sure we have the needed parameters.
    # If not given, try look for them in the params dict:
    if R_orbit is None:
        R_orbit = unlensed_bbh_params.get("R_orbit", None)
    if M_lz is None:
        M_lz = unlensed_bbh_params.get("M_lz", None)
    if src_pos is None:
        src_pos = unlensed_bbh_params.get("src_pos", None)
    if (R_orbit is None) or (M_lz is None) or (src_pos is None):
        raise IOError(
            "Insufficient lensing parameters (R_orbit, M_lz or src_pos not provided)."
        )

    # Initialise the output dictionaries
    image_1_params = unlensed_bbh_params.copy()
    image_2_params = unlensed_bbh_params.copy()

    # Casting the angles into real
    iota = unlensed_bbh_params["iota"].real
    Phicoal = unlensed_bbh_params["Phicoal"].real
    psi = unlensed_bbh_params["psi"].real
    dL = unlensed_bbh_params["dL"].real

    # Compute non-redshifted lens mass
    z = np.interp(dL, dLGridGlob, zGridGlob)
    M_lens = M_lz / (1 + z)

    # Compute image positions
    theta_E = einstein_radius(
        M_lens, dL, R_orbit
    )  # rad, used later to convert dimensionless positions into radians
    im_pos_1, im_pos_2 = get_im_pos(src_pos)  # in units of Einstein radius

    phi_L = get_phi_L(iota, R_orbit, src_pos, theta_E, dL, M_lens)

    # Get the change in parameters
    delta_cos_iota, delta_cos_phi, delta_cos_psi, z_orbit, z_grav = (
        get_lensing_induced_cosine_shifts(iota, phi_L, R_orbit, Phicoal, psi)
    )

    # Environemental effects (orbit-induced redshift and gravitational redshift) can be modeled as changes in effective chirp mass and effective luminosity distance
    # https://arxiv.org/abs/2310.16025 Eqs. 4&5
    image_1_params["Mc"] *= (1 + z_orbit) * (1 + z_grav)
    image_2_params["Mc"] *= (1 - z_orbit) * (1 + z_grav)
    image_1_params["dL"] *= (1 + z_orbit) ** 2 * (1 + z_grav)
    image_2_params["dL"] *= (1 - z_orbit) ** 2 * (1 + z_grav)

    alpha_hat = _get_alpha_hat(R_orbit)  # rad

    src_pos_in_rad = src_pos * theta_E
    im_pos_1_in_rad = im_pos_1 * theta_E
    im_pos_2_in_rad = im_pos_2 * theta_E

    # angular factors for the two images
    gamma_1 = alpha_hat - im_pos_1_in_rad + src_pos_in_rad
    gamma_2 = alpha_hat - im_pos_2_in_rad - src_pos_in_rad

    image_1_params["iota"] = np.arccos(np.cos(iota) - gamma_1 * delta_cos_iota)
    image_2_params["iota"] = np.arccos(np.cos(iota) + gamma_2 * delta_cos_iota)

    image_1_params["Phicoal"] = np.arccos(np.cos(Phicoal) + gamma_1 * delta_cos_phi)
    image_2_params["Phicoal"] = np.arccos(np.cos(Phicoal) - gamma_2 * delta_cos_phi)

    image_1_params["psi"] = np.arccos(np.cos(psi) - gamma_1 * delta_cos_psi)
    image_2_params["psi"] = np.arccos(np.cos(psi) + gamma_2 * delta_cos_psi)

    # what does this do?
    # if cplx_return:
    #     image_1_params = {key: value.astype('complex128') for key, value in image_1_params.items()}
    #     image_2_params = {key: value.astype('complex128') for key, value in image_2_params.items()}

    return image_1_params, image_2_params


def get_lensing_time_delay(unlensed_bbh_params, M_lz=None, src_pos=None):
    """
    Time difference between the two images in seconds, using point mass lens model.
    M_lz: Redshifted lens mass [M_sun]
    src_pos: Source position [Einstein radius]
    """
    # First make sure we have the needed parameters.
    # If not given, try look for them in the params dict:
    if M_lz is None:
        M_lz = unlensed_bbh_params.get("M_lz", None)
    if src_pos is None:
        src_pos = unlensed_bbh_params.get("src_pos", None)
    if (M_lz is None) or (src_pos is None):
        print(
            "Insufficient parameters (M_lz or src_pos not given). Time delay cannot be calculated."
        )
        return 0

    M_lz_in_s = M_lz * MTSUN_SI
    sqrt_term = np.sqrt(src_pos**2 + 4)
    log_diff = np.log((sqrt_term + src_pos) / (sqrt_term - src_pos))
    return 4 * M_lz_in_s * (src_pos * sqrt_term / 2 + log_diff)


def get_mag_factors(unlensed_bbh_params, src_pos=None):
    """
    Magnification factor for both images, using point mass lens model.
    src_pos: Source position [Einstein radius]
    """
    if src_pos is None:
        src_pos = unlensed_bbh_params.get("src_pos", None)

        if src_pos is None:
            print("Source position not provided, returning 1 as magnification.")
            return 1, 1

    sq_src_pos = np.square(src_pos)
    common_term = (sq_src_pos + 2) / (2 * src_pos * np.sqrt(sq_src_pos + 4))
    mag_1, mag_2 = 0.5 + common_term, 0.5 - common_term

    return mag_1, mag_2


def lens(unlensed_bbh_params):
    """
    Print lensed parameter sets, magnification factors, and time delay.
    """
    image_1_params, image_2_params = get_lensed_parameter_sets(unlensed_bbh_params)
    mag_1, mag_2 = get_mag_factors(unlensed_bbh_params)
    time_delay = get_lensing_time_delay(unlensed_bbh_params)
    print(
        "Image 1 parameters: %s \nImage 2 parameters: %s \nMagnification factors: %s, %s \nTime delay: %s s"
        % (image_1_params, image_2_params, mag_1, mag_2, time_delay)
    )

    iota = unlensed_bbh_params["iota"]
    dL = unlensed_bbh_params["dL"]
    R_orbit = unlensed_bbh_params["R_orbit"]
    M_lz = unlensed_bbh_params["R_orbit"]

    z = np.interp(dL, dLGridGlob, zGridGlob)
    M_lens = M_lz / (1 + z)

    R_orbit_in_Gpc = get_Gpc_from_R_Sch(R_orbit, M_lens)
    R_orbit_in_rad = R_orbit_in_Gpc / dL
    theta_E = einstein_radius(M_lens, dL, R_orbit)  # rad
    min_src_pos = R_orbit_in_rad * np.abs(np.cos(iota)) / theta_E
    print("Minimum source position: %s" % (min_src_pos))
    return


##############################################################################
# Compute angle changes with vectors
##############################################################################
def compute_opening_angles(
    agn_bbh_system_params
):
    pass


def compute_exact_lensed_angles(agn_bbh_system_params):
    '''
    In the following, all vectors will take shape (3, N),
    where N is the number of samples.

    This follows the convention in the paper draft at the moment.

    We abbreviate the frames as follows:
    - Source frame: `_src`
    - Lens plane frame: `_lens`
    - Wave frame: `_wav`
    '''
    iota = agn_bbh_system_params["iota"]
    phase = agn_bbh_system_params["Phicoal"]
    # phi_L = agn_bbh_system_params["phi_L"]
    r_orbit = agn_bbh_system_params["R_orbit"]  # R_Sch
    luminosity_distance = agn_bbh_system_params["dL"]  # Gpc
    lens_mass = agn_bbh_system_params["M_lz"]  # Gpc
    src_pos_y_r = agn_bbh_system_params["src_pos"]  # R_orbit
    zeros = np.zeros_like(iota)
    L_hat_src = np.array([zeros, zeros, zeros + 1])

    # Converting new src_pos to theta_E unit
    z = np.interp(luminosity_distance, dLGridGlob, zGridGlob)
    lens_mass_src = lens_mass / (1 + z)
    theta_E = einstein_radius(lens_mass_src, luminosity_distance, r_orbit)
    R_Sch = 2 * lens_mass_src * MRSUN_SI  # m
    delta = R_Sch / uGpc
    beta = src_pos_y_r * r_orbit / luminosity_distance * delta  # radian
    src_pos_y = beta / theta_E

    # rad, used later to convert dimensionless positions into radians
    _im_pos_1, _im_pos_2 = get_im_pos(src_pos_y)  # in units of Einstein radius

    img_pos_1 = _im_pos_1 * theta_E
    img_pos_2 = _im_pos_2 * theta_E
    src_pos_y_rad = src_pos_y * theta_E

    # TODO: Update this alpha calculations
    alpha_hat = _get_alpha_hat(r_orbit)  # rad
    theta_bar_p = alpha_hat - img_pos_1 + src_pos_y_rad
    theta_bar_m = alpha_hat - img_pos_2 - src_pos_y_rad

    phi_L = agn_bbh_system_params.get(
            'phi_L', get_phi_L(iota, r_orbit, src_pos_y, theta_E, luminosity_distance, lens_mass_src))

    obs_pos = np.array([-np.sin(iota), zeros, np.cos(iota)])
    lens_pos = np.array([np.cos(phi_L), np.sin(phi_L), zeros])

    lens_pln_x = obs_pos
    lens_pln_z = np.cross(lens_pos, obs_pos, axis=0)
    lens_pln_z /= np.linalg.norm(lens_pln_z, axis=0)
    lens_pln_y = np.cross(lens_pln_z, lens_pln_x, axis=0)
    lens_pln_y /= np.linalg.norm(lens_pln_y, axis=0)
    lens_pln_frame = np.array([lens_pln_x, lens_pln_y, lens_pln_z])

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
    # First define the frame
    img_p_hat_y_src = np.cross(img_p_hat_src, L_hat_src, axis=0)
    img_p_hat_y_src /= np.linalg.norm(img_p_hat_y_src, axis=0)
    img_p_hat_x_src = np.cross(img_p_hat_y_src, L_hat_src, axis=0)

    img_m_hat_y_src = np.cross(img_m_hat_src, L_hat_src, axis=0)
    img_m_hat_y_src /= np.linalg.norm(img_m_hat_y_src, axis=0)
    img_m_hat_x_src = np.cross(img_m_hat_y_src, L_hat_src, axis=0)

    # Define the line-of-separation vector
    n_coal = np.array([np.sin(phase), np.cos(phase), zeros])
    coal_dot_img_p_x = np.einsum('ij,ij->j', n_coal, img_p_hat_x_src)
    coal_dot_img_p_y = np.einsum('ij,ij->j', n_coal, img_p_hat_y_src)
    _phi_p = np.arctan2(coal_dot_img_p_y, coal_dot_img_p_x)
    _phi_p = np.arccos(coal_dot_img_p_y)

    coal_dot_img_m_x = np.einsum('ij,ij->j', n_coal, img_m_hat_x_src)
    coal_dot_img_m_y = np.einsum('ij,ij->j', n_coal, img_m_hat_y_src)
    _phi_m = np.arctan2(coal_dot_img_m_y, coal_dot_img_m_x)
    _phi_m = np.arccos(coal_dot_img_m_y)

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
