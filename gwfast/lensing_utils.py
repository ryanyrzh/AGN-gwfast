from jax import config

config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp

from astropy.cosmology import Planck18 as cosmo

zGridGlob = np.logspace(start=-6, stop=5, base=10, num=5000)
dLGridGlob = cosmo.luminosity_distance(zGridGlob).value / 1000.0

# Constants in SI
# TODO: Use the constants defined in gwfastGlobals
G = 6.6743 * 1e-11
c = 2.979246 * 1e8
M_sun = 2.9884 * 1e30
Gpc = 3.0856776 * 1e25


##############################################################################
# LENSING
##############################################################################
def theta_in_terms_of_x_0(x_0, D_l):
    """
    x_0: Minimal approach distance [R_Sch]
    D_l: Lens distance [R_Sch]
    """
    return x_0 / (D_l * jnp.sqrt(1 - 1 / x_0))


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
    x2_coef = (15 / 16) * jnp.pi - 1
    x2_coef = 0
    return 2 / x_0 + x2_coef / (x_0**2)


# def einstein_radius(D_ls, D_l):
#     '''
#     The Einstein radius
#     D_ls:   Lens-source plane distance [R_Sch]
#     D_l:    Lens distance [R_Sch]
#     '''
#     D_ratio = (1 + D_ls / D_l) * D_l**2
#     return jnp.sqrt(2 * D_ratio)


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
    z = jnp.interp(dL, dLGridGlob, zGridGlob)
    D_l_in_Gpc = dL / (1 + z) ** 2
    D_term = D_ls_in_Gpc / D_l_in_Gpc**2  # Approximating D_s=D_s
    einstein_radius_in_arcsec = jnp.sqrt(M_term * D_term)
    einstein_radius_in_rad = einstein_radius_in_arcsec * (1 / 3600) * (jnp.pi / 180)
    return einstein_radius_in_rad


def get_Gpc_from_R_Sch(qty, M):
    """
    Convert from units of Schwarzschild radius to Gpc for a given mass.
    qty: Distance to be converted [R_Sch]
    M: Mass corresponding to the Schwarzschild radius [M_sun]
    """
    M_in_kg = M * M_sun
    one_R_Sch = 2 * G * M_in_kg / c**2  # m
    qty_in_m = qty * one_R_Sch
    qty_in_Gpc = qty_in_m / Gpc
    return qty_in_Gpc


def get_R_Sch_from_Gpc(qty, M):
    """
    Convert from Gpc to units of Schwarzschild radius of a given mass
    qty: Distance to be converted [Gpc]
    M: Mass for which to calculate R_Sch [M_sun]
    """
    M_in_kg = M * M_sun
    R_Sch_in_m = 2 * G * M_in_kg / c**2
    R_Sch_in_Gpc = R_Sch_in_m / Gpc
    return qty / R_Sch_in_Gpc


def get_im_pos(src_pos):
    """
    Image positions, normalized by Einstein radius.
    src_pos: Source position [Einstein radius]
    """
    sqrt_term = jnp.sqrt(src_pos**2 / 4 + 1)
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
    approx_simp = jnp.sqrt(2 / R_orbit)
    # what do these cases mean?
    # should we do this or use the analytic expression without small angle assumptions?
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
        ## assuming α(x) to the send order
        case 3:
            idx = -0.5042733754506686
            y0 = -0.2727560615461613
            return (1 + 10 ** (y0) * R_orbit ** (idx)) * approx_simp


def find_quadratic_roots(a, b, c):
    delta = jnp.sqrt(b**2 - 4 * a * c)
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

    cos_phi_L = -jnp.sqrt(R_orbit**2 - src_pos_in_R_Sch**2) / (R_orbit * jnp.sin(iota))
    return jnp.arccos(cos_phi_L)


def _sqrt_term(iota, phi_L):
    angle_sq = jnp.cos(iota) ** 2 + jnp.sin(iota) ** 2 * jnp.sin(phi_L) ** 2
    return jnp.sqrt(angle_sq)


# def _get_cos_phi_proj(iota, phi_L):
#     '''
#     Compute projection from orbital plane onto lensing plane

#     iota -- Inclination, Unit: radian
#     phi_L -- Azimuthal angle of the lens?, unit: radian
#     '''
#     return jnp.sin(iota) * jnp.sin(phi_L) / _sqrt_term(iota, phi_L)


# def get_image_iota(iota, phi_L, alpha_hat, theta_1, theta_2, beta): # angle between total angular momentum and observer position
#     # the formula used here was derived for inclination angle, not theta_jn! need to fix this!!
#     common_term = 1 / _sqrt_term(iota, phi_L)
#     correction = common_term * (jnp.sin(iota) * jnp.cos(iota) * jnp.cos(phi_L))
#     return jnp.arccos(jnp.cos(iota) - (alpha_hat - theta_1 + beta) * correction), jnp.arccos(jnp.cos(iota) + (alpha_hat - theta_2 - beta) * correction)


# def get_image_Phicoal(iota, phi_L, Phicoal, alpha_hat, theta_1, theta_2, beta): # coalescence phase
#     common_term = 1 / _sqrt_term(iota, phi_L)
#     correction = common_term * (jnp.sin(Phicoal) * (1 / jnp.sin(iota)) * jnp.sin(phi_L))
#     return jnp.arccos(jnp.cos(Phicoal) + (alpha_hat - theta_1 + beta) * correction), jnp.arccos(jnp.cos(Phicoal) - (alpha_hat - theta_2 - beta) * correction)


# def get_image_psi(iota, phi_L, psi, alpha_hat, theta_1, theta_2, beta): # polarization angle
#     common_term = 1 / _sqrt_term(iota, phi_L)
#     correction = common_term * ((1 / jnp.tan(iota)) * jnp.sin(phi_L) * jnp.sin(psi))
#     return jnp.arccos(jnp.cos(psi) - (alpha_hat - theta_1 + beta) * correction), jnp.arccos(jnp.cos(psi) + (alpha_hat - theta_2 - beta) * correction)


def get_lensing_induced_cosine_shifts(iota, phi_L, R_orbit, phi_coal, psi):
    """
    Absolute shift = angular factor * cosine factor.
    This function calculates cosine factor, which solely depends on which angle we're shifting (iota, Phicoal, or psi),
    while the angular factor differentiates between the two images.
    Redshifts induced by environmental effects are also calculated, including orbit-induced redshift and gravitational redshift.
    """
    sqrt_term = _sqrt_term(iota, phi_L)
    common_term = 1 / _sqrt_term(iota, phi_L)

    delta_cos_iota = common_term * (jnp.sin(iota) * jnp.cos(iota) * jnp.cos(phi_L))
    delta_cos_phi = common_term * (
        jnp.sin(phi_coal) * (1 / jnp.sin(iota)) * jnp.sin(phi_L)
    )
    delta_cos_psi = common_term * ((1 / jnp.tan(iota)) * jnp.sin(phi_L) * jnp.sin(psi))

    cos_phi_proj = jnp.sin(iota) * jnp.sin(phi_L) / sqrt_term
    z_orbit = 2 * cos_phi_proj / R_orbit

    z_grav = jnp.sqrt(1 - 1 / R_orbit) - 1

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

    return jnp.arccos(jnp.cos(angle) + gamma_1 * delta_cos), jnp.arccos(
        jnp.cos(angle) - gamma_2 * delta_cos
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
    iota = unlensed_bbh_params["iota"].real.astype("float64")
    Phicoal = unlensed_bbh_params["Phicoal"].real.astype("float64")
    psi = unlensed_bbh_params["psi"].real.astype("float64")
    dL = unlensed_bbh_params["dL"].real.astype("float64")

    # Compute non-redshifted lens mass
    z = jnp.interp(dL, dLGridGlob, zGridGlob)
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

    image_1_params["iota"], image_2_params["iota"] = _new_angle_from_lensing_shift(
        iota, -delta_cos_iota, alpha_hat, theta_E, src_pos, im_pos_1, im_pos_2
    )
    image_1_params["Phicoal"], image_2_params["Phicoal"] = (
        _new_angle_from_lensing_shift(
            Phicoal, +delta_cos_phi, alpha_hat, theta_E, src_pos, im_pos_1, im_pos_2
        )
    )
    image_1_params["psi"], image_2_params["psi"] = _new_angle_from_lensing_shift(
        psi, -delta_cos_psi, alpha_hat, theta_E, src_pos, im_pos_1, im_pos_2
    )

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

    M_lz_in_kg = M_lz * M_sun
    M_lz_in_s = M_lz_in_kg * G / c**3

    time_delay = (
        4
        * M_lz_in_s
        * (
            src_pos * jnp.sqrt(src_pos**2 + 4) / 2
            + jnp.log(
                (jnp.sqrt(src_pos**2 + 4) + src_pos)
                / (jnp.sqrt(src_pos**2 + 4) - src_pos)
            )
        )
    )
    return time_delay


def get_mag_factors(unlensed_bbh_params, src_pos=None):
    """
    Magnification factor for both images, using point mass lens model.
    src_pos: Source position [Einstein radius]
    """
    if src_pos is None:
        src_pos = unlensed_bbh_params.get("src_pos", None)
    if src_pos is None:
        print(
            "Source position not provided. Magnification factors will not be calculated."
        )
        return 1, 1

    common_term = (src_pos**2 + 2) / (2 * src_pos * jnp.sqrt(src_pos**2 + 4))
    mag_1, mag_2 = 1 / 2 + common_term, 1 / 2 - common_term

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

    z = jnp.interp(dL, dLGridGlob, zGridGlob)
    M_lens = M_lz / (1 + z)

    R_orbit_in_Gpc = get_Gpc_from_R_Sch(R_orbit, M_lens)
    R_orbit_in_rad = R_orbit_in_Gpc / dL
    theta_E = einstein_radius(M_lens, dL, R_orbit)  # rad
    min_src_pos = R_orbit_in_rad * jnp.abs(jnp.cos(iota)) / theta_E
    print("Minimum source position: %s" % (min_src_pos))
    return
