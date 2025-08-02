#
#    Copyright (c) 2022 Francesco Iacovelli <francesco.iacovelli@unige.ch>, Michele Mancarella <michele.mancarella@unige.ch>
#    Copyright (c) 2025 Samson Leong <samson.leong@link.cuhk.edu.hk>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.

import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
import numpy as np
import copy

try:
    np.float128(1.0)
    typeuse = "float128"
except AttributeError:
    print(
        "WARNING: numpy float128 type not supported on this machine, resorting to float64, precision might be lower."
    )
    typeuse = "float64"

##############################################################################
# DERIVATIVES AND JACOBIANS
##############################################################################


def log_dL_to_dL_derivative_cov(or_matrix, ParNums, evParams):
    """
    Change variables in the covariance matrix from :math:`{\\rm log}(d_L)` to :math:`d_L`.

    :param array or_matrix: Array containing the covariance matrix, of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters})`.
    :param dict(int) ParNums: Dictionary specifying the position of each parameter in the Fisher matrix, as :py:class:`gwfast.waveforms.WaveFormModel.ParNums`.
    :param dict(array, array, ...) evParams: Dictionary containing the parameters of the event(s), as in :py:data:`events`.

    :return: Covariance matrix in :math:`d_L`, of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters})`.
    :rtype: 2-D array

    """
    matrix = copy.deepcopy(or_matrix)
    # for i in range(matrix.shape[-1]):
    # This has to be vectorised
    try:
        matrix = matrix.at[:, ParNums["dL"], :].set(
            matrix[:, ParNums["dL"], :] * evParams["dL"]
        )
        matrix = matrix.at[ParNums["dL"], :, :].set(
            matrix[ParNums["dL"], :, :] * evParams["dL"]
        )
    except AttributeError:
        matrix = matrix.astype(typeuse)
        matrix[:, ParNums["dL"], :] *= evParams["dL"].astype(typeuse)
        matrix[ParNums["dL"], :, :] *= evParams["dL"].astype(typeuse)
    return matrix


def log_dL_to_dL_derivative_fish(or_matrix, ParNums, evParams):
    """
    Change variables in the Fisher matrix from :math:`{\\rm log}(d_L)` to :math:`d_L`.

    :param array or_matrix: Array containing the Fisher matrix(ces), of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters})`.
    :param dict(int) ParNums: Dictionary specifying the position of each parameter in the Fisher matrix, as :py:class:`gwfast.waveforms.WaveFormModel.ParNums`.
    :param dict(array, array, ...) evParams: Dictionary containing the parameters of the event(s), as in :py:data:`events`.

    :return: Fisher matrix in :math:`d_L`, of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters})`.
    :rtype: 2-D array

    """
    matrix = copy.deepcopy(or_matrix)
    # for i in range(matrix.shape[-1]):
    # This has to be vectorised
    try:
        matrix = matrix.at[:, ParNums["dL"], :].set(
            matrix[:, ParNums["dL"], :] / evParams["dL"]
        )
        matrix = matrix.at[ParNums["dL"], :, :].set(
            matrix[ParNums["dL"], :, :] / evParams["dL"]
        )
    except AttributeError:
        matrix = matrix.astype(typeuse)
        matrix[:, ParNums["dL"], :] /= evParams["dL"].astype(typeuse)
        matrix[ParNums["dL"], :, :] /= evParams["dL"].astype(typeuse)
    return matrix


def dm1_dMc(eta):
    """
    Compute the derivative of :math:`m_1` with respect to :math:`{\cal M}_c`.

    :param array eta: The symmetric mass ratio(s), :math:`\eta`, of the objects.

    :return: :math:`\partial m_1/\partial {\cal M}_c`.
    :rtype: 1-D array

    """
    return (1 + np.sqrt(1 - 4 * eta)) * eta ** (-3.0 / 5.0) / 2


def dm2_dMc(eta):
    """
    Compute the derivative of :math:`m_2` with respect to :math:`{\cal M}_c`.

    :param array eta: The symmetric mass ratio(s), :math:`\eta`, of the objects.

    :return: :math:`\partial m_2/\partial {\cal M}_c`.
    :rtype: 1-D array

    """
    return (1 - np.sqrt(1 - 4 * eta)) * eta ** (-3.0 / 5.0) / 2


def dm1_deta(Mc, eta):
    """
    Compute the derivative of :math:`m_1` with respect to :math:`\eta`.

    :param array or float Mc: Chirp mass of the binary, :math:`{\cal M}_c`, in units of :math:`\\rm M_{\odot}`.
    :param array eta: The symmetric mass ratio(s), :math:`\eta`, of the objects.

    :return: :math:`\partial m_1/\partial \eta`.
    :rtype: 1-D array

    """
    return (
        -Mc
        * (3 - 2 * eta + 3 * np.sqrt(1 - 4 * eta))
        / (10 * np.sqrt(1 - 4 * eta) * eta ** (8.0 / 5.0))
    )


# (1-np.sqrt(1-4*eta) )*eta**(-3./5.)/2


def dm2_deta(Mc, eta):
    """
    Compute the derivative of :math:`m_2` with respect to :math:`\eta`.

    :param array or float Mc: Chirp mass of the binary, :math:`{\cal M}_c`, in units of :math:`\\rm M_{\odot}`.
    :param array eta: The symmetric mass ratio(s), :math:`\eta`, of the objects.

    :return: :math:`\partial m_2/\partial \eta`.
    :rtype: 1-D array

    """
    return (
        -Mc
        * (-3 + 2 * eta + 3 * np.sqrt(1 - 4 * eta))
        / (10 * np.sqrt(1 - 4 * eta) * eta ** (8.0 / 5.0))
    )


def dMc_dm1(m1, m2):
    """
    Compute the derivative of :math:`{\cal M}_c` with respect to :math:`m_1`.

    :param array or float m1: Mass of the first object, :math:`m_1`, in units of :math:`\\rm M_{\odot}`.
    :param array or float m2: Mass of the second object, :math:`m_2`, in units of :math:`\\rm M_{\odot}`.

    :return: :math:`\partial {\cal M}_c/\partial m_1`.
    :rtype: 1-D array

    """
    return m2 * (2 * m1 + 3 * m2) / (5 * (m1 * m2) ** (2 / 5) * (m1 + m2) ** (6 / 5))


def dMc_dm2(m1, m2):
    """
    Compute the derivative of :math:`{\cal M}_c` with respect to :math:`m_2`.

    :param array or float m1: Mass of the first object, :math:`m_1`, in units of :math:`\\rm M_{\odot}`.
    :param array or float m2: Mass of the second object, :math:`m_2`, in units of :math:`\\rm M_{\odot}`.

    :return: :math:`\partial {\cal M}_c/\partial m_2`.
    :rtype: 1-D array

    """
    return dMc_dm1(m2, m1)


def deta_dm1(m1, m2):
    """
    Compute the derivative of :math:`\eta` with respect to :math:`m_1`.

    :param array or float m1: Mass of the first object, :math:`m_1`, in units of :math:`\\rm M_{\odot}`.
    :param array or float m2: Mass of the second object, :math:`m_2`, in units of :math:`\\rm M_{\odot}`.

    :return: :math:`\partial \eta/\partial m_1`.
    :rtype: 1-D array

    """
    return m2 * (m2 - m1) / (m1 + m2) ** 3


def deta_dm2(m1, m2):
    """
    Compute the derivative of :math:`\eta` with respect to :math:`m_2`.

    :param array or float m1: Mass of the first object, :math:`m_1`, in units of :math:`\\rm M_{\odot}`.
    :param array or float m2: Mass of the second object, :math:`m_2`, in units of :math:`\\rm M_{\odot}`.

    :return: :math:`\partial \eta/\partial m_2`.
    :rtype: 1-D array

    """
    return deta_dm1(m2, m1)


def J_m1m2_Mceta(Mc, eta):
    """
    Compute the Jacobian matrix from :math:`{\cal M}_c` and :math:`\eta` to :math:`m_1` and :math:`m_2`.

    :param array or float Mc: Chirp mass of the binary, :math:`{\cal M}_c`, in units of :math:`\\rm M_{\odot}`.
    :param array eta: The symmetric mass ratio(s), :math:`\eta`, of the objects.

    :return: :math:`\partial (m_1, m_2)/\partial ({\cal M}_c, \eta)`.
    :rtype: 2-D array

    """
    return np.array(
        [[dm1_dMc(eta), dm1_deta(Mc, eta)], [dm2_dMc(eta), dm2_deta(Mc, eta)]]
    )


def J_Mceta_m1m2(m1, m2):
    """
    Compute the Jacobian matrix from :math:`m_1` and :math:`m_2` to :math:`{\cal M}_c` and :math:`\eta`.

    :param array or float m1: Mass of the first object, :math:`m_1`, in units of :math:`\\rm M_{\odot}`.
    :param array or float m2: Mass of the second object, :math:`m_2`, in units of :math:`\\rm M_{\odot}`.

    :return: :math:`\partial ({\cal M}_c, \eta)/\partial (m_1, m_2)`.
    :rtype: 2-D array

    """
    return np.array(
        [[dMc_dm1(m1, m2), dMc_dm2(m1, m2)], [deta_dm1(m1, m2), deta_dm2(m1, m2)]]
    )


def m1m2_from_Mceta(Mc, eta):
    """
    Compute the component masses of a binary given its chirp mass and symmetric mass ratio.

    :param array or float Mc: Chirp mass of the binary, :math:`{\cal M}_c`.
    :param array or float eta: The symmetric mass ratio(s), :math:`\eta`, of the objects.
    :return: :math:`m_1` and :math:`m_2`.
    :rtype: tuple(array, array) or tuple(float, float)

    """
    delta = 1 - 4 * eta
    return (1 + np.sqrt(delta)) / 2 * Mc / eta ** (3.0 / 5.0), (
        1 - np.sqrt(delta)
    ) / 2 * Mc / eta ** (3.0 / 5.0)


def Mceta_from_m1m2(m1, m2):
    """
    Compute the chirp mass and symmetric mass ratio of a binary given its component masses.

    :param array or float m1: Mass of the first object, :math:`m_1`.
    :param array or float m2: Mass of the second object, :math:`m_2`.
    :return: :math:`{\cal M}_c` and :math:`\eta`.
    :rtype: tuple(array, array) or tuple(float, float)

    """
    Mc = (m1 * m2) ** 3 / 5 / (m1 + m2) ** 1 / 5
    eta = (m1 * m2) / (m1 + m2) ** 2
    return Mc, eta


def m1m2_to_Mceta_fish(or_matrix, ParNums, evParams):
    """
    Change variables in the Fisher matrix from :math:`m_1` and :math:`m_2` to :math:`{\cal M}_c` and :math:`\eta`.

    :param array or_matrix: Array containing the Fisher matrix(ces), of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters})`.
    :param dict(int) ParNums: Dictionary specifying the position of each parameter in the Fisher matrix, as :py:class:`gwfast.waveforms.WaveFormModel.ParNums`.
    :param dict(array, array, ...) evParams: Dictionary containing the parameters of the event(s), as in :py:data:`events`.

    :return: Fisher matrix in :math:`{\cal M}_c` and :math:`\eta`, of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters})`.
    :rtype: 2-D array

    """
    nparams = len(list(ParNums.keys()))

    rotMatrix = np.identity(nparams)

    rotMatrix[
        np.ix_([ParNums["Mc"], ParNums["eta"]], [ParNums["Mc"], ParNums["eta"]])
    ] = J_m1m2_Mceta(evParams["Mc"], evParams["eta"])

    matrix = rotMatrix @ or_matrix @ rotMatrix

    return matrix


def m1m2_to_Mceta_cov(or_matrix, ParNums, evParams):
    """
    Change variables in the covariance matrix from :math:`m_1` and :math:`m_2` to :math:`{\cal M}_c` and :math:`\eta`.

    :param array or_matrix: Array containing the covariance matrix(ces), of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters})`.
    :param dict(int) ParNums: Dictionary specifying the position of each parameter in the Fisher matrix, as :py:class:`gwfast.waveforms.WaveFormModel.ParNums`.
    :param dict(array, array, ...) evParams: Dictionary containing the parameters of the event(s), as in :py:data:`events`.

    :return: Covariance matrix in :math:`{\cal M}_c` and :math:`\eta`, of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters})`.
    :rtype: 2-D array

    """
    nparams = len(list(ParNums.keys()))

    rotMatrix = np.identity(nparams)

    rotMatrix[
        np.ix_([ParNums["Mc"], ParNums["eta"]], [ParNums["Mc"], ParNums["eta"]])
    ] = J_Mceta_m1m2(*m1m2_from_Mceta(evParams["Mc"], evParams["eta"]))

    matrix = rotMatrix @ or_matrix @ rotMatrix

    return matrix


def Mceta_to_m1m2_fish(or_matrix, ParNums, evParams):
    """
    Change variables in the Fisher matrix from :math:`{\cal M}_c` and :math:`\eta` to :math:`m_1` and :math:`m_2`.

    :param array or_matrix: Array containing the Fisher matrix(ces), of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters})`.
    :param dict(int) ParNums: Dictionary specifying the position of each parameter in the Fisher matrix, as :py:class:`gwfast.waveforms.WaveFormModel.ParNums`.
    :param dict(array, array, ...) evParams: Dictionary containing the parameters of the event(s), as in :py:data:`events`.

    :return: Fisher matrix in :math:`m_1` and :math:`m_2`, of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters})`.
    :rtype: 2-D array

    """
    nparams = or_matrix.shape[0]  # len(list(ParNums.keys()))

    rotMatrix = np.identity(nparams)

    m1, m2 = m1m2_from_Mceta(evParams["Mc"], evParams["eta"])

    rotMatrix[
        np.ix_([ParNums["Mc"], ParNums["eta"]], [ParNums["Mc"], ParNums["eta"]])
    ] = J_Mceta_m1m2(m1, m2)

    matrix = rotMatrix.T @ or_matrix @ rotMatrix

    return matrix


def Mceta_to_m1m2_cov(or_matrix, ParNums, evParams):
    """
    Change variables in the covariance matrix from :math:`{\cal M}_c` and :math:`\eta` to :math:`m_1` and :math:`m_2`.

    :param array or_matrix: Array containing the covariance matrix(ces), of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters})`.
    :param dict(int) ParNums: Dictionary specifying the position of each parameter in the Fisher matrix, as :py:class:`gwfast.waveforms.WaveFormModel.ParNums`.
    :param dict(array, array, ...) evParams: Dictionary containing the parameters of the event(s), as in :py:data:`events`.

    :return: Covariance matrix in :math:`m_1` and :math:`m_2`, of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters})`.
    :rtype: 2-D array

    """
    nparams = or_matrix.shape[0]  # len(list(ParNums.keys()))

    rotMatrix = np.identity(nparams)

    rotMatrix[
        np.ix_([ParNums["Mc"], ParNums["eta"]], [ParNums["Mc"], ParNums["eta"]])
    ] = J_m1m2_Mceta(evParams["Mc"], evParams["eta"])

    matrix = rotMatrix.T @ or_matrix @ rotMatrix

    return matrix


def dchi1_dchieff(m1, m2):
    """
    Compute the derivative of :math:`\chi_{1,z}` with respect to :math:`\chi_{\\rm eff}`.

    :param array or float m1: Mass of the first object, :math:`m_1`, in units of :math:`\\rm M_{\odot}`.
    :param array or float m2: Mass of the second object, :math:`m_2`, in units of :math:`\\rm M_{\odot}`.

    :return: :math:`\partial \chi_{1,z}/\partial \chi_{\\rm eff}`.
    :rtype: 1-D array

    """
    return 1.0


def dchi2_dchieff(m1, m2):
    """
    Compute the derivative of :math:`\chi_{2,z}` with respect to :math:`\chi_{\\rm eff}`.

    :param array or float m1: Mass of the first object, :math:`m_1`, in units of :math:`\\rm M_{\odot}`.
    :param array or float m2: Mass of the second object, :math:`m_2`, in units of :math:`\\rm M_{\odot}`.

    :return: :math:`\partial \chi_{2,z}/\partial \chi_{\\rm eff}`.
    :rtype: 1-D array

    """
    return 1.0


def dchi1_dDelchi(m1, m2):
    """
    Compute the derivative of :math:`\chi_{1,z}` with respect to :math:`\Delta\chi`.

    :param array or float m1: Mass of the first object, :math:`m_1`, in units of :math:`\\rm M_{\odot}`.
    :param array or float m2: Mass of the second object, :math:`m_2`, in units of :math:`\\rm M_{\odot}`.

    :return: :math:`\partial \chi_{1,z}/\partial \Delta\chi`.
    :rtype: 1-D array

    """
    return m2 / (m1 + m2)


def dchi2_dDelchi(m1, m2):
    """
    Compute the derivative of :math:`\chi_{2,z}` with respect to :math:`\Delta\chi`.

    :param array or float m1: Mass of the first object, :math:`m_1`, in units of :math:`\\rm M_{\odot}`.
    :param array or float m2: Mass of the second object, :math:`m_2`, in units of :math:`\\rm M_{\odot}`.

    :return: :math:`\partial \chi_{2,z}/\partial \Delta\chi`.
    :rtype: 1-D array

    """
    return -m1 / (m1 + m2)


def J_chi1chi2_chieffDeltachi(m1, m2):
    """
    Compute the Jacobian matrix from :math:`\chi_{\\rm eff}` and :math:`\Delta\chi`` to :math:`\chi_{1,z}` and :math:`\chi_{2,z}`.

    :param array or float m1: Mass of the first object, :math:`m_1`, in units of :math:`\\rm M_{\odot}`.
    :param array or float m2: Mass of the second object, :math:`m_2`, in units of :math:`\\rm M_{\odot}`.

    :return: :math:`\partial (\chi_{1,z}, \chi_{2,z})/\partial (\chi_{\\rm eff}, \Delta\chi)`.
    :rtype: 2-D array

    """
    return np.array([[1.0, dchi1_dDelchi(m1, m2)], [1.0, dchi2_dDelchi(m1, m2)]])


def chi1chi2_to_chieffDeltachi_fish(or_matrix, ParNums, evParams):
    """
    Change variables in the Fisher matrix from :math:`\chi_{1,z}` and :math:`\chi_{2,z}` to :math:`\chi_{\\rm eff}` and :math:`\Delta\chi`.

    :param array or_matrix: Array containing the Fisher matrix(ces), of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters})`.
    :param dict(int) ParNums: Dictionary specifying the position of each parameter in the Fisher matrix, as :py:class:`gwfast.waveforms.WaveFormModel.ParNums`.
    :param dict(array, array, ...) evParams: Dictionary containing the parameters of the event(s), as in :py:data:`events`.

    :return: Fisher matrix in :math:`\chi_{\\rm eff}` and :math:`\Delta\chi`, of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters})`.
    :rtype: 2-D array

    """
    nparams = len(list(ParNums.keys()))

    rotMatrix = np.identity(nparams)

    rotMatrix[
        np.ix_(
            [ParNums["chi1z"], ParNums["chi2z"]], [ParNums["chi1z"], ParNums["chi2z"]]
        )
    ] = J_chi1chi2_chieffDeltachi(*m1m2_from_Mceta(evParams["Mc"], evParams["eta"]))

    matrix = rotMatrix.T @ or_matrix @ rotMatrix

    return matrix


def chiSchiA_to_chi1chi2_fish(or_matrix, ParNums, evParams):
    """
    Change variables in the Fisher matrix from :math:`\chi_s` and :math:`\chi_a` to :math:`\chi_{1,z}` and :math:`\chi_{2,z}`.

    :param array or_matrix: Array containing the Fisher matrix(ces), of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters})`.
    :param dict(int) ParNums: Dictionary specifying the position of each parameter in the Fisher matrix, as :py:class:`gwfast.waveforms.WaveFormModel.ParNums`.
    :param dict(array, array, ...) evParams: Dictionary containing the parameters of the event(s), as in :py:data:`events`.

    :return: Fisher matrix in :math:`\chi_{1,z}` and :math:`\chi_{2,z}`, of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters})`.
    :rtype: 2-D array

    """
    nparams = len(list(ParNums.keys()))

    rotMatrix = np.identity(nparams)

    J_chiSchiA_chi1chi2 = np.array([[0.5, 0.5], [0.5, -0.5]])

    rotMatrix[
        np.ix_([ParNums["chiS"], ParNums["chiA"]], [ParNums["chiS"], ParNums["chiA"]])
    ] = J_chiSchiA_chi1chi2

    matrix = rotMatrix @ or_matrix @ rotMatrix

    return matrix
