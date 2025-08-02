#
#    Copyright (c) 2022 Francesco Iacovelli <francesco.iacovelli@unige.ch>, Michele Mancarella <michele.mancarella@unige.ch>
#    Copyright (c) 2025 Samson Leong <samson.leong@link.cuhk.edu.hk>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.

import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
import numpy as np
import mpmath
from scipy.stats import norm
from scipy.linalg import eigh

try:
    np.float128(1.0)
    typeuse = "float128"
except AttributeError:
    print(
        "WARNING: numpy float128 type not supported on this machine, resorting to float64, precision might be lower."
    )
    typeuse = "float64"

__all__ = [
    'reduce_Fisher_matrix',
    'CheckFisher',
    'fixParams',
    'addPrior',
    'plot_contours',
    'plot_corners'
]


def reduce_Fisher_matrix(fisher_matrix, keys=None):
    """
    **Please use this function with care, and only removes zeroes that are expected.**

    This function remove the columns and rows of the Fisher matrix that gives identically zeroes.

    It will also remove the corresponding keys if provided.
    """
    remove_indices = []
    remove_keys = []
    for idx, row in enumerate(fisher_matrix):
        col = fisher_matrix[:, idx, :]
        zero_row = np.all(row == 0.0)
        zero_col = np.all(col == 0.0)

        if zero_row and zero_col:
            remove_indices.append(idx)

    reduced_fisher_mats = np.copy(fisher_matrix)
    for jdx in reversed(remove_indices):
        reduced_fisher_mats = np.delete(reduced_fisher_mats, (jdx), axis=0)
        reduced_fisher_mats = np.delete(reduced_fisher_mats, (jdx), axis=1)

        if keys is not None:
            rm_key = keys.pop(jdx)
            remove_keys.append(rm_key)

    return reduced_fisher_mats, remove_keys


def CheckFisher(FisherM, condNumbMax=1.0e15, use_mpmath=True, verbose=False):
    """
    Perform some sanity checks on the Fisher matrix, in particular:

        - compute the eigenvalues and eigenvectors;
        - compute the condition number (ratio of the largest to smallest eigenvalue) and check this is not large;

    :param array FisherM: Array containing the Fisher matrix(ces) to check, of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters}`, :math:`N_{\\rm events})`.
    :param float condNumbMax: Maximum allowed condition number, depending on the machine precision.
    :param bool, optional use_mpmath: Boolean specifying if the checks have to be performed using the `mpmath library <https://mpmath.org>`_.
    :param bool, optional verbose: Boolean specifying if the code has to print additional details during execution.

    :return: Eigenvalues, eigenvectors and condition number(s) of the input Fisher matrix(ces).
    :rtype: tuple(array, array, array)

    """

    # Being the Fisher symmetric by definition, we can use the scipy.linalg function 'eigh', to speed up a bit
    # The input has size (Npar,Npar,Nev), so we have to swap

    if not use_mpmath:
        evals, evecs = eigh(FisherM.transpose(2, 0, 1))
    else:
        evals = np.zeros(FisherM.shape[1:][::-1])
        evecs = np.zeros(FisherM.shape[::-1])
        for k in range(FisherM.shape[-1]):

            if np.all(np.isnan(FisherM[:, :, k])):
                if verbose:
                    print("Fisher is nan at position %s. " % k)
                evals[k, :] = np.full(FisherM.shape[0], np.nan)
                evecs[k, :, :] = np.full(FisherM[:, :, k].shape, np.nan)
            else:
                try:
                    aam = mpmath.matrix(FisherM[:, :, k].astype(typeuse))
                    E, ER = mpmath.eigh(aam)
                    evals[k, :] = np.array(E, dtype=typeuse)
                    evecs[k, :, :] = np.array(ER.tolist(), dtype=typeuse)
                except Exception as e:
                    print(e)
                    print("Trying with scipy")
                    try:
                        evals[k, :], evecs[k, :, :] = eigh(FisherM[:, :, k])
                    except Exception as e:
                        print(e)
                        print("Event is number %s" % k)
                        evals[k, :], evecs[k, :, :] = np.full(
                            FisherM.shape[0],
                            np.nan,
                        ), np.full(
                            (FisherM.shape[0], FisherM.shape[0]),
                            np.nan,
                        )
                        # condNumber = None
                        print(FisherM[:, :, k])

    if np.any(evals <= 0.0):
        print(
            "WARNING: one or more eigenvalues are negative at position(s) %s"
            % str(np.unique(np.where(evals < 0)[0]))
        )

    condNumber = np.abs(evals).max(axis=1) / np.abs(evals).min(axis=1)

    if np.any(condNumber > condNumbMax) and verbose:
        print(
            "WARNING: the condition number is too large (%s>%s)"
            % (condNumber, condNumbMax)
        )
        print("Unreliable covariance at positions " + str(condNumber > condNumbMax))
    elif verbose:
        print("Condition number= %s . Ok. " % condNumber)

    return evals, evecs, condNumber


##############################################################################
# ADDING PRIOR, ELIMINATING ROWS
##############################################################################


def fixParams(MatrIn, ParNums_inp, ParMarg):
    """
    Fix one or multiple parameters to their fiducial values in the Fisher matrix.

    :param array MatrIn: Array containing the Fisher matrix(ces), of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters}`, :math:`N_{\\rm events})`.
    :param dict(int) ParNums_inp: Dictionary specifying the position of each parameter in the input Fisher matrix, as :py:class:`gwfast.waveforms.WaveFormModel.ParNums`.
    :param list(str) ParMarg: List of the names of parameters to fix.

    :return: Fisher matrix with parameters fixed, of shape :math:`(\\tilde{N}_{\\rm parameters}`, :math:`\\tilde{N}_{\\rm parameters}`, :math:`N_{\\rm events})`, and dictionary specifying the position of each parameter in the new Fisher matrix. :math:`\\tilde{N}_{\\rm parameters}` is the original :math:`N_{\\rm parameters}` minus the number of parameters that have been fixed.
    :rtype: tuple(array, dict(int))

    """
    import copy

    ParNums = copy.deepcopy(ParNums_inp)

    IdxMarg = np.sort(np.array([ParNums[par] for par in ParMarg]))
    newdim = MatrIn.shape[0] - len(IdxMarg)

    NewMatr = np.full((newdim, newdim, MatrIn.shape[-1]), np.NaN)

    for k in range(MatrIn.shape[-1]):

        Matr = np.delete(MatrIn[:, :, k], IdxMarg, 0)
        Matr = np.delete(Matr[:, :], IdxMarg, 1)
        NewMatr[:, :, k] = Matr

    # Given that we deleted some rows and columns,
    # the meaning of the numbers of the remaining ones changes

    for pm in ParMarg:
        for k in ParNums.keys():
            if ParNums[k] > ParNums[pm]:
                ParNums[k] -= 1
        ParNums.pop(pm, None)

    return NewMatr, ParNums


def addPrior(Matr, vals, ParNums, ParAdd):
    """
    Add a Gaussian priors to the Fisher matrix on one or multiple parameters.

    :param array Matr: Array containing the Fisher matrix(ces), of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters}`, :math:`N_{\\rm events})`.
    :param list(float) vals: List of values to be added on the diagonal of the Fisher matrix.
    :param dict(int) ParNums: Dictionary specifying the position of each parameter in the Fisher matrix, as :py:class:`gwfast.waveforms.WaveFormModel.ParNums`.
    :param list(str) ParAdd: List of the names of parameters on which the prior should be added.

    :return: Fisher matrix with Gaussian priors added at the chosen positions, of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters}`, :math:`N_{\\rm events})`.
    :rtype: 3-D array

    """
    IdxAdd = np.sort(np.array([ParNums[par] for par in ParAdd]))

    pp = np.zeros((Matr.shape[0], Matr.shape[1]))

    diag = np.zeros(Matr.shape[0])
    diag[IdxAdd] = vals

    np.fill_diagonal(pp, diag)

    if Matr.ndim == 2:
        return pp + Matr
    else:
        return pp[:, :, np.newaxis] + Matr


##############################################################################
# LOCALIZATION REGION
##############################################################################


def compute_localization_region(Cov, parNum, thFid, perc_level=90, units="SqDeg"):
    """
    Compute the localisation region of one or multiple events.

    :param array Cov: Array containing the covariance matrix(ces), of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters}`, :math:`N_{\\rm events})`.
    :param dict(int) parNum: Dictionary specifying the position of each parameter in the Fisher matrix, as :py:class:`gwfast.waveforms.WaveFormModel.ParNums`.
    :param array thFid: Array containing the :math:`\\theta` sky position angle(s) of the event(s), in :math:`\\rm rad`.
    :param float perc_level: The percent level at which to compute the localisation region, from 0 to 100.
    :param str units: The units to use for the output, to choose among square degrees, ``'SqDeg'``, or steradians, ``'Sterad'``.
    :return: Localisation region(s) of the event(s).
    :rtype: 1-D array

    """
    # Cov_th_ph = Cov[ [parNum['theta'], parNum['phi']] ][:, [parNum['theta'], parNum['phi']] ]

    DelThSq = Cov[parNum["theta"], parNum["theta"]]
    DelPhiSq = Cov[parNum["phi"], parNum["phi"]]
    DelThDelPhi = Cov[parNum["phi"], parNum["theta"]]

    # From Barak, Cutler, PRD 69, 082005 (2004), gr-qc/0310125
    DelOmegaSr_base = (
        2 * np.pi * np.sqrt(DelThSq * DelPhiSq - DelThDelPhi**2) * np.abs(np.sin(thFid))
    )

    DelOmegaSr = -DelOmegaSr_base * np.log(1 - perc_level / 100)

    if units == "Sterad":
        return DelOmegaSr
    elif units == "SqDeg":
        return (180 / np.pi) ** 2 * DelOmegaSr


##############################################################################
# PLOTTING TOOLS (ELLIPSES)
##############################################################################
import matplotlib.pyplot as plt


def plot_corners(
    covariance, parameters: list, indices: list, event=None, labels={}, **kwargs
):
    """
    covariance:  The covariance matrix
    parameters:  List of parameters to plot
    indices:     Indices of that list of parameters,
                 correspond to where the parameter is in the cov. matrix
    event:       Mean values of the parameters, typically is
                 the truth value used for the Fisher analysis.
    labels:      Axis labels to use for each parameter,
                 should be a dictionary:
                 {parameters: the_labels}
    """
    color = kwargs.get("color", "C3")

    n_params = len(parameters)
    assert len(parameters) == len(indices), "Number of parameters and indices not match"

    fig, axes = plt.subplots(
        n_params,
        n_params,
        figsize=(1.7 * n_params, 1.7 * n_params),
        gridspec_kw={"wspace": 0.01, "hspace": 0.01},
        constrained_layout=True,
        sharex="col",
    )

    for col, (idx, key1) in enumerate(zip(indices, parameters)):
        for row, (jdx, key2) in enumerate(zip(indices, parameters)):
            if col > row:
                fig.delaxes(axes[row][col])
                continue

            # We want lower-triangle
            ax = axes[row, col]
            idx_pair = (idx, jdx)

            if row == n_params - 1:
                ax.set_xlabel(labels.get(key1, key1), fontsize=15)

            if row == col:
                # For this part, we use float64 for plotting purpose
                # as SciPy norm does not support float128
                mu = float(event[key1])
                std = np.sqrt(covariance[idx, jdx]).astype("f8")
                norm_rv = norm(mu, std)
                x_range = np.linspace(-2.5 * std, 2.5 * std, 500) + mu
                ax.plot(x_range, norm_rv.pdf(x_range), color=color)

                # Get the 90% Credible intervals
                # Make use of the symmetric property here
                CI_90 = norm_rv.ppf([0.05, 0.5, 0.95])
                interval = np.diff(CI_90)[0]
                ax.set_title(rf"${mu:.3f} \pm {interval:.3f}$")

                ax.axvline(CI_90[0], ls="--", color="grey")
                ax.axvline(CI_90[2], ls="--", color="grey")
                ax.set_yticks([])
                continue

            confidence_ellipse(
                covariance[np.ix_(idx_pair, idx_pair)],
                ax,
                event[key1],
                event[key2],
                edgecolor=color,
                n_std=2.0,
            )

            ax.scatter(event[key1], event[key2], c="red", s=3)
            if col == 0:
                ax.set_ylabel(labels.get(key2, key2), fontsize=15)
            else:
                ax.set_yticklabels([])
            if row != col:
                ax.tick_params(
                    axis="both",
                    which="both",
                    direction="in",
                    left=True,
                    right=True,
                    top=True,
                    bottom=True,
                )

    # Need to reset the share-y axes except the diagonal.
    for row, jdx in enumerate(indices):
        limits = axes[row, row].get_xlim()
        for ax in axes[row, :row]:
            ax.set_ylim(limits)

    # Not sure what does the `my_scales` do, skipping.
    return fig


def plot_contours(Covariance, plot_vars, plot_idxs, event, my_scales, plt_labels):

    # Example scales: scales = {'dL': lambda x: np.round(x*1000, 0), 'theta': theta_to_dec_degminsec ,'phi':phi_to_ra_hrms, 'iota':np.cos}
    # Example plt_labels:  plt_labels = [('RA', 'dec'), (r'$d_L[Mpc]$', r'$cos(\iota)$') ]

    fig, axs = plt.subplots(1, len(plot_vars), figsize=(15, 5))
    for ax, plotvar, plot_idx in zip(axs, plot_vars, plot_idxs):

        print(plotvar)
        print(plot_idx)
        # print(plotvar[1])

        confidence_ellipse(
            Covariance[np.ix_(plot_idx, plot_idx)],
            ax,
            event[plotvar[0]],
            event[plotvar[1]],
            edgecolor="red",
            n_std=2.0,
        )

        ax.scatter(event[plotvar[0]], event[plotvar[1]], c="red", s=3)
        # ax.set_title(title)
        ax.set_xlabel(plotvar[0], fontsize=15)
        ax.set_ylabel(plotvar[1], fontsize=15)

        if plotvar[1] == "theta":
            ax.set_ylim(ax.get_ylim()[::-1])
        if plotvar[0] == "phi":
            ax.set_xlim(ax.get_xlim()[::-1])

    plt.show()

    for ax, plotvar, plot_idx, plot_label in zip(axs, plot_vars, plot_idxs, plt_labels):
        if plotvar[0] in my_scales.keys():
            # transform scale on x axis
            old_labels = np.array(
                [
                    ax.get_xticklabels()[k].get_position()[0]
                    for k in range(len(ax.get_xticklabels()))
                ]
            )
            print(old_labels)

            labels = my_scales[plotvar[0]](
                np.array(
                    [
                        ax.get_xticklabels()[k].get_position()[0]
                        for k in range(len(ax.get_xticklabels()))
                    ]
                )
            )
            print(labels)

            ax.set_xticklabels(labels)

        if plotvar[1] in my_scales.keys():
            # transform scale on y axis
            labels = np.array(
                [
                    ax.get_yticklabels()[k].get_position()[1]
                    for k in range(len(ax.get_yticklabels()))
                ]
            )
            print(labels)

            new_labels = my_scales[plotvar[1]](
                np.array(
                    [
                        ax.get_yticklabels()[k].get_position()[1]
                        for k in range(len(ax.get_yticklabels()))
                    ]
                )
            )
            print(new_labels)

            ax.set_yticklabels(new_labels)

            ax.set_xlabel(plot_label[0], fontsize=15)
            ax.set_ylabel(plot_label[1], fontsize=15)

    # fig = plt.gcf()

    return fig


# From https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
def confidence_ellipse(cov, ax, mean_x, mean_y, n_std=3.0, facecolor="none", **kwargs):
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms

    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    # if x.size != y.size:
    #    raise ValueError("x and y must be the same size")

    # cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )

    # Calculating the standard deviation of x from
    # the square root of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    # mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    # mean_y = np.mean(y)

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
