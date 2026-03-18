#
#    Copyright (c) 2022 Francesco Iacovelli <francesco.iacovelli@unige.ch>, Michele Mancarella <michele.mancarella@unige.ch>
#    Copyright (c) 2025 Samson Leong <samson.leong@link.cuhk.edu.hk>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.

import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Union
from collections import OrderedDict

from jax import config, devices, tree, vmap, jacfwd
devices("cpu")
config.update("jax_enable_x64", True)

import numpy as np
import copy
import mpmath

try:
    np.float128(1.0)
    typeuse = "float128"
except AttributeError:
    print(
        "WARNING: numpy float128 type not supported on this machine, resorting to float64, precision might be lower."
    )
    typeuse = "float64"

DEFAULT_SVD = {
    "condition_max": 1e50,
    "truncate": False,
    "svals_thresh": 1e-15,
}

absmax = lambda ndarray: np.max(np.abs(ndarray))
absmin = lambda ndarray: np.min(np.abs(ndarray))

__all__ = [
    'compute_single_covariance_matrix',
    'compute_covariance_matrix',
    'covariance_change_variable',
    'covariance_change_variable_1',
    'print_single_matrix', 
    'print_matrices',
    'check_covariance',
]

##############################################################################
# INVERSION AND SANITY CHECKS
##############################################################################
def compute_single_svd(
        mpm_fisher_mat, condition, svd_kwargs={}
        ):
    condition_max = svd_kwargs.get("condition_max", DEFAULT_SVD["condition_max"])
    truncate = svd_kwargs.get("truncate", DEFAULT_SVD["truncate"])
    svals_thresh = svd_kwargs.get("svals_thresh", DEFAULT_SVD["svals_thresh"])
    
    U, sing_vals, V = mpmath.svd_r(mpm_fisher_mat)
    S_diag = np.array(sing_vals.tolist(), dtype=typeuse)

    if (truncate) and (np.abs(condition) > condition_max):
        max_sing_val = absmax(S_diag)
        S_inv = mpmath.matrix(
            np.where(
                np.abs(S_diag) / max_sing_val > svals_thresh,
                1 / S_diag,
                1 / (max_sing_val * svals_thresh)
            ).astype(typeuse)
        )
        S_trunc = mpmath.matrix(
            np.where(
                np.abs(S_diag) / max_sing_val > svals_thresh,
                S_diag,
                max_sing_val * svals_thresh
            ).astype(typeuse)
        )

        # Also copute truncated Fisher to quantify inversion error consistently
        trunc_fisher = U * mpmath.diag(S_trunc) * V
        trunc_fisher = (trunc_fisher + trunc_fisher.T) / 2
        trunc_fisher = np.array(trunc_fisher.tolist(), dtype=typeuse)

    else:
        S_inv = mpmath.matrix(1 / S_diag)

    cc = V.T * mpmath.diag(S_inv) * U.T
    return cc

def compute_single_svd_reg(mpm_fisher_mat, svd_kwargs={}):
    U, Sm, V = mpmath.svd_r(mpm_fisher_mat)

    S = np.squeeze(np.array(Sm.tolist(), dtype=typeuse))
    Um = np.array(U.tolist(), dtype=typeuse)
    Vm = np.array(V.tolist(), dtype=typeuse)

    svals_thresh = svd_kwargs.get("svals_thresh", DEFAULT_SVD["svals_thresh"])
    kVal = sum(S > svals_thresh)

    # Sinv = mpmath.matrix(np.array([1 / s for s in S]).astype(typeuse))
    return mpmath.matrix(
        Um[:, 0:kVal] @ np.diag(1.0 / S[0:kVal]) @ Vm[0:kVal, :]
    )


def compute_single_covariance_matrix(
        fisher_mat, inv_method='cho', alt_method='svd', svd_kwargs={}):
    if np.all(np.isnan(fisher_mat)):
        return np.full(fisher_mat.shape, np.nan)

    reweighted = False
    positive_definite = True
    mp_fisher = mpmath.matrix(fisher_mat)

    ## Check positive definiteness
    try:
        eigv = np.array(mpmath.eigh(mp_fisher)[0], dtype=typeuse)
        # Check positive definiteness
        cond = absmax(eigv) / absmin(eigv)
        positive_definite = min(eigv) >= 0
    except Exception as e:
        # Eigenvalue decomposition failed
        print(e)
        print("Inversion failed (Eigenvalue decomposition failed)!")
        return np.full(fisher_mat.shape, np.nan)

    try:
        # Normalize by the diagonal
        weights = mpmath.inverse(mpmath.diag(np.sqrt(np.diag(fisher_mat))))
        _mp_fisher = weights * mp_fisher * weights
        # Conditioning of the new Fisher
        new_eigv = np.array(mpmath.eigh(_mp_fisher)[0], dtype=typeuse)
        cond = absmax(new_eigv) / absmin(new_eigv)
        reweighted = True
    except ZeroDivisionError:
        print(
            "The Fisher matrix has a zero element on the diagonal. \n" + 
            "The normalization procedure will not be applied. Consider using a prior."
        )
        _mp_fisher = mp_fisher
    else:
        positive_definite = min(new_eigv) >= 0

    if inv_method == 'cho':
        if not positive_definite:
            inv_method = alt_method
        else:
            try:
            # In rare cases, the choleski decomposition still fails even if the eigenvalues are positive...
            # likely for very small eigenvalues
                cho = (mpmath.cholesky(_mp_fisher)) ** -1
            except Exception as e:
                inv_method = alt_method
                print(e)
                print(
                    f"Cholesky decomposition not usable. Eigenvalues seem ok but cholesky decomposition failed. Using method {inv_method}"
                )

    match inv_method:
        case "inv":
            cc = _mp_fisher ** -1
        case "cho":
            # c = cF**-1
            cc = cho.T * cho
        case "svd":
            cc = compute_single_svd(
                _mp_fisher, cond, svd_kwargs
                )
        case "svd_reg":
            cc = compute_single_svd_reg(
                _mp_fisher, svd_kwargs
            )
        case "lu":
            P, L, U = mpmath.lu(_mp_fisher)
            ll = P.T * L
            llinv = ll**-1
            uinv = U**-1
            cc = uinv * llinv

    # Enforce symmetry.
    cov_mat = (cc + cc.T) / 2

    if reweighted:
        # Undo the reweighting
        mp_cov_mat = weights * cov_mat * weights
    else:
        mp_cov_mat = cov_mat

    mp_cov_mat_reshaped = np.reshape(np.asarray(mp_cov_mat.tolist(), dtype='float64'), fisher_mat.shape)
    # sign, logdet = np.linalg.slogdet(mp_cov_mat_reshaped)
    # print('inversion method:', inv_method, 'single cov det sign:', sign)

    return np.asarray(mp_cov_mat.tolist(), dtype=typeuse)


def compute_covariance_matrix(
    fisher_matrix,
    inv_method='cho',
    alt_method='svd',
    cores=None,
    svd_kwargs=dict(
        condition_max=1e50,
        truncate=False,
        svals_thresh=1e-15,
    )
):
    """
    Obtain the covariance matrix(ces) by inverting the Fisher matrix(ces).

    :param array fisher_matrix: Array containing the Fisher matrix(ces) to invert, of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters}`, :math:`N_{\\rm events}, ...)`, :math:`N_{\\rm events}` can be more than one-dimension.
    :param str inv_method: Inversion method to use. To be chosen among ``'inv'``, ``'cho'``, ``'svd'``, ``'svd_reg'`` and ``'lu'``.
    :param str alt_method: Inversion method to use in case the inverison with ``invMethodIn`` fails. To be chosen among ``'inv'``, ``'cho'``, ``'svd'``, ``'svd_reg'`` and ``'lu'``. It has to be different from ``invMethodIn``.
    :param int Optional cores: Number of cores to use for parallel computation. If not specified, it will use all available cores minus 4, or the number of events if smaller.
    :param dict svd_kwargs: Dictionary containing the parameters for the SVD inversion method. It can contain the following keys:
        :param float condNumbMax: Maximum allowed condition number, above which the inverse matrix is not computed. The default value is 1e50, so the code will try to invert every matrix, irrespectively of the conditioning.
        :param bool, optional truncate: Boolean specifying if, when using the ``'svd'`` method, the function has to truncate the smallest singular values to the minimum allowed numerical precision.
        :param float svals_thresh: Threshold value to truncate the singular values when using the ``'svd'`` method, or to exclude the singular values from the inversion when using the ``'svd_reg'`` method.
    :return: Covariance matrix(ces) ((2+N)-D array) and inversion error(s) (N-D array), ``N`` being the number of dimensions of the parameters. The covariance matrix(ces) have the same shape of ``FisherMatrix``, i.e. :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters}`, :math:`N_{\\rm events}, ...)`.
    :rtype: tuple(array, array)

    """
    orig_shape = fisher_matrix.shape
    flat_fisher_mat = fisher_matrix.reshape(orig_shape[0], orig_shape[1], -1)
    flat_fisher_mat = np.asarray(flat_fisher_mat, dtype=typeuse)
    fisher_mat_loop = np.moveaxis(flat_fisher_mat, -1, 0)
    tmp_fisher_matrix = copy.deepcopy(flat_fisher_mat)

    if cores is None:
        available_cores = cpu_count()
        cores = max(1, available_cores - 4)
        cores = min(cores, flat_fisher_mat.shape[-1])

    if cores > 1:
        _compute_single_covariance_mat = partial(
            compute_single_covariance_matrix, 
            inv_method=inv_method,
            alt_method=alt_method,
            svd_kwargs=svd_kwargs
        )
        with Pool(processes=cores) as pool:
            cov_matrix_list = pool.map(
                _compute_single_covariance_mat, 
                fisher_mat_loop
            )
    else:
        cov_matrix_list = [
            compute_single_covariance_matrix(
                fisher_mat, inv_method=inv_method, 
                alt_method=alt_method, svd_kwargs=svd_kwargs
            )
            for fisher_mat in fisher_mat_loop
        ]

    cov_matrices = np.array(cov_matrix_list, dtype=typeuse)
    cov_matrices = np.moveaxis(cov_matrices, 0, -1)

    eps = compute_inversion_error(tmp_fisher_matrix, cov_matrices)

    ## Restore to the original input shape:
    cov_matrices = cov_matrices.reshape(*orig_shape)
    eps = eps.reshape(orig_shape[2:])
    return cov_matrices, eps

def high_dim_matmul(mat_1, mat_2):
    """
    Batched matrix multiplication over trailing (event/grid) axes.

    Expected shapes:
      mat_1: (N, M, *S)
      mat_2: (M, K, *S)  or (M, K) (broadcast across *S)

    Returns:
      out: (N, K, *S)
    """
    if mat_1.ndim < 2 or mat_2.ndim < 2:
        raise ValueError(f"high_dim_matmul expects arrays with ndim>=2, got {mat_1.ndim}, {mat_2.ndim}")

    N, M = mat_1.shape[0], mat_1.shape[1]
    if mat_2.shape[0] != M:
        raise ValueError(f"inner dim mismatch: mat_1 is ({N},{M},...), mat_2 is ({mat_2.shape[0]},{mat_2.shape[1]},...)")

    # Simple 2D case
    if mat_1.ndim == 2 and mat_2.ndim == 2:
        return mat_1 @ mat_2

    S1 = mat_1.shape[2:]  # batch shape
    K = mat_2.shape[1]

    # Allow mat_2 to be broadcast (constant across batch)
    if mat_2.ndim == 2:
        # reshape to (1, M, K) then broadcast later
        mat_2_batched = mat_2[None, :, :]
        S2 = S1
    else:
        S2 = mat_2.shape[2:]
        if S2 != S1:
            raise ValueError(f"batch shape mismatch: mat_1 batch={S1}, mat_2 batch={S2}")
        # Move batch axes to the front, then flatten
        perm2 = (*range(2, mat_2.ndim), 0, 1)  # (*S, M, K)
        mat_2_batched = np.transpose(mat_2, axes=perm2).reshape((-1, M, K))  # (B, M, K)

    # Move batch axes to the front, then flatten
    perm1 = (*range(2, mat_1.ndim), 0, 1)  # (*S, N, M)
    mat_1_batched = np.transpose(mat_1, axes=perm1).reshape((-1, N, M))  # (B, N, M)

    # If mat_2 is constant across the batch, broadcast it to (B, M, K)
    if mat_2.ndim == 2:
        B = mat_1_batched.shape[0]
        mat_2_batched = np.broadcast_to(mat_2_batched, (B, M, K))

    out_batched = mat_1_batched @ mat_2_batched  # (B, N, K)

    # Unflatten batch and move axes back to (N, K, *S)
    out = out_batched.reshape((*S1, N, K))
    out = np.transpose(out, axes=(len(S1), len(S1) + 1, *range(0, len(S1))))
    return out

# def high_dim_matmul(mat_1, mat_2):
#     """
#     Perform matrix multiplication for high-dimensional arrays.
#     The first two dimensions of the input arrays are treated as matrices.
#     """
#     return np.einsum('ij...,jk...->ik...', mat_1, mat_2)

def covariance_change_variable(
        covariance_matrix, injection_parameters, transform, from_params
    ):
    """
    Replace ``from_params`` in the covariance matrix with new parameters
    produced by ``transform``, allowing len(output) != len(from_params).

    ``transform`` receives the FULL ``injection_parameters`` dict so that
    ``jacfwd`` captures derivatives w.r.t. *all* original parameters (not
    only ``from_params``).  This is required whenever the transform outputs
    depend on parameters that remain in the matrix (e.g. ``iota``, ``dL``).

    Output ordering: [kept params in original order] + [new params].
    Output shape: ``(N_kept + N_new, N_kept + N_new, ...)``.

    :param array covariance_matrix: shape ``(N_orig, N_orig, ...)``.
    :param dict injection_parameters: All original parameter values.
    :param function transform: Receives dict of ALL parameters, returns dict
        of new parameters.  Must be JAX-differentiable.
    :param list from_params: Keys to REMOVE from output (replaced by the
        transform outputs).
    :return: ``(transformed_covariance, output_parameters_dict, output_keys)``
    """
    N_orig = covariance_matrix.shape[0]
    param_shape = covariance_matrix.shape[2:]
    matrix_keys = list(injection_parameters.keys())
    from_indices = set(matrix_keys.index(key) for key in from_params)
    kept_indices = [i for i in range(N_orig) if i not in from_indices]

    all_params = OrderedDict(
        {key: np.atleast_1d(injection_parameters[key]).reshape(-1)
         for key in matrix_keys}
    )

    transformed_params = transform(OrderedDict(all_params))
    out_keys = list(transformed_params.keys())

    N_to = len(out_keys)
    N_kept = len(kept_indices)
    N_out = N_kept + N_to

    kept_keys = [matrix_keys[i] for i in kept_indices]
    output_keys = kept_keys + out_keys

    if covariance_matrix.ndim == 2:
        scalar_params = OrderedDict(
            {k: all_params[k][0] for k in matrix_keys}
        )
        jac_pytree = jacfwd(transform)(scalar_params)

        J = np.zeros((N_out, N_orig), dtype=covariance_matrix.dtype)
        for out_row, in_col in enumerate(kept_indices):
            J[out_row, in_col] = 1.0
        for i, ok in enumerate(out_keys):
            for j, ik in enumerate(matrix_keys):
                J[N_kept + i, j] = np.asarray(jac_pytree[ok][ik])

        transform_covar = J @ covariance_matrix @ J.T

    else:
        _scalar_keys = [k for k, v in all_params.items() if np.ndim(v) == 0]
        if _scalar_keys:
            raise ValueError(
                f"Expected batched inputs, got scalars for keys={_scalar_keys}"
            )

        jac_pytree = vmap(jacfwd(transform))(all_params)

        J = np.zeros((N_out, N_orig, *param_shape),
                      dtype=covariance_matrix.dtype)
        for out_row, in_col in enumerate(kept_indices):
            J[out_row, in_col] = 1.0
        for i, ok in enumerate(out_keys):
            for j, ik in enumerate(matrix_keys):
                J[N_kept + i, j] = np.asarray(
                    jac_pytree[ok][ik]
                ).reshape(*param_shape)

        J_T = np.transpose(J, axes=(1, 0, *range(2, J.ndim)))
        transform_covar = high_dim_matmul(
            J, high_dim_matmul(covariance_matrix, J_T)
        )

    output_parameters = {}
    for key in output_keys:
        value = injection_parameters.get(key, None)
        if value is None:
            value = transformed_params.get(key, None)
        output_parameters[key] = (
            np.asarray(value).reshape(*param_shape)
            if covariance_matrix.ndim > 2 else value
        )

    return transform_covar, output_parameters, output_keys

# def covariance_change_variable(
#         convariance_matrix, injection_parameters, transform, from_params
#     ):
#     full_rank = convariance_matrix.shape[0]
#     param_shape = convariance_matrix.shape[2:]  # empty () if convariance_matrix is 2D
#     matrix_keys = list(injection_parameters.keys())
#     keys_indices = [matrix_keys.index(key) for key in from_params]
#
#     # Flatten inputs (support scalars too)
#     sub_injection_parameters = OrderedDict(
#         {key: np.atleast_1d(injection_parameters[key]).reshape(-1) for key in from_params}
#     )
#
#     sub_transformed_parameters = transform(sub_injection_parameters)
#     out_keys = list(sub_transformed_parameters.keys())
#
#     # Decide whether we have a batch axis to vmap over
#     # If covariance is 2D, we treat it as a single event and DO NOT vmap.
#     if convariance_matrix.ndim == 2:
#         jacobian_pytree = jacfwd(transform)(OrderedDict(
#             {k: sub_injection_parameters[k][0] for k in from_params}
#         ))
#         # jacobian_pytree[out][in] are scalars
#         jacobian_mat = np.stack(
#             [
#                 np.stack([np.asarray(jacobian_pytree[ok][ik]) for ik in from_params], axis=0)
#                 for ok in out_keys
#             ],
#             axis=0
#         )  # (Nout, Nin)
#
#         full_jacobian_mat = np.eye(full_rank, dtype=convariance_matrix.dtype)
#         full_jacobian_mat[np.ix_(keys_indices, keys_indices)] = jacobian_mat
#
#         J = full_jacobian_mat
#         transform_covar = J @ convariance_matrix @ J.T
#
#     else:
#         # Batched case: vmap over event axis
#         # Ensure leaves are rank>=1 for vmap
#         _scalar_keys = [k for k, v in sub_injection_parameters.items() if np.ndim(v) == 0]
#         if _scalar_keys:
#             raise ValueError(f"Expected batched inputs, got scalars for keys={_scalar_keys}")
#
#         jacobian_pytree = vmap(jacfwd(transform))(sub_injection_parameters)
#         # dict[out][in] -> (Nflat,)
#         jacobian_mat = np.stack(
#             [
#                 np.stack([np.asarray(jacobian_pytree[ok][ik]) for ik in from_params], axis=0)
#                 for ok in out_keys
#             ],
#             axis=0
#         ).reshape(len(out_keys), len(from_params), *param_shape)
#
#         full_jacobian_mat = np.zeros_like(convariance_matrix)
#         full_jacobian_mat[np.diag_indices(full_rank)] = 1.0
#         full_jacobian_mat[np.ix_(keys_indices, keys_indices)] = jacobian_mat
#         full_jacobian_mat_moved = np.moveaxis(full_jacobian_mat, -1, 0)
#         print(full_jacobian_mat_moved)
#
#         full_jacobian_mat_T = np.transpose(
#             full_jacobian_mat, axes=(1, 0, *range(2, full_jacobian_mat.ndim))
#         )
#
#         transform_covar = high_dim_matmul(
#             full_jacobian_mat, high_dim_matmul(convariance_matrix, full_jacobian_mat_T)
#         )
#
#     # Maintain original order of keys, replacing transformed subset
#     transform_keys = list(matrix_keys)
#     for idx, new_key_name in zip(keys_indices, out_keys):
#         transform_keys[idx] = new_key_name
#
#     transform_parameters = {}
#     for key in transform_keys:
#         value = injection_parameters.get(key, None)
#         if value is None:
#             value = sub_transformed_parameters.get(key, None)
#         # For 2D covariance, param_shape=() so reshape is a no-op for scalars/arrays
#         transform_parameters[key] = np.asarray(value).reshape(*param_shape) if convariance_matrix.ndim > 2 else value
#
#     return transform_covar, transform_parameters, transform_keys

# def covariance_change_variable(
#         convariance_matrix, injection_parameters, transform, from_params
#     ):
#     """
#     Transform the covariance matrix according to a change of variable defined by the `transform` function.

#     :param array convariance_matrix: Covariance matricies to be transformed, of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters}`, :math:`N_{\\rm events} ...)`.
#     :param dict injection_parameters: Dictionary containing the original sets of injected parameters, or the means of the parameters.
#     :param function transform: An N×N transformation function that takes a subset of injection parameters and transforms to a new set of parameters. Both the input and output must be dictionaries with same number of parameters.
#     :param list from_params: Sub-list of keys in `injection_parameters` that will be transformed. 
#     """
#     # Get the basic input and output shapes
#     full_rank = convariance_matrix.shape[0]
#     param_shape = convariance_matrix.shape[2:]
#     transform_dim = len(from_params)
#     matrix_keys = list(injection_parameters.keys())
#     keys_indices = [matrix_keys.index(key) for key in from_params]

#     # vmap cannot handle N-to-N transforms nicely, need to flatten the input first
#     sub_injection_parameters = OrderedDict(
#         {key: injection_parameters[key].reshape(-1) for key in from_params})
#     jacobian_dict = vmap(jacfwd(transform))(sub_injection_parameters)
#     # Need to re-order the output Jacobian dictionary to match with expectation
#     sub_transformed_parameters = transform(sub_injection_parameters)
#     jacobian_dict = OrderedDict(
#         {key: jacobian_dict[key] for key in sub_transformed_parameters.keys()}
#     )
#     jacobian_mat = np.array(tree.leaves(jacobian_dict)).reshape(
#         transform_dim, transform_dim, *param_shape)

#     # Compute transformed Fisher matrix
#     full_jacobian_mat = np.zeros_like(convariance_matrix)
#     full_jacobian_mat[np.diag_indices(full_rank)] = 1.0
#     full_jacobian_mat[np.ix_(keys_indices, keys_indices)] = jacobian_mat
#     full_jacobian_mat_T = np.transpose(
#         full_jacobian_mat, axes=(1, 0, *range(2, full_jacobian_mat.ndim)))
#     transform_covar = high_dim_matmul(
#         full_jacobian_mat, high_dim_matmul(
#             convariance_matrix, full_jacobian_mat_T))
    
#     # Compute transformed parameters and keys, maintaining the original order
#     transform_keys = matrix_keys
#     for idx, new_key_name in zip(keys_indices, sub_transformed_parameters.keys()):
#         transform_keys[idx] = new_key_name
#     transform_parameters = {}
#     for key in transform_keys:
#         value = injection_parameters.get(key, None)
#         if value is None:
#             value = sub_transformed_parameters.get(key, None)
#         transform_parameters[key] = value.reshape(*param_shape)

#     return transform_covar, transform_parameters, transform_keys

def covariance_change_variable_1(
        convariance_matrix, injection_parameters, target_keys, transform, from_params, to_params
    ):
    """
    Transform the covariance matrix according to a change of variable defined by the `transform` function.

    :param array convariance_matrix: Covariance matricies to be transformed, of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters}`, :math:`N_{\\rm events} ...)`.
    :param dict injection_parameters: Dictionary containing the original sets of injected parameters, or the means of the parameters.
    :param list target_keys: List containing the target sets of parameter keys.
    :param function transform: A transformation function that takes a subset of injection parameters and transforms to a new set of parameters. Both the input and output must be dictionaries.
    :param list from_params: Sub-list of keys in `injection_parameters` that will be transformed. 
    :param list to_params: Sub-list of keys in the output dictionary from `transform` corresponding to the transformed parameters.
    """
    # Get the basic input and output shapes
    inj_full_rank = convariance_matrix.shape[0]
    tar_full_rank = len(target_keys)
    param_shape = convariance_matrix.shape[2:]
    from_dim = len(from_params)
    to_dim = len(to_params)
    inj_keys = list(injection_parameters.keys())
    from_keys_indices = [inj_keys.index(key) for key in from_params]
    to_keys_indices = [target_keys.index(key) for key in to_params]

    fixed_dim = inj_full_rank - from_dim
    fixed_dim_ = tar_full_rank - to_dim
    assert fixed_dim == fixed_dim_

    # vmap cannot handle high dimensional transforms nicely, need to flatten the input first
    sub_injection_parameters = OrderedDict(
        {key: injection_parameters[key].reshape(-1) for key in from_params})
    jacobian_dict = vmap(jacfwd(transform))(sub_injection_parameters)
    # Need to re-order the output Jacobian dictionary to match with expectation
    sub_transformed_parameters = transform(sub_injection_parameters)
    jacobian_dict = OrderedDict(
        {key: jacobian_dict[key] for key in sub_transformed_parameters.keys()}
    )
    jacobian_mat = np.array(tree.leaves(jacobian_dict)).reshape(
        to_dim, from_dim, *param_shape)

    # Compute transformed Fisher matrix
    full_jacobian_mat = np.zeros((fixed_dim + to_dim, fixed_dim + from_dim, *param_shape))
    full_jacobian_mat[np.diag_indices(fixed_dim)] = 1.0
    full_jacobian_mat[np.ix_(to_keys_indices, from_keys_indices)] = jacobian_mat
    full_jacobian_mat_T = np.transpose(
        full_jacobian_mat, axes=(1, 0, *range(2, full_jacobian_mat.ndim)))
    transform_covar = high_dim_matmul(
        full_jacobian_mat, high_dim_matmul(
            convariance_matrix, full_jacobian_mat_T))
    
    # Compute transformed parameters and keys, maintaining the original order
    tar_parameters = {}
    for key in target_keys:
        value = injection_parameters.get(key, None)
        if value is None:
            value = sub_transformed_parameters.get(key, None)
        tar_parameters[key] = value.reshape(*param_shape)

    return transform_covar, tar_parameters, target_keys

def print_single_matrix(matrix, parameters:Union[dict, list]):
    """
    A helper function to print a Fisher/Covariance matrices nicely.

    :param array matrix: Array containing one matrix for prining, must be 2D.
    :param dict/list parameters: Dictionary or list containing the parameters names to be printed as headers.

    """
    assert matrix.ndim == 2, "Single matrix should be 2D."
    if isinstance(parameters, dict):
        keys = list(parameters.keys())
    elif isinstance(parameters, list):
        keys = parameters
    else:
        raise TypeError(
            "Parameters should be a dictionary or a list, got %s." % type(parameters)
        )

    max_len = len(max(keys, key=len)) + 1
    col_len = max(11, max_len)
    row = f'{"":{max_len}}   ' + '  '.join([f'{col_key:^{col_len}}' for col_key in keys])
    print(row)
    for rdx, row_key in enumerate(keys):
        row = f'{row_key:>{max_len}}  '
        for cdx, _ in enumerate(keys):
            row += f'{matrix[rdx][cdx]:+{col_len}.3e}  '
        print(row)


def print_matrices(matrices, parameters:Union[dict, list]):
    """
    A helper function to print array of Fisher/Covariance matrices nicely.

    :param array matrices: Array containing the matrix(ces) for prining, of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters}`, :math:`N_{\\rm events})`.
    :param dict/list parameters: Dictionary or list containing the parameters names to be printed as headers.

    """
    if matrices.ndim > 3:
        print("The parameter axis of the input matrices seems to be more than 1D, flattening it for iteration.")
        orig_shape = matrices.shape
        flat_matrices = matrices.reshape(orig_shape[0], orig_shape[1], -1)
    else:
        flat_matrices = matrices
    # Swapping the axes so that it can be iterated over the different sets of parameters.
    fisher_mats_iter = np.moveaxis(flat_matrices, 2, 0)
    for matrix in fisher_mats_iter:
        print_single_matrix(matrix, parameters)
        print('--------------------')


def compute_inversion_error(Fisher, Cov):
    """
    Compute the inversion error given the Fisher and covariance matrices.

    :param array Fisher: Array containing the Fisher matrix(ces), of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters}`, :math:`N_{\\rm events})`.
    :param array Cov: Array containing the covariance matrix(ces), of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters}`, :math:`N_{\\rm events})`.

    :return: Inversion error for the given matrices.
    :rtype: 1-D array

    """
    # Assuming the Fisher and Covariance take shape: (N_keys, N_keys, N_params)
    identity = np.einsum("ijl,jkl->ikl", Cov, Fisher)
    diff = identity - np.eye(Fisher.shape[0])[..., None]
    return np.max(np.abs(diff), axis=(0, 1))


# The old covariance function
def CovMatr(
    FisherMatrix,
    invMethodIn="cho",
    condNumbMax=1e50,
    truncate=False,
    svals_thresh=1e-15,
    verbose=False,
    alt_method="svd",
):
    """
    Invert the Fisher matrix(ces), obtaining the covariance matrix(ces).

    :param array FisherMatrix: Array containing the Fisher matrix(ces) to invert, of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters}`, :math:`N_{\\rm events})`.
    :param str invMethodIn: Inversion method to use. To be chosen among ``'inv'``, ``'cho'``, ``'svd'``, ``'svd_reg'`` and ``'lu'``.
    :param float condNumbMax: Maximum allowed condition number, above which the inverse matrix is not computed. The default value is 1e50, so the code will try to invert every matrix, irrespectively of the conditioning.
    :param bool, optional truncate: Boolean specifying if, when using the ``'svd'`` method, the function has to truncate the smallest singular values to the minimum allowed numerical precision.
    :param float svals_thresh: Threshold value to truncate the singular values when using the ``'svd'`` method, or to exclude the singular values from the inversion when using the ``'svd_reg'`` method.
    :param bool, optional verbose: Boolean specifying if the code has to print additional details during execution.
    :param str alt_method: Inversion method to use in case the inverison with ``invMethodIn`` fails. To be chosen among ``'inv'``, ``'cho'``, ``'svd'``, ``'svd_reg'`` and ``'lu'``. It has to be different from ``invMethodIn``.
    :return: Covariance matrix(ces) (3-D array) and inversion error(s) (1-D array). The covariance matrix(ces) have the same shape of ``FisherMatrix``, i.e. :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters}`, :math:`N_{\\rm events})`.
    :rtype: tuple(array, array)

    """
    orig_shape = FisherMatrix.shape
    flat_fisher_mat = FisherMatrix.reshape(orig_shape[0], orig_shape[1], -1)

    FisherMatrixOr = copy.deepcopy(flat_fisher_mat)

    reweighted = False
    FisherM = flat_fisher_mat.astype(typeuse)
    CovMatr = np.zeros_like(FisherM)

    cho_failed = 0
    for k, fisher_mat in enumerate(np.moveaxis(FisherM, -1, 0)):
        if np.all(np.isnan(fisher_mat)):
            if verbose:
                print("Fisher is nan at position %s. " % k)
            CovMatr[:, :, k] = np.full(fisher_mat.shape, np.nan)
            continue
        # go to mpmath
        ff = mpmath.matrix(fisher_mat.astype(typeuse))

        try:
            # Conditioning of the original Fisher
            E, _ = mpmath.eigh(ff)
            E = np.array(E.tolist(), dtype=typeuse)
            if np.any(E < 0) and verbose:
                print("Matrix is not positive definite!")

            cond = np.max(np.abs(E)) / np.min(np.abs(E))
            if verbose:
                print("Condition of original matrix: %s" % cond)

        except Exception as e:
            # Eigenvalue decomposition failed
            print(e)
            print("Inversion failed!")
            CovMatr[:, :, k] = np.full(FisherM[:, :, k].shape, np.nan)
            continue

        try:
            # Normalize by the diagonal
            ws = mpmath.diag(
                [1 / mpmath.sqrt(ff[i, i]) for i in range(FisherM.shape[-2])]
            )
            FisherM_ = ws * ff * ws
            # Conditioning of the new Fisher
            EE, _ = mpmath.eigh(FisherM_)
            E = np.array(EE.tolist(), dtype=typeuse)
            cond = np.max(np.abs(E)) / np.min(np.abs(E))
            if verbose:
                print("Condition of the new matrix: %s" % cond)
            reweighted = True
        except ZeroDivisionError:
            print(
                "The Fisher matrix has a zero element on the diagonal at position %s. The normalization procedure will not be applied. Consider using a prior."
                % k
            )
            FisherM_ = ff

        invMethod = invMethodIn
        if np.any(E < 0):
            if verbose:
                print("Matrix is not positive definite at position %s!" % k)
            if invMethodIn == "cho":
                cho_failed += 1
                invMethod = alt_method
                if verbose:
                    print(
                        "Cholesky decomposition not usable. Using method %s"
                        % invMethod
                    )
        elif invMethod == "cho":
            try:
                # In rare cases, the choleski decomposition still fails even if the eigenvalues are positive...
                # likely for very small eigenvalues
                c = (mpmath.cholesky(FisherM_)) ** -1
            except Exception as e:
                print(e)
                invMethod = alt_method
                print(
                    "Cholesky decomposition not usable. Eigenvalues seem ok but cholesky decomposition failed. Using method %s"
                    % invMethod
                )
                # print('Eigenvalues: %s' %str(E))
                cho_failed += 1

        if invMethod == "inv":
            cc = FisherM_**-1
        elif invMethod == "cho":
            # c = cF**-1
            cc = c.T * c
        elif invMethod == "svd":
            U, Sm, V = mpmath.svd_r(FisherM_)
            S = np.array(Sm.tolist(), dtype=typeuse)
            if (truncate) and (np.abs(cond) > condNumbMax):
                if verbose:
                    print("Truncating singular values below %s" % svals_thresh)

                maxev = np.max(np.abs(S))
                Sinv = mpmath.matrix(
                    np.array(
                        [
                            (
                                1 / s
                                if np.abs(s) / maxev > svals_thresh
                                else 1 / (maxev * svals_thresh)
                            )
                            for s in S
                        ]
                    ).astype(typeuse)
                )
                St = mpmath.matrix(
                    np.array(
                        [
                            (
                                s
                                if np.abs(s) / maxev > svals_thresh
                                else maxev * svals_thresh
                            )
                            for s in S
                        ]
                    ).astype(typeuse)
                )

                # Also copute truncated Fisher to quantify inversion error consistently
                truncFisher = U * mpmath.diag([s for s in St]) * V
                truncFisher = (truncFisher + truncFisher.T) / 2
                FisherMatrixOr[:, :, k] = np.array(
                    truncFisher.tolist(), dtype=typeuse
                )

                if verbose:
                    truncated = (
                        np.abs(S) / maxev < svals_thresh
                    )  # np.array([1 if np.abs(s)/maxev>svals_thresh else 0 for s in S ]
                    print("%s singular values truncated" % (truncated.sum()))
            else:
                Sinv = mpmath.matrix(
                    np.array([1 / s for s in S]).astype(typeuse)
                )
                St = S

            cc = V.T * mpmath.diag([s for s in Sinv]) * U.T

        elif invMethod == "svd_reg":

            U, Sm, V = mpmath.svd_r(FisherM_)

            S = np.squeeze(np.array(Sm.tolist(), dtype=typeuse))
            Um = np.array(U.tolist(), dtype=typeuse)
            Vm = np.array(V.tolist(), dtype=typeuse)

            kVal = sum(S > svals_thresh)

            Sinv = mpmath.matrix(np.array([1 / s for s in S]).astype(typeuse))
            cc = mpmath.matrix(
                Um[:, 0:kVal] @ np.diag(1.0 / S[0:kVal]) @ Vm[0:kVal, :]
            )

        elif invMethod == "lu":
            P, L, U = mpmath.lu(FisherM_)
            ll = P.T * L
            llinv = ll**-1
            uinv = U**-1
            cc = uinv * llinv

        # Enforce symmetry.
        cc = (cc + cc.T) / 2

        if reweighted:
            # Undo the reweighting
            CovMatr_ = ws * cc * ws
        else:
            CovMatr_ = cc

        CovMatr[:, :, k] = np.array(CovMatr_.tolist(), dtype=typeuse)
        if verbose:
            print()

    eps = compute_inversion_error(FisherMatrixOr, CovMatr)

    if verbose:
        print("Error with %s: %s\n" % (invMethod, eps))
        print(
            " Inversion error with method %s: min=%s, max=%s, mean=%s, std=%s "
            % (invMethodIn, np.min(eps), np.max(eps), np.mean(eps), np.std(eps))
        )
        print(
            "Method %s not possible on %s non-positive definite matrices, %s was used in those cases. "
            % (invMethodIn, cho_failed, alt_method)
        )

    ## Rehape to the original input shape:
    CovMatr = CovMatr.reshape(*orig_shape)
    eps = eps.reshape(orig_shape[2:])
    return CovMatr, eps


def perturb_Fisher(totF, eps=1e-10, **kwargs):
    """
    Add random small perturbations to the FIM to a specified decimal and prints the relative errors, to check if the inversion remains stable.

    :param array totF: Array containing the Fisher matrix(ces) to check, of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters}`, :math:`N_{\\rm events})`.
    :param float eps: Decimal at which to add the random perturbation.
    :param kwargs: Optional arguments to be passed to :py:class:`gwfast.fisherTools.CovMatr`, such as ``invMethodIn``.

    """
    Cov_base, _ = CovMatr(totF, **kwargs)

    totF_random = totF + np.random.rand(*totF.shape) * eps
    Cov, _ = CovMatr(totF_random, **kwargs)

    epsErr = [
        np.linalg.norm(Cov_base[i] / Cov[i] - 1, ord=np.inf)
        for i in range(Cov.shape[-1])
    ]
    print("Relative errors when perturbing at the %s level: %s" % (eps, epsErr))


def check_covariance(FisherM, Cov, tol=1e-10):
    """
    Compute the inversion error, print the difference between the product of Fisher and covariance matrices and the identity matrix on the diagonal, and print the off–diagonal elements of the product of Fisher and covariance matrices higher than a threshold.

    :param array FisherM: Array containing the Fisher matrix(ces) to check, of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters}`, :math:`N_{\\rm events})`.
    :param array Cov: Array containing the covariance matrix(ces) to check, of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters}`, :math:`N_{\\rm events})`.
    :param float tol: Threshold above which to print the off–diagonal elements of the product of Fisher and covariance matrices.

    :return: Product of Fisher and covariance matrices, of shape :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters}`, :math:`N_{\\rm events})`.
    :rtype: 3-D array

    """
    recovered_Ids = [Cov[:, :, i] @ FisherM[:, :, i] for i in range(Cov.shape[-1])]

    #
    epsErr = compute_inversion_error(FisherM, Cov)
    print("Inversion errors: %s" % epsErr)

    #
    diag_diff = [recovered_Ids[i].diagonal() - 1 for i in range(Cov.shape[-1])]
    print("diagonal-1 = %s" % str(diag_diff))

    #
    offDiag = [
        recovered_Ids[i][np.matrix(~np.eye(recovered_Ids[i].shape[0], dtype=bool))]
        for i in range(Cov.shape[-1])
    ]

    print("Max off diagonal: %s" % str([max(offDiag[i]) for i in range(Cov.shape[-1])]))

    print(
        "\nmask: where F*S(off-diagonal)>%s (--> problematic if True off diagonal)"
        % tol
    )
    print([recovered_Ids[i] > tol for i in range(Cov.shape[-1])])

    return recovered_Ids