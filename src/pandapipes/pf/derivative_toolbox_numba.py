import numpy as np
from numpy import linalg

from pandapipes.constants import P_CONVERSION, GRAVITATION_CONSTANT, NORMAL_PRESSURE, \
    NORMAL_TEMPERATURE
from pandapipes.idx_branch import LENGTH, LAMBDA, D, LOSS_COEFFICIENT as LC, PL, AREA, \
    MDOTINIT, FROM_NODE, TO_NODE, TOUTINIT
from pandapipes.idx_node import HEIGHT, PAMB, PINIT, TINIT as TINIT_NODE

try:
    from numba import jit
    from numba import int32, float64, int64
except ImportError:
    from pandapower.pf.no_numba import jit
    from numpy import int32, float64, int64


@jit((float64[:, :], float64[:], float64[:], float64[:], float64[:], float64[:]), nopython=True, cache=False)
def derivatives_hydraulic_incomp_numba(branch_pit, der_lambda, p_init_i_abs, p_init_i1_abs,
                                       height_difference, rho):
    le = der_lambda.shape[0]
    load_vec = np.zeros_like(der_lambda)
    df_dm = np.zeros_like(der_lambda)
    df_dp = np.ones_like(der_lambda)
    df_dp1 = np.ones_like(der_lambda) * (-1)
    load_vec_nodes_from = np.zeros_like(der_lambda)
    load_vec_nodes_to = np.zeros_like(der_lambda)
    df_dm_nodes = np.ones_like(der_lambda)

    for i in range(le):
        m_init_abs = np.abs(branch_pit[i][MDOTINIT])
        m_init2 = m_init_abs * branch_pit[i][MDOTINIT]
        p_diff = p_init_i_abs[i] - p_init_i1_abs[i]
        const_height = rho[i] * GRAVITATION_CONSTANT * height_difference[i] / P_CONVERSION
        friction_term = np.divide(branch_pit[i][LENGTH] * branch_pit[i][LAMBDA], branch_pit[i][D]) \
            + branch_pit[i][LC]
        const_term = np.divide(1, branch_pit[i][AREA] ** 2 * rho[i] * P_CONVERSION * 2)

        df_dm[i] = -1. * const_term * (2 * m_init_abs * friction_term + der_lambda[i]
                                   * np.divide(branch_pit[i][LENGTH], branch_pit[i][D]) * m_init2)

        load_vec[i] = p_diff + branch_pit[i][PL] + const_height - const_term * m_init2 * friction_term

        load_vec_nodes_from[i] = branch_pit[i][MDOTINIT]
        load_vec_nodes_to[i] = branch_pit[i][MDOTINIT]
    return load_vec, load_vec_nodes_from, load_vec_nodes_to, df_dm, df_dm_nodes, df_dp, df_dp1


@jit((float64[:, :], float64[:, :], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],
      float64[:], float64[:], float64[:], float64[:]), nopython=True, cache=False)
def derivatives_hydraulic_comp_numba(node_pit, branch_pit, lambda_, der_lambda, p_init_i_abs, p_init_i1_abs,
                                     height_difference, comp_fact, der_comp, der_comp1, rho, rho_n):
    le = lambda_.shape[0]
    load_vec = np.zeros_like(lambda_)
    df_dm = np.zeros_like(lambda_)
    df_dp = np.zeros_like(lambda_)
    df_dp1 = np.zeros_like(lambda_)
    load_vec_nodes_from = np.zeros_like(der_lambda)
    load_vec_nodes_to = np.zeros_like(der_lambda)
    df_dm_nodes = np.ones_like(der_lambda)
    from_nodes = branch_pit[:, FROM_NODE].astype(np.int32)

    # Formulas for gas pressure loss according to laminar version
    for i in range(le):
        # compressibility settings
        m_init_abs = np.abs(branch_pit[i][MDOTINIT])
        m_init2 = branch_pit[i][MDOTINIT] * m_init_abs
        p_diff = p_init_i_abs[i] - p_init_i1_abs[i]
        p_sum = p_init_i_abs[i] + p_init_i1_abs[i]
        p_sum_div = np.divide(1, p_sum)
        fn = from_nodes[i]
        tm = (node_pit[fn, TINIT_NODE] + branch_pit[i][TOUTINIT]) / 2

        const_height =  rho[i] * GRAVITATION_CONSTANT * height_difference[i] / P_CONVERSION
        friction_term = np.divide(lambda_[i] * branch_pit[i][LENGTH], branch_pit[i][D]) + \
                        branch_pit[i][LC]
        normal_term = np.divide(NORMAL_PRESSURE, NORMAL_TEMPERATURE * P_CONVERSION * rho_n[i] *
                                branch_pit[i][AREA] ** 2)

        load_vec[i] = p_diff + branch_pit[i][PL] + const_height \
            - normal_term * comp_fact[i] * m_init2 * friction_term * p_sum_div * tm

        const_term = normal_term * m_init2 * friction_term * tm
        df_dp[i] = 1. - const_term * p_sum_div * (der_comp[i] - comp_fact[i] * p_sum_div)
        df_dp1[i] = -1. - const_term * p_sum_div * (der_comp1[i] - comp_fact[i] * p_sum_div)

        df_dm[i] = -1. * normal_term * comp_fact[i] * p_sum_div * tm * (2 * m_init_abs * friction_term \
            + np.divide(der_lambda[i] * branch_pit[i][LENGTH] * m_init2, branch_pit[i][D]))

        load_vec_nodes_from[i] = branch_pit[i][MDOTINIT]
        load_vec_nodes_to[i] = branch_pit[i][MDOTINIT]
    return load_vec, load_vec_nodes_from, load_vec_nodes_to, df_dm, df_dm_nodes, df_dp, df_dp1


@jit((float64[:], float64[:], float64[:], float64[:], float64[:]), nopython=True)
def calc_lambda_nikuradse_incomp_numba(m, d, k, eta, area):
    lambda_nikuradse = np.empty_like(m)
    lambda_laminar = np.zeros_like(m)
    re = np.empty_like(m)
    m_abs = np.abs(m)
    for i in range(m.shape[0]):
        re[i] = np.divide(m_abs[i] * d[i], eta[i] * area[i])
        if re[i] != 0:
            lambda_laminar[i] = 64 / re[i]
        lambda_nikuradse[i] = np.power(-2 * np.log10(k[i] / (3.71 * d[i])), -2)
    return re, lambda_laminar, lambda_nikuradse


@jit((float64[:], float64[:], float64[:], float64[:], float64[:]), nopython=True)
def calc_lambda_nikuradse_comp_numba(m, d, k, eta, area):
    lambda_nikuradse = np.empty_like(m)
    lambda_laminar = np.zeros_like(m)
    re = np.empty_like(m)
    for i, mi in enumerate(m):
        m_abs = np.abs(mi)
        re[i] = np.divide(m_abs * d[i], eta[i] * area[i])
        if re[i] != 0:
            lambda_laminar[i] = np.divide(64, re[i])
        lambda_nikuradse[i] = np.divide(1, (2 * np.log10(np.divide(d[i], k[i])) + 1.14) ** 2)
    return re, lambda_laminar, lambda_nikuradse


@jit((float64[:], float64[:]), nopython=True, cache=False)
def calc_medium_pressure_with_derivative_numba(p_init_i_abs, p_init_i1_abs):
    p_m = p_init_i_abs.copy()
    der_p_m = np.ones_like(p_init_i_abs)
    der_p_m1 = der_p_m * (-1)
    val = 2 / 3
    for i in range(p_init_i_abs.shape[0]):
        if p_init_i_abs[i] != p_init_i1_abs[i]:
            diff_p_sq = p_init_i_abs[i] ** 2 - p_init_i1_abs[i] ** 2
            diff_p_sq_div = np.divide(1, diff_p_sq)
            diff_p_cub = p_init_i_abs[i] ** 3 - p_init_i1_abs[i] ** 3
            p_m[i] = val * diff_p_cub * diff_p_sq_div
            der_p_m[i] = (3 * p_init_i_abs[i] ** 2 * diff_p_sq - 2 * p_init_i_abs[i] * diff_p_cub) \
                * diff_p_sq_div ** 2 * val
            der_p_m1[i] = (-3 * p_init_i1_abs[i] ** 2 * diff_p_sq
                           + 2 * p_init_i1_abs[i] * diff_p_cub) * diff_p_sq_div ** 2 * val
    return p_m, der_p_m, der_p_m1


@jit((float64[:], float64[:], float64[:], float64[:], float64[:], int64), nopython=True)
def colebrook_numba(re, d, k, lambda_nikuradse, dummy, max_iter):
    lambda_cb = lambda_nikuradse.copy()
    lambda_cb_old = lambda_nikuradse.copy()
    converged = False
    niter = 0

    # Inner Newton-loop for calculation of lambda
    while not converged and niter < max_iter:
        for i in range(len(lambda_cb)):
            if np.isclose(re[i],0): continue
            sqt = np.sqrt(lambda_cb[i])
            add_val = np.divide(k[i], (3.71 * d[i]))
            sqt_div = np.divide(1, sqt)
            re_div = np.divide(1, re[i])
            sqt_div3 = sqt_div ** 3

            f = sqt_div + 2 * np.log10(2.51 * re_div * sqt_div + add_val)
            df_dlambda_cb = - 0.5 * sqt_div3 - 2.51 * re_div * sqt_div3 * np.divide(
                1, np.log(10) * (2.51 * re_div * sqt_div + add_val))
            x = - f / df_dlambda_cb

            lambda_cb_old[i] = lambda_cb[i]
            lambda_cb[i] += x

        dx = np.abs(lambda_cb - lambda_cb_old) * dummy
        error_lambda = linalg.norm(dx) / dx.shape[0]

        if error_lambda <= 1e-4:
            converged = True

        niter += 1

    return converged, lambda_cb


@jit((float64[:, :], int32[:], int32[:]), nopython=True)
def calc_derived_values_numba(node_pit, from_nodes, to_nodes):
    le = len(from_nodes)
    tinit_branch = np.empty(le, dtype=np.float64)
    height_difference = np.empty(le, dtype=np.float64)
    p_init_i_abs = np.empty(le, dtype=np.float64)
    p_init_i1_abs = np.empty(le, dtype=np.float64)
    for i in range(le):
        fn = from_nodes[i]
        tn = to_nodes[i]
        tinit_branch[i] = (node_pit[fn, TINIT_NODE] + node_pit[tn, TINIT_NODE]) / 2
        height_difference[i] = node_pit[fn, HEIGHT] - node_pit[tn, HEIGHT]
        p_init_i_abs[i] = node_pit[fn, PINIT] + node_pit[fn, PAMB]
        p_init_i1_abs[i] = node_pit[tn, PINIT] + node_pit[tn, PAMB]
    return tinit_branch, height_difference, p_init_i_abs, p_init_i1_abs
