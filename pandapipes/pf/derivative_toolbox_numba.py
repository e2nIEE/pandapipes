import numpy as np
from numpy import linalg

from pandapipes.constants import P_CONVERSION, GRAVITATION_CONSTANT, NORMAL_PRESSURE, \
    NORMAL_TEMPERATURE
from pandapipes.idx_branch import LENGTH, LAMBDA, D, LOSS_COEFFICIENT as LC, RHO, PL, AREA, TINIT, \
    VINIT
from pandapipes.idx_node import HEIGHT, PAMB, PINIT, TINIT as TINIT_NODE

try:
    from numba import jit
    from numba import int32, float64, int64
except ImportError:
    from pandapower.pf.no_numba import jit
    from numpy import int32, float64, int64


@jit((float64[:, :], float64[:], float64[:], float64[:], float64[:]), nopython=True, cache=False)
def derivatives_hydraulic_incomp_numba(branch_pit, der_lambda, p_init_i_abs, p_init_i1_abs,
                                       height_difference):
    le = der_lambda.shape[0]
    load_vec = np.zeros_like(der_lambda)
    df_dv = np.zeros_like(der_lambda)
    df_dp = np.ones_like(der_lambda) * (-1)
    df_dp1 = np.ones_like(der_lambda)
    load_vec_nodes = np.zeros_like(der_lambda)
    df_dv_nodes = np.zeros_like(der_lambda)

    for i in range(le):
        v_init_abs = np.abs(branch_pit[i][VINIT])
        v_init2 = v_init_abs * branch_pit[i][VINIT]
        lambda_term = np.divide(branch_pit[i][LENGTH] * branch_pit[i][LAMBDA], branch_pit[i][D]) \
            + branch_pit[i][LC]
        const_p_term = np.divide(branch_pit[i][RHO], P_CONVERSION * 2)
        df_dv[i] = const_p_term * (2 * v_init_abs * lambda_term + der_lambda[i]
                                   * np.divide(branch_pit[i][LENGTH], branch_pit[i][D]) * v_init2)
        load_vec[i] = p_init_i_abs[i] - p_init_i1_abs[i] + branch_pit[i][PL] \
            + const_p_term * (GRAVITATION_CONSTANT * 2 * height_difference[i]
                              - v_init2 * lambda_term)
        mass_flow_dv = branch_pit[i][RHO] * branch_pit[i][AREA]
        df_dv_nodes[i] = mass_flow_dv
        load_vec_nodes[i] = mass_flow_dv * branch_pit[i][VINIT]
    return load_vec, load_vec_nodes, df_dv, df_dv_nodes, df_dp, df_dp1


@jit((float64[:, :], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],
      float64[:], float64[:]), nopython=True, cache=False)
def derivatives_hydraulic_comp_numba(branch_pit, lambda_, der_lambda, p_init_i_abs, p_init_i1_abs,
                                     height_difference, comp_fact, der_comp, der_comp1):
    le = lambda_.shape[0]
    load_vec = np.zeros_like(lambda_)
    df_dv = np.zeros_like(lambda_)
    df_dp = np.zeros_like(lambda_)
    df_dp1 = np.zeros_like(lambda_) * (-1)
    load_vec_nodes = np.zeros_like(der_lambda)
    df_dv_nodes = np.zeros_like(der_lambda)

    # Formulas for gas pressure loss according to laminar version
    for i in range(le):
        # compressibility settings
        v_init_abs = np.abs(branch_pit[i][VINIT])
        v_init2 = branch_pit[i][VINIT] * v_init_abs
        p_diff = p_init_i_abs[i] - p_init_i1_abs[i]
        p_sum = p_init_i_abs[i] + p_init_i1_abs[i]
        p_sum_div = np.divide(1, p_sum)

        const_lambda = np.divide(NORMAL_PRESSURE * branch_pit[i][RHO] * branch_pit[i][TINIT],
                                 NORMAL_TEMPERATURE * P_CONVERSION)
        const_height = np.divide(
            branch_pit[i][RHO] * NORMAL_TEMPERATURE * GRAVITATION_CONSTANT * height_difference[i],
            2 * NORMAL_PRESSURE * branch_pit[i][TINIT] * P_CONVERSION)
        friction_term = np.divide(lambda_[i] * branch_pit[i][LENGTH],
                                  branch_pit[i][D]) + branch_pit[i][LC]

        load_vec[i] = p_diff + branch_pit[i][PL] + const_height * p_sum \
            - const_lambda * comp_fact[i] * v_init2 * friction_term * p_sum_div

        p_deriv = const_lambda * v_init2 * friction_term * p_sum_div
        df_dp[i] = -1. + p_deriv * (der_comp[i] - comp_fact[i] * p_sum_div) + const_height
        df_dp1[i] = 1. + p_deriv * (der_comp1[i] - comp_fact[i] * p_sum_div) + const_height

        df_dv[i] = np.divide(2 * const_lambda * comp_fact[i], p_sum) * v_init_abs * friction_term\
            + np.divide(const_lambda * comp_fact[i] * der_lambda[i] * branch_pit[i][LENGTH]
                        * v_init2, p_sum * branch_pit[i][D])
        mass_flow_dv = branch_pit[i][RHO] * branch_pit[i][AREA]
        df_dv_nodes[i] = mass_flow_dv
        load_vec_nodes[i] = mass_flow_dv * branch_pit[i][VINIT]
    return load_vec, load_vec_nodes, df_dv, df_dv_nodes, df_dp, df_dp1


@jit((float64[:], float64[:], float64[:], float64[:], float64[:]), nopython=True)
def calc_lambda_nikuradse_incomp_numba(v, d, k, eta, rho):
    lambda_nikuradse = np.empty_like(v)
    lambda_laminar = np.zeros_like(v)
    re = np.empty_like(v)
    v_abs = np.abs(v)
    for i in range(v.shape[0]):
        if v_abs[i] < 1e-6:
            re[i] = np.divide(rho[i] * 1e-6 * d[i], eta[i])
        else:
            re[i] = np.divide(rho[i] * v_abs[i] * d[i], eta[i])
        if v[i] != 0:
            lambda_laminar[i] = 64 / re[i]
        lambda_nikuradse[i] = np.power(-2 * np.log10(k[i] / (3.71 * d[i])), -2)
    return re, lambda_laminar, lambda_nikuradse


@jit((float64[:], float64[:], float64[:], float64[:], float64[:]), nopython=True)
def calc_lambda_nikuradse_comp_numba(v, d, k, eta, rho):
    lambda_nikuradse = np.empty_like(v)
    lambda_laminar = np.zeros_like(v)
    re = np.empty_like(v)
    for i in range(len(v)):
        v_abs = np.abs(v[i])
        if v_abs < 1e-6:
            re[i] = np.divide(rho[i] * 1e-6 * d[i], eta[i])
        else:
            re[i] = np.divide(rho[i] * v_abs * d[i], eta[i])
        if v[i] != 0:
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
    lambda_cb_old = np.empty_like(lambda_nikuradse)
    converged = False
    niter = 0
    factor = np.log(10) * 2.51

    # Inner Newton-loop for calculation of lambda
    while not converged and niter < max_iter:
        for i in range(len(lambda_cb)):
            sqt = np.sqrt(lambda_cb[i])
            add_val = np.divide(k[i], (3.71 * d[i]))
            sqt_div = np.divide(1, sqt)
            re_div = np.divide(1, re[i])
            sqt_div3 = sqt_div ** 3

            f = sqt_div + 2 * np.log10(2.51 * re_div * sqt_div + add_val)
            df_dlambda_cb = - 0.5 * sqt_div3 - 2.51 * re_div * sqt_div3 * np.divide(
                re[i] * sqt + add_val, factor)
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
