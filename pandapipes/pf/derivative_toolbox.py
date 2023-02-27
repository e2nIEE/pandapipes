# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import linalg

from pandapipes.constants import P_CONVERSION, GRAVITATION_CONSTANT, NORMAL_PRESSURE, \
    NORMAL_TEMPERATURE
from pandapipes.idx_branch import LENGTH, LAMBDA, D, LOSS_COEFFICIENT as LC, RHO, PL, AREA, TINIT, \
    VINIT
from pandapipes.idx_node import HEIGHT, PINIT, PAMB, TINIT as TINIT_NODE


def derivatives_hydraulic_incomp_np(branch_pit, der_lambda, p_init_i_abs, p_init_i1_abs,
                                    height_difference):
    v_init_abs = np.abs(branch_pit[:, VINIT])
    v_init2 = v_init_abs * branch_pit[:, VINIT]
    lambda_term = np.divide(branch_pit[:, LENGTH] * branch_pit[:, LAMBDA], branch_pit[:, D]) \
                  + branch_pit[:, LC]
    const_p_term = np.divide(branch_pit[:, RHO], P_CONVERSION * 2)
    df_dv = const_p_term * (2 * v_init_abs * lambda_term + der_lambda
                            * np.divide(branch_pit[:, LENGTH], branch_pit[:, D]) * v_init2)
    load_vec = p_init_i_abs - p_init_i1_abs + branch_pit[:, PL] \
               + const_p_term * (GRAVITATION_CONSTANT * 2 * height_difference
                                 - v_init2 * lambda_term)
    mass_flow_dv = branch_pit[:, RHO] * branch_pit[:, AREA]
    df_dv_nodes = mass_flow_dv
    load_vec_nodes = mass_flow_dv * branch_pit[:, VINIT]
    df_dp = np.ones_like(der_lambda) * (-1)
    df_dp1 = np.ones_like(der_lambda)
    return load_vec, load_vec_nodes, df_dv, df_dv_nodes, df_dp, df_dp1


def derivatives_hydraulic_comp_np(branch_pit, lambda_, der_lambda, p_init_i_abs, p_init_i1_abs,
                                  height_difference, comp_fact, der_comp, der_comp1):
    # Formulas for gas pressure loss according to laminar version
    v_init_abs = np.abs(branch_pit[:, VINIT])
    v_init2 = branch_pit[:, VINIT] * v_init_abs
    p_diff = p_init_i_abs - p_init_i1_abs
    p_sum = p_init_i_abs + p_init_i1_abs
    p_sum_div = np.divide(1, p_sum)

    const_lambda = np.divide(NORMAL_PRESSURE * branch_pit[:, RHO] * branch_pit[:, TINIT],
                             NORMAL_TEMPERATURE * P_CONVERSION)
    const_height = np.divide(
        branch_pit[:, RHO] * NORMAL_TEMPERATURE * GRAVITATION_CONSTANT * height_difference,
        2 * NORMAL_PRESSURE * branch_pit[:, TINIT] * P_CONVERSION)
    friction_term = np.divide(lambda_ * branch_pit[:, LENGTH], branch_pit[:, D]) + branch_pit[:, LC]

    load_vec = p_diff + branch_pit[:, PL] + const_height * p_sum \
               - const_lambda * comp_fact * v_init2 * friction_term * p_sum_div

    p_deriv = const_lambda * v_init2 * friction_term * p_sum_div
    df_dp = -1. + p_deriv * (der_comp - comp_fact * p_sum_div) + const_height
    df_dp1 = 1. + p_deriv * (der_comp1 - comp_fact * p_sum_div) + const_height

    df_dv = np.divide(2 * const_lambda * comp_fact, p_sum) * v_init_abs * friction_term \
            + np.divide(const_lambda * comp_fact * der_lambda * branch_pit[:, LENGTH] * v_init2,
                        p_sum * branch_pit[:, D])
    mass_flow_dv = branch_pit[:, RHO] * branch_pit[:, AREA]
    df_dv_nodes = mass_flow_dv
    load_vec_nodes = mass_flow_dv * branch_pit[:, VINIT]

    return load_vec, load_vec_nodes, df_dv, df_dv_nodes, df_dp, df_dp1


def calc_lambda_nikuradse_incomp_np(v, d, k, eta, rho):
    v_abs = np.abs(v)
    v_abs[v_abs < 1e-6] = 1e-6
    re = np.divide(rho * v_abs * d, eta)
    lambda_laminar = np.zeros_like(v)
    lambda_laminar[v != 0] = 64 / re[v != 0]
    lambda_nikuradse = np.divide(1, (-2 * np.log10(k / (3.71 * d))) ** 2)
    return re, lambda_laminar, lambda_nikuradse


def calc_lambda_nikuradse_comp_np(v, d, k, eta, rho):
    v_abs = np.abs(v)
    v_abs[v_abs < 1e-6] = 1e-6
    re = np.divide(rho * v_abs * d, eta)
    lambda_laminar = np.zeros_like(v)
    lambda_laminar[v != 0] = 64 / re[v != 0]
    lambda_nikuradse = np.divide(1, (2 * np.log10(d / k) + 1.14) ** 2)
    return re, lambda_laminar, lambda_nikuradse


def calc_medium_pressure_with_derivative_np(p_init_i_abs, p_init_i1_abs):
    val = 2 / 3
    p_m = p_init_i_abs.copy()
    der_p_m = np.ones_like(p_init_i_abs)
    der_p_m1 = np.ones_like(p_init_i_abs) * (-1)
    p_differs = p_init_i_abs != p_init_i1_abs

    if not np.any(p_differs):
        return p_m, der_p_m, der_p_m1

    p_sq = p_init_i_abs[p_differs] ** 2
    p1_sq = p_init_i1_abs[p_differs] ** 2
    diff_p_sq = p_sq - p1_sq
    diff_p_sq_div = np.divide(1, diff_p_sq)
    diff_p_cub = p_init_i_abs[p_differs] ** 3 - p_init_i1_abs[p_differs] ** 3
    factor = diff_p_sq_div ** 2 * val

    p_m[p_differs] = val * diff_p_cub * diff_p_sq_div
    der_p_m[p_differs] = (3 * p_sq * diff_p_sq - 2 * p_init_i_abs[p_differs] * diff_p_cub) * factor
    der_p_m1[p_differs] = (-3 * p1_sq * diff_p_sq + 2 * p_init_i1_abs[p_differs] * diff_p_cub) \
                          * factor

    return p_m, der_p_m, der_p_m1


def colebrook_np(re, d, k, lambda_nikuradse, dummy, max_iter):
    """

    :param re:
    :type re:
    :param d:
    :type d:
    :param k:
    :type k:
    :param lambda_nikuradse:
    :type lambda_nikuradse:
    :param dummy:
    :type dummy:
    :param max_iter:
    :type max_iter:
    :return: lambda_cb
    :rtype:
    """
    lambda_cb = lambda_nikuradse
    converged = False
    error_lambda = []
    niter = 0

    # Inner Newton-loop for calculation of lambda
    while not converged and niter < max_iter:
        f = lambda_cb ** (-1 / 2) + 2 * np.log10(2.51 / (re * np.sqrt(lambda_cb)) + k / (3.71 * d))

        df_dlambda_cb = (-1 / 2 * lambda_cb ** (-3 / 2)) - (2.51 / re) * lambda_cb ** (-3 / 2) \
                        / (np.log(10) * 2.51 / (re * np.sqrt(lambda_cb) + k / (3.71 * d)))

        x = - f / df_dlambda_cb

        lambda_cb_old = lambda_cb
        lambda_cb = lambda_cb + x

        dx = np.abs(lambda_cb - lambda_cb_old) * dummy
        error_lambda.append(linalg.norm(dx) / (len(dx)))

        if error_lambda[niter] <= 1e-4:
            converged = True

        niter += 1

    return converged, lambda_cb


def calc_derived_values_np(node_pit, from_nodes, to_nodes):
    tinit_branch = (node_pit[from_nodes, TINIT_NODE] + node_pit[to_nodes, TINIT_NODE]) / 2
    height_difference = node_pit[from_nodes, HEIGHT] - node_pit[to_nodes, HEIGHT]
    p_init_i_abs = node_pit[from_nodes, PINIT] + node_pit[from_nodes, PAMB]
    p_init_i1_abs = node_pit[to_nodes, PINIT] + node_pit[to_nodes, PAMB]
    return tinit_branch, height_difference, p_init_i_abs, p_init_i1_abs
