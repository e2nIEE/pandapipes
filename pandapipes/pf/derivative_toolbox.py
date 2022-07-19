# Copyright (c) 2020-2021 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import linalg

from pandapipes.constants import P_CONVERSION, GRAVITATION_CONSTANT, NORMAL_PRESSURE, \
    NORMAL_TEMPERATURE


def derivatives_hydraulic_incomp_np(pit_cols, branch_pit, der_lambda, p_init_i_abs, p_init_i1_abs,
                                    height_difference):
    v_init_abs = np.abs(branch_pit[:, pit_cols[0]])
    v_init2 = v_init_abs * branch_pit[:, pit_cols[0]]
    lambda_term = np.divide(branch_pit[:, pit_cols[1]] *
                            branch_pit[:, pit_cols[2]],
                            branch_pit[:, pit_cols[3]]) + \
                            branch_pit[:, pit_cols[4]]
    const_p_term = np.divide(branch_pit[:, pit_cols[5]], P_CONVERSION * 2)
    df_dv = const_p_term * (2 * v_init_abs * lambda_term + der_lambda
                            * np.divide(branch_pit[:, pit_cols[1]],
                                        branch_pit[:, pit_cols[3]]) * v_init2)
    load_vec = p_init_i_abs - p_init_i1_abs + branch_pit[:, pit_cols[6]] \
               + const_p_term * (GRAVITATION_CONSTANT * 2 * height_difference
                                 - v_init2 * lambda_term)
    mass_flow_dv = branch_pit[:, pit_cols[5]] * \
                   branch_pit[:, pit_cols[7]]
    df_dv_nodes = mass_flow_dv
    load_vec_nodes = mass_flow_dv * branch_pit[:, pit_cols[0]]
    df_dp = np.ones_like(der_lambda) * (-1)
    df_dp1 = np.ones_like(der_lambda)
    return load_vec, load_vec_nodes, df_dv, df_dv_nodes, df_dp, df_dp1


def derivatives_hydraulic_comp_np(pit_cols, branch_pit, lambda_, der_lambda, p_init_i_abs, p_init_i1_abs,
                                  height_difference, comp_fact, der_comp, der_comp1):
    # Formulas for gas pressure loss according to laminar version
    v_init_abs = np.abs(branch_pit[:, pit_cols[0]])
    v_init2 = branch_pit[:, pit_cols[0]] * v_init_abs
    p_diff = p_init_i_abs - p_init_i1_abs
    p_sum = p_init_i_abs + p_init_i1_abs
    p_sum_div = np.divide(1, p_sum)

    const_lambda = np.divide(NORMAL_PRESSURE *
                             branch_pit[:, pit_cols[5]] *
                             branch_pit[:, pit_cols[8]],
                             NORMAL_TEMPERATURE * P_CONVERSION)
    const_height = np.divide(
        branch_pit[:, pit_cols[5]] * NORMAL_TEMPERATURE * GRAVITATION_CONSTANT * height_difference,
        2 * NORMAL_PRESSURE * branch_pit[:, pit_cols[8]] * P_CONVERSION)
    friction_term = np.divide(lambda_ * branch_pit[:, pit_cols[1]],
                              branch_pit[:, pit_cols[3]]) + \
                              branch_pit[:, pit_cols[4]]

    load_vec = p_diff + branch_pit[:, pit_cols[6]] + const_height * p_sum \
               - const_lambda * comp_fact * v_init2 * friction_term * p_sum_div

    p_deriv = const_lambda * v_init2 * friction_term * p_sum_div
    df_dp = -1. + p_deriv * (der_comp - comp_fact * p_sum_div) + const_height
    df_dp1 = 1. + p_deriv * (der_comp1 - comp_fact * p_sum_div) + const_height

    df_dv = np.divide(2 * const_lambda * comp_fact, p_sum) * v_init_abs * friction_term \
            + np.divide(const_lambda * comp_fact * der_lambda * branch_pit[:, pit_cols[1]] * v_init2,
                        p_sum * branch_pit[:, pit_cols[3]])
    mass_flow_dv = branch_pit[:, pit_cols[5]] * branch_pit[:, pit_cols[7]]
    df_dv_nodes = mass_flow_dv
    load_vec_nodes = mass_flow_dv * branch_pit[:, pit_cols[0]]

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


def calc_derived_values_np(pit_cols, node_pit, from_nodes, to_nodes):
    tinit_branch = (node_pit[from_nodes, pit_cols[0]] +
                    node_pit[to_nodes, pit_cols[0]]) / 2
    height_difference = node_pit[from_nodes, pit_cols[1]] - \
                        node_pit[to_nodes, pit_cols[1]]
    p_init_i_abs = node_pit[from_nodes, pit_cols[2]] + \
                   node_pit[from_nodes, pit_cols[3]]
    p_init_i1_abs = node_pit[to_nodes, pit_cols[2]] + \
                    node_pit[to_nodes, pit_cols[3]]
    return tinit_branch, height_difference, p_init_i_abs, p_init_i1_abs


def get_derivative_density_diff(mass_fraction, density_list):
    rho_prod = np.prod(density_list, axis=1)
    shape = np.shape(mass_fraction)
    loop = np.arange(0, shape[1])
    nom = np.zeros(shape[0])
    rho_select = np.zeros(shape)
    for i in loop:
        select = loop != i
        nom += mass_fraction[:, i] * np.prod(density_list[:, select], axis=1)
        rho_select[:, i] += np.prod(density_list[:, select], axis=1)
    res = -rho_prod[:, np.newaxis] * rho_select * nom[:, np.newaxis] ** -2
    return res


def get_derivative_density_same(mass_fraction, density_list):
    rho_prod = np.prod(density_list, axis=1)
    shape = np.shape(mass_fraction)
    loop = np.arange(0, shape[1])
    nom = np.zeros(shape[0])
    rho_select = np.zeros(shape)
    for i in loop:
        select = loop != i
        nom += mass_fraction[:, i] * np.prod(density_list[:, select], axis=1)
        rho_select[:, i] += np.prod(density_list[:, select], axis=1) * mass_fraction[:, i]
    res = rho_prod[:, np.newaxis] * (-rho_select+nom[:, np.newaxis]) * nom[:, np.newaxis] ** -2
    return res