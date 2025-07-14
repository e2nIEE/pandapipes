# Copyright (c) 2020-2025 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import linalg

from pandapipes.constants import P_CONVERSION, GRAVITATION_CONSTANT, NORMAL_PRESSURE, \
    NORMAL_TEMPERATURE
from pandapipes.idx_branch import LENGTH, LAMBDA, D, LOSS_COEFFICIENT as LC, PL, AREA, \
    MDOTINIT, TOUTINIT, FROM_NODE
from pandapipes.idx_node import HEIGHT, PINIT, PAMB, TINIT as TINIT_NODE


def derivatives_hydraulic_incomp_np(branch_pit, der_lambda, p_init_i_abs, p_init_i1_abs,
                                    height_difference, rho):
    # Formulas for pressure loss in incompressible flow
    # Use medium density ((rho_from + rho_to) / 2) for Darcy Weisbach according to
    # https://www.schweizer-fn.de/rohr/rohrleitung/rohrleitung.php#fluessigkeiten
    m_init_abs = np.abs(branch_pit[:, MDOTINIT])
    m_init2 = m_init_abs * branch_pit[:, MDOTINIT]
    p_diff = p_init_i_abs - p_init_i1_abs
    const_height = rho * GRAVITATION_CONSTANT * height_difference / P_CONVERSION
    friction_term = np.divide(branch_pit[:, LENGTH] * branch_pit[:, LAMBDA], branch_pit[:, D]) + branch_pit[:, LC]
    const_term = np.divide(1, branch_pit[:, AREA] ** 2 * rho * P_CONVERSION * 2)

    df_dm = - const_term * (2 * m_init_abs * friction_term + der_lambda
                            * np.divide(branch_pit[:, LENGTH], branch_pit[:, D]) * m_init2)

    load_vec = p_diff + branch_pit[:, PL] + const_height - const_term * m_init2 * friction_term

    df_dp = np.ones_like(der_lambda)
    df_dp1 = np.ones_like(der_lambda) * (-1)

    df_dm_nodes = np.ones_like(der_lambda)

    load_vec_nodes_from = branch_pit[:, MDOTINIT]
    load_vec_nodes_to = branch_pit[:, MDOTINIT]

    return load_vec, load_vec_nodes_from, load_vec_nodes_to, df_dm, df_dm_nodes, df_dp, df_dp1


def derivatives_hydraulic_comp_np(node_pit, branch_pit, lambda_, der_lambda, p_init_i_abs, p_init_i1_abs,
                                  height_difference, comp_fact, der_comp, der_comp1, rho, rho_n):
    # Formulas for gas pressure loss according to laminar version
    m_init_abs = np.abs(branch_pit[:, MDOTINIT])
    m_init2 = branch_pit[:, MDOTINIT] * m_init_abs
    p_diff = p_init_i_abs - p_init_i1_abs
    p_sum = p_init_i_abs + p_init_i1_abs
    p_sum_div = np.divide(1, p_sum)
    from_nodes = branch_pit[:, FROM_NODE].astype(np.int32)
    tm = (node_pit[from_nodes, TINIT_NODE] + branch_pit[:, TOUTINIT]) / 2
    const_height = rho * GRAVITATION_CONSTANT * height_difference / P_CONVERSION
    friction_term = np.divide(lambda_ * branch_pit[:, LENGTH], branch_pit[:, D]) + branch_pit[:, LC]
    normal_term = np.divide(NORMAL_PRESSURE, NORMAL_TEMPERATURE * P_CONVERSION * rho_n * branch_pit[:, AREA] ** 2)

    const_term_p = normal_term * m_init2 * friction_term * tm
    df_dp = 1. - const_term_p * p_sum_div * (der_comp - comp_fact * p_sum_div)
    df_dp1 = -1. - const_term_p * p_sum_div * (der_comp1 - comp_fact * p_sum_div)

    const_term_m = normal_term * p_sum_div * tm * comp_fact
    df_dm = - const_term_m * (2 * m_init_abs * friction_term +
                            np.divide(der_lambda * branch_pit[:, LENGTH] * m_init2, branch_pit[:, D]))

    load_vec = p_diff + branch_pit[:, PL] + const_height \
               - normal_term * comp_fact * m_init2 * friction_term * p_sum_div * tm

    df_dm_nodes = np.ones_like(lambda_)

    load_vec_nodes_from = branch_pit[:, MDOTINIT]
    load_vec_nodes_to = branch_pit[:, MDOTINIT]

    return load_vec, load_vec_nodes_from, load_vec_nodes_to, df_dm, df_dm_nodes, df_dp, df_dp1


def calc_lambda_nikuradse_incomp_np(m, d, k, eta, area):
    m_abs = np.abs(m)
    re = np.divide(m_abs * d, eta * area)
    lambda_laminar = np.zeros_like(m)
    lambda_laminar[~np.isclose(re, 0)] = 64 / re[~np.isclose(re, 0)]
    lambda_nikuradse = np.divide(1, (-2 * np.log10(k / (3.71 * d))) ** 2)
    return re, lambda_laminar, lambda_nikuradse


def calc_lambda_nikuradse_comp_np(m, d, k, eta, area):
    m_abs = np.abs(m)
    re = np.divide(m_abs * d, eta * area)
    lambda_laminar = np.zeros_like(m)
    lambda_laminar[~np.isclose(re, 0)] = 64 / re[~np.isclose(re, 0)]
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
    mask = ~np.isclose(re, 0)
    f = np.zeros_like(lambda_cb)
    df = np.zeros_like(lambda_cb)
    x = np.zeros_like(lambda_cb)
    re_nz = re[mask]
    k_nz = k[mask]
    d_nz = d[mask]
    # Inner Newton-loop for calculation of lambda
    while not converged and niter < max_iter:

        f[mask] = lambda_cb[mask] ** (-1 / 2) + 2 * np.log10(2.51 / (re_nz * np.sqrt(lambda_cb[mask])) + k_nz / (3.71 * d_nz))

        df[mask]= -1 / 2 * lambda_cb[mask] ** (-3 / 2) - (2.51 / re_nz) * lambda_cb[mask] ** (-3 / 2) \
                        / (np.log(10) * (2.51 / (re_nz * np.sqrt(lambda_cb[mask])) + k_nz / (3.71 * d_nz)))

        x[mask] = - f[mask] / df[mask]

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
