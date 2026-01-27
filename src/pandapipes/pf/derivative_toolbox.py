# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import logging
import numpy as np
from numpy import linalg
from pandapipes.pf.internals_toolbox import _sum_by_group
from pandapipes.constants import P_CONVERSION, GRAVITATION_CONSTANT, NORMAL_PRESSURE, \
    NORMAL_TEMPERATURE
from pandapipes.idx_branch import LENGTH, LAMBDA, D, LOSS_COEFFICIENT as LC, PL, AREA, \
    MDOTINIT, TOUTINIT, FROM_NODE, TEXT, ALPHA, TL, QEXT, T_OUT_OLD
from pandapipes.idx_node import HEIGHT, PINIT, PAMB, TINIT as TINIT_NODE, LOAD, MDOTSLACKINIT, TINIT_OLD

logger = logging.getLogger(__name__)


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

def derivatives_thermal_np(node_pit, branch_pit,
                              from_nodes, to_nodes,
                              t_init_i, t_init_i1, t_init_n,
                              cp_i, cp_i1, cp_n, cp,
                              rho, dt, transient,
                              amb):
    # this is not required currently, but useful when implementing leakages
    # m_init_i = np.abs(branch_pit[:, MDOTINIT])
    # m_init_i1 = np.abs(branch_pit[:, MDOTINIT])
    mdot = np.abs(branch_pit[:, MDOTINIT])
    t_amb = branch_pit[:, TEXT]
    length = branch_pit[:, LENGTH]
    alpha = branch_pit[:, ALPHA] * np.pi * branch_pit[:, D]
    tl = branch_pit[:, TL]
    qext = branch_pit[:, QEXT]

    branches_flow = _branches_not_zero_flow(branch_pit)
    # ToDo: Is it relevant to consider slack streams?
    nodes_flow = np.isin(np.arange(len(node_pit)),
                         np.concatenate([from_nodes[branches_flow], to_nodes[branches_flow]]))

    fn = np.zeros_like(cp_n)
    fn[nodes_flow] = (node_pit[nodes_flow, LOAD] * cp_n[nodes_flow] * t_init_n[nodes_flow] +
                      node_pit[nodes_flow, MDOTSLACKINIT] * cp_n[nodes_flow] * t_init_n[nodes_flow])
    fn[~nodes_flow] = amb - t_init_n[~nodes_flow]

    dfn_dt = np.zeros_like(cp_n)
    dfn_dt[nodes_flow] = - node_pit[nodes_flow, LOAD] * cp_n[nodes_flow]
    dfn_dt[~nodes_flow] = np.ones_like(cp_n[~nodes_flow])
    dfn_dts = - node_pit[:, MDOTSLACKINIT] * cp_n

    fbf = mdot * t_init_i * cp_i
    fbt = mdot * t_init_i1 * cp_i1
    dfbf_dt = - mdot * cp_i
    dfbt_dtout = mdot * cp_i1

    if transient:
        area = branch_pit[:, AREA]
        tvor = branch_pit[:, T_OUT_OLD]

        fb = (
                rho * area * cp * (t_init_i1 - tvor) * (1 / dt) * length
                + cp * mdot * (-t_init_i + t_init_i1 - tl)
                - alpha * (t_amb - t_init_i1) * length + qext
        )

        dfb_dt = - cp * mdot
        dfb_dtout = rho * area * cp / dt * length + cp * mdot + alpha

        if np.any(~branches_flow):
            # TODO: maybe replace this statement with a component lookup
            zero_length = np.isclose(branch_pit[:, LENGTH], 0, rtol=1e-6, atol=1e-10)
            mask = zero_length & ~branches_flow
            if np.any(mask):
                fb[mask] = (
                        rho[mask] * area[mask] * cp[mask] * (t_init_i1[mask] - tvor[mask]) * (1 / dt)
                        - alpha[mask] * (t_amb[mask] - t_init_i1[mask]) + qext[mask]
                )
                dfb_dt[mask] = 0
                dfb_dtout[mask] = (rho[mask] * area[mask] * cp[mask] / dt +
                                                     alpha[mask])

        if np.any(~nodes_flow):
            fn_zero = ~nodes_flow[from_nodes]
            tn_zero = ~nodes_flow[to_nodes]

            t_from_node_vor_zero = node_pit[from_nodes[fn_zero], TINIT_OLD]
            t_to_node_vor_zero = node_pit[to_nodes[tn_zero], TINIT_OLD]
            t_to_node = node_pit[to_nodes[tn_zero], TINIT_NODE]

            fn_eq = (rho[fn_zero] * area[fn_zero] * cp[fn_zero] * (1 / dt)
                     * (t_init_i[fn_zero] - t_from_node_vor_zero)
                     - alpha[fn_zero] * (t_amb[fn_zero] - t_init_i[fn_zero]))

            tn_eq = (rho[tn_zero] * area[tn_zero] * cp[tn_zero] * (1 / dt)
                     * (t_to_node - t_to_node_vor_zero)
                     - alpha[tn_zero] * (t_amb[tn_zero] - t_to_node))

            fn_deriv = (rho[fn_zero] * area[fn_zero] * cp[fn_zero] * (1 / dt) + alpha[fn_zero])
            tn_deriv = (rho[tn_zero] * area[tn_zero] * cp[tn_zero] * (1 / dt) + alpha[tn_zero])

            fn_nodes, fn_eq_sum, fn_deriv_sum= _sum_by_group(False,
                from_nodes[fn_zero], fn_eq, fn_deriv
            )

            tn_nodes, tn_eq_sum, tn_deriv_sum = _sum_by_group(False,
                to_nodes[tn_zero], tn_eq, tn_deriv
            )

            fn[~nodes_flow] = 0
            fn[fn_nodes] += fn_eq_sum
            fn[tn_nodes] += tn_eq_sum
            dfn_dt[~nodes_flow] = 0
            dfn_dt[fn_nodes] -= fn_deriv_sum
            dfn_dt[tn_nodes] -= tn_deriv_sum
            dfn_dts[~nodes_flow] = 0

            fbf[fn_zero] = 0
            fbt[tn_zero] = 0
            dfbf_dt[fn_zero] = 0
            dfbt_dtout[tn_zero] = 0
    else:
        non_zero_length_mask = ~np.isclose(branch_pit[:, LENGTH], 0, rtol=1e-6, atol=1e-10)
        if np.any(non_zero_length_mask & (np.abs(branch_pit[:, QEXT]) > 1e-12)):
            logger.warning(
                "A branch with non zero length has a non zero external heat load. This might lead "
                "to errors in the calculation, as the overlap of temperature reduction from heat "
                "losses to ambient and a constant heat flux cannot be solved with the implmented "
                "method."
            )

        fb = np.zeros_like(cp)
        fb[branches_flow] = (
                t_amb[branches_flow] + (t_init_i[branches_flow]  - t_amb[branches_flow])
                * np.exp(- alpha[branches_flow] * length[branches_flow]  / (cp[branches_flow] * mdot[branches_flow]))
                - t_init_i1[branches_flow] + tl[branches_flow]
                - qext[branches_flow] / (cp[branches_flow] * mdot[branches_flow])
        )
        fb[~branches_flow] = amb - t_init_i1[~branches_flow]

        dfb_dt = np.zeros_like(cp)
        dfb_dt[branches_flow] = np.exp(- alpha[branches_flow] * length[branches_flow] /
                                       (cp[branches_flow] * mdot[branches_flow]))

        dfb_dtout = - np.ones_like(cp)

    infeed = np.setdiff1d(from_nodes[branches_flow], to_nodes[branches_flow])


    # This approach can be used if you consider the effect of sources with given temperature (checkout issue #656)

    # branch_pit[:, LOAD_VEC_NODES_FROM_T] = mdot * t_init_i * cp_i
    # --> cp_i is calculated by fluid.get_heat_capacity(t_init_i)
    # branch_pit[:, LOAD_VEC_NODES_TO_T] = mdot * t_init_i1 * cp_i1
    # --> still missing is the derivative of loads
    # t_init = node_pit[:, TINIT_NODE]
    # cp_n = fluid.get_heat_capacity(t_init)
    # node_pit[:, LOAD_T] = cp_n * node_pit[:, LOAD] * t_init

    return fn, dfn_dt, dfn_dts, fb, dfb_dt, dfb_dtout, fbf, fbt, dfbf_dt, dfbt_dtout, infeed


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

def _branches_not_zero_flow(branch_pit):
    """
    Simple function to identify branches with flow based on the calculated velocity.

    :param branch_pit: The pandapipes internal table of the network (including hydraulics results)
    :type branch_pit: np.array
    :return: branches_connected_flow - lookup array if branch is connected wrt. flow
    :rtype: np.array
    """
    # TODO: is this formulation correct or could there be any caveats?
    return ~np.isnan(branch_pit[:, MDOTINIT]) & ~np.isclose(branch_pit[:, MDOTINIT], 0, rtol=1e-10, atol=1e-10)