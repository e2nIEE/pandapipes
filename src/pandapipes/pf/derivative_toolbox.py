# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import logging
import numpy as np
from scipy.optimize import newton
from pandapipes.pf.internals_toolbox import _sum_by_group, branch_area
from pandapipes.pf.pipeflow_setup import branches_not_zero_flow
from pandapipes.constants import P_CONVERSION, GRAVITATION_CONSTANT, NORMAL_PRESSURE, \
    NORMAL_TEMPERATURE
from pandapipes.idx_branch import IdxBranch
from pandapipes.idx_node import IdxNode

logger = logging.getLogger(__name__)


def derivatives_hydraulic_incomp_np(branch_pit, der_lambda, p_init_i_abs, p_init_i1_abs,
                                    height_difference, rho, area):
    # Formulas for pressure loss in incompressible flow
    # Use medium density ((rho_from + rho_to) / 2) for Darcy Weisbach according to
    # https://www.schweizer-fn.de/rohr/rohrleitung/rohrleitung.php#fluessigkeiten
    m_init_abs = np.abs(branch_pit[:, IdxBranch.MDOTINIT])
    m_abs_deriv = np.maximum(m_init_abs, 1e-8)
    m_init2 = m_init_abs * branch_pit[:, IdxBranch.MDOTINIT]
    p_diff = p_init_i_abs - p_init_i1_abs
    length = branch_pit[:, IdxBranch.LENGTH]
    lambd = branch_pit[:, IdxBranch.LAMBDA]
    lc = branch_pit[:, IdxBranch.LOSS_COEFFICIENT]
    pl = branch_pit[:, IdxBranch.PL]

    d = branch_pit[:, IdxBranch.D]

    const_height = rho * GRAVITATION_CONSTANT * height_difference / P_CONVERSION
    friction_term = length * lambd / d + lc
    const_term = 1 / (area ** 2 * rho * P_CONVERSION * 2)

    df_dm = - const_term * (2 * m_abs_deriv * friction_term + der_lambda
                            * length / d * m_init2)

    load_vec = p_diff + pl + const_height - const_term * m_init2 * friction_term

    df_dp = np.ones_like(der_lambda)
    df_dp1 = np.ones_like(der_lambda) * (-1)

    df_dm_nodes = np.ones_like(der_lambda)

    load_vec_nodes_from = branch_pit[:, IdxBranch.MDOTINIT]
    load_vec_nodes_to = branch_pit[:, IdxBranch.MDOTINIT]

    dp_frict_loss = const_term * m_init2 * friction_term

    return load_vec, load_vec_nodes_from, load_vec_nodes_to, df_dm, df_dm_nodes, df_dp, df_dp1, dp_frict_loss


def derivatives_hydraulic_comp_np(node_pit, branch_pit, lambda_, der_lambda, p_init_i_abs, p_init_i1_abs,
                                  height_difference, comp_fact, der_comp, der_comp1, rho, rho_n, area):
    # Formulas for gas pressure loss according to laminar version
    m_init_abs = np.abs(branch_pit[:, IdxBranch.MDOTINIT])
    m_abs_deriv = np.maximum(m_init_abs, 1e-8)
    m_init2 = branch_pit[:, IdxBranch.MDOTINIT] * m_init_abs
    p_diff = p_init_i_abs - p_init_i1_abs
    p_sum = p_init_i_abs + p_init_i1_abs
    p_sum_div = np.divide(1, p_sum)
    from_nodes = branch_pit[:, IdxBranch.FROM_NODE].astype(np.int32)
    tm = (node_pit[from_nodes, IdxNode.TINIT] + branch_pit[:, IdxBranch.TOUTINIT]) / 2
    const_height = rho * GRAVITATION_CONSTANT * height_difference / P_CONVERSION
    friction_term = np.divide(lambda_ * branch_pit[:, IdxBranch.LENGTH], branch_pit[:, IdxBranch.D]) + branch_pit[:, IdxBranch.LOSS_COEFFICIENT]
    normal_term = np.divide(NORMAL_PRESSURE, NORMAL_TEMPERATURE * P_CONVERSION * rho_n)
    const_term = normal_term / area ** 2

    const_term_p = const_term * m_init2 * friction_term * tm
    df_dp = 1. - const_term_p * p_sum_div * (der_comp - comp_fact * p_sum_div)
    df_dp1 = -1. - const_term_p * p_sum_div * (der_comp1 - comp_fact * p_sum_div)

    const_term_m = const_term * p_sum_div * tm * comp_fact
    df_dm = - const_term_m * (2 * m_abs_deriv * friction_term +
                            np.divide(der_lambda * branch_pit[:, IdxBranch.LENGTH] * m_init2, branch_pit[:, IdxBranch.D]))
    df_dm[np.isclose(m_init_abs, 0)] = 1.

    load_vec = p_diff + branch_pit[:, IdxBranch.PL] + const_height \
               - const_term * comp_fact * m_init2 * friction_term * p_sum_div * tm

    df_dm_nodes = np.ones_like(lambda_)

    load_vec_nodes_from = branch_pit[:, IdxBranch.MDOTINIT]
    load_vec_nodes_to = branch_pit[:, IdxBranch.MDOTINIT]
    dp_frict_loss = const_term * comp_fact * m_init2 * friction_term * p_sum_div * tm

    return load_vec, load_vec_nodes_from, load_vec_nodes_to, df_dm, df_dm_nodes, df_dp, df_dp1, dp_frict_loss

def derivatives_branch_thermal_np(branch_pit,
                                   branch_pit_old, branch_pit_old_lookup,
                                   t_init_i, t_init_i1, t_init_nt,
                                   cp_n, cp_b,
                                   rho, dt, transient, amb):
    """Branch-level thermal derivatives: fnt, dfnt_dt, dfnt_dtout, fb, dfb_dt, dfb_dtout."""
    mdot = np.abs(branch_pit[:, IdxBranch.MDOTINIT])
    t_amb = branch_pit[:, IdxBranch.TEXT]
    length = branch_pit[:, IdxBranch.LENGTH]
    alpha = branch_pit[:, IdxBranch.ALPHA] * np.pi * branch_pit[:, IdxBranch.DO]
    tl = branch_pit[:, IdxBranch.TL]
    qext = branch_pit[:, IdxBranch.QEXT]

    branches_flow = branches_not_zero_flow(branch_pit)

    fnt = cp_n * mdot * (t_init_i1 - t_init_nt)
    dfnt_dt = - cp_n * mdot
    dfnt_dtout = cp_n * mdot

    if transient:
        area = branch_area(branch_pit)
        tvor = branch_pit_old[:, branch_pit_old_lookup[IdxBranch.TOUTINIT]]

        fb = (
                rho * area * cp_b * (t_init_i1 - tvor) * (1 / dt) * length
                + cp_b * mdot * (-t_init_i + t_init_i1 - tl)
                - alpha * (t_amb - t_init_i1) * length + qext
        )

        dfb_dt = - cp_b * mdot
        dfb_dtout = rho * area * cp_b / dt * length + cp_b * mdot + alpha * length

        if np.any(~branches_flow):
            # TODO: maybe replace this statement with a component lookup
            zero_length = np.isclose(branch_pit[:, IdxBranch.LENGTH], 0, atol=1e-10)
            mask = zero_length & ~branches_flow
            if np.any(mask):
                fb[mask] = (
                        rho[mask] * area[mask] * cp_b[mask] * (t_init_i1[mask] - tvor[mask]) * (1 / dt)
                        - alpha[mask] * (t_amb[mask] - t_init_i1[mask]) + qext[mask]
                )
                dfb_dt[mask] = 0
                dfb_dtout[mask] = (rho[mask] * area[mask] * cp_b[mask] / dt +
                                   alpha[mask])
    else:
        non_zero_length_mask = ~np.isclose(branch_pit[:, IdxBranch.LENGTH], 0, rtol=1e-6, atol=1e-10)
        if np.any(non_zero_length_mask & (np.abs(branch_pit[:, IdxBranch.QEXT]) > 1e-12)):
            logger.warning(
                "A branch with non zero length has a non zero external heat load. This might lead "
                "to errors in the calculation, as the overlap of temperature reduction from heat "
                "losses to ambient and a constant heat flux cannot be solved with the implmented "
                "method."
            )

        fb = np.zeros_like(cp_b)
        fb[branches_flow] = (
                t_amb[branches_flow] + (t_init_i[branches_flow] - t_amb[branches_flow])
                * np.exp(- alpha[branches_flow] * length[branches_flow] / (cp_b[branches_flow] * mdot[branches_flow]))
                - t_init_i1[branches_flow] + tl[branches_flow]
                - qext[branches_flow] / (cp_b[branches_flow] * mdot[branches_flow])
        )
        fb[~branches_flow] = amb - t_init_i1[~branches_flow]
        dfb_dt = np.zeros_like(cp_b)
        dfb_dt[branches_flow] = np.exp(- alpha[branches_flow] * length[branches_flow] /
                                       (cp_b[branches_flow] * mdot[branches_flow]))
        dfb_dtout = - np.ones_like(cp_b)

        fnt[~branches_flow] = 0
        dfnt_dt[~branches_flow] = 0
        dfnt_dtout[~branches_flow] = 0

    return fnt, dfnt_dt, dfnt_dtout, fb, dfb_dt, dfb_dtout


def derivatives_node_thermal_np(node_pit, branch_pit,
                                 node_pit_old, node_pit_old_lookup,
                                 from_nodes, to_nodes,
                                 t_init_i, t_init_n,
                                 cp_b, rho, dt, transient, amb):
    """Node stagnant thermal derivatives: fn, dfn_dt. Must be called with the full branch pit."""
    branches_flow = branches_not_zero_flow(branch_pit)
    nodes_flow = np.isin(np.arange(len(node_pit)),
                         np.concatenate([from_nodes[branches_flow], to_nodes[branches_flow]]))

    fn = np.zeros_like(t_init_n)
    dfn_dt = np.zeros_like(t_init_n)

    if transient:
        area = branch_area(branch_pit)
        alpha = branch_pit[:, IdxBranch.ALPHA] * np.pi * branch_pit[:, IdxBranch.DO]
        t_amb = branch_pit[:, IdxBranch.TEXT]

        if np.any(~nodes_flow):
            fn_zero = ~nodes_flow[from_nodes]
            tn_zero = ~nodes_flow[to_nodes]

            t_from_node_vor_zero = node_pit_old[from_nodes[fn_zero], node_pit_old_lookup[IdxNode.TINIT]]
            t_to_node_vor_zero = node_pit_old[to_nodes[tn_zero], node_pit_old_lookup[IdxNode.TINIT]]
            t_to_node = node_pit[to_nodes[tn_zero], IdxNode.TINIT]

            fn_eq = (rho[fn_zero] * area[fn_zero] * cp_b[fn_zero] * (1 / dt)
                     * (t_init_i[fn_zero] - t_from_node_vor_zero)
                     - alpha[fn_zero] * (t_amb[fn_zero] - t_init_i[fn_zero]))

            tn_eq = (rho[tn_zero] * area[tn_zero] * cp_b[tn_zero] * (1 / dt)
                     * (t_to_node - t_to_node_vor_zero)
                     - alpha[tn_zero] * (t_amb[tn_zero] - t_to_node))

            fn_deriv = (rho[fn_zero] * area[fn_zero] * cp_b[fn_zero] * (1 / dt) + alpha[fn_zero])
            tn_deriv = (rho[tn_zero] * area[tn_zero] * cp_b[tn_zero] * (1 / dt) + alpha[tn_zero])

            fn_nodes, fn_eq_sum, fn_deriv_sum = _sum_by_group(False, from_nodes[fn_zero], fn_eq, fn_deriv)
            tn_nodes, tn_eq_sum, tn_deriv_sum = _sum_by_group(False, to_nodes[tn_zero], tn_eq, tn_deriv)

            fn[fn_nodes] += fn_eq_sum
            fn[tn_nodes] += tn_eq_sum
            dfn_dt[fn_nodes] += fn_deriv_sum
            dfn_dt[tn_nodes] += tn_deriv_sum
    else:
        fn[~nodes_flow] = amb - t_init_n[~nodes_flow]
        dfn_dt[~nodes_flow] = - np.ones(np.sum(~nodes_flow))

    return fn, dfn_dt


def calc_lambda_nikuradse_incomp_np(m, d, k, eta, area):
    m_abs = np.abs(m)
    re = m_abs * d / (eta * area)
    lambda_laminar = np.zeros_like(m)
    lambda_laminar[~np.isclose(re, 0)] = 64 / re[~np.isclose(re, 0)]
    lambda_nikuradse = 1 / ((-2 * np.log10(k / (3.71 * d))) ** 2)
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


def colebrook_np(re, d, k, lambda_nikuradse, max_iter, lengths, tolerance=1e-4):
    """Function calculates the friction factor of a pipe using the Colebrook-White equation.

    It is an implicit equation which is solved using the Newton-Raphson method. For pipes with
    zero flow or zero length, the initial guess is returned. This should be uncritical, as the
    pressure loss term will equal zero (lambda * u^2 * l / d).

    :param re: Reynolds number [dimensionless]
    :type re: np.array
    :param d: Diameter [m]
    :type d: np.array
    :param k: Roughness [m]
    :type k: np.array
    :param lambda_nikuradse: Initial guess for lambda (from Nikuradse)
    :type lambda_nikuradse: np.array
    :param max_iter: Maximum number of iterations for the Colebrook-White calculation
    :type max_iter: int
    :param lengths: Length of the pipes [m] - only used to identify zero-length pipes
    :type lengths: np.array
    :param tolerance: Tolerance for the Colebrook-White calculation
    :type tolerance: float
    :return: lambda_cb, converged
    1. lambda_cb: Friction factor according to Colebrook-White
    2. converged: True, if the Colebrook-White calculation converged for all pipes
    :rtype: (np.array, bool)
    """
    def colebrook_white_implicit(lambda_cb, re_nz, k_nz, d_nz):
        return lambda_cb ** (-1 / 2) + 2 * np.log10(2.51 / (re_nz * np.sqrt(lambda_cb)) + k_nz / (3.71 * d_nz))

    def cw_derivative(lambda_cb, re_nz, k_nz, d_nz):
        return -1 / 2 * lambda_cb ** (-3 / 2) - (2.51 / re_nz) * lambda_cb ** (-3 / 2) / (
                    np.log(10) * (2.51 / (re_nz * np.sqrt(lambda_cb)) + k_nz / (3.71 * d_nz)))

    mask = ~np.isclose(re, 0) & ~np.isclose(lengths, 0, rtol=1e-10, atol=1e-11)
    lambda_res = lambda_nikuradse

    if not mask.any():
        return True, lambda_res

    res = newton(colebrook_white_implicit, lambda_res[mask], maxiter=max_iter, args=(re[mask], k[mask], d[mask]),
                 tol=tolerance, full_output=True, fprime=cw_derivative)  # , fprime2=cw_derivative_2)

    if lambda_res[mask].size == 1:
        lambda_res[mask] = res[0]
        converged = res[1].converged
    else:
        lambda_res[mask] = res.root
        converged = np.all(res.converged)

    return converged, lambda_res


def calc_derived_values_np(node_pit, from_nodes, to_nodes):
    tinit_branch = (node_pit[from_nodes, IdxNode.TINIT] + node_pit[to_nodes, IdxNode.TINIT]) / 2
    height_difference = node_pit[from_nodes, IdxNode.HEIGHT] - node_pit[to_nodes, IdxNode.HEIGHT]
    p_init_i_abs = node_pit[from_nodes, IdxNode.PINIT] + node_pit[from_nodes, IdxNode.PAMB]
    p_init_i1_abs = node_pit[to_nodes, IdxNode.PINIT] + node_pit[to_nodes, IdxNode.PAMB]
    return tinit_branch, height_difference, p_init_i_abs, p_init_i1_abs