import numpy as np

from pandapipes.constants import P_CONVERSION, GRAVITATION_CONSTANT, NORMAL_PRESSURE, \
    NORMAL_TEMPERATURE
from pandapipes.idx_branch import IdxBranch
from pandapipes.idx_node import IdxNode

try:
    from numba import jit
    from numba import int32, float64, int64, bool, optional, none
except ImportError:
    from pandapower.pf.no_numba import jit
    from numpy import int32, float64, int64, bool
    from typing import Optional as optional

# numba's nopython mode can resolve a plain module-level int global, or a plain int accessed via
# module.CONST, as a compile-time constant - but IdxBranch/IdxNode are classes (see
# pandapipes.idx.IndexMeta), and numba has no typing rule for an arbitrary class object used as a
# global, so `IdxBranch.MDOTINIT` inside a @jit(nopython=True) function body fails to compile
# ("Untyped global name 'IdxBranch'..."). Bind the specific columns this module's jitted functions
# need as plain module-level ints once here, and reference those bare names inside every
# @jit(nopython=True) function below instead of the class attribute.
BRANCH_MDOTINIT = IdxBranch.MDOTINIT
BRANCH_LENGTH = IdxBranch.LENGTH
BRANCH_D = IdxBranch.D
BRANCH_DO = IdxBranch.DO
BRANCH_LOSS_COEFFICIENT = IdxBranch.LOSS_COEFFICIENT
BRANCH_LAMBDA = IdxBranch.LAMBDA
BRANCH_PL = IdxBranch.PL
BRANCH_TL = IdxBranch.TL
BRANCH_QEXT = IdxBranch.QEXT
BRANCH_TEXT = IdxBranch.TEXT
BRANCH_ALPHA = IdxBranch.ALPHA
BRANCH_FROM_NODE = IdxBranch.FROM_NODE
BRANCH_TOUTINIT = IdxBranch.TOUTINIT

NODE_TINIT = IdxNode.TINIT
NODE_HEIGHT = IdxNode.HEIGHT
NODE_PINIT = IdxNode.PINIT
NODE_PAMB = IdxNode.PAMB


@jit((float64[:, :], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:]), nopython=True, cache=False)
def derivatives_hydraulic_incomp_numba(branch_pit, der_lambda, p_init_i_abs, p_init_i1_abs,
                                       height_difference, rho, area):
    le = der_lambda.shape[0]
    load_vec = np.zeros_like(der_lambda)
    df_dm = np.zeros_like(der_lambda)
    df_dp = np.ones_like(der_lambda)
    df_dp1 = np.ones_like(der_lambda) * (-1)
    load_vec_nodes_from = np.zeros_like(der_lambda)
    load_vec_nodes_to = np.zeros_like(der_lambda)
    df_dm_nodes = np.ones_like(der_lambda)
    dp_frict_loss = np.zeros_like(der_lambda)

    for i in range(le):
        m_init_abs = np.abs(branch_pit[i][BRANCH_MDOTINIT])
        m_abs_deriv = max(m_init_abs, 1e-8)
        m_init2 = m_init_abs * branch_pit[i][BRANCH_MDOTINIT]
        p_diff = p_init_i_abs[i] - p_init_i1_abs[i]
        const_height = rho[i] * GRAVITATION_CONSTANT * height_difference[i] / P_CONVERSION
        friction_term = np.divide(branch_pit[i][BRANCH_LENGTH] * branch_pit[i][BRANCH_LAMBDA], branch_pit[i][BRANCH_D]) \
            + branch_pit[i][BRANCH_LOSS_COEFFICIENT]
        const_term = np.divide(1, area[i] ** 2 * rho[i] * P_CONVERSION * 2)

        df_dm[i] = -1. * const_term * (2 * m_abs_deriv * friction_term + der_lambda[i]
                                   * np.divide(branch_pit[i][BRANCH_LENGTH], branch_pit[i][BRANCH_D]) * m_init2)

        load_vec[i] = p_diff + branch_pit[i][BRANCH_PL] + const_height - const_term * m_init2 * friction_term

        load_vec_nodes_from[i] = branch_pit[i][BRANCH_MDOTINIT]
        load_vec_nodes_to[i] = branch_pit[i][BRANCH_MDOTINIT]
        dp_frict_loss[i] = const_term * m_init2 * friction_term
    return load_vec, load_vec_nodes_from, load_vec_nodes_to, df_dm, df_dm_nodes, df_dp, df_dp1, dp_frict_loss


@jit((float64[:, :], float64[:, :], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],
      float64[:], float64[:], float64[:], float64[:], float64[:]), nopython=True, cache=False)
def derivatives_hydraulic_comp_numba(node_pit, branch_pit, lambda_, der_lambda, p_init_i_abs, p_init_i1_abs,
                                     height_difference, comp_fact, der_comp, der_comp1, rho, rho_n, area):
    le = lambda_.shape[0]
    load_vec = np.zeros_like(lambda_)
    df_dm = np.zeros_like(lambda_)
    df_dp = np.zeros_like(lambda_)
    df_dp1 = np.zeros_like(lambda_)
    load_vec_nodes_from = np.zeros_like(der_lambda)
    load_vec_nodes_to = np.zeros_like(der_lambda)
    df_dm_nodes = np.ones_like(der_lambda)
    from_nodes = branch_pit[:, BRANCH_FROM_NODE].astype(np.int32)
    dp_frict_loss = np.zeros_like(der_lambda)

    # Formulas for gas pressure loss according to laminar version
    for i in range(le):
        # compressibility settings
        m_init_abs = np.abs(branch_pit[i][BRANCH_MDOTINIT])
        m_abs_deriv = max(m_init_abs, 1e-8)
        m_init2 = branch_pit[i][BRANCH_MDOTINIT] * m_init_abs
        p_diff = p_init_i_abs[i] - p_init_i1_abs[i]
        p_sum = p_init_i_abs[i] + p_init_i1_abs[i]
        p_sum_div = np.divide(1, p_sum)
        fn = from_nodes[i]
        tm = (node_pit[fn, NODE_TINIT] + branch_pit[i][BRANCH_TOUTINIT]) / 2

        const_height =  rho[i] * GRAVITATION_CONSTANT * height_difference[i] / P_CONVERSION
        friction_term = np.divide(lambda_[i] * branch_pit[i][BRANCH_LENGTH], branch_pit[i][BRANCH_D]) + \
                        branch_pit[i][BRANCH_LOSS_COEFFICIENT]
        normal_term = np.divide(NORMAL_PRESSURE, NORMAL_TEMPERATURE * P_CONVERSION * rho_n[i] * area[i] ** 2)

        load_vec[i] = p_diff + branch_pit[i][BRANCH_PL] + const_height \
            - normal_term * comp_fact[i] * m_init2 * friction_term * p_sum_div * tm

        const_term = normal_term * m_init2 * friction_term * tm
        df_dp[i] = 1. - const_term * p_sum_div * (der_comp[i] - comp_fact[i] * p_sum_div)
        df_dp1[i] = -1. - const_term * p_sum_div * (der_comp1[i] - comp_fact[i] * p_sum_div)

        df_dm[i] = -1. * normal_term * comp_fact[i] * p_sum_div * tm * (2 * m_abs_deriv * friction_term
            + np.divide(der_lambda[i] * branch_pit[i][BRANCH_LENGTH] * m_init2, branch_pit[i][BRANCH_D]))
        # matches derivatives_hydraulic_comp_np's df_dm[np.isclose(m_init_abs, 0)] = 1. - at
        # exactly zero flow, m_init2 is 0 so only the friction_term part of df_dm survives, and
        # for a zero-length/zero-loss-coefficient branch (e.g. a parallel dummy branch)
        # friction_term is 0 too, leaving df_dm exactly 0: a singular Jacobian diagonal entry for
        # that branch's momentum row.
        if m_init_abs <= 1e-8:
            df_dm[i] = 1.

        load_vec_nodes_from[i] = branch_pit[i][BRANCH_MDOTINIT]
        load_vec_nodes_to[i] = branch_pit[i][BRANCH_MDOTINIT]
        dp_frict_loss[i] = normal_term * comp_fact[i] * m_init2 * friction_term * p_sum_div * tm
    return load_vec, load_vec_nodes_from, load_vec_nodes_to, df_dm, df_dm_nodes, df_dp, df_dp1, dp_frict_loss


@jit((float64[:, :], int32[:], int32[:]), nopython=True, cache=False)
def _make_lookups(branch_pit, to_nodes, from_nodes):
    max_val_to = np.max(to_nodes)
    max_val_from = np.max(from_nodes)
    club_to = np.zeros(max_val_to + 1, dtype=bool)
    club_from = np.zeros(max_val_from + 1, dtype=bool)
    branches_flow = np.zeros_like(to_nodes, dtype=bool)
    for i in range(len(to_nodes)):
        mdot = branch_pit[i, BRANCH_MDOTINIT]
        branches_flow[i] = (not np.isnan(mdot)) and (abs(mdot) > 1e-10)
        if branches_flow[i]:
            club_to[to_nodes[i]] = True
            club_from[from_nodes[i]] = True

    return club_to, club_from, branches_flow


@jit((float64[:, :], float64[:, :],
      float64[:, :], int32[:],
      float64[:, :], int32[:],
      int32[:], int32[:],
      float64[:], float64[:], float64[:], float64[:],
      float64[:], float64[:],
      float64[:], optional(float64), bool, float64), nopython=True, cache=False)
def derivatives_thermal_numba(node_pit, branch_pit,
                              node_pit_old, node_pit_old_lookup,
                              branch_pit_old, branch_pit_old_lookup,
                              from_nodes, to_nodes,
                              t_init_i, t_init_i1, t_init_nt, t_init_n,
                              cp_n, cp_b,
                              rho, dt, transient, amb):
    n = t_init_n.shape[0]
    b = t_init_nt.shape[0]

    nodes_flow = np.zeros_like(t_init_n, dtype=bool)

    fn = np.zeros_like(t_init_n)
    dfn_dt = np.zeros_like(t_init_n)

    fnt = np.zeros_like(t_init_nt)
    dfnt_dt = np.zeros_like(t_init_nt)
    dfnt_dtout = np.zeros_like(t_init_nt)

    club_to, club_from, branches_flow = _make_lookups(branch_pit, to_nodes, from_nodes)

    fb = np.zeros_like(t_init_nt)
    dfb_dt = np.zeros_like(t_init_nt)
    dfb_dtout = np.zeros_like(t_init_nt)

    infeed = np.zeros_like(t_init_n, dtype=bool)

    for i in range(n):
        result_from = club_from[i] if (i < len(club_from)) else False
        result_to = club_to[i] if (i < len(club_to)) else False
        nodes_flow[i] = result_from | result_to

        if ~transient and ~nodes_flow[i]:
            fn[i] = amb - t_init_n[i]
            dfn_dt[i] = -1.

    for i in range(b):
        # this is not required currently, but useful when implementing leakages
        # m_init_i = np.abs(branch_pit[:, BRANCH_MDOTINIT])
        # m_init_i1 = np.abs(branch_pit[:, BRANCH_MDOTINIT])
        mdot = np.abs(branch_pit[i][BRANCH_MDOTINIT])
        t_amb = branch_pit[i][BRANCH_TEXT]
        length = branch_pit[i][BRANCH_LENGTH]
        alpha = branch_pit[i][BRANCH_ALPHA] * np.pi * branch_pit[i][BRANCH_DO]
        tl = branch_pit[i][BRANCH_TL]
        qext = branch_pit[i][BRANCH_QEXT]

        fnt[i] = cp_n[i] * mdot * (t_init_i1[i] - t_init_nt[i])
        dfnt_dt[i] = - cp_n[i] * mdot
        dfnt_dtout[i] = cp_n[i] * mdot

        if transient:
            area = np.pi * (branch_pit[i][BRANCH_D] / 2) ** 2
            tvor = branch_pit_old[i][branch_pit_old_lookup[BRANCH_TOUTINIT]]

            fb[i] = (
                    rho[i] * area * cp_b[i] * (t_init_i1[i] - tvor) * (1 / dt) * length
                    + cp_b[i] * mdot * (-t_init_i[i] + t_init_i1[i] - tl)
                    - alpha * (t_amb - t_init_i1[i]) * length + qext
            )

            dfb_dt[i] = - cp_b[i] * mdot
            dfb_dtout[i] = rho[i] * area * cp_b[i] / dt * length + cp_b[i] * mdot + alpha * length

            if ~branches_flow[i] & (abs(branch_pit[i][BRANCH_LENGTH] < 1.e-8)):
                fb[i] = rho[i] * area * cp_b[i] * (t_init_i1[i] - tvor) * (1 / dt) - alpha * (t_amb - t_init_i1[i]) + qext
                dfb_dt[i] = 0
                dfb_dtout[i] = rho[i] * area * cp_b[i] / dt + alpha

            fn_zero = ~nodes_flow[from_nodes[i]]
            tn_zero = ~nodes_flow[to_nodes[i]]
            if fn_zero:
                t_from_node_vor_zero = node_pit_old[from_nodes[i], node_pit_old_lookup[NODE_TINIT]]
                fn_eq = (rho[i] * area * cp_b[i] * (1 / dt) * (t_init_i[i] - t_from_node_vor_zero)
                         - alpha * (t_amb - t_init_i[i]))
                fn_deriv = rho[i] * area * cp_b[i] * (1 / dt) + alpha
                dfn_dt[from_nodes[i]] += fn_deriv
                fn[from_nodes[i]] += fn_eq
            if tn_zero:
                t_to_node_vor_zero = node_pit_old[to_nodes[i], node_pit_old_lookup[NODE_TINIT]]
                t_to_node = node_pit[to_nodes[i], NODE_TINIT]
                tn_eq = (rho[i] * area * cp_b[i] * (1 / dt) * (t_to_node - t_to_node_vor_zero)
                         - alpha * (t_amb - t_to_node))
                tn_deriv = (rho[i]* area * cp_b[i] * (1 / dt) + alpha)
                dfn_dt[to_nodes[i]] += tn_deriv
                fn[to_nodes[i]] += tn_eq
        else:
            if branches_flow[i]:
                fb[i] = (
                        t_amb + (t_init_i[i] - t_amb) * np.exp(- alpha * length / (cp_b[i] * mdot))
                        - t_init_i1[i] + tl - qext / (cp_b[i] * mdot)
                )
                dfb_dt[i] = np.exp(- alpha * length / (cp_b[i] * mdot))
            else:
                fb[i] = amb - t_init_i1[i]
            dfb_dtout[i] = -1
        if branches_flow[i]:
            result_from = club_to[from_nodes[i]] if (from_nodes[i] < len(club_to)) else False
            infeed[from_nodes[i]] = ~result_from

    return fn, dfn_dt, fnt, dfnt_dt, dfnt_dtout, fb, dfb_dt, dfb_dtout, infeed


@jit((float64[:, :],
      float64[:, :], int32[:],
      float64[:], float64[:], float64[:],
      float64[:], float64[:],
      float64[:], optional(float64), bool, float64), nopython=True, cache=False)
def derivatives_branch_thermal_numba(branch_pit,
                                      branch_pit_old, branch_pit_old_lookup,
                                      t_init_i, t_init_i1, t_init_nt,
                                      cp_n, cp_b,
                                      rho, dt, transient, amb):
    b = t_init_nt.shape[0]

    fnt = np.zeros_like(t_init_nt)
    dfnt_dt = np.zeros_like(t_init_nt)
    dfnt_dtout = np.zeros_like(t_init_nt)
    fb = np.zeros_like(t_init_nt)
    dfb_dt = np.zeros_like(t_init_nt)
    dfb_dtout = np.zeros_like(t_init_nt)

    branches_flow = np.zeros(b, dtype=bool)
    for i in range(b):
        mdot_val = branch_pit[i, BRANCH_MDOTINIT]
        branches_flow[i] = (not np.isnan(mdot_val)) and (abs(mdot_val) > 1e-10)

    for i in range(b):
        mdot = np.abs(branch_pit[i][BRANCH_MDOTINIT])
        t_amb = branch_pit[i][BRANCH_TEXT]
        length = branch_pit[i][BRANCH_LENGTH]
        alpha = branch_pit[i][BRANCH_ALPHA] * np.pi * branch_pit[i][BRANCH_DO]
        tl = branch_pit[i][BRANCH_TL]
        qext = branch_pit[i][BRANCH_QEXT]

        fnt[i] = cp_n[i] * mdot * (t_init_i1[i] - t_init_nt[i])
        dfnt_dt[i] = -cp_n[i] * mdot
        dfnt_dtout[i] = cp_n[i] * mdot

        if transient:
            area = np.pi * (branch_pit[i][BRANCH_D] / 2) ** 2
            tvor = branch_pit_old[i][branch_pit_old_lookup[BRANCH_TOUTINIT]]

            fb[i] = (
                rho[i] * area * cp_b[i] * (t_init_i1[i] - tvor) * (1 / dt) * length
                + cp_b[i] * mdot * (-t_init_i[i] + t_init_i1[i] - tl)
                - alpha * (t_amb - t_init_i1[i]) * length + qext
            )
            dfb_dt[i] = -cp_b[i] * mdot
            dfb_dtout[i] = rho[i] * area * cp_b[i] / dt * length + cp_b[i] * mdot + alpha * length

            if not branches_flow[i] and abs(branch_pit[i][BRANCH_LENGTH]) < 1e-8:
                fb[i] = (rho[i] * area * cp_b[i] * (t_init_i1[i] - tvor) * (1 / dt)
                         - alpha * (t_amb - t_init_i1[i]) + qext)
                dfb_dt[i] = 0
                dfb_dtout[i] = rho[i] * area * cp_b[i] / dt + alpha
        else:
            if branches_flow[i]:
                fb[i] = (
                    t_amb + (t_init_i[i] - t_amb) * np.exp(-alpha * length / (cp_b[i] * mdot))
                    - t_init_i1[i] + tl - qext / (cp_b[i] * mdot)
                )
                dfb_dt[i] = np.exp(-alpha * length / (cp_b[i] * mdot))
            else:
                fb[i] = amb - t_init_i1[i]
                fnt[i] = 0.
                dfnt_dt[i] = 0.
                dfnt_dtout[i] = 0.
            dfb_dtout[i] = -1.

    return fnt, dfnt_dt, dfnt_dtout, fb, dfb_dt, dfb_dtout


@jit((float64[:, :], float64[:, :],
      float64[:, :], int32[:],
      int32[:], int32[:],
      float64[:], float64[:],
      float64[:], float64[:],
      optional(float64), bool, float64), nopython=True, cache=False)
def derivatives_node_thermal_numba(node_pit, branch_pit,
                                    node_pit_old, node_pit_old_lookup,
                                    from_nodes, to_nodes,
                                    t_init_i, t_init_n,
                                    cp_b, rho,
                                    dt, transient, amb):
    n = t_init_n.shape[0]
    b = from_nodes.shape[0]

    fn = np.zeros_like(t_init_n)
    dfn_dt = np.zeros_like(t_init_n)

    if b == 0:
        if not transient:
            for i in range(n):
                fn[i] = amb - t_init_n[i]
                dfn_dt[i] = -1.
        return fn, dfn_dt

    club_to, club_from, _ = _make_lookups(branch_pit, to_nodes, from_nodes)
    nodes_flow = np.zeros(n, dtype=bool)
    for i in range(n):
        result_from = club_from[i] if (i < len(club_from)) else False
        result_to = club_to[i] if (i < len(club_to)) else False
        nodes_flow[i] = result_from | result_to

        if not transient and not nodes_flow[i]:
            fn[i] = amb - t_init_n[i]
            dfn_dt[i] = -1.

    if transient:
        for i in range(b):
            area = np.pi * (branch_pit[i][BRANCH_D] / 2) ** 2
            t_amb = branch_pit[i][BRANCH_TEXT]
            alpha = branch_pit[i][BRANCH_ALPHA] * np.pi * branch_pit[i][BRANCH_DO]

            fn_zero = not nodes_flow[from_nodes[i]]
            tn_zero = not nodes_flow[to_nodes[i]]

            if fn_zero:
                t_from_vor = node_pit_old[from_nodes[i], node_pit_old_lookup[NODE_TINIT]]
                fn_eq = (rho[i] * area * cp_b[i] * (1 / dt) * (t_init_i[i] - t_from_vor)
                         - alpha * (t_amb - t_init_i[i]))
                fn_deriv = rho[i] * area * cp_b[i] * (1 / dt) + alpha
                dfn_dt[from_nodes[i]] += fn_deriv
                fn[from_nodes[i]] += fn_eq
            if tn_zero:
                t_to_vor = node_pit_old[to_nodes[i], node_pit_old_lookup[NODE_TINIT]]
                t_to = node_pit[to_nodes[i], NODE_TINIT]
                tn_eq = (rho[i] * area * cp_b[i] * (1 / dt) * (t_to - t_to_vor)
                         - alpha * (t_amb - t_to))
                tn_deriv = rho[i] * area * cp_b[i] * (1 / dt) + alpha
                dfn_dt[to_nodes[i]] += tn_deriv
                fn[to_nodes[i]] += tn_eq

    return fn, dfn_dt


@jit((float64[:], float64[:], float64[:], float64[:], float64[:]), nopython=True)
def calc_lambda_nikuradse_incomp_numba(m, d, k, eta, area):
    lambda_nikuradse = np.zeros_like(m)
    lambda_laminar = np.zeros_like(m)
    re = np.zeros_like(m)
    m_abs = np.abs(m)
    for i in range(m.shape[0]):
        re[i] = np.divide(m_abs[i] * d[i], eta[i] * area[i])
        if (abs(re[i]) > 1.e-8):
            lambda_laminar[i] = 64 / re[i]
        lambda_nikuradse[i] = np.power(-2 * np.log10(k[i] / (3.71 * d[i])), -2)
    return re, lambda_laminar, lambda_nikuradse


@jit((float64[:], float64[:], float64[:], float64[:], float64[:]), nopython=True)
def calc_lambda_nikuradse_comp_numba(m, d, k, eta, area):
    lambda_nikuradse = np.zeros_like(m)
    lambda_laminar = np.zeros_like(m)
    re = np.zeros_like(m)
    for i, mi in enumerate(m):
        m_abs = np.abs(mi)
        re[i] = np.divide(m_abs * d[i], eta[i] * area[i])
        if (abs(re[i]) > 1.e-8):
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


@jit((float64[:], float64[:], float64[:], float64[:], int64, float64[:], float64), nopython=True, cache=False)
def colebrook_numba(re, d, k, lambda_nikuradse, max_iter, lengths, tolerance):
    """Numba counterpart of derivative_toolbox.colebrook_np, for calc_lambda's use_numba=True dispatch.

    Imported there as ``colebrook_numba as colebrook``, mirroring ``colebrook_np as
    colebrook`` on the non-numba side - both share this same positional signature.
    scipy.optimize.newton (colebrook_np's own Newton solver) isn't numba-jittable, so this
    reimplements the same implicit equation
    ``lambda_cb**(-1/2) + 2*log10(2.51/(re*sqrt(lambda_cb)) + k/(3.71*d)) = 0`` as a hand-rolled,
    jit-compiled Newton loop instead. Mirrors colebrook_np's own masking - branches with ~zero Re
    or ~zero length are left at their input lambda_nikuradse guess rather than Newton-iterated
    (colebrook_np: ``~np.isclose(re, 0) & ~np.isclose(lengths, 0, rtol=1e-10, atol=1e-11)``,
    reproduced here via explicit thresholds since np.isclose itself isn't numba-jittable) - and
    takes the same configurable ``tolerance`` (net option "tolerance_colebrook").

    :return: (converged, lambda_cb) - converged is False if max_iter is reached before every
        active branch's Newton step drops below tolerance.
    """
    n = re.shape[0]
    lambda_cb = lambda_nikuradse.copy()
    active = np.zeros(n, dtype=bool)
    n_active = 0
    for i in range(n):
        if abs(re[i]) > 1e-8 and abs(lengths[i]) > 1e-11:
            active[i] = True
            n_active += 1

    if n_active == 0:
        return True, lambda_cb

    converged = False
    niter = 0

    # Inner Newton-loop for calculation of lambda
    while not converged and niter < max_iter:
        max_step = 0.0
        for i in range(n):
            if not active[i]:
                continue
            sqt = np.sqrt(lambda_cb[i])
            add_val = np.divide(k[i], (3.71 * d[i]))
            sqt_div = np.divide(1, sqt)
            re_div = np.divide(1, re[i])
            sqt_div3 = sqt_div ** 3

            f = sqt_div + 2 * np.log10(2.51 * re_div * sqt_div + add_val)
            df_dlambda_cb = - 0.5 * sqt_div3 - 2.51 * re_div * sqt_div3 * np.divide(
                1, np.log(10) * (2.51 * re_div * sqt_div + add_val))
            step = - f / df_dlambda_cb
            lambda_cb[i] += step

            abs_step = abs(step)
            max_step = max(max_step, abs_step)

        if max_step < tolerance:
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
        tinit_branch[i] = (node_pit[fn, NODE_TINIT] + node_pit[tn, NODE_TINIT]) / 2
        height_difference[i] = node_pit[fn, NODE_HEIGHT] - node_pit[tn, NODE_HEIGHT]
        p_init_i_abs[i] = node_pit[fn, NODE_PINIT] + node_pit[fn, NODE_PAMB]
        p_init_i1_abs[i] = node_pit[tn, NODE_PINIT] + node_pit[tn, NODE_PAMB]
    return tinit_branch, height_difference, p_init_i_abs, p_init_i1_abs
