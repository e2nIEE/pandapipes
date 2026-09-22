# pylint: disable=import-outside-toplevel
# Every local import below picks between a numba and a plain-numpy implementation based on a
# runtime option (options["use_numba"]). This can't be hoisted to module level: when numba isn't
# installed, derivative_toolbox_numba's @jit(...) decorators fall back to plain numpy types for
# their explicit signatures (e.g. float64[:, :]), and numpy's own float64 doesn't support that
# subscript syntax at all ("TypeError: There are no type variables left in numpy.float64" on
# numpy>=2) - so importing that module eagerly would break `import pandapipes` outright whenever
# numba isn't installed, not just waste time JIT-compiling functions nobody asked for. Verified by
# actually trying it: any use_numba dispatch import from derivative_toolbox_numba hoisted to this
# file's top level reproduces that exact crash immediately.
import numpy as np
from pandapipes.constants import NORMAL_TEMPERATURE
from pandapipes.idx_branch import IdxBranch
from pandapipes.idx_node import IdxNode
from pandapipes.pf.internals_toolbox import get_from_nodes_corrected, get_to_nodes_corrected, _sum_by_group, \
    branch_area
from pandapipes.pf.pipeflow_setup import get_net_option, get_lookup, PipeflowNotConverged
from pandapipes.properties.fluids import get_fluid
from pandapipes.properties.properties_toolbox import get_branch_real_density, get_branch_real_eta, get_branch_cp


def calculate_derivatives_hydraulic(net, branch_pit_slice, node_pit, options):
    """Compute hydraulic derivatives for *branch_pit_slice* (a view of the global branch pit) and write results back in-place via the view.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param branch_pit_slice: view of the global branch pit for the component's active branches
    :param node_pit: global node internal table
    :param options: solver options dict (use_numba, friction_model, …)
    :return: df_dm, df_dp, df_dp1, df_dm_nodes, load_vec, load_vec_nodes_from, load_vec_nodes_to
    """
    if options["use_numba"]:
        from pandapipes.pf.derivative_toolbox_numba import (
            derivatives_hydraulic_incomp_numba as derivatives_hydraulic_incomp,
            derivatives_hydraulic_comp_numba as derivatives_hydraulic_comp,
            calc_medium_pressure_with_derivative_numba as calc_medium_pressure_with_derivative)
    else:
        from pandapipes.pf.derivative_toolbox import (derivatives_hydraulic_incomp_np as derivatives_hydraulic_incomp,
                                                      derivatives_hydraulic_comp_np as derivatives_hydraulic_comp,
                                                      calc_medium_pressure_with_derivative_np as calc_medium_pressure_with_derivative)
    fluid = get_fluid(net)
    gas_mode = fluid.is_gas
    friction_model = options["friction_model"]

    b_pit = branch_pit_slice
    from_nodes = b_pit[:, IdxBranch.FROM_NODE].astype(np.int32)
    to_nodes = b_pit[:, IdxBranch.TO_NODE].astype(np.int32)
    tinit_branch, height_difference, p_init_i_abs, p_init_i1_abs = get_derived_values(
        node_pit, from_nodes, to_nodes, options["use_numba"])

    if gas_mode:
        p_m, der_p_m, der_p_m1 = calc_medium_pressure_with_derivative(p_init_i_abs, p_init_i1_abs)
    else:
        p_m, der_p_m, der_p_m1 = (p_init_i_abs + p_init_i1_abs) / 2, None, None

    rho = get_branch_real_density(fluid, node_pit, b_pit)
    eta = get_branch_real_eta(fluid, node_pit, b_pit, p_m)

    # computed once here, then threaded through to calc_lambda/calc_der_lambda AND
    # derivatives_hydraulic_incomp/comp below - not re-derived at each of those call sites for
    # the same branch set/iteration (D never changes mid-solve here, only across outer
    # optimize_dn iterations, so this is safe to reuse for the remainder of this call, but never
    # cached anywhere longer-lived than that - see branch_area's own docstring for why)
    area = branch_area(b_pit)
    lambda_, re = calc_lambda(b_pit[:, IdxBranch.MDOTINIT], eta, b_pit[:, IdxBranch.D], b_pit[:, IdxBranch.K], gas_mode,
        friction_model, b_pit[:, IdxBranch.LENGTH], options, area)
    der_lambda = calc_der_lambda(b_pit[:, IdxBranch.MDOTINIT], eta, b_pit[:, IdxBranch.D], b_pit[:, IdxBranch.K], friction_model,
                                 lambda_, area, re, b_pit[:, IdxBranch.LENGTH])
    b_pit[:, IdxBranch.RE]     = re
    b_pit[:, IdxBranch.LAMBDA] = lambda_

    if not gas_mode:
        load_vec, load_vec_nodes_from, load_vec_nodes_to, df_dm, df_dm_nodes, df_dp, df_dp1, dp_frict_loss = (
            derivatives_hydraulic_incomp(b_pit, der_lambda, p_init_i_abs, p_init_i1_abs, height_difference, rho, area))
    else:
        rho_n = np.full(len(b_pit), fluid.get_density(NORMAL_TEMPERATURE))
        comp_fact = fluid.get_compressibility(p_m, tinit_branch)
        dc = fluid.get_der_compressibility()
        der_comp = dc * der_p_m
        der_comp1 = dc * der_p_m1
        load_vec, load_vec_nodes_from, load_vec_nodes_to, df_dm, df_dm_nodes, df_dp, df_dp1, dp_frict_loss = (
            derivatives_hydraulic_comp(node_pit, b_pit, lambda_, der_lambda, p_init_i_abs, p_init_i1_abs,
                height_difference, comp_fact, der_comp, der_comp1, rho, rho_n, area))

    b_pit[:, IdxBranch.DP_FRICT_LOSS] = dp_frict_loss

    return df_dm, df_dp, df_dp1, df_dm_nodes, load_vec, load_vec_nodes_from, load_vec_nodes_to


def calculate_derivatives_branch_thermal(net, branch_pit_slice, node_pit, branch_pit_old_slice, options):
    """Compute branch-level thermal derivatives for *branch_pit_slice*.

    Stagnant node equations are excluded — see calculate_derivatives_node_thermal.

    :return: fnt, dfnt_dt, dfnt_dtout, fb, dfb_dt, dfb_dtout
    """
    branch_pit_old_lookup = get_lookup(net, "branch", "old_pit_cols")

    if options["use_numba"]:
        from pandapipes.pf.derivative_toolbox_numba import derivatives_branch_thermal_numba as deriv_fn
    else:
        from pandapipes.pf.derivative_toolbox import derivatives_branch_thermal_np as deriv_fn

    fluid = get_fluid(net)
    b_pit = branch_pit_slice
    b_pit_old = branch_pit_old_slice

    from_nodes = get_from_nodes_corrected(b_pit)
    to_nodes = get_to_nodes_corrected(b_pit)
    t_init_i = node_pit[from_nodes, IdxNode.TINIT]
    t_init_i1 = b_pit[:, IdxBranch.TOUTINIT]
    t_init_nt = node_pit[to_nodes, IdxNode.TINIT]
    cp_b = get_branch_cp(fluid, node_pit, b_pit)
    cp_i1 = fluid.get_heat_capacity(t_init_i1)
    cp_nt = fluid.get_heat_capacity(t_init_nt)
    cp_n = fluid.get_heat_capacity((cp_i1 + cp_nt) / 2)
    rho = get_branch_real_density(fluid, node_pit, b_pit)
    transient = get_net_option(net, "transient")
    dt = get_net_option(net, "dt")
    amb = get_net_option(net, 'ambient_temperature')

    return deriv_fn(
        b_pit,
        b_pit_old, branch_pit_old_lookup,
        t_init_i, t_init_i1, t_init_nt,
        cp_n, cp_b,
        rho, dt, transient, amb,
    )


def calculate_derivatives_node_thermal(net, branch_pit, node_pit, node_pit_old, options):
    """Compute stagnant node thermal derivatives from the FULL active thermal branch pit.

    Must be called with the complete branch pit so that nodes_flow is computed globally.

    :return: fn_node, dfn_dt  (arrays indexed by global node index)
    """
    node_pit_old_lookup = get_lookup(net, "node", "old_pit_cols")

    if options["use_numba"]:
        from pandapipes.pf.derivative_toolbox_numba import derivatives_node_thermal_numba as deriv_fn
    else:
        from pandapipes.pf.derivative_toolbox import derivatives_node_thermal_np as deriv_fn

    fluid = get_fluid(net)
    from_nodes = get_from_nodes_corrected(branch_pit)
    to_nodes = get_to_nodes_corrected(branch_pit)
    t_init_i = node_pit[from_nodes, IdxNode.TINIT]
    t_init_n = node_pit[:, IdxNode.TINIT]
    cp_b = get_branch_cp(fluid, node_pit, branch_pit)
    rho = get_branch_real_density(fluid, node_pit, branch_pit)
    transient = get_net_option(net, "transient")
    dt = get_net_option(net, "dt")
    amb = get_net_option(net, 'ambient_temperature')

    return deriv_fn(
        node_pit, branch_pit,
        node_pit_old, node_pit_old_lookup,
        from_nodes, to_nodes,
        t_init_i, t_init_n,
        cp_b, rho, dt, transient, amb,
    )


def calculate_load_hydraulic(net, junctions, mdot, scaling, in_service, sign, junction_table_name):
    """Compute the aggregated nodal mass-flow loads for a ConstFlow-type component.

    :param junctions: per-row junction table index (not a pit position) - grouped by
        :func:`~pandapipes.pf.internals_toolbox._sum_by_group` before being resolved against the
        active node pit
    :param mdot: per-row mass flow (may contain NaN, treated as 0)
    :param scaling: per-row scaling factor
    :param in_service: per-row in-service flag
    :return: the active node-pit indices and the corresponding summed load values
        (sign-corrected, NaN-safe, filtered to hydraulically active nodes)
    """
    helper = in_service * scaling * sign
    mf = np.nan_to_num(mdot)
    juncts, loads_sum = _sum_by_group(
        get_net_option(net, "use_numba"), junctions, -mf * helper)
    junction_idx_lookup = get_lookup(net, "node", "index_active_hydraulics")[junction_table_name]
    index = junction_idx_lookup[juncts.astype(np.int32)]
    valid = index >= 0
    return index[valid].astype(np.int32), loads_sum[valid]


def get_derived_values(node_pit, from_nodes, to_nodes, use_numba):
    if use_numba:
        from pandapipes.pf.derivative_toolbox_numba import calc_derived_values_numba
        return calc_derived_values_numba(node_pit, from_nodes, to_nodes)
    from pandapipes.pf.derivative_toolbox import calc_derived_values_np
    return calc_derived_values_np(node_pit, from_nodes, to_nodes)


def calc_lambda(m, eta, d, k, gas_mode, friction_model, lengths, options, area):
    """Function calculates the friction factor of a pipe.

    Turbulence is calculated based on Nikuradse. If v equals 0, a value of 0.001 is used in order
    to avoid division by zero. This should not be a problem as the pressure loss term will equal
    zero (lambda * u^2).

    :param m:
    :type m:
    :param eta:
    :type eta:
    :param d:
    :type d:
    :param k:
    :type k:
    :param gas_mode:
    :type gas_mode:
    :param friction_model:
    :type friction_model:
    :param lengths:
    :type lengths:
    :param options:
    :type options:
    :param area:
    :type area:
    :return:
    :rtype:
    """
    if options["use_numba"]:
        from pandapipes.pf.derivative_toolbox_numba import (
            calc_lambda_nikuradse_incomp_numba as calc_lambda_nikuradse_incomp,
            calc_lambda_nikuradse_comp_numba as calc_lambda_nikuradse_comp,
            colebrook_numba as colebrook)
    else:
        from pandapipes.pf.derivative_toolbox import (calc_lambda_nikuradse_incomp_np as calc_lambda_nikuradse_incomp,
                                                      calc_lambda_nikuradse_comp_np as calc_lambda_nikuradse_comp,
                                                      colebrook_np as colebrook)
    if gas_mode:
        re, lambda_laminar, lambda_nikuradse = calc_lambda_nikuradse_comp(m, d, k, eta, area)
    else:
        re, lambda_laminar, lambda_nikuradse = calc_lambda_nikuradse_incomp(m, d, k, eta, area)

    if friction_model == "colebrook":
        max_iter = options.get("max_iter_colebrook", 100)
        tolerance = options.get("tolerance_colebrook", 1e-4)
        converged, lambda_colebrook = colebrook(re, d, k, lambda_nikuradse, max_iter, lengths, tolerance)
        if not converged:
            raise PipeflowNotConverged("The Colebrook-White algorithm did not converge. There might be model "
                                       "inconsistencies. The maximum iterations can be given as 'max_iter_colebrook' "
                                       "argument to the pipeflow.")
        return lambda_colebrook, re
    elif friction_model == "swamee-jain":
        # 1.325 instead of 0.25???
        lambda_swamee_jain = 0.25 / ((np.log10(k / (3.7 * d) + 5.74 / (re ** 0.9))) ** 2)
        return lambda_swamee_jain, re
    else:
        # lambda_tot = np.where(re > 2300, lambda_laminar + lambda_nikuradse, lambda_laminar)
        lambda_tot = lambda_laminar + lambda_nikuradse
        return lambda_tot, re


def calc_der_lambda(m, eta, d, k, friction_model, lambda_pipe, area, re, lengths):
    """Function calculates the derivative of lambda with respect to v.

    Turbulence is calculated based on Nikuradse. This should not be a problem as the pressure loss
    term will equal zero (lambda * u^2).

    :param m:
    :type m:
    :param eta:
    :type eta:
    :param d:
    :type d:
    :param k:
    :type k:
    :param friction_model:
    :type friction_model:
    :param lambda_pipe:
    :type lambda_pipe:
    :param area:
    :type area:
    :return:
    :rtype:
    """
    b_term = np.zeros_like(m)
    df_dm = np.zeros_like(m)
    df_dlambda = np.zeros_like(m)
    lambda_der = np.zeros_like(m)
    pos = ~np.isclose(re, 0)

    if friction_model == "colebrook":
        pos &= ~np.isclose(lengths, 0, rtol=1e-10, atol=1e-11)
        m_abs = np.abs(m)
        # b_term is "2.51/(Re*sqrt(lambda)) + k/(3.71*d)" from the colebrook-white implicit
        # equation F(lambda, m) = 0 (see colebrook_white_implicit/cw_derivative in
        # derivative_toolbox.colebrook_np); Re is defined
        # via |m| (see calc_lambda_nikuradse_*_np's own re = m_abs*d/(eta*area)), so this needs
        # m_abs here too, not signed m - a previous version used signed m, which is only correct
        # for m > 0 and flips b_term's first term to the wrong sign for m < 0.
        b_term[pos] = (2.51 * eta[pos] * area[pos] / (m_abs[pos] * d[pos] * np.sqrt(lambda_pipe[pos])) + k[pos] / (
                    3.71 * d[pos]))

        # dF/dm - the 1/|m| term in b_term contributes a sign(m) chain-rule factor (d|m|/dm); a
        # previous version omitted it, same class of bug as the "nikuradse"/"swamee-jain" branches
        # below.
        df_dm[pos] = -np.sign(m[pos]) * 2 * 2.51 * eta[pos] * area[pos] / (
                    m[pos] ** 2 * np.sqrt(lambda_pipe[pos]) * d[pos]) / (np.log(10) * b_term[pos])

        df_dlambda[pos] = -0.5 * lambda_pipe[pos] ** (-3 / 2) - (2.51 * eta[pos] * area[pos] / (d[pos] * m_abs[pos])) * \
                          lambda_pipe[pos] ** (-3 / 2) / (np.log(10) * b_term[pos])

        # implicit function theorem: F(lambda(m), m) = 0 => dlambda/dm = -(dF/dm)/(dF/dlambda) -
        # a previous version omitted the leading minus sign, verified against finite differences
        # of calc_lambda(..., friction_model="colebrook") for both signs of m (see git history).
        lambda_der[pos] = -df_dm[pos] / df_dlambda[pos]

        return lambda_der
    elif friction_model == "swamee-jain":
        param = (k[pos] / (3.7 * d[pos]) + 5.74 * ((eta[pos] * area[pos]) / (np.abs(m[pos]) * d[pos])) ** 0.9)
        # 0.5 / (log(10) * log(param)^3 * param) * 5.166 * abs(eta)^0.9  / (abs(rho * d)^0.9
        # * abs(v_corr)^1.9)
        # lambda_swamee_jain is a function of |m| only, so d(lambda)/dm picks up a d|m|/dm =
        # sign(m) chain-rule factor - verified against finite differences of calc_lambda(...,
        # friction_model="swamee-jain") for both signs of m (see git history); a previous version
        # of this branch omitted it, returning the same value for m < 0 as for m > 0 instead of
        # flipping its sign.
        lambda_der[pos] = np.sign(m[pos]) * 0.5 * np.log(10) ** 2 / (np.log(param) ** 3) / param * 5.166 * (
                    (eta[pos] * area[pos]) / (d[pos])) ** 0.9 * np.abs(m[pos]) ** -1.9
        return lambda_der
    else:
        # lambda_laminar = 64/Re = 64*eta*area/(|m|*d), so d(lambda_laminar)/dm carries a
        # sign(m) factor from d(1/|m|)/dm = -sign(m)/m**2 - verified against finite differences
        # of calc_lambda(..., friction_model=None/"nikuradse") for both signs of m (see git
        # history); a previous version of this branch omitted sign(m), returning the m > 0 value
        # unchanged for m < 0 instead of flipping its sign.
        lambda_der[pos] = -np.sign(m[pos]) * (64 * eta[pos] * area[pos]) / (m[pos] ** 2 * d[pos])
        return lambda_der
