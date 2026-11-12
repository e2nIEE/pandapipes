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
    :return: df_dm, df_dp, df_dp1, df_dm_nodes, residual_vec, residual_vec_nodes_from, residual_vec_nodes_to
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
    area = branch_area(b_pit)

    # Darcy Friction factor: lambda
    re = (
        np.abs(b_pit[:, IdxBranch.MDOTINIT])
        * b_pit[:, IdxBranch.D]
        / (eta * area)
    )
    mask = (
        ~np.isclose(re, 0)
        & ~np.isclose(b_pit[:, IdxBranch.LENGTH], 0, rtol=1e-10, atol=1e-11)
    )
    k_over_D = b_pit[mask, IdxBranch.K] / b_pit[mask, IdxBranch.D]
    lambda_ = np.zeros_like(re)
    der_lambda = np.zeros_like(re)
    friction_factor_model = options["friction_model"]
    lambda_[mask], der_lambda[mask] = (
        friction_factor_model.compute_lambda_and_dlambda_dm(
            k_over_D,
            re[mask],
            b_pit[mask, IdxBranch.MDOTINIT],
        )
    )

    b_pit[:, IdxBranch.RE]     = re
    b_pit[:, IdxBranch.LAMBDA] = lambda_

    if not gas_mode:
        residual_vec, residual_vec_nodes_from, residual_vec_nodes_to, df_dm, df_dm_nodes, df_dp, df_dp1, dp_frict_loss = (
            derivatives_hydraulic_incomp(b_pit, der_lambda, p_init_i_abs, p_init_i1_abs, height_difference, rho, area))
    else:
        rho_n = np.full(len(b_pit), fluid.get_density(NORMAL_TEMPERATURE))
        comp_fact = fluid.get_compressibility(p_m, tinit_branch)
        dc = fluid.get_der_compressibility()
        der_comp = dc * der_p_m
        der_comp1 = dc * der_p_m1
        residual_vec, residual_vec_nodes_from, residual_vec_nodes_to, df_dm, df_dm_nodes, df_dp, df_dp1, dp_frict_loss = (
            derivatives_hydraulic_comp(node_pit, b_pit, lambda_, der_lambda, p_init_i_abs, p_init_i1_abs,
                height_difference, comp_fact, der_comp, der_comp1, rho, rho_n, area))

    b_pit[:, IdxBranch.DP_FRICT_LOSS] = dp_frict_loss

    return df_dm, df_dp, df_dp1, df_dm_nodes, residual_vec, residual_vec_nodes_from, residual_vec_nodes_to


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
    cp_n = (cp_i1 + cp_nt) / 2
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
