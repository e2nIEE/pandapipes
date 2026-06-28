import numpy as np

from pandapipes.constants import NORMAL_TEMPERATURE
from pandapipes.idx_branch import (LENGTH, D, K, RE, LAMBDA, LOAD_VEC_BRANCHES, JAC_DERIV_DM, JAC_DERIV_DP,
                                   JAC_DERIV_DP1, JAC_DERIV_DM_NODE, FROM_NODE, TO_NODE, TOUTINIT, AREA,
                                   LOAD_VEC_BRANCHES_T, JAC_DERIV_DT, LOAD_VEC_NODES_TO_T,
                                   LOAD_VEC_NODES_FROM, LOAD_VEC_NODES_TO, JAC_DERIV_DT_NODE, JAC_DERIV_DTOUT_NODE,
                                   JAC_DERIV_DTOUT, MDOTINIT, DP_FRICT_LOSS)
from pandapipes.idx_node import TINIT as TINIT_NODE, INFEED, LOAD_T, JAC_DERIV_DT_N
from pandapipes.pf.internals_toolbox import get_from_nodes_corrected, get_to_nodes_corrected
from pandapipes.pf.pipeflow_setup import get_net_option, get_lookup
from pandapipes.properties.fluids import get_fluid
from pandapipes.properties.properties_toolbox import get_branch_real_density, get_branch_real_eta, get_branch_cp


def calculate_derivatives_hydraulic(net,
                                    branch_pit, node_pit,
                                    branch_pit_old, node_pit_old,
                                    options):
    """
    Function which creates derivatives.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param branch_pit:
    :type branch_pit:
    :param node_pit:
    :type node_pit:
    :param options:
    :type options:
    :return: No Output.
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

    from_nodes = branch_pit[:, FROM_NODE].astype(np.int32)
    to_nodes = branch_pit[:, TO_NODE].astype(np.int32)
    tinit_branch, height_difference, p_init_i_abs, p_init_i1_abs = get_derived_values(node_pit, from_nodes, to_nodes,
                                                                                      options["use_numba"])

    if gas_mode:
        p_m, der_p_m, der_p_m1 = calc_medium_pressure_with_derivative(p_init_i_abs, p_init_i1_abs)
    else:
        p_m, der_p_m, der_p_m1 = (p_init_i_abs + p_init_i1_abs) / 2, None, None

    rho = get_branch_real_density(fluid, node_pit, branch_pit)
    eta = get_branch_real_eta(fluid, node_pit, branch_pit, p_m)

    # Darcy Friction factor: lambda
    re = (
        np.abs(branch_pit[:, MDOTINIT])
        * branch_pit[:, D]
        / (eta * branch_pit[:, AREA])
    )
    mask = (
        ~np.isclose(re, 0)
        & ~np.isclose(branch_pit[:, LENGTH], 0, rtol=1e-10, atol=1e-11)
    )
    k_over_D = branch_pit[mask, K] / branch_pit[mask, D]
    lambda_ = np.zeros_like(re)
    der_lambda = np.zeros_like(re)
    friction_factor_model = options["friction_model"]
    lambda_[mask], der_lambda[mask] = (
        friction_factor_model.compute_lambda_and_dlambda_dm(
            k_over_D,
            re[mask],
            branch_pit[mask, MDOTINIT],
        )
    )

    branch_pit[:, RE] = re
    branch_pit[:, LAMBDA] = lambda_

    if not gas_mode:
        load_vec, load_vec_nodes_from, load_vec_nodes_to, df_dm, df_dm_nodes, df_dp, df_dp1, dp_frict_loss = (
            derivatives_hydraulic_incomp(branch_pit, der_lambda, p_init_i_abs, p_init_i1_abs, height_difference, rho))
    else:
        rho_n = np.full(len(branch_pit), fluid.get_density(NORMAL_TEMPERATURE))
        comp_fact = fluid.get_compressibility(p_m, tinit_branch)
        dc = fluid.get_der_compressibility()
        # TODO: this might not be required
        der_comp = dc * der_p_m
        der_comp1 = dc * der_p_m1
        load_vec, load_vec_nodes_from, load_vec_nodes_to, df_dm, df_dm_nodes, df_dp, df_dp1, dp_frict_loss = (
            derivatives_hydraulic_comp(node_pit, branch_pit, lambda_, der_lambda, p_init_i_abs, p_init_i1_abs,
                height_difference, comp_fact, der_comp, der_comp1, rho, rho_n))

    branch_pit[:, LOAD_VEC_BRANCHES] = load_vec
    branch_pit[:, JAC_DERIV_DM] = df_dm
    branch_pit[:, JAC_DERIV_DP] = df_dp
    branch_pit[:, JAC_DERIV_DP1] = df_dp1
    branch_pit[:, LOAD_VEC_NODES_FROM] = load_vec_nodes_from
    branch_pit[:, LOAD_VEC_NODES_TO] = load_vec_nodes_to
    branch_pit[:, JAC_DERIV_DM_NODE] = df_dm_nodes
    branch_pit[:, DP_FRICT_LOSS] = dp_frict_loss


def calculate_derivatives_thermal(net,
                                  branch_pit, node_pit,
                                  branch_pit_old, node_pit_old,
                                  options):
    node_pit_old_lookup = get_lookup(net, "node", "old_pit_cols")
    branch_pit_old_lookup = get_lookup(net, "branch", "old_pit_cols")

    if options["use_numba"]:
        from pandapipes.pf.derivative_toolbox_numba import derivatives_thermal_numba as derivatives_termal
    else:
        from pandapipes.pf.derivative_toolbox import derivatives_thermal_np as derivatives_termal
    fluid = get_fluid(net)
    cp_b = get_branch_cp(fluid, node_pit, branch_pit)
    # this is not required currently, but useful when implementing leakages
    # m_init_i = np.abs(branch_pit[:, MDOTINIT])
    # m_init_i1 = np.abs(branch_pit[:, MDOTINIT])
    from_nodes = get_from_nodes_corrected(branch_pit)
    to_nodes = get_to_nodes_corrected(branch_pit)
    t_init_i = node_pit[from_nodes, TINIT_NODE]
    t_init_i1 = branch_pit[:, TOUTINIT]
    t_init_nt = node_pit[to_nodes, TINIT_NODE]
    t_init_n = node_pit[:, TINIT_NODE]
    cp_i1 = fluid.get_heat_capacity(t_init_i1)
    cp_nt = fluid.get_heat_capacity(t_init_nt)
    cp_n = fluid.get_heat_capacity((cp_i1 + cp_nt) / 2)
    transient = get_net_option(net, "transient")
    dt = get_net_option(net, "dt")
    rho = get_branch_real_density(fluid, node_pit, branch_pit)
    amb = get_net_option(net, 'ambient_temperature')

    fn, dfn_dt, fnt, dfnt_dt, dfnt_dtout, fb, dfb_dt, dfb_dtout, infeed = (
        derivatives_termal(node_pit, branch_pit,
                           node_pit_old, node_pit_old_lookup,
                           branch_pit_old, branch_pit_old_lookup,
                           from_nodes, to_nodes,
                           t_init_i, t_init_i1, t_init_nt, t_init_n,
                           cp_n, cp_b,
                           rho, dt, transient, amb))

    node_pit[:, LOAD_T] = fn
    node_pit[:, JAC_DERIV_DT_N] = dfn_dt

    branch_pit[:, LOAD_VEC_BRANCHES_T] = fb
    branch_pit[:, JAC_DERIV_DT] = dfb_dt
    branch_pit[:, JAC_DERIV_DTOUT] = dfb_dtout

    branch_pit[:, LOAD_VEC_NODES_TO_T] = fnt
    branch_pit[:, JAC_DERIV_DT_NODE] = dfnt_dt
    branch_pit[:, JAC_DERIV_DTOUT_NODE] = dfnt_dtout

    node_pit[:, INFEED] = False
    node_pit[infeed, INFEED] = True


def get_derived_values(node_pit, from_nodes, to_nodes, use_numba):
    if use_numba:
        from pandapipes.pf.derivative_toolbox_numba import calc_derived_values_numba
        return calc_derived_values_numba(node_pit, from_nodes, to_nodes)
    from pandapipes.pf.derivative_toolbox import calc_derived_values_np
    return calc_derived_values_np(node_pit, from_nodes, to_nodes)
