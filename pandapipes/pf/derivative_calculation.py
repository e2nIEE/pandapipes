import numpy as np


from pandapipes.properties.fluids import is_fluid_gas, get_fluid, get_mixture_compressibility, \
    get_mixture_der_cmpressibility
from pandapipes.pf.pipeflow_setup import get_lookup


def calculate_derivatives_hydraulic(net, branch_pit, node_pit, options):
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

    gas_mode = is_fluid_gas(net)
    friction_model = options["friction_model"]

    lambda_, re = calc_lambda(
        branch_pit[:, net['_idx_branch']['VINIT']],
        branch_pit[:, net['_idx_branch']['ETA']],
        branch_pit[:, net['_idx_branch']['RHO']],
        branch_pit[:, net['_idx_branch']['D']],
        branch_pit[:, net['_idx_branch']['K']],
        gas_mode, friction_model,
        branch_pit[:, net['_idx_branch']['LENGTH']], options)
    der_lambda = calc_der_lambda(
        branch_pit[:, net['_idx_branch']['VINIT']],
        branch_pit[:, net['_idx_branch']['ETA']],
        branch_pit[:, net['_idx_branch']['RHO']],
        branch_pit[:, net['_idx_branch']['D']],
        branch_pit[:, net['_idx_branch']['K']],
        friction_model, lambda_)
    branch_pit[:, net['_idx_branch']['RE']] = re
    branch_pit[:, net['_idx_branch']['LAMBDA']] = lambda_
    from_nodes = branch_pit[:, net['_idx_branch']['FROM_NODE']].astype(np.int32)
    to_nodes = branch_pit[:, net['_idx_branch']['TO_NODE']].astype(np.int32)
    tinit_branch, height_difference, p_init_i_abs, p_init_i1_abs = \
        get_derived_values(net, node_pit, from_nodes, to_nodes, options["use_numba"])
    branch_pit[:, net['_idx_branch']['TINIT']] = tinit_branch

    pit_cols = np.array([net['_idx_branch']['VINIT'], net['_idx_branch']['LENGTH'], net['_idx_branch']['LAMBDA'],
                         net['_idx_branch']['D'], net['_idx_branch']['LOSS_COEFFICIENT'], net['_idx_branch']['RHO'],
                         net['_idx_branch']['PL'], net['_idx_branch']['AREA'], net['_idx_branch']['TINIT']],
                         dtype=np.int32)

    if not gas_mode:
        if options["use_numba"]:
            from pandapipes.pf.derivative_toolbox_numba import derivatives_hydraulic_incomp_numba \
                as derivatives_hydraulic_incomp
        else:
            from pandapipes.pf.derivative_toolbox import derivatives_hydraulic_incomp_np \
                as derivatives_hydraulic_incomp

        load_vec, load_vec_nodes, df_dv, df_dv_nodes, df_dp, df_dp1 = derivatives_hydraulic_incomp(
            pit_cols, branch_pit, der_lambda, p_init_i_abs, p_init_i1_abs, height_difference)
    else:
        if options["use_numba"]:
            from pandapipes.pf.derivative_toolbox_numba import derivatives_hydraulic_comp_numba \
                as derivatives_hydraulic_comp, calc_medium_pressure_with_derivative_numba as \
                calc_medium_pressure_with_derivative
        else:
            from pandapipes.pf.derivative_toolbox import derivatives_hydraulic_comp_np \
                as derivatives_hydraulic_comp, calc_medium_pressure_with_derivative_np as \
                calc_medium_pressure_with_derivative

        p_m, der_p_m, der_p_m1 = calc_medium_pressure_with_derivative(p_init_i_abs, p_init_i1_abs)
        if len(net._fluid) == 1:
            fluid = net._fluid[0]
            comp_fact = get_fluid(net, fluid).get_compressibility(p_m)
            der_comp_fact = get_fluid(net, fluid).get_der_compressibility()
            der_comp = der_comp_fact * der_p_m
            der_comp1 = der_comp_fact * der_p_m1
        else:
            w = get_lookup(net, 'branch', 'w')
            mf = branch_pit[:, w]
            comp_fact = get_mixture_compressibility(net, p_m, mf, branch_pit[:, net['_idx_branch']['TINIT']])
            der_comp_fact = get_mixture_der_cmpressibility(net, p_m, mf)
            der_comp = der_comp_fact * der_p_m
            der_comp1 = der_comp_fact * der_p_m1
        # TODO: this might not be required
        load_vec, load_vec_nodes, df_dv, df_dv_nodes, df_dp, df_dp1 = derivatives_hydraulic_comp(
            pit_cols, branch_pit, lambda_, der_lambda, p_init_i_abs, p_init_i1_abs, height_difference,
            comp_fact, der_comp, der_comp1)

    branch_pit[:, net['_idx_branch']['LOAD_VEC_BRANCHES']] = load_vec
    branch_pit[:, net['_idx_branch']['JAC_DERIV_DV']] = df_dv
    branch_pit[:, net['_idx_branch']['JAC_DERIV_DP']] = df_dp
    branch_pit[:, net['_idx_branch']['JAC_DERIV_DP1']] = df_dp1
    branch_pit[:, net['_idx_branch']['LOAD_VEC_NODES']] = load_vec_nodes
    branch_pit[:, net['_idx_branch']['JAC_DERIV_DV_NODE']] = df_dv_nodes


def get_derived_values(net, node_pit, from_nodes, to_nodes, use_numba):
    pit_cols = np.array([net['_idx_node']['TINIT'], net['_idx_node']['HEIGHT'],
                         net['_idx_node']['PINIT'], net['_idx_node']['PAMB']], dtype=np.int32)
    if use_numba:
        from pandapipes.pf.derivative_toolbox_numba import calc_derived_values_numba
        return calc_derived_values_numba(pit_cols, node_pit, from_nodes, to_nodes)
    from pandapipes.pf.derivative_toolbox import calc_derived_values_np
    return calc_derived_values_np(pit_cols, node_pit, from_nodes, to_nodes)


def calc_lambda(v, eta, rho, d, k, gas_mode, friction_model, lengths, options):
    """
    Function calculates the friction factor of a pipe. Turbulence is calculated based on
    Nikuradse. If v equals 0, a value of 0.001 is used in order to avoid division by zero.
    This should not be a problem as the pressure loss term will equal zero (lambda * u^2).

    :param v:
    :type v:
    :param eta:
    :type eta:
    :param rho:
    :type rho:
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
    :return:
    :rtype:
    """
    if options["use_numba"]:
        from pandapipes.pf.derivative_toolbox_numba import calc_lambda_nikuradse_incomp_numba as \
            calc_lambda_nikuradse_incomp, colebrook_numba as colebrook,\
            calc_lambda_nikuradse_comp_numba as calc_lambda_nikuradse_comp
    else:
        from pandapipes.pf.derivative_toolbox import calc_lambda_nikuradse_incomp_np as \
            calc_lambda_nikuradse_incomp, colebrook_np as colebrook,\
            calc_lambda_nikuradse_comp_np as calc_lambda_nikuradse_comp
    if gas_mode:
        re, lambda_laminar, lambda_nikuradse = calc_lambda_nikuradse_comp(v, d, k, eta, rho)
    else:
        re, lambda_laminar, lambda_nikuradse = calc_lambda_nikuradse_incomp(v, d, k, eta, rho)

    if friction_model == "colebrook":
        # TODO: move this import to top level if possible
        from pandapipes.pipeflow import PipeflowNotConverged
        max_iter = options.get("max_iter_colebrook", 100)
        dummy = (lengths != 0).astype(np.float64)
        converged, lambda_colebrook = colebrook(re, d, k, lambda_nikuradse, dummy, max_iter)
        if not converged:
            raise PipeflowNotConverged(
                "The Colebrook-White algorithm did not converge. There might be model "
                "inconsistencies. The maximum iterations can be given as 'max_iter_colebrook' "
                "argument to the pipeflow.")
        return lambda_colebrook, re
    elif friction_model == "swamee-jain":
        lambda_swamee_jain = 0.25 / ((np.log10(k/(3.7*d) + 5.74/(re**0.9)))**2)
        return lambda_swamee_jain, re
    else:
        # lambda_tot = np.where(re > 2300, lambda_laminar + lambda_nikuradse, lambda_laminar)
        lambda_tot = lambda_laminar + lambda_nikuradse
        return lambda_tot, re


def calc_der_lambda(v, eta, rho, d, k, friction_model, lambda_pipe):
    """
    Function calculates the derivative of lambda with respect to v. Turbulence is calculated based
    on Nikuradse. This should not be a problem as the pressure loss term will equal zero
    (lambda * u^2).

    :param v:
    :type v:
    :param eta:
    :type eta:
    :param rho:
    :type rho:
    :param d:
    :type d:
    :param k:
    :type k:
    :param friction_model:
    :type friction_model:
    :param lambda_pipe:
    :type lambda_pipe:
    :return:
    :rtype:
    """

    # TODO: check if some formulas with constants can be shortened
    v_corr = np.where(v == 0, 0.00001, v)

    if friction_model == "colebrook":
        b_term = 2.51 * eta / (rho * d * np.sqrt(lambda_pipe) * v_corr) + k / (3.71 * d)

        df_dv = -2 * (2.51 * eta / (rho * np.sqrt(lambda_pipe) * v_corr ** 2)) \
                / (np.log(10) * b_term)

        df_dlambda = -0.5 * lambda_pipe ** (-3 / 2) - (2.51 * eta / (rho * d * v_corr)) \
                     * lambda_pipe ** (-3 / 2) / (np.log(10) * b_term)

        lambda_colebrook_der = df_dv / df_dlambda

        return lambda_colebrook_der
    elif friction_model == "swamee-jain":
        param = k/(3.7*d) + 5.74 * (np.abs(eta))**0.9 / ((np.abs(rho*v_corr*d))**0.9)
        # 0.5 / (log(10) * log(param)^3 * param) * 5.166 * abs(eta)^0.9  / (abs(rho * d)^0.9
        # * abs(v_corr)^1.9)
        lambda_swamee_jain_der = 0.5 / np.log(10) / (np.log(param) ** 3) / param * 5.166 \
                                 * np.abs(eta) ** 0.9 / ((np.abs(rho * d) ** 0.9)
                                                         * np.abs(v_corr) ** 1.9)
        return lambda_swamee_jain_der
    else:
        lambda_laminar_der = -(64 * eta) / (rho * v_corr ** 2 * d)
        return lambda_laminar_der