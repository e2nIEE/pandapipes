import numpy as np

from pandapipes.idx_branch import LENGTH, D, K, RE, LAMBDA, LOAD_VEC_BRANCHES, \
    JAC_DERIV_DM, JAC_DERIV_DP, JAC_DERIV_DP1, LOAD_VEC_NODES, JAC_DERIV_DM_NODE, \
    FROM_NODE, TO_NODE, FROM_NODE_T, TOUTINIT, TEXT, AREA, ALPHA, TL, QEXT, LOAD_VEC_NODES_T, \
    LOAD_VEC_BRANCHES_T, JAC_DERIV_DT, JAC_DERIV_DT1, JAC_DERIV_DT_NODE, MINIT, MINIT_T
from pandapipes.idx_node import TINIT as TINIT_NODE
from pandapipes.properties.fluids import get_fluid
from pandapipes.constants import NORMAL_TEMPERATURE
from pandapipes.properties.properties_toolbox import get_branch_density, get_branch_eta, get_branch_cp


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
    fluid = get_fluid(net)
    gas_mode = fluid.is_gas
    friction_model = options["friction_model"]
    rho = get_branch_density(net, fluid, node_pit, branch_pit)
    eta = get_branch_eta(net, fluid, node_pit, branch_pit)
    rho_n = fluid.get_density([NORMAL_TEMPERATURE] * len(branch_pit))

    lambda_, re = calc_lambda(
        branch_pit[:, MINIT], eta, branch_pit[:, D],
        branch_pit[:, K], gas_mode, friction_model, branch_pit[:, LENGTH], options, branch_pit[:, AREA])
    der_lambda = calc_der_lambda(branch_pit[:, MINIT], eta,
                                 branch_pit[:, D], branch_pit[:, K], friction_model, lambda_, branch_pit[:, AREA])
    branch_pit[:, RE] = re
    branch_pit[:, LAMBDA] = lambda_
    from_nodes = branch_pit[:, FROM_NODE].astype(np.int32)
    to_nodes = branch_pit[:, TO_NODE].astype(np.int32)
    tinit_branch, height_difference, p_init_i_abs, p_init_i1_abs = \
        get_derived_values(node_pit, from_nodes, to_nodes, options["use_numba"])

    if not gas_mode:
        if options["use_numba"]:
            from pandapipes.pf.derivative_toolbox_numba import derivatives_hydraulic_incomp_numba \
                as derivatives_hydraulic_incomp
        else:
            from pandapipes.pf.derivative_toolbox import derivatives_hydraulic_incomp_np \
                as derivatives_hydraulic_incomp

        load_vec, load_vec_nodes, df_dm, df_dm_nodes, df_dp, df_dp1 = derivatives_hydraulic_incomp(
            branch_pit, der_lambda, p_init_i_abs, p_init_i1_abs, height_difference, rho, rho_n)
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
        comp_fact = fluid.get_compressibility(p_m)
        # TODO: this might not be required
        der_comp = fluid.get_der_compressibility() * der_p_m
        der_comp1 = fluid.get_der_compressibility() * der_p_m1
        load_vec, load_vec_nodes, df_dm, df_dm_nodes, df_dp, df_dp1 = derivatives_hydraulic_comp(
            node_pit, branch_pit, lambda_, der_lambda, p_init_i_abs, p_init_i1_abs, height_difference,
            comp_fact, der_comp, der_comp1, rho, rho_n)

    branch_pit[:, LOAD_VEC_BRANCHES] = load_vec
    branch_pit[:, JAC_DERIV_DM] = df_dm
    branch_pit[:, JAC_DERIV_DP] = df_dp
    branch_pit[:, JAC_DERIV_DP1] = df_dp1
    branch_pit[:, LOAD_VEC_NODES] = load_vec_nodes
    branch_pit[:, JAC_DERIV_DM_NODE] = df_dm_nodes


def calculate_derivatives_thermal(net, branch_pit, node_pit, options):
    fluid = get_fluid(net)
    cp = get_branch_cp(net, fluid, node_pit, branch_pit)
    m_init = branch_pit[:, MINIT_T]
    from_nodes = branch_pit[:, FROM_NODE_T].astype(np.int32)
    t_init_i = node_pit[from_nodes, TINIT_NODE]
    t_init_i1 = branch_pit[:, TOUTINIT]
    t_amb = branch_pit[:, TEXT]
    length = branch_pit[:, LENGTH]
    alpha = branch_pit[:, ALPHA] * np.pi * branch_pit[:, D]
    tl = branch_pit[:, TL]
    qext = branch_pit[:, QEXT]
    t_m = (t_init_i1 + t_init_i) / 2

    branch_pit[:, LOAD_VEC_BRANCHES_T] = \
        -(cp * m_init * (-t_init_i + t_init_i1 - tl) - alpha * (t_amb - t_m) * length + qext)

    branch_pit[:, JAC_DERIV_DT] = - cp * m_init + alpha / 2 * length
    branch_pit[:, JAC_DERIV_DT1] = cp * m_init + alpha / 2 * length

    branch_pit[:, JAC_DERIV_DT_NODE] = m_init
    branch_pit[:, LOAD_VEC_NODES_T] = m_init * t_init_i1


def get_derived_values(node_pit, from_nodes, to_nodes, use_numba):
    if use_numba:
        from pandapipes.pf.derivative_toolbox_numba import calc_derived_values_numba
        return calc_derived_values_numba(node_pit, from_nodes, to_nodes)
    from pandapipes.pf.derivative_toolbox import calc_derived_values_np
    return calc_derived_values_np(node_pit, from_nodes, to_nodes)


def calc_lambda(m, eta, d, k, gas_mode, friction_model, lengths, options, area):
    """
    Function calculates the friction factor of a pipe. Turbulence is calculated based on
    Nikuradse. If v equals 0, a value of 0.001 is used in order to avoid division by zero.
    This should not be a problem as the pressure loss term will equal zero (lambda * u^2).

    :param m:
    :type m:
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
            calc_lambda_nikuradse_incomp, colebrook_numba as colebrook, \
            calc_lambda_nikuradse_comp_numba as calc_lambda_nikuradse_comp
    else:
        from pandapipes.pf.derivative_toolbox import calc_lambda_nikuradse_incomp_np as \
            calc_lambda_nikuradse_incomp, colebrook_np as colebrook, \
            calc_lambda_nikuradse_comp_np as calc_lambda_nikuradse_comp
    if gas_mode:
        re, lambda_laminar, lambda_nikuradse = calc_lambda_nikuradse_comp(m, d, k, eta, area)
    else:
        re, lambda_laminar, lambda_nikuradse = calc_lambda_nikuradse_incomp(m, d, k, eta, area)

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
        lambda_swamee_jain = 0.25 / ((np.log10(k / (3.7 * d) + 5.74 / (re ** 0.9))) ** 2)
        return lambda_swamee_jain, re
    else:
        # lambda_tot = np.where(re > 2300, lambda_laminar + lambda_nikuradse, lambda_laminar)
        lambda_tot = lambda_laminar + lambda_nikuradse
        return lambda_tot, re


def calc_der_lambda(m, eta, d, k, friction_model, lambda_pipe, area):
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
    m_corr = np.where(np.abs(m) < 0.00001, 0.00001, m)

    if friction_model == "colebrook":
        b_term = 2.51 * eta * area / (m_corr * d * np.sqrt(lambda_pipe)) + k / (3.71 * d)

        df_dm = -2 * 2.51 * eta * area / (m_corr ** 2 * np.sqrt(lambda_pipe) * d) \
                / (np.log(10) * b_term)

        df_dlambda = -0.5 * lambda_pipe ** (-3 / 2) - (2.51 * eta * area / (d * m_corr)) \
                     * lambda_pipe ** (-3 / 2) / (np.log(10) * b_term)

        lambda_colebrook_der = df_dm / df_dlambda

        return lambda_colebrook_der
    elif friction_model == "swamee-jain":
        param = k / (3.7 * d) + 5.74 * (np.abs(eta * area)) ** 0.9 / ((np.abs(m_corr * d)) ** 0.9)
        # 0.5 / (log(10) * log(param)^3 * param) * 5.166 * abs(eta)^0.9  / (abs(rho * d)^0.9
        # * abs(v_corr)^1.9)
        lambda_swamee_jain_der = 0.5 * np.log(10) ** 2 / (np.log(param) ** 3) / param * 5.166 \
                                 * np.abs(eta * area) ** 0.9 / ((np.abs(d) ** 0.9)
                                                         * np.abs(m_corr) ** 1.9)
        return lambda_swamee_jain_der
    else:
        lambda_laminar_der = -(64 * eta * area) / (m_corr ** 2 * d)
        return lambda_laminar_der