import numpy as np
from pandapipes.constants import NORMAL_TEMPERATURE
from pandapipes.idx_branch import (LENGTH, D, K, RE, LAMBDA, LOAD_VEC_BRANCHES, JAC_DERIV_DM, JAC_DERIV_DP,
                                   JAC_DERIV_DP1, JAC_DERIV_DM_NODE, FROM_NODE, TO_NODE, TOUTINIT, AREA,
                                   LOAD_VEC_BRANCHES_T, JAC_DERIV_DT, JAC_DERIV_DT_NODE,
                                   LOAD_VEC_NODES_FROM, LOAD_VEC_NODES_TO, LOAD_VEC_NODES_FROM_T, LOAD_VEC_NODES_TO_T,
                                   JAC_DERIV_DTOUT, JAC_DERIV_DTOUT_NODE, MDOTINIT)
from pandapipes.idx_node import (TINIT as TINIT_NODE, INFEED, LOAD_T, JAC_DERIV_DT_LOAD, JAC_DERIV_DT_SLACK,
                                 )
from pandapipes.pf.internals_toolbox import get_from_nodes_corrected, get_to_nodes_corrected
from pandapipes.pf.pipeflow_setup import get_net_option, get_lookup
from pandapipes.properties.fluids import get_fluid
from pandapipes.properties.properties_toolbox import get_branch_real_density, get_branch_real_eta, get_branch_cp
from scipy.optimize import newton


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
    lambda_, re = calc_lambda(branch_pit[:, MDOTINIT], eta, branch_pit[:, D], branch_pit[:, K], gas_mode,
        friction_model, branch_pit[:, LENGTH], options, branch_pit[:, AREA])
    der_lambda = calc_der_lambda(branch_pit[:, MDOTINIT], eta, branch_pit[:, D], branch_pit[:, K], friction_model,
                                 lambda_, branch_pit[:, AREA], re, branch_pit[:, LENGTH])
    branch_pit[:, RE] = re
    branch_pit[:, LAMBDA] = lambda_

    if not gas_mode:
        load_vec, load_vec_nodes_from, load_vec_nodes_to, df_dm, df_dm_nodes, df_dp, df_dp1 = (
            derivatives_hydraulic_incomp(branch_pit, der_lambda, p_init_i_abs, p_init_i1_abs, height_difference, rho))
    else:
        rho_n = np.full(len(branch_pit), fluid.get_density(NORMAL_TEMPERATURE))
        comp_fact = fluid.get_compressibility(p_m, tinit_branch)
        dc = fluid.get_der_compressibility()
        # TODO: this might not be required
        der_comp = dc * der_p_m
        der_comp1 = dc * der_p_m1
        load_vec, load_vec_nodes_from, load_vec_nodes_to, df_dm, df_dm_nodes, df_dp, df_dp1 = (
            derivatives_hydraulic_comp(node_pit, branch_pit, lambda_, der_lambda, p_init_i_abs, p_init_i1_abs,
                height_difference, comp_fact, der_comp, der_comp1, rho, rho_n))

    branch_pit[:, LOAD_VEC_BRANCHES] = load_vec
    branch_pit[:, JAC_DERIV_DM] = df_dm
    branch_pit[:, JAC_DERIV_DP] = df_dp
    branch_pit[:, JAC_DERIV_DP1] = df_dp1
    branch_pit[:, LOAD_VEC_NODES_FROM] = load_vec_nodes_from
    branch_pit[:, LOAD_VEC_NODES_TO] = load_vec_nodes_to
    branch_pit[:, JAC_DERIV_DM_NODE] = df_dm_nodes


def calculate_derivatives_thermal(net, branch_pit, node_pit, options):
    if options["use_numba"]:
        from pandapipes.pf.derivative_toolbox_numba import derivatives_thermal_numba as derivatives_termal
    else:
        from pandapipes.pf.derivative_toolbox import derivatives_thermal_np as derivatives_termal
    fluid = get_fluid(net)
    cp = get_branch_cp(fluid, node_pit, branch_pit)
    # this is not required currently, but useful when implementing leakages
    # m_init_i = np.abs(branch_pit[:, MDOTINIT])
    # m_init_i1 = np.abs(branch_pit[:, MDOTINIT])
    from_nodes = get_from_nodes_corrected(branch_pit)
    to_nodes = get_to_nodes_corrected(branch_pit)
    t_init_i = node_pit[from_nodes, TINIT_NODE]
    t_init_i1 = branch_pit[:, TOUTINIT]
    t_init_n =  node_pit[:, TINIT_NODE]
    cp_i = fluid.get_heat_capacity(t_init_i)
    cp_i1 = fluid.get_heat_capacity(t_init_i1)
    cp_n = fluid.get_heat_capacity(t_init_n)
    transient = get_net_option(net, "transient")
    dt = get_net_option(net, "dt")
    rho = get_branch_real_density(fluid, node_pit, branch_pit)
    amb = get_net_option(net, 'ambient_temperature')

    fn, dfn_dt, dfn_dts, fb, dfb_dt, dfb_dtout, fbf, fbt, dfbf_dt, dfbt_dtout, infeed = (
        derivatives_termal(node_pit, branch_pit, from_nodes, to_nodes, t_init_i, t_init_i1, t_init_n, cp_i, cp_i1, cp_n,
                           cp, rho, dt, transient, amb))

    node_pit[:, LOAD_T] = fn
    node_pit[:, JAC_DERIV_DT_LOAD] = dfn_dt
    node_pit[:, JAC_DERIV_DT_SLACK] = dfn_dts

    branch_pit[:, LOAD_VEC_BRANCHES_T] = fb
    branch_pit[:, JAC_DERIV_DT] = dfb_dt
    branch_pit[:, JAC_DERIV_DTOUT] = dfb_dtout

    branch_pit[:, LOAD_VEC_NODES_FROM_T] = fbf
    branch_pit[:, LOAD_VEC_NODES_TO_T] = fbt
    branch_pit[:, JAC_DERIV_DT_NODE] = dfbf_dt
    branch_pit[:, JAC_DERIV_DTOUT_NODE] = dfbt_dtout

    node_pit[:, INFEED] = False
    node_pit[infeed, INFEED] = True


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
            calc_lambda_nikuradse_comp_numba as calc_lambda_nikuradse_comp)
    else:
        from pandapipes.pf.derivative_toolbox import (calc_lambda_nikuradse_incomp_np as calc_lambda_nikuradse_incomp,
                                                      calc_lambda_nikuradse_comp_np as calc_lambda_nikuradse_comp)
    if gas_mode:
        re, lambda_laminar, lambda_nikuradse = calc_lambda_nikuradse_comp(m, d, k, eta, area)
    else:
        re, lambda_laminar, lambda_nikuradse = calc_lambda_nikuradse_incomp(m, d, k, eta, area)

    if friction_model == "colebrook":
        # TODO: move this import to top level if possible
        from pandapipes.pipeflow import PipeflowNotConverged
        max_iter = options.get("max_iter_colebrook", 100)
        tolerance = options.get("tolerance_colebrook", 1e-4)
        converged, lambda_colebrook = colebrook_white(re, d, k, lambda_nikuradse, max_iter, lengths, tolerance)
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
    """
    Function calculates the derivative of lambda with respect to v. Turbulence is calculated based
    on Nikuradse. This should not be a problem as the pressure loss term will equal zero
    (lambda * u^2).

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
        b_term[pos] = (2.51 * eta[pos] * area[pos] / (m[pos] * d[pos] * np.sqrt(lambda_pipe[pos])) + k[pos] / (
                    3.71 * d[pos]))

        df_dm[pos] = -2 * 2.51 * eta[pos] * area[pos] / (m[pos] ** 2 * np.sqrt(lambda_pipe[pos]) * d[pos]) / (
                    np.log(10) * b_term[pos])

        df_dlambda[pos] = -0.5 * lambda_pipe[pos] ** (-3 / 2) - (2.51 * eta[pos] * area[pos] / (d[pos] * m[pos])) * \
                          lambda_pipe[pos] ** (-3 / 2) / (np.log(10) * b_term[pos])

        lambda_der[pos] = df_dm[pos] / df_dlambda[pos]

        return lambda_der
    elif friction_model == "swamee-jain":
        param = (k[pos] / (3.7 * d[pos]) + 5.74 * ((eta[pos] * area[pos]) / (np.abs(m[pos]) * d[pos])) ** 0.9)
        # 0.5 / (log(10) * log(param)^3 * param) * 5.166 * abs(eta)^0.9  / (abs(rho * d)^0.9
        # * abs(v_corr)^1.9)
        lambda_der[pos] = 0.5 * np.log(10) ** 2 / (np.log(param) ** 3) / param * 5.166 * (
                    (eta[pos] * area[pos]) / (d[pos])) ** 0.9 * np.abs(m[pos]) ** -1.9
        return lambda_der
    else:
        lambda_der[pos] = -(64 * eta[pos] * area[pos]) / (m[pos] ** 2 * d[pos])
        return lambda_der


def colebrook_white(re, d, k, lambda_nikuradse, max_iter, lengths, tolerance=1e-4):
    """
    Function calculates the friction factor of a pipe using the Colebrook-White equation. It is an
    implicit equation which is solved using the Newton-Raphson method. For pipes with zero flow or
    zero length, the initial guess is returned. This should be uncritical, as the pressure loss
    term will equal zero (lambda * u^2 * l / d).

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

    res = newton(colebrook_white_implicit, lambda_res[mask], maxiter=max_iter, args=(re[mask], k[mask], d[mask]),
                 tol=tolerance, full_output=True, fprime=cw_derivative)  # , fprime2=cw_derivative_2)

    if lambda_res[mask].size == 1:
        lambda_res[mask] = res[0]
        converged = res[1].converged
    else:
        lambda_res[mask] = res.root
        converged = np.all(res.converged)

    return converged, lambda_res
