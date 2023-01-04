# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import linalg
from pandapower.auxiliary import ppException
from scipy.sparse.linalg import spsolve

from pandapipes.idx_branch import ACTIVE as ACTIVE_BR, FROM_NODE, TO_NODE, FROM_NODE_T, \
    TO_NODE_T, VINIT, T_OUT, VINIT_T
from pandapipes.idx_node import PINIT, TINIT, ACTIVE as ACTIVE_ND
from pandapipes.pf.build_system_matrix import build_system_matrix
from pandapipes.pf.derivative_calculation import calculate_derivatives_hydraulic
from pandapipes.pf.pipeflow_setup import get_net_option, get_net_options, set_net_option, \
    init_options, create_internal_results, write_internal_results, get_lookup, create_lookups, \
    initialize_pit, check_connectivity, reduce_pit, \
    set_user_pf_options, init_all_result_tables
from pandapipes.pf.result_extraction import extract_all_results, extract_results_active_pit

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def set_logger_level_pipeflow(level):
    """
    Set logger level from outside to reduce/extend pipeflow() printout.
    :param level: levels according to 'logging' (i.e. DEBUG, INFO, WARNING, ERROR and CRITICAL)
    :type level: str
    :return: No output

    EXAMPLE:
        set_logger_level_pipeflow('WARNING')

    """
    logger.setLevel(level)


def pipeflow(net, sol_vec=None, **kwargs):
    """
    The main method used to start the solver to calculate the velocity, pressure and temperature\
    distribution for a given net. Different options can be entered for \\**kwargs, which control\
    the solver behaviour (see function :func:`init_options` for more information).

    :param net: The pandapipes net for which to perform the pipeflow
    :type net: pandapipesNet
    :param sol_vec: Initializes the start values for the heating network calculation
    :type sol_vec: numpy.ndarray, default None
    :param kwargs: A list of options controlling the solver behaviour
    :return: No output

    :Example:
        >>> pipeflow(net, mode="hydraulics")

    """
    local_params = dict(locals())

    # Inputs & initialization of variables
    # ------------------------------------------------------------------------------------------

    # Init physical constants and options
    init_options(net, local_params)

    # init result tables
    net["converged"] = False
    init_all_result_tables(net)

    create_lookups(net)
    node_pit, branch_pit = initialize_pit(net)
    if (len(node_pit) == 0) & (len(branch_pit) == 0):
        logger.warning("There are no node and branch entries defined. This might mean that your net"
                       " is empty")
        return
    calculation_mode = get_net_option(net, "mode")
    calculate_hydraulics = calculation_mode in ["hydraulics", "all"]
    calculate_heat = calculation_mode in ["heat", "all"]

    if get_net_option(net, "check_connectivity"):
        nodes_connected, branches_connected = check_connectivity(
            net, branch_pit, node_pit, check_heat=calculate_heat)
    else:
        nodes_connected = node_pit[:, ACTIVE_ND].astype(np.bool)
        branches_connected = branch_pit[:, ACTIVE_BR].astype(np.bool)

    reduce_pit(net, node_pit, branch_pit, nodes_connected, branches_connected)

    if calculation_mode == "heat" and not net.user_pf_options["hyd_flag"]:
        raise UserWarning("Converged flag not set. Make sure that hydraulic calculation results "
                          "are available.")
    elif calculation_mode == "heat" and net.user_pf_options["hyd_flag"]:
        net["_active_pit"]["node"][:, PINIT] = sol_vec[:len(node_pit)]
        net["_active_pit"]["branch"][:, VINIT] = sol_vec[len(node_pit):]

    if calculate_hydraulics:
        converged, _ = hydraulics(net)
        if not converged:
            raise PipeflowNotConverged("The hydraulic calculation did not converge to a solution.")

    if calculate_heat:
        converged, _ = heat_transfer(net)
        if not converged:
            raise PipeflowNotConverged("The heat transfer calculation did not converge to a "
                                       "solution.")
    elif not calculate_hydraulics:
        raise UserWarning("No proper calculation mode chosen.")

    extract_results_active_pit(net, node_pit, branch_pit, nodes_connected, branches_connected)
    extract_all_results(net, nodes_connected, branches_connected)


def hydraulics(net):
    max_iter, nonlinear_method, tol_p, tol_v, tol_res = get_net_options(
        net, "iter", "nonlinear_method", "tol_p", "tol_v", "tol_res")

    # Start of nonlinear loop
    # ---------------------------------------------------------------------------------------------
    niter = 0
    create_internal_results(net)
    if not get_net_option(net, "reuse_internal_data") or "_internal_data" not in net:
        net["_internal_data"] = dict()

    # This branch is used to stop the solver after a specified error tolerance is reached
    error_v, error_p, residual_norm = [], [], None

    # This loop is left as soon as the solver converged
    while not get_net_option(net, "converged") and niter <= max_iter:
        logger.debug("niter %d" % niter)

        # solve_hydraulics is where the calculation takes place
        v_init, p_init, v_init_old, p_init_old, epsilon = solve_hydraulics(net)

        # Error estimation & convergence plot
        dv_init = np.abs(v_init - v_init_old)
        dp_init = np.abs(p_init - p_init_old)

        residual_norm = linalg.norm(epsilon) / len(epsilon)
        error_v.append(linalg.norm(dv_init) / len(dv_init) if len(dv_init) else 0)
        error_p.append(linalg.norm(dp_init / len(dp_init)))

        finalize_iteration(net, niter, error_p, error_v, residual_norm, nonlinear_method, tol_p,
                           tol_v, tol_res, p_init_old, v_init_old)
        niter += 1
    write_internal_results(net, iterations=niter, error_p=error_p[niter - 1],
                           error_v=error_v[niter - 1], residual_norm=residual_norm)

    converged = get_net_option(net, "converged")
    net['converged'] = converged
    if converged:
        set_user_pf_options(net, hyd_flag=True)

    log_final_results(net, converged, niter, residual_norm)
    if not get_net_option(net, "reuse_internal_data"):
        net.pop("_internal_data", None)

    return converged, niter


def heat_transfer(net):
    max_iter, nonlinear_method, tol_t, tol_res = get_net_options(
        net, "iter", "nonlinear_method", "tol_T", "tol_res")

    # Start of nonlinear loop
    # ---------------------------------------------------------------------------------------------

    if net.fluid.is_gas:
        logger.info("Caution! Temperature calculation does currently not affect hydraulic "
                    "properties!")

    error_t, error_t_out, residual_norm = [], [], None

    set_net_option(net, "converged", False)
    niter = 0

    # This loop is left as soon as the solver converged
    while not get_net_option(net, "converged") and niter <= max_iter:
        logger.debug("niter %d" % niter)

        # solve_hydraulics is where the calculation takes place
        t_out, t_out_old, t_init, t_init_old, epsilon = solve_temperature(net)

        # Error estimation & convergence plot
        delta_t_init = np.abs(t_init - t_init_old)
        delta_t_out = np.abs(t_out - t_out_old)

        residual_norm = (linalg.norm(epsilon) / (len(epsilon)))
        error_t.append(linalg.norm(delta_t_init) / (len(delta_t_init)))
        error_t_out.append(linalg.norm(delta_t_out) / (len(delta_t_out)))

        finalize_iteration(net, niter, error_t, error_t_out, residual_norm, nonlinear_method, tol_t,
                           tol_t, tol_res, t_init_old, t_out_old, hyraulic_mode=True)
        logger.debug("F: %s" % epsilon.round(4))
        logger.debug("T_init_: %s" % t_init.round(4))
        logger.debug("T_out_: %s" % t_out.round(4))
        niter += 1

    write_internal_results(net, iterations_T=niter, error_T=error_t[niter - 1],
                           residual_norm_T=residual_norm)

    converged = get_net_option(net, "converged")
    net['converged'] = converged
    log_final_results(net, converged, niter, residual_norm, hyraulic_mode=False)

    return converged, niter


def solve_hydraulics(net):
    """
    Create and solve the linearized system of equations (based on a jacobian in form of a scipy
    sparse matrix and a load vector in form of a numpy array) in order to calculate the hydraulic
    magnitudes (pressure and velocity) for the network nodes and branches.

    :param net: The pandapipesNet for which to solve the hydraulic matrix
    :type net: pandapipesNet
    :return:

    """
    options = net["_options"]
    branch_pit = net["_active_pit"]["branch"]
    node_pit = net["_active_pit"]["node"]

    branch_lookups = get_lookup(net, "branch", "from_to_active")
    for comp in net['component_list']:
        comp.adaption_before_derivatives_hydraulic(
            net, branch_pit, node_pit, branch_lookups, options)
    calculate_derivatives_hydraulic(net, branch_pit, node_pit, options)
    for comp in net['component_list']:
        comp.adaption_after_derivatives_hydraulic(
            net, branch_pit, node_pit, branch_lookups, options)
    jacobian, epsilon = build_system_matrix(net, branch_pit, node_pit, False)

    v_init_old = branch_pit[:, VINIT].copy()
    p_init_old = node_pit[:, PINIT].copy()

    x = spsolve(jacobian, epsilon)
    branch_pit[:, VINIT] += x[len(node_pit):]
    node_pit[:, PINIT] += x[:len(node_pit)] * options["alpha"]

    return branch_pit[:, VINIT], node_pit[:, PINIT], v_init_old, p_init_old, epsilon


def solve_temperature(net):
    """
    This function contains the procedure to build and solve a linearized system of equation based on
    an underlying net and the necessary graph data structures. Temperature values are calculated.
    Returned are the solution vectors for the new iteration, the original solution vectors and a
    vector containing component indices for the system matrix entries

    :param net: The pandapipesNet for which to solve the temperature matrix
    :type net: pandapipesNet
    :return: branch_pit

    """

    options = net["_options"]
    branch_pit = net["_active_pit"]["branch"]
    node_pit = net["_active_pit"]["node"]
    branch_lookups = get_lookup(net, "branch", "from_to_active")

    # Negative velocity values are turned to positive ones (including exchange of from_node and
    # to_node for temperature calculation
    branch_pit[:, VINIT_T] = branch_pit[:, VINIT]
    branch_pit[:, FROM_NODE_T] = branch_pit[:, FROM_NODE]
    branch_pit[:, TO_NODE_T] = branch_pit[:, TO_NODE]
    mask = branch_pit[:, VINIT] < 0
    branch_pit[mask, VINIT_T] = -branch_pit[mask, VINIT]
    branch_pit[mask, FROM_NODE_T] = branch_pit[mask, TO_NODE]
    branch_pit[mask, TO_NODE_T] = branch_pit[mask, FROM_NODE]

    for comp in net['component_list']:
        comp.calculate_derivatives_thermal(net, branch_pit, node_pit, branch_lookups, options)
    jacobian, epsilon = build_system_matrix(net, branch_pit, node_pit, True)

    t_init_old = node_pit[:, TINIT].copy()
    t_out_old = branch_pit[:, T_OUT].copy()

    x = spsolve(jacobian, epsilon)
    node_pit[:, TINIT] += x[:len(node_pit)] * options["alpha"]
    branch_pit[:, T_OUT] += x[len(node_pit):]

    return branch_pit[:, T_OUT], t_out_old, node_pit[:, TINIT], t_init_old, epsilon


def set_damping_factor(net, niter, error):
    """
    Set the value of the damping factor (factor for the newton step width) from current results.

    :param net: the net for which to perform the pipeflow
    :type net: pandapipesNet
    :param niter:
    :type niter:
    :param error: an array containing the current residuals of all field variables solved for
    :return: No Output.

    EXAMPLE:
        set_damping_factor(net, niter, [error_p, error_v])
    """
    error_x0 = error[0]
    error_x1 = error[1]

    error_x0_increased = error_x0[niter] > error_x0[niter - 1]
    error_x1_increased = error_x1[niter] > error_x1[niter - 1]
    current_alpha = get_net_option(net, "alpha")

    if error_x0_increased and error_x1_increased:
        set_net_option(net, "alpha", current_alpha / 10 if current_alpha >= 0.1 else current_alpha)
    else:
        set_net_option(net, "alpha", current_alpha * 10 if current_alpha <= 0.1 else 1.0)

    return error_x0_increased, error_x1_increased


def finalize_iteration(net, niter, error_1, error_2, residual_norm, nonlinear_method, tol_1, tol_2,
                       tol_res, vals_1_old, vals_2_old, hyraulic_mode=True):
    col1, col2 = (PINIT, VINIT) if hyraulic_mode else (TINIT, T_OUT)

    # Control of damping factor
    if nonlinear_method == "automatic":
        error_x0_increased, error_x1_increased = set_damping_factor(net, niter,
                                                                    [error_1, error_2])
        if error_x0_increased:
            net["_active_pit"]["node"][:, col1] = vals_1_old
        if error_x1_increased:
            net["_active_pit"]["branch"][:, col2] = vals_2_old
    elif nonlinear_method != "constant":
        logger.warning("No proper nonlinear method chosen. Using constant settings.")

    # Setting convergence flag
    if error_2[niter] <= tol_2 and error_1[niter] <= tol_1 and residual_norm < tol_res:
        if nonlinear_method != "automatic":
            set_net_option(net, "converged", True)
        elif get_net_option(net, "alpha") == 1:
            set_net_option(net, "converged", True)

    if hyraulic_mode:
        logger.debug("errorv: %s" % error_1[niter])
        logger.debug("errorp: %s" % error_2[niter])
        logger.debug("alpha: %s" % get_net_option(net, "alpha"))
    else:
        logger.debug("errorT: %s" % error_1[niter])
        logger.debug("alpha: %s" % get_net_option(net, "alpha"))


def log_final_results(net, converged, niter, residual_norm, hyraulic_mode=True):
    if hyraulic_mode:
        solver = "hydraulics"
        outputs = ["tol_p", "tol_v"]
    else:
        solver = "heat transfer"
        outputs = ["tol_T"]
    logger.debug("--------------------------------------------------------------------------------")
    if not converged:
        logger.debug("Maximum number of iterations reached but %s solver did not converge."
                     % solver)
        logger.debug("Norm of residual: %s" % residual_norm)
    else:
        logger.debug("Calculation completed. Preparing results...")
        logger.debug("Converged after %d iterations." % niter)
        logger.debug("Norm of residual: %s" % residual_norm)
        for out in outputs:
            logger.debug("%s: %s" % (out, get_net_option(net, out)))


class PipeflowNotConverged(ppException):
    """
    Exception being raised in case pipeflow did not converge.
    """
    pass
