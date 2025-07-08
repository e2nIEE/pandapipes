# Copyright (c) 2020-2025 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import linalg
from scipy.sparse.linalg import spsolve

from pandapipes.idx_branch import MDOTINIT, TOUTINIT, FROM_NODE_T_SWITCHED
from pandapipes.idx_node import PINIT, TINIT, MDOTSLACKINIT, NODE_TYPE, P
from pandapipes.pf.build_system_matrix import build_system_matrix
from pandapipes.pf.derivative_calculation import (calculate_derivatives_hydraulic,
                                                  calculate_derivatives_thermal)
from pandapipes.pf.pipeflow_setup import (
    get_net_option, get_net_options, set_net_option, init_options, create_internal_results,
    write_internal_results, get_lookup, create_lookups, initialize_pit, reduce_pit,
    set_user_pf_options, init_all_result_tables, identify_active_nodes_branches,
    check_infeed_number, PipeflowNotConverged
)
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
    calculation_mode = get_net_option(net, "mode")

    # init result tables
    net.converged = False
    init_all_result_tables(net)

    create_lookups(net)
    initialize_pit(net)

    calculation_mode = get_net_option(net, "mode")
    calculate_hydraulics = calculation_mode in ["hydraulics", 'sequential']
    calculate_heat = calculation_mode in ["heat", 'sequential']
    calculate_bidrect = calculation_mode == "bidirectional"


    # TODO: This is not necessary in every time step, but we need the result! The result of the
    #       connectivity check is currently not saved anywhere!
    # cannot be moved to calculate_hydraulics as the active node/branch hydraulics lookup is also required to
    # determine the active node/branch heat transfer lookup
    identify_active_nodes_branches(net)

    if calculation_mode == 'heat':
        use_given_hydraulic_results(net, sol_vec)

    if not (calculate_hydraulics | calculate_heat | calculate_bidrect):
        raise UserWarning("No proper calculation mode chosen.")
    elif calculate_bidrect:
        bidirectional(net)
    else:
        if calculate_hydraulics:
            hydraulics(net)
        if calculate_heat:
            heat_transfer(net)

    extract_all_results(net, calculation_mode)


def use_given_hydraulic_results(net, sol_vec):
    node_pit = net["_pit"]["node"]
    branch_pit = net["_pit"]["branch"]

    if not net.user_pf_options["hyd_flag"]:
        raise UserWarning("Converged flag not set. Make sure that hydraulic calculation "
                          "results are available.")
    else:
        node_pit[:, PINIT] = sol_vec[:len(node_pit)]
        branch_pit[:, MDOTINIT] = sol_vec[len(node_pit):]


def newton_raphson(net, funct, mode, solver_vars, tols, pit_names, iter_name):
    max_iter, nonlinear_method, tol_res = get_net_options(
        net, iter_name, "nonlinear_method", "tol_res"
    )
    niter = 0
    # This branch is used to stop the solver after a specified error tolerance is reached
    errors = {var: [] for var in solver_vars}
    create_internal_results(net)
    residual_norm = None
    # This loop is left as soon as the solver converged
    # Assumes this loop is the Newton-Raphson iteration loop
    # 1: ODE -> integrate to get function y(0)
    # 2: Build Jacobian matrix df1/dx1, df1/dx2 etc. (this means take derivative of each variable x1,x2,x3...)
    # 3: Consider initial guess for x1,x2,x3,... this is a vector x(0) = [x1,x2,x3,x4,]
    # 4: Compute value of Jacobian at these guesses x(0) above
    # 5: Take inverse of Jacobian (not always able to thus LU decomposition, spsolve...)
    # 6: Evaluate function from step 1 at the initial guesses from step 3
    # 7 The first iteration is the: initial_guess_vector - Jacobian@initial_guess * function vector@initial_guess
    #                            x(1)   = x(0) - J^-1(x(0) *F(0)
    # The repeat from step 3 again until error convergence
    #                            x(2)   = x(1) - J^-1(x(1) *F(1)
    # note: Jacobian equations don't change, just the X values subbed in at each iteration which
    # makes the jacobian different
    while not net.converged and niter < max_iter:
        logger.debug("niter %d" % niter)

        # solve_hydraulics is where the calculation takes place
        results, residual = funct(net)
        residual_norm = linalg.norm(residual / len(residual))
        logger.debug("residual: %s" % residual_norm.round(4))
        pos = np.arange(len(solver_vars) * 2)
        results = np.array(results, object)
        vals_new = results[pos[::2]]
        vals_old = results[pos[1::2]]
        for var, val_new, val_old in zip(solver_vars, vals_new, vals_old):
            dval = val_new - val_old
            errors[var].append(linalg.norm(dval) / len(dval) if len(dval) else 0)
        finalize_iteration(
            net, niter, residual_norm, nonlinear_method, errors=errors, tols=tols, tol_res=tol_res,
            vals_old=vals_old, solver_vars=solver_vars, pit_names=pit_names
        )
        niter += 1
    write_internal_results(net, **errors)
    kwargs = dict()
    kwargs['residual_norm_%s' % mode] = residual_norm
    kwargs['iterations_%s' % mode] = niter
    write_internal_results(net, **kwargs)
    log_final_results(net, mode, niter, residual_norm, solver_vars, tols)


def bidirectional(net):
    net.converged = False
    if not get_net_option(net, "reuse_internal_data") or "_internal_data" not in net:
        net["_internal_data"] = dict()
    solver_vars = ['mdot', 'p', 'TOUT', 'T']
    tol_m, tol_p, tol_temp = get_net_options(net, 'tol_m', 'tol_p', 'tol_T')
    newton_raphson(
        net, solve_bidirectional, 'bidirectional', solver_vars, [tol_m, tol_p, tol_temp, tol_temp],
        ['branch', 'node', 'branch', 'node'], 'max_iter_bidirect'
    )
    if net.converged:
        set_user_pf_options(net, hyd_flag=True)
    if not get_net_option(net, "reuse_internal_data"):
        net.pop("_internal_data", None)
    if not net.converged:
        raise PipeflowNotConverged("The bidrectional calculation did not converge to a solution.")


def hydraulics(net):
    # Start of nonlinear loop
    # ---------------------------------------------------------------------------------------------
    net.converged = False
    reduce_pit(net, mode="hydraulics")
    if not get_net_option(net, "reuse_internal_data") or "_internal_data" not in net:
        net["_internal_data"] = dict()
    solver_vars = ['mdot', 'p', 'mdotslack']
    tol_p, tol_m, tol_msl = get_net_options(net, 'tol_m', 'tol_p', 'tol_m')
    newton_raphson(net, solve_hydraulics, 'hydraulics', solver_vars, [tol_m, tol_p, tol_msl],
                   ['branch', 'node', 'node'], 'max_iter_hyd')
    if net.converged:
        set_user_pf_options(net, hyd_flag=True)

    if not get_net_option(net, "reuse_internal_data"):
        net.pop("_internal_data", None)

    if not net.converged:
        msg = "The hydraulic calculation did not converge to a solution."
        raise PipeflowNotConverged(msg)
    extract_results_active_pit(net, mode="hydraulics")


def heat_transfer(net):
    # Start of nonlinear loop
    # ---------------------------------------------------------------------------------------------
    net.converged = False
    identify_active_nodes_branches(net, False)
    reduce_pit(net, mode="heat_transfer")
    if net.fluid.is_gas:
        logger.info("Caution! Temperature calculation does currently not affect hydraulic "
                    "properties!")
    solver_vars = ['Tout', 'T']
    tol_temp = next(get_net_options(net, 'tol_T'))
    newton_raphson(net, solve_temperature, 'heat', solver_vars, [tol_temp, tol_temp], ['branch', 'node'],
                   'max_iter_therm')
    if not net.converged:
        msg = "The heat transfer calculation did not converge to a solution."
        raise PipeflowNotConverged(msg)
    extract_results_active_pit(net, mode="heat_transfer")


def solve_bidirectional(net):
    reduce_pit(net, mode="hydraulics")
    res_hyd, residual_hyd = solve_hydraulics(net)
    extract_results_active_pit(net, mode="hydraulics")
    identify_active_nodes_branches(net, False)
    reduce_pit(net, mode="heat_transfer")
    res_heat, residual_heat = solve_temperature(net)
    extract_results_active_pit(net, mode="heat_transfer")
    residual = np.concatenate([residual_hyd, residual_heat])
    res = res_hyd + res_heat
    return res, residual


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

    branch_lookups = get_lookup(net, "branch", "from_to_active_hydraulics")
    for comp in net['component_list']:
        comp.adaption_before_derivatives_hydraulic(net, branch_pit, node_pit, branch_lookups,
                                                   options)
    calculate_derivatives_hydraulic(net, branch_pit, node_pit, options)
    for comp in net['component_list']:
        comp.adaption_after_derivatives_hydraulic(
            net, branch_pit, node_pit, branch_lookups, options)
    # epsilon is node [pressure] slack nodes and load vector branch prsr difference
    # jacobian is the derivatives
    jacobian, epsilon = build_system_matrix(net, branch_pit, node_pit, False)

    m_init_old = branch_pit[:, MDOTINIT].copy()
    p_init_old = node_pit[:, PINIT].copy()
    slack_nodes = np.where(node_pit[:, NODE_TYPE] == P)[0]
    msl_init_old = node_pit[slack_nodes, MDOTSLACKINIT].copy()

    # x is next step pressures and velocity
    x = spsolve(jacobian, epsilon)

    branch_pit[:, MDOTINIT] -= x[len(node_pit):len(node_pit) + len(branch_pit)] * options["alpha"]
    node_pit[:, PINIT] -= x[:len(node_pit)] * options["alpha"]
    node_pit[slack_nodes, MDOTSLACKINIT] -= x[len(node_pit) + len(branch_pit):]

    return [branch_pit[:, MDOTINIT], m_init_old, node_pit[:, PINIT], p_init_old, msl_init_old,
            node_pit[slack_nodes, MDOTSLACKINIT]], epsilon


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
    branch_lookups = get_lookup(net, "branch", "from_to_active_heat_transfer")

    # Negative velocity values are turned to positive ones (including exchange of from_node and
    # to_node for temperature calculation
    branch_pit[:, FROM_NODE_T_SWITCHED] = branch_pit[:, MDOTINIT] < -2e-11

    for comp in net['component_list']:
        comp.adaption_before_derivatives_thermal(net, branch_pit, node_pit, branch_lookups, options)
    calculate_derivatives_thermal(net, branch_pit, node_pit, options)
    for comp in net['component_list']:
        comp.adaption_after_derivatives_thermal(net, branch_pit, node_pit, branch_lookups, options)

    t_init_old = node_pit[:, TINIT].copy()
    t_out_old = branch_pit[:, TOUTINIT].copy()

    if not check_infeed_number(node_pit):
        return [branch_pit[:, TOUTINIT], t_out_old, node_pit[:, TINIT], t_init_old], np.array([
            np.nan])

    jacobian, epsilon = build_system_matrix(net, branch_pit, node_pit, True)

    x = spsolve(jacobian, epsilon)

    node_pit[:, TINIT] -= x[:len(node_pit)] * options["alpha"]
    branch_pit[:, TOUTINIT] -= x[len(node_pit):] * options["alpha"]

    return [branch_pit[:, TOUTINIT], t_out_old, node_pit[:, TINIT], t_init_old], epsilon


def set_damping_factor(net, niter, errors):
    """
    Set the value of the damping factor (factor for the newton step width) from current results.

    :param net: the net for which to perform the pipeflow
    :type net: pandapipesNet
    :param niter:
    :type niter:
    :param errors: an array containing the current residuals of all field variables solved for
    :return: No Output.

    EXAMPLE:
        set_damping_factor(net, niter, [error_p, error_v])
    """
    error_increased = []
    for error in errors.values():
        error_increased.append(error[niter] > error[niter - 1])
    current_alpha = get_net_option(net, "alpha")
    if np.all(error_increased):
        set_net_option(net, "alpha", current_alpha / 10 if current_alpha >= 0.1 else current_alpha)
    else:
        set_net_option(net, "alpha", current_alpha * 10 if current_alpha <= 0.1 else 1.0)
    return error_increased


def finalize_iteration(net, niter, residual_norm, nonlinear_method, errors, tols, tol_res, vals_old,
                       solver_vars, pit_names):
    # Control of damping factor
    if nonlinear_method == "automatic":
        errors_increased = set_damping_factor(net, niter, errors)
        logger.debug("alpha: %s" % get_net_option(net, "alpha"))
        for error_increased, var, val, pit in zip(errors_increased, solver_vars, vals_old,
                                                  pit_names):
            if error_increased:
                # todo: not working in bidirectional mode as bidirectional is not distinguishing \
                #  between hydraulics and heat transfer active pit
                net["_active_pit"][pit][:, globals()[var.upper() + 'INIT']] = val
        if get_net_option(net, "alpha") != 1:
            net.converged = False
            return
    elif nonlinear_method != "constant":
        logger.warning("No proper nonlinear method chosen. Using constant settings.")
    converged = True
    for error, var, tol in zip(errors.values(), solver_vars, tols):
        converged = error[niter] <= tol
        if not converged: break
        logger.debug("error_%s: %s" % (var, error[niter]))
    net.converged = converged and residual_norm <= tol_res


def log_final_results(net, solver, niter, residual_norm, solver_vars, tols):
    logger.debug("--------------------------------------------------------------------------------")
    if not net.converged:
        logger.debug(
            "Maximum number of iterations reached but %s solver did not converge." % solver)
        logger.debug("Norm of residual: %s" % residual_norm)
    else:
        logger.debug("Calculation completed. Preparing results...")
        logger.debug("Converged after %d iterations." % niter)
        logger.debug("Norm of residual: %s" % residual_norm)
        for var, tol in zip(solver_vars, tols):
            logger.debug("tolerance for %s: %s" % (var, tol))
