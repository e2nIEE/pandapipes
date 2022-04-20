# Copyright (c) 2020-2022 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import linalg
from scipy.sparse.linalg import spsolve

from pandapipes.component_models import Junction
from pandapipes.component_models.auxiliaries import build_system_matrix
from pandapipes.pipeflow_setup import get_net_option, get_net_options, set_net_option, \
    init_options, create_internal_results, write_internal_results, extract_all_results, \
    get_lookup, create_lookups, initialize_pit, check_connectivity, reduce_pit, \
    extract_results_active_pit, set_user_pf_options, update_pit, init_idx
from pandapipes.pipeflow_setup import init_fluid
from pandapipes.properties.fluids import is_fluid_gas
from pandapower.auxiliary import ppException

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def set_logger_level_pipeflow(level):
    """
    Set logger level from outside to reduce/extend pipeflow() printout.
    :param level: levels according to 'logging' (i. e. DEBUG, INFO, WARNING, ERROR and CRITICAL)
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
        >>> pipeflow(net, mode="hydraulic")

    """

    local_params = dict(locals())

    # Inputs & initialization of variables
    # ------------------------------------------------------------------------------------------
    # Init physical constants and options
    init_options(net, local_params)

    init_fluid(net)

    init_idx(net)

    create_lookups(net)

    node_pit, branch_pit, node_element_pit = initialize_pit(net, Junction.table_name())

    if (len(node_pit) == 0) & (len(branch_pit) == 0):
        logger.warning("There are no node and branch entries defined. This might mean that your net"
                       " is empty")
        return

    calculation_mode = get_net_option(net, "mode")

    if get_net_option(net, "check_connectivity"):
        nodes_connected, branches_connected, node_elements_connected = check_connectivity(
            net, branch_pit, node_pit, node_element_pit, check_heat=calculation_mode in ["heat", "all"])
    else:
        nodes_connected = node_pit[:, net['_idx_node']['ACTIVE']].astype(bool)
        branches_connected = branch_pit[:, net['_idx_branch']['ACTIVE']].astype(bool)
        node_elements_connected = node_element_pit[:, net['_idx_node_element']['ACTIVE']].astype(bool)

    reduce_pit(net, node_pit, branch_pit, node_element_pit,
               nodes_connected, branches_connected, node_elements_connected)

    update_pit(net, Junction.table_name())

    if calculation_mode == "hydraulics":
        hydraulics(net)
    elif calculation_mode == "heat":
        if net.user_pf_options["hyd_flag"]:
            node_pit = net["_active_pit"]["node"]
            node_pit[:, net['_idx_node']['PINIT']] = sol_vec[:len(node_pit)]
            branch_pit = net["_active_pit"]["branch"]
            branch_pit[:, net['_idx_branch']['VINIT']] = sol_vec[len(node_pit):]
            heat_transfer(net)
        else:
            logger.warning("Converged flag not set. Make sure that hydraulic calculation results "
                           "are available.")
    elif calculation_mode == "all":
        hydraulics(net)
        heat_transfer(net)
    else:
        logger.warning("No proper calculation mode chosen.")

    extract_results_active_pit(net, node_pit, branch_pit, nodes_connected, branches_connected)
    extract_all_results(net, Junction.table_name())


def hydraulics(net):
    net["converged"] = False
    net["OPF_converged"] = False
    max_iter, nonlinear_method, tol_p, tol_v, tol_m, tol_w, tol_res = get_net_options(
        net, "iter", "nonlinear_method", "tol_p", "tol_v", "tol_m", "tol_w", "tol_res")

    # Start of nonlinear loop
    # ---------------------------------------------------------------------------------------------
    niter = 0
    create_internal_results(net)
    if not get_net_option(net, "reuse_internal_data") or "_internal_data" not in net:
        net["_internal_data"] = dict()

    # This branch is used to stop the solver after a specified error tolerance is reached
    error_v, error_p, error_m, error_w, residual_norm = [], [], [], [], None

    # This loop is left as soon as the solver converged
    while not get_net_option(net, "converged") and niter <= max_iter:
        logger.debug("niter %d" % niter)

        # solve_hydraulics is where the calculation takes place
        if niter == 0:
            first_iter = True
            len_fluid = 0
        else:
            first_iter = False
            len_fluid = len(net._fluid) - 1

        results, epsilon = solve_hydraulics(net, first_iter)

        # Error estimation & convergence plot
        dv_init = np.abs(results[2 + len_fluid + 1] - results[0])
        dp_init = np.abs(results[3 + len_fluid + 1] - results[1])
        dm_init = np.abs(results[4 + len_fluid + 1] - results[2])
        if len_fluid:
            w_list = [res[1] - res[0] for res in zip(results[3:2 + len_fluid + 1], results[4 + len_fluid + 2:])]
            w_error = [linalg.norm(w) / (len(w)) if len(w) else 0 for w in w_list]
            w_error = max(w_error)
        residual_norm = (linalg.norm(epsilon) / (len(epsilon)))
        error_v.append(linalg.norm(dv_init) / (len(dv_init)) if len(dv_init) else 0)
        error_p.append(linalg.norm(dp_init / (len(dp_init))))
        error_m.append(linalg.norm(dm_init) / (len(dm_init)) if len(dm_init) else 0)
        if len_fluid:
            error_w.append(w_error)
        else:
            error_w.append(0)

        # Control of damping factor
        if nonlinear_method == "automatic":
            error_x0_increased, error_x1_increased = set_damping_factor(net, niter,
                                                                        [error_p, error_v])
            if error_x0_increased:
                net["_active_pit"]["node"][:, net['_idx_node']['PINIT']] = results[3 + len(net._fluid)]
            if error_x1_increased:
                net["_active_pit"]["branch"][:, net['_idx_branch']['VINIT']] = results[2 + len(net._fluid)]
        elif nonlinear_method != "constant":
            logger.warning("No proper nonlinear method chosen. Using constant settings.")

        # Setting convergence flag
        if error_v[niter] <= tol_v and error_p[niter] <= tol_p and \
                error_m[niter] <= tol_m and error_w[niter] <= tol_w and \
                residual_norm < tol_res:
            if nonlinear_method != "automatic":
                set_net_option(net, "converged", True)
            elif get_net_option(net, "alpha") == 1:
                set_net_option(net, "converged", True)

        logger.debug("errorv: %s" % error_v[niter])
        logger.debug("errorp: %s" % error_p[niter])
        logger.debug("errorm: %s" % error_m[niter])
        logger.debug("errorw: %s" % error_w[niter])
        logger.debug("alpha: %s" % get_net_option(net, "alpha"))
        niter += 1
    write_internal_results(net, iterations=niter, error_p=error_p[niter - 1],
                           error_v=error_v[niter - 1], residual_norm=residual_norm)

    logger.debug(
        "---------------------------------------------------------------------------------")
    if get_net_option(net, "converged") is False:
        logger.warning("Maximum number of iterations reached but hydraulics solver did not "
                       "converge.")
        logger.debug("Norm of residual: %s" % residual_norm)
    else:
        logger.debug("Calculation completed. Preparing results...")
        logger.debug("Converged after %d iterations." % niter)
        logger.debug("Norm of residual: %s" % residual_norm)
        logger.debug("tol_p: %s" % get_net_option(net, "tol_p"))
        logger.debug("tol_v: %s" % get_net_option(net, "tol_v"))
        logger.debug("tol_m: %s" % get_net_option(net, "tol_m"))
        net['converged'] = True

    if not get_net_option(net, "reuse_internal_data"):
        net.pop("_internal_data", None)
    set_user_pf_options(net, hyd_flag=True)

    return niter


def heat_transfer(net):
    max_iter, nonlinear_method, tol_t, tol_res = get_net_options(
        net, "iter", "nonlinear_method", "tol_T", "tol_res")

    # Start of nonlinear loop
    # ---------------------------------------------------------------------------------------------

    if is_fluid_gas(net):
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

        # Control of damping factor
        if nonlinear_method == "automatic":
            error_x0_increased, error_x1_increased = set_damping_factor(net, niter,
                                                                        [error_t, error_t_out])
            if error_x0_increased:
                net["_active_pit"]["node"][:, net['_idx_node']['TINIT']] = t_init_old
            if error_x1_increased:
                net["_active_pit"]["branch"][:, net['_idx_branch']['T_OUT']] = t_out_old
        elif nonlinear_method != "constant":
            logger.warning("No proper nonlinear method chosen. Using constant settings.")

        # Setting convergence flag
        if error_t[niter] <= tol_t and error_t_out[niter] <= tol_t \
                and residual_norm < tol_res:
            if nonlinear_method != "automatic":
                set_net_option(net, "converged", True)
            elif get_net_option(net, "alpha") == 1:
                set_net_option(net, "converged", True)

        logger.debug("errorT: %s" % error_t[niter])
        logger.debug("alpha: %s" % get_net_option(net, "alpha"))
        niter += 1
        logger.debug("F: %s" % epsilon.round(4))
        logger.debug("T_init_: %s" % t_init.round(4))
        logger.debug("T_out_: %s" % t_out.round(4))

    write_internal_results(net, iterations_T=niter, error_T=error_t[niter - 1],
                           residual_norm_T=residual_norm)
    logger.debug(
        "---------------------------------------------------------------------------------")
    if get_net_option(net, "converged") is False:
        logger.warning("Maximum number of iterations reached but heat transfer solver did not "
                       "converge.")
        logger.debug("Norm of residual: %s" % residual_norm)
    else:
        logger.debug("Calculation completed. Preparing results...")
        logger.debug("Converged after %d iterations." % niter)
        logger.debug("Norm of residual: %s" % residual_norm)
        logger.debug("tol_T: %s" % get_net_option(net, "tol_T"))

        net['converged'] = True

    return niter


def solve_hydraulics(net, first_iter):
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
    node_element_pit = net['_active_pit']['node_element']

    branch_lookups = get_lookup(net, "branch", "from_to_active")
    ne_mask = node_element_pit[:, net._idx_node_element['NODE_ELEMENT_TYPE']].astype(bool)
    if len(net._fluid) != 1:
        for comp in np.concatenate([net['node_element_list'], net['node_list'], net['branch_list']]):
            comp.create_property_pit_node_entries(net, node_pit, Junction.table_name())
            comp.create_property_pit_branch_entries(net, node_pit, branch_pit, Junction.table_name())
    for comp in net['branch_list']:
        comp.calculate_derivatives_hydraulic(net, branch_pit, node_pit, branch_lookups, options)
    jacobian, epsilon = build_system_matrix(net, branch_pit, node_pit, node_element_pit, False, first_iter)

    if first_iter:
        len_fl = 0
    else:
        len_fl = len(net._fluid) - 1
    results = []
    results += [branch_pit[:, net['_idx_branch']['VINIT']].copy()]
    results += [node_pit[:, net['_idx_node']['PINIT']].copy()]
    results += [node_element_pit[ne_mask, net['_idx_node_element']['MINIT']].copy()]
    if len_fl:
        w_node = get_lookup(net, 'node', 'w')
        for no in node_pit[:, w_node[:-1]].T.copy():
            results += [no]

    x = spsolve(jacobian, epsilon)
    branch_pit[:, net['_idx_branch']['VINIT']] += x[len(node_pit):len(branch_pit) + len(node_pit)]
    node_pit[:, net['_idx_node']['PINIT']] += x[:len(node_pit)] * options["alpha"]
    ne_len = len(branch_pit) + len(node_pit) + len(node_element_pit[ne_mask])
    node_element_pit[ne_mask, net['_idx_node_element']['MINIT']] += \
        x[len(branch_pit) + len(node_pit): ne_len]
    if len_fl:
        node_pit[:, w_node[:-1]] += x[ne_len:].reshape((len_fl, -1)).T
        node_pit[:, w_node[-1]] = 1 - node_pit[:, w_node[:-1]].sum(axis=1)

    results += [branch_pit[:, net['_idx_branch']['VINIT']].copy()]
    results += [node_pit[:, net['_idx_node']['PINIT']].copy()]
    results += [node_element_pit[ne_mask, net['_idx_node_element']['MINIT']].copy()]
    if len_fl:
        for no in node_pit[:, w_node[:-1]].T.copy():
            results += [no]

    return results, epsilon


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
    node_element_pit = net['_active_pit']['node_element']
    branch_lookups = get_lookup(net, "branch", "from_to_active")

    # Negative velocity values are turned to positive ones (including exchange of from_node and
    # to_node for temperature calculation
    branch_pit[:, net['_idx_branch']['VINIT_T']] = branch_pit[:, net['_idx_branch']['VINIT']]
    branch_pit[:, net['_idx_branch']['FROM_NODE_T']] = branch_pit[:, net['_idx_branch']['FROM_NODE']]
    branch_pit[:, net['_idx_branch']['TO_NODE_T']] = branch_pit[:, net['_idx_branch']['TO_NODE']]
    mask = branch_pit[:, net['_idx_branch']['VINIT']] < 0
    branch_pit[mask, net['_idx_branch']['VINIT_T']] = -branch_pit[mask, net['_idx_branch']['VINIT']]
    branch_pit[mask, net['_idx_branch']['FROM_NODE_T']] = branch_pit[mask, net['_idx_branch']['TO_NODE']]
    branch_pit[mask, net['_idx_branch']['TO_NODE_T']] = branch_pit[mask, net['_idx_branch']['FROM_NODE']]

    for comp in net['branch_list']:
        comp.calculate_derivatives_thermal(net, branch_pit, node_pit, branch_lookups, options)
    jacobian, epsilon = build_system_matrix(net, branch_pit, node_pit, node_element_pit, True, False)

    t_init_old = node_pit[:, net['_idx_node']['TINIT']].copy()
    t_out_old = branch_pit[:, net['_idx_branch']['T_OUT']].copy()

    x = spsolve(jacobian, epsilon)
    node_pit[:, net['_idx_node']['TINIT']] += x[:len(node_pit)] * options["alpha"]
    branch_pit[:, net['_idx_branch']['T_OUT']] += x[len(node_pit):]

    return branch_pit[:, net['_idx_branch']['T_OUT']], t_out_old, node_pit[:,
                                                                  net['_idx_node']['TINIT']], t_init_old, epsilon


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


class PipeflowNotConverged(ppException):
    """
    Exception being raised in case pipeflow did not converge.
    """
    pass
