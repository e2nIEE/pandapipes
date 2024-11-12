# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import copy
import inspect

import numpy as np
from scipy.sparse import coo_matrix, csgraph

from pandapipes.component_init import COMPONENT_REGISTRY
from pandapipes.idx_branch import FROM_NODE, TO_NODE, ACTIVE as ACTIVE_BR
from pandapipes.idx_node import NODE_TYPE, P, NODE_TYPE_T, T, ACTIVE as ACTIVE_ND, \
    TABLE_IDX as TABLE_IDX_ND, ELEMENT_IDX as ELEMENT_IDX_ND
from pandapipes.properties.fluids import get_fluid
from pandapipes.utils.internals import get_table_name, create_empty_pit, branches_connected_flow, \
    PipeflowNotConverged, get_net_option, get_lookup, _sum_by_group


try:
    import numba
    from numba import jit

    numba_installed = True
except ImportError:
    from pandapower.pf.no_numba import jit

    numba_installed = False

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

default_options = {"friction_model": "nikuradse", "tol_p": 1e-5, "tol_m": 1e-5,
                   "tol_T": 1e-3, "tol_res": 1e-3, "max_iter_hyd": 10, "max_iter_therm": 10, "max_iter_bidirect": 10,
                   "error_flag": False, "alpha": 1,
                   "nonlinear_method": "constant", "mode": "hydraulics",
                   "ambient_temperature": 293.15, "check_connectivity": True,
                   "max_iter_colebrook": 10, "only_update_hydraulic_matrix": False,
                   "reuse_internal_data": False, "use_numba": True,
                   "quit_on_inconsistency_connectivity": False, "calc_compression_power": True}


def init_options(net, local_parameters):
    """
    Initializes physical and mathematical constants included in pandapipes. In addition, options
    for the nonlinear and time-dependent solver are also set.

    Those are the options that can be set and their default values:

        - **max_iter_hyd** (int): 10 - If the hydraulics simulation is terminated after a certain amount of \
                               iterations, this is the number of iterations.

        - **max_iter_therm** (int): 10 - If the thermal simulation is terminated after a certain amount of \
                               iterations, this is the number of iterations.

        - **tol_p** (float): 1e-4 - The relative tolerance for the pressure. A result is accepted \
                                    if the relative error is smaller than this factor.

        - **tol_m** (float): 1e-4 - The relative tolerance for the velocity. A result is accepted \
                                    if the relative error is smaller than this factor.

        - **tol_T** (float): 1e-4 - The relative tolerance for the temperature. A result is \
                                    accepted if the relative error is smaller than this factor.

        - **tol_res** (float): 1e-3 - The relative tolerance for the residual. A result is accepted\
                                      if the relative error is smaller than this factor.

        - **ambient_temperature** (float): 293.0 - The assumed ambient temperature for the\
                calculation of the barometric formula

        - **friction_model** (str): "nikuradse" - The friction model that shall be used to identify\
                the value for lambda (can be "nikuradse" or "colebrook")

        - **alpha** (float): 1 - The step width for the Newton iterations. If the Newton steps \
                shall be damped, **alpha** can be reduced. See also the **nonlinear_method** \
                parameter.

        - **nonlinear_method** (str): "constant" - The option of how the damping factor **alpha** \
                is determined in each iteration. It can be "constant" (i.e. **alpha** is always the\
                 same in each iteration) or "automatic", in which case **alpha** is adapted \
                 automatically with respect to the convergence behaviour.

        - **mode** (str): "hydraulics" - Define the calculation mode: what shall be calculated - \
                solely hydraulics ('hydraulics'), solely heat transfer('heat') or both combined sequentially \
                ('sequential') or bidirectionally ('bidirectional').

        - **only_update_hydraulic_matrix** (bool): False - If True, the system matrix is not \
                created in every iteration, but only the data is updated according to a lookup that\
                is identified in the first iteration. This speeds up calculation, but has not yet\
                been tested extensively.

        - **check_connectivity** (bool): True - If True, a connectivity check is performed at the\
                beginning of the pipeflow and parts of the net that are not connected to external\
                grids are set inactive.

        - **quit_on_inconsistency_connectivity** (bool): False - If True, inconsistencies in the\
                connectivity check raise an error, otherwise they are handled. Inconsistencies mean\
                that out of service nodes are connected to in service branches. If that is the case\
                and the flag is set to False, the connected nodes are activated.

        - **use_numba** (bool): True - If True, use numba for more efficient internal calculations

    :param net: The pandapipesNet for which the options are initialized
    :type net: pandapipesNet
    :param local_parameters: Dictionary with local parameters that were passed to the pipeflow call.
    :type local_parameters: dict
    :return: No output

    :Example:
        >>> init_options(net)

    """
    from pandapipes.pipeflow import pipeflow # todo: this is only needed becaus eof circular imports

    # the base layer of the options consists of the default options
    net["_options"] = copy.deepcopy(default_options)
    excluded_params = {"net", "interactive_plotting", "t_start", "sol_vec", "kwargs"}

    # the base layer is overwritten and extended by options given by the default parameters of the
    # pipeflow function definition
    args_pf = inspect.getfullargspec(pipeflow)
    pf_func_options = dict(zip(args_pf.args[-len(args_pf.defaults):], args_pf.defaults))
    pf_func_options = {k: pf_func_options[k] for k in set(pf_func_options.keys()) - excluded_params}
    net["_options"].update(pf_func_options)

    # the third layer is the user defined pipeflow options
    if "user_pf_options" in net and len(net.user_pf_options) > 0:
        opts = _iteration_check(net.user_pf_options)
        opts = _check_mode(opts)
        net["_options"].update(opts)


    # the last layer is the layer of passeed parameters by the user, it is defined as the local
    # existing parameters during the pipeflow call which diverges from the default parameters of the
    # function definition in the second layer
    params = dict()
    for k, v in local_parameters.items():
        if k in excluded_params or (k in pf_func_options and pf_func_options[k] == v):
            continue
        params[k] = v

    opts = _iteration_check(local_parameters["kwargs"])
    opts = _check_mode(opts)
    params.update(opts)
    net["_options"].update(params)
    net["_options"]["fluid"] = get_fluid(net).name
    if not net["_options"]["only_update_hydraulic_matrix"]:
        net["_options"]["reuse_internal_data"] = False

    if not numba_installed:
        if net["_options"]["use_numba"]:
            logger.info("numba is not installed. Install numba first before you set the 'use_numba'"
                        " flag to True. The pipeflow will be performed without numba speedup.")
        net["_options"]["use_numba"] = False

def _iteration_check(opts):
    opts = copy.deepcopy(opts)
    iter_defined = False
    params = dict()
    if 'iter' in opts:
        params['max_iter_hyd'] = params['max_iter_therm'] = params['max_iter_bidirect'] = opts["iter"]
        iter_defined = True
    if 'max_iter_hyd' in opts:
        max_iter_hyd = opts["max_iter_hyd"]
        if iter_defined: logger.info("You defined 'iter' and 'max_iter_hyd. "
                                     "'max_iter_hyd' will overwrite 'iter'")
        params['max_iter_hyd'] = max_iter_hyd
    if 'max_iter_therm' in opts:
        max_iter_therm = opts["max_iter_therm"]
        if iter_defined: logger.info("You defined 'iter' and 'max_iter_therm. "
                                     "'max_iter_therm' will overwrite 'iter'")
        params['max_iter_therm'] = max_iter_therm
    if 'max_iter_bidirect' in opts:
        max_iter_bidirect = opts["max_iter_bidirect"]
        if iter_defined: logger.info("You defined 'iter' and 'max_iter_bidirect. "
                                     "'max_iter_bidirect' will overwrite 'iter'")
        params['max_iter_bidirect'] = max_iter_bidirect
    opts.update(params)
    return opts

def _check_mode(opts):
    opts = copy.deepcopy(opts)
    if 'mode' in opts and opts['mode'] == 'all':
        logger.warning("mode 'all' is deprecated and will be removed in a future release. "
                       "Use 'sequential' or 'bidirectional' instead. "
                       "For now 'all' is set equal to 'sequential'.")
        opts['mode'] = 'sequential'
    return opts


def initialize_pit(net):
    """
    Initializes and fills the internal structure which is called pit (pandapipes internal tables).
    The structure is a dictionary which should contain one array for all nodes and one array for all
    branches of the net (c.f. also `create_empty_pit`).

    :param net: The pandapipes network for which to create and fill the internal structure
    :type net: pandapipesNet
    :return: (node_pit, branch_pit) - The two internal structure arrays
    :rtype: tuple(np.array)

    """
    pit = create_empty_pit(net)

    for comp in net['component_list']:
        if hasattr(COMPONENT_REGISTRY[comp], "create_pit_node_entries"):
            COMPONENT_REGISTRY[comp].create_pit_node_entries(net, pit["node"])
        if hasattr(COMPONENT_REGISTRY[comp], "create_pit_branch_entries"):
            COMPONENT_REGISTRY[comp].create_pit_branch_entries(net, pit["branch"])
        if hasattr(COMPONENT_REGISTRY[comp], "create_component_array"):
            COMPONENT_REGISTRY[comp].create_component_array(net, pit["components"])

    if len(pit["node"]) == 0:
        logger.warning("There are no nodes defined. "
                       "You need at least one node! "
                       "Without any nodes, you are not able to conduct a pipeflow!")
        return


def init_all_result_tables(net):
    """
    Initialize the result tables of all components in the net.

    :param net: pandapipes net for which to extract results into net.res_xy
    :type net: pandapipesNet
    :return: No output

    """
    for comp in net['component_list']:
        COMPONENT_REGISTRY[comp].init_results(net)


def create_lookups(net):
    """
    Create all lookups necessary for the pipeflow of the given net.
    The lookups are usually:

      - node_from_to: The start and end indices of all node component tables within the pit
      - branch_from_to: The start and end indices of all branch component tables within the pit
      - node_table: Dictionary to determine indices for node component tables (e.g. \
                    {"junction": 0}). Can be arbitrary and strongly depends on the component order \
                    given by `get_component_list`.
      - branch_table: Dictionary to determine indices for branch component tables (e.g.\
                      {"pipe": 0, "valve": 1}). Can be arbitrary and strongly depends on the\
                      component order given by `get_component_list`.
      - node_index: Lookup from component index (e.g. junction 2) to pit index (e.g. 0) for nodes.
      - branch_index: Lookup from component index (e.g. pipe 1) to pit index (e.g. 5) for branches.
      - internal_nodes_lookup: Lookup for internal nodes of branch components that makes result\
                               extraction a lot easier.

    :param net: The pandapipes network for which to create the lookups
    :type net: pandapipesNet
    :return: No output

    """
    node_ft_lookups, node_idx_lookups, node_from, node_table_nr = dict(), dict(), 0, 0
    branch_ft_lookups, branch_idx_lookups, branch_from, branch_table_nr = dict(), dict(), 0, 0
    branch_table_lookups = {"t2n": dict(), "n2t": dict()}
    node_table_lookups = {"t2n": dict(), "n2t": dict()}
    internal_nodes_lookup = dict()

    for comp in net['component_list']:
        if hasattr(COMPONENT_REGISTRY[comp], "create_branch_lookups"):
            branch_from, branch_table_nr = COMPONENT_REGISTRY[comp].create_branch_lookups(
                net, branch_ft_lookups, branch_table_lookups, branch_idx_lookups, branch_table_nr,
                branch_from)
        if hasattr(COMPONENT_REGISTRY[comp], "create_node_lookups"):
            node_from, node_table_nr = COMPONENT_REGISTRY[comp].create_node_lookups(
                net, node_ft_lookups, node_table_lookups, node_idx_lookups, node_from, node_table_nr,
                internal_nodes_lookup)

    net["_lookups"] = {"node_from_to": node_ft_lookups, "branch_from_to": branch_ft_lookups,
                       "node_table": node_table_lookups, "branch_table": branch_table_lookups,
                       "node_index": node_idx_lookups, "branch_index": branch_idx_lookups,
                       "node_length": node_from, "branch_length": branch_from,
                       "internal_nodes_lookup": internal_nodes_lookup}


def identify_active_nodes_branches(net, hydraulic=True):
    """
    Function that creates the connectivity lookup for nodes and branches. If the option \
    "check_connectivity" is set, a full connectivity check is performed based on a sparse matrix \
    graph search. Otherwise, only the nodes and branches are identified that are inactive, which \
    means:\
      - in case of hydraulics, just use the "ACTIVE" identifier of the respective components\
      - in case of heat transfer, use the hydraulic result to check which branches are traversed \
        by the fluid and a simple rule to make sure that active nodes are connected to at least one\
        traversed branch\
    The result of this connectivity search is stored in the lookups (e.g. as \
    net["_lookups"]["node_active_hydraulics"])

    :param net: the pandapipes net for which to identify the connectivity
    :type net: pandapipes.pandapipesNet
    :param branch_pit: Internal array with branch entries
    :type branch_pit: np.array
    :param node_pit: Internal array with node entries
    :type node_pit: np.array
    :param hydraulic: flag for the mode (if True, do the check for the hydraulic simulation, \
        otherwise for the heat transfer simulation with other considerations)
    :type hydraulic: bool, default True
    :return: No output
    """

    node_pit = net["_pit"]["node"]
    branch_pit = net["_pit"]["branch"]

    if hydraulic:
        # connectivity check for hydraulic simulation
        if get_net_option(net, "check_connectivity"):
            nodes_connected, branches_connected = check_connectivity(net, branch_pit, node_pit)
        else:
            # if connectivity check is switched off, still consider oos elements
            nodes_connected = node_pit[:, ACTIVE_ND].astype(np.bool_)
            branches_connected = branch_pit[:, ACTIVE_BR].astype(np.bool_)
    else:
        # connectivity check for heat simulation (needs to consider branches with 0 velocity as
        # well)
        if get_net_option(net, "check_connectivity"):
            # full connectivity check for hydraulic simulation
            nodes_connected, branches_connected = check_connectivity(net, branch_pit, node_pit,
                                                                     mode="heat_transfer")
        else:
            # if no full connectivity check is performed, all nodes that are not connected to the
            # rest of the network wrt. flow can be identified by a more performant sum_by_group_call
            # check for branches that are not traversed (for temperature calculation, this means
            # that they are "out of service")
            branches_connected = get_lookup(net, "branch", "active_hydraulics") \
                                 & branches_connected_flow(branch_pit)
            fn = branch_pit[:, FROM_NODE].astype(np.int32)
            tn = branch_pit[:, TO_NODE].astype(np.int32)
            fn_tn, flow = _sum_by_group(
                get_net_option(net, "use_numba"), np.concatenate([fn, tn]),
                np.concatenate([branches_connected, branches_connected]).astype(np.int32)
            )
            nodes_connected = np.copy(get_lookup(net, "node", "active_hydraulics"))
            # set nodes oos that are not connected to any branches with flow > 0 (0.1 is arbitrary
            # here, any value between 0 and 1 should work, excluding 0 and 1)
            nodes_connected[fn_tn] = nodes_connected[fn_tn] & (flow > 0.1)
    mode = "hydraulics" if hydraulic else "heat_transfer"
    if np.all(~nodes_connected):
        mode = 'hydraulic' if hydraulic else 'heat transfer'
        raise PipeflowNotConverged(" All nodes are set out of service. Probably they are not supplied."
                                   " Therefore, the %s pipeflow did not converge. "
                                   " Have you forgotten to define an external grid?" % mode)
    net["_lookups"]["node_active_" + mode] = nodes_connected
    net["_lookups"]["branch_active_" + mode] = branches_connected


def check_connectivity(net, branch_pit, node_pit, mode="hydraulics"):
    """
    Perform a connectivity check which means that network nodes are identified that don't have any
    connection to an external grid component. Quick overview over the steps of this function:

      - Build a sparse matrix graph (scipy.sparse.csr_matrix) from all branches that are in_service\
        (nodes of this graph are taken from FROM_NODE and TO_NODE column in pit).
      - Add a node that represents all external grids and connect all nodes that are connected to\
        external grids to that node.
      - Perform a breadth first order search to identify all nodes that are reachable from the \
        added external grid node.
      - Create masks for exisiting nodes and branches to show if they are reachable from an \
        external grid.
      - Compare the reachable nodes with the initial in_service nodes.\n
        - If nodes are reachable that were set out of service by the user, they are either set \
          in_service or an error is raised. The behavior depends on the pipeflow option \
          **quit_on_inconsistency_connectivity**.
        - If nodes are not reachable that were set in_service by the user, they will be set out of\
          service automatically (this is the desired functionality of the connectivity check).

    :param net: The pandapipesNet for which to perform the check
    :type net: pandapipesNet
    :param branch_pit: Internal array with branch entries
    :type branch_pit: np.array
    :param node_pit: Internal array with node entries
    :type node_pit: np.array
    :return: (nodes_connected, branches_connected) - Lookups of np.arrays stating which of the
            internal nodes and branches are reachable from any of the hyd_slacks (np mask).
    :rtype: tuple(np.array)
    """
    if mode == "hydraulics":
        active_branch_lookup = branch_pit[:, ACTIVE_BR].astype(np.bool_)
        active_node_lookup = node_pit[:, ACTIVE_ND].astype(np.bool_)
        slacks = np.where((node_pit[:, NODE_TYPE] == P) & active_node_lookup)[0]
    else:
        active_branch_lookup = branches_connected_flow(branch_pit) \
                               & get_lookup(net, "branch", "active_hydraulics")
        active_node_lookup = node_pit[:, ACTIVE_ND].astype(np.bool_) \
                             & get_lookup(net, "node", "active_hydraulics")
        slacks = np.where((node_pit[:, NODE_TYPE_T] == T) & active_node_lookup)[0]

    return perform_connectivity_search(net, node_pit, branch_pit, slacks, active_node_lookup,
                                       active_branch_lookup, mode=mode)


def perform_connectivity_search(net, node_pit, branch_pit, slack_nodes,
                                active_node_lookup, active_branch_lookup, mode="hydraulics"):
    len_nodes = len(node_pit)
    from_nodes = branch_pit[:, FROM_NODE].astype(np.int32)
    to_nodes = branch_pit[:, TO_NODE].astype(np.int32)
    nobranch = np.sum(active_branch_lookup)
    active_from_nodes = from_nodes[active_branch_lookup]
    active_to_nodes = to_nodes[active_branch_lookup]

    # we create a "virtual" node that is connected to all slack nodes and start the connectivity
    # search at this node
    fn_matrix = np.concatenate([active_from_nodes, slack_nodes])
    tn_matrix = np.concatenate([active_to_nodes,
                                np.full(len(slack_nodes), len_nodes, dtype=np.int32)])

    adj_matrix = coo_matrix((np.ones(nobranch + len(slack_nodes)), (fn_matrix, tn_matrix)),
                            shape=(len_nodes + 1, len_nodes + 1))

    # check which nodes are reachable from the virtual heat slack node
    reachable_nodes = csgraph.breadth_first_order(adj_matrix, len_nodes, False, False)
    # throw out the virtual heat slack node
    reachable_nodes = reachable_nodes[reachable_nodes != len_nodes]

    nodes_connected = np.zeros(len(active_node_lookup), dtype=bool)
    nodes_connected[reachable_nodes] = True

    if not np.all(nodes_connected[active_from_nodes] == nodes_connected[active_to_nodes]):
        raise ValueError(
            "An error occured in the %s connectivity check. Please contact the pandapipes "
            "development team!" % mode)
    branches_connected = active_branch_lookup & nodes_connected[from_nodes]

    oos_nodes = np.where(~nodes_connected & active_node_lookup)[0]
    is_nodes = np.where(nodes_connected & ~active_node_lookup)[0]

    if len(oos_nodes) > 0:
        msg = "\n".join("In table %s: %s" % (tbl, nds) for tbl, nds in
                        get_table_index_list(net, node_pit, oos_nodes))
        logger.info("Setting the following nodes out of service for %s calculation in connectivity"
                    " check:\n%s" % (mode, msg))

    if len(is_nodes) > 0:
        node_type_message = "\n".join("In table %s: %s" % (tbl, nds) for tbl, nds in
                                      get_table_index_list(net, node_pit, is_nodes))
        if get_net_option(net, "quit_on_inconsistency_connectivity"):
            raise UserWarning(
                "The following nodes are connected to in_service branches in the %s calculation "
                "although being out of service, which leads to an inconsistency in the connectivity"
                " check!\n%s" % (mode, node_type_message))
        logger.info("Setting the following nodes back in service for %s calculation in connectivity"
                    " check as they are connected to in_service branches:\n%s"
                    % (mode, node_type_message))

    return nodes_connected, branches_connected


def get_table_index_list(net, pit_array, pit_indices, pit_type="node"):
    """
    Auxiliary function to get a list of tables and the table indices that belong to a number of pit
    indices.

    :param net: pandapipes net for which the list is requested
    :type net: pandapipesNet
    :param pit_array: Internal structure from which to derive the tables and table indices
    :type pit_array: np.array
    :param pit_indices: Indices for which the table name and index list are requested
    :type pit_indices: list, np.array, ....
    :param pit_type: Type of the pit ("node" or "branch")
    :type pit_type: str, default "node"
    :return: List of table names and table indices belonging to the pit indices
    """
    int_pit = pit_array[pit_indices, :]
    tables = np.unique(int_pit[:, TABLE_IDX_ND])
    table_lookup = get_lookup(net, pit_type, "table")
    return [(get_table_name(table_lookup, tbl), list(int_pit[int_pit[:, TABLE_IDX_ND] == tbl,
    ELEMENT_IDX_ND].astype(np.int32)))
            for tbl in tables]


def reduce_pit(net, mode="hydraulics"):
    """
    Create an internal ("active") pit with all nodes and branches that are actually in_service. This
    is also done for different lookups (e.g. the from_to indices for this pit and the node index
    lookup). A specialty that needs to be considered is that from_nodes and to_nodes change to new
    indices.

    :param net: The pandapipesNet for which the pit shall be reduced
    :type net: pandapipesNet
    :param node_pit: The internal structure node array
    :type node_pit: np.array
    :param branch_pit: The internal structure branch array
    :type branch_pit: np.array
    :param mode: the mode of the calculation (either "hydraulics" or "heat_transfer") for storing /\
        retrieving correct lookups
    :type mode: str, default "hydraulics"
    :return: No output
    """

    node_pit = net["_pit"]["node"]
    branch_pit = net["_pit"]["branch"]

    active_pit = dict()
    els = dict()
    reduced_node_lookup = None
    nodes_connected = get_lookup(net, "node", "active_" + mode)
    branches_connected = get_lookup(net, "branch", "active_" + mode)
    if np.all(nodes_connected):
        net["_lookups"]["node_from_to_active_" + mode] = copy.deepcopy(
            get_lookup(net, "node", "from_to"))
        net["_lookups"]["node_index_active_" + mode] = copy.deepcopy(
            get_lookup(net, "node", "index"))
        active_pit["node"] = np.copy(node_pit)
    else:
        active_pit["node"] = np.copy(node_pit[nodes_connected, :])
        reduced_node_lookup = np.cumsum(nodes_connected) - 1
        node_idx_lookup = get_lookup(net, "node", "index")
        net["_lookups"]["node_index_active_" + mode] = {
            tbl: reduced_node_lookup[idx_lookup[idx_lookup != -1]]
            for tbl, idx_lookup in node_idx_lookup.items()}
        els["node"] = nodes_connected
    if np.all(branches_connected):
        net["_lookups"]["branch_from_to_active_" + mode] = copy.deepcopy(
            get_lookup(net, "branch", "from_to"))
        active_pit["branch"] = np.copy(branch_pit)
        net["_lookups"]["branch_index_active_" + mode] = copy.deepcopy(
            get_lookup(net, "branch", "index"))
    else:
        active_pit["branch"] = np.copy(branch_pit[branches_connected, :])
        branch_idx_lookup = get_lookup(net, "branch", "index")
        if len(branch_idx_lookup):
            reduced_branch_lookup = np.cumsum(branches_connected) - 1
            net["_lookups"]["branch_index_active_" + mode] = {
                tbl: reduced_branch_lookup[idx_lookup[idx_lookup != -1]]
                for tbl, idx_lookup in branch_idx_lookup.items()}
        else:
            net["_lookups"]["branch_index_active_" + mode] = dict()
        els["branch"] = branches_connected
    if reduced_node_lookup is not None:
        active_pit["branch"][:, FROM_NODE] = reduced_node_lookup[
            branch_pit[branches_connected, FROM_NODE].astype(np.int32)]
        active_pit["branch"][:, TO_NODE] = reduced_node_lookup[
            branch_pit[branches_connected, TO_NODE].astype(np.int32)]
    net["_active_pit"] = active_pit

    for el, connected_els in els.items():
        ft_lookup = get_lookup(net, el, "from_to")
        aux_lookup = {table: (ft[0], ft[1], np.sum(connected_els[ft[0]: ft[1]]))
                      for table, ft in ft_lookup.items() if ft is not None}
        from_to_active_lookup = copy.deepcopy(ft_lookup)
        count = 0
        for table, (_, _, len_new) in sorted(aux_lookup.items(), key=lambda x: x[1][0]):
            from_to_active_lookup[table] = (count, count + len_new)
            count += len_new
        net["_lookups"]["%s_from_to_active_%s" % (el, mode)] = from_to_active_lookup
