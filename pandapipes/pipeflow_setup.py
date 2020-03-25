# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import copy
import inspect
import numpy as np

from pandapipes.idx_branch import FROM_NODE, TO_NODE, FROM_NODE_T, TO_NODE_T, VINIT, branch_cols, \
    ACTIVE as ACTIVE_BR
from pandapipes.idx_node import NODE_TYPE, P, PINIT, NODE_TYPE_T, T, node_cols, \
    ACTIVE as ACTIVE_ND, TABLE_IDX as TABLE_IDX_ND, ELEMENT_IDX as ELEMENT_IDX_ND
from pandapipes.properties.fluids import get_fluid
from scipy.sparse import coo_matrix, csgraph

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

default_options = {"friction_model": "nikuradse", "converged": False, "tol_p": 1e-4, "tol_v": 1e-4,
                   "tol_T": 1e-3, "tol_res": 1e-3, "iter": 10, "error_flag": False, "alpha": 1,
                   "nonlinear_method": "constant", "p_scale": 1, "mode": "hydraulics",
                   "ambient_temperature": 293, "check_connectivity": True,
                   "only_update_hydraulic_matrix": False,
                   "quit_on_inconsistency_connectivity": False}


def get_net_option(net, option_name):
    """
    Returns the requested option of the given net. Raises a UserWarning if the option was not found.

    :param net: pandapipesNet for which option is requested
    :type net: pandapipesNet
    :param option_name: name of requested option
    :type option_name: str
    :return: option - The value of the option
    """
    try:
        return net["_options"][option_name]
    except KeyError:
        raise UserWarning("The option %s is not stored in the pandapipes net." % option_name)


def get_net_options(net, *option_names):
    """
    Returns several requested options of the given net. Raises a UserWarning if any of the options
    was not found.

    :param net: pandapipesNet for which option is requested
    :type net: pandapipesNet
    :param option_names: names of requested options (as args)
    :type option_names: str
    :return: option - Tuple with values of the options
    """
    return (get_net_option(net, option) for option in list(option_names))


def set_net_option(net, option_name, option_value):
    """
    Auxiliary function to set the value of a specific option (options are saved in a dict)

    :param net: pandapipesNet for which option shall be set
    :type net: pandapipesNet
    :param option_name: name under which the option shall be saved
    :type option_name: str
    :param option_value: value that shall be set for the given option
    :return: No output.
    """
    net["_options"][option_name] = option_value


def warn_high_index(element_name, element_length, max_element_index):
    if (element_length > 100 and max_element_index > 1000 * element_length) \
            or (element_length <= 100 and max_element_index > 50000):
        logger.warning("High index in %s table!!!" % element_name)


def add_table_lookup(table_lookup, table_name, table_number):
    """
    Auxiliary function to add a lookup between table name in the pandapipes net and table number in
    the internal structure (pit).

    :param table_lookup: The lookup dictionary from table names to internal number (n2t) and vice \
                versa (t2n)
    :type table_lookup: dict
    :param table_name: name of the table that shall be mapped to number
    :type table_name: str
    :param table_number: number under which the table is saved in the pit
    :type table_number: int
    :return: No ouput.
    """
    table_lookup["n2t"][table_number] = table_name
    table_lookup["t2n"][table_name] = table_number


def get_table_number(table_lookup, table_name):
    """
    Auxiliary function to retrieve the internal pit number for a given pandapipesNet table name from
    the table lookup.

    :param table_lookup: The lookup dictionary from table names to internal number (n2t) and vice \
                versa (t2n)
    :type table_lookup: dict
    :param table_name: name of the table for which the internal number shall be retrieved
    :type table_name: str
    :return: table_number - internal number of the given table name within the pit
    :rtype: int
    """
    if table_name not in table_lookup["t2n"]:
        return None
    return table_lookup["t2n"][table_name]


def get_table_name(table_lookup, table_number):
    """
    Auxiliary function to retrieve the pandapipesNet table name for a given internal pit number from
    the table lookup.

    :param table_lookup: The lookup dictionary from table names to internal number (n2t) and vice \
                versa (t2n)
    :type table_lookup: dict
    :param table_number: internal number of the table for which the name shall be retrieved
    :type table_number: int
    :return: table_name - pandapipesNet table name for the internal pit number
    :rtype: str
    """
    if table_number not in table_lookup["n2t"]:
        return None
    return table_lookup["n2t"][table_number]


def get_lookup(net, pit_type="node", lookup_type="index"):
    """
    Returns internal lookups which are mostly defined in the function `create_lookups`.

    :param net: The pandapipesNet for which the lookup is requested
    :type net: pandapipesNet
    :param pit_type: identifier which of the two pits ("branch" or "node") the lookup belongs to
    :type pit_type: str
    :param lookup_type: name of the lookup type
    :type lookup_type: str
    :return: lookup - A lookup (mostly a dict with mappings from pandapipesNet to internal
            structure)
    :rtype: dict, np.array, ...
    """
    pit_type = pit_type.lower()
    lookup_type = lookup_type.lower()
    all_lookup_types = ["index", "table", "from_to", "active", "length", "from_to_active",
                        "index_active"]
    if lookup_type not in all_lookup_types:
        type_names = "', '".join(all_lookup_types)
        logger.error("No lookup type '%s' exists. Please choose one of '%s'."
                     % (lookup_type, type_names))
        return None
    if pit_type not in ["node", "branch"]:
        logger.error("No pit type '%s' exists. Please choose one of 'node' and 'branch'."
                     % pit_type)
        return None
    return net["_lookups"]["%s_%s" % (pit_type, lookup_type)]


def set_user_pf_options(net, reset=False, **kwargs):
    """
    This function sets the 'user_pf_options' dict for net. These options overrule
    net.__internal_options once they are added to net. These options are used in configuration of
    load flow calculation.
    At the same time, user-defined arguments for pandapower.runpp() always have a higher priority.
    To remove user_pf_options, set overwrite=True and provide no additional arguments

    :param net: pandapipes network for which to create user options
    :type net: pandapipesNet
    :param reset: specifies whether the user_pf_options is removed before setting new options
    :type reset: bool, default False
    :param kwargs: pipeflow options that shall be set, e. g. tol_v = 1e-7
    :return: No output
    """
    if reset or 'user_pf_options' not in net.keys():
        net['user_pf_options'] = dict()

    additional_kwargs = set(kwargs.keys()) - set(default_options.keys()) - {"fluid", "hyd_flag"}
    if len(additional_kwargs) > 0:
        logger.info('parameters %s are not in the list of standard options'
                    % list(additional_kwargs))

    net.user_pf_options.update(kwargs)


def init_options(net, local_parameters):
    """
    Initializes physical and mathematical constants included in Pandapipes. In addition, options
    for the nonlinear and time-dependent solver are also set.

    Those are the options that can be set and their default values:

        - **iter** (int): 10 - If the simulation is terminated after a certain amount of iterations\
                               , this is the number of iterations.

        - **tol_p** (float): 1e-4 - The relative tolerance for the pressure. A result is accepted \
                                    if the relative error is smaller than this factor.

        - **tol_v** (float): 1e-4 - The relative tolerance for the velocity. A result is accepted \
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
                parameter

        - **nonlinear_method** (str): "constant" - The option of how the damping factor **alpha** \
                is determined in each iteration. It can be "constant" (i.e. **alpha** is always the\
                 same in each iteration) or "automatic", in which case **alpha** is adapted \
                 automatically with respect to the convergence behaviour.

        - **gas_impl** (str): "pandapipes" - Implementation of the gas model. It can be set to\
                "pandapipes" with calculations according to  "Handbuch der Gasversorgungstechnik"\
                 or to "stanet" with calculations according to the stanet reference.

        - **heat_transfer** (bool): False - Flag to determine if the heat transfer shall be\
                calculated.

        - **only_update_hydraulic_matrix** (bool): False - If true, the system matrix is not \
                created in every iteration, but only the data isupdated according to a lookup that\
                is identified in the first iteration. This speeds up calculation, but has not yet\
                been tested extensively.

        - **check_connectivity** (bool): True - If true, a connectivity check is performed at the\
                beginning of the pipeflow and parts of the net that are not connected to external\
                grids are set inactive.

        - **quit_on_inconsistency_connectivity** (bool): False - If true, inconsistencies in the\
                connectivity check raise an error, otherwise they are handled. Inconsistencies mean\
                that out of service nodes are connected to in service branches. If that is the case\
                and the flag is set to False, the connected nodes are activated.

    :param net: The pandapipesNet for which the options are initialized
    :type net: pandapipesNet
    :param local_parameters: dictionary with local parameters that were passed to the pipeflow call.
    :type local_parameters: dict
    :return: No output

    EXAMPLE:
        init_constants(net)

    """
    from pandapipes.pipeflow import pipeflow

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
        net["_options"].update(net.user_pf_options)

    # the last layer is the layer of passeed parameters by the user, it is defined as the local
    # existing parameters during the pipeflow call which diverges from the default parameters of the
    # function definition in the second layer
    params = dict()
    for k, v in local_parameters.items():
        if k in excluded_params or (k in pf_func_options and pf_func_options[k] == v):
            continue
        params[k] = v
    params.update(local_parameters["kwargs"])
    net["_options"].update(params)
    net["_options"]["fluid"] = get_fluid(net).name


def create_internal_results(net):
    """
    Initializes a dictionary that shall contain some internal results later.

    :param net: pandapipesNet to which internal result dict will be added
    :type net: pandapipesNet
    :return: No output.
    """
    net["_internal_results"] = dict()


def write_internal_results(net, **kwargs):
    """
    Adds specified values to the internal result dictionary of the given pandapipesNet. If internal
    results are not yet defined for the net, they are created as well.

    :param net: pandapipesNet for which to update internal result dict
    :type net: pandapipesNet
    :param kwargs: additional keyword arguments with the internal result values
    :return: No output.
    """
    if "_internal_results" not in net:
        create_internal_results(net)
    net["_internal_results"].update(kwargs)


def initialize_pit(net, node_name, NodeComponent, NodeElementComponent, BranchComponent,
                   BranchWInternalsComponent):
    """
    Initializes and fills the internal structure which is called pit (pandapipes internal tables).
    The structure is a dictionary which should contain one array for all nodes and one array for all
    branches of the net (c.f. also `create_empty_pit`).

    :param net: The pandapipesNet for which to create and fill the internal structure
    :type net: pandapipesNet
    :return: (node_pit, branch_pit) - the two internal structure arrays
    :rtype: tuple(np.array)
    """
    pit = create_empty_pit(net)

    for comp in net['component_list']:
        if issubclass(comp, NodeComponent) | \
                issubclass(comp, BranchWInternalsComponent) | \
                issubclass(comp, NodeElementComponent):
            comp.create_pit_node_entries(net, pit["node"], node_name)
        if issubclass(comp, BranchComponent):
            comp.create_pit_branch_entries(net, pit["branch"], node_name)
    return pit["node"], pit["branch"]


def create_empty_pit(net):
    """
    Creates an empty internal structure which is called pit (pandapipes internal tables). The\
    structure is a dictionary which should contain one array for all nodes and one array for all\
    branches of the net. It is very often referred to within the pipeflow. So the structure in\
    general looks like this:

    net["_pit"] = {"node": np.array((no_nodes, col_nodes), dtype=np.float64),\
                   "branch": np.array((no_branches, col_branches), dtype=np.float64)}

    :param net: The pandapipesNet to which to add the empty structure
    :type net: pandapipesNet
    :return: pit - The dict of arrays with the internal node / branch structure
    :rtype: dict
    """
    node_length = get_lookup(net, "node", "length")
    branch_length = get_lookup(net, "branch", "length")
    # init empty pit
    pit = {"node": np.empty((node_length, node_cols), dtype=np.float64),
           "branch": np.empty((branch_length, branch_cols), dtype=np.float64)}
    net["_pit"] = pit
    return pit


def extract_all_results(net, node_name):
    """
    Extract results from branch pit and node pit and write them to the different tables of the net,
    as defined by the component models.

    :param net: pandapipesNet for which to extract results into net.res_xy
    :type net: pandapipesNet
    :return: No output.
    """
    for comp in net['component_list']:
        comp.extract_results(net, net["_options"], node_name)


def create_lookups(net, NodeComponent, BranchComponent, BranchWInternalsComponent):
    """
    Create all lookups necessary for the pipeflow of the given net.
    The lookups are usually:

      - node_from_to: the start and end indices of all node component tables within the pit
      - branch_from_to: the start and end indices of all branch component tables within the pit
      - node_table: dict to determine indices for node component tables (e.g. {"junction": 0}).\
                    Can be arbitrary and strongly depends on the component order given by\
                    `get_component_list`.
      - branch_table: dict to determine indices for branch component tables (e.g.\
                      {"pipe": 0, "valve": 1}). Can be arbitrary and strongly depends on the\
                      component order given by `get_component_list`.
      - node_index: Lookup from component index (e.g. junction 2) to pit index (e.g. 0) for nodes.
      - branch_index: Lookup from component index (e.g. pipe 1) to pit index (e.g. 5) for branches.
      - internal_nodes_lookup: Lookup for internal nodes of branch components that makes result\
                               extraction a lot easier.

    :param net: The pandapipesNet for which to create the lookups
    :type net: pandapipesNet
    :return: No output.

    """
    node_ft_lookups, node_idx_lookups, node_from, node_table_nr = dict(), dict(), 0, 0
    branch_ft_lookups, branch_idx_lookups, branch_from, branch_table_nr = dict(), dict(), 0, 0
    branch_table_lookups = {"t2n": dict(), "n2t": dict()}
    node_table_lookups = {"t2n": dict(), "n2t": dict()}
    internal_nodes_lookup = dict()

    for comp in net['component_list']:
        if issubclass(comp, BranchComponent):
            branch_from, branch_table_nr = comp.create_branch_lookups(
                net, branch_ft_lookups, branch_table_lookups, branch_idx_lookups, branch_table_nr,
                branch_from)
        if issubclass(comp, NodeComponent) | issubclass(comp, BranchWInternalsComponent):
            node_from, node_table_nr = comp.create_node_lookups(
                net, node_ft_lookups, node_table_lookups, node_idx_lookups, node_from,
                node_table_nr, internal_nodes_lookup)

    net["_lookups"] = {"node_from_to": node_ft_lookups, "branch_from_to": branch_ft_lookups,
                       "node_table": node_table_lookups, "branch_table": branch_table_lookups,
                       "node_index": node_idx_lookups, "branch_index": branch_idx_lookups,
                       "node_length": node_from, "branch_length": branch_from,
                       "internal_nodes_lookup": internal_nodes_lookup}


def check_connectivity(net, branch_pit, node_pit, check_heat):
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
    :param branch_pit: internal array with branch entries
    :type branch_pit: np.array
    :param node_pit: internal array with node entries
    :type node_pit: np.array
    :param check_heat: flag which determines whether to also check for connectivity to heat \
        external grids
    :type check_heat: bool
    :return: (nodes_connected_hyd, branches_connected) - lookups of np.arrays stating which of the
            internal nodes and branches are reachable from any of the hyd_slacks (np mask).
    :rtype: tuple(np.array)
    """
    active_branch_lookup = branch_pit[:, ACTIVE_BR].astype(np.bool)
    active_node_lookup = node_pit[:, ACTIVE_ND].astype(np.bool)
    from_nodes = branch_pit[:, FROM_NODE].astype(np.int32)
    to_nodes = branch_pit[:, TO_NODE].astype(np.int32)
    hyd_slacks = np.where(node_pit[:, NODE_TYPE] == P & node_pit[:, ACTIVE_ND].astype(np.bool))[0]

    nodes_connected, branches_connected = perform_connectivity_search(
        net, node_pit, hyd_slacks, from_nodes, to_nodes, active_node_lookup, active_branch_lookup,
        mode="hydraulics")
    if not check_heat:
        return nodes_connected, branches_connected

    heat_slacks = np.where((node_pit[:, NODE_TYPE_T] == T) & nodes_connected)[0]
    if len(heat_slacks) == len(hyd_slacks) and np.all(heat_slacks == hyd_slacks):
        return nodes_connected, branches_connected

    nodes_connected, branches_connected = perform_connectivity_search(
        net, node_pit, heat_slacks, from_nodes, to_nodes, nodes_connected, branches_connected,
        mode="heat transfer")
    return nodes_connected, branches_connected


def perform_connectivity_search(net, node_pit, slack_nodes, from_nodes, to_nodes,
                                active_node_lookup, active_branch_lookup, mode="hydraulics"):
    len_nodes = len(node_pit)
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

    nodes_connected = np.zeros(len(active_node_lookup), dtype=np.bool)
    nodes_connected[reachable_nodes] = True

    if not np.all(nodes_connected[active_from_nodes] == nodes_connected[active_to_nodes]):
        raise ValueError(
            "An error occured in the %s connectivity check. Please contact the pandapipes development" \
            " team!" % mode)
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

    :param net: pandapipseNet for which the list is requested
    :type net: pandapipesNet
    :param pit_array: internal structure from which to derive the tables and table indices
    :type pit_array: np.array
    :param pit_indices: indices for which the table name and index list are requested
    :type pit_indices: list, np.array, ...
    :param pit_type: type of the pit ("node" or "branch")
    :type pit_type: str, default "node"
    :return: list of table names and table indices belonging to the pit indices
    """
    int_pit = pit_array[pit_indices, :]
    tables = np.unique(int_pit[:, TABLE_IDX_ND])
    table_lookup = get_lookup(net, pit_type, "table")
    return [(get_table_name(table_lookup, tbl), list(int_pit[int_pit[:, TABLE_IDX_ND] == tbl,
                                                             ELEMENT_IDX_ND].astype(np.int32)))
            for tbl in tables]


def reduce_pit(net, node_pit, branch_pit, nodes_connected, branches_connected):
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
    :param nodes_connected: A mask array stating which nodes are actually connected to the rest of\
            the net
    :type nodes_connected: np.array
    :param branches_connected: A mask array stating which branches are actually connected to the \
             rest of the net
    :type branches_connected: np.array
    :return: No output
    """
    active_pit = dict()
    els = dict()
    reduced_node_lookup = None
    if np.alltrue(nodes_connected):
        net["_lookups"]["node_from_to_active"] = copy.deepcopy(get_lookup(net, "node", "from_to"))
        net["_lookups"]["node_index_active"] = copy.deepcopy(get_lookup(net, "node", "index"))
        active_pit["node"] = np.copy(node_pit)
    else:
        active_pit["node"] = np.copy(node_pit[nodes_connected, :])
        reduced_node_lookup = np.cumsum(nodes_connected) - 1
        node_idx_lookup = get_lookup(net, "node", "index")
        net["_lookups"]["node_index_active"] = {
            tbl: reduced_node_lookup[idx_lookup[idx_lookup != -1]]
            for tbl, idx_lookup in node_idx_lookup.items()}
        els["node"] = nodes_connected
    if np.alltrue(branches_connected):
        net["_lookups"]["branch_from_to_active"] = copy.deepcopy(get_lookup(net, "branch",
                                                                            "from_to"))
        active_pit["branch"] = np.copy(branch_pit)
        net["_lookups"]["branch_index_active"] = copy.deepcopy(get_lookup(net, "branch", "index"))
    else:
        active_pit["branch"] = np.copy(branch_pit[branches_connected, :])
        if reduced_node_lookup is not None:
            active_pit["branch"][:, FROM_NODE] = reduced_node_lookup[
                branch_pit[branches_connected, FROM_NODE].astype(np.int32)]
            active_pit["branch"][:, TO_NODE] = reduced_node_lookup[
                branch_pit[branches_connected, TO_NODE].astype(np.int32)]
        branch_idx_lookup = get_lookup(net, "branch", "index")
        if len(branch_idx_lookup):
            reduced_branch_lookup = np.cumsum(branches_connected) - 1
            net["_lookups"]["branch_index_active"] = {
                tbl: reduced_branch_lookup[idx_lookup[idx_lookup != -1]]
                for tbl, idx_lookup in branch_idx_lookup.items()}
        else:
            net["_lookups"]["branch_index_active"] = dict()
        els["branch"] = branches_connected
    net["_active_pit"] = active_pit
    net["_lookups"]["node_active"] = nodes_connected
    net["_lookups"]["branch_active"] = branches_connected

    for el, connected_els in els.items():
        ft_lookup = get_lookup(net, el, "from_to")
        aux_lookup = {table: (ft[0], ft[1], np.sum(connected_els[ft[0]: ft[1]]))
                      for table, ft in ft_lookup.items() if ft is not None}
        from_to_active_lookup = copy.deepcopy(ft_lookup)
        count = 0
        for table, (f_old, t_old, len_new) in sorted(aux_lookup.items(), key=lambda x: x[1][0]):
            from_to_active_lookup[table] = (count, count + len_new)
            count += len_new
        net["_lookups"]["%s_from_to_active" % el] = from_to_active_lookup


def extract_results_active_pit(net, node_pit, branch_pit, nodes_connected, branches_connected):
    """
    Extract the pipeflow results from the internal pit structure ("_active_pit") to the general pit
    structure.

    :param net: The pandapipesNet that the internal structure belongs to
    :type net: pandapipesNet
    :param node_pit: The internal structure node array
    :type node_pit: np.array
    :param branch_pit: The internal structure branch array
    :type branch_pit: np.array
    :param nodes_connected: A mask array stating which nodes are actually connected to the rest of\
            the net
    :type nodes_connected: np.array
    :param branches_connected: A mask array stating which branches are actually connected to the \
             rest of the net
    :type branches_connected: np.array
    :return: No output.
    """
    if not np.alltrue(nodes_connected):
        node_pit[~nodes_connected, PINIT] = np.NaN
        node_pit[nodes_connected, :] = net["_active_pit"]["node"]
    else:
        net["_pit"]["node"] = np.copy(net["_active_pit"]["node"])
    if not np.alltrue(branches_connected):
        branch_pit[~branches_connected, VINIT] = np.NaN
        rows_active_br = np.where(branches_connected)[0]
        cols_br = np.array([i for i in range(branch_pit.shape[1])
                            if i not in [FROM_NODE, TO_NODE, FROM_NODE_T, TO_NODE_T]])
        branch_pit[rows_active_br[:, np.newaxis], cols_br[np.newaxis, :]] = \
            net["_active_pit"]["branch"][:, cols_br]
    else:
        net["_pit"]["branch"] = np.copy(net["_active_pit"]["branch"])
