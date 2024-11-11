import numpy as np
from pandas import DataFrame

from pandapower.auxiliary import ppException

from pandapipes.constants import NORMAL_PRESSURE, TEMP_GRADIENT_KPM, AVG_TEMPERATURE_K, HEIGHT_EXPONENT
from pandapipes.idx_node import NODE_TYPE_T, T, INFEED, node_cols
from pandapipes.idx_branch import MDOTINIT, branch_cols, FROM_NODE_T_SWITCHED, TO_NODE, FROM_NODE

try:
    from numba import jit
    numba_installed = True
except ImportError:
    from pandapower.pf.no_numba import jit
    numba_installed = False

try:
    from pandaplan.core.pplog.logging import getLogger
except ImportError:
    from logging import getLogger

logger = getLogger(__name__)


def check_infeed_number(node_pit):
    slack_nodes = node_pit[:, NODE_TYPE_T] == T
    if len(node_pit) == 1:
        node_pit[slack_nodes, INFEED] = True
    infeed_nodes = node_pit[:, INFEED]
    if np.sum(infeed_nodes) != np.sum(slack_nodes):
        raise PipeflowNotConverged(r'The number of infeeding nodes and slacks do not match')


class PipeflowNotConverged(ppException):
    """
    Exception being raised in case pipeflow did not converge.
    """
    pass


def get_net_option(net, option_name):
    """
    Returns the requested option of the given net. Raises a UserWarning if the option was not found.

    :param net: pandapipesNet for which option is requested
    :type net: pandapipesNet
    :param option_name: Name of requested option
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
    :param option_names: Names of requested options (as args)
    :type option_names: str
    :return: option - Tuple with values of the options
    """
    return (get_net_option(net, option) for option in list(option_names))


def set_net_option(net, option_name, option_value):
    """
    Auxiliary function to set the value of a specific option (options are saved in a dict).

    :param net: pandapipesNet for which option shall be set
    :type net: pandapipesNet
    :param option_name: Name under which the option shall be saved
    :type option_name: str
    :param option_value: Value that shall be set for the given option
    :return: No output
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
    :param table_name: Name of the table that shall be mapped to number
    :type table_name: str
    :param table_number: Number under which the table is saved in the pit
    :type table_number: int
    :return: No output
    """
    table_lookup["n2t"][table_number] = table_name
    table_lookup["t2n"][table_name] = table_number


def get_table_number(table_lookup, table_name):
    """
    Auxiliary function to retrieve the internal pit number for a given pandapipes net table name \
    from the table lookup.

    :param table_lookup: The lookup dictionary from table names to internal number (n2t) and vice \
                versa (t2n)
    :type table_lookup: dict
    :param table_name: Name of the table for which the internal number shall be retrieved
    :type table_name: str
    :return: table_number - Internal number of the given table name within the pit
    :rtype: int
    """
    if table_name not in table_lookup["t2n"]:
        return None
    return table_lookup["t2n"][table_name]


def get_table_name(table_lookup, table_number):
    """
    Auxiliary function to retrieve the pandapipes net table name for a given internal pit number \
    from the table lookup.

    :param table_lookup: The lookup dictionary from table names to internal number (n2t) and vice \
                versa (t2n)
    :type table_lookup: dict
    :param table_number: Internal number of the table for which the name shall be retrieved
    :type table_number: int
    :return: table_name - pandapipes net table name for the internal pit number
    :rtype: str

    """
    if table_number not in table_lookup["n2t"]:
        return None
    return table_lookup["n2t"][table_number]


def get_lookup(net, pit_type="node", lookup_type="index"):
    """
    Returns internal lookups which are mostly defined in the function `create_lookups`.

    :param net: The pandapipes net for which the lookup is requested
    :type net: pandapipesNet
    :param pit_type: Identifier which of the two pits ("branch" or "node") the lookup belongs to
    :type pit_type: str
    :param lookup_type: Name of the lookup type
    :type lookup_type: str
    :return: lookup - A lookup (mostly a dict with mappings from pandapipesNet to internal
            structure)
    :rtype: dict, np.array, ....

    """
    pit_type = pit_type.lower()
    lookup_type = lookup_type.lower()
    all_lookup_types = ["index", "table", "from_to", "active_hydraulics", "active_heat_transfer",
                        "length", "from_to_active_hydraulics", "from_to_active_heat_transfer",
                        "index_active_hydraulics", "index_active_heat_transfer"]
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
    This function sets the "user_pf_options" dictionary for net. These options overrule
    net._internal_options once they are added to net. These options are used in configuration of
    load flow calculation.
    At the same time, user-defined arguments for `pandapipes.pipeflow()` always have a higher
    priority. To remove user_pf_options, set "reset = True" and provide no additional arguments.

    :param net: pandapipes network for which to create user options
    :type net: pandapipesNet
    :param reset: Specifies whether the user_pf_options is removed before setting new options
    :type reset: bool, default False
    :param kwargs: pipeflow options that shall be set, e.g. tol_m = 1e-7
    :return: No output
    """
    if reset or 'user_pf_options' not in net.keys():
        net['user_pf_options'] = dict()

    additional_kwargs = set(kwargs.keys()) - set(default_options.keys()) - {"fluid", "hyd_flag"}
    if len(additional_kwargs) > 0:
        logger.info('parameters %s are not in the list of standard options'
                    % list(additional_kwargs))

    net.user_pf_options.update(kwargs)


def create_internal_results(net):
    """
    Initializes a dictionary that shall contain some internal results later.

    :param net: pandapipes net to which internal result dict will be added
    :type net: pandapipesNet
    :return: No output
    """
    net["_internal_results"] = dict()


def write_internal_results(net, **kwargs):
    """
    Adds specified values to the internal result dictionary of the given pandapipes net. If internal
    results are not yet defined for the net, they are created as well.

    :param net: pandapipes net for which to update internal result dict
    :type net: pandapipesNet
    :param kwargs: Additional keyword arguments with the internal result values
    :return: No output

    """
    if "_internal_results" not in net:
        create_internal_results(net)
    net["_internal_results"].update(kwargs)


def create_empty_pit(net):
    """
    Creates an empty internal structure which is called pit (pandapipes internal tables). The\
    structure is a dictionary which should contain one array for all nodes and one array for all\
    branches of the net. It is very often referred to within the pipeflow. So the structure in\
    general looks like this:

    >>> net["_pit"] = {"node": np.array((no_nodes, col_nodes), dtype=np.float64),
    >>>                "branch": np.array((no_branches, col_branches), dtype=np.float64)}

    :param net: The pandapipes net to which to add the empty structure
    :type net: pandapipesNet
    :return: pit - The dict of arrays with the internal node / branch structure
    :rtype: dict

    """
    node_length = get_lookup(net, "node", "length")
    branch_length = get_lookup(net, "branch", "length")
    # init empty pit
    pit = {"node": np.empty((node_length, node_cols), dtype=np.float64),
           "branch": np.empty((branch_length, branch_cols), dtype=np.float64),
           "components": {}}
    net["_pit"] = pit
    return pit


def branches_connected_flow(branch_pit):
    """
    Simple function to identify branches with flow based on the calculated velocity.

    :param branch_pit: The pandapipes internal table of the network (including hydraulics results)
    :type branch_pit: np.array
    :return: branches_connected_flow - lookup array if branch is connected wrt. flow
    :rtype: np.array
    """
    # TODO: is this formulation correct or could there be any caveats?
    return ~np.isnan(branch_pit[:, MDOTINIT]) \
        & ~np.isclose(branch_pit[:, MDOTINIT], 0, rtol=1e-10, atol=1e-10)


def _sum_by_group_np(indices, *values):
    """
    Auxiliary function to sum up values by some given indices (both as numpy arrays).

    :param indices:
    :type indices:
    :param values:
    :type values:
    :return:
    :rtype:
    """

    # sort indices and values by indices
    order = np.argsort(indices)
    indices = indices[order]
    val = list(values)
    for i, _ in enumerate(val):
        val[i] = val[i][order]

    return _sum_by_group_sorted(indices, *val)


def _sum_by_group(use_numba, indices, *values):
    """
    Auxiliary function to sum up values by some given indices (both as numpy arrays).

    :param use_numba:
    :type use_numba:
    :param indices:
    :type indices:
    :param values:
    :type values:
    :return:
    :rtype:
    """
    if not use_numba:
        return _sum_by_group_np(indices, *values)
    elif not numba_installed:
        logger.info("The numba import did not work out, it will not be used.")
        return _sum_by_group_np(indices, *values)
    if len(indices) == 0:
        return tuple([indices] + list(values))
    # idea: shift this into numba function and raise ValueError if condition not accepted,
    # has not yet worked...
    ind_dt = indices.dtype
    indices = indices.astype(np.int32)
    max_ind = max_nb(indices)
    if (max_ind < 1e5 or max_ind < 2 * len(indices)) and max_ind < 10 * len(indices):
        dtypes = [v.dtype for v in values]
        val_arr = np.array(list(values), dtype=np.float64).transpose()
        new_ind, new_arr = _sum_values_by_index(indices, val_arr, max_ind, len(indices),
                                                len(values))
        return tuple([new_ind.astype(ind_dt)]
                     + [new_arr[:, i].astype(dtypes[i]) for i in range(len(values))])
    return _sum_by_group_np(indices, *values)


def select_from_pit(table_index_array, input_array, data):
    """
        Auxiliary function to retrieve values from a table like a pit. Each data entry corresponds
        to a table_index_array entry. Example: velocities are indexed by the corresponding
        from_nodes stored in the pipe pit.

        The function inputs another array which consists of some table_index_array entries the user
        wants to retrieve. The function is used in pandapipes results evaluation. The input array is
        the list of from_junction entries, corresponding only to the junction elements, not
        containing additional pipe nodes. The table_index_array is the complete list of from_nodes
        consisting of junction element entries and additional pipe section nodes. Data corresponds
        to the gas velocities.

        :param table_index_array:
        :type table_index_array:
        :param input_array:
        :type input_array:
        :param data:
        :type data:
        :return:
        :rtype:
        """
    sorter = np.argsort(table_index_array)
    indices = sorter[np.searchsorted(table_index_array, input_array, sorter=sorter)]

    return data[indices]


@jit(nopython=True)
def _sum_values_by_index(indices, value_arr, max_ind, le, n_vals):
    ind1 = indices + 1
    new_indices = np.zeros(max_ind + 2, dtype=np.int32)
    summed_values = np.zeros((max_ind + 2, n_vals), dtype=np.float64)
    for i in range(le):
        new_indices[int(ind1[i])] = ind1[i]
        for j in range(n_vals):
            summed_values[int(ind1[i]), j] += value_arr[i, j]
    summed_values = summed_values[new_indices > 0]
    new_indices = new_indices[new_indices > 0] - 1
    return new_indices, summed_values


@jit(nopython=True)
def max_nb(arr):
    return np.max(arr)


def get_from_nodes_corrected(branch_pit, switch_from_to_col=None):
    """
    Function to get corrected from nodes from the branch pit.

    Usually, this should be used if the velocity in a branch is negative, so that the\
    flow goes from the to_node to the from_node. The parameter switch_from_to_col indicates\
    whether the two columns shall be switched (for each row) or not.

    :param branch_pit: The branch pit
    :type branch_pit: np.ndarray
    :param switch_from_to_col: Indicates for each branch, whether to use the from (True) or \
        to (False) node. If None, the column FROM_NODE_T_SWITCHED is used.
    :type switch_from_to_col: np.ndarray, default None
    :return:
    :rtype:
    """
    if switch_from_to_col is None:
        switch_from_to_col = branch_pit[:, FROM_NODE_T_SWITCHED]
    from_node_col = switch_from_to_col.astype(np.int32) * (TO_NODE - FROM_NODE) + FROM_NODE
    return branch_pit[np.arange(len(branch_pit)), from_node_col].astype(np.int32)


def get_to_nodes_corrected(branch_pit, switch_from_to_col=None):
    """
    Function to get corrected to nodes from the branch pit.

    Usually, this should be used if the velocity in a branch is negative, so that the\
    flow goes from the to_node to the from_node. The parameter switch_from_to_col indicates\
    whether the two columns shall be switched (for each row) or not.

    :param branch_pit: The branch pit
    :type branch_pit: np.ndarray
    :param switch_from_to_col: Indicates for each branch, whether to use the from (False) or \
        to (True) node. If set to None, the column FROM_NODE_T_SWITCHED is used.
    :type switch_from_to_col: np.ndarray, default None
    :return:
    :rtype:
    """
    if switch_from_to_col is None:
        switch_from_to_col = branch_pit[:, FROM_NODE_T_SWITCHED]
    to_node_col = switch_from_to_col.astype(np.int32) * (FROM_NODE - TO_NODE) + TO_NODE
    return branch_pit[np.arange(len(branch_pit)), to_node_col].astype(np.int32)


def _sum_by_group_sorted(indices, *values):
    """Auxiliary function to sum up values by some given indices (both as numpy arrays). Expects the
    indices and values to already be sorted.

    :param indices:
    :type indices:
    :param values:
    :type values:
    :return:
    :rtype:
    """
    # Index defines whether a specific index has already appeared in the index array before.
    index = np.ones(len(indices), 'bool')
    index[:-1] = indices[1:] != indices[:-1]

    # make indices unique for output
    indices = indices[index]

    val = list(values)
    for i, _ in enumerate(val):
        # sum up values, chose only those with unique indices and then subtract the previous sums
        # --> this way for each index the sum of all values belonging to this index is returned
        nans = np.isnan(val[i])
        if np.any(nans):
            np.nan_to_num(val[i], copy=False)
            np.cumsum(val[i], out=val[i])
            val[i] = val[i][index]
            still_na = nans[index]
            val[i][1:] = val[i][1:] - val[i][:-1]
            val[i][still_na] = np.nan
        else:
            np.cumsum(val[i], out=val[i])
            val[i] = val[i][index]
            val[i][1:] = val[i][1:] - val[i][:-1]
    return [indices] + val


def p_correction_height_air(height):
    """

    :param height:
    :type height:
    :return:
    :rtype:
    """
    return NORMAL_PRESSURE * np.power(1 - height * TEMP_GRADIENT_KPM / AVG_TEMPERATURE_K,
                                      HEIGHT_EXPONENT)


def vinterp(min_vals, max_vals, lengths):
    """

    :param min_vals:
    :type min_vals:
    :param max_vals:
    :type max_vals:
    :param lengths: lengths for each range (same length as starts)
    :type lengths: numpy.array
    :return:
    :rtype:
    """
    intervals = (max_vals - min_vals) / (lengths + 1)
    steps = np.repeat(intervals, lengths)
    counter = np.arange(lengths.sum()) - np.repeat(lengths.cumsum() - lengths, lengths) + 1
    return np.repeat(min_vals, lengths) + steps * counter


def vrange(starts, lengths):
    """
    Create concatenated ranges of integers for multiple start/length

    :param starts: starts for each range
    :type starts: numpy.array
    :param lengths: lengths for each range (same length as starts)
    :type lengths: numpy.array
    :return: cat_range - concatenated ranges
    :rtype: numpy.array

    :Example:
    >>> starts = np.array([1, 3, 4, 6])
    >>> lengths = np.array([0, 2, 3, 0])
    >>> print vrange(starts, lengths)
    """
    # Repeat start position index length times and concatenate
    starting_array = np.repeat(starts, lengths)
    # Create group counter that resets for each start/length
    length_ranges = np.arange(lengths.sum()) - np.repeat(lengths.cumsum() - lengths, lengths)
    # Add group counter to group specific starts
    return starting_array + length_ranges


def init_results_element(net, element, output, all_float):
    """

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param element:
    :type element:
    :param output:
    :type output:
    :param all_float:
    :type all_float:
    :return: No Output.
    """
    res_element = "res_" + element
    if all_float:
        net[res_element] = DataFrame(np.nan, columns=output, index=net[element].index,
                                        dtype=np.float64)
    else:
        net[res_element] = DataFrame(np.zeros(0, dtype=output), index=[])
        net[res_element] = DataFrame(np.nan, index=net[element].index,
                                        columns=net[res_element].columns)


def set_entry_check_repeat(pit, column, entry, repeat_number, repeated=True):
    if repeated:
        pit[:, column] = np.repeat(entry, repeat_number)
    else:
        pit[:, column] = entry
