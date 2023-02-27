# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import pandas as pd

from pandapipes.constants import NORMAL_PRESSURE, TEMP_GRADIENT_KPM, AVG_TEMPERATURE_K, \
    HEIGHT_EXPONENT
from pandapipes.idx_branch import LOAD_VEC_NODES, FROM_NODE, TO_NODE
from pandapipes.idx_node import EXT_GRID_OCCURENCE, EXT_GRID_OCCURENCE_T
from pandapipes.idx_node import PINIT, NODE_TYPE, P, TINIT, NODE_TYPE_T, T, LOAD
from pandapipes.pf.internals_toolbox import _sum_by_group
from pandapipes.pf.pipeflow_setup import get_net_option, get_lookup


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
        net[res_element] = pd.DataFrame(np.NAN, columns=output, index=net[element].index,
                                        dtype=np.float64)
    else:
        net[res_element] = pd.DataFrame(np.zeros(0, dtype=output), index=[])
        net[res_element] = pd.DataFrame(np.NaN, index=net[element].index,
                                        columns=net[res_element].columns)


def add_new_component(net, component, overwrite=False):
    """

    :param net:
    :type net:
    :param component:
    :type component:
    :param overwrite:
    :type overwrite:
    :return:
    :rtype:
    """
    name = component.table_name()
    if not overwrite and name in net:
        # logger.info('%s is already in net. Try overwrite if you want to get a new entry' %name)
        return
    else:
        if hasattr(component, 'geodata'):
            geodata = component.geodata()
        else:
            geodata = None

        comp_input = component.get_component_input()
        if name not in net:
            net['component_list'].append(component)
        net.update({name: comp_input})
        if isinstance(net[name], list):
            net[name] = pd.DataFrame(np.zeros(0, dtype=net[name]), index=[])
        # init_empty_results_table(net, name, component.get_result_table(net))

        if geodata is not None:
            net.update({name + '_geodata': geodata})
            if isinstance(net[name + '_geodata'], list):
                net[name + '_geodata'] = pd.DataFrame(np.zeros(0, dtype=net[name + '_geodata']),
                                                      index=[])


def set_entry_check_repeat(pit, column, entry, repeat_number, repeated=True):
    if repeated:
        pit[:, column] = np.repeat(entry, repeat_number)
    else:
        pit[:, column] = entry


def set_fixed_node_entries(net, node_pit, junctions, eg_types, p_values, t_values, node_comp,
                           mode="all"):
    junction_idx_lookups = get_lookup(net, "node", "index")[node_comp.table_name()]
    for eg_type in ("p", "t"):
        if eg_type not in mode and mode != "all":
            continue
        if eg_type == "p":
            val_col, type_col, eg_count_col, typ, valid_types, values = \
                PINIT, NODE_TYPE, EXT_GRID_OCCURENCE, P, ["p", "pt"], p_values
        else:
            val_col, type_col, eg_count_col, typ, valid_types, values = \
                TINIT, NODE_TYPE_T, EXT_GRID_OCCURENCE_T, T, ["t", "pt"], t_values
        mask = np.isin(eg_types, valid_types)
        if not np.any(mask):
            continue
        use_numba = get_net_option(net, "use_numba")
        juncts, press_sum, number = _sum_by_group(use_numba, junctions[mask], values[mask],
                                                  np.ones_like(values[mask], dtype=np.int32))
        index = junction_idx_lookups[juncts]
        node_pit[index, val_col] = (node_pit[index, val_col] * node_pit[index, eg_count_col]
                                    + press_sum) / (number + node_pit[index, eg_count_col])
        node_pit[index, type_col] = typ
        node_pit[index, eg_count_col] += number


def get_mass_flow_at_nodes(net, node_pit, branch_pit, eg_nodes, comp):
    node_uni, inverse_nodes, counts = np.unique(eg_nodes, return_counts=True, return_inverse=True)
    eg_from_branches = np.isin(branch_pit[:, FROM_NODE], node_uni)
    eg_to_branches = np.isin(branch_pit[:, TO_NODE], node_uni)
    from_nodes = branch_pit[eg_from_branches, FROM_NODE]
    to_nodes = branch_pit[eg_to_branches, TO_NODE]
    mass_flow_from = branch_pit[eg_from_branches, LOAD_VEC_NODES]
    mass_flow_to = branch_pit[eg_to_branches, LOAD_VEC_NODES]
    loads = node_pit[node_uni, LOAD]
    all_index_nodes = np.concatenate([from_nodes, to_nodes, node_uni])
    all_mass_flows = np.concatenate([-mass_flow_from, mass_flow_to, -loads])
    nodes, sum_mass_flows = _sum_by_group(get_net_option(net, "use_numba"), all_index_nodes,
                                          all_mass_flows)
    if not np.all(nodes == node_uni):
        raise UserWarning("In component %s: Something went wrong with the mass flow balance. "
                          "Please report this error at github." % comp.__name__)
    return sum_mass_flows, inverse_nodes, counts
