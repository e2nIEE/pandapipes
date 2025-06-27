# Copyright (c) 2020-2025 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import pandas as pd

from pandapipes import get_fluid
from pandapipes.constants import NORMAL_PRESSURE, TEMP_GRADIENT_KPM, AVG_TEMPERATURE_K, \
    HEIGHT_EXPONENT
from pandapipes.idx_branch import LOAD_VEC_NODES_FROM, LOAD_VEC_NODES_TO, FROM_NODE, TO_NODE
from pandapipes.idx_node import (EXT_GRID_OCCURENCE, EXT_GRID_OCCURENCE_T,
                                 PINIT, NODE_TYPE, P, TINIT, NODE_TYPE_T, T, LOAD)
from pandapipes.pf.pipeflow_setup import get_net_option, get_lookup
from pandapipes.pf.internals_toolbox import _sum_by_group


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
        net[res_element] = pd.DataFrame(np.nan, columns=output, index=net[element].index,
                                        dtype=np.float64)
    else:
        net[res_element] = pd.DataFrame(np.zeros(0, dtype=output), index=[])
        net[res_element] = pd.DataFrame(np.nan, index=net[element].index,
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


def set_fixed_node_entries(net, node_pit, junctions, types, values, node_comp, mode):
    if not len(junctions):
        return [], []

    junction_idx_lookups = get_lookup(net, "node", "index")[node_comp.table_name()]
    use_numba = get_net_option(net, "use_numba")

    if mode == "p":
        val_col, type_col, count_col, typ, valid_types, values = \
            PINIT, NODE_TYPE, EXT_GRID_OCCURENCE, P, ["p", "pt"], values
    elif mode == "t":
        val_col, type_col, count_col, typ, valid_types, values = \
            TINIT, NODE_TYPE_T, EXT_GRID_OCCURENCE_T, T, ["t", "pt"], values
    else:
        raise UserWarning(r'The mode %s is not supported. Choose either mode "p" or "t"' % mode)

    mask = np.isin(types, valid_types)

    juncts, val_sum, number = _sum_by_group(use_numba, junctions[mask], values[mask],
                                            np.ones_like(values[mask], dtype=np.int32))

    index = junction_idx_lookups[juncts]

    node_pit[index, val_col] = (node_pit[index, val_col] * node_pit[index, count_col] + val_sum) / \
                               (number + node_pit[index, count_col])

    node_pit[index, count_col] += number
    node_pit[index, type_col] = typ

    return index


def get_mass_flow_at_nodes(net, node_pit, branch_pit, eg_nodes, comp):
    node_uni, inverse_nodes, counts = np.unique(eg_nodes, return_counts=True, return_inverse=True)
    eg_from_branches = np.isin(branch_pit[:, FROM_NODE], node_uni)
    eg_to_branches = np.isin(branch_pit[:, TO_NODE], node_uni)
    from_nodes = branch_pit[eg_from_branches, FROM_NODE]
    to_nodes = branch_pit[eg_to_branches, TO_NODE]
    mass_flow_from = branch_pit[eg_from_branches, LOAD_VEC_NODES_FROM]
    mass_flow_to = branch_pit[eg_to_branches, LOAD_VEC_NODES_TO]
    loads = node_pit[node_uni, LOAD]
    all_index_nodes = np.concatenate([from_nodes, to_nodes, node_uni])
    all_mass_flows = np.concatenate([-mass_flow_from, mass_flow_to, -loads])
    nodes, sum_mass_flows = _sum_by_group(get_net_option(net, "use_numba"), all_index_nodes,
                                          all_mass_flows)
    if not np.all(nodes == node_uni):
        raise UserWarning("In component %s: Something went wrong with the mass flow balance. "
                          "Please report this error at github." % comp.__name__)
    return sum_mass_flows, inverse_nodes, counts


def standard_branch_wo_internals_result_lookup(net):
    required_results_hyd = [
        ("p_from_bar", "p_from"), ("p_to_bar", "p_to"), ("mdot_to_kg_per_s", "mf_to"),
        ("mdot_from_kg_per_s", "mf_from")
    ]
    required_results_ht = [("t_from_k", "temp_from"), ("t_to_k", "temp_to"), ("t_outlet_k", "t_outlet")]

    if get_fluid(net).is_gas:
        required_results_hyd.extend([
            ("normfactor_from", "normfactor_from"),
            ("normfactor_to", "normfactor_to"), ("vdot_norm_m3_per_s", "vf")
        ])
    else:
        required_results_hyd.extend([("vdot_m3_per_s", "vf")])

    return required_results_hyd, required_results_ht


def get_component_array(net, component_name, component_type="branch", mode='hydraulics', only_active=True):
    """
    Returns the internal array of a component.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param component_name: Table name of the component for which to extract internal array
    :type component_name: str
    :param component_type: Type of component that is considered ("branch" or "node")
    :type component_type: str, default "branch"
    :param only_active: If True, only return entries of active elements (included in _active_pit)
    :type only_active: bool
    :return: component_array - internal array of the component
    :rtype: numpy.ndarray
    """
    if not only_active:
        return net["_pit"]["components"][component_name]
    f_all, t_all = get_lookup(net, component_type, "from_to")[component_name]
    in_service_elm = get_lookup(net, component_type, "active_%s"%mode)[f_all:t_all]
    return net["_pit"]["components"][component_name][in_service_elm]


def get_std_type_lookup(net, table_name):
    return np.array(list(net.std_types[table_name].keys()))
