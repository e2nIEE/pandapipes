# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import pandas as pd


from pandapipes import is_fluid_gas
from pandapipes.constants import NORMAL_PRESSURE, TEMP_GRADIENT_KPM, AVG_TEMPERATURE_K, \
    HEIGHT_EXPONENT
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
                net['_idx_node']['PINIT'], net['_idx_node']['NODE_TYPE'], \
                net['_idx_node']['EXT_GRID_OCCURENCE'], net['_idx_node']['P'], ["p", "pt"], p_values
        else:
            val_col, type_col, eg_count_col, typ, valid_types, values = \
                net['_idx_node']['TINIT'], net['_idx_node']['NODE_TYPE_T'], \
                net['_idx_node']['EXT_GRID_OCCURENCE_T'], net['_idx_node']['T'], ["t", "pt"], t_values
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
    eg_from_branches = np.isin(branch_pit[:, net['_idx_branch']['FROM_NODE']], node_uni)
    eg_to_branches = np.isin(branch_pit[:, net['_idx_branch']['TO_NODE']], node_uni)
    from_nodes = branch_pit[eg_from_branches, net['_idx_branch']['FROM_NODE']]
    to_nodes = branch_pit[eg_to_branches, net['_idx_branch']['TO_NODE']]
    mass_flow_from = branch_pit[eg_from_branches, net['_idx_branch']['LOAD_VEC_NODES']]
    mass_flow_to = branch_pit[eg_to_branches, net['_idx_branch']['LOAD_VEC_NODES']]
    loads = node_pit[node_uni, net['_idx_node']['LOAD']]
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
        ("mdot_from_kg_per_s", "mf_from"), ("vdot_norm_m3_per_s", "vf"), ("lambda", "lambda"),
        ("reynolds", "reynolds")
    ]
    required_results_ht = [("t_from_k", "temp_from"), ("t_to_k", "temp_to")]

    if is_fluid_gas(net):
        required_results_hyd.extend([
            ("v_from_m_per_s", "v_gas_from"), ("v_to_m_per_s", "v_gas_to"),
            ("v_mean_m_per_s", "v_gas_mean"), ("normfactor_from", "normfactor_from"),
            ("normfactor_to", "normfactor_to")
        ])
    else:
        required_results_hyd.extend([("v_mean_m_per_s", "v_mps")])

    return required_results_hyd, required_results_ht


def get_component_array(net, component_name, component_type="branch", only_active=True):
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
    f_all, t_all = get_lookup(net, component_type, "from_to")[component_name]
    if not only_active:
        return net["_pit"]["components"][component_name]
    in_service_elm = get_lookup(net, component_type, "active_hydraulics")[f_all:t_all]
    return net["_pit"]["components"][component_name][in_service_elm]
