# Copyright (c) 2020-2021 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import pandas as pd
from pandapipes.constants import NORMAL_PRESSURE, TEMP_GRADIENT_KPM, AVG_TEMPERATURE_K, \
    HEIGHT_EXPONENT, NORMAL_TEMPERATURE
from pandapipes.idx_branch import ELEMENT_IDX as ELEMENT_IDX_BRANCH, VINIT, LAMBDA, RE, FROM_NODE,\
    TO_NODE, TINIT as TINIT_BRANCH
from pandapipes.idx_node import PAMB, PINIT
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


def calculate_branch_results(res_table, branch_pit, node_pit, placement_table, fluid, use_numba):
    idx_active = branch_pit[:, ELEMENT_IDX_BRANCH]
    v_mps = branch_pit[:, VINIT]
    _, v_sum, internal_pipes = _sum_by_group(use_numba, idx_active, v_mps,
                                             np.ones_like(idx_active))
    idx_pit = branch_pit[:, ELEMENT_IDX_BRANCH]
    _, lambda_sum, reynolds_sum, = \
        _sum_by_group(use_numba, idx_pit, branch_pit[:, LAMBDA], branch_pit[:, RE])
    if fluid.is_gas:
        from_nodes = branch_pit[:, FROM_NODE].astype(np.int32)
        to_nodes = branch_pit[:, TO_NODE].astype(np.int32)
        numerator = NORMAL_PRESSURE * branch_pit[:, TINIT_BRANCH]
        p_from = node_pit[from_nodes, PAMB] + node_pit[from_nodes, PINIT]
        p_to = node_pit[to_nodes, PAMB] + node_pit[to_nodes, PINIT]
        mask = ~np.isclose(p_from, p_to)
        p_mean = np.empty_like(p_to)
        p_mean[~mask] = p_from[~mask]
        p_mean[mask] = 2 / 3 * (p_from[mask] ** 3 - p_to[mask] ** 3) \
                       / (p_from[mask] ** 2 - p_to[mask] ** 2)
        normfactor_mean = numerator * fluid.get_property("compressibility", p_mean) \
                          / (p_mean * NORMAL_TEMPERATURE)
        v_gas_mean = v_mps * normfactor_mean
        _, v_gas_mean_sum = _sum_by_group(use_numba, idx_active, v_gas_mean)
        res_table["v_mean_m_per_s"].values[placement_table] = v_gas_mean_sum / internal_pipes
    else:
        res_table["v_mean_m_per_s"].values[placement_table] = v_sum / internal_pipes
    res_table["lambda"].values[placement_table] = lambda_sum / internal_pipes
    res_table["reynolds"].values[placement_table] = reynolds_sum / internal_pipes
