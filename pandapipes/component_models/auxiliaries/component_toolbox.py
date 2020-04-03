# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import pandas as pd
from pandapipes.constants import NORMAL_PRESSURE, R_UNIVERSAL, \
    MOLAR_MASS_AIR, GRAVITATION_CONSTANT, TEMP_GRADIENT_KPM, AVG_TEMPERATURE_K, HEIGHT_EXPONENT


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
    cat_start = np.repeat(starts, lengths)
    # Create group counter that resets for each start/length
    cat_counter = np.arange(lengths.sum()) - np.repeat(lengths.cumsum() - lengths, lengths)
    # Add group counter to group specific starts
    cat_range = cat_start + cat_counter
    return cat_range


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
