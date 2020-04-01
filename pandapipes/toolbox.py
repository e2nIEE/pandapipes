# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import pandas as pd
from pandapipes.pandapipes_net import pandapipesNet, logger
from pandapower.toolbox import dataframes_equal
from pandapower.auxiliary import get_indices
import os

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def nets_equal(net1, net2, check_only_results=False, exclude_elms=None, **kwargs):
    """
    Compares the DataFrames of two networks. The networks are considered equal if they share the
    same keys and values, except of the 'et' (elapsed time) entry which differs depending on
    runtime conditions and entries stating with '_'.

    :param net1:
    :type net1: pandapipesNet
    :param net2:
    :type net2:pandapipesNet
    :param check_only_results:
    :type check_only_results: bool, default False
    :param exclude_elms:
    :type exclude_elms: ?, default None
    :param kwargs:
    :type kwargs:
    :return:
    :rtype:
    """

    eq = isinstance(net1, pandapipesNet) and isinstance(net2, pandapipesNet)
    exclude_elms = [] if exclude_elms is None else list(exclude_elms)
    exclude_elms += ["res_" + ex for ex in exclude_elms]
    not_equal = []

    if eq:
        # for two networks make sure both have the same keys that do not start with "_"...
        net1_keys = [key for key in net1.keys() if not (key.startswith("_") or key in exclude_elms)]
        net2_keys = [key for key in net2.keys() if not (key.startswith("_") or key in exclude_elms)]
        keys_to_check = set(net1_keys) & set(net2_keys)
        key_difference = set(net1_keys) ^ set(net2_keys)

        if len(key_difference) > 0:
            logger.info("Networks entries mismatch at: %s" % key_difference)
            if not check_only_results:
                return False

        # ... and then iter through the keys, checking for equality for each table
        for df_name in list(keys_to_check):
            # skip 'et' (elapsed time) and entries starting with '_' (internal vars)
            if df_name != 'et' and not df_name.startswith("_"):
                if check_only_results and not df_name.startswith("res_"):
                    continue  # skip anything that is not a result table

                if isinstance(net1[df_name], pd.DataFrame) and isinstance(net2[df_name],
                                                                          pd.DataFrame):
                    frames_equal = dataframes_equal(net1[df_name], net2[df_name], **kwargs)
                    eq &= frames_equal

                    if not frames_equal:
                        not_equal.append(df_name)

    if len(not_equal) > 0:
        logger.info("Networks do not match in DataFrame(s): %s" % (', '.join(not_equal)))

    return eq


def _sum_by_group_sorted(indices, *values):
    """
    Auxiliary function to sum up values by some given indices (both as numpy arrays). Expects the
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
    for i in range(len(val)):
        # sum up values, chose only those with unique indices and then subtract the previous sums
        # --> this way for each index the sum of all values belonging to this index is returned
        np.cumsum(val[i], out=val[i])
        val[i] = val[i][index]
        val[i][1:] = val[i][1:] - val[i][:-1]
    return [indices] + val


def _sum_by_group(indices, *values):
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
    for i in range(len(val)):
        val[i] = val[i][order]

    return _sum_by_group_sorted(indices, *val)


def element_junction_tuples(junction_elements=True, branch_elements=True, res_elements=False):
    """
    Utility function
    Provides the tuples of elements and corresponding columns for junctions they are connected to
    :param junction_elements: whether tuples for junction elements e.g. sink, source,
    ... are included
    :param branch_elements: whether branch elements e.g. pipe, pumps, ... are included
    :return: set of tuples with element names and column names
    """
    ejts = set()
    if junction_elements:
        elements = ["sink", "source", "ext_grid"]
        for elm in elements:
            ejts.update([(elm, "junction")])
    if branch_elements:
        elements = ["pipe", "valve", "pump", "circ_pump_mass", "circ_pump_pressure",
                    "heat_exchanger"]
        for elm in elements:
            ejts.update([(elm, "from_junction"), (elm, "to_junction")])
    if res_elements:
        elements_without_res = ["valve"]
        ejts.update(
            [("res_" + ejt[0], ejt[1]) for ejt in ejts if ejt[0] not in elements_without_res])
    return ejts


def reindex_junctions(net, junction_lookup):
    """
    Changes the index of net.junction and considers the new junction indices in all other
    pandapipes element tables.

    INPUT:
      **net** - pandapipes network

      **junction_lookup** (dict) - the keys are the old junction indices, the values the new junction indices
    """
    not_fitting_junction_lookup_keys = set(junction_lookup.keys()) - set(net.junction.index)
    if len(not_fitting_junction_lookup_keys):
        logger.error("These junction indices are unknown to net. Thus, they cannot be reindexed: " +
                     str(not_fitting_junction_lookup_keys))

    missing_junction_indices = sorted(set(net.junction.index) - set(junction_lookup.keys()))
    if len(missing_junction_indices):
        duplicates = set(missing_junction_indices).intersection(set(junction_lookup.values()))
        if len(duplicates):
            logger.error("The junctions %s are not listed in junction_lookup but their index is "
                         "used as a new index. This would result in duplicated junction indices."
                         % (str(duplicates)))
        else:
            junction_lookup.update({j: j for j in missing_junction_indices})

    net.junction.index = get_indices(net.junction.index, junction_lookup)
    if hasattr(net, "res_junction"):
        net.res_junction.index = get_indices(net.res_junction.index, junction_lookup)

    for element, value in element_junction_tuples():
        if element in net.keys():
            net[element][value] = get_indices(net[element][value], junction_lookup)
    net["junction_geodata"].set_index(get_indices(net["junction_geodata"].index, junction_lookup),
                                      inplace=True)
    return junction_lookup
