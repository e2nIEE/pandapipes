# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import logging

import numpy as np
import numpy.typing as npt

from pandapipes.idx_branch import FROM_NODE, FROM_NODE_T_SWITCHED, TO_NODE

logger = logging.getLogger(__name__)


def sum_by_group(
    groups: npt.NDArray,
    *values: npt.NDArray,
) -> tuple[npt.NDArray, list[npt.NDArray]]:
    """
    Sum values by group.

    :param groups: group labels for each element.
        Will be converted to ``int``.
    :param values: arrays of items
        One or more value arrays to sum by group.
    :return:
        A tuple of ``(unique_groups, grouped_sums)``, where:

        - unique_groups
            sorted unique group labels.
        - grouped_sums[i]
            summed values for the i-th input array, same length as ``unique_groups``.

    .. warning::
        We use ``np.bincount`` which creates an array of ``groups.max() + 1`` size,
        so having large numbers (> ``1<<24``) in the ``groups`` array is disadvantageous.

    .. note::
        NaNs are propagated: if an item of some value array is NaN, the sum of this value
        for a corresponding group will be NaN.
    """
    if groups.size == 0:
        return np.empty(0, dtype=int), [np.empty(0, dtype=val.dtype) for val in values]

    groups = groups.astype(int, copy=False)
    unique_groups = np.bincount(groups).nonzero()[0]
    grouped_sums = [
        np.bincount(groups, weights=val)[unique_groups].astype(val.dtype, copy=False)
        for val in values
    ]
    return unique_groups, grouped_sums


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
