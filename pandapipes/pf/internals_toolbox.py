# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import logging
try:
    from numba import jit
    numba_installed = True
except ImportError:
    from pandapower.pf.no_numba import jit
    numba_installed = False


logger = logging.getLogger(__name__)


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
            val[i][still_na] = np.NaN
        else:
            np.cumsum(val[i], out=val[i])
            val[i] = val[i][index]
            val[i][1:] = val[i][1:] - val[i][:-1]
    return [indices] + val


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
