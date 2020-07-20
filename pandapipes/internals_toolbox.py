import numpy as np


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


def select_from_pit(table_index_array, input_array, data):
    """
        Auxiliary function to retrieve values from a table like a pit. Each data entry corresponds to a
        table_index_array entry. Example: velocities are indexed by the corresponding from_nodes stored in the
        pipe pit.

        The function inputs another array which consists of some table_index_array entries the user wants to retrieve.
        The function is used in pandapipes results evaluation. The input array is the list of from_junction entries,
        corresponding only to the junction elements, not containing additional pipe nodes. The table_index_array
        is the complete list of from_nodes consisting of junction element entries and additional pipe section nodes.
        Data corresponds to the gas velocities.

        :param indices:
        :type indices:
        :param values:
        :type values:
        :return:
        :rtype:
        """
    sorter = np.argsort(table_index_array)
    indices = sorter[np.searchsorted(table_index_array, input_array, sorter=sorter)]

    return data[indices]