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