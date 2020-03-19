# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from pandapipes.idx_branch import FROM_NODE, TO_NODE, JAC_DERIV_DV, JAC_DERIV_DP, JAC_DERIV_DP1, \
    JAC_DERIV_DV_NODE, LOAD_VEC_NODES, LOAD_VEC_BRANCHES, JAC_DERIV_DT, JAC_DERIV_DT1, \
    JAC_DERIV_DT_NODE, LOAD_VEC_NODES_T, LOAD_VEC_BRANCHES_T, FROM_NODE_T, TO_NODE_T
from pandapipes.idx_node import LOAD, TINIT
from pandapipes.idx_node import P, NODE_TYPE, T, NODE_TYPE_T
from pandapipes.toolbox import _sum_by_group, _sum_by_group_sorted
from scipy.sparse import csr_matrix
from pandapipes.pipeflow_setup import get_net_option


def build_system_matrix(net, branch_pit, node_pit, heat_mode):
    """

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param branch_pit:
    :type branch_pit:
    :param node_pit:
    :type node_pit:
    :param heat_mode:
    :type heat_mode:
    :return: system_matrix, load_vector
    :rtype:
    """
    update_option = get_net_option(net, "only_update_hydraulic_matrix")
    update_only = update_option and "hydraulic_data_sorting" in net["_internal_data"] \
                  and "hydraulic_matrix" in net["_internal_data"]

    len_b = len(branch_pit)
    len_n = len(node_pit)
    branch_matrix_indices = np.arange(len_b) + len_n
    fn_col, tn_col, ntyp_col, slack_type, num_der = (FROM_NODE, TO_NODE, NODE_TYPE, P, 3) \
        if not heat_mode else (FROM_NODE_T, TO_NODE_T, NODE_TYPE_T, T, 2)

    fn = branch_pit[:, fn_col].astype(np.int32)
    tn = branch_pit[:, tn_col].astype(np.int32)
    not_slack_fn_branch_mask = node_pit[fn, ntyp_col] != slack_type
    not_slack_tn_branch_mask = node_pit[tn, ntyp_col] != slack_type
    slack_nodes = np.where(node_pit[:, ntyp_col] == slack_type)[0]

    if not heat_mode:
        len_fn_not_slack = np.sum(not_slack_fn_branch_mask)
        len_tn_not_slack = np.sum(not_slack_tn_branch_mask)
        len_fn1 = num_der * len_b + len_fn_not_slack
        len_tn1 = len_fn1 + len_tn_not_slack
        full_len = len_tn1 + slack_nodes.shape[0]
    else:
        inc_flow_sum = np.zeros(len(node_pit[:, LOAD]))
        tn_unique_der, tn_sums_der = _sum_by_group(tn, branch_pit[:, JAC_DERIV_DT_NODE])
        inc_flow_sum[tn_unique_der] += tn_sums_der
        len_fn1 = num_der * len_b + len(tn_unique_der)
        len_tn1 = len_fn1 + len_b
        full_len = len_tn1 + slack_nodes.shape[0]

    system_data = np.zeros(full_len, dtype=np.float64)

    if not heat_mode:
        # pdF_dv
        system_data[:len_b] = branch_pit[:, JAC_DERIV_DV]
        # pdF_dpi
        system_data[len_b:2 * len_b] = branch_pit[:, JAC_DERIV_DP]
        # pdF_dpi1
        system_data[2 * len_b:3 * len_b] = branch_pit[:, JAC_DERIV_DP1]
        # jdF_dv_from_nodes
        system_data[3 * len_b:len_fn1] = branch_pit[not_slack_fn_branch_mask, JAC_DERIV_DV_NODE]
        # jdF_dv_to_nodes
        system_data[len_fn1:len_tn1] = branch_pit[not_slack_tn_branch_mask,
                                                  JAC_DERIV_DV_NODE] * (-1)
        # p_nodes
        system_data[len_tn1:] = 1
    else:
        system_data[:len_b] = branch_pit[:, JAC_DERIV_DT]
        # pdF_dpi1
        system_data[len_b:2 * len_b] = branch_pit[:, JAC_DERIV_DT1]
        # jdF_dv_from_nodes
        system_data[2 * len_b:len_fn1] = inc_flow_sum[tn_unique_der]
        # jdF_dv_to_nodes
        data = branch_pit[:, JAC_DERIV_DT_NODE] * (-1)
        rows = tn
        index = np.argsort(rows)
        data = data[index]
        system_data[len_fn1:len_fn1 + len_b] = data
        system_data[len_fn1 + len_b:] = 1

    if not update_only:
        system_cols = np.zeros(full_len, dtype=np.int32)
        system_rows = np.zeros(full_len, dtype=np.int32)

        if not heat_mode:
            # pdF_dv
            system_cols[:len_b] = branch_matrix_indices
            system_rows[:len_b] = branch_matrix_indices

            # pdF_dpi
            system_cols[len_b:2 * len_b] = fn
            system_rows[len_b:2 * len_b] = branch_matrix_indices

            # pdF_dpi1
            system_cols[2 * len_b:3 * len_b] = tn
            system_rows[2 * len_b:3 * len_b] = branch_matrix_indices

            # jdF_dv_from_nodes
            system_cols[3 * len_b:len_fn1] = branch_matrix_indices[not_slack_fn_branch_mask]
            system_rows[3 * len_b:len_fn1] = fn[not_slack_fn_branch_mask]

            # jdF_dv_to_nodes
            system_cols[len_fn1:len_tn1] = branch_matrix_indices[not_slack_tn_branch_mask]
            system_rows[len_fn1:len_tn1] = tn[not_slack_tn_branch_mask]

            # p_nodes
            system_cols[len_tn1:] = slack_nodes
            system_rows[len_tn1:] = slack_nodes
        else:
            # pdF_dTfromnode
            system_cols[:len_b] = fn
            system_rows[:len_b] = branch_matrix_indices

            # pdF_dTout
            system_cols[len_b:2 * len_b] = branch_matrix_indices
            system_rows[len_b:2 * len_b] = branch_matrix_indices

            # t_nodes
            system_cols[len_fn1 + len_b:] = slack_nodes
            system_rows[len_fn1 + len_b:] = np.arange(0, len(slack_nodes))

            # jdF_dTnode_
            tn_unique_idx = np.unique(tn, return_index=True)
            system_cols[2 * len_b:len_fn1] = tn_unique_idx[0]
            system_rows[2 * len_b:len_fn1] = len(slack_nodes) + np.arange(0, len(tn_unique_der))

            # jdF_dTout
            branch_order = np.argsort(tn)
            tn_uni, tn_uni_counts = np.unique(tn[branch_order], return_counts=True)
            row_index = np.repeat(np.arange(len(tn_uni)), tn_uni_counts)
            system_cols[len_fn1:len_fn1 + len_b] = branch_matrix_indices[branch_order]
            system_rows[len_fn1:len_fn1 + len_b] = len(slack_nodes) + row_index

        if not update_option:
            system_matrix = csr_matrix((system_data, (system_rows, system_cols)),
                                       shape=(len_n + len_b, len_n + len_b))



        else:
            data_order = np.lexsort([system_cols, system_rows])
            system_data = system_data[data_order]
            system_cols = system_cols[data_order]
            system_rows = system_rows[data_order]

            row_counter = np.zeros(len_b + len_n + 1, dtype=np.int32)
            unique_rows, row_counts = _sum_by_group_sorted(system_rows, np.ones_like(system_rows))
            row_counter[unique_rows + 1] += row_counts
            ptr = row_counter.cumsum()
            system_matrix = csr_matrix((system_data, system_cols, ptr),
                                       shape=(len_n + len_b, len_n + len_b))
            net["_internal_data"]["hydraulic_data_sorting"] = data_order
            net["_internal_data"]["hydraulic_matrix"] = system_matrix
    else:
        data_order = net["_internal_data"]["hydraulic_data_sorting"]
        system_data = system_data[data_order]
        system_matrix = net["_internal_data"]["hydraulic_matrix"]
        system_matrix.data = system_data

    if not heat_mode:
        load_vector = np.empty(len_n + len_b)
        load_vector[len_n:] = branch_pit[:, LOAD_VEC_BRANCHES]
        load_vector[:len_n] = node_pit[:, LOAD] * (-1)
        fn_unique, fn_sums = _sum_by_group(fn, branch_pit[:, LOAD_VEC_NODES])
        tn_unique, tn_sums = _sum_by_group(tn, branch_pit[:, LOAD_VEC_NODES])
        load_vector[fn_unique] -= fn_sums
        load_vector[tn_unique] += tn_sums
        load_vector[slack_nodes] = 0
    else:
        tn_unique, tn_sums = _sum_by_group(tn, branch_pit[:, LOAD_VEC_NODES_T])
        load_vector = np.zeros(len_n + len_b)
        load_vector[len(slack_nodes) + np.arange(0, len(tn_unique_der))] += tn_sums
        load_vector[len(slack_nodes) + np.arange(0, len(tn_unique_der))] -= tn_sums_der * node_pit[
            tn_unique_der, TINIT]
        load_vector[0:len(slack_nodes)] = 0.

        load_vector[len_n:] = branch_pit[:, LOAD_VEC_BRANCHES_T]

    return system_matrix, load_vector
