# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from scipy.sparse import csr_matrix

from pandapipes.idx_branch import FROM_NODE, TO_NODE, JAC_DERIV_DM, JAC_DERIV_DP, JAC_DERIV_DP1, \
    JAC_DERIV_DM_NODE, LOAD_VEC_NODES, LOAD_VEC_BRANCHES, JAC_DERIV_DT, JAC_DERIV_DTOUT, CIRC, PUMP_TYPE, \
    JAC_DERIV_DT_NODE_B, JAC_DERIV_DT_NODE_N, LOAD_VEC_NODES_T, LOAD_VEC_BRANCHES_T, FROM_NODE_T, TO_NODE_T, BRANCH_TYPE

from pandapipes.idx_node import P, PC, NODE_TYPE, T, NODE_TYPE_T, LOAD, INFEED, MDOTSLACKINIT, JAC_DERIV_MSL
from pandapipes.pf.internals_toolbox import _sum_by_group_sorted, _sum_by_group
from pandapipes.pf.pipeflow_setup import get_net_option


def build_system_matrix(net, branch_pit, node_pit, heat_mode):
    """
    Builds the system matrix.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param branch_pit: pandapipes internal table for branching components such as pipes or valves
    :type branch_pit: numpy.ndarray
    :param node_pit:  pandapipes internal table for node components
    :type node_pit: numpy.ndarray
    :param heat_mode: Is it a heat network calculation: True or False
    :type heat_mode: bool
    :return: system_matrix, load_vector
    :rtype: system_matrix - scipy.sparse.csr.csr_matrix, load_vector - numpy.ndarray
    """
    update_option = get_net_option(net, "only_update_hydraulic_matrix")
    update_only = update_option and "hydraulic_data_sorting" in net["_internal_data"] \
                  and "hydraulic_matrix" in net["_internal_data"]
    use_numba = get_net_option(net, "use_numba")

    len_b = len(branch_pit)
    len_n = len(node_pit)
    branch_matrix_indices = np.arange(len_b) + len_n
    fn_col, tn_col, ntyp_col, slack_type, pc_type, num_der = \
        (FROM_NODE, TO_NODE, NODE_TYPE, P, PC, 3) \
            if not heat_mode else (FROM_NODE_T, TO_NODE_T, NODE_TYPE_T, T, PC, 2)
    pc_nodes = np.where(node_pit[:, ntyp_col] == pc_type)[0]
    fn = branch_pit[:, fn_col].astype(np.int32)
    tn = branch_pit[:, tn_col].astype(np.int32)

    pc_branch_mask = branch_pit[:, BRANCH_TYPE] == pc_type
    slack_nodes = np.where(node_pit[:, ntyp_col] == slack_type)[0]
    pc_matrix_indices = branch_matrix_indices[pc_branch_mask]

    # size of the matrix
    if not heat_mode:
        len_sl = len(slack_nodes)
        slack_mass_matrix_indices = np.arange(len_sl) + len_b + len_n
        slack_masses_from, slack_branches_from = np.where(branch_pit[:, fn_col] == slack_nodes[:, None])
        slack_masses_to, slack_branches_to = np.where(branch_pit[:, tn_col] == slack_nodes[:, None])
        not_slack_fn_branch_mask = node_pit[fn, ntyp_col] != slack_type
        not_slack_tn_branch_mask = node_pit[tn, ntyp_col] != slack_type
        len_fn_not_slack = np.sum(not_slack_fn_branch_mask)
        len_tn_not_slack = np.sum(not_slack_tn_branch_mask)
        len_fn1 = num_der * len_b + len_fn_not_slack
        len_tn1 = len_fn1 + len_tn_not_slack
        len_pc = len_tn1 + pc_nodes.shape[0]
        len_slack = len_pc + slack_nodes.shape[0]
        len_fsb = len_slack + len(slack_branches_from)
        len_tsb = len_fsb + len(slack_branches_to)
        full_len = len_tsb + slack_nodes.shape[0]
    else:
        len_sl = 0
        not_slack_branch_mask = branch_pit[:, PUMP_TYPE] != CIRC
        len_tn_not_slack = np.sum(not_slack_branch_mask)
        infeed_node = np.arange(len_n)[node_pit[:, INFEED].astype(np.bool_)]
        len_tn1 = num_der * len_b + len_tn_not_slack
        len_tn2 = len_tn1 + len_tn_not_slack
        full_len = len_tn2 + slack_nodes.shape[0]

    system_data = np.zeros(full_len, dtype=np.float64)

    # entries in the matrix
    if not heat_mode:

        # branch equations
        # ----------------
        # branch_dF_dm
        system_data[:len_b] = branch_pit[:, JAC_DERIV_DM]
        # branch_dF_dp_from
        system_data[len_b:2 * len_b] = branch_pit[:, JAC_DERIV_DP]
        # branch_dF_dp_to
        system_data[2 * len_b:3 * len_b] = branch_pit[:, JAC_DERIV_DP1]

        # node equations
        # --------------
        # from_node_dF_dm
        system_data[3 * len_b:len_fn1] = branch_pit[not_slack_fn_branch_mask, JAC_DERIV_DM_NODE] * (-1)
        # to_node_dF_dm
        system_data[len_fn1:len_tn1] = branch_pit[not_slack_tn_branch_mask, JAC_DERIV_DM_NODE]

        # fixed pressure equations
        # ------------------------
        # pc_nodes and slack_nodes
        system_data[len_tn1:len_slack] = 1

        # mass flow slack equation
        # --------------
        # from_slack_dF_dm
        system_data[len_slack:len_fsb] = branch_pit[slack_branches_from, JAC_DERIV_DM_NODE] * (-1)
        # to_slack_dF_dm
        system_data[len_fsb:len_tsb] = branch_pit[slack_branches_to, JAC_DERIV_DM_NODE]
        # slackmass_dF_dmslack
        system_data[len_tsb:] = node_pit[slack_nodes, JAC_DERIV_MSL]
    else:

        # branch equations
        # ----------------
        # branch_dF_dT_from
        system_data[:len_b] = branch_pit[:, JAC_DERIV_DT]
        # branch_dF_dT_out
        system_data[len_b:2 * len_b] = branch_pit[:, JAC_DERIV_DTOUT]

        # node equations
        # --------------
        # node_dF_dT_to
        system_data[2 * len_b:len_tn1] = branch_pit[not_slack_branch_mask, JAC_DERIV_DT_NODE_N]
        # node_dF_dT_out
        system_data[len_tn1:len_tn2] = branch_pit[not_slack_branch_mask, JAC_DERIV_DT_NODE_B]

        # fixed temperature equations
        # ---------------------------
        # t_nodes
        system_data[len_tn2:] = 1

    # position in the matrix
    if not update_only:
        system_cols = np.zeros(full_len, dtype=np.int32)
        system_rows = np.zeros(full_len, dtype=np.int32)

        if not heat_mode:

            # branch equations
            # ----------------
            # branch_dF_dm
            system_cols[:len_b] = branch_matrix_indices
            system_rows[:len_b] = branch_matrix_indices
            # branch_dF_dp_from
            system_cols[len_b:2 * len_b] = fn
            system_rows[len_b:2 * len_b] = branch_matrix_indices
            # branch_dF_dp_to
            system_cols[2 * len_b:3 * len_b] = tn
            system_rows[2 * len_b:3 * len_b] = branch_matrix_indices

            # node equations
            # --------------
            # from_node_dF_dm
            system_cols[3 * len_b:len_fn1] = branch_matrix_indices[not_slack_fn_branch_mask]
            system_rows[3 * len_b:len_fn1] = fn[not_slack_fn_branch_mask]
            # to_node_dF_dm
            system_cols[len_fn1:len_tn1] = branch_matrix_indices[not_slack_tn_branch_mask]
            system_rows[len_fn1:len_tn1] = tn[not_slack_tn_branch_mask]

            # fixed pressure equations
            # -----------------------
            # pc_nodes
            system_cols[len_tn1:len_pc] = pc_nodes
            system_rows[len_tn1:len_pc] = pc_matrix_indices
            # slack_nodes
            system_cols[len_pc:len_slack] = slack_nodes
            system_rows[len_pc:len_slack] = slack_nodes

            # mass flow slack equation
            # --------------
            # from_slack_dF_dm
            system_cols[len_slack:len_fsb] = branch_matrix_indices[slack_branches_from]
            system_rows[len_slack:len_fsb] = slack_mass_matrix_indices[slack_masses_from]
            # to_slack_dF_dm
            system_cols[len_fsb:len_tsb] = branch_matrix_indices[slack_branches_to]
            system_rows[len_fsb:len_tsb] = slack_mass_matrix_indices[slack_masses_to]
            # to_slack_dF_dm
            system_cols[len_fsb:len_tsb] = branch_matrix_indices[slack_branches_to]
            system_rows[len_fsb:len_tsb] = slack_mass_matrix_indices[slack_masses_to]
            # slackmass_dF_dmslack
            system_cols[len_tsb:] = slack_mass_matrix_indices
            system_rows[len_tsb:] = slack_mass_matrix_indices

        else:
            # branch equations
            # ----------------
            # branch_dF_dT_from
            system_cols[:len_b] = fn
            system_rows[:len_b] = branch_matrix_indices
            # branch_dF_dT_out
            system_cols[len_b:2 * len_b] = branch_matrix_indices
            system_rows[len_b:2 * len_b] = branch_matrix_indices

            # node equations
            # --------------
            # node_dF_dT_to
            system_cols[2 * len_b:len_tn1] = tn[not_slack_branch_mask]
            system_rows[2 * len_b:len_tn1] = tn[not_slack_branch_mask]
            # node_dF_dT_out
            system_cols[len_tn1:len_tn2] = branch_matrix_indices[not_slack_branch_mask]
            system_rows[len_tn1:len_tn2] = tn[not_slack_branch_mask]

            # fixed temperature equations
            # ---------------------------
            # t_nodes (overwrites only infeeding nodes' equation)
            system_cols[len_tn2:] = slack_nodes
            system_rows[len_tn2:] = infeed_node

        if not update_option:
            system_matrix = csr_matrix((system_data, (system_rows, system_cols)),
                                       shape=(len_n + len_b + len_sl, len_n + len_b + len_sl))

        else:
            data_order = np.lexsort([system_cols, system_rows])
            system_data = system_data[data_order]
            system_cols = system_cols[data_order]
            system_rows = system_rows[data_order]

            row_counter = np.zeros(len_b + len_n + len_sl + 1, dtype=np.int32)
            unique_rows, row_counts = _sum_by_group_sorted(system_rows, np.ones_like(system_rows))
            row_counter[unique_rows + 1] += row_counts
            ptr = row_counter.cumsum()
            system_matrix = csr_matrix((system_data, system_cols, ptr),
                                       shape=(len_n + len_b + len_sl, len_n + len_b + len_sl))
            net["_internal_data"]["hydraulic_data_sorting"] = data_order
            net["_internal_data"]["hydraulic_matrix"] = system_matrix
    else:
        data_order = net["_internal_data"]["hydraulic_data_sorting"]
        system_data = system_data[data_order]
        system_matrix = net["_internal_data"]["hydraulic_matrix"]
        system_matrix.data = system_data

    # load vector on the right side
    if not heat_mode:
        load_vector = np.empty(len_n + len_b + len_sl)
        load_vector[len_n:len_b + len_n] = branch_pit[:, LOAD_VEC_BRANCHES]
        load_vector[:len_n] = node_pit[:, LOAD] * (-1)
        fn_unique, fn_sums = _sum_by_group(use_numba, fn, branch_pit[:, LOAD_VEC_NODES])
        tn_unique, tn_sums = _sum_by_group(use_numba, tn, branch_pit[:, LOAD_VEC_NODES])
        load_vector[fn_unique] -= fn_sums
        load_vector[tn_unique] += tn_sums
        load_vector[slack_nodes] = 0
        load_vector[pc_matrix_indices] = branch_pit[pc_branch_mask, LOAD_VEC_BRANCHES]

        load_vector[slack_mass_matrix_indices] = node_pit[slack_nodes, LOAD] * (-1)
        fsb_unique, fsb_sums = _sum_by_group(use_numba, slack_masses_from, branch_pit[slack_branches_from, LOAD_VEC_NODES])
        tsb_unique, tsb_sums = _sum_by_group(use_numba, slack_masses_to, branch_pit[slack_branches_to, LOAD_VEC_NODES])
        load_vector[slack_mass_matrix_indices[fsb_unique]] -= fsb_sums
        load_vector[slack_mass_matrix_indices[tsb_unique]] += tsb_sums
        load_vector[slack_mass_matrix_indices] -= node_pit[slack_nodes, MDOTSLACKINIT]
    else:
        load_vector = np.zeros(len_n + len_b)
        load_vector[len_n:] = branch_pit[:, LOAD_VEC_BRANCHES_T]
        tn_unique_n, tn_sums_n = _sum_by_group(use_numba, tn, branch_pit[:, LOAD_VEC_NODES_T])
        load_vector[tn_unique_n] += tn_sums_n
        load_vector[infeed_node] = 0

    return system_matrix, load_vector
