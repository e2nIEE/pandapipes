# Copyright (c) 2020-2021 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from scipy.sparse import csr_matrix

from pandapipes.internals_toolbox import _sum_by_group_sorted, _sum_by_group
from pandapipes.pipeflow_setup import get_net_option, get_lookup


def build_system_matrix(net, branch_pit, node_pit, node_element_pit, heat_mode, first_iter):
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

    len_b = len(branch_pit)
    len_n = len(node_pit)

    slack_element_mask = (node_element_pit[:, net._idx_node_element['NODE_ELEMENT_TYPE']] == 1)
    len_s = len(slack_element_mask[slack_element_mask])
    len_ne = len(node_element_pit[slack_element_mask])

    len_fluid = 0 if first_iter else len(net._fluid) - 1
    len_nw, len_bw, len_sw = (len_n * len_fluid, len_b * len_fluid, len_s * len_fluid)

    extra = 0 if heat_mode else len_ne
    branch_matrix_indices = np.arange(len_b) + len_n
    node_element_matrix_indices = np.arange(len_ne) + len_b + len_n
    fn_col, tn_col, ntyp_col, slack_type, pc_type, nej_col, min_col, num_der = \
        (net['_idx_branch']['FROM_NODE'], net['_idx_branch']['TO_NODE'], net['_idx_node']['NODE_TYPE'],
         net['_idx_node']['P'], net['_idx_node']['PC'], net['_idx_node_element']['JUNCTION'],
         net['_idx_node_element']['MINIT'], 3) \
            if not heat_mode else (
            net['_idx_branch']['FROM_NODE_T'], net['_idx_branch']['TO_NODE_T'], net['_idx_node']['NODE_TYPE_T'],
            net['_idx_node']['T'], net['_idx_node']['PC'], net['_idx_node_element']['JUNCTION'],
            net['_idx_node_element']['MINIT'], 2)
    pc_nodes = np.where(node_pit[:, ntyp_col] == pc_type)[0]
    slack_masses_from, slack_branches_from = np.where(
        branch_pit[:, fn_col] == node_element_pit[slack_element_mask, nej_col][:, np.newaxis])
    slack_masses_to, slack_branches_to = np.where(
        branch_pit[:, tn_col] == node_element_pit[slack_element_mask, nej_col][:, np.newaxis])
    slack_mass = node_element_pit[slack_element_mask, min_col]
    _, inv, count = np.unique(node_element_pit[slack_element_mask, nej_col], return_inverse=True, return_counts=True)
    fn = branch_pit[:, fn_col].astype(np.int32)
    tn = branch_pit[:, tn_col].astype(np.int32)

    w_ne_col = get_lookup(net, "node_element", "w")[:-1]
    w_n_col = get_lookup(net, "node", "w")[:-1]

    w_node_matrix_indices = np.arange(len_nw) + len_ne + len_b + len_n
    not_slack_fn_branch_mask = node_pit[fn, ntyp_col] != slack_type
    not_slack_tn_branch_mask = node_pit[tn, ntyp_col] != slack_type
    pc_branch_mask = branch_pit[:, net['_idx_branch']['BRANCH_TYPE']] == pc_type
    slack_nodes = np.where(node_pit[:, ntyp_col] == slack_type)[0]
    pc_matrix_indices = branch_matrix_indices[pc_branch_mask]

    vfn = branch_pit[:, net['_idx_branch']['V_FROM_NODE']].astype(int)
    vtn = branch_pit[:, net['_idx_branch']['V_TO_NODE']].astype(int)
    not_slack_vfn_branch_mask = node_pit[vfn, ntyp_col] != slack_type
    not_slack_vtn_branch_mask = node_pit[vtn, ntyp_col] != slack_type

    if len_fluid and not first_iter:
        # get nodes the fluid is moving from and to (ignoring the from_nodes and to_nodes convention)
        fn_w = get_w_like_node_vector(vfn, len_fluid, len_n)
        tn_w = get_w_like_node_vector(vtn, len_fluid, len_n)

        # get slack nodes as w like vector
        slack_nodes_w = get_w_like_node_vector(node_element_pit[slack_element_mask, nej_col].astype(int),
                                               len_fluid, len_n)

        # derivate w after slack mass
        slack_wdF_dm = get_slack_wdF_dm(net, node_pit, node_element_pit, slack_element_mask, w_n_col, w_ne_col,
                                        count, inv)

        # derivate load from node after w
        n_mdF_dw = get_n_mdF_dw(net, node_pit, node_element_pit, slack_element_mask, len_fluid, len_n)

        # derivate branches mass flow from node after w
        n_wdF_dw = get_n_wdF_dw(net, branch_pit)

        # derivate branches w after v
        wdF_dv = get_wdF_dv(net, node_pit, branch_pit, w_n_col)

        # derivate branches from node rho after w
        fn_rhodF_dw = get_fn_rhodF_dw(net, branch_pit, not_slack_fn_branch_mask)

        # derivate branches to node rho after w
        tn_rhodF_dw = get_tn_rhodF_dw(net, branch_pit, not_slack_tn_branch_mask)

        # derivate from slack mass rho after w
        fslack_rhodF_dw = get_fslack_rhodF_dw(net, branch_pit, slack_branches_from)

        # derivate to slack mass rho after w
        tslack_rhodF_dw = get_tslack_rhodF_dw(net, branch_pit, slack_branches_to)

        # derivate nodes rho after w
        # TODO: still missing, needs to be implemented

        extra += len_nw
    else:
        fn_w, tn_w, slack_nodes_w, slack_wdF_dm, n_mdF_dw, n_wdF_dw, wdF_dv, \
        fn_rhodF_dw, tn_rhodF_dw, fslack_rhodF_dw, tslack_rhodF_dw = \
            [], [], [], [], [], [], [], \
            [], [], [], []


    if not heat_mode:
        len_fn_not_slack = np.sum(not_slack_fn_branch_mask)
        len_tn_not_slack = np.sum(not_slack_tn_branch_mask)
        #len_vfn_not_slack = np.sum(not_slack_vfn_branch_mask)
        #len_vtn_not_slack = np.sum(not_slack_vtn_branch_mask)
        len_fn1 = num_der * len_b + len_fn_not_slack
        len_tn1 = len_fn1 + len_tn_not_slack
        len_pc = len_tn1 + pc_nodes.shape[0]
        len_slack = len_pc + slack_nodes.shape[0]
        len_fne = len_slack + len(slack_masses_from)
        len_tne = len_fne + len(slack_masses_to)
        len_m_ne = len_tne + len_ne

        len_slack_wdF_dm = len_sw + len_m_ne
        len_n_mdF_dw = len_nw + len_slack_wdF_dm
        len_fn_wdF_dw = len_bw + len_n_mdF_dw
        len_tn_wdF_dw = len_bw + len_fn_wdF_dw
        len_fn_wdF_dv = len_bw + len_tn_wdF_dw
        len_tn_wdF_dv = len_bw + len_fn_wdF_dv
        len_fn_rhodF_dw = len_fn_not_slack * len_fluid + len_tn_wdF_dv
        len_tn_rhodF_dw = len_tn_not_slack * len_fluid + len_fn_rhodF_dw
        len_fslack_rhodF_dw = len(slack_branches_from) * len_fluid + len_tn_rhodF_dw
        len_tslack_rhodF_dw = len(slack_branches_to) * len_fluid + len_fslack_rhodF_dw
        full_len = len_tslack_rhodF_dw
    else:
        inc_flow_sum = np.zeros(len(node_pit[:, net['_idx_node']['LOAD']]))
        tn_unique_der, tn_sums_der = _sum_by_group(tn, branch_pit[:, net['_idx_branch']['JAC_DERIV_DT_NODE']])
        inc_flow_sum[tn_unique_der] += tn_sums_der
        len_fn1 = num_der * len_b + len(tn_unique_der)
        len_tn1 = len_fn1 + len_b
        full_len = len_tn1 + slack_nodes.shape[0]

    system_data = np.zeros(full_len, dtype=np.float64)

    if not heat_mode:
        # pdF_dv
        system_data[:len_b] = branch_pit[:, net['_idx_branch']['JAC_DERIV_DV']]
        # pdF_dpi
        system_data[len_b:2 * len_b] = branch_pit[:, net['_idx_branch']['JAC_DERIV_DP']]
        # pdF_dpi1
        system_data[2 * len_b:3 * len_b] = branch_pit[:, net['_idx_branch']['JAC_DERIV_DP1']]
        # jdF_dv_from_nodes
        system_data[3 * len_b:len_fn1] = branch_pit[not_slack_fn_branch_mask, net['_idx_branch']['JAC_DERIV_DV_NODE']]
        # jdF_dv_to_nodes
        system_data[len_fn1:len_tn1] = branch_pit[not_slack_tn_branch_mask,
                                                  net['_idx_branch']['JAC_DERIV_DV_NODE']] * (-1)
        # pc_nodes and p_nodes
        system_data[len_tn1:len_slack] = 1
        # fne_mdF_dv
        system_data[len_slack:len_fne] = branch_pit[slack_branches_from, net['_idx_branch']['JAC_DERIV_DV_NODE']]
        # tne_mdF_dv
        system_data[len_fne:len_tne] = branch_pit[slack_branches_to, net['_idx_branch']['JAC_DERIV_DV_NODE']] * (-1)
        # ne_mdF_dm
        system_data[len_tne:len_m_ne] = 1

        # slack_wdF_dm
        system_data[len_m_ne:len_slack_wdF_dm] = slack_wdF_dm
        # n_mdF_dw
        system_data[len_slack_wdF_dm:len_n_mdF_dw] = n_mdF_dw
        # fn_wdF_dw
        system_data[len_n_mdF_dw:len_fn_wdF_dw] = n_wdF_dw
        # tn_wdF_dw
        system_data[len_fn_wdF_dw:len_tn_wdF_dw] = n_wdF_dw * (-1)
        # fn_wdF_dv
        system_data[len_tn_wdF_dw:len_fn_wdF_dv] = wdF_dv
        # tn_wdF_dv
        system_data[len_fn_wdF_dv:len_tn_wdF_dv] = wdF_dv * (-1)
        # fn_rhodF_dw
        system_data[len_tn_wdF_dv:len_fn_rhodF_dw] = fn_rhodF_dw
        # tn_rhodF_dw
        system_data[len_fn_rhodF_dw:len_tn_rhodF_dw] = tn_rhodF_dw * (-1)
        # fslack_rhodF_dw
        system_data[len_tn_rhodF_dw:len_fslack_rhodF_dw] = fslack_rhodF_dw
        # fslack_rhodF_dw
        system_data[len_fslack_rhodF_dw:] = tslack_rhodF_dw * (-1)
    else:
        system_data[:len_b] = branch_pit[:, net['_idx_branch']['JAC_DERIV_DT']]
        # pdF_dpi1
        system_data[len_b:2 * len_b] = branch_pit[:, net['_idx_branch']['JAC_DERIV_DT1']]
        # jdF_dv_from_nodes
        system_data[2 * len_b:len_fn1] = inc_flow_sum[tn_unique_der]
        # jdF_dv_to_nodes
        data = branch_pit[:, net['_idx_branch']['JAC_DERIV_DT_NODE']] * (-1)
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

            # pc_nodes
            system_cols[len_tn1:len_pc] = pc_nodes
            system_rows[len_tn1:len_pc] = pc_matrix_indices

            # p_nodes
            system_cols[len_pc:len_slack] = slack_nodes
            system_rows[len_pc:len_slack] = slack_nodes

            # fne_mdF_dv
            system_cols[len_slack:len_fne] = branch_matrix_indices[slack_branches_from]
            system_rows[len_slack:len_fne] = node_element_matrix_indices[slack_masses_from]

            # tne_mdF_dv
            system_cols[len_fne:len_tne] = branch_matrix_indices[slack_branches_to]
            system_rows[len_fne:len_tne] = node_element_matrix_indices[slack_masses_to]

            # ne_mdF_dm
            system_cols[len_tne:len_m_ne] = node_element_matrix_indices
            system_rows[len_tne:len_m_ne] = node_element_matrix_indices

            # slack_wdF_dm
            system_cols[len_m_ne:len_slack_wdF_dm] = get_w_like_vector(node_element_matrix_indices, len_fluid)
            system_rows[len_m_ne:len_slack_wdF_dm] = w_node_matrix_indices[slack_nodes_w]

            # n_mdF_dw
            system_cols[len_slack_wdF_dm:len_n_mdF_dw] = w_node_matrix_indices
            system_rows[len_slack_wdF_dm:len_n_mdF_dw] = w_node_matrix_indices

            # fn_wdF_dw
            system_cols[len_n_mdF_dw:len_fn_wdF_dw] = w_node_matrix_indices[fn_w]
            system_rows[len_n_mdF_dw:len_fn_wdF_dw] = w_node_matrix_indices[fn_w]

            # tn_wdF_dw
            system_cols[len_fn_wdF_dw:len_tn_wdF_dw] = w_node_matrix_indices[fn_w]
            system_rows[len_fn_wdF_dw:len_tn_wdF_dw] = w_node_matrix_indices[tn_w]

            # fn_wdF_dv
            system_cols[len_tn_wdF_dw:len_fn_wdF_dv] = get_w_like_vector(branch_matrix_indices, len_fluid)
            system_rows[len_tn_wdF_dw:len_fn_wdF_dv] = w_node_matrix_indices[fn_w]

            # tn_wdF_dv
            system_cols[len_fn_wdF_dv:len_tn_wdF_dv] = get_w_like_vector(branch_matrix_indices, len_fluid)
            system_rows[len_fn_wdF_dv:len_tn_wdF_dv] = w_node_matrix_indices[tn_w]

            # fn_rhodF_dw
            system_cols[len_tn_wdF_dv:len_fn_rhodF_dw] = w_node_matrix_indices[
                get_w_like_node_vector(vfn[not_slack_fn_branch_mask], len_fluid, len_n)]
            system_rows[len_tn_wdF_dv:len_fn_rhodF_dw] = \
                get_w_like_vector(fn[not_slack_fn_branch_mask], len_fluid)

            # tn_rhodF_dw
            system_cols[len_fn_rhodF_dw:len_tn_rhodF_dw] = w_node_matrix_indices[
                get_w_like_node_vector(vfn[not_slack_tn_branch_mask], len_fluid, len_n)]
            system_rows[len_fn_rhodF_dw:len_tn_rhodF_dw] = \
                get_w_like_vector(tn[not_slack_tn_branch_mask], len_fluid)

            # fslack_rhodF_dw
            system_cols[len_tn_rhodF_dw:len_fslack_rhodF_dw] = w_node_matrix_indices[
                get_w_like_node_vector(vfn[slack_branches_from], len_fluid, len_n)]
            system_rows[len_tn_rhodF_dw:len_fslack_rhodF_dw] = node_element_matrix_indices[
                get_w_like_vector(fn[slack_branches_from], len_fluid)]

            # tslack_rhodF_dw
            system_cols[len_fslack_rhodF_dw:] = w_node_matrix_indices[
                get_w_like_node_vector(vfn[slack_branches_to], len_fluid, len_n)]
            system_rows[len_fslack_rhodF_dw:] = node_element_matrix_indices[
                get_w_like_vector(tn[slack_branches_to], len_fluid)]

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
                                       shape=(len_n + len_b + extra, len_n + len_b + extra))

        else:
            data_order = np.lexsort([system_cols, system_rows])
            system_data = system_data[data_order]
            system_cols = system_cols[data_order]
            system_rows = system_rows[data_order]

            row_counter = np.zeros(len_b + len_n + extra + 1, dtype=np.int32)
            unique_rows, row_counts = _sum_by_group_sorted(system_rows, np.ones_like(system_rows))
            row_counter[unique_rows + 1] += row_counts
            ptr = row_counter.cumsum()
            system_matrix = csr_matrix((system_data, system_cols, ptr),
                                       shape=(len_n + len_b + extra, len_n + len_b + extra))
            net["_internal_data"]["hydraulic_data_sorting"] = data_order
            net["_internal_data"]["hydraulic_matrix"] = system_matrix
    else:
        data_order = net["_internal_data"]["hydraulic_data_sorting"]
        system_data = system_data[data_order]
        system_matrix = net["_internal_data"]["hydraulic_matrix"]
        system_matrix.data = system_data

    if not heat_mode:
        load_vector = np.empty(len_n + len_b + extra)
        load_vector[len_n:len_n + len_b] = branch_pit[:, net['_idx_branch']['LOAD_VEC_BRANCHES']]
        load_vector[:len_n] = node_pit[:, net['_idx_node']['LOAD']] * (-1)
        load_vector[len_n + len_b:len_n + len_b + len_ne] = \
            node_pit[node_element_pit[slack_element_mask, nej_col].astype(int), net['_idx_node']['LOAD']] * (-1)
        _, inv_from, count_from = np.unique(slack_branches_from, return_inverse=True, return_counts=True)
        _, inv_to, count_to = np.unique(slack_branches_to, return_inverse=True, return_counts=True)
        load_vector[len_n + len_b:len_n + len_b + len_ne] /= count[inv]
        fn_unique, fn_sums = _sum_by_group(fn, branch_pit[:, net['_idx_branch']['LOAD_VEC_NODES']])
        tn_unique, tn_sums = _sum_by_group(tn, branch_pit[:, net['_idx_branch']['LOAD_VEC_NODES']])
        fn_slack_unique, fn_slack_sums = _sum_by_group(slack_masses_from, branch_pit[
            slack_branches_from, net['_idx_branch']['LOAD_VEC_NODES']] / count_from[inv_from])
        tn_slack_unique, tn_slack_sums = _sum_by_group(slack_masses_to, branch_pit[
            slack_branches_to, net['_idx_branch']['LOAD_VEC_NODES']] / count_to[inv_to])
        load_vector[fn_unique] -= fn_sums
        load_vector[tn_unique] += tn_sums
        load_vector[slack_nodes] = 0
        load_vector[pc_matrix_indices] = 0
        load_vector[node_element_matrix_indices[fn_slack_unique]] -= fn_slack_sums
        load_vector[node_element_matrix_indices[tn_slack_unique]] += tn_slack_sums
        load_vector[node_element_matrix_indices] -= slack_mass
        if len_fluid and not first_iter:
            branch_deriv = np.abs(branch_pit[:, net['_idx_branch']['LOAD_VEC_NODES']])
            node_load = get_load_vec(net, node_pit) * (-1) * node_pit[:, w_n_col].T
            load_vector[w_node_matrix_indices] = node_load.flatten()
            node_w_out, load_branch = _sum_by_group(vfn, np.abs(branch_deriv))
            branch_from_load = load_branch * node_pit[node_w_out, :][:, w_n_col].T
            node_w_outw = np.tile(node_w_out, (len_fluid, 1)) + np.arange(len_fluid)[:, np.newaxis] * (len_n)
            branch_to_load = np.abs(branch_deriv) * node_pit[vfn, :][:, w_n_col].T
            node_w_inw, branch_to_load = _sum_by_group(tn_w, branch_to_load.flatten())
            load_vector[w_node_matrix_indices[node_w_outw.flatten()]] -= branch_from_load.flatten()
            load_vector[w_node_matrix_indices[node_w_inw.flatten()]] += branch_to_load.flatten()
            mass_slack = slack_wdF_dm * np.tile(slack_mass * count[inv], len_fluid)
            sl_nods, mass_slack = _sum_by_group(slack_nodes_w, mass_slack)
            load_vector[w_node_matrix_indices[sl_nods]] -= mass_slack
            s_nods = node_element_pit[~slack_element_mask, nej_col].astype(int)
            mass_load = node_element_pit[~slack_element_mask, min_col] * \
                        node_element_pit[~slack_element_mask, :][:, w_ne_col].T
            s_nods = np.tile(s_nods, (len_fluid, 1)) + np.arange(len_fluid)[:, np.newaxis] * (len_n)
            s_nods, mass_load = _sum_by_group(s_nods.flatten(), mass_load.flatten())
            load_vector[w_node_matrix_indices[s_nods]] += mass_load
            # print(node_pit[:, w_n_col], '*******', slack_w)
    else:
        tn_unique, tn_sums = _sum_by_group(tn, branch_pit[:, net['_idx_branch']['LOAD_VEC_NODES_T']])
        load_vector = np.zeros(len_n + len_b)
        load_vector[len(slack_nodes) + np.arange(0, len(tn_unique_der))] += tn_sums
        load_vector[len(slack_nodes) + np.arange(0, len(tn_unique_der))] -= tn_sums_der * node_pit[
            tn_unique_der, net['_idx_node']['TINIT']]
        load_vector[0:len(slack_nodes)] = 0.

        load_vector[len_n:] = branch_pit[:, net['_idx_branch']['LOAD_VEC_BRANCHES_T']]
    return system_matrix, load_vector


def get_slack_wdF_dm(net, node_pit, node_element_pit, slack_element_mask, w_n_col, w_ne_col,
                     number_slacks_per_junction, inverse_position):
    len_fluid = len(net['_fluid']) - 1
    w = np.zeros((len(slack_element_mask[slack_element_mask]), len_fluid))
    slack_mass = node_element_pit[slack_element_mask, net['_idx_node_element']['MINIT']]
    slack_nodes = node_element_pit[slack_element_mask, net['_idx_node_element']['JUNCTION']].astype(int)
    w_div_to = node_element_pit[slack_element_mask, :][:, w_ne_col] / \
               np.tile(number_slacks_per_junction[inverse_position][:, np.newaxis], len_fluid)
    w_div_from = node_pit[slack_nodes, :][:, w_n_col] / \
                 np.tile(number_slacks_per_junction[inverse_position][:, np.newaxis], len_fluid)
    w[slack_mass < 0, :] = w_div_to[slack_mass < 0]
    w[slack_mass >= 0, :] = -w_div_from[slack_mass >= 0]
    return w.T.flatten()


def get_slack_element_nodes_w(net, node_element_pit, slack_element_mask, number_of_fluids, number_of_nodes):
    slack_element_nodes = node_element_pit[slack_element_mask, net['_idx_node_element']['JUNCTION']].astype(int)
    slack_element_nodes_w = np.tile(slack_element_nodes, (number_of_fluids, 1)) + \
                            np.arange(number_of_fluids)[:, np.newaxis] * (number_of_nodes)
    slack_element_nodes_w = slack_element_nodes_w.flatten()
    return slack_element_nodes_w


def get_n_mdF_dw(net, node_pit, node_element_pit, slack_element_mask, number_of_fluids, number_of_nodes):
    load = get_w_like_vector(node_pit[:, net['_idx_node']['LOAD']], number_of_fluids)
    slack_mass = node_element_pit[slack_element_mask, net['_idx_node_element']['MINIT']]
    slack_mass = get_w_like_vector(slack_mass, number_of_fluids)
    slack_nodes = node_element_pit[slack_element_mask, net['_idx_node_element']['JUNCTION']].astype(int)
    slack_nodes = get_w_like_node_vector(slack_nodes, number_of_fluids, number_of_nodes)
    mdF_dw = np.zeros(len(load))
    mdF_dw[load >= 0] += load[load >= 0]
    mdF_dw[slack_nodes.flatten()[slack_mass >= 0]] += slack_mass[slack_mass >= 0]
    return mdF_dw


def get_n_wdF_dw(net, branch_pit):
    der_rho_same = get_lookup(net, 'branch', 'deriv_rho_same')[:-1]
    v_a = np.abs(branch_pit[:, net['_idx_branch']['VINIT']] * branch_pit[:, net['_idx_branch']['AREA']])
    jac_deriv_rho = branch_pit[:, der_rho_same]
    wdF_dw = v_a[:, np.newaxis] * jac_deriv_rho
    return wdF_dw.flatten()


def get_fn_rhodF_dw(net, branch_pit, not_slack_from_branches):
    rhodF_dw = get_n_rhodF_dw(net, branch_pit, not_slack_from_branches)
    return rhodF_dw


def get_tn_rhodF_dw(net, branch_pit, not_slack_to_branches):
    rhodF_dw = get_n_rhodF_dw(net, branch_pit, not_slack_to_branches)
    return rhodF_dw


def get_fslack_rhodF_dw(net, branch_pit, slack_from_branches):
    rhodF_dw = get_n_rhodF_dw(net, branch_pit, slack_from_branches)
    return rhodF_dw


def get_tslack_rhodF_dw(net, branch_pit, slack_to_branches):
    rhodF_dw = get_n_rhodF_dw(net, branch_pit, slack_to_branches)
    return rhodF_dw


def get_n_rhodF_dw(net, branch_pit, branches):
    der_rho_diff = get_lookup(net, 'branch', 'deriv_rho_diff')[:-1]
    v_a = branch_pit[branches, net['_idx_branch']['VINIT']] * branch_pit[branches, net['_idx_branch']['AREA']]
    jac_deriv_rho = branch_pit[:, der_rho_diff][branches, :]
    rhodF_dw = v_a[:, np.newaxis] * jac_deriv_rho
    return rhodF_dw.flatten()


def get_wdF_dv(net, node_pit, branch_pit, w_n_col):
    load_vec = branch_pit[:, net['_idx_branch']['JAC_DERIV_DV_NODE']] * \
               np.sign(branch_pit[:, net['_idx_branch']['LOAD_VEC_NODES']])
    v_from_b = branch_pit[:, net['_idx_branch']['V_FROM_NODE']].astype(int)
    w = node_pit[:, w_n_col][v_from_b, :]
    wdF_dv = load_vec[:, np.newaxis] * w
    return wdF_dv.flatten()


####################### auxiliaries ###############################

def get_w_like_node_vector(nodes, number_of_fluids, number_of_nodes):
    nodes_w = np.tile(nodes, (number_of_fluids, 1)) + \
              np.arange(number_of_fluids)[:, np.newaxis] * (number_of_nodes)
    nodes_w = nodes_w.flatten()
    return nodes_w


def get_w_like_vector(entry, number_of_fluids):
    entry_w = np.tile(entry, number_of_fluids)
    entry_w = entry_w.flatten()
    return entry_w


def get_load_vec(net, node_pit):
    l = node_pit[:, net['_idx_node']['LOAD']]
    load = np.zeros(len(l))
    load[l >= 0] = l[l >= 0]
    return load
