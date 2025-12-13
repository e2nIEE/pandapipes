# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from pandapipes.component_models.component_toolbox import get_component_array
from pandapipes.component_models.pressure_control_component import PressureControlComponent
from pandapipes.idx_branch import BRANCH_TYPE, LOAD_VEC_BRANCHES, JAC_DERIV_DM, JAC_DERIV_DP, JAC_DERIV_DP1, ACTIVE, \
    TO_NODE, FROM_NODE, MDOTINIT
from pandapipes.idx_node import PINIT, NODE_TYPE, PC as PC_NODE, L
from pandapipes.idx_branch import PC as PC_BRANCH
from pandapipes.pf.pipeflow_setup import get_lookup
from pandapipes.pf.result_extraction import extract_branch_results_without_internals
from pandapipes.properties.fluids import get_fluid


class ThrottleValve(PressureControlComponent):
    """

    """
    MODE = 0
    PINIT = 1
    JUNCTS = 2
    MAXV = 3
    PC = 4

    internal_cols = 5

    CLOSED = 1
    PCTRL = 2
    OPEN = 3
    MAXVEXT = 4

    @classmethod
    def table_name(cls):
        return "throttle_valve"

    @classmethod
    def create_component_array(cls, net, component_pits):
        """
        Function which creates an internal array of the component in analogy to the pit, but with
        component specific entries, that are not needed in the pit.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param component_pits: dictionary of component specific arrays
        :type component_pits: dict
        :return:
        :rtype:
        """
        tbl = net[cls.table_name()]
        pc_array = np.zeros(shape=(len(tbl), cls.internal_cols), dtype=np.float64)
        pc_array[net[cls.table_name()].control_active.values, cls.PC] = cls.PC
        pc_branch = pc_array[:, cls.PC] == cls.PC
        pc_array[pc_branch, cls.MODE] = cls.PCTRL
        pc_array[pc_branch, cls.PINIT] = tbl['controlled_p_bar'].values[pc_branch]
        pc_array[pc_branch, cls.JUNCTS] = tbl['controlled_junction'].values[pc_branch]
        pc_array[pc_branch, cls.MAXV] = tbl['max_mdot_kg_per_s'].values[pc_branch]
        component_pits[cls.table_name()] = pc_array

    @classmethod
    def adaption_before_derivatives_hydraulic(cls, net, branch_pit, node_pit, idx_lookups, options):
        f, t = idx_lookups[cls.table_name()]
        pc_pit = branch_pit[f:t, :]
        pc_array = get_component_array(net, cls.table_name())
        pc_branch = pc_array[:, cls.PC] == cls.PC
        pc_pit[pc_branch, BRANCH_TYPE] = PC_BRANCH
        mask_maxv = (pc_array[:, cls.MODE] == cls.MAXVEXT)
        if any(mask_maxv):
            pc_pit[mask_maxv, MDOTINIT] = pc_array[mask_maxv, cls.MAXV]

    @classmethod
    def adaption_after_derivatives_hydraulic(cls, net, branch_pit, node_pit, idx_lookups, options):
        junction_idx_lookups = get_lookup(net, "node", "index_active_hydraulics")[
            cls.get_connected_node_type().table_name()]
        f, t = idx_lookups[cls.table_name()]
        pc_pit = branch_pit[f:t, :]
        pc_array = get_component_array(net, cls.table_name())
        pc_branch = pc_pit[:, BRANCH_TYPE] == PC_BRANCH
        mask_ctrl = (pc_array[:, cls.MODE] == cls.PCTRL) & pc_branch
        mask_open = (pc_array[:, cls.MODE] == cls.OPEN) & pc_branch
        mask_maxv = (pc_array[:, cls.MODE] == cls.MAXVEXT) & pc_branch

        if any(mask_ctrl):
            pc_pit[mask_ctrl, BRANCH_TYPE] = PC_BRANCH
            pc_pit[mask_ctrl, JAC_DERIV_DP] = 0
            pc_pit[mask_ctrl, JAC_DERIV_DP1] = 0
            pc_pit[mask_ctrl, JAC_DERIV_DM] = 0
            pc_pit[mask_ctrl, LOAD_VEC_BRANCHES] = 0.

            index_pc = junction_idx_lookups[pc_array[mask_ctrl, cls.JUNCTS].astype(np.int32)]
            node_pit[index_pc, NODE_TYPE] = PC_NODE

        if any(mask_maxv):
            pc_pit[mask_maxv, JAC_DERIV_DP] = 0
            pc_pit[mask_maxv, JAC_DERIV_DP1] = 0
            pc_pit[mask_maxv, JAC_DERIV_DM] = 1
            pc_pit[mask_maxv, LOAD_VEC_BRANCHES] = 0
            index_pc = junction_idx_lookups[pc_array[mask_maxv, cls.JUNCTS].astype(np.int32)]
            node_pit[index_pc, NODE_TYPE] = L
            pc_pit[mask_maxv, BRANCH_TYPE] = 0

        if any(mask_open):
            pc_pit[mask_open, BRANCH_TYPE] = 0
            index_pc = junction_idx_lookups[pc_array[mask_open, cls.JUNCTS].astype(np.int32)]
            node_pit[index_pc, NODE_TYPE] = L

    @classmethod
    def rerun_hydraulics(cls, net):
        rerun = False
        branch_pit = net["_active_pit"]["branch"]
        node_pit = net["_active_pit"]["node"]
        branch_lookups = get_lookup(net, "branch", "from_to_active_hydraulics")

        f, t = branch_lookups[cls.table_name()]
        pc_pit = branch_pit[f:t, :]
        pc_array = get_component_array(net, cls.table_name())

        pc_branch = pc_pit[:, BRANCH_TYPE] == PC_BRANCH
        to_nodes = pc_pit[:, TO_NODE].astype(np.int32)
        from_nodes = pc_pit[:, FROM_NODE].astype(np.int32)
        p_to = node_pit[to_nodes, PINIT]
        p_from = node_pit[from_nodes, PINIT]
        mf = pc_pit[:, MDOTINIT]

        is_closed = pc_array[:, cls.MODE] == cls.CLOSED
        is_open = pc_array[:, cls.MODE] == cls.OPEN
        is_controlled = pc_array[:, cls.MODE] == cls.PCTRL
        is_maxv = pc_array[:, cls.MODE] == cls.MAXVEXT

        open = (p_to > p_from) & ~np.isclose(p_to, p_from) & (is_controlled | is_open)
        mask_open = open & pc_branch

        if any(mask_open):
            pc_array[mask_open, cls.MODE] = cls.OPEN
            pc_pit[mask_open, ACTIVE] = True
            rerun = True

        reverse = (mf < 0) & (is_controlled | is_closed) & ~np.isclose(mf, 0)
        mask_reverse = reverse & pc_branch

        if any(mask_reverse):
            junction_idx_lookups = get_lookup(net, "node", "index_active_hydraulics")[
                cls.get_connected_node_type().table_name()]
            pc_array[mask_reverse, cls.MODE] = cls.CLOSED
            pc_pit[mask_reverse, ACTIVE] = False
            index_pc = junction_idx_lookups[pc_array[mask_reverse, cls.JUNCTS].astype(np.int32)]
            node_pit[index_pc, NODE_TYPE] = L
            rerun = True

        mask_non_nan = ~np.isnan(pc_array[:, cls.MAXV])
        mask_non_nan &= (is_controlled | is_maxv)
        if np.any(mask_non_nan):
            mask_maxv = pc_array[:, cls.MAXV] < pc_pit[:, MDOTINIT]
            pc_pit[mask_non_nan & mask_maxv, ACTIVE] = True
            pc_array[mask_non_nan & mask_maxv, cls.MODE] = cls.MAXVEXT
            if any(mask_non_nan & mask_maxv):
                rerun = True

        return rerun
