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


class ThrottleValve(PressureControlComponent):
    """

    """
    JUNCTS = 0
    PCTRL = 1
    MAXV = 2

    internal_cols = 3

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
        pc_ctrl = net[cls.table_name()].control_active.values
        tbl = net[cls.table_name()].loc[pc_ctrl, :]
        pc_array = np.zeros(shape=(len(tbl), cls.internal_cols), dtype=np.float64)
        pc_array[:, cls.PCTRL] = tbl['controlled_p_bar'].values[pc_ctrl]
        pc_array[:, cls.JUNCTS] = tbl['controlled_junction'].values[pc_ctrl]
        pc_array[:, cls.MAXV] = tbl['max_mdot_kg_per_s'].values[pc_ctrl]
        component_pits[cls.table_name()] = pc_array

    @classmethod
    def adaption_before_derivatives_hydraulic(cls, net, branch_pit, node_pit, idx_lookups, options):
        f, t = idx_lookups[cls.table_name()]
        pc_pit = branch_pit[f:t, :]
        pc_array = get_component_array(net, cls.table_name())
        mask_maxv = pc_array[:, cls.MAXV] < pc_pit[:, MDOTINIT]
        pc_pit[mask_maxv, MDOTINIT] = pc_array[mask_maxv, cls.MAXV]

    @classmethod
    def adaption_after_derivatives_hydraulic(cls, net, branch_pit, node_pit, idx_lookups, options):
        junction_idx_lookups = get_lookup(net, "node", "index_active_hydraulics")[
            cls.get_connected_node_type().table_name()]
        f, t = idx_lookups[cls.table_name()]

        pc_array = get_component_array(net, cls.table_name())

        pc_pit = branch_pit[f:t, :]

        to_nodes = pc_pit[:, TO_NODE].astype(np.int32)
        from_nodes = pc_pit[:, FROM_NODE].astype(np.int32)
        p_to = node_pit[to_nodes, PINIT]
        p_from = node_pit[from_nodes, PINIT]
        mf = pc_pit[:, MDOTINIT]

        # pressure p_to lower than p_from
        pc_ctrl = (p_to < p_from) | np.isclose(p_to, p_from)
        index_pc = junction_idx_lookups[pc_array[pc_ctrl, cls.JUNCTS].astype(np.int32)]

        pc_pit[pc_ctrl, ACTIVE] = True
        pc_pit[pc_ctrl, BRANCH_TYPE] = PC_BRANCH
        pc_pit[pc_ctrl, JAC_DERIV_DP] = 0
        pc_pit[pc_ctrl, JAC_DERIV_DP1] = 0
        pc_pit[pc_ctrl, JAC_DERIV_DM] = 0
        pc_pit[pc_ctrl, LOAD_VEC_BRANCHES] = 0.
        node_pit[index_pc, NODE_TYPE] = PC_NODE

        # pressure p_ctrl higher than p_from
        pc_higher = p_to > p_from
        index_pc = junction_idx_lookups[pc_array[pc_higher, cls.JUNCTS].astype(np.int32)]

        pc_pit[pc_ctrl, ACTIVE] = True
        pc_pit[pc_higher, BRANCH_TYPE] = 0
        node_pit[index_pc, NODE_TYPE] = L

        # mass flow is reversed
        pc_reverse = (mf < 0)  & ~np.isclose(mf, 0)
        pc_pit[pc_reverse, ACTIVE] = False

        # maximum mass flow
        mask_maxv = (pc_array[:, cls.MAXV] < pc_pit[:, MDOTINIT]) | np.isclose(pc_array[:, cls.MAXV], pc_pit[:, MDOTINIT])

        index_pc = junction_idx_lookups[pc_array[mask_maxv, cls.JUNCTS].astype(np.int32)]

        pc_pit[mask_maxv, BRANCH_TYPE] = 0
        pc_pit[mask_maxv, JAC_DERIV_DP] = 0
        pc_pit[mask_maxv, JAC_DERIV_DP1] = 0
        pc_pit[mask_maxv, JAC_DERIV_DM] = 1
        pc_pit[mask_maxv, LOAD_VEC_BRANCHES] = 0
        node_pit[index_pc, NODE_TYPE] = L
