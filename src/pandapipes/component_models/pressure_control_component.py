# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype

from pandapipes.component_models.abstract_models.branch_wzerolength_models import BranchWZeroLengthComponent
from pandapipes.component_models.component_toolbox import get_component_array
from pandapipes.component_models.junction_component import Junction
from pandapipes.idx_branch import D, AREA, BRANCH_TYPE, LOSS_COEFFICIENT as LC, LOAD_VEC_BRANCHES, JAC_DERIV_DM, \
    JAC_DERIV_DP, JAC_DERIV_DP1, ACTIVE, TO_NODE, FROM_NODE, MDOTINIT
from pandapipes.idx_node import PINIT, NODE_TYPE, PC, L
from pandapipes.pf.pipeflow_setup import get_lookup, identify_active_nodes_branches
from pandapipes.pf.result_extraction import extract_branch_results_without_internals
from pandapipes.properties.fluids import get_fluid


class PressureControlComponent(BranchWZeroLengthComponent):
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
        return "press_control"

    @classmethod
    def active_identifier(cls):
        return "in_service"

    @classmethod
    def from_to_node_cols(cls):
        return "from_junction", "to_junction"

    @classmethod
    def get_connected_node_type(cls):
        return Junction

    @classmethod
    def create_pit_node_entries(cls, net, node_pit):
        pcs = net[cls.table_name()]
        controlled = pcs.in_service & pcs.control_active
        juncts = pcs['controlled_junction'].values[controlled]
        press = pcs['controlled_p_bar'].values[controlled]
        junction_idx_lookups = get_lookup(net, "node", "index")[cls.get_connected_node_type().table_name()]
        index_pc = junction_idx_lookups[juncts]
        node_pit[index_pc, NODE_TYPE] = PC
        node_pit[index_pc, PINIT] = press

    @classmethod
    def create_pit_branch_entries(cls, net, branch_pit):
        """
        Function which creates pit branch entries with a specific table.
        :param net: The pandapipes network
        :type net: pandapipesNet
        :param branch_pit:
        :type branch_pit:
        :return: No Output.
        """
        pc_pit = super().create_pit_branch_entries(net, branch_pit)
        pc_pit[:, D] = 0.1
        pc_pit[:, AREA] = pc_pit[:, D] ** 2 * np.pi / 4
        pc_pit[net[cls.table_name()].control_active.values, BRANCH_TYPE] = PC
        pc_pit[:, LC] = net[cls.table_name()].loss_coefficient.values

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
        pc_array[net[cls.table_name()].control_active.values, cls.PC] = PC
        pc_branch = pc_array[net[cls.table_name()].control_active.values, cls.PC] == PC
        pc_array[pc_branch, cls.MODE] = cls.PCTRL
        pc_array[pc_branch, cls.PINIT] = tbl['controlled_p_bar'].values
        pc_array[pc_branch, cls.JUNCTS] = tbl['controlled_junction'].values
        pc_array[pc_branch, cls.MAXV] = tbl['max_mdot_kg_per_s'].values
        component_pits[cls.table_name()] = pc_array

    @classmethod
    def adaption_before_derivatives_hydraulic(cls, net, branch_pit, node_pit, idx_lookups, options):
        f, t = idx_lookups[cls.table_name()]
        pc_pit = branch_pit[f:t, :]
        pc_array = get_component_array(net, cls.table_name())
        pc_branch = pc_array[:, cls.PC] == PC
        pc_pit[pc_branch, BRANCH_TYPE] = PC
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
        pc_branch = pc_pit[:, BRANCH_TYPE] == PC
        mask_ctrl = (pc_array[:, cls.MODE] == cls.PCTRL) & pc_branch
        mask_open = (pc_array[:, cls.MODE] == cls.OPEN) & pc_branch
        mask_maxv = (pc_array[:, cls.MODE] == cls.MAXVEXT) & pc_branch

        if any(mask_ctrl):
            pc_pit[mask_ctrl, BRANCH_TYPE] = PC
            pc_pit[mask_ctrl, JAC_DERIV_DP] = 0
            pc_pit[mask_ctrl, JAC_DERIV_DP1] = 0
            pc_pit[mask_ctrl, JAC_DERIV_DM] = 0
            pc_pit[mask_ctrl, LOAD_VEC_BRANCHES] = 0.

            index_pc = junction_idx_lookups[pc_array[mask_ctrl, cls.JUNCTS].astype(np.int32)]
            node_pit[index_pc, NODE_TYPE] = PC

        if any(mask_maxv):
            pc_pit[mask_maxv, JAC_DERIV_DP] = 0
            pc_pit[mask_maxv, JAC_DERIV_DP1] = 0
            pc_pit[mask_maxv, JAC_DERIV_DM] = 1
            pc_pit[mask_maxv, LOAD_VEC_BRANCHES] = 0
            index_pc = junction_idx_lookups[pc_array[mask_maxv, cls.JUNCTS].astype(np.int32)]
            node_pit[index_pc, NODE_TYPE] = L
            pc_pit[mask_maxv, BRANCH_TYPE] = L

        if any(mask_open):
            pc_pit[mask_open, BRANCH_TYPE] = L
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

        pc_branch = pc_pit[:, BRANCH_TYPE] == PC
        to_nodes = pc_pit[:, TO_NODE].astype(np.int32)
        from_nodes = pc_pit[:, FROM_NODE].astype(np.int32)
        p_to = node_pit[to_nodes, PINIT]
        p_from = node_pit[from_nodes, PINIT]
        mf = pc_pit[:, MDOTINIT]

        is_closed = pc_array[:, cls.MODE] == cls.CLOSED
        is_open = pc_array[:, cls.MODE] == cls.OPEN
        is_controlled = pc_array[:, cls.MODE] == cls.PCTRL
        is_maxv = pc_array[:, cls.MODE] == cls.MAXVEXT

        #ctrlable = (p_to <= p_ctrl) & (is_closed | is_open)
        #mask_ctrl = ctrlable & pc_branch

        #if any(mask_ctrl):
        #    pc_array[mask_ctrl, cls.MODE] = cls.PCTRL
        #    pc_pit[mask_ctrl, ACTIVE] = True
        #    rerun = True

        open = (p_to > p_from) & ~np.isclose(p_to, p_from)  & (is_controlled | is_open)#(is_controlled | ~is_closed)
        mask_open = open & pc_branch

        if any(mask_open):
            pc_array[mask_open, cls.MODE] = cls.OPEN
            pc_pit[mask_open, ACTIVE] = True
            rerun = True

        reverse = (mf < 0) & (is_controlled | is_closed) & ~np.isclose(mf, 0) #(is_controlled | is_open)
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
        #maxv_isclose = pc_array[:, cls.MODE] == cls.CLOSED
        #maxv_isopen = pc_array[:, cls.MODE] == cls.OPEN
        mask_non_nan &= (is_controlled | is_maxv)#~(maxv_isclose | maxv_isopen )
        if np.any(mask_non_nan):
            mask_maxv = pc_array[:, cls.MAXV] < pc_pit[:, MDOTINIT]
            pc_pit[mask_non_nan & mask_maxv, ACTIVE] = True
            pc_array[mask_non_nan & mask_maxv, cls.MODE] = cls.MAXVEXT
            if any(mask_non_nan & mask_maxv):
                rerun =True

        return rerun

    @classmethod
    def extract_results(cls, net, options, branch_results, mode):
        """
        Function that extracts certain results.

        :param mode:
        :type mode:
        :param branch_results:
        :type branch_results:
        :param net: The pandapipes network
        :type net: pandapipesNet
        :param options:
        :type options:
        :return: No Output.
        """
        required_results_hyd = [("p_from_bar", "p_from"), ("p_to_bar", "p_to"), ("mdot_from_kg_per_s", "mf_from"),
                                ("mdot_to_kg_per_s", "mf_to"), ("vdot_norm_m3_per_s", "vf")]
        required_results_ht = [("t_from_k", "temp_from"), ("t_to_k", "temp_to")]

        if get_fluid(net).is_gas:
            required_results_hyd.extend(
                [("v_from_m_per_s", "v_gas_from"), ("v_to_m_per_s", "v_gas_to"), ("normfactor_from", "normfactor_from"),
                 ("normfactor_to", "normfactor_to")])
        else:
            required_results_hyd.extend([("v_mean_m_per_s", "v_mps")])

        extract_branch_results_without_internals(net, branch_results, required_results_hyd, required_results_ht,
                                                 cls.table_name(), mode)

        res_table = net["res_" + cls.table_name()]
        f, t = get_lookup(net, "branch", "from_to")[cls.table_name()]
        p_to = branch_results["p_to"][f:t]
        p_from = branch_results["p_from"][f:t]
        res_table["deltap_bar"].values[:] = p_to - p_from

    @classmethod
    def get_component_input(cls):
        """

        Get component input.

        :return:
        :rtype:
        """
        return [("name", dtype(object)), ("from_junction", "u4"), ("to_junction", "u4"), ("controlled_junction", "u4"),
                ("controlled_p_bar", "f8"), ("control_active", "bool"), ("loss_coefficient", "f8"),
                ("max_mdot_kg_per_s", "f8"), ("in_service", 'bool'), ("type", dtype(object))]

    @classmethod
    def get_result_table(cls, net):
        """

        Gets the result table.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :return: (columns, all_float) - the column names and whether they are all float type. Only
                if False, returns columns as tuples also specifying the dtypes
        :rtype: (list, bool)
        """
        if get_fluid(net).is_gas:
            output = ["v_from_m_per_s", "v_to_m_per_s", "p_from_bar", "p_to_bar", "t_from_k", "t_to_k",
                      "mdot_from_kg_per_s", "mdot_to_kg_per_s", "vdot_norm_m3_per_s", "normfactor_from",
                      "normfactor_to"]
        else:
            output = ["v_mean_m_per_s", "p_from_bar", "p_to_bar", "t_from_k", "t_to_k", "mdot_from_kg_per_s",
                      "mdot_to_kg_per_s", "vdot_norm_m3_per_s"]
        output += ["deltap_bar"]
        return output, True
