# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype
from pandapipes.component_models.abstract_models import BranchWZeroLengthComponent, get_fluid, \
    TINIT_NODE
from pandapipes.idx_branch import D, AREA, PL, TL, \
    JAC_DERIV_DP, JAC_DERIV_DP1, JAC_DERIV_DV, BRANCH_TYPE, FROM_NODE, TO_NODE, VINIT, \
    LOAD_VEC_NODES, ELEMENT_IDX, LOSS_COEFFICIENT as LC
from pandapipes.idx_node import PINIT, NODE_TYPE, PC
from pandapipes.internals_toolbox import _sum_by_group
from pandapipes.pipeflow_setup import get_lookup, get_net_option


class PressureControlComponent(BranchWZeroLengthComponent):
    """

    """

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
    def create_pit_node_entries(cls, net, node_pit, node_name):
        pcs = net[cls.table_name()]
        controlled = pcs.in_service & pcs.control_active
        juncts = pcs['controlled_junction'].values[controlled]
        press = pcs['controlled_p_bar'].values[controlled]
        junction_idx_lookups = get_lookup(net, "node", "index")[node_name]
        index_pc = junction_idx_lookups[juncts]
        node_pit[index_pc, NODE_TYPE] = PC
        node_pit[index_pc, PINIT] = press

    @classmethod
    def create_pit_branch_entries(cls, net, pc_pit, node_name):
        """
        Function which creates pit branch entries with a specific table.
        :param net: The pandapipes network
        :type net: pandapipesNet
        :param pc_pit:
        :type pc_pit:
        :param node_name:
        :type node_name:
        :return: No Output.
        """
        pc_pit = super().create_pit_branch_entries(net, pc_pit, node_name)
        pc_pit[:, D] = 0.1
        pc_pit[:, AREA] = pc_pit[:, D] ** 2 * np.pi / 4
        pc_pit[net[cls.table_name()].control_active.values, BRANCH_TYPE] = PC
        pc_pit[:, LC] = net[cls.table_name()].loss_coefficient.values

    @classmethod
    def calculate_derivatives_hydraulic(cls, net, branch_pit, node_pit, idx_lookups, options):
        press_pit = super().calculate_derivatives_hydraulic(net, branch_pit, node_pit, idx_lookups,
                                                            options)
        pc_branch = press_pit[:, BRANCH_TYPE] == PC
        press_pit[pc_branch, JAC_DERIV_DP] = 0
        press_pit[pc_branch, JAC_DERIV_DP1] = 0
        press_pit[pc_branch, JAC_DERIV_DV] = 0

    @classmethod
    def calculate_pressure_lift(cls, net, pc_pit, node_pit):
        """

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param pc_pit:
        :type pc_pit:
        :param node_pit:
        :type node_pit:
        :return: power stroke
        :rtype: float
        """
        pc_pit[:, PL] = 0

    @classmethod
    def calculate_temperature_lift(cls, net, pc_pit, node_pit):
        """

        :param net:
        :type net:
        :param pc_pit:
        :type pc_pit:
        :param node_pit:
        :type node_pit:
        :return:
        :rtype:
        """
        pc_pit[:, TL] = 0

    @classmethod
    def extract_results(cls, net, options, node_name):
        """
        Function that extracts certain results.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param options:
        :type options:
        :param node_name:
        :type node_name:
        :return: No Output.
        """
        placement_table, pc_pit, res_table = cls.prepare_result_tables(net, options, node_name)

        node_pit = net["_active_pit"]["node"]
        node_active_idx_lookup = get_lookup(net, "node", "index_active")[node_name]
        junction_idx_lookup = get_lookup(net, "node", "index")[node_name]
        from_junction_nodes = node_active_idx_lookup[junction_idx_lookup[
            net[cls.table_name()]["from_junction"].values[placement_table]]]
        to_junction_nodes = node_active_idx_lookup[junction_idx_lookup[
            net[cls.table_name()]["to_junction"].values[placement_table]]]

        res_table['deltap_bar'].values[placement_table] = node_pit[to_junction_nodes, PINIT] - \
            node_pit[from_junction_nodes, PINIT]

        from_nodes = pc_pit[:, FROM_NODE].astype(np.int32)
        to_nodes = pc_pit[:, TO_NODE].astype(np.int32)

        t0 = node_pit[from_nodes, TINIT_NODE]
        t1 = node_pit[to_nodes, TINIT_NODE]
        mf = pc_pit[:, LOAD_VEC_NODES]
        vf = pc_pit[:, LOAD_VEC_NODES] / get_fluid(net).get_density((t0 + t1) / 2)

        idx_active = pc_pit[:, ELEMENT_IDX]
        idx_sort, mf_sum, vf_sum, internal_pipes = \
            _sum_by_group(idx_active, mf, vf, np.ones_like(idx_active))
        res_table["mdot_to_kg_per_s"].values[placement_table] = -mf_sum / internal_pipes
        res_table["mdot_from_kg_per_s"].values[placement_table] = mf_sum / internal_pipes
        res_table["vdot_norm_m3_per_s"].values[placement_table] = vf_sum / internal_pipes

    @classmethod
    def get_component_input(cls):
        """

        Get component input.

        :return:
        :rtype:
        """
        return [("name", dtype(object)),
                ("from_junction", "u4"),
                ("to_junction", "u4"),
                ("controlled_junction", "u4"),
                ("controlled_p_bar", "f8"),
                ("control_active", "bool"),
                ("loss_coefficient", "f8"),
                ("in_service", 'bool'),
                ("type", dtype(object))]

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
        return ["deltap_bar", "mdot_from_kg_per_s", "mdot_to_kg_per_s", "vdot_norm_m3_per_s"], True
