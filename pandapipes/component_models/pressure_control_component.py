# Copyright (c) 2020-2022 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype

from pandapipes.component_models.abstract_models import BranchWZeroLengthComponent, get_fluid
from pandapipes.idx_branch import D, AREA, PL, TL, \
    JAC_DERIV_DP, JAC_DERIV_DP1, JAC_DERIV_DV, BRANCH_TYPE, LOSS_COEFFICIENT as LC
from pandapipes.idx_node import PINIT, NODE_TYPE, PC
from pandapipes.pipeflow_setup import get_lookup


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

        placement_table, res_table, pc_pit, node_pit = super().extract_results(net, options, node_name)

        node_active_idx_lookup = get_lookup(net, "node", "index_active")[node_name]
        junction_idx_lookup = get_lookup(net, "node", "index")[node_name]
        from_junction_nodes = node_active_idx_lookup[junction_idx_lookup[
            net[cls.table_name()]["from_junction"].values[placement_table]]]
        to_junction_nodes = node_active_idx_lookup[junction_idx_lookup[
            net[cls.table_name()]["to_junction"].values[placement_table]]]

        p_to = node_pit[to_junction_nodes, PINIT]
        p_from = node_pit[from_junction_nodes, PINIT]
        res_table['deltap_bar'].values[placement_table] = p_to - p_from

        return placement_table, res_table, pc_pit

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
        if get_fluid(net).is_gas:
            output = ["v_from_m_per_s", "v_to_m_per_s", "p_from_bar", "p_to_bar",
                      "t_from_k", "t_to_k", "mdot_from_kg_per_s", "mdot_to_kg_per_s",
                      "vdot_norm_m3_per_s", "normfactor_from", "normfactor_to"]
        else:
            output = ["v_mean_m_per_s", "p_from_bar", "p_to_bar", "t_from_k", "t_to_k",
                      "mdot_from_kg_per_s", "mdot_to_kg_per_s", "vdot_norm_m3_per_s"]
        output += ["deltap_bar"]
        return output, True
