# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype
from pandapipes.component_models.abstract_models import BranchWZeroLengthComponent
from pandapipes.idx_branch import D, AREA, PL, TL, \
    JAC_DERIV_DP, JAC_DERIV_DP1, JAC_DERIV_DV, BRANCH_TYPE
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
    def create_pit_node_entries(cls, net, node_pit, node_name):
        pcs = net[cls.table_name()]
        juncts = pcs['controlled_junction'].values[pcs.in_service]
        press = pcs['controlled_p_bar'].values[pcs.in_service]
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
        :param pump_pit:
        :type pump_pit:
        :param internal_pipe_number:
        :type internal_pipe_number:
        :return: No Output.
        """
        pc_pit = super().create_pit_branch_entries(net, pc_pit, node_name)
        pc_pit[:, D] = 0.1
        pc_pit[:, AREA] = pc_pit[:, D] ** 2 * np.pi / 4
        pc_pit[:, BRANCH_TYPE] = PC

    @classmethod
    def calculate_derivatives_hydraulic(cls, net, branch_pit, node_pit, idx_lookups, options):
        press_pit = super().calculate_derivatives_hydraulic(net, branch_pit, node_pit, idx_lookups, options)
        press_pit[:, JAC_DERIV_DP] = 0
        press_pit[:, JAC_DERIV_DP1] = 0
        press_pit[:, JAC_DERIV_DV] = 0

    @classmethod
    def calculate_pressure_lift(cls, net, pc_pit, node_pit):
        """

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param pump_pit:
        :type pump_pit:
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
        :param pump_pit:
        :type pump_pit:
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
        :return: No Output.
        """
        placement_table, pc_pit, res_table = super().extract_results(net, options, node_name)

        node_pit = net["_active_pit"]["node"]
        node_active_idx_lookup = get_lookup(net, "node", "index_active")[node_name]
        junction_idx_lookup = get_lookup(net, "node", "index")[node_name]
        from_junction_nodes = node_active_idx_lookup[junction_idx_lookup[
            net[cls.table_name()]["from_junction"].values[placement_table]]]
        to_junction_nodes = node_active_idx_lookup[junction_idx_lookup[
            net[cls.table_name()]["to_junction"].values[placement_table]]]

        res_table['deltap_bar'].values[placement_table] = node_pit[to_junction_nodes , PINIT] - \
                                                          node_pit[from_junction_nodes, PINIT]

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
        return ['deltap_bar'], True
