# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype

from pandapipes.component_models.component_toolbox import standard_branch_wo_internals_result_lookup
from pandapipes.component_models.abstract_models.branch_w_internals_models import \
    BranchWInternalsComponent
from pandapipes.component_models.junction_component import Junction
from pandapipes.idx_branch import LENGTH, K, TEXT, ALPHA
from pandapipes.pf.result_extraction import extract_branch_results_without_internals
from pandapipes.properties.fluids import get_fluid
from pandapipes.pf.pipeflow_setup import add_table_lookup


class Valve(BranchWInternalsComponent):
    """
    Valves are branch elements that can separate two junctions.
    They have a length of 0, but can introduce a lumped pressure loss.
    """

    @classmethod
    def from_to_node_cols(cls):
        return "junction", "element"

    @classmethod
    def table_name(cls):
        return "valve"

    @classmethod
    def active_identifier(cls):
        return "opened"

    @classmethod
    def get_connected_node_type(cls):
        return Junction

    @classmethod
    def internal_node_name(cls):
        return "valve_nodes"

    @classmethod
    def get_internal_node_number(cls, net):
        """

        :param net: The pandapipes network
        :type net: pandapipesNet
        :return:
        :rtype:
        """

        int_nodes = np.empty(len(net[cls.table_name()]), dtype=np.int32)
        int_nodes[net[cls.table_name()]['et'] == 'p'] = 1
        int_nodes[net[cls.table_name()]['et'] == 'j'] = 0
        return int_nodes

    @classmethod
    def get_internal_branch_number(cls, net):
        """

        :param net: The pandapipes network
        :type net: pandapipesNet
        :return:
        :rtype:
        """

        return np.ones(len(net[cls.table_name()]), dtype=np.int32)

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
        valve_pit = super().create_pit_branch_entries(net, branch_pit)
        valve_pit[:, LENGTH] = 0
        valve_pit[:, K] = 1000
        valve_pit[:, TEXT] = 293.15
        valve_pit[:, ALPHA] = 0

    @classmethod
    def get_component_input(cls):
        """

        :return:
        :rtype:
        """
        return [("name", dtype(object)),
                ("junction", "i8"),
                ("element", "i8"),
                ("et", dtype(object)),
                ("diameter_m", "f8"),
                ("opened", "bool"),
                ("loss_coefficient", "f8"),
                ("type", dtype(object))]

    @classmethod
    def extract_results(cls, net, options, branch_results, mode):
        required_results_hyd, required_results_ht = standard_branch_wo_internals_result_lookup(net)

        required_results_hyd.extend([("v_mean_m_per_s", "v_mps"), ("lambda", "lambda"), ("reynolds", "reynolds")])

        if get_fluid(net).is_gas:
            required_results_hyd.extend([("v_from_m_per_s", "v_gas_from"), ("v_to_m_per_s", "v_gas_to")])

        extract_branch_results_without_internals(net, branch_results, required_results_hyd,
                                                 required_results_ht, cls.table_name(), mode)

    @classmethod
    def get_result_table(cls, net):
        """

        :param net: The pandapipes network
        :type net: pandapipesNet
        :return: (columns, all_float) - the column names and whether they are all float type. Only
                if False, returns columns as tuples also specifying the dtypes
        :rtype: (list, bool)
        """
        if get_fluid(net).is_gas:
            output = ["v_from_m_per_s", "v_to_m_per_s", "v_mean_m_per_s", "p_from_bar", "p_to_bar",
                      "t_from_k", "t_to_k", "t_outlet_k", "mdot_from_kg_per_s", "mdot_to_kg_per_s",
                      "vdot_norm_m3_per_s", "reynolds", "lambda", "normfactor_from",
                      "normfactor_to"]
        else:
            output = ["v_mean_m_per_s", "p_from_bar", "p_to_bar", "t_from_k", "t_to_k", "t_outlet_k",
                      "mdot_from_kg_per_s", "mdot_to_kg_per_s", "vdot_m3_per_s", "reynolds",
                      "lambda"]
        return output, True
