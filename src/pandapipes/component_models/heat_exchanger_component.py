# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from numpy import dtype

from pandapipes.component_models._branch_element_models import BranchElementComponent
from pandapipes.component_models.component_toolbox import standard_branch_wo_internals_result_lookup
from pandapipes.idx_branch import QEXT, LOSS_COEFFICIENT as LC
from pandapipes.properties.fluids import get_fluid
from pandapipes.utils.result_extraction import extract_branch_results_without_internals


class HeatExchanger(BranchElementComponent):
    @property
    def table_name(self):
        return "heat_exchanger"

    def create_pit_branch_entries(self, net, branch_pit):
        """
        Function which creates pit branch entries with a specific table.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param branch_pit:
        :type branch_pit:
        :return: No Output.
        """
        heat_exchanger_pit = super().create_pit_branch_entries(net, branch_pit)
        heat_exchanger_pit[:, LC] = net[self.table_name].loss_coefficient.values
        heat_exchanger_pit[:, QEXT] = net[self.table_name].qext_w.values

    def extract_results(self, net, options, branch_results, mode):
        """
        Class method to extract pipeflow results from the internal structure into the results table.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param options: pipeflow options
        :type options: dict
        :param branch_results: important branch results
        :type branch_results: dict
        :param mode: simulation mode
        :type mode: str
        :return: No Output.
        :rtype: None
        """
        required_results_hyd, required_results_ht = standard_branch_wo_internals_result_lookup(net)

        extract_branch_results_without_internals(net, branch_results, required_results_hyd,
                                                 required_results_ht, self.table_name, mode)

    def get_component_input(self):
        """

        :return:
        :rtype:
        """
        return [("name", dtype(object)),
                ("from_junction", "u4"),
                ("to_junction", "u4"),
                ("diameter_m", "f8"),
                ("qext_w", 'f8'),
                ("loss_coefficient", "f8"),
                ("in_service", 'bool'),
                ("type", dtype(object))]

    def get_result_table(self, net):
        """

        :param net: The pandapipes network
        :type net: pandapipesNet
        :return: (columns, all_float) - the column names and whether they are all float type. Only
                if False, returns columns as tuples also specifying the dtypes
        :rtype: (list, bool)
        """
        if get_fluid(net).is_gas:
            output = ["p_from_bar", "p_to_bar",
                      "t_from_k", "t_to_k", "t_outlet_k", "mdot_from_kg_per_s", "mdot_to_kg_per_s",
                      "vdot_norm_m3_per_s", "reynolds", "lambda", "normfactor_from",
                      "normfactor_to"]
        else:
            output = ["p_from_bar", "p_to_bar", "t_from_k", "t_to_k", "t_outlet_k",
                      "mdot_from_kg_per_s", "mdot_to_kg_per_s", "vdot_m3_per_s", "reynolds",
                      "lambda"]
        return output, True
