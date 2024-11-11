# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from numpy import dtype, pi

from pandapipes.component_models._branch_element_models import BranchElementComponent
from pandapipes.component_models.component_toolbox import standard_branch_wo_internals_result_lookup
from pandapipes.idx_branch import D, AREA, LOSS_COEFFICIENT as LC
from pandapipes.properties.fluids import get_fluid
from pandapipes.utils.result_extraction import extract_branch_results_without_internals


class Valve(BranchElementComponent):
    """
    Valves are branch elements that can separate two junctions.
    They have a length of 0, but can introduce a lumped pressure loss.
    """
    @property
    def table_name(self):
        return "valve"

    @property
    def active_identifier(self):
        return "opened"

    def create_pit_branch_entries(self, net, branch_pit):
        """
        Function which creates pit branch entries with a specific table.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param branch_pit:
        :type branch_pit:
        :return: No Output.
        """
        valve_pit = super().create_pit_branch_entries(net, branch_pit)
        valve_pit[:, D] = net[self.table_name].diameter_m.values
        valve_pit[:, AREA] = valve_pit[:, D] ** 2 * pi / 4
        valve_pit[:, LC] = net[self.table_name].loss_coefficient.values

    def get_component_input(self):
        """

        :return:
        :rtype:
        """
        return [("name", dtype(object)),
                ("from_junction", "i8"),
                ("to_junction", "i8"),
                ("diameter_m", "f8"),
                ("opened", "bool"),
                ("loss_coefficient", "f8"),
                ("type", dtype(object))]

    def extract_results(self, net, options, branch_results, mode):
        required_results_hyd, required_results_ht = standard_branch_wo_internals_result_lookup(net)

        required_results_hyd.extend([("v_mean_m_per_s", "v_mps"), ("lambda", "lambda"), ("reynolds", "reynolds")])

        if get_fluid(net).is_gas:
            required_results_hyd.extend([("v_from_m_per_s", "v_gas_from"), ("v_to_m_per_s", "v_gas_to")])

        extract_branch_results_without_internals(net, branch_results, required_results_hyd,
                                                 required_results_ht, self.table_name, mode)

    def get_result_table(self, net):
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
