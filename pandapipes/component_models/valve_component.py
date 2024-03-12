# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype

from pandapipes.component_models.component_toolbox import standard_branch_wo_internals_result_lookup
from pandapipes.component_models.abstract_models.branch_wzerolength_models import \
    BranchWZeroLengthComponent
from pandapipes.component_models.junction_component import Junction
from pandapipes.pf.result_extraction import extract_branch_results_without_internals
from pandapipes.properties.fluids import is_fluid_gas


class Valve(BranchWZeroLengthComponent):
    """
    Valves are branch elements that can separate two junctions.
    They have a length of 0, but can introduce a lumped pressure loss.
    """

    @classmethod
    def from_to_node_cols(cls):
        return "from_junction", "to_junction"

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
        valve_pit[:, net['_idx_branch']['D']] = net[cls.table_name()].diameter_m.values
        valve_pit[:, net['_idx_branch']['AREA']] = valve_pit[:, net['_idx_branch']['D']] ** 2 * np.pi / 4
        valve_pit[:, net['_idx_branch']['LOSS_COEFFICIENT']] = net[cls.table_name()].loss_coefficient.values

    @classmethod
    def calculate_temperature_lift(cls, net, branch_component_pit, node_pit):
        """

        :param net:
        :type net:
        :param branch_component_pit:
        :type branch_component_pit:
        :param node_pit:
        :type node_pit:
        :return:
        :rtype:
        """
        branch_component_pit[:, net['_idx_branch']['TL']] = 0

    @classmethod
    def get_component_input(cls):
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

    @classmethod
    def extract_results(cls, net, options, branch_results, mode):
        required_results_hyd, required_results_ht = standard_branch_wo_internals_result_lookup(net)

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
        if is_fluid_gas(net):
            output = ["v_from_m_per_s", "v_to_m_per_s", "v_mean_m_per_s", "p_from_bar", "p_to_bar",
                      "t_from_k", "t_to_k", "mdot_from_kg_per_s", "mdot_to_kg_per_s",
                      "vdot_norm_m3_per_s", "reynolds", "lambda", "normfactor_from",
                      "normfactor_to"]

            add = ["w_%s" % fluid for fluid in net._fluid]

        else:
            output = ["v_mean_m_per_s", "p_from_bar", "p_to_bar", "t_from_k", "t_to_k",
                      "mdot_from_kg_per_s", "mdot_to_kg_per_s", "vdot_norm_m3_per_s", "reynolds",
                      "lambda"]

            add = []
        return output + add, True
