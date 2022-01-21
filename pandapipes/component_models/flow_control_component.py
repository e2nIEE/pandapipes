# Copyright (c) 2020-2021 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype

from pandapipes.component_models.component_toolbox import calculate_branch_results
from pandapipes.component_models.abstract_models import BranchWZeroLengthComponent, get_fluid
from pandapipes.idx_branch import D, AREA, TL, JAC_DERIV_DP, JAC_DERIV_DP1, JAC_DERIV_DV, VINIT, \
    RHO, LOAD_VEC_BRANCHES
from pandapipes.pf.pipeflow_setup import get_net_option


class FlowControlComponent(BranchWZeroLengthComponent):
    """

    """

    @classmethod
    def table_name(cls):
        return "flow_control"

    @classmethod
    def active_identifier(cls):
        return "in_service"

    @classmethod
    def from_to_node_cols(cls):
        return "from_junction", "to_junction"

    @classmethod
    def create_pit_branch_entries(cls, net, branch_pit, node_name):
        """
        Function which creates pit branch entries with a specific table.
        :param net: The pandapipes network
        :type net: pandapipesNet
        :param branch_pit:
        :type branch_pit:
        :param node_name:
        :type node_name:
        :return: No Output.
        """
        fc_pit = super().create_pit_branch_entries(net, branch_pit, node_name)
        fc_pit[:, D] = net[cls.table_name()].diameter_m.values
        fc_pit[:, AREA] = fc_pit[:, D] ** 2 * np.pi / 4
        fc_pit[:, VINIT] = net[cls.table_name()].controlled_mdot_kg_per_s.values / \
                           (fc_pit[:, AREA] * fc_pit[:, RHO])

    @classmethod
    def adaption_before_derivatives(cls, net, branch_pit, node_pit, idx_lookups, options):
        pass

    @classmethod
    def adaption_after_derivatives(cls, net, branch_pit, node_pit, idx_lookups, options):
        # set all pressure derivatives to 0 and velocity to 1; load vector must be 0, as no change
        # of velocity is allowed during the pipeflow iteration
        f, t = idx_lookups[cls.table_name()]
        fc_pit = branch_pit[f:t, :]
        fc_pit[:, JAC_DERIV_DP] = 0
        fc_pit[:, JAC_DERIV_DP1] = 0
        fc_pit[:, JAC_DERIV_DV] = 1
        fc_pit[:, LOAD_VEC_BRANCHES] = 0

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

        placement_table, res_table, fc_pit, node_pit = super().extract_results(net, options,
                                                                               node_name)
        fluid = get_fluid(net)
        use_numba = get_net_option(net, "use_numba")

        calculate_branch_results(res_table, fc_pit, node_pit, placement_table, fluid, use_numba)

        return placement_table, res_table, fc_pit

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
                ("controlled_mdot_kg_per_s", "f8"),
                ("diameter_m", "f8"),
                ("control_active", "bool"),
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
            output = ["v_from_m_per_s", "v_to_m_per_s", "v_mean_m_per_s", "p_from_bar", "p_to_bar",
                      "t_from_k", "t_to_k", "mdot_from_kg_per_s", "mdot_to_kg_per_s",
                      "vdot_norm_m3_per_s", "reynolds", "lambda", "normfactor_from",
                      "normfactor_to"]
        else:
            output = ["v_mean_m_per_s", "p_from_bar", "p_to_bar", "t_from_k", "t_to_k",
                      "mdot_from_kg_per_s", "mdot_to_kg_per_s", "vdot_norm_m3_per_s", "reynolds",
                      "lambda"]
        return output, True
