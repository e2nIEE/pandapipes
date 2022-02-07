# Copyright (c) 2020-2022 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype

from pandapipes.component_models.abstract_models import BranchWZeroLengthComponent
from pandapipes.constants import NORMAL_PRESSURE, NORMAL_TEMPERATURE
from pandapipes.idx_branch import D, AREA, LOSS_COEFFICIENT as LC, PL, TL
from pandapipes.idx_branch import FROM_NODE, TO_NODE, TINIT, VINIT, RE, LAMBDA, ELEMENT_IDX
from pandapipes.idx_node import PINIT, PAMB
from pandapipes.internals_toolbox import _sum_by_group
from pandapipes.properties.fluids import get_fluid


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
    def create_pit_branch_entries(cls, net, valve_pit, node_name):
        """
        Function which creates pit branch entries with a specific table.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param valve_pit:
        :type valve_pit:
        :param node_name:
        :type node_name:
        :return: No Output.
        """
        valve_pit = super().create_pit_branch_entries(net, valve_pit, node_name)
        valve_pit[:, D] = net[cls.table_name()].diameter_m.values
        valve_pit[:, AREA] = valve_pit[:, D] ** 2 * np.pi / 4
        valve_pit[:, LC] = net[cls.table_name()].loss_coefficient.values

    @classmethod
    def calculate_pressure_lift(cls, net, valve_pit, node_pit):
        """

        :param net:
        :type net:
        :param valve_pit:
        :type valve_pit:
        :param node_pit:
        :type node_pit:
        :return:
        :rtype:
        """
        valve_pit[:, PL] = 0

    @classmethod
    def calculate_temperature_lift(cls, net, valve_pit, node_pit):
        """

        :param net:
        :type net:
        :param valve_pit:
        :type valve_pit:
        :param node_pit:
        :type node_pit:
        :return:
        :rtype:
        """
        valve_pit[:, TL] = 0

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
    def extract_results(cls, net, options, node_name):
        placement_table, res_table, branch_pit, node_pit = super().extract_results(net, options, node_name)
        fluid = get_fluid(net)

        idx_active = branch_pit[:, ELEMENT_IDX]
        v_mps = branch_pit[:, VINIT]
        _, v_sum, internal_pipes = _sum_by_group(idx_active, v_mps, np.ones_like(idx_active))
        idx_pit = branch_pit[:, ELEMENT_IDX]
        _, lambda_sum, reynolds_sum, = \
            _sum_by_group(idx_pit, branch_pit[:, LAMBDA], branch_pit[:, RE])
        if fluid.is_gas:
            from_nodes = branch_pit[:, FROM_NODE].astype(np.int32)
            to_nodes = branch_pit[:, TO_NODE].astype(np.int32)
            numerator = NORMAL_PRESSURE * branch_pit[:, TINIT]
            p_from = node_pit[from_nodes, PAMB] + node_pit[from_nodes, PINIT]
            p_to = node_pit[to_nodes, PAMB] + node_pit[to_nodes, PINIT]
            mask = ~np.isclose(p_from, p_to)
            p_mean = np.empty_like(p_to)
            p_mean[~mask] = p_from[~mask]
            p_mean[mask] = 2 / 3 * (p_from[mask] ** 3 - p_to[mask] ** 3) \
                           / (p_from[mask] ** 2 - p_to[mask] ** 2)
            normfactor_mean = numerator * fluid.get_property("compressibility", p_mean) \
                              / (p_mean * NORMAL_TEMPERATURE)
            v_gas_mean = v_mps * normfactor_mean
            _, v_gas_mean_sum = _sum_by_group(idx_active, v_gas_mean)
            res_table["v_mean_m_per_s"].values[placement_table] = v_gas_mean_sum / internal_pipes
        else:
            res_table["v_mean_m_per_s"].values[placement_table] = v_sum / internal_pipes
        res_table["lambda"].values[placement_table] = lambda_sum / internal_pipes
        res_table["reynolds"].values[placement_table] = reynolds_sum / internal_pipes

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
                      "t_from_k", "t_to_k", "mdot_from_kg_per_s", "mdot_to_kg_per_s",
                      "vdot_norm_m3_per_s", "reynolds", "lambda", "normfactor_from",
                      "normfactor_to"]
        else:
            output = ["v_mean_m_per_s", "p_from_bar", "p_to_bar", "t_from_k", "t_to_k",
                      "mdot_from_kg_per_s", "mdot_to_kg_per_s", "vdot_norm_m3_per_s", "reynolds",
                      "lambda"]
        return output, True
