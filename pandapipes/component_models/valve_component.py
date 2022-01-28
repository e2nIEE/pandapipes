# Copyright (c) 2020-2021 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype

from pandapipes.component_models.abstract_models import BranchWZeroLengthComponent
from pandapipes.constants import NORMAL_PRESSURE, NORMAL_TEMPERATURE
from pandapipes.internals_toolbox import _sum_by_group
from pandapipes.properties.fluids import is_fluid_gas, get_compressibility
from operator import itemgetter


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
        valve_pit[:, net['_idx_branch']['D']] = net[cls.table_name()].diameter_m.values
        valve_pit[:, net['_idx_branch']['AREA']] = valve_pit[:, net['_idx_branch']['D']] ** 2 * np.pi / 4
        valve_pit[:, net['_idx_branch']['LOSS_COEFFICIENT']] = net[cls.table_name()].loss_coefficient.values

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
        valve_pit[:, net['_idx_branch']['PL']] = 0

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
        valve_pit[:, net['_idx_branch']['TL']] = 0

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

        idx_active = branch_pit[:, net['_idx_branch']['ELEMENT_IDX']]
        v_mps = branch_pit[:, net['_idx_branch']['VINIT']]
        _, v_sum, internal_pipes = _sum_by_group(idx_active, v_mps, np.ones_like(idx_active))
        idx_pit = branch_pit[:, net['_idx_branch']['ELEMENT_IDX']]
        _, lambda_sum, reynolds_sum, = \
            _sum_by_group(idx_pit, branch_pit[:, net['_idx_branch']['LAMBDA']], branch_pit[:, net['_idx_branch']['RE']])
        if is_fluid_gas(net):
            from_nodes = branch_pit[:, net['_idx_branch']['FROM_NODE']].astype(np.int32)
            to_nodes = branch_pit[:, net['_idx_branch']['TO_NODE']].astype(np.int32)
            numerator = NORMAL_PRESSURE * branch_pit[:, net['_idx_branch']['TINIT']]
            p_from = node_pit[from_nodes, net['_idx_node']['PAMB']] + node_pit[from_nodes, net['_idx_node']['PINIT']]
            p_to = node_pit[to_nodes, net['_idx_node']['PAMB']] + node_pit[to_nodes, net['_idx_node']['PINIT']]
            mask = ~np.isclose(p_from, p_to)
            p_mean = np.empty_like(p_to)
            p_mean[~mask] = p_from[~mask]
            p_mean[mask] = 2 / 3 * (p_from[mask] ** 3 - p_to[mask] ** 3) \
                           / (p_from[mask] ** 2 - p_to[mask] ** 2)
            if len(net._fluid) == 1:
                normfactor_mean = numerator * get_compressibility(net, p_mean) \
                                  / (p_mean * NORMAL_TEMPERATURE)
            else:
                node_pit = net['_pit']['node']
                vinit = branch_pit[:, net['_idx_branch']['VINIT']]
                nodes = np.zeros(len(vinit), dtype=int)
                nodes[vinit >= 0] = branch_pit[:, net['_idx_branch']['FROM_NODE']][vinit >= 0]
                nodes[vinit < 0] = branch_pit[:, net['_idx_branch']['TO_NODE']][vinit < 0]
                slacks = node_pit[nodes, net['_idx_node']['SLACK']]
                mf = net['_mass_fraction']
                mf = np.array(itemgetter(*slacks)(mf))
                comp_fact = get_compressibility(net, p_mean, mass_fraction=mf)
                normfactor_mean = numerator * comp_fact / (p_mean * NORMAL_TEMPERATURE)
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
        if is_fluid_gas(net):
            output = ["v_from_m_per_s", "v_to_m_per_s", "v_mean_m_per_s", "p_from_bar", "p_to_bar",
                      "t_from_k", "t_to_k", "mdot_from_kg_per_s", "mdot_to_kg_per_s",
                      "vdot_norm_m3_per_s", "reynolds", "lambda", "normfactor_from",
                      "normfactor_to"]
        else:
            output = ["v_mean_m_per_s", "p_from_bar", "p_to_bar", "t_from_k", "t_to_k",
                      "mdot_from_kg_per_s", "mdot_to_kg_per_s", "vdot_norm_m3_per_s", "reynolds",
                      "lambda"]
        return output, True
