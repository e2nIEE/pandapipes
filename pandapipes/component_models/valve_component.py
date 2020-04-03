# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from pandapipes.pipeflow_setup import get_fluid, get_lookup, get_net_option
from pandapipes.component_models.abstract_models import BranchWZeroLengthComponent
from pandapipes.idx_node import ELEMENT_IDX, PINIT, TINIT as TINIT_NODE, PAMB
from pandapipes.idx_branch import FROM_NODE, TO_NODE, D, TINIT, AREA, VINIT, \
    LOAD_VEC_NODES, LOSS_COEFFICIENT as LC, PL, TL, RE, LAMBDA
from pandapipes.toolbox import _sum_by_group
from numpy import dtype
from pandapipes.constants import NORMAL_TEMPERATURE, NORMAL_PRESSURE


class Valve(BranchWZeroLengthComponent):
    """

    """

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
        :param internal_pipe_number:
        :type internal_pipe_number:
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
    def extract_results(cls, net, options, node_name):
        """
        Function that extracts certain results.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param options:
        :type options:
        :return: No Output.
        """
        placement_table, valve_pit, res_table = super().extract_results(net, options, node_name)

        node_pit = net["_active_pit"]["node"]
        node_active_idx_lookup = get_lookup(net, "node", "index_active")[node_name]
        junction_idx_lookup = get_lookup(net, "node", "index")[node_name]
        from_junction_nodes = node_active_idx_lookup[junction_idx_lookup[
            net[cls.table_name()]["from_junction"].values[placement_table]]]
        to_junction_nodes = node_active_idx_lookup[junction_idx_lookup[
            net[cls.table_name()]["to_junction"].values[placement_table]]]

        from_nodes = valve_pit[:, FROM_NODE].astype(np.int32)
        to_nodes = valve_pit[:, TO_NODE].astype(np.int32)
        p_scale = get_net_option(net, "p_scale")
        fluid = get_fluid(net)

        v_mps = valve_pit[:, VINIT]

        t0 = node_pit[from_nodes, TINIT_NODE]
        t1 = node_pit[to_nodes, TINIT_NODE]
        mf = valve_pit[:, LOAD_VEC_NODES]
        vf = valve_pit[:, LOAD_VEC_NODES] / get_fluid(net).get_density((t0 + t1) / 2)

        idx_active = valve_pit[:, ELEMENT_IDX]
        idx_sort, v_sum, mf_sum, vf_sum = \
            _sum_by_group(idx_active, v_mps, mf, vf)

        if fluid.is_gas:
            # derived from the ideal gas law
            p_from = node_pit[from_nodes, PAMB] + node_pit[from_nodes, PINIT] * p_scale
            p_to = node_pit[to_nodes, PAMB] + node_pit[to_nodes, PINIT] * p_scale
            numerator = NORMAL_PRESSURE * valve_pit[:, TINIT]
            normfactor_from = numerator * fluid.get_property("compressibility", p_from) \
                              / (p_from * NORMAL_TEMPERATURE)
            normfactor_to = numerator * fluid.get_property("compressibility", p_to) \
                            / (p_to * NORMAL_TEMPERATURE)
            v_gas_from = v_mps * normfactor_from
            v_gas_to = v_mps * normfactor_to
            mask = p_from != p_to
            p_mean = np.empty_like(p_to)
            p_mean[~mask] = p_from[~mask]
            p_mean[mask] = 2 / 3 * (p_from[mask] ** 3 - p_to[mask] ** 3) \
                           / (p_from[mask] ** 2 - p_to[mask] ** 2)
            normfactor_mean = numerator * fluid.get_property("compressibility", p_mean) \
                              / (p_mean * NORMAL_TEMPERATURE)
            v_gas_mean = v_mps * normfactor_mean

            idx_sort, v_gas_from_sum, v_gas_to_sum, v_gas_mean_sum, nf_from_sum, nf_to_sum = \
                _sum_by_group(idx_active, v_gas_from, v_gas_to, v_gas_mean, normfactor_from,
                              normfactor_to)

            res_table["v_from_m_per_s"].values[placement_table] = v_gas_from_sum
            res_table["v_to_m_per_s"].values[placement_table] = v_gas_to_sum
            res_table["v_mean_m_per_s"].values[placement_table] = v_gas_mean_sum
            res_table["normfactor_from"].values[placement_table] = nf_from_sum
            res_table["normfactor_to"].values[placement_table] = nf_to_sum
        else:
            res_table["v_mean_m_per_s"].values[placement_table] = v_sum

        res_table["p_from_bar"].values[placement_table] = node_pit[from_junction_nodes, PINIT]
        res_table["p_to_bar"].values[placement_table] = node_pit[to_junction_nodes, PINIT]
        res_table["t_from_k"].values[placement_table] = node_pit[from_junction_nodes, TINIT_NODE]
        res_table["t_to_k"].values[placement_table] = node_pit[to_junction_nodes, TINIT_NODE]
        res_table["mdot_to_kg_per_s"].values[placement_table] = -mf_sum
        res_table["mdot_from_kg_per_s"].values[placement_table] = mf_sum
        res_table["vdot_norm_m3_per_s"].values[placement_table] = vf_sum
        idx_pit = valve_pit[:, ELEMENT_IDX]
        idx_sort, lambda_sum, reynolds_sum, = \
            _sum_by_group(idx_pit, valve_pit[:, LAMBDA], valve_pit[:, RE])
        res_table["lambda"].values[placement_table] = lambda_sum
        res_table["reynolds"].values[placement_table] = reynolds_sum

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
