# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype
from operator import itemgetter

from pandapipes.component_models.abstract_models import BranchWZeroLengthComponent

from pandapipes.idx_node import PINIT, PAMB
from pandapipes.idx_branch import STD_TYPE, VINIT, D, AREA, TL, \
    LOSS_COEFFICIENT as LC, FROM_NODE, TO_NODE, TINIT, PL

from pandapipes.constants import NORMAL_TEMPERATURE, NORMAL_PRESSURE

from pandapipes.pipeflow_setup import get_net_option, get_fluid


class Pump(BranchWZeroLengthComponent):
    """

    """

    @classmethod
    def table_name(cls):
        return "pump"

    @classmethod
    def active_identifier(cls):
        return "in_service"

    @classmethod
    def create_pit_branch_entries(cls, net, pump_pit, node_name):
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
        pump_pit = super().create_pit_branch_entries(net, pump_pit, node_name)
        std_types_lookup = np.array(list(net.std_type[cls.table_name()].keys()))
        std_type, pos = np.where(net[cls.table_name()]['std_type'].values
                                 == std_types_lookup[:, np.newaxis])
        pump_pit[pos, STD_TYPE] = std_type
        pump_pit[:, D] = 0.1
        pump_pit[:, AREA] = pump_pit[:, D] ** 2 * np.pi / 4
        pump_pit[:, LC] = 0

    @classmethod
    def calculate_pressure_lift(cls, net, pump_pit, node_pit):
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
        area = pump_pit[:, AREA]
        idx = pump_pit[:, STD_TYPE].astype(int)
        std_types = np.array(list(net.std_type['pump'].keys()))[idx]
        p_scale = get_net_option(net, "p_scale")
        from_nodes = pump_pit[:, FROM_NODE].astype(np.int32)
        to_nodes = pump_pit[:, TO_NODE].astype(np.int32)
        fluid = get_fluid(net)
        p_from = node_pit[from_nodes, PAMB] + node_pit[from_nodes, PINIT] * p_scale
        p_to = node_pit[to_nodes, PAMB] + node_pit[to_nodes, PINIT] * p_scale
        numerator = NORMAL_PRESSURE * pump_pit[:, TINIT]
        v_mps = pump_pit[:, VINIT]
        if fluid.is_gas:
            mask = p_from != p_to
            p_mean = np.empty_like(p_to)
            p_mean[~mask] = p_from[~mask]
            p_mean[mask] = 2 / 3 * (p_from[mask] ** 3 - p_to[mask] ** 3) \
                           / (p_from[mask] ** 2 - p_to[mask] ** 2)
            normfactor_mean = numerator * fluid.get_property("compressibility", p_mean) \
                              / (p_mean * NORMAL_TEMPERATURE)
            v_mean = v_mps * normfactor_mean
        else:
            v_mean = v_mps
        vol = v_mean * area
        fcts = itemgetter(*std_types)(net['std_type']['pump'])
        fcts = [fcts] if not isinstance(fcts, tuple) else fcts
        pl = np.array(list(map(lambda x, y: x.get_pressure(y), fcts, vol)))
        pump_pit[:, PL] = pl

    @classmethod
    def calculate_temperature_lift(cls, net, pump_pit, node_pit):
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
        pump_pit[:, TL] = 0

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
        placement_table, pump_pit, res_table = super().extract_results(net, options, node_name)
        res_table['deltap_bar'].values[placement_table] = pump_pit[:, PL]

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
                ("std_type", dtype(object)),
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
        return ["deltap_bar"], True
