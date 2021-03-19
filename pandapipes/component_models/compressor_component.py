# Copyright (c) 2020-2021 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from operator import itemgetter

import numpy as np
from numpy import dtype
from pandapipes.component_models.pump_component import Pump
from pandapipes.constants import NORMAL_TEMPERATURE, NORMAL_PRESSURE
from pandapipes.idx_branch import STD_TYPE, VINIT, D, AREA, TL, \
    LOSS_COEFFICIENT as LC, FROM_NODE, TINIT, PL, BOOST_RATIO
from pandapipes.idx_node import PINIT, PAMB
from pandapipes.pipeflow_setup import get_net_option, get_fluid


class Compressor(Pump):
    """

    """

    @classmethod
    def table_name(cls):
        return "compressor"

    @classmethod
    def create_pit_branch_entries(cls, net, compressor_pit, node_name):
        """
        Function which creates pit branch entries with a specific table.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param compressor_pit:
        :type compressor_pit:
        :param internal_pipe_number:
        :type internal_pipe_number:
        :return: No Output.
        """
        compressor_pit = super(Pump, cls).create_pit_branch_entries(net, compressor_pit, node_name)

        # std_types_lookup = np.array(list(net.std_type[cls.table_name()].keys()))
        # std_type, pos = np.where(net[cls.table_name()]['std_type'].values
        #                          == std_types_lookup[:, np.newaxis])
        # compressor_pit[:, STD_TYPE] = std_type
        compressor_pit[:, D] = 0.1
        compressor_pit[:, AREA] = compressor_pit[:, D] ** 2 * np.pi / 4
        compressor_pit[:, LC] = 0
        compressor_pit[:, BOOST_RATIO] = net.compressor.boost_ratio.values

    @classmethod
    def calculate_pressure_lift(cls, net, compressor_pit, node_pit):
        """

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param compressor_pit:
        :type compressor_pit:
        :param node_pit:
        :type node_pit:
        :return: power stroke
        :rtype: float
        """
        # area = compressor_pit[:, AREA]
        # idx = compressor_pit[:, STD_TYPE].astype(int)
        # std_types = np.array(list(net.std_type['pump'].keys()))[idx]
        boost_ratio = net[cls.table_name()].boost_ratio.values  # TODO: get boost ratio for each

        # compressor
        p_scale = get_net_option(net, "p_scale")
        from_nodes = compressor_pit[:, FROM_NODE].astype(np.int32)
        # to_nodes = compressor_pit[:, TO_NODE].astype(np.int32)
        # fluid = get_fluid(net)
        p_from = node_pit[from_nodes, PAMB] + node_pit[from_nodes, PINIT] * p_scale
        p_to_calc = p_from * boost_ratio
        pl_abs = p_to_calc - p_from

        v_mps = compressor_pit[:, VINIT]
        pl_abs *= (v_mps >= 0)  # force pressure lift = 0 for reverse flow

        compressor_pit[:, PL] = pl_abs

        # # p_to = node_pit[to_nodes, PAMB] + node_pit[to_nodes, PINIT] * p_scale
        # # numerator = NORMAL_PRESSURE * compressor_pit[:, TINIT]
        # if fluid.is_gas:
        #     # consider volume flow at inlet
        #     normfactor_from = numerator * fluid.get_property("compressibility", p_from) \
        #                       / (p_from * NORMAL_TEMPERATURE)
        #     v_mean = v_mps * normfactor_from
        # else:
        #     v_mean = v_mps
        # vol = v_mean * area
        # fcts = itemgetter(*std_types)(net['std_type']['pump'])
        # fcts = [fcts] if not isinstance(fcts, tuple) else fcts
        # pl = np.array(list(map(lambda x, y: x.get_pressure(y), fcts, vol)))

    #  --- keep as in Pump class ---
    # @classmethod
    # def calculate_temperature_lift(cls, net, compressor_pit, node_pit):
    #     """
    #
    #     :param net:
    #     :type net:
    #     :param compressor_pit:
    #     :type compressor_pit:
    #     :param node_pit:
    #     :type node_pit:
    #     :return:
    #     :rtype:
    #     """
    #     compressor_pit[:, TL] = 0


    #  --- keep as in Pump class ---
    # @classmethod
    # def extract_results(cls, net, options, node_name):
    #     """
    #     Function that extracts certain results.
    #
    #     :param net: The pandapipes network
    #     :type net: pandapipesNet
    #     :param options:
    #     :type options:
    #     :return: No Output.
    #     """
    #     placement_table, compressor_pit, res_table = super().prepare_result_tables(net, options, node_name)
    #     res_table['deltap_bar'].values[placement_table] = compressor_pit[:, PL]



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
                # ("std_type", dtype(object)),
                ("boost_ratio", "f8"),
                ("in_service", 'bool'),
                ("type", dtype(object))]
