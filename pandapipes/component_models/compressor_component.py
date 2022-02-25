# Copyright (c) 2020-2022 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from operator import itemgetter

import numpy as np
from numpy import dtype
from pandapipes.component_models.pump_component import Pump
from pandapipes.constants import NORMAL_TEMPERATURE, NORMAL_PRESSURE
from pandapipes.idx_branch import STD_TYPE, VINIT, D, AREA, TL, \
    LOSS_COEFFICIENT as LC, FROM_NODE, TINIT, PL, PRESSURE_RATIO
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
        :param compressor_pit: a part of the pit that includes only those columns relevant for
                               compressors
        :type compressor_pit:
        :param node_name:
        :type node_name:
        :return: No Output.
        """
        compressor_pit = super(Pump, cls).create_pit_branch_entries(net, compressor_pit, node_name)

        compressor_pit[:, D] = 0.1
        compressor_pit[:, AREA] = compressor_pit[:, D] ** 2 * np.pi / 4
        compressor_pit[:, LC] = 0
        compressor_pit[:, PRESSURE_RATIO] = net[cls.table_name()].pressure_ratio.values

    @classmethod
    def calculate_pressure_lift(cls, net, compressor_pit, node_pit):
        """ absolute inlet pressure multiplied by the compressor's boost ratio.

        If the flow is reversed, the pressure lift is set to 0.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param compressor_pit:
        :type compressor_pit:
        :param node_pit:
        :type node_pit:
        """
        pressure_ratio = compressor_pit[:, PRESSURE_RATIO]

        from_nodes = compressor_pit[:, FROM_NODE].astype(np.int32)
        p_from = node_pit[from_nodes, PAMB] + node_pit[from_nodes, PINIT]
        p_to_calc = p_from * pressure_ratio
        pl_abs = p_to_calc - p_from

        v_mps = compressor_pit[:, VINIT]
        pl_abs *= (v_mps >= 0)  # force pressure lift = 0 for reverse flow

        compressor_pit[:, PL] = pl_abs


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
                ("pressure_ratio", "f8"),
                ("in_service", 'bool')]
