# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype

from pandapipes.component_models.component_toolbox import get_component_array
from pandapipes.component_models.junction_component import Junction
from pandapipes.component_models.pump_component import Pump


class Compressor(Pump):
    """

    """
    PRESSURE_RATIO = 0

    internal_cols = 1

    @classmethod
    def table_name(cls):
        return "compressor"

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
        compressor_pit = super(Pump, cls).create_pit_branch_entries(net, branch_pit)

        compressor_pit[:, net['_idx_branch']['D']] = 0.1
        compressor_pit[:, net['_idx_branch']['AREA']] = compressor_pit[:, net['_idx_branch']['D']] ** 2 * np.pi / 4
        compressor_pit[:, net['_idx_branch']['LOSS_COEFFICIENT']] = 0

    @classmethod
    def create_component_array(cls, net, component_pits):
        """
        Function which creates an internal array of the component in analogy to the pit, but with
        component specific entries, that are not needed in the pit.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param component_pits: dictionary of component specific arrays
        :type component_pits: dict
        :return:
        :rtype:
        """
        tbl = net[cls.table_name()]
        compr_array = np.zeros(shape=(len(tbl), cls.internal_cols), dtype=np.float64)
        compr_array[:, cls.PRESSURE_RATIO] = net[cls.table_name()].pressure_ratio.values
        component_pits[cls.table_name()] = compr_array

    @classmethod
    def adaption_before_derivatives_hydraulic(cls, net, branch_pit, node_pit, branch_lookups, node_lookups, options):
        """ absolute inlet pressure multiplied by the compressor's boost ratio.
        If the flow is reversed, the pressure lift is set to 0."""
        f, t = branch_lookups[cls.table_name()]
        compressor_branch_pit = branch_pit[f:t, :]
        compressor_array = get_component_array(net, cls.table_name())

        from_nodes = compressor_branch_pit[:, net['_idx_branch']['FROM_NODE']].astype(np.int32)
        p_from = node_pit[from_nodes, net['_idx_node']['PAMB']] + node_pit[from_nodes, net['_idx_node']['PINIT']]
        p_to_calc = p_from * compressor_array[:, cls.PRESSURE_RATIO]
        pl_abs = p_to_calc - p_from

        v_mps = compressor_branch_pit[:, net['_idx_branch']['VINIT']]
        pl_abs[v_mps < 0] = 0  # force pressure lift = 0 for reverse flow

        compressor_branch_pit[:, net['_idx_branch']['PL']] = pl_abs

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
