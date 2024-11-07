# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from numpy import dtype, zeros, float64, int32

from pandapipes.component_models.component_toolbox import get_component_array
from pandapipes.component_models.pump_component import Pump
from pandapipes.idx_branch import MDOTINIT, LOSS_COEFFICIENT as LC, FROM_NODE, PL
from pandapipes.idx_node import PINIT, PAMB


class Compressor(Pump):
    """

    """
    PRESSURE_RATIO = 0

    internal_cols = 1

    @property
    def table_name(self):
        return "compressor"

    def create_pit_branch_entries(self, net, branch_pit):
        """
        Function which creates pit branch entries with a specific table.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param branch_pit:
        :type branch_pit:
        :return: No Output.
        """
        compressor_pit = super().create_pit_branch_entries(net, branch_pit)
        compressor_pit[:, LC] = 0

    def create_component_array(self, net, component_pits):
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
        tbl = net[self.table_name]
        compr_array = zeros(shape=(len(tbl), self.internal_cols), dtype=float64)
        compr_array[:, self.PRESSURE_RATIO] = net[self.table_name].pressure_ratio.values
        component_pits[self.table_name] = compr_array

    def adaption_before_derivatives_hydraulic(self, net, branch_pit, node_pit, idx_lookups, options):
        # calculation of pressure lift
        f, t = idx_lookups[self.table_name]
        compressor_branch_pit = branch_pit[f:t, :]
        compressor_array = get_component_array(net, self.table_name)

        from_nodes = compressor_branch_pit[:, FROM_NODE].astype(int32)
        p_from = node_pit[from_nodes, PAMB] + node_pit[from_nodes, PINIT]

        p_to_calc = p_from * compressor_array[:, self.PRESSURE_RATIO]
        pl_abs = p_to_calc - p_from

        m_mps = compressor_branch_pit[:, MDOTINIT]
        pl_abs[m_mps < 0] = 0  # force pressure lift = 0 for reverse flow

        compressor_branch_pit[:, PL] = pl_abs

    def get_component_input(self):
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
