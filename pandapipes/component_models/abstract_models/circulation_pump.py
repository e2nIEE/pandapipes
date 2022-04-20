# Copyright (c) 2020-2022 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np

from pandapipes.component_models.ext_grid_component import ExtGrid
from pandapipes.pipeflow_setup import get_lookup, get_table_number

try:
    from pandaplan.core import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class CirculationPump(ExtGrid):

    @classmethod
    def sign(cls):
        return -1.

    @classmethod
    def junction_name(cls):
        return 'from_junction'


    @classmethod
    def create_pit_node_element_entries(cls, net, node_element_pit, node_name):
        super().create_pit_node_element_entries(net, node_element_pit, node_name)


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

        node_pit = net._active_pit['node']
        node_lookup = get_lookup(net, "node", "index")[node_name]

        res_table, circ_pump = super().extract_results(net, options, node_name)

        nodes_from = node_lookup[circ_pump.from_junction]
        nodes_to = node_lookup[circ_pump.to_junction]

        deltap_bar = node_pit[nodes_from, net['_idx_node']['PINIT']] - node_pit[nodes_to, net['_idx_node']['PINIT']]
        res_table["deltap_bar"].values[:] = deltap_bar

    @classmethod
    def get_connected_junction(cls, net):
        junction = net[cls.table_name()].from_junction
        return junction

    @classmethod
    def get_result_table(cls, net):
        """

        :param net: The pandapipes network
        :type net: pandapipesNet
        :return: (columns, all_float) - the column names and whether they are all float type. Only
                if False, returns columns as tuples also specifying the dtypes
        :rtype: (list, bool)
        """
        return ["mdot_kg_per_s", "deltap_bar"], True
