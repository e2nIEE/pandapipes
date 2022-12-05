# Copyright (c) 2020-2022 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np

from pandapipes.component_models.ext_grid_component import ExtGrid
from pandapipes.idx_node import PINIT
from pandapipes.pf.pipeflow_setup import get_lookup

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class CirculationPump(ExtGrid):

    @classmethod
    def sign(cls):
        return -1.

    @classmethod
    def get_connected_node_type(cls):
        raise NotImplementedError

    @classmethod
    def extract_results(cls, net, options, branch_results, nodes_connected, branches_connected):
        """
        Function that extracts certain results.

        :param nodes_connected:
        :type nodes_connected:
        :param branches_connected:
        :type branches_connected:
        :param branch_results:
        :type branch_results:
        :param net: The pandapipes network
        :type net: pandapipesNet
        :param options:
        :type options:
        :return: No Output.
        """
        res_table, circ_pump, index_nodes_from, node_pit, _ = \
            super().extract_results(net, options, None, nodes_connected, branches_connected)

        index_juncts_to = circ_pump.to_junction.values
        junct_uni_to = np.array(list(set(index_juncts_to)))
        index_nodes_to = get_lookup(net, "node", "index")[
            cls.get_connected_node_type().table_name()][junct_uni_to]

        deltap_bar = node_pit[index_nodes_from, PINIT] - node_pit[index_nodes_to, PINIT]
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
