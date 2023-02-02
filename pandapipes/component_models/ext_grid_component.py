# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype

from pandapipes.component_models.abstract_models.node_element_models import NodeElementComponent
from pandapipes.component_models.component_toolbox import set_fixed_node_entries, \
    get_mass_flow_at_nodes
from pandapipes.pf.pipeflow_setup import get_lookup

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class ExtGrid(NodeElementComponent):
    """

    """

    @classmethod
    def table_name(cls):
        return "ext_grid"

    @classmethod
    def sign(cls):
        return 1.

    @classmethod
    def get_connected_node_type(cls):
        from pandapipes.component_models.junction_component import Junction
        return Junction

    @classmethod
    def active_identifier(cls):
        return "in_service"

    @classmethod
    def create_pit_node_entries(cls, net, node_pit):
        """
        Function which creates pit node entries.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param node_pit:
        :type node_pit:
        :return: No Output.
        """
        ext_grids = net[cls.table_name()]
        ext_grids = ext_grids[ext_grids[cls.active_identifier()].values]

        junction = ext_grids[cls.get_node_col()].values
        press = ext_grids.p_bar.values
        set_fixed_node_entries(net, node_pit, junction, ext_grids.type.values, press,
                               ext_grids.t_k.values, cls.get_connected_node_type())

        return ext_grids, press

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
        ext_grids = net[cls.table_name()]

        if len(ext_grids) == 0:
            return

        res_table = net["res_" + cls.table_name()]

        branch_pit = net['_pit']['branch']
        node_pit = net["_pit"]["node"]

        p_grids = np.isin(ext_grids.type.values, ["p", "pt"]) & ext_grids.in_service.values
        junction = cls.get_connected_junction(net).values
        # get indices in internal structure for junctions in ext_grid tables which are "active"
        eg_nodes = get_lookup(net, "node", "index")[cls.get_connected_node_type().table_name()][
            junction[p_grids]]
        sum_mass_flows, inverse_nodes, counts = get_mass_flow_at_nodes(net, node_pit, branch_pit,
                                                                       eg_nodes, cls)

        # positive results mean that the ext_grid feeds in, negative means that the ext grid
        # extracts (like a load)
        res_table["mdot_kg_per_s"].values[p_grids] = \
            cls.sign() * (sum_mass_flows / counts)[inverse_nodes]
        return res_table, ext_grids, node_pit, branch_pit

    @classmethod
    def get_connected_junction(cls, net):
        junction = net[cls.table_name()].junction
        return junction

    @classmethod
    def get_node_col(cls):
        return "junction"

    @classmethod
    def get_component_input(cls):
        """

        :return:
        :rtype:
        """
        return [("name", dtype(object)),
                ("junction", "u4"),
                ("p_bar", "f8"),
                ("t_k", "f8"),
                ("in_service", "bool"),
                ('type', dtype(object))]

    @classmethod
    def get_result_table(cls, net):
        """

        :param net: The pandapipes network
        :type net: pandapipesNet
        :return: (columns, all_float) - the column names and whether they are all float type. Only
                if False, returns columns as tuples also specifying the dtypes
        :rtype: (list, bool)
        """
        return ["mdot_kg_per_s"], True
