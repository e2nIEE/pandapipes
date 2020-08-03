# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype

from pandapipes.component_models.abstract_models import NodeElementComponent
from pandapipes.constants import GRAVITATION_CONSTANT, P_CONVERSION
from pandapipes.idx_branch import FROM_NODE, TO_NODE, LOAD_VEC_NODES
from pandapipes.idx_node import PINIT, LOAD, NODE_TYPE, P, EXT_GRID_OCCURENCE
from pandapipes.internals_toolbox import _sum_by_group
from pandapipes.pipeflow_setup import get_lookup

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class WaterTower(NodeElementComponent):
    """

    """

    @classmethod
    def table_name(cls):
        return "water_tower"

    @classmethod
    def sign(cls):
        return 1.

    @classmethod
    def create_pit_node_entries(cls, net, node_pit, node_name):
        """
        Function which creates pit node entries.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param node_pit:
        :type node_pit:
        :return: No Output.
        """
        water_towers = net[cls.table_name()]
        density = net.fluid.get_density(water_towers.t_k.values)
        press = density * water_towers.height_m.values * GRAVITATION_CONSTANT / P_CONVERSION * \
                water_towers.in_service.values
        junction_idx_lookups = get_lookup(net, "node", "index")[node_name]
        junction = cls.get_connected_junction(net)
        juncts_p, press_sum, number = _sum_by_group(junction.values, press,
                                                    np.ones_like(press, dtype=np.int32))
        index_p = junction_idx_lookups[juncts_p]
        node_pit[index_p, PINIT] = press_sum / number
        node_pit[index_p, NODE_TYPE] = P
        node_pit[index_p, EXT_GRID_OCCURENCE] += number

        net["_lookups"]["water_tower"] = \
            np.array(list(set(np.concatenate([net["_lookups"]["water_tower"], index_p])))) if \
                "water_tower" in net['_lookups'] else index_p
        return water_towers, press

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
        water_towers = net[cls.table_name()]

        if len(water_towers) == 0:
            return

        res_table = super().extract_results(net, options, node_name)

        branch_pit = net['_pit']['branch']
        node_pit = net["_pit"]["node"]

        junction = cls.get_connected_junction(net)
        index_juncts = np.array(junction.values[junction])
        junct_uni = np.array(list(set(index_juncts)))
        index_nodes = get_lookup(net, "node", "index")[node_name][junct_uni]
        eg_from_branches = np.isin(branch_pit[:, FROM_NODE], index_nodes)
        eg_to_branches = np.isin(branch_pit[:, TO_NODE], index_nodes)
        from_nodes = branch_pit[eg_from_branches, FROM_NODE]
        to_nodes = branch_pit[eg_to_branches, TO_NODE]
        mass_flow_from = branch_pit[eg_from_branches, LOAD_VEC_NODES]
        mass_flow_to = branch_pit[eg_to_branches, LOAD_VEC_NODES]
        press = node_pit[index_nodes, PINIT]
        loads = node_pit[index_nodes, LOAD]
        counts = node_pit[index_nodes, EXT_GRID_OCCURENCE]
        all_index_nodes = np.concatenate([from_nodes, to_nodes, index_nodes])
        all_mass_flows = np.concatenate([-mass_flow_from, mass_flow_to, -loads])
        nodes, sum_mass_flows = _sum_by_group(all_index_nodes, all_mass_flows)

        # positive results mean that the ext_grid feeds in, negative means that the ext grid
        # extracts (like a load)
        res_table["mdot_kg_per_s"].values[:] = np.repeat(cls.sign() * sum_mass_flows / counts,
                                                         counts.astype(int))
        res_table["p_bar"].values[:] = press
        return res_table, water_towers, index_nodes, node_pit, branch_pit

    @classmethod
    def get_connected_junction(cls, net):
        junction = net[cls.table_name()].junction
        return junction

    @classmethod
    def get_component_input(cls):
        """

        :return:
        """

        return [("name", dtype(object)),
                ("junction", "u4"),
                ("height_m", "f8"),
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
        return ["mdot_kg_per_s", "p_bar"], True
