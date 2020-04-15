# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype

from pandapipes.component_models.abstract_models import NodeElementComponent

from pandapipes.idx_node import PINIT, LOAD, TINIT, NODE_TYPE, NODE_TYPE_T, P, T, \
    EXT_GRID_OCCURENCE, EXT_GRID_OCCURENCE_T
from pandapipes.idx_branch import FROM_NODE, TO_NODE, LOAD_VEC_NODES

from pandapipes.pipeflow_setup import get_lookup
from pandapipes.toolbox import _sum_by_group

try:
    import pplog as logging
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
    def create_pit_node_entries(cls, net, node_pit, node_name):
        """
        Function which creates pit node entries.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param node_pit:
        :type node_pit:
        :return: No Output.
        """
        ext_grids = net[cls.table_name()]

        p_mask = np.where(np.isin(ext_grids.type.values, ["p", "pt"]))
        press = ext_grids.p_bar.values[p_mask] * ext_grids.in_service.values[p_mask]
        junction_idx_lookups = get_lookup(net, "node", "index")[node_name]
        junction = cls.get_connected_junction(net)
        juncts_p, press_sum, number = _sum_by_group(junction.values[p_mask], press,
                                                    np.ones_like(press, dtype=np.int32))
        index_p = junction_idx_lookups[juncts_p]
        node_pit[index_p, PINIT] = press_sum / number
        node_pit[index_p, NODE_TYPE] = P
        node_pit[index_p, EXT_GRID_OCCURENCE] += number

        t_mask = np.where(np.isin(ext_grids.type.values, ["t", "pt"]))
        t_k = ext_grids.t_k.values[t_mask] * ext_grids.in_service.values[t_mask]
        juncts_t, t_sum, number = _sum_by_group(junction.values[t_mask], t_k,
                                                np.ones_like(t_k, dtype=np.int32))
        index = junction_idx_lookups[juncts_t]
        node_pit[index, TINIT] = t_sum / number
        node_pit[index, NODE_TYPE_T] = T
        node_pit[index, EXT_GRID_OCCURENCE_T] += number

        net["_lookups"]["ext_grid"] = \
            np.array(list(set(np.concatenate([net["_lookups"]["ext_grid"], index_p])))) if \
            "ext_grid" in net['_lookups'] else index_p
        return ext_grids, press

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
        ext_grids = net[cls.table_name()]

        if len(ext_grids) == 0:
            return

        res_table = super().extract_results(net, options, node_name)

        branch_pit = net['_pit']['branch']
        node_pit = net["_pit"]["node"]

        p_grids = np.isin(ext_grids.type.values, ["p", "pt"])
        junction = cls.get_connected_junction(net)
        index_juncts = np.array(junction.values[p_grids])
        junct_uni = np.array(list(set(index_juncts)))
        index_nodes = get_lookup(net, "node", "index")[node_name][junct_uni]
        eg_from_branches = np.isin(branch_pit[:, FROM_NODE], index_nodes)
        eg_to_branches = np.isin(branch_pit[:, TO_NODE], index_nodes)
        from_nodes = branch_pit[eg_from_branches, FROM_NODE]
        to_nodes = branch_pit[eg_to_branches, TO_NODE]
        mass_flow_from = branch_pit[eg_from_branches, LOAD_VEC_NODES]
        mass_flow_to = branch_pit[eg_to_branches, LOAD_VEC_NODES]
        loads = node_pit[index_nodes, LOAD]
        counts = node_pit[index_nodes, EXT_GRID_OCCURENCE]
        all_index_nodes = np.concatenate([from_nodes, to_nodes, index_nodes])
        all_mass_flows = np.concatenate([-mass_flow_from, mass_flow_to, -loads])
        nodes, sum_mass_flows = _sum_by_group(all_index_nodes, all_mass_flows)

        # positive results mean that the ext_grid feeds in, negative means that the ext grid
        # extracts (like a load)
        res_table["mdot_kg_per_s"].values[p_grids] = np.repeat(cls.sign() * sum_mass_flows / counts,
                                                               counts.astype(int))
        return res_table, ext_grids, index_nodes, node_pit, branch_pit

    @classmethod
    def get_connected_junction(cls, net):
        junction = net[cls.table_name()].junction
        return junction

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
