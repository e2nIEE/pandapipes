# Copyright (c) 2020-2021 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import copy
from operator import itemgetter

import numpy as np
import pandas as pd
from numpy import dtype

from pandapipes.component_models.abstract_models import NodeElementComponent
from pandapipes.internals_toolbox import _sum_by_group
from pandapipes.pipeflow_setup import get_lookup, add_table_lookup, get_table_number

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
    def node_element_relevant(cls, net):
        return True

    @classmethod
    def create_pit_node_element_entries(cls, net, node_element_pit, node_name):
        ext_grids = net[cls.table_name()]
        ext_grids = ext_grids[ext_grids.in_service.values]
        ext_grid_pit = super().create_pit_node_element_entries(net, node_element_pit, node_name)
        p_mask = np.where(np.isin(ext_grids.type.values, ["p", "pt"]))
        t_mask = np.where(np.isin(ext_grids.type.values, ["t"]))
        ext_grid_pit[p_mask, net._idx_node_element['NODE_ELEMENT_TYPE']] = net._idx_node_element['P']
        ext_grid_pit[t_mask, net._idx_node_element['NODE_ELEMENT_TYPE']] = net._idx_node_element['T']
        ext_grid_pit[:, net._idx_node_element['MINIT']] = 0.005
        return ext_grid_pit

    @classmethod
    def create_pit_node_entries(cls, net, node_pit, node_name):
        """
        Function which creates pit node entries.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param node_pit:
        :type node_pit:
        :param node_name:
        :type node_name:
        :return: No Output.
        """

        ext_grids = net[cls.table_name()]
        ext_grids = ext_grids[ext_grids.in_service.values]

        p_mask = np.where(np.isin(ext_grids.type.values, ["p", "pt"]))
        press = ext_grids.p_bar.values[p_mask]
        junction_idx_lookups = get_lookup(net, "node", "index")[node_name]
        junction = cls.get_connected_junction(net)
        juncts_p, press_sum, number = _sum_by_group(junction.values[p_mask], press,
                                                    np.ones_like(press, dtype=np.int32))
        index_p = junction_idx_lookups[juncts_p]
        node_pit[index_p, net['_idx_node']['PINIT']] = press_sum / number
        node_pit[index_p, net['_idx_node']['NODE_TYPE']] = net['_idx_node']['P']
        node_pit[index_p, net['_idx_node']['EXT_GRID_OCCURENCE']] += number

        t_mask = np.where(np.isin(ext_grids.type.values, ["t", "pt"]))
        t_k = ext_grids.t_k.values[t_mask]
        juncts_t, t_sum, number = _sum_by_group(junction.values[t_mask], t_k,
                                                np.ones_like(t_k, dtype=np.int32))
        index = junction_idx_lookups[juncts_t]
        node_pit[index, net['_idx_node']['TINIT']] = t_sum / number
        node_pit[index, net['_idx_node']['NODE_TYPE_T']] = net['_idx_node']['T']
        node_pit[index, net['_idx_node']['EXT_GRID_OCCURENCE_T']] += number

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
        :param node_name:
        :type node_name:
        :return: No Output.
        """

        ext_grids = net[cls.table_name()]

        if len(ext_grids) == 0:
            return

        #branch_pit = net['_pit']['branch']
        #node_pit = net["_pit"]["node"]

        #eg_nodes, p_grids, sum_mass_flows, counts, inverse_nodes, node_uni = \
        #    cls.get_mass_flow(net, ext_grids, node_pit, branch_pit, node_name)

        res_table = super().extract_results(net, options, node_name)

        f, t = get_lookup(net, "node_element", "from_to")[cls.table_name()]
        fa, ta = get_lookup(net, "node_element", "from_to_active")[cls.table_name()]

        node_element_pit  = net["_active_pit"]["node_element"][fa:ta, :]
        node_elements_active = get_lookup(net, "node_element", "active")[f:t]

        # positive results mean that the ext_grid feeds in, negative means that the ext grid
        # extracts (like a load)
        res_table["mdot_kg_per_s"].values[node_elements_active] = node_element_pit[:, net['_idx_node_element']['MINIT']]
        return res_table, ext_grids

    @classmethod
    def get_mass_flow(cls, net, ext_grids, node_pit, branch_pit, node_name):

        p_grids = np.isin(ext_grids.type.values, ["p", "pt"])
        junction = cls.get_connected_junction(net)
        eg_nodes = get_lookup(net, "node", "index")[node_name][np.array(junction.values[p_grids])]
        node_uni, inverse_nodes, counts = np.unique(eg_nodes, return_counts=True,
                                                    return_inverse=True)
        eg_from_branches = np.isin(branch_pit[:, net['_idx_branch']['FROM_NODE']], node_uni)
        eg_to_branches = np.isin(branch_pit[:, net['_idx_branch']['TO_NODE']], node_uni)
        from_nodes = branch_pit[eg_from_branches, net['_idx_branch']['FROM_NODE']]
        to_nodes = branch_pit[eg_to_branches, net['_idx_branch']['TO_NODE']]
        mass_flow_from = branch_pit[eg_from_branches, net['_idx_branch']['LOAD_VEC_NODES']]
        mass_flow_to = branch_pit[eg_to_branches, net['_idx_branch']['LOAD_VEC_NODES']]
        loads = node_pit[node_uni, net['_idx_node']['LOAD']]
        all_index_nodes = np.concatenate([from_nodes, to_nodes, node_uni])
        all_mass_flows = np.concatenate([-mass_flow_from, mass_flow_to, -loads])
        nodes, sum_mass_flows = _sum_by_group(all_index_nodes, all_mass_flows)
        return eg_nodes, p_grids, sum_mass_flows, counts, inverse_nodes, node_uni

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
                ("fluid", dtype(object)),
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
