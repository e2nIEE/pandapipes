# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np

from pandapipes.idx_branch import FROM_NODE, TO_NODE, TINIT, ELEMENT_IDX, RHO, ETA, CP, ACTIVE
from pandapipes.idx_node import TINIT as TINIT_NODE, L, node_cols

from pandapipes.pipeflow_setup import add_table_lookup, get_lookup, get_table_number
from pandapipes.properties.fluids import get_fluid

from pandapipes.component_models.abstract_models.branch_models import BranchComponent

try:
    from numba import jit
except ImportError:
    from pandapower.pf.no_numba import jit

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class BranchWInternalsComponent(BranchComponent):
    """

    """

    @classmethod
    def internal_node_name(cls):
        return NotImplementedError

    @classmethod
    def create_node_lookups(cls, net, ft_lookups, table_lookup, idx_lookups, current_start,
                            current_table, internal_nodes_lookup):
        """
        Function which creates node lookups.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param ft_lookups:
        :type ft_lookups:
        :param table_lookup:
        :type table_lookup:
        :param idx_lookups:
        :type idx_lookups:
        :param current_start:
        :type current_start:
        :param current_table:
        :type current_table:
        :param internal_nodes_lookup:
        :type internal_nodes_lookup:
        :return:
        :rtype:
        """
        internal_nodes = cls.get_internal_pipe_number(net) - 1
        end = current_start
        ft_lookups[cls.table_name()] = None
        if np.any(internal_nodes > 0):
            int_nodes_num = int(np.sum(internal_nodes))
            internal_pipes = internal_nodes + 1
            int_pipes_num = int(np.sum(internal_pipes))
            end = current_start + int_nodes_num
            add_table_lookup(table_lookup, cls.internal_node_name(), current_table)
            ft_lookups[cls.internal_node_name()] = (current_start, end)
            return end, current_table + 1, internal_nodes, internal_pipes, int_nodes_num, int_pipes_num
        else:
            return end, current_table + 1, 0, 0, 0, 0

    @classmethod
    def create_branch_lookups(cls, net, ft_lookups, table_lookup, idx_lookups, current_table,
                              current_start):
        """
        Function which creates branch lookups.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param ft_lookups:
        :type ft_lookups:
        :param table_lookup:
        :type table_lookup:
        :param idx_lookups:
        :type idx_lookups:
        :param current_table:
        :type current_table:
        :param current_start:
        :type current_start:
        :return:
        :rtype:
        """
        end = current_start + int(np.sum(cls.get_internal_pipe_number(net)))
        ft_lookups[cls.table_name()] = (current_start, end)
        add_table_lookup(table_lookup, cls.table_name(), current_table)
        return end, current_table + 1

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
        table_lookup = get_lookup(net, "node", "table")
        table_nr = get_table_number(table_lookup, cls.internal_node_name())
        if table_nr is None:
            return None, 0, 0, None, None, None
        ft_lookup = get_lookup(net, "node", "from_to")
        f, t = ft_lookup[cls.internal_node_name()]

        int_node_pit = node_pit[f:t, :]
        int_node_pit[:, :] = np.array([table_nr, 0, L] + [0] * (node_cols - 3))
        int_node_number = cls.get_internal_pipe_number(net) - 1

        int_node_pit[:, ELEMENT_IDX] = np.arange(t - f)

        f_junction, t_junction = ft_lookup[node_name]
        junction_pit = node_pit[f_junction:t_junction, :]
        from_junctions = net[cls.table_name()].from_junction.values.astype(np.int32)
        to_junctions = net[cls.table_name()].to_junction.values.astype(np.int32)
        return table_nr, int_node_number, int_node_pit, junction_pit, from_junctions, to_junctions

    @classmethod
    def create_pit_branch_entries(cls, net, branch_winternals_pit, node_name):
        """
        Function which creates pit branch entries.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param branch_pit:
        :type branch_pit:
        :return: No Output.
        """
        branch_winternals_pit, node_pit, from_nodes, to_nodes \
            = super().create_pit_branch_entries(net, branch_winternals_pit, node_name)

        internal_pipe_number = cls.get_internal_pipe_number(net)
        node_ft_lookups = get_lookup(net, "node", "from_to")

        if cls.internal_node_name() in node_ft_lookups:
            pipe_nodes_from, pipe_nodes_to = node_ft_lookups[cls.internal_node_name()]
            pipe_nodes_idx = np.arange(pipe_nodes_from, pipe_nodes_to)
            insert_places = np.repeat(np.arange(len(from_nodes)), internal_pipe_number - 1)
            from_nodes = np.insert(from_nodes, insert_places + 1, pipe_nodes_idx)
            to_nodes = np.insert(to_nodes, insert_places, pipe_nodes_idx)

        branch_winternals_pit[:, ELEMENT_IDX] = np.repeat(net[cls.table_name()].index.values,
                                                          internal_pipe_number)
        branch_winternals_pit[:, FROM_NODE] = from_nodes
        branch_winternals_pit[:, TO_NODE] = to_nodes
        branch_winternals_pit[:, TINIT] = (node_pit[from_nodes, TINIT_NODE] + node_pit[
            to_nodes, TINIT_NODE]) / 2
        fluid = get_fluid(net)
        branch_winternals_pit[:, RHO] = fluid.get_density(branch_winternals_pit[:, TINIT])
        branch_winternals_pit[:, ETA] = fluid.get_viscosity(branch_winternals_pit[:, TINIT])
        branch_winternals_pit[:, CP] = fluid.get_heat_capacity(branch_winternals_pit[:, TINIT])
        branch_winternals_pit[:, ACTIVE] = \
            np.repeat(net[cls.table_name()][cls.active_identifier()].values,internal_pipe_number)

        return branch_winternals_pit, internal_pipe_number

    @classmethod
    def get_internal_pipe_number(cls, net):
        """

        :param net: The pandapipes network
        :type net: pandapipesNet
        :return:
        :rtype:
        """
        return net[cls.table_name()].sections.values

    @classmethod
    def get_internal_results(cls, net, branch):
        """

        :param net:
        :type net:
        :param pipe:
        :type pipe:
        :return:
        :rtype:
        """
        raise NotImplementedError
