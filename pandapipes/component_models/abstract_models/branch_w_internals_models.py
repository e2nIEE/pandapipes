# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np

from pandapipes.component_models.abstract_models.branch_models import BranchComponent
from pandapipes.component_models.component_toolbox import set_entry_check_repeat
from pandapipes.idx_branch import ACTIVE, FROM_NODE, TO_NODE, TINIT, RHO, ETA, CP, ELEMENT_IDX
from pandapipes.idx_node import L, node_cols, TINIT as TINIT_NODE
from pandapipes.pf.pipeflow_setup import add_table_lookup, get_lookup, get_table_number
from pandapipes.properties.fluids import get_fluid

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class BranchWInternalsComponent(BranchComponent):
    """

    """

    @classmethod
    def table_name(cls):
        raise NotImplementedError

    @classmethod
    def get_component_input(cls):
        raise NotImplementedError

    @classmethod
    def get_result_table(cls, net):
        raise NotImplementedError

    @classmethod
    def active_identifier(cls):
        raise NotImplementedError

    @classmethod
    def calculate_temperature_lift(cls, net, branch_component_pit, node_pit):
        raise NotImplementedError

    @classmethod
    def from_to_node_cols(cls):
        raise NotImplementedError

    @classmethod
    def internal_node_name(cls):
        """

        :return: internal_node_name - name of the internal nodes for this class
        :rtype: str
        """
        raise NotImplementedError

    @classmethod
    def get_connected_node_type(cls):
        raise NotImplementedError

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
            return end, current_table + 1, internal_nodes, internal_pipes, int_nodes_num, \
                int_pipes_num
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
    def create_pit_node_entries(cls, net, node_pit):
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

        junction_table_name = cls.get_connected_node_type().table_name()
        fj_name, tj_name = "from_" + junction_table_name, "to_" + junction_table_name
        f_junction, t_junction = ft_lookup[junction_table_name]
        junction_pit = node_pit[f_junction:t_junction, :]
        from_junctions = net[cls.table_name()][fj_name].values.astype(np.int32)
        to_junctions = net[cls.table_name()][tj_name].values.astype(np.int32)
        junction_indices = get_lookup(net, "node", "index")[junction_table_name]
        fj_nodes = junction_indices[from_junctions]
        tj_nodes = junction_indices[to_junctions]
        return table_nr, int_node_number, int_node_pit, junction_pit, fj_nodes, tj_nodes

    @classmethod
    def create_pit_branch_entries(cls, net, branch_pit):
        """
        Function which creates pit branch entries.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param branch_pit:
        :type branch_pit:
        :return: No Output.
        """
        branch_w_internals_pit, node_pit, from_nodes, to_nodes \
            = super().create_pit_branch_entries(net, branch_pit)

        if not len(branch_w_internals_pit):
            return branch_w_internals_pit, np.array([], dtype=np.int32)

        internal_pipe_number = cls.get_internal_pipe_number(net).astype(np.int32)
        node_ft_lookups = get_lookup(net, "node", "from_to")

        has_internals = cls.internal_node_name() in node_ft_lookups
        if has_internals:
            pipe_nodes_from, pipe_nodes_to = node_ft_lookups[cls.internal_node_name()]
            pipe_nodes_idx = np.arange(pipe_nodes_from, pipe_nodes_to)
            insert_places = np.repeat(np.arange(len(from_nodes)), internal_pipe_number - 1)
            from_nodes = np.insert(from_nodes, insert_places + 1, pipe_nodes_idx)
            to_nodes = np.insert(to_nodes, insert_places, pipe_nodes_idx)

        set_entry_check_repeat(
            branch_w_internals_pit, ELEMENT_IDX, net[cls.table_name()].index.values,
            internal_pipe_number, has_internals)
        set_entry_check_repeat(
            branch_w_internals_pit, ACTIVE, net[cls.table_name()][cls.active_identifier()].values,
            internal_pipe_number, has_internals)
        branch_w_internals_pit[:, FROM_NODE] = from_nodes
        branch_w_internals_pit[:, TO_NODE] = to_nodes
        branch_w_internals_pit[:, TINIT] = (node_pit[from_nodes, TINIT_NODE] + node_pit[
            to_nodes, TINIT_NODE]) / 2
        fluid = get_fluid(net)
        branch_w_internals_pit[:, RHO] = fluid.get_density(branch_w_internals_pit[:, TINIT])
        branch_w_internals_pit[:, ETA] = fluid.get_viscosity(branch_w_internals_pit[:, TINIT])
        branch_w_internals_pit[:, CP] = fluid.get_heat_capacity(branch_w_internals_pit[:, TINIT])

        return branch_w_internals_pit, internal_pipe_number

    @classmethod
    def get_internal_pipe_number(cls, net):
        """

        :param net: The pandapipes network
        :type net: pandapipesNet
        :return:
        :rtype:
        """
        return np.array(net[cls.table_name()].sections.values)

    @classmethod
    def extract_results(cls, net, options, branch_results, nodes_connected, branches_connected):
        raise NotImplementedError

    @classmethod
    def get_internal_results(cls, net, branch):
        """

        :param net:
        :type net:
        :param branch:
        :type branch:
        :return:
        :rtype:
        """
        raise NotImplementedError
