# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np

from pandapipes.component_models.abstract_models.branch_models import BranchComponent
from pandapipes.component_models.component_toolbox import set_entry_check_repeat, get_internal_lookup_structure
from pandapipes.idx_branch import (
    ACTIVE,
    ELEMENT_IDX,
    D,
    LOSS_COEFFICIENT as LC,
    AREA,
    QEXT,
)
from pandapipes.idx_node import L, node_cols
from pandapipes.pf.pipeflow_setup import add_table_lookup, get_lookup, get_table_number, get_net_option

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
    def get_internal_node_number(cls, net, return_internal_only=True):
        raise NotImplementedError

    @classmethod
    def get_internal_branch_number(cls, net):
        return NotImplementedError

    @classmethod
    def create_node_lookups(cls, net, ft_lookups, table_lookup, idx_lookups, current_start, current_table, internals):
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
        :param internals:
        :type internals:
        :return:
        :rtype:
        """
        internal_nodes = cls.get_internal_node_number(net)
        internal_nodes_num = int(np.sum(internal_nodes))
        end = current_start + internal_nodes_num
        if internal_nodes_num > 0:
            add_table_lookup(table_lookup, cls.internal_node_name(), current_table)
            ft_lookups[cls.internal_node_name()] = (current_start, end)
            get_internal_lookup_structure(internals, cls.table_name(), internal_nodes, current_start)
        return end, current_table + 1

    @classmethod
    def create_branch_lookups(cls, net, ft_lookups, table_lookup, idx_lookups, current_start, current_table, internals):
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
        :param current_start:
        :type current_start:
        :param current_table:
        :type current_table:
        :param internals:
        :type internals:
        :return:
        :rtype:
        """
        internal_branches = cls.get_internal_branch_number(net)
        internal_branches_num = int(np.sum(internal_branches))
        end = current_start + internal_branches_num
        add_table_lookup(table_lookup, cls.table_name(), current_table)
        ft_lookups[cls.table_name()] = (current_start, end)
        get_internal_lookup_structure(internals, cls.table_name(), internal_branches)
        table_indices = net[cls.table_name()].index
        table_len = len(table_indices)
        if not table_len:
            idx_lookups[cls.table_name()] = np.array([], dtype=np.int32)
            idx_lookups[cls.table_name()][table_indices] = np.arange(table_len) + current_start
        else:
            idx_lookups[cls.table_name()] = -np.ones(table_indices.max() + 1, dtype=np.int32)
            idx_lookups[cls.table_name()][table_indices] = np.arange(table_len) + current_start
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
        if table_nr is not None:
            ft_lookup = get_lookup(net, "node", "from_to")
            f, t = ft_lookup[cls.internal_node_name()]

            int_node_pit = node_pit[f:t, :]
            if not get_net_option(net, "transient") or get_net_option(net, "simulation_time_step") == 0:
                int_node_pit[:, :] = np.array([table_nr, 0, L] + [0] * (node_cols - 3))
            return int_node_pit

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
        branch_w_internals_pit, node_pit = super().create_pit_branch_entries(net, branch_pit)

        if not len(branch_w_internals_pit):
            return branch_w_internals_pit, node_pit

        tbl = cls.table_name()
        node_ft_lookups = get_lookup(net, "node", "from_to")
        has_internals = cls.internal_node_name() in node_ft_lookups
        internal_branch_number = cls.get_internal_branch_number(net)
        if not get_net_option(net, "transient") or get_net_option(net, "simulation_time_step") == 0:
            set_entry_check_repeat(branch_w_internals_pit, ELEMENT_IDX, net[tbl].index.values, internal_branch_number,
                has_internals)
            set_entry_check_repeat(branch_w_internals_pit, ACTIVE, net[tbl][cls.active_identifier()].values,
                internal_branch_number, has_internals)
            set_entry_check_repeat(branch_w_internals_pit, D, net[tbl].diameter_m.values, internal_branch_number,
                has_internals)
            set_entry_check_repeat(branch_w_internals_pit, LC, net[tbl].loss_coefficient.values, internal_branch_number,
                has_internals)

            branch_w_internals_pit[:, AREA] = branch_w_internals_pit[:, D] ** 2 * np.pi / 4
            branch_w_internals_pit[:, QEXT] = 0.0
        return branch_w_internals_pit, node_pit

    @classmethod
    def extract_results(cls, net, options, branch_results, mode):
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
