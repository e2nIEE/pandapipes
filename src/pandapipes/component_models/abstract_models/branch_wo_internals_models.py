# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np

from pandapipes.component_models.abstract_models.branch_models import BranchComponent
from pandapipes.idx_branch import (FROM_NODE, TO_NODE, TOUTINIT, ELEMENT_IDX, ACTIVE, T_OUT_OLD, LENGTH, K, TEXT, ALPHA,
                                   D, AREA)
from pandapipes.idx_node import TINIT as TINIT_NODE
from pandapipes.pf.pipeflow_setup import add_table_lookup, get_net_option, get_lookup

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class BranchWOInternalsComponent(BranchComponent):

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
    def from_to_node_cols(cls):
        raise NotImplementedError

    @classmethod
    def get_connected_node_type(cls):
        raise NotImplementedError

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
        :param current_table:
        :type current_table:
        :param current_start:
        :type current_start:
        :param internals:
        :type internals:
        :return:
        :rtype:
        """
        end = current_start + len(net[cls.table_name()])
        ft_lookups[cls.table_name()] = (current_start, end)
        add_table_lookup(table_lookup, cls.table_name(), current_table)
        return end, current_table + 1

    @classmethod
    def create_pit_branch_entries(cls, net, branch_pit):
        """
        Function which creates pit branch entries with a specific table.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param branch_pit:
        :type branch_pit:
        :return: No Output.
        """
        branch_wo_internals_pit, node_pit = super().create_pit_branch_entries(net, branch_pit)
        junction_idx_lookup = get_lookup(net, "node", "index")[
            cls.get_connected_node_type().table_name()]
        fn_col, tn_col = cls.from_to_node_cols()
        from_nodes = junction_idx_lookup[net[cls.table_name()][fn_col].values]
        to_nodes = junction_idx_lookup[net[cls.table_name()][tn_col].values]

        if not len(branch_wo_internals_pit):
            return branch_wo_internals_pit

        if not get_net_option(net, "transient") or get_net_option(net, "simulation_time_step") == 0:
            branch_wo_internals_pit[:, FROM_NODE] = from_nodes
            branch_wo_internals_pit[:, TO_NODE] = to_nodes
            branch_wo_internals_pit[:, TOUTINIT] = node_pit[to_nodes, TINIT_NODE]
            branch_wo_internals_pit[:, ELEMENT_IDX] = net[cls.table_name()].index.values
            branch_wo_internals_pit[:, ACTIVE] = net[cls.table_name()][cls.active_identifier()].values
            branch_wo_internals_pit[:, LENGTH] = 0
            branch_wo_internals_pit[:, K] = 1e-3
            branch_wo_internals_pit[:, TEXT] = 293.15
            branch_wo_internals_pit[:, ALPHA] = 0
            branch_wo_internals_pit[:, D] = 0.1
            branch_wo_internals_pit[:, AREA] = branch_wo_internals_pit[:, D] ** 2 * np.pi / 4
        if get_net_option(net, "transient"):
            branch_wo_internals_pit[:, T_OUT_OLD] = branch_wo_internals_pit[:, TOUTINIT]
        return branch_wo_internals_pit

    @classmethod
    def calculate_temperature_lift(cls, net, branch_component_pit, node_pit):
        raise NotImplementedError

    @classmethod
    def extract_results(cls, net, options, branch_results, mode):
        raise NotImplementedError
