# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import pandas as pd

from pandapipes.component_models.abstract_models.branch_models import BranchComponent
from pandapipes.component_models.component_toolbox import get_internal_lookup_structure, build_pit_entries
from pandapipes.idx_branch import IdxBranch
from pandapipes.idx_node import IdxNode
from pandapipes.pf.pipeflow_setup import add_table_lookup, get_lookup, get_table_number, get_net_option
from pandapipes.pf.system_index import PitEntries

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class BranchWInternalsComponent(BranchComponent):
    """Abstract base class for branch components with internal nodes."""

    @classmethod
    def table_name(cls):
        raise NotImplementedError

    @classmethod
    def active_identifier(cls):
        raise NotImplementedError

    @classmethod
    def get_connected_node_type(cls):
        raise NotImplementedError

    @classmethod
    def from_to_node_cols(cls):
        raise NotImplementedError

    @classmethod
    def internal_node_name(cls):
        """Return the name of the internal nodes for this class.

        :return: internal_node_name - name of the internal nodes for this class
        :rtype: str
        """
        raise NotImplementedError

    @classmethod
    def get_internal_node_number(cls, net, return_internal_only=True):
        raise NotImplementedError

    @classmethod
    def get_internal_branch_number(cls, net):
        return NotImplementedError

    @classmethod
    def get_component_input(cls):
        raise NotImplementedError

    @classmethod
    def create_node_lookups(cls, net, ft_lookups, table_lookup, idx_lookups, current_start, current_table, internals):
        """Function which creates node lookups.

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
        """Function which creates branch lookups.

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
    def register_pit_node_entries(cls, net, node_pit, registry) -> None:
        table_lookup = get_lookup(net, "node", "table")
        table_nr = get_table_number(table_lookup, cls.internal_node_name())
        if table_nr is None:
            return
        ft_lookup = get_lookup(net, "node", "from_to")
        f, t = ft_lookup[cls.internal_node_name()]
        if not get_net_option(net, "transient") or get_net_option(net, "simulation_time_step") == 0:
            rows = np.arange(f, t, dtype=np.int32)
            registry.add(PitEntries(*build_pit_entries(
                rows,
                [IdxNode.TABLE_IDX, IdxNode.NODE_TYPE],
                [float(table_nr), float(IdxNode.L)],
            )))

    @classmethod
    def register_pit_branch_entries(cls, net, branch_pit, node_pit, registry) -> None:
        super().register_pit_branch_entries(net, branch_pit, node_pit, registry)

        f, t = get_lookup(net, "branch", "from_to")[cls.table_name()]
        if not len(net[cls.table_name()]):
            return

        tbl = cls.table_name()
        node_ft_lookups = get_lookup(net, "node", "from_to")
        has_internals = cls.internal_node_name() in node_ft_lookups
        internal_branch_number = cls.get_internal_branch_number(net)
        rows = np.arange(f, t, dtype=np.int32)

        if not get_net_option(net, "transient") or get_net_option(net, "simulation_time_step") == 0:
            def _rep(vals):
                return np.repeat(vals, internal_branch_number) if has_internals else vals

            d_vals = _rep(net[tbl].inner_diameter_mm.values / 1000.)
            lc_vals = _rep(net[tbl].loss_coefficient.values)
            elem_idx_vals = _rep(net[tbl].index.values.astype(float))
            active_vals = _rep(net[tbl][cls.active_identifier()].values.astype(float))

            if "outer_diameter_mm" in net[tbl]:
                outer = net[tbl].outer_diameter_mm.values.copy()
                inner = net[tbl].inner_diameter_mm.values
                outer[pd.isnull(outer)] = inner[pd.isnull(outer)]
                do_vals = _rep(outer / 1000.)
                do_vals[np.isnan(do_vals)] = d_vals[np.isnan(do_vals)]
            else:
                do_vals = d_vals.copy()

            registry.add(PitEntries(*build_pit_entries(
                rows,
                [IdxBranch.ELEMENT_IDX, IdxBranch.ACTIVE, IdxBranch.D, IdxBranch.DO, IdxBranch.LOSS_COEFFICIENT],
                [elem_idx_vals, active_vals, d_vals, do_vals, lc_vals],
            )))

    @classmethod
    def calculate_temperature_lift(cls, net, branch_component_pit, node_pit):
        raise NotImplementedError

    @classmethod
    def extract_results(cls, net, options, branch_results, mode):
        raise NotImplementedError

    @classmethod
    def get_internal_results(cls, net, branch):
        """Get internal results for a branch.

        :param net:
        :type net:
        :param branch:
        :type branch:
        :return:
        :rtype:
        """
        raise NotImplementedError

    @classmethod
    def get_result_table(cls, net):
        raise NotImplementedError
