# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np

from pandapipes.component_models.abstract_models.base_component import Component
from pandapipes.idx_branch import IdxBranch
from pandapipes.component_models.component_toolbox import build_pit_entries
from pandapipes.pf.pipeflow_setup import get_net_option, get_table_number, get_lookup
from pandapipes.pf.system_index import PitEntries

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class BranchComponent(Component):

    @classmethod
    def table_name(cls):
        raise NotImplementedError

    @classmethod
    def active_identifier(cls):
        raise NotImplementedError()

    @classmethod
    def get_connected_node_type(cls):
        raise NotImplementedError

    @classmethod
    def from_to_node_cols(cls):
        raise NotImplementedError

    @classmethod
    def get_component_input(cls):
        raise NotImplementedError

    @classmethod
    def create_branch_lookups(cls, net, ft_lookups, table_lookup, idx_lookups, current_start,
                              current_table, internals):
        """Function which creates branch lookups.

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
        raise NotImplementedError

    @classmethod
    def register_pit_branch_entries(cls, net, branch_pit, node_pit, registry) -> None:
        f, t = get_lookup(net, "branch", "from_to")[cls.table_name()]
        if not len(net[cls.table_name()]):
            return

        rows = np.arange(f, t, dtype=np.int32)

        if not get_net_option(net, "transient") or get_net_option(net, "simulation_time_step") == 0:
            branch_table_nr = get_table_number(get_lookup(net, "branch", "table"), cls.table_name())
            registry.add(PitEntries(*build_pit_entries(
                rows,
                [IdxBranch.TABLE_IDX],
                [float(branch_table_nr)],
            )))

    @classmethod
    def get_result_table(cls, net):
        raise NotImplementedError

    @classmethod
    def extract_results(cls, net, options, branch_results, mode):
        raise NotImplementedError
