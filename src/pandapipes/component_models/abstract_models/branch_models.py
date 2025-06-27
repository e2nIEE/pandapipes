# Copyright (c) 2020-2025 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np

from pandapipes.component_models.abstract_models.base_component import Component
from pandapipes.idx_branch import (
    MDOTINIT,
    branch_cols,
    TEXT,
    FLOW_RETURN_CONNECT,
)
from pandapipes.pf.pipeflow_setup import get_net_option
from pandapipes.pf.pipeflow_setup import get_table_number, get_lookup

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
    def get_component_input(cls):
        raise NotImplementedError

    @classmethod
    def get_result_table(cls, net):
        raise NotImplementedError

    @classmethod
    def from_to_node_cols(cls):
        raise NotImplementedError

    @classmethod
    def active_identifier(cls):
        raise NotImplementedError()

    @classmethod
    def get_connected_node_type(cls):
        raise NotImplementedError

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
        :return: No Output.
        """
        raise NotImplementedError

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
        node_pit = net["_pit"]["node"]
        f, t = get_lookup(net, "branch", "from_to")[cls.table_name()]
        branch_table_nr = get_table_number(get_lookup(net, "branch", "table"), cls.table_name())
        branch_component_pit = branch_pit[f:t, :]
        if not len(net[cls.table_name()]):
            return branch_component_pit, node_pit, [], []

        junction_idx_lookup = get_lookup(net, "node", "index")[
            cls.get_connected_node_type().table_name()]
        fn_col, tn_col = cls.from_to_node_cols()
        from_nodes = junction_idx_lookup[net[cls.table_name()][fn_col].values]
        to_nodes = junction_idx_lookup[net[cls.table_name()][tn_col].values]
        if not get_net_option(net, "transient") or get_net_option(net, "simulation_time_step") == 0:
            branch_component_pit[:, :] = np.array([branch_table_nr] + [0] * (branch_cols - 1))
            branch_component_pit[:, MDOTINIT] = 0.1
            branch_component_pit[:, TEXT] = get_net_option(net, 'ambient_temperature')
            branch_component_pit[:, FLOW_RETURN_CONNECT] = False
        return branch_component_pit, node_pit, from_nodes, to_nodes

    @classmethod
    def extract_results(cls, net, options, branch_results, mode):
        raise NotImplementedError
