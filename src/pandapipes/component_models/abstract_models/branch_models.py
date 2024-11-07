# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from abc import abstractmethod
from numpy import array

from pandapipes.component_models.abstract_models.base_component import Component
from pandapipes.component_models.junction_component import Junction
from pandapipes.idx_branch import MDOTINIT, branch_cols, TEXT
from pandapipes.pf.pipeflow_setup import get_table_number, get_lookup, get_net_option


class BranchComponent(Component):

    @property
    def from_to_node_cols(self):
        return "return_junction", "flow_junction"

    @property
    def connected_node_type(self):
        return Junction

    @abstractmethod
    def create_branch_lookups(self, net, ft_lookups, table_lookup, idx_lookups, current_table,
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
        ...

    def create_pit_branch_entries(self, net, branch_pit):
        """
        Function which creates pit branch entries.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param branch_pit:
        :type branch_pit:
        :return: No Output.
        """
        node_pit = net["_pit"]["node"]
        f, t = get_lookup(net, "branch", "from_to")[self.table_name]
        branch_table_nr = get_table_number(get_lookup(net, "branch", "table"), self.table_name)
        branch_component_pit = branch_pit[f:t, :]
        if not len(net[self.table_name]):
            return branch_component_pit, node_pit, [], []

        junction_idx_lookup = get_lookup(net, "node", "index")[
            self.connected_node_type.table_name]
        fn_col, tn_col = self.from_to_node_cols
        from_nodes = junction_idx_lookup[net[self.table_name][fn_col].values]
        to_nodes = junction_idx_lookup[net[self.table_name][tn_col].values]
        branch_component_pit[:, :] = array([branch_table_nr] + [0] * (branch_cols - 1))
        branch_component_pit[:, MDOTINIT] = 0.1
        branch_component_pit[:, TEXT] = get_net_option(net, 'ambient_temperature')
        return branch_component_pit, node_pit, from_nodes, to_nodes
