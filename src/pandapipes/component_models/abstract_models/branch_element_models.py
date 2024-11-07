# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from numpy import pi

from pandapipes.component_models.abstract_models.branch_models import BranchComponent

from pandapipes.idx_branch import FROM_NODE, TO_NODE, TOUTINIT, ELEMENT_IDX, ACTIVE, LENGTH, K, TEXT, ALPHA, D, AREA
from pandapipes.idx_node import TINIT as TINIT_NODE

from pandapipes.pf.pipeflow_setup import add_table_lookup


class BranchElementComponent(BranchComponent):
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
        :return:
        :rtype:
        """
        end = current_start + len(net[self.table_name])
        ft_lookups[self.table_name] = (current_start, end)
        add_table_lookup(table_lookup, self.table_name, current_table)
        return end, current_table + 1

    def create_pit_branch_entries(self, net, branch_pit):
        """
        Function which creates pit branch entries with a specific table.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param branch_pit:
        :type branch_pit:
        :return: No Output.
        """
        branch_element_pit, node_pit, from_nodes, to_nodes \
            = super().create_pit_branch_entries(net, branch_pit)
        branch_element_pit[:, ELEMENT_IDX] = net[self.table_name].index.values
        branch_element_pit[:, FROM_NODE] = from_nodes
        branch_element_pit[:, TO_NODE] = to_nodes
        branch_element_pit[:, TOUTINIT] = node_pit[to_nodes, TINIT_NODE]
        branch_element_pit[:, ACTIVE] = net[self.table_name][self.active_identifier].values
        ##
        branch_element_pit[:, LENGTH] = 0
        branch_element_pit[:, K] = 1000
        branch_element_pit[:, TEXT] = 293.15
        branch_element_pit[:, ALPHA] = 0
        branch_element_pit[:, D] = 0.1
        branch_element_pit[:, AREA] = branch_element_pit[:, D] ** 2 * pi / 4
        return branch_element_pit
