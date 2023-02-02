# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from pandapipes.component_models.abstract_models.branch_models import BranchComponent

from pandapipes.idx_branch import FROM_NODE, TO_NODE, TINIT, ELEMENT_IDX, RHO, ETA, CP, ACTIVE
from pandapipes.idx_node import TINIT as TINIT_NODE

from pandapipes.pf.pipeflow_setup import add_table_lookup
from pandapipes.properties.fluids import get_fluid

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
        branch_wo_internals_pit, node_pit, from_nodes, to_nodes \
            = super().create_pit_branch_entries(net, branch_pit)
        branch_wo_internals_pit[:, ELEMENT_IDX] = net[cls.table_name()].index.values
        branch_wo_internals_pit[:, FROM_NODE] = from_nodes
        branch_wo_internals_pit[:, TO_NODE] = to_nodes
        branch_wo_internals_pit[:, TINIT] = (node_pit[from_nodes, TINIT_NODE]
                                             + node_pit[to_nodes, TINIT_NODE]) / 2
        fluid = get_fluid(net)
        branch_wo_internals_pit[:, RHO] = fluid.get_density(branch_wo_internals_pit[:, TINIT])
        branch_wo_internals_pit[:, ETA] = fluid.get_viscosity(branch_wo_internals_pit[:, TINIT])
        branch_wo_internals_pit[:, CP] = fluid.get_heat_capacity(branch_wo_internals_pit[:, TINIT])
        branch_wo_internals_pit[:, ACTIVE] = net[cls.table_name()][cls.active_identifier()].values
        return branch_wo_internals_pit

    @classmethod
    def calculate_temperature_lift(cls, net, branch_component_pit, node_pit):
        raise NotImplementedError

    @classmethod
    def get_connected_node_type(cls):
        raise NotImplementedError

    @classmethod
    def extract_results(cls, net, options, branch_results, nodes_connected, branches_connected):
        raise NotImplementedError
