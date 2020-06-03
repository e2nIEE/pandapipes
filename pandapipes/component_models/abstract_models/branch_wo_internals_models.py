# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from pandapipes.component_models.abstract_models.branch_models import BranchComponent

from pandapipes.idx_branch import FROM_NODE, TO_NODE, TINIT, ELEMENT_IDX, RHO, ETA, CP, ACTIVE
from pandapipes.idx_node import TINIT as TINIT_NODE

from pandapipes.pipeflow_setup import add_table_lookup
from pandapipes.properties.fluids import get_fluid

try:
    from numba import jit
except ImportError:
    from pandapower.pf.no_numba import jit

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class BranchWOInternalsComponent(BranchComponent):
    """

    """

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
    def create_pit_branch_entries(cls, net, branch_wo_internals_pit, node_name):
        """
        Function which creates pit branch entries with a specific table.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param valve_pit:
        :type valve_pit:
        :param internal_pipe_number:
        :type internal_pipe_number:
        :return: No Output.
        """
        branch_wo_internals_pit, node_pit, from_nodes, to_nodes \
            = super().create_pit_branch_entries(net, branch_wo_internals_pit, node_name)
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
