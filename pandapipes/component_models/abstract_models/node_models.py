# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

try:
    from numba import jit
except ImportError:
    from pandapower.pf.no_numba import jit

from pandapipes.component_models.abstract_models.component_models import Component

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class NodeComponent(Component):
    """

    """

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
        :return: No Output.
        """
        raise NotImplementedError

    @classmethod
    def create_pit_node_entries(cls, net, node_pit, node_name):
        """

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param node_pit:
        :type node_pit:
        :return: No Output.
        """
        raise NotImplementedError
