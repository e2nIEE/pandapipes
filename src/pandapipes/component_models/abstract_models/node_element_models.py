# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from abc import abstractmethod
from pandapipes.component_models.abstract_models.base_component import Component
from pandapipes.component_models.junction_component import Junction

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class NodeElementComponent(Component):
    """

    """
    @property
    @abstractmethod
    def sign(self):
        ...

    @property
    def connected_node_type(self):
        return Junction

    @abstractmethod
    def create_pit_node_entries(self, net, node_pit):
        """
        Function that creates pit node entries.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param node_pit:
        :type node_pit:
        :return: No Output.
        """
        ...

    def get_result_table(self, net):
        """Get results.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :return: (columns, all_float) - the column names and whether they are all float type. Only
                if False, returns columns as tuples also specifying the dtypes
        :rtype: (list, bool)
        """
        return ["mdot_kg_per_s"], True
