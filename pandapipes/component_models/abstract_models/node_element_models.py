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


class NodeElementComponent(Component):
    """

    """

    @classmethod
    def create_pit_node_entries(cls, net, node_pit, node_name):
        """
        Function that creates pit node entries.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param node_pit:
        :type node_pit:
        :return: No Output.
        """
        raise NotImplementedError
