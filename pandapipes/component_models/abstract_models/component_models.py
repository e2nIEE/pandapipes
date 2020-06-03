# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from pandapipes.component_models.auxiliaries.component_toolbox import init_results_element

try:
    from numba import jit
except ImportError:
    from pandapower.pf.no_numba import jit

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class Component:
    """

    """

    @classmethod
    def table_name(self):
        raise NotImplementedError()

    @classmethod
    def extract_results(cls, net, options, node_name):
        """
        Function that extracts certain results.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param options:
        :type options:
        :return: No Output.
        """
        output, all_float = cls.get_result_table(net)
        init_results_element(net, cls.table_name(), output, all_float)
        res_table = net["res_" + cls.table_name()]
        return res_table

    @classmethod
    def get_component_input(cls):
        """

        :return:
        :rtype:
        """
        raise NotImplementedError

    @classmethod
    def get_result_table(cls, net):
        """Get result table.

        Parameters:
        :param net:
        :type net:
        :return:
        :rtype:
        """
        raise NotImplementedError
