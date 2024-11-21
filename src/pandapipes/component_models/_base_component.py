# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from abc import ABC, abstractmethod
from pandapipes.utils.internals import init_results_element


class Component(ABC):

    @property
    @abstractmethod
    def table_name(self):
        ...

    @property
    def active_identifier(self):
        return "in_service"

    @abstractmethod
    def extract_results(self, net, options, branch_results, mode):
        """
        Function that extracts certain results.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param options:
        :type options:
        :param branch_results:
        :type branch_results:
        :param mode:
        :type mode:
        :return: No Output.
        """
        ...

    @abstractmethod
    def get_component_input(self):
        """

        :return:
        :rtype:
        """
        ...

    @abstractmethod
    def get_result_table(self, net):
        """
        Get result table.

        :param net: a pandapipes net
        :type net: pandapipes.pandapipesNet
        :return:
        :rtype:
        """
        ...

    def init_results(self, net):
        """
        Function that intializes the result table for the component.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :return: No Output.
        """
        output, all_float = self.get_result_table(net)
        init_results_element(net, self.table_name, output, all_float)
        res_table = net["res_" + self.table_name]
        return res_table