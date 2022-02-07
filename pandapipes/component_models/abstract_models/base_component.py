# Copyright (c) 2020-2022 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from pandapipes.component_models.auxiliaries.component_toolbox import init_results_element

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class Component:

    @classmethod
    def table_name(cls):
        raise NotImplementedError()

    @classmethod
    def extract_results(cls, net, options, node_name):
        """
        Function that extracts certain results.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param options:
        :type options:
        :param node_name:
        :type node_name:
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
        """
        Get result table.

        :param net: a pandapipes net
        :type net: pandapipes.pandapipesNet
        :return:
        :rtype:
        """
        raise NotImplementedError

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
        :return:
        :rtype:
        """
        return current_start, current_table

    @classmethod
    def create_branch_lookups(cls, net, ft_lookups, table_lookup, idx_lookups, current_table, current_start):
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
        return current_start, current_table

    @classmethod
    def create_pit_node_entries(cls, net, node_pit, node_name):
        """
        Function which creates pit branch entries.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param branch_pit:
        :type branch_pit:
        :return: No Output.
        """

        pass

    @classmethod
    def create_pit_branch_entries(cls, net, branch_pit, node_name):
        """
        Function which creates pit branch entries.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param node_pit:
        :type node_pit:
        :return: No Output.
        """

        pass

    @classmethod
    def calculate_derivatives_hydraulic(cls, net, branch_pit, node_pit, idx_lookups, options):
        """
        Function which creates derivatives.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param branch_component_pit:
        :type branch_component_pit:
        :param node_pit:
        :type node_pit:
        :param idx_lookups:
        :type idx_lookups:
        :param options:
        :type options:
        :return: No Output.
        """
        pass

    @classmethod
    def calculate_derivatives_thermal(cls, net, branch_pit, node_pit, idx_lookups, options):
        """
        Function which creates derivatives.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param branch_component_pit:
        :type branch_component_pit:
        :param node_pit:
        :type node_pit:
        :param idx_lookups:
        :type idx_lookups:
        :param options:
        :type options:
        :return: No Output.
        """
        pass
