# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from pandapipes.component_models.abstract_models.base_component import Component
from pandapipes.pf.pipeflow_setup import get_lookup, add_table_lookup, get_table_number
import numpy as np

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class NodeElementComponent(Component):
    """

    """

    @classmethod
    def node_element_relevant(cls, net):
        return False

    @classmethod
    def table_name(cls):
        raise NotImplementedError

    @classmethod
    def get_connected_node_type(cls):
        raise NotImplementedError

    #ToDo: remove as soon as the circulation pumps are redefined, replace by get_connected_node_type
    @classmethod
    def junction_column_name(cls):
        return 'junction'

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
    def create_node_element_lookups(cls, net, ft_lookups, table_lookup, idx_lookups, current_start,
                                    current_table):
        if cls.node_element_relevant(net):
            table_indices = net[cls.table_name()].index
            table_len = len(table_indices)
            end = current_start + table_len
            ft_lookups[cls.table_name()] = (current_start, end)
            add_table_lookup(table_lookup, cls.table_name(), current_table)
            if not table_len:
                idx_lookups[cls.table_name()] = np.array([], dtype=np.int32)
                idx_lookups[cls.table_name()][table_indices] = np.arange(table_len) + current_start
            else:
                idx_lookups[cls.table_name()] = -np.ones(table_indices.max() + 1, dtype=np.int32)
                idx_lookups[cls.table_name()][table_indices] = np.arange(table_len) + current_start
            return end, current_table + 1
        else:
            return current_start, current_table

    @classmethod
    def create_pit_node_entries(cls, net, node_pit):
        """
        Function that creates pit node entries.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param node_pit:
        :type node_pit:
        :return: No Output.
        """
        raise NotImplementedError

    @classmethod
    def create_pit_node_element_entries(cls, net, node_element_pit):
        if cls.node_element_relevant(net):
            ft_lookup = get_lookup(net, "node_element", "from_to")
            node_lookup = get_lookup(net, "node", "index")[cls.get_connected_node_type().table_name()]
            node_element_table_nr = get_table_number(get_lookup(net, "node_element", "table"), cls.table_name())
            f, t = ft_lookup[cls.table_name()]

            node_elements = net[cls.table_name()]
            node_element_pit = node_element_pit[f:t, :]
            node_element_pit[:, :] = np.array([node_element_table_nr] + [0] *
                                              (net['_idx_node_element']['node_element_cols'] - 1))
            node_element_pit[:, net['_idx_node']['ELEMENT_IDX']] = node_elements.index.values
            node_element_pit[:, net['_idx_node_element']['JUNCTION']] = \
                node_lookup[node_elements[cls.junction_column_name()].values]
            node_element_pit[:, net['_idx_node_element']['ACTIVE']] = node_elements.in_service.values
            if len(net._fluid) != 1:
                w_lookup = np.array(get_lookup(net, "node_element", "w"))
                flp, nep = np.where(node_elements.fluid.values == net._fluid[:, np.newaxis])
                node_element_pit[nep, w_lookup[flp]] = 1.
            return node_element_pit

    @classmethod
    def extract_results(cls, net, options, branch_results, mode):
        raise NotImplementedError
