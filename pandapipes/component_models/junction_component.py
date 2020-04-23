# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from warnings import warn
import numpy as np
from numpy import dtype

from pandapipes.component_models.auxiliaries.component_toolbox import p_correction_height_air
from pandapipes.component_models.abstract_models import NodeComponent

from pandapipes.idx_node import L, ELEMENT_IDX, RHO, PINIT, node_cols, HEIGHT, TINIT, PAMB, \
    ACTIVE as ACTIVE_ND

from pandapipes.pipeflow_setup import add_table_lookup, get_table_number, \
    get_lookup
from pandapipes.properties.fluids import get_fluid


class Junction(NodeComponent):
    """

    """

    @classmethod
    def table_name(cls):
        return "junction"

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
        table_indices = net[cls.table_name()].index
        table_len = len(table_indices)
        end = current_start + table_len
        ft_lookups[cls.table_name()] = (current_start, end)
        add_table_lookup(table_lookup, cls.table_name(), current_table)
        idx_lookups[cls.table_name()] = -np.ones(table_indices.max() + 1, dtype=np.int32)
        idx_lookups[cls.table_name()][table_indices] = np.arange(table_len) + current_start
        return end, current_table + 1

    @classmethod
    def create_pit_node_entries(cls, net, node_pit, node_name):
        """
        Function which creates pit node entries.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param node_pit:
        :type node_pit:
        :return: No Output.
        """
        ft_lookup = get_lookup(net, "node", "from_to")
        table_nr = get_table_number(get_lookup(net, "node", "table"), cls.table_name())
        f, t = ft_lookup[cls.table_name()]

        junctions = net[cls.table_name()]
        junction_pit = node_pit[f:t, :]
        junction_pit[:, :] = np.array([table_nr, 0, L] + [0] * (node_cols - 3))

        junction_pit[:, ELEMENT_IDX] = junctions.index.values
        junction_pit[:, HEIGHT] = junctions.height_m.values
        junction_pit[:, PINIT] = junctions.pn_bar.values
        junction_pit[:, TINIT] = junctions.tfluid_k.values
        junction_pit[:, RHO] = get_fluid(net).get_density(junction_pit[:, TINIT])
        junction_pit[:, PAMB] = p_correction_height_air(junction_pit[:, HEIGHT])
        junction_pit[:, ACTIVE_ND] = junctions.in_service.values

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
        res_table = super().extract_results(net, options, node_name)

        f, t = get_lookup(net, "node", "from_to")[cls.table_name()]
        fa, ta = get_lookup(net, "node", "from_to_active")[cls.table_name()]

        junction_pit = net["_active_pit"]["node"][fa:ta, :]
        junctions_active = get_lookup(net, "node", "active")[f:t]

        if np.any(junction_pit[:, PINIT] < 0):
            warn(UserWarning('Pipeflow converged, however, the results are phyisically incorrect '
                             'as pressure is at the nodes %s are negative'
                             % junction_pit[junction_pit[:, PINIT] < 0, ELEMENT_IDX]))

        res_table["p_bar"].values[junctions_active] = junction_pit[:, PINIT]
        res_table["t_k"].values[junctions_active] = junction_pit[:, TINIT]

    @classmethod
    def get_component_input(cls):
        """

        :return:
        :rtype:
        """
        return [('name', dtype(object)),
                ('pn_bar', 'f8'),
                ("tfluid_k", 'f8'),
                ("height_m", 'f8'),
                ('in_service', 'bool'),
                ('type', dtype(object))]

    @classmethod
    def geodata(cls):
        """

        :return:
        :rtype:
        """
        return [("x", "f8"), ("y", "f8")]

    @classmethod
    def get_result_table(cls, net):
        """

        :param net: The pandapipes network
        :type net: pandapipesNet
        :return: (columns, all_float) - the column names and whether they are all float type. Only
                if False, returns columns as tuples also specifying the dtypes
        :rtype: (list, bool)
        """
        return ["p_bar", "t_k"], True
