# Copyright (c) 2020-2025 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from warnings import warn

import numpy as np
import pandas as pd
from numpy import dtype

from pandapipes.component_models.abstract_models.node_models import NodeComponent
from pandapipes.component_models.component_toolbox import p_correction_height_air
from pandapipes.idx_node import L, ELEMENT_IDX, PINIT, node_cols, HEIGHT, TINIT, PAMB, \
    ACTIVE as ACTIVE_ND, TINIT_OLD, EXT_GRID_OCCURENCE, EXT_GRID_OCCURENCE_T, LOAD
from pandapipes.pf.pipeflow_setup import add_table_lookup, get_table_number, \
    get_lookup
from pandapipes.pf.pipeflow_setup import get_net_option


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
        if not table_len:
            idx_lookups[cls.table_name()] = np.array([], dtype=np.int32)
            idx_lookups[cls.table_name()][table_indices] = np.arange(table_len) + current_start
        else:
            idx_lookups[cls.table_name()] = -np.ones(table_indices.max() + 1, dtype=np.int32)
            idx_lookups[cls.table_name()][table_indices] = np.arange(table_len) + current_start
        return end, current_table + 1

    @classmethod
    def create_pit_node_entries(cls, net, node_pit):
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

        if not get_net_option(net, "transient") or get_net_option(net, "simulation_time_step") == 0:
            junction_pit[:, :] = np.array([table_nr, 0, L] + [0] * (node_cols - 3))
            junction_pit[:, TINIT] = junctions.tfluid_k.values
            junction_pit[:, ELEMENT_IDX] = junctions.index.values
            junction_pit[:, HEIGHT] = junctions.height_m.values
            junction_pit[:, PINIT] = junctions.pn_bar.values
            junction_pit[:, TINIT] = junctions.tfluid_k.values
            junction_pit[:, PAMB] = p_correction_height_air(junction_pit[:, HEIGHT])
            junction_pit[:, ACTIVE_ND] = junctions.in_service.values
        else:
            junction_pit[:, EXT_GRID_OCCURENCE] = 0
            junction_pit[:, EXT_GRID_OCCURENCE_T] = 0
            junction_pit[:, LOAD] = 0

        if get_net_option(net, "transient"):
            junction_pit[:, TINIT_OLD] = junction_pit[:, TINIT]

    @classmethod
    def extract_results(cls, net, options, branch_results, mode):
        """
        Function that extracts certain results.

        :param mode:
        :type mode:
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
        res_table = net["res_" + cls.table_name()]

        if get_net_option(net, "transient"):
            # output, all_float = cls.get_result_table(net)
            # TODO: This must be made more precise in different components
            net["res_internal"] = pd.DataFrame(
                np.nan, columns=["t_k"], index=np.arange(len(net["_active_pit"]["node"][:,
                                                           TINIT])),
                dtype=np.float64
            )
            net["res_internal"]["t_k"] = net["_active_pit"]["node"][:, TINIT]

        f, t = get_lookup(net, "node", "from_to")[cls.table_name()]
        junction_pit = net["_pit"]["node"][f:t, :]

        if mode in ["hydraulics", "sequential", "bidirectional"]:
            junctions_connected_hydraulic = get_lookup(net, "node", "active_hydraulics")[f:t]

            if np.any(junction_pit[junctions_connected_hydraulic, PINIT] < 0):
                warn(UserWarning('Pipeflow converged, however, the results are physically incorrect '
                                 'as pressure is negative at nodes %s'
                                 % junction_pit[junction_pit[:, PINIT] < 0, ELEMENT_IDX]))

        #     res_table["p_bar"].values[junctions_connected_hydraulic] = junction_pit[:, PINIT]
        #     if mode == "hydraulics":
        #         res_table["t_k"].values[junctions_connected_hydraulic] = junction_pit[:, TINIT]
        #
        # if mode in ["heat", "sequential", "bidirectional]:
        #     junctions_connected_ht = get_lookup(net, "node", "active_heat_transfer")[f:t]
        #     res_table["t_k"].values[junctions_connected_ht] = junction_pit[:, TINIT]
        res_table["p_bar"].values[:] = junction_pit[:, PINIT]
        res_table["t_k"].values[:] = junction_pit[:, TINIT]

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
