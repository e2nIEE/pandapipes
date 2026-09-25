# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from warnings import warn

import numpy as np
import pandas as pd
from numpy import dtype

from pandapipes.component_models.abstract_models.node_models import NodeComponent
from pandapipes.component_models.component_toolbox import (
    build_pit_entries, p_correction_height_air, get_thermal_options,
)
from pandapipes.idx_node import IdxNode
from pandapipes.pf.derivative_calculation import calculate_derivatives_node_thermal
from pandapipes.pf.system_index import PitEntries, ComponentEquations, ThermVarEq
from pandapipes.pf.pipeflow_setup import add_table_lookup, get_table_number, \
    get_lookup, get_net_option


class Junction(NodeComponent):
    """Junction node component."""

    @classmethod
    def table_name(cls):
        return "junction"

    @classmethod
    def get_component_input(cls):
        """Get the component input columns for this table.

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
    def create_node_lookups(cls, net, ft_lookups, table_lookup, idx_lookups, current_start,
                            current_table, internals):
        """Create node lookups.

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
        :param internals:
        :type internals:
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
    def register_pit_node_entries(cls, net, node_pit, registry) -> None:
        ft_lookup = get_lookup(net, "node", "from_to")
        table_nr = get_table_number(get_lookup(net, "node", "table"), cls.table_name())
        f, t = ft_lookup[cls.table_name()]
        junctions = net[cls.table_name()]
        rows = np.arange(f, t, dtype=np.int32)

        if not get_net_option(net, "transient") or get_net_option(net, "simulation_time_step") == 0:
            height_vals = junctions.height_m.values
            registry.add(PitEntries(*build_pit_entries(
                rows,
                [IdxNode.TABLE_IDX, IdxNode.ELEMENT_IDX, IdxNode.NODE_TYPE, IdxNode.HEIGHT, IdxNode.PAMB, IdxNode.ACTIVE, IdxNode.TINIT, IdxNode.PINIT],
                [float(table_nr), junctions.index.values.astype(float), float(IdxNode.L),
                 height_vals, p_correction_height_air(height_vals),
                 junctions.in_service.values.astype(float),
                 junctions.tfluid_k.values, junctions.pn_bar.values],
            )))
        else:
            registry.add(PitEntries(*build_pit_entries(
                rows,
                [IdxNode.TINIT, IdxNode.PINIT],
                [junctions.tfluid_k.values, junctions.pn_bar.values],
            )))

    @classmethod
    def register_thermal_equations(cls, net, branch_pit, node_pit, sys_idx, registry) -> None:
        node_pit_old = net["_active_old_pit"]["node"]

        fn_node, dfn_dt = calculate_derivatives_node_thermal(
            net, branch_pit, node_pit, node_pit_old, get_thermal_options(net)
        )

        stagnant = np.where(dfn_dt != 0)[0].astype(np.int32)
        if not len(stagnant):
            return

        # variables
        t_n_col = sys_idx.idx(ThermVarEq.TINIT, stagnant)

        # equation position node
        n_eq = sys_idx.idx(ThermVarEq.NODE, stagnant)

        # system matrix node
        rows_node = n_eq.astype(np.int32)
        cols_node = t_n_col.astype(np.int32)
        data_node = dfn_dt[stagnant].astype(np.float64)
        load_rows_node = n_eq.astype(np.int32)
        load_node = fn_node[stagnant].astype(np.float64)

        registry.add(ComponentEquations(
            rows=rows_node,
            cols=cols_node,
            data=data_node,
            load_rows=load_rows_node,
            load_data=load_node,
        ))

    @classmethod
    def geodata(cls):
        """Get geodata columns.

        :return:
        :rtype:
        """
        return [("x", "f8"), ("y", "f8")]

    @classmethod
    def get_result_table(cls, net):
        """Get the result table columns.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :return: (columns, all_float) - the column names and whether they are all float type. Only
                if False, returns columns as tuples also specifying the dtypes
        :rtype: (list, bool)
        """
        return ["p_bar", "t_k"], True

    @classmethod
    def extract_results(cls, net, options, branch_results, mode):
        """Extract certain results.

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
                                                           IdxNode.TINIT])),
                dtype=np.float64
            )
            net["res_internal"]["t_k"] = net["_active_pit"]["node"][:, IdxNode.TINIT]

        f, t = get_lookup(net, "node", "from_to")[cls.table_name()]
        junction_pit = net["_pit"]["node"][f:t, :]

        if mode in ["hydraulics", "sequential", "bidirectional"]:
            junctions_connected_hydraulic = get_lookup(net, "node", "active_hydraulics")[f:t]

            if np.any(junction_pit[junctions_connected_hydraulic, IdxNode.PINIT] < 0):
                warn(UserWarning('Pipeflow converged, however, the results are physically incorrect '
                                 'as pressure is negative at nodes %s'
                                 % junction_pit[junction_pit[:, IdxNode.PINIT] < 0, IdxNode.ELEMENT_IDX]))

        res_table["p_bar"].values[:] = junction_pit[:, IdxNode.PINIT]
        res_table["t_k"].values[:] = junction_pit[:, IdxNode.TINIT]
