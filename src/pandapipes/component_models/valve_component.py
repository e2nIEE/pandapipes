# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype

from pandapipes.component_models.component_toolbox import standard_branch_wo_internals_result_lookup
from pandapipes.component_models.abstract_models.branch_w_internals_models import \
    BranchWInternalsComponent
from pandapipes.component_models.junction_component import Junction
from pandapipes.idx_node import TINIT as TINIT_NODE, HEIGHT, PINIT, ACTIVE as ACTIVE_ND, PAMB, TINIT_OLD
from pandapipes.idx_branch import LENGTH, K, TEXT, ALPHA, FROM_NODE, TO_NODE, TOUTINIT
from pandapipes.pf.result_extraction import extract_branch_results_without_internals
from pandapipes.pf.pipeflow_setup import get_fluid, get_net_option, get_lookup


class Valve(BranchWInternalsComponent):
    """
    Valves are branch elements that can separate two junctions.
    They have a length of 0, but can introduce a lumped pressure loss.
    """

    @classmethod
    def from_to_node_cols(cls):
        return "junction", "element"

    @classmethod
    def table_name(cls):
        return "valve"

    @classmethod
    def active_identifier(cls):
        return "opened"

    @classmethod
    def get_connected_node_type(cls):
        return Junction

    @classmethod
    def internal_node_name(cls):
        return "valve_nodes"

    @classmethod
    def get_internal_node_number(cls, net):
        """

        :param net: The pandapipes network
        :type net: pandapipesNet
        :return:
        :rtype:
        """

        int_nodes = np.zeros(len(net[cls.table_name()]), dtype=np.int32)
        mask_p = np.flatnonzero(net[cls.table_name()]['et'].values == 'p')
        val = net[cls.table_name()][['junction', 'element']].iloc[mask_p].values
        _, idx = np.unique(val, return_index=True, axis=0)
        int_nodes[mask_p[idx]] = 1
        return int_nodes

    @classmethod
    def get_internal_branch_number(cls, net):
        """

        :param net: The pandapipes network
        :type net: pandapipesNet
        :return:
        :rtype:
        """

        return np.ones(len(net[cls.table_name()]), dtype=np.int32)

    @classmethod
    def create_pit_node_entries(cls, net, node_pit):
        int_node_pit = super().create_pit_node_entries(net, node_pit)
        if int_node_pit is not None:
            if not get_net_option(net, "transient") or get_net_option(net, "simulation_time_step") == 0:
                junction_table_name = cls.get_connected_node_type().table_name()
                ft_lookup = get_lookup(net, "node", "from_to")
                fj_name, _ = cls.from_to_node_cols()
                f_junction, t_junction = ft_lookup[junction_table_name]
                junction_pit = node_pit[f_junction:t_junction, :]
                internal_nodes_lookups = net._lookups['internal_nodes'][cls.table_name()]
                from_junctions = net[cls.table_name()].loc[internal_nodes_lookups[:, 0], fj_name].values.astype(np.int32)
                active = net[cls.table_name()].loc[internal_nodes_lookups[:, 0], cls.active_identifier()].values
                junction_indices = get_lookup(net, "node", "index")[junction_table_name]
                fj_nodes = junction_indices[from_junctions]

                int_node_pit[:, TINIT_NODE] = junction_pit[fj_nodes, TINIT_NODE]
                int_node_pit[:, HEIGHT] = junction_pit[fj_nodes, HEIGHT]
                int_node_pit[:, PINIT] = junction_pit[fj_nodes, PINIT]
                int_node_pit[:, PAMB] = junction_pit[fj_nodes, PAMB]
                int_node_pit[:, ACTIVE_ND] = active
            if get_net_option(net, "transient"):
                int_node_pit[:, TINIT_OLD] = int_node_pit[:, TINIT_NODE].astype(np.float64)

    @classmethod
    def create_pit_branch_entries(cls, net, branch_pit):
        """
        Function which creates pit branch entries with a specific table.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param branch_pit:
        :type branch_pit:
        :return: No Output.
        """
        valve_pit, node_pit = super().create_pit_branch_entries(net, branch_pit)
        if not get_net_option(net, "transient") or get_net_option(net, "simulation_time_step") == 0:
            internal_node_number = cls.get_internal_node_number(net)

            fn_col, tn_col = cls.from_to_node_cols()
            junction_idx_lookup = get_lookup(net, "node", "index")[Junction.table_name()]
            from_nodes = junction_idx_lookup[net[cls.table_name()][fn_col].values]
            has_internals = np.any(internal_node_number > 0)
            if has_internals:
                f, t = get_lookup(net, "branch", "from_to")['pipe']
                pipe_pit = branch_pit[f:t, :]
                mask_p = net[cls.table_name()]['et'].values == 'p'
                pipe_idx_lookup = get_lookup(net, "branch", "index")['pipe']
                pipe_idx = pipe_idx_lookup[net[cls.table_name()].loc[mask_p, 'element']]
                pipes = net['pipe'].loc[net[cls.table_name()].loc[mask_p, 'element'], ['from_junction', 'to_junction']]
                change_to_nodes = np.where(net[cls.table_name()]['junction'].values == pipes['to_junction'].values, True, False)
                internal_nodes_lookups = net._lookups['internal_nodes'][cls.table_name()]
                to_nodes = internal_nodes_lookups[:, 1]
                pipe_pit[pipe_idx[change_to_nodes], TO_NODE] = to_nodes
                pipe_pit[pipe_idx[~change_to_nodes], FROM_NODE] = to_nodes
            else:
                to_nodes = junction_idx_lookup[net[cls.table_name()][tn_col].values]

            valve_pit[:, FROM_NODE] = from_nodes
            valve_pit[:, TO_NODE] = to_nodes
            valve_pit[:, TOUTINIT] = node_pit[to_nodes, TINIT_NODE]
            valve_pit[:, LENGTH] = 0
            valve_pit[:, K] = 1000
            valve_pit[:, TEXT] = 293.15
            valve_pit[:, ALPHA] = 0

    @classmethod
    def get_component_input(cls):
        """

        :return:
        :rtype:
        """
        return [("name", dtype(object)),
                ("junction", "i8"),
                ("element", "i8"),
                ("et", dtype(object)),
                ("diameter_m", "f8"),
                ("opened", "bool"),
                ("loss_coefficient", "f8"),
                ("type", dtype(object))]

    @classmethod
    def extract_results(cls, net, options, branch_results, mode):
        required_results_hyd, required_results_ht = standard_branch_wo_internals_result_lookup(net)

        required_results_hyd.extend([("v_mean_m_per_s", "v_mps"), ("lambda", "lambda"), ("reynolds", "reynolds")])

        if get_fluid(net).is_gas:
            required_results_hyd.extend([("v_from_m_per_s", "v_gas_from"), ("v_to_m_per_s", "v_gas_to")])

        extract_branch_results_without_internals(net, branch_results, required_results_hyd,
                                                 required_results_ht, cls.table_name(), mode)

    @classmethod
    def get_result_table(cls, net):
        """

        :param net: The pandapipes network
        :type net: pandapipesNet
        :return: (columns, all_float) - the column names and whether they are all float type. Only
                if False, returns columns as tuples also specifying the dtypes
        :rtype: (list, bool)
        """
        if get_fluid(net).is_gas:
            output = ["v_from_m_per_s", "v_to_m_per_s", "v_mean_m_per_s", "p_from_bar", "p_to_bar",
                      "t_from_k", "t_to_k", "t_outlet_k", "mdot_from_kg_per_s", "mdot_to_kg_per_s",
                      "vdot_norm_m3_per_s", "reynolds", "lambda", "normfactor_from",
                      "normfactor_to"]
        else:
            output = ["v_mean_m_per_s", "p_from_bar", "p_to_bar", "t_from_k", "t_to_k", "t_outlet_k",
                      "mdot_from_kg_per_s", "mdot_to_kg_per_s", "vdot_m3_per_s", "reynolds",
                      "lambda"]
        return output, True
