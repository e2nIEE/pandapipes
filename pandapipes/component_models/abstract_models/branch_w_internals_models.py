# Copyright (c) 2020-2021 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np

from pandapipes.component_models.abstract_models.branch_models import BranchComponent
from pandapipes.constants import NORMAL_PRESSURE, NORMAL_TEMPERATURE
from pandapipes.idx_branch import ACTIVE
from pandapipes.idx_branch import FROM_NODE, TO_NODE, TINIT, RHO, ETA, \
    VINIT, RE, LAMBDA, CP, ELEMENT_IDX
from pandapipes.idx_node import L, node_cols
from pandapipes.idx_node import PINIT, TINIT as TINIT_NODE, PAMB
from pandapipes.pf.internals_toolbox import _sum_by_group
from pandapipes.pf.pipeflow_setup import add_table_lookup, get_lookup, get_table_number, \
    get_net_option
from pandapipes.properties.fluids import get_fluid

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class BranchWInternalsComponent(BranchComponent):
    """

    """

    @classmethod
    def table_name(cls):
        raise NotImplementedError

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
    def calculate_temperature_lift(cls, net, pipe_pit, node_pit):
        raise NotImplementedError

    @classmethod
    def from_to_node_cols(cls):
        raise NotImplementedError

    @classmethod
    def internal_node_name(cls):
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
        internal_nodes = cls.get_internal_pipe_number(net) - 1
        end = current_start
        ft_lookups[cls.table_name()] = None
        if np.any(internal_nodes > 0):
            int_nodes_num = int(np.sum(internal_nodes))
            internal_pipes = internal_nodes + 1
            int_pipes_num = int(np.sum(internal_pipes))
            end = current_start + int_nodes_num
            add_table_lookup(table_lookup, cls.internal_node_name(), current_table)
            ft_lookups[cls.internal_node_name()] = (current_start, end)
            return end, current_table + 1, internal_nodes, internal_pipes, int_nodes_num, int_pipes_num
        else:
            return end, current_table + 1, 0, 0, 0, 0

    @classmethod
    def create_branch_lookups(cls, net, ft_lookups, table_lookup, idx_lookups, current_table,
                              current_start):
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
        :return:
        :rtype:
        """
        end = current_start + int(np.sum(cls.get_internal_pipe_number(net)))
        ft_lookups[cls.table_name()] = (current_start, end)
        add_table_lookup(table_lookup, cls.table_name(), current_table)
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
        table_lookup = get_lookup(net, "node", "table")
        table_nr = get_table_number(table_lookup, cls.internal_node_name())
        if table_nr is None:
            return None, 0, 0, None, None, None
        ft_lookup = get_lookup(net, "node", "from_to")
        f, t = ft_lookup[cls.internal_node_name()]

        int_node_pit = node_pit[f:t, :]
        int_node_pit[:, :] = np.array([table_nr, 0, L] + [0] * (node_cols - 3))
        int_node_number = cls.get_internal_pipe_number(net) - 1

        int_node_pit[:, ELEMENT_IDX] = np.arange(t - f)

        f_junction, t_junction = ft_lookup[cls.get_connected_node_type().table_name()]
        junction_pit = node_pit[f_junction:t_junction, :]
        from_junctions = net[cls.table_name()].from_junction.values.astype(np.int32)
        to_junctions = net[cls.table_name()].to_junction.values.astype(np.int32)
        return table_nr, int_node_number, int_node_pit, junction_pit, from_junctions, to_junctions

    @classmethod
    def create_pit_branch_entries(cls, net, branch_pit):
        """
        Function which creates pit branch entries.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param branch_pit:
        :type branch_pit:
        :return: No Output.
        """
        branch_w_internals_pit, node_pit, from_nodes, to_nodes \
            = super().create_pit_branch_entries(net, branch_pit)

        if not len(branch_w_internals_pit):
            return branch_w_internals_pit, []

        internal_pipe_number = cls.get_internal_pipe_number(net)
        node_ft_lookups = get_lookup(net, "node", "from_to")

        if cls.internal_node_name() in node_ft_lookups:
            pipe_nodes_from, pipe_nodes_to = node_ft_lookups[cls.internal_node_name()]
            pipe_nodes_idx = np.arange(pipe_nodes_from, pipe_nodes_to)
            insert_places = np.repeat(np.arange(len(from_nodes)), internal_pipe_number - 1)
            from_nodes = np.insert(from_nodes, insert_places + 1, pipe_nodes_idx)
            to_nodes = np.insert(to_nodes, insert_places, pipe_nodes_idx)

        branch_w_internals_pit[:, ELEMENT_IDX] = np.repeat(net[cls.table_name()].index.values,
                                                           internal_pipe_number)
        branch_w_internals_pit[:, FROM_NODE] = from_nodes
        branch_w_internals_pit[:, TO_NODE] = to_nodes
        branch_w_internals_pit[:, TINIT] = (node_pit[from_nodes, TINIT_NODE] + node_pit[
            to_nodes, TINIT_NODE]) / 2
        fluid = get_fluid(net)
        branch_w_internals_pit[:, RHO] = fluid.get_density(branch_w_internals_pit[:, TINIT])
        branch_w_internals_pit[:, ETA] = fluid.get_viscosity(branch_w_internals_pit[:, TINIT])
        branch_w_internals_pit[:, CP] = fluid.get_heat_capacity(branch_w_internals_pit[:, TINIT])
        branch_w_internals_pit[:, ACTIVE] = \
            np.repeat(net[cls.table_name()][cls.active_identifier()].values, internal_pipe_number)

        return branch_w_internals_pit, internal_pipe_number

    @classmethod
    def extract_results(cls, net, options, branch_results, nodes_connected, branches_connected):
        # # placement_table, branch_pit, res_table = cls.prepare_result_tables(net, options, node_name)
        # # res_table = Component.extract_results(net, options, node_name, None)
        # res_table = net["res_" + cls.table_name()]
        #
        # f, t = get_lookup(net, "branch", "from_to")[cls.table_name()]
        # fa, ta = get_lookup(net, "branch", "from_to_active")[cls.table_name()]
        #
        # placement_table = np.argsort(net[cls.table_name()].index.values)
        # idx_pit = net["_pit"]["branch"][f:t, ELEMENT_IDX]
        # pipe_considered = get_lookup(net, "branch", "active")[f:t]
        # _, active_pipes = _sum_by_group(get_net_option(net, "use_numba"), idx_pit,
        #                                 pipe_considered.astype(np.int32))
        # active_pipes = active_pipes > 0.99
        # placement_table = placement_table[active_pipes]
        # branch_pit = net["_active_pit"]["branch"][fa:ta, :]
        #
        # node_pit = net["_active_pit"]["node"]
        #
        # if not len(branch_pit):
        #     return placement_table, res_table, branch_pit, node_pit
        #
        # node_active_idx_lookup = get_lookup(net, "node", "index_active")[cls.get_connected_node_type().table_name()]
        # junction_idx_lookup = get_lookup(net, "node", "index")[cls.get_connected_node_type().table_name()]
        # from_junction_nodes = node_active_idx_lookup[junction_idx_lookup[
        #     net[cls.table_name()]["from_junction"].values[placement_table]]]
        # to_junction_nodes = node_active_idx_lookup[junction_idx_lookup[
        #     net[cls.table_name()]["to_junction"].values[placement_table]]]
        #
        # from_nodes = branch_pit[:, FROM_NODE].astype(np.int32)
        # to_nodes = branch_pit[:, TO_NODE].astype(np.int32)
        # fluid = get_fluid(net)
        #
        # v_mps = branch_results["v_mps"][fa:ta]
        #
        # t0 = node_pit[from_nodes, TINIT_NODE]
        # t1 = node_pit[to_nodes, TINIT_NODE]
        # mf = branch_pit[:, LOAD_VEC_NODES]
        # vf = branch_pit[:, LOAD_VEC_NODES] / get_fluid(net).get_density((t0 + t1) / 2)
        #
        # use_numba = get_net_option(net, "use_numba")
        # idx_active = branch_pit[:, ELEMENT_IDX]
        # _, v_sum, mf_sum, vf_sum, internal_pipes = _sum_by_group(use_numba, idx_active, v_mps, mf,
        #                                                          vf, np.ones_like(idx_active))
        #
        # if fluid.is_gas:
        #     # derived from the ideal gas law
        #     p_from = node_pit[from_nodes, PAMB] + node_pit[from_nodes, PINIT]
        #     p_to = node_pit[to_nodes, PAMB] + node_pit[to_nodes, PINIT]
        #     numerator = NORMAL_PRESSURE * branch_pit[:, TINIT]
        #     normfactor_from = numerator * fluid.get_property("compressibility", p_from) \
        #                       / (p_from * NORMAL_TEMPERATURE)
        #     normfactor_to = numerator * fluid.get_property("compressibility", p_to) \
        #                     / (p_to * NORMAL_TEMPERATURE)
        #     v_gas_from = v_mps * normfactor_from
        #     v_gas_to = v_mps * normfactor_to
        #
        #     _, nf_from_sum, nf_to_sum = _sum_by_group(use_numba, idx_active, normfactor_from,
        #                                               normfactor_to)
        #
        #     v_gas_from_ordered = select_from_pit(from_nodes, from_junction_nodes, v_gas_from)
        #     v_gas_to_ordered = select_from_pit(to_nodes, to_junction_nodes, v_gas_to)
        #
        #     res_table["v_from_m_per_s"].values[placement_table] = v_gas_from_ordered
        #     res_table["v_to_m_per_s"].values[placement_table] = v_gas_to_ordered
        #     res_table["normfactor_from"].values[placement_table] = nf_from_sum / internal_pipes
        #     res_table["normfactor_to"].values[placement_table] = nf_to_sum / internal_pipes
        #
        # res_table["p_from_bar"].values[placement_table] = node_pit[from_junction_nodes, PINIT]
        # res_table["p_to_bar"].values[placement_table] = node_pit[to_junction_nodes, PINIT]
        # res_table["t_from_k"].values[placement_table] = node_pit[from_junction_nodes, TINIT_NODE]
        # res_table["t_to_k"].values[placement_table] = node_pit[to_junction_nodes, TINIT_NODE]
        # res_table["mdot_to_kg_per_s"].values[placement_table] = -mf_sum / internal_pipes
        # res_table["mdot_from_kg_per_s"].values[placement_table] = mf_sum / internal_pipes
        # res_table["vdot_norm_m3_per_s"].values[placement_table] = vf_sum / internal_pipes
        # return placement_table, res_table, branch_pit, node_pit
        #
        #
        #
        #
        #

        placement_table, res_table, branch_pit, node_pit = super().extract_results(net, options,
                                                                                   branch_results,
                                                                                   nodes_connected,
                                                                                   branches_connected)
        fluid = get_fluid(net)
        use_numba = get_net_option(net, "use_numba")

        idx_active = branch_pit[:, ELEMENT_IDX]
        v_mps = branch_pit[:, VINIT]
        _, v_sum, internal_pipes = _sum_by_group(use_numba, idx_active, v_mps, np.ones_like(idx_active))
        idx_pit = branch_pit[:, ELEMENT_IDX]
        _, lambda_sum, reynolds_sum, = \
            _sum_by_group(use_numba, idx_pit, branch_pit[:, LAMBDA], branch_pit[:, RE])
        if fluid.is_gas:
            from_nodes = branch_pit[:, FROM_NODE].astype(np.int32)
            to_nodes = branch_pit[:, TO_NODE].astype(np.int32)
            numerator = NORMAL_PRESSURE * branch_pit[:, TINIT]
            p_from = node_pit[from_nodes, PAMB] + node_pit[from_nodes, PINIT]
            p_to = node_pit[to_nodes, PAMB] + node_pit[to_nodes, PINIT]
            mask = ~np.isclose(p_from, p_to)
            p_mean = np.empty_like(p_to)
            p_mean[~mask] = p_from[~mask]
            p_mean[mask] = 2 / 3 * (p_from[mask] ** 3 - p_to[mask] ** 3) \
                           / (p_from[mask] ** 2 - p_to[mask] ** 2)
            normfactor_mean = numerator * fluid.get_property("compressibility", p_mean) \
                              / (p_mean * NORMAL_TEMPERATURE)
            v_gas_mean = v_mps * normfactor_mean
            _, v_gas_mean_sum = _sum_by_group(use_numba, idx_active, v_gas_mean)
            res_table["v_mean_m_per_s"].values[placement_table] = v_gas_mean_sum / internal_pipes
        else:
            res_table["v_mean_m_per_s"].values[placement_table] = v_sum / internal_pipes
        res_table["lambda"].values[placement_table] = lambda_sum / internal_pipes
        res_table["reynolds"].values[placement_table] = reynolds_sum / internal_pipes

    @classmethod
    def get_internal_pipe_number(cls, net):
        """

        :param net: The pandapipes network
        :type net: pandapipesNet
        :return:
        :rtype:
        """
        return net[cls.table_name()].sections.values

    @classmethod
    def get_internal_results(cls, net, branch):
        """

        :param net:
        :type net:
        :param branch:
        :type branch:
        :return:
        :rtype:
        """
        raise NotImplementedError
