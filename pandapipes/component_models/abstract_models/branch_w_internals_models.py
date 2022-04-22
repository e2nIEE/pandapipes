# Copyright (c) 2020-2022 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np

from pandapipes.component_models.abstract_models.branch_models import BranchComponent
from pandapipes.component_models.auxiliaries.component_toolbox import set_entry_check_repeat
from pandapipes.constants import NORMAL_PRESSURE, NORMAL_TEMPERATURE
from pandapipes.internals_toolbox import _sum_by_group
from pandapipes.pipeflow_setup import add_table_lookup, get_lookup, get_table_number
from pandapipes.properties.fluids import get_mixture_compressibility, is_fluid_gas, get_fluid

try:
    from pandaplan.core import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class BranchWInternalsComponent(BranchComponent):
    """

    """

    @classmethod
    def active_identifier(cls):
        raise NotImplementedError

    @classmethod
    def calculate_pressure_lift(cls, net, pipe_pit, node_pit):
        raise NotImplementedError

    @classmethod
    def calculate_temperature_lift(cls, net, pipe_pit, node_pit):
        raise NotImplementedError

    @classmethod
    def from_to_node_cols(cls):
        raise NotImplementedError

    @classmethod
    def internal_node_name(cls):
        return NotImplementedError

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
    def create_pit_node_entries(cls, net, node_pit, node_name):
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
        int_node_pit[:, :] = np.array([table_nr, 0, net['_idx_node']['L']] + [0] * (net['_idx_node']['node_cols'] - 3))
        int_node_number = cls.get_internal_pipe_number(net) - 1

        int_node_pit[:, net['_idx_node']['ELEMENT_IDX']] = np.arange(t - f)

        f_junction, t_junction = ft_lookup[node_name]
        junction_pit = node_pit[f_junction:t_junction, :]
        from_junctions = net[cls.table_name()].from_junction.values.astype(np.int32)
        to_junctions = net[cls.table_name()].to_junction.values.astype(np.int32)
        return table_nr, int_node_number, int_node_pit, junction_pit, from_junctions, to_junctions

    @classmethod
    def create_pit_branch_entries(cls, net, branch_winternals_pit, node_name):
        """
        Function which creates pit branch entries.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param branch_winternals_pit:
        :type branch_winternals_pit:
        :param node_name:
        :type node_name:
        :return: No Output.
        """
        branch_winternals_pit, node_pit, from_nodes, to_nodes \
            = super().create_pit_branch_entries(net, branch_winternals_pit, node_name)

        if not len(branch_winternals_pit):
            return branch_winternals_pit, np.array([], dtype=np.int32)

        internal_pipe_number = cls.get_internal_pipe_number(net).astype(np.int32)
        node_ft_lookups = get_lookup(net, "node", "from_to")

        has_internals = cls.internal_node_name() in node_ft_lookups
        if has_internals:
            pipe_nodes_from, pipe_nodes_to = node_ft_lookups[cls.internal_node_name()]
            pipe_nodes_idx = np.arange(pipe_nodes_from, pipe_nodes_to)
            insert_places = np.repeat(np.arange(len(from_nodes)), internal_pipe_number - 1)
            from_nodes = np.insert(from_nodes, insert_places + 1, pipe_nodes_idx)
            to_nodes = np.insert(to_nodes, insert_places, pipe_nodes_idx)
        set_entry_check_repeat(
            branch_winternals_pit, net['_idx_branch']['ELEMENT_IDX'], net[cls.table_name()].index.values,
            internal_pipe_number, has_internals)
        set_entry_check_repeat(
            branch_winternals_pit, net['_idx_branch']['ACTIVE'], net[cls.table_name()][cls.active_identifier()].values,
            internal_pipe_number, has_internals)
        branch_winternals_pit[:, net['_idx_branch']['FROM_NODE']] = from_nodes
        branch_winternals_pit[:, net['_idx_branch']['TO_NODE']] = to_nodes
        branch_winternals_pit[:, net['_idx_branch']['TINIT']] = (node_pit[from_nodes, net['_idx_node']['TINIT']] +
                                                                 node_pit[to_nodes, net['_idx_node']['TINIT']]) / 2

        if len(net._fluid) == 1:
            branch_winternals_pit[:, net['_idx_branch']['RHO']] = \
                get_fluid(net, net._fluid[0]).get_density(branch_winternals_pit[:, net['_idx_branch']['TINIT']])
            branch_winternals_pit[:, net['_idx_branch']['ETA']] = \
                get_fluid(net, net._fluid[0]).get_viscosity(branch_winternals_pit[:, net['_idx_branch']['TINIT']])
            branch_winternals_pit[:, net['_idx_branch']['CP']] = \
                get_fluid(net, net._fluid[0]).get_heat_capacity(branch_winternals_pit[:, net['_idx_branch']['TINIT']])
        else:
            for fluid in net._fluid:
                branch_winternals_pit[:, net['_idx_branch'][fluid + '_RHO']] = \
                    get_fluid(net, fluid).get_density(branch_winternals_pit[:, net['_idx_branch']['TINIT']])

        return branch_winternals_pit, internal_pipe_number

    @classmethod
    def extract_results(cls, net, options, node_name):
        placement_table, res_table, branch_pit, node_pit = super().extract_results(net, options, node_name)

        idx_active = branch_pit[:, net['_idx_branch']['ELEMENT_IDX']]
        v_mps = branch_pit[:, net['_idx_branch']['VINIT']]
        _, v_sum, internal_pipes = _sum_by_group(idx_active, v_mps, np.ones_like(idx_active))
        idx_pit = branch_pit[:, net['_idx_branch']['ELEMENT_IDX']]
        _, lambda_sum, reynolds_sum, = \
            _sum_by_group(idx_pit, branch_pit[:, net['_idx_branch']['LAMBDA']], branch_pit[:, net['_idx_branch']['RE']])
        if is_fluid_gas(net):
            from_nodes = branch_pit[:, net['_idx_branch']['FROM_NODE']].astype(np.int32)
            to_nodes = branch_pit[:, net['_idx_branch']['TO_NODE']].astype(np.int32)
            numerator = NORMAL_PRESSURE * branch_pit[:, net['_idx_branch']['TINIT']]
            p_from = node_pit[from_nodes, net['_idx_node']['PAMB']] + node_pit[from_nodes, net['_idx_node']['PINIT']]
            p_to = node_pit[to_nodes, net['_idx_node']['PAMB']] + node_pit[to_nodes, net['_idx_node']['PINIT']]
            mask = ~np.isclose(p_from, p_to)
            p_mean = np.empty_like(p_to)
            p_mean[~mask] = p_from[~mask]
            p_mean[mask] = 2 / 3 * (p_from[mask] ** 3 - p_to[mask] ** 3) \
                           / (p_from[mask] ** 2 - p_to[mask] ** 2)

            if len(net._fluid) == 1:
                fluid = net._fluid[0]
                normfactor_mean = numerator * get_fluid(net, fluid).get_compressibility(p_mean) \
                                  / (p_mean * NORMAL_TEMPERATURE)
            else:
                w = get_lookup(net, 'branch', 'w')
                mf = branch_pit[:, w]
                comp_fact = get_mixture_compressibility(net, p_mean, mf)
                normfactor_mean = numerator * comp_fact / (p_mean * NORMAL_TEMPERATURE)

            v_gas_mean = v_mps * normfactor_mean
            _, v_gas_mean_sum = _sum_by_group(idx_active, v_gas_mean)
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
        return np.array(net[cls.table_name()].sections.values)

    @classmethod
    def get_internal_results(cls, net, branch):
        """

        :param net:
        :type net:
        :param pipe:
        :type pipe:
        :return:
        :rtype:
        """
        raise NotImplementedError
