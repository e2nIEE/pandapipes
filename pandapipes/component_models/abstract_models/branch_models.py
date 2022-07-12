# Copyright (c) 2020-2022 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np

from pandapipes.component_models.abstract_models.base_component import Component
from pandapipes.pf.pipeflow_setup import get_table_number, get_lookup
from pandapipes.properties.fluids import get_mixture_density, \
    get_mixture_viscosity, get_mixture_heat_capacity, \
    get_derivative_density_diff, get_derivative_density_same

try:
    from pandaplan.core import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class BranchComponent(Component):

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
    def from_to_node_cols(cls):
        raise NotImplementedError

    @classmethod
    def active_identifier(cls):
        raise NotImplementedError()

    @classmethod
    def get_connected_node_type(cls):
        raise NotImplementedError

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
        :return: No Output.
        """
        raise NotImplementedError

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
        node_pit = net["_pit"]["node"]
        f, t = get_lookup(net, "branch", "from_to")[cls.table_name()]
        branch_table_nr = get_table_number(get_lookup(net, "branch", "table"), cls.table_name())
        branch_component_pit = branch_pit[f:t, :]
        if not len(net[cls.table_name()]):
            return branch_component_pit, node_pit, [], []

        junction_idx_lookup = get_lookup(net, "node", "index")[
            cls.get_connected_node_type().table_name()]
        from_nodes = junction_idx_lookup[net[cls.table_name()]["from_junction"].values]
        to_nodes = junction_idx_lookup[net[cls.table_name()]["to_junction"].values]
        branch_component_pit[:, :] = np.array([branch_table_nr] + [0] * (net['_idx_branch']['branch_cols'] - 1))
        branch_component_pit[:, net['_idx_branch']['VINIT']] = 0.1
        return branch_component_pit, node_pit, from_nodes, to_nodes

    @classmethod
    def adaption_before_derivatives_hydraulic(cls, net, branch_pit, node_pit, branch_lookups, node_lookups, options):
        if len(net._fluid) != 1:
            f, t = branch_lookups[cls.table_name()]
            branch_component_pit = branch_pit[f:t, :]
            create_v_node(net, branch_pit)
            v_from_b = branch_component_pit[:, net['_idx_branch']['V_FROM_NODE']].astype(int)

            w = get_lookup(net, 'node', 'w')
            w_branch = get_lookup(net, 'branch', 'w')
            mf = node_pit[:, w][v_from_b]
            branch_component_pit[:, w_branch] = mf

            branch_component_pit[:, net['_idx_branch']['RHO']] = \
                get_mixture_density(net, branch_component_pit[:, net['_idx_branch']['TINIT']], mf)
            branch_component_pit[:, net['_idx_branch']['ETA']] = \
                get_mixture_viscosity(net, branch_component_pit[:, net['_idx_branch']['TINIT']], mf)
            branch_component_pit[:, net['_idx_branch']['CP']] = \
                get_mixture_heat_capacity(net, branch_component_pit[:, net['_idx_branch']['TINIT']], mf)

            der_rho_same = get_lookup(net, 'branch', 'deriv_rho_same')
            der_rho_diff = get_lookup(net, 'branch', 'deriv_rho_diff')
            rho = get_lookup(net, 'branch', 'rho')
            rl = branch_component_pit[:, rho]
            branch_component_pit[:, der_rho_same] = get_derivative_density_same(mf, rl)
            branch_component_pit[:, der_rho_diff] = get_derivative_density_diff(mf, rl)

    @classmethod
    def calculate_derivatives_thermal(cls, net, branch_pit, node_pit, idx_lookups, options):
        """
        Function which creates derivatives of the temperature.

        :param net:
        :type net:
        :param branch_pit:
        :type branch_pit:
        :param node_pit:
        :type node_pit:
        :param idx_lookups:
        :type idx_lookups:
        :param options:
        :type options:
        :return: No Output.
        """
        f, t = idx_lookups[cls.table_name()]
        branch_component_pit = branch_pit[f:t, :]
        cp = branch_component_pit[:, net['_idx_branch']['CP']]
        rho = branch_component_pit[:, net['_idx_branch']['RHO']]
        v_init = branch_component_pit[:, net['_idx_branch']['VINIT_T']]
        from_nodes = branch_component_pit[:, net['_idx_branch']['FROM_NODE_T']].astype(np.int32)
        t_init_i = node_pit[from_nodes, net['_idx_node']['TINIT']]
        t_init_i1 = branch_component_pit[:, net['_idx_branch']['T_OUT']]
        t_amb = branch_component_pit[:, net['_idx_branch']['TEXT']]
        area = branch_component_pit[:, net['_idx_branch']['AREA']]
        length = branch_component_pit[:, net['_idx_branch']['LENGTH']]
        alpha = branch_component_pit[:, net['_idx_branch']['ALPHA']] * np.pi * branch_component_pit[:,
                                                                               net['_idx_branch']['D']]
        cls.calculate_temperature_lift(net, branch_component_pit, node_pit)
        tl = branch_component_pit[:, net['_idx_branch']['TL']]
        qext = branch_component_pit[:, net['_idx_branch']['QEXT']]
        t_m = (t_init_i1 + t_init_i) / 2

        branch_component_pit[:, net['_idx_branch']['LOAD_VEC_BRANCHES_T']] = \
            -(rho * area * cp * v_init * (-t_init_i + t_init_i1 - tl)
              - alpha * (t_amb - t_m) * length + qext)

        branch_component_pit[:, net['_idx_branch']['JAC_DERIV_DT']] = - rho * area * cp * v_init + alpha / 2 * length
        branch_component_pit[:, net['_idx_branch']['JAC_DERIV_DT1']] = rho * area * cp * v_init + alpha / 2 * length

        branch_component_pit[:, net['_idx_branch']['JAC_DERIV_DT_NODE']] = \
            rho * v_init * branch_component_pit[:, net['_idx_branch']['AREA']]
        branch_component_pit[:, net['_idx_branch']['LOAD_VEC_NODES_T']] = \
            rho * v_init * branch_component_pit[:, net['_idx_branch']['AREA']] * t_init_i1

    @classmethod
    def calculate_temperature_lift(cls, net, branch_pit, node_pit):
        """

        :param net:
        :type net:
        :param branch_pit:
        :type branch_pit:
        :param node_pit:
        :type node_pit:
        :return:
        :rtype:
        """
        raise NotImplementedError

    @classmethod
    def extract_results(cls, net, options, branch_results, nodes_connected, branches_connected):
        raise NotImplementedError


def create_v_node(net, branch_pit):
    v = branch_pit[:, net['_idx_branch']['VINIT']]
    fn_w = branch_pit[v >= 0, net['_idx_branch']['FROM_NODE']]
    tn_w = branch_pit[v < 0, net['_idx_branch']['TO_NODE']]
    branch_pit[v >= 0, net['_idx_branch']['V_FROM_NODE']] = fn_w
    branch_pit[v < 0, net['_idx_branch']['V_FROM_NODE']] = tn_w

    tn_w = branch_pit[v >= 0, net['_idx_branch']['TO_NODE']]
    fn_w = branch_pit[v < 0, net['_idx_branch']['FROM_NODE']]
    branch_pit[v >= 0, net['_idx_branch']['V_TO_NODE']] = tn_w
    branch_pit[v < 0, net['_idx_branch']['V_TO_NODE']] = fn_w
