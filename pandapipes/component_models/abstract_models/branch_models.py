# Copyright (c) 2020-2022 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np

from pandapipes.component_models.abstract_models import Component
from pandapipes.component_models.auxiliaries.derivative_toolbox import calc_der_lambda, calc_lambda
from pandapipes.constants import NORMAL_PRESSURE, GRAVITATION_CONSTANT, NORMAL_TEMPERATURE, \
    P_CONVERSION
from pandapipes.idx_branch import FROM_NODE, TO_NODE, LENGTH, D, TINIT, AREA, K, RHO, ETA, \
    VINIT, RE, LAMBDA, LOAD_VEC_NODES, ALPHA, QEXT, TEXT, LOSS_COEFFICIENT as LC, branch_cols, \
    T_OUT, CP, VINIT_T, FROM_NODE_T, PL, TL, \
    JAC_DERIV_DP, JAC_DERIV_DP1, JAC_DERIV_DT, JAC_DERIV_DT1, JAC_DERIV_DT_NODE, JAC_DERIV_DV, \
    JAC_DERIV_DV_NODE, \
    LOAD_VEC_BRANCHES, LOAD_VEC_BRANCHES_T, LOAD_VEC_NODES_T, ELEMENT_IDX
from pandapipes.idx_node import PINIT, HEIGHT, TINIT as TINIT_NODE, PAMB
from pandapipes.internals_toolbox import _sum_by_group, select_from_pit
from pandapipes.pipeflow_setup import get_table_number, get_lookup
from pandapipes.properties.fluids import get_fluid

try:
    import pplog as logging
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
        raise NotImplementedError

    @classmethod
    def create_pit_branch_entries(cls, net, branch_pit, node_name):
        """
        Function which creates pit branch entries.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param branch_pit:
        :type branch_pit:
        :param node_name:
        :type node_name:
        :return: No Output.
        """
        node_pit = net["_pit"]["node"]
        f, t = get_lookup(net, "branch", "from_to")[cls.table_name()]
        branch_table_nr = get_table_number(get_lookup(net, "branch", "table"), cls.table_name())
        branch_component_pit = branch_pit[f:t, :]
        if not len(net[cls.table_name()]):
            return branch_component_pit, node_pit, [], []

        junction_idx_lookup = get_lookup(net, "node", "index")[node_name]
        from_nodes = junction_idx_lookup[net[cls.table_name()]["from_junction"].values]
        to_nodes = junction_idx_lookup[net[cls.table_name()]["to_junction"].values]
        branch_component_pit[:, :] = np.array([branch_table_nr] + [0] * (branch_cols - 1))
        branch_component_pit[:, VINIT] = 0.1
        return branch_component_pit, node_pit, from_nodes, to_nodes

    @classmethod
    def calculate_derivatives_hydraulic(cls, net, branch_pit, node_pit, idx_lookups, options):
        """
        Function which creates derivatives.

        :param net: The pandapipes network
        :type net: pandapipesNet
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
        if branch_component_pit.size == 0:
            return
        fluid = get_fluid(net)
        gas_mode = fluid.is_gas
        friction_model = options["friction_model"]
        g_const = GRAVITATION_CONSTANT

        rho = branch_component_pit[:, RHO]
        eta = branch_component_pit[:, ETA]
        d = branch_component_pit[:, D]
        k = branch_component_pit[:, K]
        length = branch_component_pit[:, LENGTH]
        from_nodes = branch_component_pit[:, FROM_NODE].astype(np.int32)
        to_nodes = branch_component_pit[:, TO_NODE].astype(np.int32)
        loss_coef = branch_component_pit[:, LC]
        t_init = (node_pit[from_nodes, TINIT_NODE] + node_pit[to_nodes, TINIT_NODE]) / 2
        branch_component_pit[:, TINIT] = t_init
        v_init = branch_component_pit[:, VINIT]

        p_init_i = node_pit[from_nodes, PINIT]
        p_init_i1 = node_pit[to_nodes, PINIT]
        p_init_i_abs = p_init_i + node_pit[from_nodes, PAMB]
        p_init_i1_abs = p_init_i1 + node_pit[to_nodes, PAMB]
        v_init2 = v_init * np.abs(v_init)

        height_difference = node_pit[from_nodes, HEIGHT] - node_pit[to_nodes, HEIGHT]
        dummy = length != 0
        lambda_pipe, re = calc_lambda(v_init, eta, rho, d, k, gas_mode, friction_model, dummy,
                                      options)
        der_lambda_pipe = calc_der_lambda(v_init, eta, rho, d, k, friction_model, lambda_pipe)
        branch_component_pit[:, RE] = re
        branch_component_pit[:, LAMBDA] = lambda_pipe
        cls.calculate_pressure_lift(net, branch_component_pit, node_pit)
        pl = branch_component_pit[:, PL]

        if not gas_mode:
            branch_component_pit[:, JAC_DERIV_DV] = \
                rho / (P_CONVERSION * 2) * (length / d * (der_lambda_pipe * v_init2 + 2 *
                                                          lambda_pipe * np.abs(v_init)) + 2 * loss_coef * np.abs(
                    v_init))

            branch_component_pit[:, LOAD_VEC_BRANCHES] = \
                - (-p_init_i_abs + p_init_i1_abs - pl
                   - rho * g_const * height_difference / P_CONVERSION
                   + (length * lambda_pipe / d + loss_coef) / (P_CONVERSION * 2) * rho * v_init2)

            branch_component_pit[:, JAC_DERIV_DP] = -1
            branch_component_pit[:, JAC_DERIV_DP1] = 1
        else:
            # compressibility settings
            p_m = np.empty_like(p_init_i_abs)
            mask = p_init_i_abs != p_init_i1_abs
            p_m[~mask] = p_init_i_abs[~mask]
            p_m[mask] = 2 / 3 * (p_init_i_abs[mask] ** 3 - p_init_i1_abs[mask] ** 3) \
                        / (p_init_i_abs[mask] ** 2 - p_init_i1_abs[mask] ** 2)
            comp_fact = get_fluid(net).get_property("compressibility", p_m)

            const_lambda = NORMAL_PRESSURE * rho * comp_fact * t_init \
                           / (NORMAL_TEMPERATURE * P_CONVERSION)
            const_height = rho * NORMAL_TEMPERATURE / (2 * NORMAL_PRESSURE * t_init * P_CONVERSION)

            branch_component_pit[:, LOAD_VEC_BRANCHES] = \
                -(-p_init_i_abs + p_init_i1_abs - pl + const_lambda * v_init2 * (
                        lambda_pipe * length / d + loss_coef)
                  * (p_init_i_abs + p_init_i1_abs) ** (-1)
                  - const_height * (p_init_i_abs + p_init_i1_abs) * g_const * height_difference)

            branch_component_pit[:, JAC_DERIV_DP] = \
                -1. - const_lambda * v_init2 * (lambda_pipe * length / d + loss_coef) \
                * (p_init_i_abs + p_init_i1_abs) ** (-2) \
                - const_height * g_const * height_difference

            branch_component_pit[:, JAC_DERIV_DP1] = \
                1. - const_lambda * v_init2 * (lambda_pipe * length / d + loss_coef) \
                * (p_init_i_abs + p_init_i1_abs) ** (-2) \
                - const_height * g_const * height_difference

            branch_component_pit[:, JAC_DERIV_DV] = \
                2 * const_lambda * (p_init_i_abs + p_init_i1_abs) ** (-1) \
                * np.abs(v_init) * lambda_pipe * length / d \
                + const_lambda * (p_init_i_abs + p_init_i1_abs) ** (-1) * v_init2 \
                * der_lambda_pipe * length / d \
                + 2 * const_lambda * (p_init_i_abs + p_init_i1_abs) ** (-1) * np.abs(v_init) \
                * loss_coef

        mass_flow_dv = rho * branch_component_pit[:, AREA]
        branch_component_pit[:, JAC_DERIV_DV_NODE] = mass_flow_dv
        branch_component_pit[:, LOAD_VEC_NODES] = mass_flow_dv * v_init
        return branch_component_pit

    @classmethod
    def calculate_derivatives_thermal(cls, net, branch_pit, node_pit, idx_lookups, options):
        """
        Function which creates derivatives of the temperature.

        :param net:
        :type net:
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
        f, t = idx_lookups[cls.table_name()]
        branch_component_pit = branch_pit[f:t, :]
        cp = branch_component_pit[:, CP]
        rho = branch_component_pit[:, RHO]
        v_init = branch_component_pit[:, VINIT_T]
        from_nodes = branch_component_pit[:, FROM_NODE_T].astype(np.int32)
        t_init_i = node_pit[from_nodes, TINIT_NODE]
        t_init_i1 = branch_component_pit[:, T_OUT]
        t_amb = branch_component_pit[:, TEXT]
        area = branch_component_pit[:, AREA]
        length = branch_component_pit[:, LENGTH]
        alpha = branch_component_pit[:, ALPHA] * np.pi * branch_component_pit[:, D]
        cls.calculate_temperature_lift(net, branch_component_pit, node_pit)
        tl = branch_component_pit[:, TL]
        qext = branch_component_pit[:, QEXT]
        t_m = (t_init_i1 + t_init_i) / 2

        branch_component_pit[:, LOAD_VEC_BRANCHES_T] = \
            -(rho * area * cp * v_init * (-t_init_i + t_init_i1 - tl)
              - alpha * (t_amb - t_m) * length + qext)

        branch_component_pit[:, JAC_DERIV_DT] = - rho * area * cp * v_init + alpha / 2 * length
        branch_component_pit[:, JAC_DERIV_DT1] = rho * area * cp * v_init + alpha / 2 * length

        branch_component_pit[:, JAC_DERIV_DT_NODE] = rho * v_init * branch_component_pit[:, AREA]
        branch_component_pit[:, LOAD_VEC_NODES_T] = rho * v_init * branch_component_pit[:,
                                                                   AREA] * t_init_i1

    @classmethod
    def calculate_pressure_lift(cls, net, branch_pit, node_pit):
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
    def prepare_result_tables(cls, net, options, node_name):
        res_table = super().extract_results(net, options, node_name)

        f, t = get_lookup(net, "branch", "from_to")[cls.table_name()]
        fa, ta = get_lookup(net, "branch", "from_to_active")[cls.table_name()]

        placement_table = np.argsort(net[cls.table_name()].index.values)
        idx_pit = net["_pit"]["branch"][f:t, ELEMENT_IDX]
        pipe_considered = get_lookup(net, "branch", "active")[f:t]
        _, active_pipes = _sum_by_group(idx_pit, pipe_considered.astype(np.int32))
        active_pipes = active_pipes > 0.99
        placement_table = placement_table[active_pipes]
        branch_pit = net["_active_pit"]["branch"][fa:ta, :]
        return placement_table, branch_pit, res_table

    @classmethod
    def extract_results(cls, net, options, node_name):
        placement_table, branch_pit, res_table = cls.prepare_result_tables(net, options, node_name)

        node_pit = net["_active_pit"]["node"]

        if not len(branch_pit):
            return placement_table, res_table, branch_pit, node_pit

        node_active_idx_lookup = get_lookup(net, "node", "index_active")[node_name]
        junction_idx_lookup = get_lookup(net, "node", "index")[node_name]
        from_junction_nodes = node_active_idx_lookup[junction_idx_lookup[
            net[cls.table_name()]["from_junction"].values[placement_table]]]
        to_junction_nodes = node_active_idx_lookup[junction_idx_lookup[
            net[cls.table_name()]["to_junction"].values[placement_table]]]

        from_nodes = branch_pit[:, FROM_NODE].astype(np.int32)
        to_nodes = branch_pit[:, TO_NODE].astype(np.int32)
        fluid = get_fluid(net)

        v_mps = branch_pit[:, VINIT]

        t0 = node_pit[from_nodes, TINIT_NODE]
        t1 = node_pit[to_nodes, TINIT_NODE]
        mf = branch_pit[:, LOAD_VEC_NODES]
        vf = branch_pit[:, LOAD_VEC_NODES] / get_fluid(net).get_density((t0 + t1) / 2)

        idx_active = branch_pit[:, ELEMENT_IDX]
        _, v_sum, mf_sum, vf_sum, internal_pipes = _sum_by_group(idx_active, v_mps, mf, vf, np.ones_like(idx_active))

        if fluid.is_gas:
            # derived from the ideal gas law
            p_from = node_pit[from_nodes, PAMB] + node_pit[from_nodes, PINIT]
            p_to = node_pit[to_nodes, PAMB] + node_pit[to_nodes, PINIT]
            numerator = NORMAL_PRESSURE * branch_pit[:, TINIT]
            normfactor_from = numerator * fluid.get_property("compressibility", p_from) \
                              / (p_from * NORMAL_TEMPERATURE)
            normfactor_to = numerator * fluid.get_property("compressibility", p_to) \
                            / (p_to * NORMAL_TEMPERATURE)
            v_gas_from = v_mps * normfactor_from
            v_gas_to = v_mps * normfactor_to

            _, nf_from_sum, nf_to_sum = _sum_by_group(idx_active, normfactor_from, normfactor_to)

            v_gas_from_ordered = select_from_pit(from_nodes, from_junction_nodes, v_gas_from)
            v_gas_to_ordered = select_from_pit(to_nodes, to_junction_nodes, v_gas_to)

            res_table["v_from_m_per_s"].values[placement_table] = v_gas_from_ordered
            res_table["v_to_m_per_s"].values[placement_table] = v_gas_to_ordered
            res_table["normfactor_from"].values[placement_table] = nf_from_sum / internal_pipes
            res_table["normfactor_to"].values[placement_table] = nf_to_sum / internal_pipes

        res_table["p_from_bar"].values[placement_table] = node_pit[from_junction_nodes, PINIT]
        res_table["p_to_bar"].values[placement_table] = node_pit[to_junction_nodes, PINIT]
        res_table["t_from_k"].values[placement_table] = node_pit[from_junction_nodes, TINIT_NODE]
        res_table["t_to_k"].values[placement_table] = node_pit[to_junction_nodes, TINIT_NODE]
        res_table["mdot_to_kg_per_s"].values[placement_table] = -mf_sum / internal_pipes
        res_table["mdot_from_kg_per_s"].values[placement_table] = mf_sum / internal_pipes
        res_table["vdot_norm_m3_per_s"].values[placement_table] = vf_sum / internal_pipes
        return placement_table, res_table, branch_pit, node_pit
