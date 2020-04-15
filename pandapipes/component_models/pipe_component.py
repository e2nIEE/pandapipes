# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import matplotlib.pyplot as plt
import numpy as np

from pandapipes.component_models.abstract_models import BranchWInternalsComponent
from pandapipes.component_models.auxiliaries.component_toolbox import p_correction_height_air, \
    vinterp
from pandapipes.component_models.junction_component import Junction

from pandapipes.idx_node import ELEMENT_IDX, PINIT, HEIGHT, TINIT as TINIT_NODE, \
    RHO as RHO_NODES, PAMB, ACTIVE as ACTIVE_ND
from pandapipes.idx_branch import FROM_NODE, TO_NODE, LENGTH, D, TINIT, AREA, K, \
    VINIT, RE, LAMBDA, LOAD_VEC_NODES, ALPHA, QEXT, TEXT, LOSS_COEFFICIENT as LC, T_OUT, PL, TL
from pandapipes.constants import NORMAL_TEMPERATURE, NORMAL_PRESSURE

from pandapipes.pipeflow_setup import get_net_option, get_fluid, get_lookup
from pandapipes.toolbox import _sum_by_group
from numpy import dtype

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Pipe(BranchWInternalsComponent):
    """

    """

    @classmethod
    def table_name(cls):
        return "pipe"

    @classmethod
    def internal_node_name(cls):
        return "pipe_nodes"

    @classmethod
    def active_identifier(cls):
        return "in_service"

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
        end, current_table, internal_nodes, internal_pipes, int_nodes_num, int_pipes_num = \
            super().create_node_lookups(net, ft_lookups, table_lookup, idx_lookups,
                                        current_start, current_table, internal_nodes_lookup)
        if np.any(internal_nodes > 0):
            internal_nodes_lookup["TPINIT"] = np.empty((int_nodes_num, 2), dtype=np.int32)
            internal_nodes_lookup["TPINIT"][:, 0] = np.repeat(net[cls.table_name()].index,
                                                              internal_nodes)
            internal_nodes_lookup["TPINIT"][:, 1] = np.arange(current_start, end)

            internal_nodes_lookup["VINIT"] = np.empty((int_pipes_num, 2), dtype=np.int32)
            internal_nodes_lookup["VINIT"][:, 0] = np.repeat(net[cls.table_name()].index,
                                                             internal_pipes)
            internal_nodes_lookup["VINIT"][:, 1] = np.arange(int_pipes_num)

        return end, current_table

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
        table_nr, int_node_number, int_node_pit, junction_pit, from_junctions, to_junctions = \
            super().create_pit_node_entries(net, node_pit, node_name)
        if table_nr is None:
            return
        int_node_pit[:, HEIGHT] = vinterp(junction_pit[from_junctions, HEIGHT],
                                          junction_pit[to_junctions, HEIGHT], int_node_number)
        int_node_pit[:, PINIT] = vinterp(junction_pit[from_junctions, PINIT],
                                         junction_pit[to_junctions, PINIT], int_node_number)
        int_node_pit[:, TINIT_NODE] = vinterp(junction_pit[from_junctions, TINIT_NODE],
                                              junction_pit[to_junctions, TINIT_NODE],
                                              int_node_number)
        int_node_pit[:, PAMB] = p_correction_height_air(int_node_pit[:, HEIGHT])
        int_node_pit[:, RHO_NODES] = get_fluid(net).get_density(int_node_pit[:, TINIT_NODE])
        int_node_pit[:, ACTIVE_ND] = \
            np.repeat(net[cls.table_name()][cls.active_identifier()].values, int_node_number)

    @classmethod
    def create_pit_branch_entries(cls, net, pipe_pit, node_name):
        """
        Function which creates pit branch entries.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param branch_pit:
        :type branch_pit:
        :return: No Output.
        """
        pipe_pit, internal_pipe_number = \
            super().create_pit_branch_entries(net, pipe_pit, node_name)

        pipe_pit[:, LENGTH] = np.repeat(net[cls.table_name()].length_km.values * 1000 /
                                        internal_pipe_number, internal_pipe_number)
        pipe_pit[:, K] = np.repeat(net[cls.table_name()].k_mm.values / 1000,
                                   internal_pipe_number)
        pipe_pit[:, T_OUT] = 293
        pipe_pit[:, ALPHA] = np.repeat(net[cls.table_name()].alpha_w_per_m2k.values,
                                       internal_pipe_number)
        pipe_pit[:, QEXT] = np.repeat(net[cls.table_name()].qext_w.values,
                                      internal_pipe_number)
        pipe_pit[:, TEXT] = np.repeat(net[cls.table_name()].text_k.values,
                                      internal_pipe_number)
        pipe_pit[:, D] = np.repeat(net[cls.table_name()].diameter_m.values, internal_pipe_number)
        pipe_pit[:, AREA] = pipe_pit[:, D] ** 2 * np.pi / 4
        pipe_pit[:, LC] = np.repeat(net[cls.table_name()].loss_coefficient.values,
                                    internal_pipe_number)

    @classmethod
    def calculate_pressure_lift(cls, net, pipe_pit, node_pit):
        """

        :param net:
        :type net:
        :param pipe_pit:
        :type pipe_pit:
        :param node_pit:
        :type node_pit:
        :return:
        :rtype:
        """
        pipe_pit[:, PL] = 0

    @classmethod
    def calculate_temperature_lift(cls, net, pipe_pit, node_pit):
        """

        :param net:
        :type net:
        :param pipe_pit:
        :type pipe_pit:
        :param node_pit:
        :type node_pit:
        :return:
        :rtype:
        """
        pipe_pit[:, TL] = 0

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

        placement_table, pipe_pit, res_table = super().extract_results(net, options, node_name)

        node_pit = net["_active_pit"]["node"]
        node_active_idx_lookup = get_lookup(net, "node", "index_active")[node_name]
        junction_idx_lookup = get_lookup(net, "node", "index")[node_name]
        from_junction_nodes = node_active_idx_lookup[junction_idx_lookup[
            net[cls.table_name()]["from_junction"].values[placement_table]]]
        to_junction_nodes = node_active_idx_lookup[junction_idx_lookup[
            net[cls.table_name()]["to_junction"].values[placement_table]]]

        from_nodes = pipe_pit[:, FROM_NODE].astype(np.int32)
        to_nodes = pipe_pit[:, TO_NODE].astype(np.int32)
        p_scale = get_net_option(net, "p_scale")
        fluid = get_fluid(net)

        v_mps = pipe_pit[:, VINIT]

        t0 = node_pit[from_nodes, TINIT_NODE]
        t1 = node_pit[to_nodes, TINIT_NODE]
        mf = pipe_pit[:, LOAD_VEC_NODES]
        vf = pipe_pit[:, LOAD_VEC_NODES] / get_fluid(net).get_density((t0 + t1) / 2)

        idx_active = pipe_pit[:, ELEMENT_IDX]
        idx_sort, v_sum, mf_sum, vf_sum, internal_pipes = \
            _sum_by_group(idx_active, v_mps, mf, vf, np.ones_like(idx_active))

        if fluid.is_gas:
            # derived from the ideal gas law
            p_from = node_pit[from_nodes, PAMB] + node_pit[from_nodes, PINIT] * p_scale
            p_to = node_pit[to_nodes, PAMB] + node_pit[to_nodes, PINIT] * p_scale
            numerator = NORMAL_PRESSURE * pipe_pit[:, TINIT]
            normfactor_from = numerator * fluid.get_property("compressibility", p_from) \
                              / (p_from * NORMAL_TEMPERATURE)
            normfactor_to = numerator * fluid.get_property("compressibility", p_to) \
                            / (p_to * NORMAL_TEMPERATURE)
            v_gas_from = v_mps * normfactor_from
            v_gas_to = v_mps * normfactor_to
            mask = p_from != p_to
            p_mean = np.empty_like(p_to)
            p_mean[~mask] = p_from[~mask]
            p_mean[mask] = 2 / 3 * (p_from[mask] ** 3 - p_to[mask] ** 3) \
                           / (p_from[mask] ** 2 - p_to[mask] ** 2)
            normfactor_mean = numerator * fluid.get_property("compressibility", p_mean) \
                              / (p_mean * NORMAL_TEMPERATURE)
            v_gas_mean = v_mps * normfactor_mean

            idx_sort, v_gas_from_sum, v_gas_to_sum, v_gas_mean_sum, nf_from_sum, nf_to_sum, \
            internal_pipes = _sum_by_group(
                idx_active, v_gas_from, v_gas_to, v_gas_mean, normfactor_from, normfactor_to,
                np.ones_like(idx_active))

            res_table["v_from_m_per_s"].values[placement_table] = v_gas_from_sum / internal_pipes
            res_table["v_to_m_per_s"].values[placement_table] = v_gas_to_sum / internal_pipes
            res_table["v_mean_m_per_s"].values[placement_table] = v_gas_mean_sum / internal_pipes
            res_table["normfactor_from"].values[placement_table] = nf_from_sum / internal_pipes
            res_table["normfactor_to"].values[placement_table] = nf_to_sum / internal_pipes
        else:
            res_table["v_mean_m_per_s"].values[placement_table] = v_sum / internal_pipes

        res_table["p_from_bar"].values[placement_table] = node_pit[from_junction_nodes, PINIT]
        res_table["p_to_bar"].values[placement_table] = node_pit[to_junction_nodes, PINIT]
        res_table["t_from_k"].values[placement_table] = node_pit[from_junction_nodes, TINIT_NODE]
        res_table["t_to_k"].values[placement_table] = node_pit[to_junction_nodes, TINIT_NODE]
        res_table["mdot_to_kg_per_s"].values[placement_table] = -mf_sum / internal_pipes
        res_table["mdot_from_kg_per_s"].values[placement_table] = mf_sum / internal_pipes
        res_table["vdot_norm_m3_per_s"].values[placement_table] = vf_sum / internal_pipes
        idx_pit = pipe_pit[:, ELEMENT_IDX]
        idx_sort, lambda_sum, reynolds_sum, = \
            _sum_by_group(idx_pit, pipe_pit[:, LAMBDA], pipe_pit[:, RE])
        res_table["lambda"].values[placement_table] = lambda_sum / internal_pipes
        res_table["reynolds"].values[placement_table] = reynolds_sum / internal_pipes

    @classmethod
    def get_internal_results(cls, net, pipe):
        """

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param pipe:
        :type pipe:
        :return: pipe_results
        :rtype:
        """
        internal_sections = cls.get_internal_pipe_number(net)
        internal_p_nodes = internal_sections - 1
        p_node_idx = np.repeat(pipe, internal_p_nodes[pipe])
        v_pipe_idx = np.repeat(pipe, internal_sections[pipe])
        pipe_results = dict()
        pipe_results["PINIT"] = np.zeros((len(p_node_idx), 2), dtype=np.float64)
        pipe_results["TINIT"] = np.zeros((len(p_node_idx), 2), dtype=np.float64)
        pipe_results["VINIT"] = np.zeros((len(v_pipe_idx), 2), dtype=np.float64)

        if np.all(internal_sections[pipe] >= 2):
            fluid = get_fluid(net)
            f, t = get_lookup(net, "branch", "from_to")[cls.table_name()]
            pipe_pit = net["_pit"]["branch"][f:t, :]
            node_pit = net["_pit"]["node"]
            int_p_lookup = net["_lookups"]["internal_nodes_lookup"]["TPINIT"]
            int_v_lookup = net["_lookups"]["internal_nodes_lookup"]["VINIT"]

            selected_indices_p = []
            selected_indices_v = []
            for i in pipe:
                selected_indices_p.append(np.where(int_p_lookup[:, 0] == i, True, False))
                selected_indices_v.append(np.where(int_v_lookup[:, 0] == i, True, False))

            selected_indices_p_final = np.logical_or.reduce(selected_indices_p[:])
            selected_indices_v_final = np.logical_or.reduce(selected_indices_v[:])
            # a = np.where(int_p_lookup[:,0] ==  True, False)
            # b = np.where(int_v_lookup[:, 0] == v_pipe_idx, True, False)
            p_nodes = int_p_lookup[:, 1][selected_indices_p_final]
            v_nodes = int_v_lookup[:, 1][selected_indices_v_final]

            v_pipe_data = pipe_pit[v_nodes, VINIT]
            p_node_data = node_pit[p_nodes, PINIT]
            t_node_data = node_pit[p_nodes, TINIT_NODE]

            gas_mode = fluid.is_gas

            if gas_mode:
                p_scale = get_net_option(net, "p_scale")

                from_nodes = pipe_pit[v_nodes, FROM_NODE].astype(np.int32)
                to_nodes = pipe_pit[v_nodes, TO_NODE].astype(np.int32)
                p_from = node_pit[from_nodes, PAMB] + node_pit[from_nodes, PINIT] * p_scale
                p_to = node_pit[to_nodes, PAMB] + node_pit[to_nodes, PINIT] * p_scale
                p_mean = np.where(p_from == p_to, p_from,
                                  2 / 3 * (p_from ** 3 - p_to ** 3) / (p_from ** 2 - p_to ** 2))
                numerator = NORMAL_PRESSURE * node_pit[v_nodes, TINIT_NODE]
                normfactor = numerator * fluid.get_property("compressibility", p_mean) \
                             / (p_mean * NORMAL_TEMPERATURE)

                v_pipe_data = v_pipe_data * normfactor

            pipe_results["PINIT"][:, 0] = p_node_idx
            pipe_results["PINIT"][:, 1] = p_node_data
            pipe_results["TINIT"][:, 0] = p_node_idx
            pipe_results["TINIT"][:, 1] = t_node_data
            pipe_results["VINIT"][:, 0] = v_pipe_idx
            pipe_results["VINIT"][:, 1] = v_pipe_data
        else:
            logger.warning("For at least one pipe no internal data is available.")

        return pipe_results

    @classmethod
    def get_component_input(cls):
        """

        :return:
        :rtype:
        """
        return [("name", dtype(object)),
                ("from_junction", "u4"),
                ("to_junction", "u4"),
                ("std_type", dtype(object)),
                ("length_km", "f8"),
                ("diameter_m", "f8"),
                ("k_mm", "f8"),
                ("loss_coefficient", "f8"),
                ("alpha_w_per_m2k", 'f8'),
                ("text_k", 'f8'),
                ("qext_w", 'f8'),
                ("sections", "u4"),
                ("in_service", 'bool'),
                ("type", dtype(object))]

    @classmethod
    def geodata(cls):
        """

        :return:
        :rtype:
        """
        return [("coords", dtype(object))]

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
                      "t_from_k", "t_to_k", "mdot_from_kg_per_s", "mdot_to_kg_per_s",
                      "vdot_norm_m3_per_s", "reynolds", "lambda", "normfactor_from",
                      "normfactor_to"]
        else:
            output = ["v_mean_m_per_s", "p_from_bar", "p_to_bar", "t_from_k", "t_to_k",
                      "mdot_from_kg_per_s", "mdot_to_kg_per_s", "vdot_norm_m3_per_s", "reynolds",
                      "lambda"]
        return output, True

    @classmethod
    def plot_pipe(cls, net, pipe, pipe_results):
        """

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param pipe:
        :type pipe:
        :param pipe_results:
        :type pipe_results:
        :return: No Output.
        """
        pipe_p_data_idx = np.where(pipe_results["PINIT"][:, 0] == pipe)
        pipe_v_data_idx = np.where(pipe_results["VINIT"][:, 0] == pipe)
        pipe_p_data = pipe_results["PINIT"][pipe_p_data_idx, 1]
        pipe_t_data = pipe_results["TINIT"][pipe_p_data_idx, 1]
        pipe_v_data = pipe_results["VINIT"][pipe_v_data_idx, 1]
        node_pit = net["_pit"]["node"]

        junction_idx_lookup = get_lookup(net, "node", "index")[Junction.table_name()]
        from_junction_nodes = junction_idx_lookup[net[cls.table_name]["from_junction"].values]
        to_junction_nodes = junction_idx_lookup[net[cls.table_name]["to_junction"].values]

        p_values = np.zeros(len(pipe_p_data[0]) + 2)
        p_values[0] = node_pit[from_junction_nodes[pipe], PINIT]
        p_values[1:-1] = pipe_p_data[:]
        p_values[-1] = node_pit[to_junction_nodes[pipe], PINIT]

        t_values = np.zeros(len(pipe_t_data[0]) + 2)
        t_values[0] = node_pit[from_junction_nodes[pipe], TINIT_NODE]
        t_values[1:-1] = pipe_t_data[:]
        t_values[-1] = node_pit[to_junction_nodes[pipe], TINIT_NODE]

        v_values = pipe_v_data[0, :]

        x_pt = np.linspace(0, net.pipe["length_km"], len(p_values))
        x_v = np.linspace(0, net.pipe["length_km"], len(v_values))
        f, axes = plt.subplots(3, 1, sharex="all")
        axes[0].plot(x_pt, p_values)
        axes[0].set_title("Pressure [bar]")
        axes[1].plot(x_v, v_values)
        axes[1].set_title("Velocity [m/s]")
        axes[2].plot(x_pt, t_values)
        axes[2].set_title("Temperature [K]")

        plt.show()
