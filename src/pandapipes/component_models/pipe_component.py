# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import matplotlib.pyplot as plt
import numpy as np
from numpy import dtype

from pandapipes.component_models.abstract_models import BranchComponent
from pandapipes.component_models.component_toolbox import set_entry_check_repeat, vinterp, p_correction_height_air
from pandapipes.constants import NORMAL_TEMPERATURE, NORMAL_PRESSURE
from pandapipes.idx_branch import FROM_NODE, TO_NODE, LENGTH, D, AREA, K, \
    MDOTINIT, ALPHA, QEXT, TEXT, LOSS_COEFFICIENT as LC, ACTIVE, ELEMENT_IDX, TOUTINIT
from pandapipes.idx_node import (PINIT, TINIT as TINIT_NODE, PAMB, L, node_cols, HEIGHT, ACTIVE as ACTIVE_ND)
from pandapipes.pf.pipeflow_setup import get_fluid, get_lookup, get_net_option, add_table_lookup, get_table_number
from pandapipes.pf.result_extraction import extract_branch_results_with_internals, \
    extract_branch_results_without_internals

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Pipe(BranchComponent):
    """

    """
    @property
    def table_name(self):
        return "pipe"

    @property
    def internal_node_name(self):
        return "pipe_nodes"

    @property
    def geodata(self):
        """

        :return:
        :rtype:
        """
        return [("coords", dtype(object))]

    def create_node_lookups(self, net, ft_lookups, table_lookup, idx_lookups, current_start,
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
        internal_nodes = self.get_internal_pipe_number(net) - 1
        end = current_start
        ft_lookups[self.table_name] = None
        if np.any(internal_nodes > 0):
            int_nodes_num = int(np.sum(internal_nodes))
            internal_pipes = internal_nodes + 1
            int_pipes_num = int(np.sum(internal_pipes))
            end = current_start + int_nodes_num
            add_table_lookup(table_lookup, self.internal_node_name, current_table)
            ft_lookups[self.internal_node_name] = (current_start, end)
        else:
            internal_nodes = 0
            internal_pipes = 0
            int_nodes_num = 0
            int_pipes_num = 0

        current_table = current_table + 1

        if np.any(internal_nodes > 0):
            internal_nodes_lookup["TPINIT"] = np.empty((int_nodes_num, 2), dtype=np.int32)
            internal_nodes_lookup["TPINIT"][:, 0] = np.repeat(net[self.table_name].index,
                                                              internal_nodes.astype(np.int32))
            internal_nodes_lookup["TPINIT"][:, 1] = np.arange(current_start, end)

            internal_nodes_lookup["VINIT"] = np.empty((int_pipes_num, 2), dtype=np.int32)
            internal_nodes_lookup["VINIT"][:, 0] = np.repeat(net[self.table_name].index,
                                                             internal_pipes.astype(np.int32))
            internal_nodes_lookup["VINIT"][:, 1] = np.arange(int_pipes_num)

        return end, current_table

    def create_branch_lookups(self, net, ft_lookups, table_lookup, idx_lookups, current_table,
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
        end = current_start + int(np.sum(self.get_internal_pipe_number(net)))
        ft_lookups[self.table_name] = (current_start, end)
        add_table_lookup(table_lookup, self.table_name, current_table)
        return end, current_table + 1

    def create_pit_node_entries(self, net, node_pit):
        """
        Function which creates pit node entries.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param node_pit:
        :type node_pit:
        :return: No Output.
        """
        table_lookup = get_lookup(net, "node", "table")
        table_nr = get_table_number(table_lookup, self.internal_node_name)
        if table_nr is None:
            return None, 0, 0, None, None, None
        ft_lookup = get_lookup(net, "node", "from_to")
        f, t = ft_lookup[self.internal_node_name]

        int_node_pit = node_pit[f:t, :]
        int_node_pit[:, :] = np.array([table_nr, 0, L] + [0] * (node_cols - 3))
        int_node_number = self.get_internal_pipe_number(net) - 1

        int_node_pit[:, ELEMENT_IDX] = np.arange(t - f)

        junction_table_name = self.connected_node_type.table_name  # todo:
        fj_name, tj_name = "from_" + junction_table_name, "to_" + junction_table_name
        f_junction, t_junction = ft_lookup[junction_table_name]
        junction_pit = node_pit[f_junction:t_junction, :]
        from_junctions = net[self.table_name][fj_name].values.astype(np.int32)
        to_junctions = net[self.table_name][tj_name].values.astype(np.int32)
        junction_indices = get_lookup(net, "node", "index")[junction_table_name]
        fj_nodes = junction_indices[from_junctions]
        tj_nodes = junction_indices[to_junctions]

        int_node_pit[:, HEIGHT] = vinterp(junction_pit[fj_nodes, HEIGHT],
                                          junction_pit[tj_nodes, HEIGHT], int_node_number)
        int_node_pit[:, PINIT] = vinterp(junction_pit[fj_nodes, PINIT],
                                         junction_pit[tj_nodes, PINIT], int_node_number)
        int_node_pit[:, TINIT_NODE] = vinterp(junction_pit[fj_nodes, TINIT_NODE],
                                              junction_pit[tj_nodes, TINIT_NODE],
                                              int_node_number)
        int_node_pit[:, PAMB] = p_correction_height_air(int_node_pit[:, HEIGHT])
        int_node_pit[:, ACTIVE_ND] = \
            np.repeat(net[self.table_name][self.active_identifier].values, int_node_number)
        return table_nr, int_node_number, int_node_pit, junction_pit, fj_nodes, tj_nodes

    def create_pit_branch_entries(self, net, branch_pit):
        """
        Function which creates pit branch entries.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param branch_pit:
        :type branch_pit:
        :return: No Output.
        """
        pipe_pit, node_pit, from_nodes, to_nodes \
            = super().create_pit_branch_entries(net, branch_pit)

        if len(pipe_pit):
            internal_pipe_number = self.get_internal_pipe_number(net).astype(np.int32)
            node_ft_lookups = get_lookup(net, "node", "from_to")

            has_internals = self.internal_node_name in node_ft_lookups
            if has_internals:
                pipe_nodes_from, pipe_nodes_to = node_ft_lookups[self.internal_node_name]
                pipe_nodes_idx = np.arange(pipe_nodes_from, pipe_nodes_to)
                insert_places = np.repeat(np.arange(len(from_nodes)), internal_pipe_number - 1)
                from_nodes = np.insert(from_nodes, insert_places + 1, pipe_nodes_idx)
                to_nodes = np.insert(to_nodes, insert_places, pipe_nodes_idx)

            set_entry_check_repeat(
                pipe_pit, ELEMENT_IDX, net[self.table_name].index.values,
                internal_pipe_number, has_internals)
            set_entry_check_repeat(
                pipe_pit, ACTIVE, net[self.table_name][self.active_identifier].values,
                internal_pipe_number, has_internals)
            pipe_pit[:, FROM_NODE] = from_nodes
            pipe_pit[:, TO_NODE] = to_nodes
            pipe_pit[:, TOUTINIT] = node_pit[to_nodes, TINIT_NODE]
        else:
            internal_pipe_number = np.array([], dtype=np.int32)

        ############

        has_internals = np.any(internal_pipe_number > 1)
        tbl = self.table_name
        set_entry_check_repeat(
            pipe_pit, LENGTH, net[tbl].length_km.values * 1000 / internal_pipe_number,
            internal_pipe_number, has_internals)
        set_entry_check_repeat(
            pipe_pit, K, net[tbl].k_mm.values / 1000, internal_pipe_number, has_internals)
        set_entry_check_repeat(
            pipe_pit, ALPHA, net[tbl].u_w_per_m2k.values, internal_pipe_number, has_internals)
        set_entry_check_repeat(
            pipe_pit, QEXT, net[tbl].qext_w.values, internal_pipe_number, has_internals)
        set_entry_check_repeat(
            pipe_pit, TEXT, net[tbl].text_k.values, internal_pipe_number, has_internals)
        set_entry_check_repeat(
            pipe_pit, D, net[tbl].diameter_m.values, internal_pipe_number, has_internals)
        set_entry_check_repeat(
            pipe_pit, LC, net[tbl].loss_coefficient.values, internal_pipe_number, has_internals)

        nan_mask = np.isnan(pipe_pit[:, TEXT])
        pipe_pit[nan_mask, TEXT] = get_net_option(net, 'ambient_temperature')
        pipe_pit[:, AREA] = pipe_pit[:, D] ** 2 * np.pi / 4
        pipe_pit[:, MDOTINIT] *= pipe_pit[:, AREA] * get_fluid(net).get_density(NORMAL_TEMPERATURE)

    def extract_results(self, net, options, branch_results, mode):
        res_nodes_from_hyd = [("p_from_bar", "p_from"), ("mdot_from_kg_per_s", "mf_from")]
        res_nodes_from_ht = [("t_from_k", "temp_from")]
        res_nodes_to_hyd = [("p_to_bar", "p_to"), ("mdot_to_kg_per_s", "mf_to")]
        res_nodes_to_ht = [("t_to_k", "temp_to")]
        res_mean_hyd = [("lambda", "lambda"), ("reynolds", "reynolds")]
        res_branch_ht = [("t_outlet_k", "t_outlet")]

        if get_fluid(net).is_gas:
            res_nodes_from_hyd.extend([("v_from_m_per_s", "v_gas_from"),
                                       ("normfactor_from", "normfactor_from")])
            res_nodes_to_hyd.extend([("v_to_m_per_s", "v_gas_to"),
                                     ("normfactor_to", "normfactor_to")])
            res_mean_hyd.extend([("v_mean_m_per_s", "v_gas_mean"), ("vdot_norm_m3_per_s", "vf")])
        else:
            res_mean_hyd.extend([("v_mean_m_per_s", "v_mps"), ("vdot_m3_per_s", "vf")])

        if np.any(self.get_internal_pipe_number(net) > 1):
            extract_branch_results_with_internals(
                net, branch_results, self.table_name, res_nodes_from_hyd, res_nodes_from_ht,
                res_nodes_to_hyd, res_nodes_to_ht, res_mean_hyd, res_branch_ht, [],
                self.connected_node_type.table_name, mode) #todo
        else:
            required_results_hyd = res_nodes_from_hyd + res_nodes_to_hyd + res_mean_hyd
            required_results_ht = res_nodes_from_ht + res_nodes_to_ht + res_branch_ht
            extract_branch_results_without_internals(
                net, branch_results, required_results_hyd, required_results_ht, self.table_name,
                mode
            )

    def get_internal_results(self, net, pipe):
        """
        Retrieve velocity (at to/from node; mean), pressure and temperature of the internal sections
        of pipes. The pipes have to have at least 2 internal sections.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param pipe: indices of pipes to evaluate
        :type pipe: np.array
        :return: pipe_results
        :rtype:
        """
        internal_sections = self.get_internal_pipe_number(net)
        internal_p_nodes = internal_sections - 1
        p_node_idx = np.repeat(pipe, internal_p_nodes[pipe])
        v_pipe_idx = np.repeat(pipe, internal_sections[pipe])
        pipe_results = dict()
        pipe_results["PINIT"] = np.zeros((len(p_node_idx), 2), dtype=np.float64)
        pipe_results["TINIT"] = np.zeros((len(p_node_idx), 2), dtype=np.float64)
        pipe_results["VINIT_FROM"] = np.zeros((len(v_pipe_idx), 2), dtype=np.float64)
        pipe_results["VINIT_TO"] = np.zeros((len(v_pipe_idx), 2), dtype=np.float64)
        pipe_results["VINIT_MEAN"] = np.zeros((len(v_pipe_idx), 2), dtype=np.float64)

        if np.all(internal_sections[pipe] >= 2):
            fluid = get_fluid(net)
            f, t = get_lookup(net, "branch", "from_to")[self.table_name]
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

            p_nodes = int_p_lookup[:, 1][selected_indices_p_final]
            m_nodes = int_v_lookup[:, 1][selected_indices_v_final]

            v_pipe_data = pipe_pit[m_nodes, MDOTINIT] / fluid.get_density(NORMAL_TEMPERATURE) / pipe_pit[m_nodes, AREA]
            p_node_data = node_pit[p_nodes, PINIT]
            t_node_data = node_pit[p_nodes, TINIT_NODE]

            gas_mode = fluid.is_gas

            if gas_mode:
                from_nodes = pipe_pit[m_nodes, FROM_NODE].astype(np.int32)
                to_nodes = pipe_pit[m_nodes, TO_NODE].astype(np.int32)
                p_from = node_pit[from_nodes, PAMB] + node_pit[from_nodes, PINIT]
                p_to = node_pit[to_nodes, PAMB] + node_pit[to_nodes, PINIT]
                p_mean = np.where(p_from == p_to, p_from,
                                  2 / 3 * (p_from ** 3 - p_to ** 3) / (p_from ** 2 - p_to ** 2))
                numerator = NORMAL_PRESSURE * node_pit[m_nodes, TINIT_NODE]
                normfactor_mean = numerator * fluid.get_property("compressibility", p_mean) \
                    / (p_mean * NORMAL_TEMPERATURE)
                normfactor_from = numerator * fluid.get_property("compressibility", p_from) \
                    / (p_from * NORMAL_TEMPERATURE)
                normfactor_to = numerator * fluid.get_property("compressibility", p_to) \
                    / (p_to * NORMAL_TEMPERATURE)

                v_pipe_data_mean = v_pipe_data * normfactor_mean
                v_pipe_data_from = v_pipe_data * normfactor_from
                v_pipe_data_to = v_pipe_data * normfactor_to

                pipe_results["VINIT_FROM"][:, 0] = v_pipe_idx
                pipe_results["VINIT_FROM"][:, 1] = v_pipe_data_from
                pipe_results["VINIT_TO"][:, 0] = v_pipe_idx
                pipe_results["VINIT_TO"][:, 1] = v_pipe_data_to
                pipe_results["VINIT_MEAN"][:, 0] = v_pipe_idx
                pipe_results["VINIT_MEAN"][:, 1] = v_pipe_data_mean
            else:
                pipe_results["VINIT_FROM"][:, 0] = v_pipe_idx
                pipe_results["VINIT_FROM"][:, 1] = v_pipe_data
                pipe_results["VINIT_TO"][:, 0] = v_pipe_idx
                pipe_results["VINIT_TO"][:, 1] = v_pipe_data
                pipe_results["VINIT_MEAN"][:, 0] = v_pipe_idx
                pipe_results["VINIT_MEAN"][:, 1] = v_pipe_data

            pipe_results["PINIT"][:, 0] = p_node_idx
            pipe_results["PINIT"][:, 1] = p_node_data
            pipe_results["TINIT"][:, 0] = p_node_idx
            pipe_results["TINIT"][:, 1] = t_node_data

        else:
            logger.warning("For at least one pipe no internal data is available.")

        return pipe_results

    def get_component_input(self):
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
                ("u_w_per_m2k", 'f8'),
                ("text_k", 'f8'),
                ("qext_w", 'f8'),
                ("sections", "u4"),
                ("in_service", 'bool'),
                ("type", dtype(object))]

    def get_result_table(self, net):
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

    def get_internal_pipe_number(self, net):
        """

        :param net: The pandapipes network
        :type net: pandapipesNet
        :return:
        :rtype:
        """
        return np.array(net[self.table_name].sections.values)

    def plot_pipe(self, net, pipe, pipe_results):
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
        pipe_v_data_idx = np.where(pipe_results["VINIT_MEAN"][:, 0] == pipe)
        pipe_p_data = pipe_results["PINIT"][pipe_p_data_idx, 1]
        pipe_t_data = pipe_results["TINIT"][pipe_p_data_idx, 1]
        pipe_v_data = pipe_results["VINIT_MEAN"][pipe_v_data_idx, 1]
        node_pit = net["_pit"]["node"]

        junction_idx_lookup = get_lookup(net, "node", "index")[self.connected_node_type.table_name]
        from_junction_nodes = junction_idx_lookup[net[self.table_name]["from_junction"].values]
        to_junction_nodes = junction_idx_lookup[net[self.table_name]["to_junction"].values]
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
        _, axes = plt.subplots(3, 1, sharex="all")
        axes[0].plot(x_pt, p_values)
        axes[0].set_title("Pressure [bar]")
        axes[1].plot(x_v, v_values)
        axes[1].set_title("Velocity [m/s]")
        axes[2].plot(x_pt, t_values)
        axes[2].set_title("Temperature [K]")

        plt.show()
