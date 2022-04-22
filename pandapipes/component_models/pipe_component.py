# Copyright (c) 2020-2022 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import matplotlib.pyplot as plt
import numpy as np
from numpy import dtype

from pandapipes.component_models.abstract_models import BranchWInternalsComponent
from pandapipes.component_models.auxiliaries.component_toolbox import p_correction_height_air, \
    vinterp, set_entry_check_repeat
from pandapipes.component_models.junction_component import Junction
from pandapipes.constants import NORMAL_TEMPERATURE, NORMAL_PRESSURE
from pandapipes.pipeflow_setup import get_lookup
from pandapipes.pipeflow_setup import get_table_number
from pandapipes.properties.fluids import get_mixture_density, is_fluid_gas, get_mixture_compressibility, get_fluid

try:
    from pandaplan.core import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Pipe(BranchWInternalsComponent):
    """

    """

    @classmethod
    def from_to_node_cols(cls):
        return "from_junction", "to_junction"

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
                                                              internal_nodes.astype(np.int32))
            internal_nodes_lookup["TPINIT"][:, 1] = np.arange(current_start, end)

            internal_nodes_lookup["VINIT"] = np.empty((int_pipes_num, 2), dtype=np.int32)
            internal_nodes_lookup["VINIT"][:, 0] = np.repeat(net[cls.table_name()].index,
                                                             internal_pipes.astype(np.int32))
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
        :param node_name:
        :type node_name:
        :return: No Output.
        """
        table_nr, int_node_number, int_node_pit, junction_pit, from_junctions, to_junctions = \
            super().create_pit_node_entries(net, node_pit, node_name)
        if table_nr is None:
            return
        int_node_pit[:, net['_idx_node']['HEIGHT']] = vinterp(junction_pit[from_junctions, net['_idx_node']['HEIGHT']],
                                                              junction_pit[to_junctions, net['_idx_node']['HEIGHT']],
                                                              int_node_number)
        int_node_pit[:, net['_idx_node']['PINIT']] = vinterp(junction_pit[from_junctions, net['_idx_node']['PINIT']],
                                                             junction_pit[to_junctions, net['_idx_node']['PINIT']],
                                                             int_node_number)
        int_node_pit[:, net['_idx_node']['TINIT']] = vinterp(
            junction_pit[from_junctions, net['_idx_node']['TINIT']],
            junction_pit[to_junctions, net['_idx_node']['TINIT']],
            int_node_number)
        int_node_pit[:, net['_idx_node']['PAMB']] = p_correction_height_air(int_node_pit[:, net['_idx_node']['HEIGHT']])
        int_node_pit[:, net['_idx_node']['ACTIVE']] = \
            np.repeat(net[cls.table_name()][cls.active_identifier()].values, int_node_number)
        if len(net._fluid) == 1:
            junction_pit[:, net['_idx_node']['RHO']] = \
                get_fluid(net, net._fluid[0]).get_density(junction_pit[:, net['_idx_node']['TINIT']])

    @classmethod
    def create_property_pit_node_entries(cls, net, node_pit, node_name):
        if len(net._fluid) != 1:
            table_lookup = get_lookup(net, "node", "table")
            table_nr = get_table_number(table_lookup, cls.internal_node_name())
            if table_nr is None:
                return
            ft_lookup = get_lookup(net, "node", "from_to")
            f, t = ft_lookup[cls.internal_node_name()]

            junction_pit = node_pit[f:t, :]
            w = get_lookup(net, 'node', 'w')
            mass_fraction = junction_pit[:, w]
            junction_pit[:, net['_idx_node']['RHO']] = \
                get_mixture_density(net, junction_pit[:, net['_idx_node']['TINIT']], mass_fraction=mass_fraction)

    @classmethod
    def create_pit_branch_entries(cls, net, pipe_pit, node_name):
        """
        Function which creates pit branch entries.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param pipe_pit:
        :type pipe_pit:
        :param node_name:
        :type node_name:
        :return: No Output.
        """
        pipe_pit, internal_pipe_number = \
            super().create_pit_branch_entries(net, pipe_pit, node_name)

        has_internals = np.any(internal_pipe_number > 1)
        tbl = cls.table_name()
        set_entry_check_repeat(
            pipe_pit, net['_idx_branch']['LENGTH'], net[tbl].length_km.values * 1000 / internal_pipe_number,
            internal_pipe_number, has_internals)
        set_entry_check_repeat(
            pipe_pit, net['_idx_branch']['K'], net[tbl].k_mm.values / 1000, internal_pipe_number, has_internals)
        set_entry_check_repeat(
            pipe_pit, net['_idx_branch']['ALPHA'], net[tbl].alpha_w_per_m2k.values, internal_pipe_number, has_internals)
        set_entry_check_repeat(
            pipe_pit, net['_idx_branch']['QEXT'], net[tbl].qext_w.values, internal_pipe_number, has_internals)
        set_entry_check_repeat(
            pipe_pit, net['_idx_branch']['TEXT'], net[tbl].text_k.values, internal_pipe_number, has_internals)
        set_entry_check_repeat(
            pipe_pit, net['_idx_branch']['D'], net[tbl].diameter_m.values, internal_pipe_number, has_internals)
        set_entry_check_repeat(
            pipe_pit, net['_idx_branch']['LOSS_COEFFICIENT'], net[tbl].loss_coefficient.values, internal_pipe_number,
            has_internals)

        pipe_pit[:, net['_idx_branch']['T_OUT']] = 293
        pipe_pit[:, net['_idx_branch']['AREA']] = pipe_pit[:, net['_idx_branch']['D']] ** 2 * np.pi / 4

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
        pipe_pit[:, net['_idx_branch']['PL']] = 0

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
        pipe_pit[:, net['_idx_branch']['TL']] = 0

    @classmethod
    def get_internal_results(cls, net, pipe):
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
        internal_sections = cls.get_internal_pipe_number(net)
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

            p_nodes = int_p_lookup[:, 1][selected_indices_p_final]
            v_nodes = int_v_lookup[:, 1][selected_indices_v_final]

            v_pipe_data = pipe_pit[v_nodes, net['_idx_branch']['VINIT']]
            p_node_data = node_pit[p_nodes, net['_idx_node']['PINIT']]
            t_node_data = node_pit[p_nodes, net['_idx_node']['TINIT']]

            gas_mode = is_fluid_gas(net)

            if gas_mode:
                from_nodes = pipe_pit[v_nodes, net['_idx_branch']['FROM_NODE']].astype(np.int32)
                to_nodes = pipe_pit[v_nodes, net['_idx_branch']['TO_NODE']].astype(np.int32)
                p_from = node_pit[from_nodes, net['_idx_node']['PAMB']] + node_pit[
                    from_nodes, net['_idx_node']['PINIT']]
                p_to = node_pit[to_nodes, net['_idx_node']['PAMB']] + node_pit[to_nodes, net['_idx_node']['PINIT']]
                p_mean = np.where(p_from == p_to, p_from,
                                  2 / 3 * (p_from ** 3 - p_to ** 3) / (p_from ** 2 - p_to ** 2))
                numerator = NORMAL_PRESSURE * node_pit[v_nodes, net['_idx_node']['TINIT']]
                if len(net._fluid) == 1:
                    fluid = get_fluid(net, net._fluid[0])
                    normfactor_mean = numerator * fluid.get_compressibility(p_mean) \
                                      / (p_mean * NORMAL_TEMPERATURE)
                    normfactor_from = numerator * fluid.get_compressibility(p_from) \
                                      / (p_from * NORMAL_TEMPERATURE)
                    normfactor_to = numerator * fluid.get_compressibility(p_to) \
                                    / (p_to * NORMAL_TEMPERATURE)
                else:
                    w = get_lookup(net, 'branch', 'w')
                    mass_fraction = pipe_pit[:, w]
                    normfactor_mean = numerator * get_mixture_compressibility(net, p_mean, mass_fraction) \
                                      / (p_mean * NORMAL_TEMPERATURE)
                    normfactor_from = numerator * get_mixture_compressibility(net, p_from, mass_fraction) \
                                      / (p_from * NORMAL_TEMPERATURE)
                    normfactor_to = numerator * get_mixture_compressibility(net, p_to, mass_fraction) \
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

    @classmethod
    def get_component_input(cls):
        """
        Column names and types of the corresponding DataFrame

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
        if is_fluid_gas(net):
            output = ["v_from_m_per_s", "v_to_m_per_s", "v_mean_m_per_s", "p_from_bar", "p_to_bar",
                      "t_from_k", "t_to_k", "mdot_from_kg_per_s", "mdot_to_kg_per_s",
                      "vdot_norm_m3_per_s", "reynolds", "lambda", "normfactor_from",
                      "normfactor_to"]
            add = ["w_%s" % fluid for fluid in net._fluid]

        else:
            output = ["v_mean_m_per_s", "p_from_bar", "p_to_bar", "t_from_k", "t_to_k",
                      "mdot_from_kg_per_s", "mdot_to_kg_per_s", "vdot_norm_m3_per_s", "reynolds",
                      "lambda"]
            add = []

        return output + add, True

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
        pipe_v_data_idx = np.where(pipe_results["VINIT_MEAN"][:, 0] == pipe)
        pipe_p_data = pipe_results["PINIT"][pipe_p_data_idx, 1]
        pipe_t_data = pipe_results["TINIT"][pipe_p_data_idx, 1]
        pipe_v_data = pipe_results["VINIT_MEAN"][pipe_v_data_idx, 1]
        node_pit = net["_pit"]["node"]

        junction_idx_lookup = get_lookup(net, "node", "index")[Junction.table_name()]
        from_junction_nodes = junction_idx_lookup[net[cls.table_name()]["from_junction"].values]
        to_junction_nodes = junction_idx_lookup[net[cls.table_name()]["to_junction"].values]
        p_values = np.zeros(len(pipe_p_data[0]) + 2)
        p_values[0] = node_pit[from_junction_nodes[pipe], net['_idx_node']['PINIT']]
        p_values[1:-1] = pipe_p_data[:]
        p_values[-1] = node_pit[to_junction_nodes[pipe], net['_idx_node']['PINIT']]

        t_values = np.zeros(len(pipe_t_data[0]) + 2)
        t_values[0] = node_pit[from_junction_nodes[pipe], net['_idx_node']['TINIT_NODE']]
        t_values[1:-1] = pipe_t_data[:]
        t_values[-1] = node_pit[to_junction_nodes[pipe], net['_idx_node']['TINIT_NODE']]

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
