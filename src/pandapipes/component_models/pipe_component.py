# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import matplotlib.pyplot as plt
import numpy as np
from numpy import dtype

from pandapipes.component_models.abstract_models import BranchWInternalsComponent
from pandapipes.component_models.component_toolbox import (
    build_pit_entries, vinterp, p_correction_height_air, get_hydraulic_options, get_thermal_options,
    register_branch_node_mass_balance, register_branch_node_thermal_balance,
)
from pandapipes.component_models.junction_component import Junction
from pandapipes.constants import NORMAL_TEMPERATURE, NORMAL_PRESSURE
from pandapipes.idx_branch import IdxBranch
from pandapipes.idx_node import IdxNode
from pandapipes.pf.derivative_calculation import calculate_derivatives_hydraulic, calculate_derivatives_branch_thermal
from pandapipes.pf.internals_toolbox import branch_area, get_from_nodes_corrected, get_to_nodes_corrected
from pandapipes.pf.pipeflow_setup import get_fluid, get_lookup, get_net_option, get_table_number
from pandapipes.pf.result_extraction import extract_branch_results_with_internals, \
    extract_branch_results_without_internals
from pandapipes.pf.system_index import ComponentEquations, BaseSystemIndex, HydVarEq, PitEntries, ThermVarEq

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Pipe(BranchWInternalsComponent):
    """Pipe branch component with internal sections."""

    @classmethod
    def table_name(cls):
        return "pipe"

    @classmethod
    def active_identifier(cls):
        return "in_service"

    @classmethod
    def get_connected_node_type(cls):
        return Junction

    @classmethod
    def from_to_node_cols(cls):
        return "from_junction", "to_junction"

    @classmethod
    def internal_node_name(cls):
        return "pipe_nodes"

    @classmethod
    def get_internal_node_number(cls, net, return_internal_only=True):
        """Get the number of internal nodes per pipe.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :return:
        :rtype:
        """
        return cls.get_internal_branch_number(net) - 1

    @classmethod
    def get_internal_branch_number(cls, net):
        """Get the number of internal branches (sections) per pipe.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :return:
        :rtype:
        """
        return np.array(net[cls.table_name()].sections.values).astype(np.int32)

    @classmethod
    def get_component_input(cls):
        """Get the component input columns for this table.

        :return:
        :rtype:
        """
        return [("name", dtype(object)), ("from_junction", "u4"), ("to_junction", "u4"), ("std_type", dtype(object)),
                ("length_km", "f8"), ("inner_diameter_mm", "f8"), ("outer_diameter_mm", "f8"),
                ("k_mm", "f8"), ("loss_coefficient", "f8"),
                ("u_w_per_m2k", 'f8'), ("text_k", 'f8'), ("sections", "u4"), ("in_service", 'bool'),
                ("type", dtype(object))]

    @classmethod
    def register_pit_node_entries(cls, net, node_pit, registry) -> None:
        super().register_pit_node_entries(net, node_pit, registry)

        table_lookup = get_lookup(net, "node", "table")
        if get_table_number(table_lookup, cls.internal_node_name()) is None:
            return

        ft_lookup = get_lookup(net, "node", "from_to")
        f, t = ft_lookup[cls.internal_node_name()]
        int_node_number = cls.get_internal_node_number(net)
        junction_table_name = cls.get_connected_node_type().table_name()
        fj_name, tj_name = cls.from_to_node_cols()
        from_junctions = net[cls.table_name()][fj_name].values.astype(np.int32)
        to_junctions = net[cls.table_name()][tj_name].values.astype(np.int32)
        junction_table = net[junction_table_name]

        rows = np.arange(f, t, dtype=np.int32)
        tinit_vals = vinterp(junction_table.loc[from_junctions, "tfluid_k"].values,
                             junction_table.loc[to_junctions, "tfluid_k"].values, int_node_number)
        pinit_vals = vinterp(junction_table.loc[from_junctions, "pn_bar"].values,
                             junction_table.loc[to_junctions, "pn_bar"].values, int_node_number)

        if not get_net_option(net, "transient") or get_net_option(net, "simulation_time_step") == 0:
            height_vals = vinterp(junction_table.loc[from_junctions, "height_m"].values,
                                  junction_table.loc[to_junctions, "height_m"].values, int_node_number)
            pamb_vals = p_correction_height_air(height_vals)
            active_vals = np.repeat(net[cls.table_name()][cls.active_identifier()].values, int_node_number).astype(float)
            registry.add(PitEntries(*build_pit_entries(
                rows,
                [IdxNode.TINIT, IdxNode.PINIT, IdxNode.HEIGHT, IdxNode.PAMB, IdxNode.ACTIVE],
                [tinit_vals, pinit_vals, height_vals, pamb_vals, active_vals],
            )))
        else:
            registry.add(PitEntries(*build_pit_entries(
                rows,
                [IdxNode.TINIT, IdxNode.PINIT],
                [tinit_vals, pinit_vals])))

    @classmethod
    def register_pit_branch_entries(cls, net, branch_pit, node_pit, registry) -> None:
        super().register_pit_branch_entries(net, branch_pit, node_pit, registry)

        f, t = get_lookup(net, "branch", "from_to")[cls.table_name()]
        junction_idx_lookup = get_lookup(net, "node", "index")[cls.get_connected_node_type().table_name()]
        fn_col, tn_col = cls.from_to_node_cols()
        from_nodes = junction_idx_lookup[net[cls.table_name()][fn_col].values]
        to_nodes = junction_idx_lookup[net[cls.table_name()][tn_col].values]
        internal_pipe_number = cls.get_internal_branch_number(net)
        has_internals = np.any(internal_pipe_number > 1)

        if has_internals:
            internal_node_number = cls.get_internal_node_number(net)
            node_ft_lookups = get_lookup(net, "node", "from_to")
            pipe_nodes_from, pipe_nodes_to = node_ft_lookups[cls.internal_node_name()]
            pipe_nodes_idx = np.arange(pipe_nodes_from, pipe_nodes_to)
            insert_places = np.repeat(np.arange(len(from_nodes)), internal_node_number)
            from_nodes = np.insert(from_nodes, insert_places + 1, pipe_nodes_idx)
            to_nodes = np.insert(to_nodes, insert_places, pipe_nodes_idx)

        rows = np.arange(f, t, dtype=np.int32)
        tbl = cls.table_name()
        junction_table_name = cls.get_connected_node_type().table_name()

        def _rep(vals):
            return np.repeat(vals, internal_pipe_number) if has_internals else vals

        to_junctions_br = net[tbl][tn_col].values
        toutinit_vals = _rep(net[junction_table_name].loc[to_junctions_br, "tfluid_k"].values)
        d_vals = _rep(net[tbl].inner_diameter_mm.values / 1000.)
        area_vals = d_vals ** 2 * np.pi / 4
        mdotinit_vals = 0.1 * area_vals * get_fluid(net).get_density(NORMAL_TEMPERATURE)

        if not get_net_option(net, "transient") or get_net_option(net, "simulation_time_step") == 0:
            length_vals = _rep(net[tbl].length_km.values * 1000 / internal_pipe_number)
            k_vals = _rep(net[tbl].k_mm.values / 1000)
            alpha_vals = _rep(net[tbl].u_w_per_m2k.values)
            text_vals = _rep(net[tbl].text_k.values)
            text_vals[np.isnan(text_vals)] = get_net_option(net, 'ambient_temperature')

            registry.add(PitEntries(*build_pit_entries(
                rows,
                [IdxBranch.FROM_NODE, IdxBranch.TO_NODE, IdxBranch.LENGTH, IdxBranch.K, IdxBranch.ALPHA, IdxBranch.TEXT, IdxBranch.TOUTINIT],
                [from_nodes.astype(float), to_nodes.astype(float),
                 length_vals, k_vals, alpha_vals, text_vals, toutinit_vals],
            )))
        else:
            registry.add(PitEntries(*build_pit_entries(
                rows, [IdxBranch.TOUTINIT], [toutinit_vals],
            )))

        registry.add(PitEntries(*build_pit_entries(
            rows, [IdxBranch.MDOTINIT], [mdotinit_vals],
        )))

    @classmethod
    def register_hydraulic_equations(cls, net, branch_pit, node_pit,
                                     sys_idx: BaseSystemIndex, registry) -> None:
        f, t = get_lookup(net, "branch", "from_to_active_hydraulics")[cls.table_name()]
        branch_idx = np.arange(f, t, dtype=np.int32)
        if not len(branch_idx):
            return
        df_dm, df_dp, df_dp1, df_dm_node, load, load_fn, load_tn = (
            calculate_derivatives_hydraulic(net, branch_pit[f:t], node_pit, get_hydraulic_options(net))
        )

        b_pit = branch_pit[f:t]
        fn = b_pit[:, IdxBranch.FROM_NODE].astype(np.int32)
        tn = b_pit[:, IdxBranch.TO_NODE].astype(np.int32)

        # variables
        mdot_col   = sys_idx.idx(HydVarEq.MDOTINIT, branch_idx)
        p_from_col = sys_idx.idx(HydVarEq.PINIT, fn)
        p_to_col   = sys_idx.idx(HydVarEq.PINIT, tn)

        # equation position branch
        branch_eq  = sys_idx.idx(HydVarEq.BRANCH, branch_idx)

        # system matrix branch
        rows_branch = np.concatenate([branch_eq, branch_eq, branch_eq]).astype(np.int32)
        cols_branch = np.concatenate([mdot_col, p_from_col, p_to_col]).astype(np.int32)
        data_branch = np.concatenate([df_dm, df_dp, df_dp1]).astype(np.float64)
        load_rows_branch = branch_eq.astype(np.int32)
        load_branch = load.astype(np.float64)

        registry.add(ComponentEquations(
            rows=rows_branch,
            cols=cols_branch,
            data=data_branch,
            load_rows=load_rows_branch,
            load_data=load_branch,
        ))

        register_branch_node_mass_balance(sys_idx, registry, fn, tn, mdot_col, df_dm_node,
                                          -load_fn, load_tn)

    @classmethod
    def register_thermal_equations(cls, net, branch_pit, node_pit,
                                   sys_idx: BaseSystemIndex, registry) -> None:
        f, t = get_lookup(net, "branch", "from_to_active_heat_transfer")[cls.table_name()]
        branch_idx = np.arange(f, t, dtype=np.int32)
        if not len(branch_idx):
            return
        branch_pit_old = net["_active_old_pit"]["branch"]
        fnt, dfnt_dt, dfnt_dtout, fb, dfb_dt, dfb_dtout = (
            calculate_derivatives_branch_thermal(net, branch_pit[f:t], node_pit, branch_pit_old[f:t],
                                                 get_thermal_options(net))
        )

        pipe_pit = branch_pit[f:t]
        fn = get_from_nodes_corrected(pipe_pit).astype(np.int32)
        tn = get_to_nodes_corrected(pipe_pit).astype(np.int32)

        # variables
        t_out_col  = sys_idx.idx(ThermVarEq.TOUTINIT, branch_idx)
        t_from_col = sys_idx.idx(ThermVarEq.TINIT, fn)
        t_tn_col   = sys_idx.idx(ThermVarEq.TINIT, tn)

        # equation position branch
        branch_eq  = sys_idx.idx(ThermVarEq.BRANCH, branch_idx)

        # system matrix branch
        rows_branch = np.concatenate([branch_eq, branch_eq]).astype(np.int32)
        cols_branch = np.concatenate([t_from_col, t_out_col]).astype(np.int32)
        data_branch = np.concatenate([dfb_dt, dfb_dtout]).astype(np.float64)
        load_rows_branch = branch_eq.astype(np.int32)
        load_branch = fb.astype(np.float64)

        registry.add(ComponentEquations(
            rows=rows_branch,
            cols=cols_branch,
            data=data_branch,
            load_rows=load_rows_branch,
            load_data=load_branch,
        ))

        register_branch_node_thermal_balance(sys_idx, registry, tn, t_tn_col, t_out_col,
                                             dfnt_dt, dfnt_dtout, fnt)

    @classmethod
    def geodata(cls):
        """Get geodata columns.

        :return:
        :rtype:
        """
        return [("coords", dtype(object))]

    @classmethod
    def get_result_table(cls, net):
        """Get the result table columns.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :return: (columns, all_float) - the column names and whether they are all float type. Only
                if False, returns columns as tuples also specifying the dtypes
        :rtype: (list, bool)
        """
        if get_fluid(net).is_gas:
            output = ["v_from_m_per_s", "v_to_m_per_s", "v_mean_m_per_s", "p_from_bar", "p_to_bar", "t_from_k",
                      "t_to_k", "t_outlet_k", "mdot_from_kg_per_s", "mdot_to_kg_per_s", "vdot_norm_m3_per_s",
                      "reynolds", "lambda", "normfactor_from", "normfactor_to", "dp_friction_loss_bar"]
        else:
            output = ["v_mean_m_per_s", "p_from_bar", "p_to_bar", "t_from_k", "t_to_k", "t_outlet_k",
                      "mdot_from_kg_per_s", "mdot_to_kg_per_s", "vdot_m3_per_s", "reynolds", "lambda", "dp_friction_loss_bar"]
        return output, True

    @classmethod
    def extract_results(cls, net, options, branch_results, mode):
        res_nodes_from_hyd = [("p_from_bar", "p_from"), ("mdot_from_kg_per_s", "mf_from")]
        res_nodes_from_ht = [("t_from_k", "temp_from")]
        res_nodes_to_hyd = [("p_to_bar", "p_to"), ("mdot_to_kg_per_s", "mf_to")]
        res_nodes_to_ht = [("t_to_k", "temp_to")]
        res_mean_hyd = [("lambda", "lambda"), ("reynolds", "reynolds"), ("dp_friction_loss_bar", "dp_frict_loss")]
        res_branch_ht = [("t_outlet_k", "t_outlet")]

        if get_fluid(net).is_gas:
            res_nodes_from_hyd.extend([("v_from_m_per_s", "v_gas_from"), ("normfactor_from", "normfactor_from")])
            res_nodes_to_hyd.extend([("v_to_m_per_s", "v_gas_to"), ("normfactor_to", "normfactor_to")])
            res_mean_hyd.extend([("v_mean_m_per_s", "v_gas_mean"), ("vdot_norm_m3_per_s", "vf")])
        else:
            res_mean_hyd.extend([("v_mean_m_per_s", "v_mps"), ("vdot_m3_per_s", "vf")])

        if np.any(cls.get_internal_node_number(net) > 0):
            extract_branch_results_with_internals(net, branch_results, cls.table_name(), res_nodes_from_hyd,
                res_nodes_from_ht, res_nodes_to_hyd, res_nodes_to_ht, res_mean_hyd, res_branch_ht, [],
                cls.internal_node_name(), mode)
        else:
            required_results_hyd = res_nodes_from_hyd + res_nodes_to_hyd + res_mean_hyd
            required_results_ht = res_nodes_from_ht + res_nodes_to_ht + res_branch_ht
            extract_branch_results_without_internals(net, branch_results, required_results_hyd, required_results_ht,
                cls.table_name(), mode)

    @classmethod
    def get_internal_results(cls, net, pipe):
        """Retrieve velocity (at to/from node; mean), pressure and temperature of the internal sections of pipes.

        The pipes have to have at least 2 internal sections.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param pipe: indices of pipes to evaluate
        :type pipe: np.array
        :return: pipe_results
        :rtype:
        """
        internal_nodes = cls.get_internal_node_number(net)
        internal_sections = internal_nodes + 1
        p_node_idx = np.repeat(pipe, internal_nodes[pipe])
        v_pipe_idx = np.repeat(pipe, internal_sections[pipe])
        pipe_results = dict()
        pipe_results["PINIT"] = np.zeros((len(p_node_idx), 2), dtype=np.float64)
        pipe_results["TINIT"] = np.zeros((len(p_node_idx), 2), dtype=np.float64)
        pipe_results["VINIT_FROM"] = np.zeros((len(v_pipe_idx), 2), dtype=np.float64)
        pipe_results["VINIT_TO"] = np.zeros((len(v_pipe_idx), 2), dtype=np.float64)
        pipe_results["VINIT_MEAN"] = np.zeros((len(v_pipe_idx), 2), dtype=np.float64)

        if np.all(internal_sections[pipe] >= 2):
            fluid = get_fluid(net)
            f, t = get_lookup(net, "branch", "from_to")[cls.table_name()]
            pipe_pit = net["_pit"]["branch"][f:t, :]
            node_pit = net["_pit"]["node"]
            int_p_lookup = net["_lookups"]["internal_nodes"][cls.table_name()]
            int_v_lookup = net["_lookups"]["internal_branches"][cls.table_name()]

            pipe_lookup_index = get_lookup(net, 'branch', 'index')['pipe'][pipe]

            p_nodes = int_p_lookup[pipe_lookup_index]
            p_nodes = [np.arange(x, y + 1) for x,y in zip(p_nodes[:, 0], p_nodes[:, 1])]
            m_nodes = int_v_lookup[pipe_lookup_index]
            m_nodes = [np.arange(x, y + 1) for x,y in zip(m_nodes[:, 0], m_nodes[:, 1])]

            v_pipe_data = pipe_pit[m_nodes, IdxBranch.MDOTINIT] / fluid.get_density(NORMAL_TEMPERATURE) / (
                branch_area(pipe_pit)[m_nodes])
            p_node_data = node_pit[p_nodes, IdxNode.PINIT]
            t_node_data = node_pit[p_nodes, IdxNode.TINIT]

            gas_mode = fluid.is_gas

            if gas_mode:
                from_nodes = pipe_pit[m_nodes, IdxBranch.FROM_NODE].astype(np.int32)
                to_nodes = pipe_pit[m_nodes, IdxBranch.TO_NODE].astype(np.int32)
                p_from = node_pit[from_nodes, IdxNode.PAMB] + node_pit[from_nodes, IdxNode.PINIT]
                p_to = node_pit[to_nodes, IdxNode.PAMB] + node_pit[to_nodes, IdxNode.PINIT]
                p_mean = np.where(p_from == p_to, p_from, 2 / 3 * (p_from ** 3 - p_to ** 3) / (p_from ** 2 - p_to ** 2))
                factor = NORMAL_PRESSURE * node_pit[m_nodes, IdxNode.TINIT] / NORMAL_TEMPERATURE

                args_from, args_to, args_mean = [p_from], [p_to], [p_mean]
                if (hasattr(fluid.all_properties["compressibility"], "allow_2d")
                        and fluid.all_properties["compressibility"].allow_2d):
                    # TODO: this is only allowed without temperature calculation (assumed for gases)
                    t_from = node_pit[from_nodes, IdxNode.TINIT]
                    t_to = node_pit[to_nodes, IdxNode.TINIT]
                    args_from.append(t_from)
                    args_to.append(t_to)
                    args_mean.append((t_from + t_to) / 2)

                normfactor_mean = factor * fluid.get_compressibility(*args_mean) / p_mean
                normfactor_from = factor * fluid.get_compressibility(*args_from) / p_from
                normfactor_to = factor * fluid.get_compressibility(*args_to) / p_to

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
    def plot_pipe(cls, net, pipe, pipe_results):
        """Plot pressure, velocity and temperature profiles along a pipe.

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
        p_values[0] = node_pit[from_junction_nodes[pipe], IdxNode.PINIT]
        p_values[1:-1] = pipe_p_data[:]
        p_values[-1] = node_pit[to_junction_nodes[pipe], IdxNode.PINIT]

        t_values = np.zeros(len(pipe_t_data[0]) + 2)
        t_values[0] = node_pit[from_junction_nodes[pipe], IdxNode.TINIT]
        t_values[1:-1] = pipe_t_data[:]
        t_values[-1] = node_pit[to_junction_nodes[pipe], IdxNode.TINIT]

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
