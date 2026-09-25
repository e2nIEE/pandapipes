# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype

from pandapipes.component_models.abstract_models.branch_w_internals_models import BranchWInternalsComponent
from pandapipes.component_models.component_toolbox import (
    build_pit_entries, p_correction_height_air, standard_branch_wo_internals_result_lookup,
    get_hydraulic_options, get_thermal_options, register_branch_node_mass_balance,
    register_branch_node_thermal_balance,
)
from pandapipes.component_models.junction_component import Junction
from pandapipes.constants import NORMAL_TEMPERATURE
from pandapipes.idx_branch import IdxBranch
from pandapipes.idx_node import IdxNode
from pandapipes.pf.derivative_calculation import (
    calculate_derivatives_hydraulic, calculate_derivatives_branch_thermal,
)
from pandapipes.pf.internals_toolbox import get_from_nodes_corrected, get_to_nodes_corrected
from pandapipes.pf.pipeflow_setup import get_fluid, get_net_option, get_lookup
from pandapipes.pf.result_extraction import extract_branch_results_without_internals
from pandapipes.pf.system_index import ComponentEquations, HydVarEq, PitEntries, ThermVarEq


class Valve(BranchWInternalsComponent):
    """Valves are branch elements that can separate two junctions.

    They have a length of 0, but can introduce a lumped pressure loss.
    """

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
    def from_to_node_cols(cls):
        return "junction", "element"

    @classmethod
    def internal_node_name(cls):
        return "valve_nodes"

    @classmethod
    def get_internal_node_number(cls, net, return_internal_only=True):
        int_nodes = np.zeros(len(net[cls.table_name()]), dtype=np.int32)
        mask_p = np.flatnonzero(net[cls.table_name()]['et'].values == 'pi')
        val = net[cls.table_name()][list(cls.from_to_node_cols())].values[mask_p]
        _, idx, inv = np.unique(val, return_index=True, return_inverse=True, axis=0)
        idx_inv = np.empty_like(idx)
        order = np.argsort(idx)
        idx_inv[order] = np.arange(len(idx))
        int_nodes[mask_p[idx]] = 1
        return int_nodes if return_internal_only else (int_nodes, idx_inv[inv], mask_p)

    @classmethod
    def get_internal_branch_number(cls, net):
        return np.ones(len(net[cls.table_name()]), dtype=np.int32)

    @classmethod
    def get_component_input(cls):
        return [
            ("name", dtype(object)),
            ("junction", "i8"),
            ("element", "i8"),
            ("et", dtype(object)),
            ("inner_diameter_mm", "f8"),
            ("opened", "bool"),
            ("loss_coefficient", "f8"),
            ("type", dtype(object))
        ]

    @classmethod
    def register_pit_node_entries(cls, net, node_pit, registry) -> None:
        super().register_pit_node_entries(net, node_pit, registry)

        node_ft_lookups = get_lookup(net, "node", "from_to")
        if cls.internal_node_name() not in node_ft_lookups:
            return
        f, t = node_ft_lookups[cls.internal_node_name()]
        if f == t:
            return

        int_node_number = cls.get_internal_node_number(net)
        junction_table_name = cls.get_connected_node_type().table_name()
        fj_name, _ = cls.from_to_node_cols()

        from_junctions = net[cls.table_name()][fj_name].values.astype(np.int32)
        junction_indices = get_lookup(net, "node", "index")[junction_table_name]
        junct_pit_index = junction_indices[from_junctions]
        fj_nodes = np.repeat(junct_pit_index, int_node_number)

        f_junc, _ = node_ft_lookups[junction_table_name]
        junc_df = net[junction_table_name]
        local_idx = fj_nodes - f_junc

        rows = np.arange(f, t, dtype=np.int32)

        registry.add(PitEntries(*build_pit_entries(
            rows,
            [IdxNode.TINIT, IdxNode.PINIT],
            [junc_df.tfluid_k.values[local_idx], junc_df.pn_bar.values[local_idx]],
        )))
        if not get_net_option(net, "transient") or get_net_option(net, "simulation_time_step") == 0:
            height_vals = junc_df.height_m.values[local_idx]
            registry.add(PitEntries(*build_pit_entries(
                rows,
                [IdxNode.HEIGHT, IdxNode.PAMB, IdxNode.ACTIVE],
                [height_vals, p_correction_height_air(height_vals),
                 junc_df.in_service.values[local_idx].astype(float)],
            )))

    @classmethod
    def register_pit_branch_entries(cls, net, branch_pit, node_pit, registry) -> None:
        super().register_pit_branch_entries(net, branch_pit, node_pit, registry)

        f, t = get_lookup(net, "branch", "from_to")[cls.table_name()]
        tbl = net[cls.table_name()]
        if not len(tbl):
            return

        internal_node_number, inverse_index, mask_p = cls.get_internal_node_number(net, False)

        fn_col, tn_col = cls.from_to_node_cols()
        junction_table_name = cls.get_connected_node_type().table_name()
        junction_idx_lookup = get_lookup(net, "node", "index")[junction_table_name]
        f_junc, _ = get_lookup(net, "node", "from_to")[junction_table_name]
        junc_df = net[junction_table_name]

        from_junctions_raw = tbl[fn_col].values.astype(np.int32)
        from_nodes = junction_idx_lookup[from_junctions_raw]
        to_nodes = np.zeros(len(tbl), dtype=np.int64)
        mask_j = tbl.et.values == 'ju'
        to_elements = tbl[tn_col].values
        to_nodes[mask_j] = junction_idx_lookup[to_elements[mask_j].astype(np.int32)]

        rows = np.arange(f, t, dtype=np.int32)

        if not get_net_option(net, "transient") or get_net_option(net, "simulation_time_step") == 0:
            has_internals = np.any(internal_node_number > 0)
            if has_internals:
                pipe_idx_lookup = get_lookup(net, "branch", "index")['pipe']
                mask_p_uni = internal_node_number.astype(bool)
                pipes = pipe_idx_lookup[to_elements[mask_p_uni].astype(np.int32)]

                internal = net['_lookups']['internal_branches']['pipe']

                # Determine pipe direction from net.pipe (branch_pit not yet filled at registry phase)
                pipe_from_junctions = net['pipe'].loc[
                    to_elements[mask_p_uni].astype(np.int32), 'from_junction'
                ].values
                fn_pipe = junction_idx_lookup[pipe_from_junctions.astype(np.int32)]
                fp = fn_pipe == from_nodes[mask_p_uni]

                vn_f, vn_t = get_lookup(net, "node", "from_to")['valve_nodes']
                valve_nodes = np.arange(vn_f, vn_t, dtype=np.int32)

                if np.any(fp):
                    registry.add_override(PitEntries(*build_pit_entries(
                        internal[pipes[fp], 0].astype(np.int32),
                        [IdxBranch.FROM_NODE], [valve_nodes[fp].astype(float)],
                    )))
                if np.any(~fp):
                    registry.add_override(PitEntries(*build_pit_entries(
                        internal[pipes[~fp], 1].astype(np.int32),
                        [IdxBranch.TO_NODE], [valve_nodes[~fp].astype(float)],
                    )))

                to_nodes[mask_p] = valve_nodes[inverse_index]

            registry.add(PitEntries(*build_pit_entries(
                rows,
                [IdxBranch.FROM_NODE, IdxBranch.TO_NODE, IdxBranch.K, IdxBranch.TEXT],
                [from_nodes.astype(float), to_nodes.astype(float),
                 np.full(len(rows), 1e-3),
                 np.full(len(rows), get_net_option(net, 'ambient_temperature'))],
            )))

        # TOUTINIT — always set (not conditional on transient)
        toutinit = np.zeros(len(rows))
        if np.any(mask_j):
            toutinit[mask_j] = junc_df.tfluid_k.values[
                junction_idx_lookup[to_elements[mask_j].astype(np.int32)] - f_junc
            ]
        if len(mask_p):
            toutinit[mask_p] = junc_df.tfluid_k.values[
                junction_idx_lookup[from_junctions_raw[mask_p]] - f_junc
            ]
        registry.add(PitEntries(*build_pit_entries(rows, [IdxBranch.TOUTINIT], [toutinit])))

        d_vals = tbl.inner_diameter_mm.values / 1000.
        area_vals = d_vals ** 2 * np.pi / 4
        mdotinit_vals = 0.1 * area_vals * get_fluid(net).get_density(NORMAL_TEMPERATURE)
        registry.add(PitEntries(*build_pit_entries(rows, [IdxBranch.MDOTINIT], [mdotinit_vals])))

    @classmethod
    def register_hydraulic_equations(cls, net, branch_pit, node_pit, sys_idx, registry) -> None:
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
    def register_thermal_equations(cls, net, branch_pit, node_pit, sys_idx, registry):
        f, t = get_lookup(net, "branch", "from_to_active_heat_transfer")[cls.table_name()]
        branch_idx = np.arange(f, t, dtype=np.int32)
        if not len(branch_idx):
            return
        branch_pit_old = net["_active_old_pit"]["branch"]
        fnt, dfnt_dt, dfnt_dtout, fb, dfb_dt, dfb_dtout = (
            calculate_derivatives_branch_thermal(net, branch_pit[f:t], node_pit, branch_pit_old[f:t],
                                                 get_thermal_options(net))
        )

        b_pit = branch_pit[f:t]
        fn = get_from_nodes_corrected(b_pit).astype(np.int32)
        tn = get_to_nodes_corrected(b_pit).astype(np.int32)

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
    def get_result_table(cls, net):
        if get_fluid(net).is_gas:
            output = ["v_from_m_per_s", "v_to_m_per_s", "v_mean_m_per_s", "p_from_bar", "p_to_bar", "t_from_k",
                      "t_to_k", "t_outlet_k", "mdot_from_kg_per_s", "mdot_to_kg_per_s", "vdot_norm_m3_per_s",
                      "reynolds", "lambda", "normfactor_from", "normfactor_to"]
        else:
            output = ["v_mean_m_per_s", "p_from_bar", "p_to_bar", "t_from_k", "t_to_k", "t_outlet_k",
                      "mdot_from_kg_per_s", "mdot_to_kg_per_s", "vdot_m3_per_s", "reynolds", "lambda"]
        return output, True

    @classmethod
    def extract_results(cls, net, options, branch_results, mode):
        required_results_hyd, required_results_ht = standard_branch_wo_internals_result_lookup(net)

        required_results_hyd.extend([("v_mean_m_per_s", "v_mps"), ("lambda", "lambda"), ("reynolds", "reynolds")])

        if get_fluid(net).is_gas:
            required_results_hyd.extend([("v_from_m_per_s", "v_gas_from"), ("v_to_m_per_s", "v_gas_to")])

        extract_branch_results_without_internals(net, branch_results, required_results_hyd, required_results_ht,
                                                 cls.table_name(), mode)
