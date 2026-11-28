# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype

from pandapipes.component_models import (get_fluid, BranchWOInternalsComponent, get_component_array,
                                         standard_branch_wo_internals_result_lookup)
from pandapipes.component_models.component_toolbox import (
    build_pit_entries, get_hydraulic_options, get_thermal_options, register_node_hydraulic_equation,
    register_node_thermal_equation, register_branch_thermal_equation, register_circ_pump_slack_equations,
)
from pandapipes.component_models.junction_component import Junction
from pandapipes.constants import GRAVITATION_CONSTANT, P_CONVERSION
from pandapipes.idx_branch import IdxBranch
from pandapipes.idx_node import IdxNode
from pandapipes.pf.derivative_calculation import (calculate_derivatives_hydraulic,
                                                  calculate_derivatives_branch_thermal)
from pandapipes.pf.internals_toolbox import get_from_nodes_corrected, get_to_nodes_corrected
from pandapipes.pf.pipeflow_setup import get_lookup
from pandapipes.pf.result_extraction import extract_branch_results_without_internals
from pandapipes.pf.system_index import ComponentEquations, EqWriteMode, HydVarEq, PitEntries, PitWriteMode
from pandapipes.properties.properties_toolbox import get_branch_cp, get_branch_real_density

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)

class HeatGenerator(BranchWOInternalsComponent):
    """Heat generator component that feeds in heat via a prescribed qext, mass flow, temperature lift, flow
    temperature, or - like a pressure circulation pump - a return pressure and pressure lift."""

    # columns for internal array
    MASS = 0
    QEXT = 1
    DELTAT = 2
    TFLOW = 3
    MODE = 4
    # flag for PR_PL generators, required by register_circ_pump_slack_equations
    TYPE_P = 5

    internal_cols = 6

    # heat generator modes (sum of combinations of given parameters)
    MF_DT = 1
    MF_TR = 2
    QE_MF = 3
    QE_DT = 4
    QE_TR = 5
    PR_PL = 6

    @classmethod
    def table_name(cls):
        return "heat_generator"

    @classmethod
    def active_identifier(cls):
        return "in_service"

    @classmethod
    def from_to_node_cols(cls):
        return "return_junction", "flow_junction"

    @classmethod
    def get_connected_node_type(cls):
        return Junction

    @classmethod
    def get_component_input(cls):
        """Get component input.

        :return:
        :rtype:
        """
        return [("name", dtype(object)), ("return_junction", "u4"), ("flow_junction", "u4"), ("qext_w", "f8"),
                ("tflow_k", "f8"), ("controlled_mdot_kg_per_s", "f8"), ("deltat_k", "f8"), ("preturn_bar", "f8"),
                ("plift_bar", "f8"), ("in_service", "bool"), ("type", dtype(object))]

    @classmethod
    def register_pit_node_entries(cls, net, node_pit, registry) -> None:
        tbl = net[cls.table_name()]
        hg_tbl = tbl[tbl[cls.active_identifier()].values]
        mask_pr_pl = (~hg_tbl.preturn_bar.isna() & ~hg_tbl.plift_bar.isna()).values
        if not np.any(mask_pr_pl):
            return

        _, tn_col = cls.from_to_node_cols()
        junction_lookup = get_lookup(net, "node", "index")[cls.get_connected_node_type().table_name()]
        index_p = junction_lookup[hg_tbl[tn_col].values[mask_pr_pl]]
        p_values = hg_tbl.preturn_bar.values[mask_pr_pl] + hg_tbl.plift_bar.values[mask_pr_pl]

        # MEAN: several generators may fix the pressure at the same flow junction
        registry.add_override(PitEntries(*build_pit_entries(
            index_p,
            [IdxNode.PINIT, IdxNode.NODE_TYPE, IdxNode.NODE_TYPE_T],
            [p_values, float(IdxNode.P), float(IdxNode.GE)],
        ), mode=PitWriteMode.MEAN))

    @classmethod
    def register_pit_branch_entries(cls, net, branch_pit, node_pit, registry) -> None:
        super().register_pit_branch_entries(net, branch_pit, node_pit, registry)

        f, t = get_lookup(net, "branch", "from_to")[cls.table_name()]
        tbl = net[cls.table_name()]
        if not len(tbl):
            return

        rows = np.arange(f, t, dtype=np.int32)

        qext = tbl.qext_w.values
        mask_qext = ~np.isnan(qext)
        if np.any(mask_qext):
            registry.add_override(PitEntries(*build_pit_entries(rows[mask_qext], [IdxBranch.QEXT],
                                                                [-qext[mask_qext]])))

        mdot = tbl.controlled_mdot_kg_per_s.values
        mask_mdot = ~np.isnan(mdot)
        if np.any(mask_mdot):
            registry.add_override(PitEntries(*build_pit_entries(rows[mask_mdot], [IdxBranch.MDOTINIT],
                                                                [mdot[mask_mdot]])))

        tflow = tbl.tflow_k.values
        mask_tf = ~np.isnan(tflow)
        if np.any(mask_tf):
            registry.add_override(PitEntries(*build_pit_entries(rows[mask_tf], [IdxBranch.TOUTINIT],
                                                                [tflow[mask_tf]])))

        pl = tbl.plift_bar.values
        mask_pl = ~np.isnan(pl)
        if np.any(mask_pl):
            registry.add_override(PitEntries(*build_pit_entries(rows[mask_pl], [IdxBranch.PL], [pl[mask_pl]])))

        mask_pr_pl = ~np.isnan(tbl.preturn_bar.values) & mask_pl
        if np.any(~mask_pr_pl):
            registry.add_override(PitEntries(*build_pit_entries(rows[~mask_pr_pl], [IdxBranch.FLOW_RETURN_CONNECT],
                                                                [1.])))

    @classmethod
    def create_component_array(cls, net, component_pits):
        """Create an internal array of the component in analogy to the pit.

        Holds component-specific entries that are not needed in the pit.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param component_pits: dictionary of component specific arrays
        :type component_pits: dict
        :return:
        :rtype:
        """
        tbl = net[cls.table_name()]
        hg_array = np.zeros(shape=(len(tbl), cls.internal_cols), dtype=np.float64)
        hg_array[:, cls.DELTAT] = tbl.deltat_k.values
        hg_array[:, cls.TFLOW] = tbl.tflow_k.values
        hg_array[:, cls.QEXT] = tbl.qext_w.values
        hg_array[:, cls.MASS] = tbl.controlled_mdot_kg_per_s.values
        mf = ~np.isnan(tbl.controlled_mdot_kg_per_s.values)
        tf = ~np.isnan(tbl.tflow_k.values)
        dt = ~np.isnan(tbl.deltat_k.values)
        qe = ~np.isnan(tbl.qext_w.values)
        pr = ~np.isnan(tbl.preturn_bar.values)
        pl = ~np.isnan(tbl.plift_bar.values)
        hg_array[pr & pl & tf, cls.MODE] = cls.PR_PL
        hg_array[mf & dt, cls.MODE] = cls.MF_DT
        hg_array[mf & tf, cls.MODE] = cls.MF_TR
        hg_array[qe & mf, cls.MODE] = cls.QE_MF
        hg_array[qe & dt, cls.MODE] = cls.QE_DT
        hg_array[qe & tf, cls.MODE] = cls.QE_TR
        hg_array[:, cls.TYPE_P] = hg_array[:, cls.MODE] == cls.PR_PL
        component_pits[cls.table_name()] = hg_array

    @classmethod
    def register_equations(cls, net, branch_pit, node_pit, sys_idx, registry, mode) -> None:
        if mode == "hydraulics":
            cls._register_hydraulic_equations(net, branch_pit, node_pit, sys_idx, registry)
        elif mode == "heat_transfer":
            cls._register_thermal_equations(net, branch_pit, node_pit, sys_idx, registry)

    @classmethod
    def _register_hydraulic_equations(cls, net, branch_pit, node_pit, sys_idx, registry) -> None:
        f, t = get_lookup(net, "branch", "from_to_active_hydraulics")[cls.table_name()]
        branch_idx = np.arange(f, t, dtype=np.int32)
        if not len(branch_idx):
            return
        b_pit = branch_pit[f:t]
        fn = b_pit[:, IdxBranch.FROM_NODE].astype(np.int32)
        tn = b_pit[:, IdxBranch.TO_NODE].astype(np.int32)

        hg_array = get_component_array(net, cls.table_name(), mode='hydraulics')
        mask_qe_tr = hg_array[:, cls.MODE] == cls.QE_TR
        mask_pr_pl = hg_array[:, cls.MODE] == cls.PR_PL
        mask_generic = ~mask_qe_tr & ~mask_pr_pl

        # QE_DT only provides the initial mass flow, which is then solved like a zero-length branch
        mask_qe_dt = hg_array[:, cls.MODE] == cls.QE_DT
        if np.any(mask_qe_dt):
            cp = get_branch_cp(get_fluid(net), node_pit, b_pit[mask_qe_dt])
            b_pit[mask_qe_dt, IdxBranch.MDOTINIT] = (b_pit[mask_qe_dt, IdxBranch.QEXT]
                                                     / (cp * hg_array[mask_qe_dt, cls.DELTAT]))

        mdot_col = sys_idx.idx(HydVarEq.MDOTINIT, branch_idx)
        branch_eq = sys_idx.idx(HydVarEq.BRANCH, branch_idx)

        if np.any(mask_generic):
            df_dm, df_dp, df_dp1, _, residual, _, _ = calculate_derivatives_hydraulic(
                net, b_pit, node_pit, get_hydraulic_options(net))
            p_from_col = sys_idx.idx(HydVarEq.PINIT, fn[mask_generic])
            p_to_col = sys_idx.idx(HydVarEq.PINIT, tn[mask_generic])
            eq = branch_eq[mask_generic]
            registry.add(ComponentEquations(
                rows=np.concatenate([eq, eq, eq]).astype(np.int32),
                cols=np.concatenate([mdot_col[mask_generic], p_from_col, p_to_col]).astype(np.int32),
                data=np.concatenate([df_dm[mask_generic], df_dp[mask_generic],
                                     df_dp1[mask_generic]]).astype(np.float64),
                residual_rows=eq.astype(np.int32),
                residual_data=residual[mask_generic].astype(np.float64),
                mode=EqWriteMode.UNIQUE,
            ))

        if np.any(mask_qe_tr):
            cp = get_branch_cp(get_fluid(net), node_pit, b_pit[mask_qe_tr])
            from_nodes = get_from_nodes_corrected(b_pit[mask_qe_tr]).astype(np.int32)
            t_in = node_pit[from_nodes, IdxNode.TINIT]
            t_out = b_pit[mask_qe_tr, IdxBranch.TOUTINIT]
            qext = b_pit[mask_qe_tr, IdxBranch.QEXT]
            df_dm_qetr = -cp * (t_out - t_in)
            mask_equal = np.where(qext < 0, t_in >= t_out, t_out >= t_in)
            mask_ign = mask_equal | (qext == 0)

            # reset degenerate rows in the pit itself, so no stale mass flow enters the node balance
            rows_ign = np.flatnonzero(mask_qe_tr)[mask_ign]
            b_pit[rows_ign, IdxBranch.MDOTINIT] = 0.

            df_dm = np.where(mask_ign, 1., df_dm_qetr)
            residual = -qext + df_dm_qetr * b_pit[mask_qe_tr, IdxBranch.MDOTINIT]
            registry.add(ComponentEquations(
                rows=branch_eq[mask_qe_tr].astype(np.int32),
                cols=mdot_col[mask_qe_tr].astype(np.int32),
                data=df_dm.astype(np.float64),
                residual_rows=branch_eq[mask_qe_tr].astype(np.int32),
                residual_data=residual.astype(np.float64),
                mode=EqWriteMode.UNIQUE,
            ))

        if np.any(mask_pr_pl):
            fn_p, tn_p = fn[mask_pr_pl], tn[mask_pr_pl]
            p_from_abs = node_pit[fn_p, IdxNode.PINIT] + node_pit[fn_p, IdxNode.PAMB]
            p_to_abs = node_pit[tn_p, IdxNode.PINIT] + node_pit[tn_p, IdxNode.PAMB]
            rho = get_branch_real_density(get_fluid(net), node_pit, b_pit[mask_pr_pl])
            height_diff = node_pit[fn_p, IdxNode.HEIGHT] - node_pit[tn_p, IdxNode.HEIGHT]
            const_height = rho * GRAVITATION_CONSTANT * height_diff / P_CONVERSION
            residual = p_from_abs - p_to_abs + b_pit[mask_pr_pl, IdxBranch.PL] + const_height

            eq = branch_eq[mask_pr_pl]
            n = len(eq)
            registry.add(ComponentEquations(
                rows=np.concatenate([eq, eq]).astype(np.int32),
                cols=np.concatenate([sys_idx.idx(HydVarEq.PINIT, fn_p),
                                     sys_idx.idx(HydVarEq.PINIT, tn_p)]).astype(np.int32),
                data=np.concatenate([np.ones(n), -np.ones(n)]).astype(np.float64),
                residual_rows=eq.astype(np.int32),
                residual_data=residual.astype(np.float64),
                mode=EqWriteMode.UNIQUE,
            ))
            register_circ_pump_slack_equations(net, branch_pit, node_pit, sys_idx, registry, cls.table_name(),
                                               cls.TYPE_P)

        mass = b_pit[:, IdxBranch.MDOTINIT]
        register_node_hydraulic_equation(sys_idx, registry, branch_idx, fn, tn, np.ones(len(branch_idx)),
                                         -mass, mass)

    @classmethod
    def _register_thermal_equations(cls, net, branch_pit, node_pit, sys_idx, registry) -> None:
        f, t = get_lookup(net, "branch", "from_to_active_heat_transfer")[cls.table_name()]
        branch_idx = np.arange(f, t, dtype=np.int32)
        if not len(branch_idx):
            return
        branch_pit_old = net["_active_old_pit"]["branch"]

        b_pit = branch_pit[f:t]
        hg_array = get_component_array(net, cls.table_name(), mode='heat_transfer')

        mask_mf_dt = hg_array[:, cls.MODE] == cls.MF_DT
        if np.any(mask_mf_dt):
            cp = get_branch_cp(get_fluid(net), node_pit, b_pit[mask_mf_dt])
            b_pit[mask_mf_dt, IdxBranch.QEXT] = (cp * b_pit[mask_mf_dt, IdxBranch.MDOTINIT]
                                                 * hg_array[mask_mf_dt, cls.DELTAT])

        mask_mf_tr = hg_array[:, cls.MODE] == cls.MF_TR
        if np.any(mask_mf_tr):
            cp = get_branch_cp(get_fluid(net), node_pit, b_pit[mask_mf_tr])
            from_nodes = get_from_nodes_corrected(b_pit[mask_mf_tr]).astype(np.int32)
            t_in = node_pit[from_nodes, IdxNode.TINIT]
            t_out = hg_array[mask_mf_tr, cls.TFLOW]
            b_pit[mask_mf_tr, IdxBranch.QEXT] = cp * b_pit[mask_mf_tr, IdxBranch.MDOTINIT] * (t_in - t_out)

        fnt, dfnt_dt, dfnt_dtout, fb, dfb_dt, dfb_dtout = calculate_derivatives_branch_thermal(
            net, b_pit, node_pit, branch_pit_old[f:t], get_thermal_options(net)
        )

        fn = get_from_nodes_corrected(b_pit).astype(np.int32)
        tn = get_to_nodes_corrected(b_pit).astype(np.int32)

        # outlet temperature is fixed to its initial value (tflow_k)
        mask_fixed_tout = (((hg_array[:, cls.MODE] == cls.QE_TR) & (b_pit[:, IdxBranch.QEXT] != 0))
                           | (hg_array[:, cls.MODE] == cls.PR_PL))
        dfb_dt[mask_fixed_tout] = 0
        dfb_dtout[mask_fixed_tout] = 1
        fb[mask_fixed_tout] = 0

        register_branch_thermal_equation(sys_idx, registry, branch_idx, fn, dfb_dt, dfb_dtout, fb,
                                         mode=EqWriteMode.UNIQUE)
        register_node_thermal_equation(sys_idx, registry, branch_idx, tn, dfnt_dt, dfnt_dtout, fnt)

    @classmethod
    def get_result_table(cls, net):
        """Gets the result table.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :return: (columns, all_float) - the column names and whether they are all float type. Only
                if False, returns columns as tuples also specifying the dtypes
        :rtype: (list, bool)
        """
        if get_fluid(net).is_gas:
            output = ["p_from_bar", "p_to_bar", "t_from_k",
                      "t_to_k", "t_outlet_k", "mdot_from_kg_per_s", "mdot_to_kg_per_s", "vdot_norm_m3_per_s",
                      "normfactor_from", "normfactor_to"]
        else:
            output = ["p_from_bar", "p_to_bar", "t_from_k", "t_to_k", "t_outlet_k", "mdot_from_kg_per_s",
                      "mdot_to_kg_per_s", "vdot_m3_per_s"]
        output += ['deltat_k', 'qext_w']
        return output, True

    @classmethod
    def extract_results(cls, net, options, branch_results, mode):
        """Extract heat generator results from the pipeflow internal structure.

        :param net:
        :type net:
        :param options:
        :type options:
        :param branch_results:
        :type branch_results:
        :param mode:
        :type mode:
        :return:
        :rtype:
        """
        required_results_hyd, required_results_ht = standard_branch_wo_internals_result_lookup(net)

        extract_branch_results_without_internals(net, branch_results, required_results_hyd, required_results_ht,
            cls.table_name(), mode)

        node_pit = net['_pit']['node']
        branch_pit = net['_pit']['branch']
        branch_lookups = get_lookup(net, "branch", "from_to")
        f, t = branch_lookups[cls.table_name()]

        res_table = net["res_" + cls.table_name()]

        from_nodes = get_from_nodes_corrected(branch_pit[f:t])
        t_from = node_pit[from_nodes, IdxNode.TINIT]
        tout = branch_pit[f:t, IdxBranch.TOUTINIT]
        mdot = branch_pit[f:t, IdxBranch.MDOTINIT]

        hg_array = get_component_array(net, cls.table_name(), only_active=False)
        mask_prpl = hg_array[:, cls.MODE] == cls.PR_PL
        mask_other = ~mask_prpl

        res_table['deltat_k'].values[:] = t_from - tout

        if np.any(mask_prpl):
            fluid = get_fluid(net)
            cp_i = fluid.get_heat_capacity(t_from[mask_prpl])
            cp_i1 = fluid.get_heat_capacity(tout[mask_prpl])
            res_table['qext_w'].values[mask_prpl] = mdot[mask_prpl] * (cp_i1 * tout[mask_prpl]
                                                                       - cp_i * t_from[mask_prpl])

            mask_reverse = (mdot[mask_prpl] < 0) & ~np.isclose(mdot[mask_prpl], 0)
            if np.any(mask_reverse):
                raise UserWarning(
                    r'Your grid is badly modelled and would lead to a direction change in heat generator %s'
                    % str(net[cls.table_name()].index[mask_prpl][mask_reverse].tolist())
                )

        if np.any(mask_other):
            res_table['qext_w'].values[mask_other] = -branch_pit[f:t, IdxBranch.QEXT][mask_other]
