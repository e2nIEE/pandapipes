# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype

from pandapipes.component_models.abstract_models.circulation_pump import CirculationPump
from pandapipes.component_models.component_toolbox import (
    build_pit_entries, get_thermal_options, register_branch_node_thermal_balance,
    register_circ_pump_node_continuity, register_circ_pump_slack_equations,
)
from pandapipes.component_models.junction_component import Junction
from pandapipes.idx_branch import IdxBranch
from pandapipes.pf.derivative_calculation import calculate_derivatives_branch_thermal
from pandapipes.pf.internals_toolbox import get_to_nodes_corrected
from pandapipes.pf.pipeflow_setup import get_lookup
from pandapipes.pf.system_index import ComponentEquations, EqWriteMode, PitEntries, HydVarEq, ThermVarEq

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class CirculationPumpMass(CirculationPump):

    @classmethod
    def table_name(cls):
        return "circ_pump_mass"

    @classmethod
    def active_identifier(cls):
        return "in_service"

    @classmethod
    def get_connected_node_type(cls):
        return Junction

    @classmethod
    def get_component_input(cls):
        """Get component input.

        :return:
        :rtype:
        """
        return [("name", dtype(object)),
                ("return_junction", "u4"),
                ("flow_junction", "u4"),
                ("p_flow_bar", "f8"),
                ("t_flow_k", "f8"),
                ("mdot_flow_kg_per_s", "f8"),
                ("in_service", 'bool'),
                ("type", dtype(object))]

    @classmethod
    def register_pit_branch_entries(cls, net, branch_pit, node_pit, registry) -> None:
        super().register_pit_branch_entries(net, branch_pit, node_pit, registry)

        f, t = get_lookup(net, "branch", "from_to")[cls.table_name()]
        tbl = net[cls.table_name()]
        if not len(tbl):
            return

        rows = np.arange(f, t, dtype=np.int32)
        registry.add(PitEntries(*build_pit_entries(
            rows, [IdxBranch.MDOTINIT], [tbl.mdot_flow_kg_per_s.values],
        )))

    @classmethod
    def register_hydraulic_equations(cls, net, branch_pit, node_pit, sys_idx, registry):
        register_circ_pump_node_continuity(net, branch_pit, sys_idx, registry, cls.table_name())
        register_circ_pump_slack_equations(net, branch_pit, node_pit, sys_idx, registry, cls.table_name())

        f, t = get_lookup(net, "branch", "from_to_active_hydraulics")[cls.table_name()]
        if f == t:
            return

        branch_idx = np.arange(f, t, dtype=np.int32)

        # variables
        mdot_col = sys_idx.idx(HydVarEq.MDOTINIT, branch_idx)

        # equation position branch
        branch_eq = sys_idx.idx(HydVarEq.BRANCH, branch_idx)

        # system matrix branch: 1 * δm = 0  (mass flow is fixed, no change; override)
        rows_branch = branch_eq.astype(np.int32)
        cols_branch = mdot_col.astype(np.int32)
        data_branch = np.ones(len(branch_idx), dtype=np.float64)
        load_rows_branch = branch_eq.astype(np.int32)
        load_branch = np.zeros(len(branch_idx), dtype=np.float64)

        registry.add_override(ComponentEquations(
            rows=rows_branch,
            cols=cols_branch,
            data=data_branch,
            load_rows=load_rows_branch,
            load_data=load_branch,
            mode=EqWriteMode.UNIQUE,
        ))

    @classmethod
    def register_thermal_equations(cls, net, branch_pit, node_pit, sys_idx, registry):
        f, t = get_lookup(net, "branch", "from_to_active_heat_transfer")[cls.table_name()]
        if f == t:
            return

        branch_idx = np.arange(f, t, dtype=np.int32)
        branch_pit_old = net["_active_old_pit"]["branch"]
        fnt, dfnt_dt, dfnt_dtout, _, _, _ = calculate_derivatives_branch_thermal(
            net, branch_pit[f:t], node_pit, branch_pit_old[f:t], get_thermal_options(net)
        )

        b_pit = branch_pit[f:t]
        tn = get_to_nodes_corrected(b_pit).astype(np.int32)

        # variables
        t_out_col = sys_idx.idx(ThermVarEq.TOUTINIT, branch_idx)

        # equation position branch
        branch_eq = sys_idx.idx(ThermVarEq.BRANCH, branch_idx)

        # system matrix branch: outlet temperature fixed at t_flow_k (override)
        rows_branch = branch_eq.astype(np.int32)
        cols_branch = branch_eq.astype(np.int32)
        data_branch = np.ones(len(branch_idx), dtype=np.float64)
        load_rows_branch = branch_eq.astype(np.int32)
        load_branch = np.zeros(len(branch_idx), dtype=np.float64)

        registry.add_override(ComponentEquations(
            rows=rows_branch,
            cols=cols_branch,
            data=data_branch,
            load_rows=load_rows_branch,
            load_data=load_branch,
            mode=EqWriteMode.UNIQUE,
        ))

        # t_tn_col == the node equation's own row index here (ThermVarEq.NODE and ThermVarEq.TINIT
        # share the same block in a square system, see HeatSystemIndex) - computed explicitly
        # rather than reusing tn_eq, to match register_branch_node_thermal_balance's own signature
        t_tn_col = sys_idx.idx(ThermVarEq.TINIT, tn)
        register_branch_node_thermal_balance(sys_idx, registry, tn, t_tn_col, t_out_col,
                                             dfnt_dt, dfnt_dtout, fnt)
