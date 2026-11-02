# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype

from pandapipes.component_models import standard_branch_wo_internals_result_lookup
from pandapipes.component_models.abstract_models.branch_wo_internals_models import \
    BranchWOInternalsComponent
from pandapipes.component_models.component_toolbox import (
    build_pit_entries, get_hydraulic_options, get_thermal_options, register_node_hydraulic_equation,
    register_node_thermal_equation, register_branch_hydraulic_equation,
    register_branch_thermal_equation,
)
from pandapipes.component_models.junction_component import Junction
from pandapipes.idx_branch import IdxBranch
from pandapipes.pf.derivative_calculation import calculate_derivatives_hydraulic, calculate_derivatives_branch_thermal
from pandapipes.pf.internals_toolbox import get_from_nodes_corrected, get_to_nodes_corrected
from pandapipes.pf.pipeflow_setup import get_fluid, get_lookup
from pandapipes.pf.result_extraction import extract_branch_results_without_internals
from pandapipes.pf.system_index import PitEntries, PitWriteMode

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class HeatExchanger(BranchWOInternalsComponent):

    @classmethod
    def table_name(cls):
        return "heat_exchanger"

    @classmethod
    def active_identifier(cls):
        return "in_service"

    @classmethod
    def from_to_node_cols(cls):
        return "from_junction", "to_junction"

    @classmethod
    def get_connected_node_type(cls):
        return Junction

    @classmethod
    def get_component_input(cls):
        return [("name", dtype(object)),
                ("from_junction", "u4"),
                ("to_junction", "u4"),
                ("inner_diameter_mm", "f8"),
                ("qext_w", 'f8'),
                ("loss_coefficient", "f8"),
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
        d_vals = tbl.inner_diameter_mm.values / 1000.

        registry.add_override(PitEntries(*build_pit_entries(
            rows,
            [IdxBranch.QEXT, IdxBranch.D, IdxBranch.DO, IdxBranch.LOSS_COEFFICIENT],
            [tbl.qext_w.values, d_vals, d_vals, tbl.loss_coefficient.values],
        ), mode=PitWriteMode.UNIQUE))

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
        df_dm, df_dp, df_dp1, df_dm_node, residual, residual_fn, residual_tn = (
            calculate_derivatives_hydraulic(net, branch_pit[f:t], node_pit, get_hydraulic_options(net))
        )

        b_pit = branch_pit[f:t]
        fn = b_pit[:, IdxBranch.FROM_NODE].astype(np.int32)
        tn = b_pit[:, IdxBranch.TO_NODE].astype(np.int32)

        register_branch_hydraulic_equation(sys_idx, registry, branch_idx, fn, tn,
                                           df_dm, df_dp, df_dp1, residual)
        register_node_hydraulic_equation(sys_idx, registry, branch_idx, fn, tn, df_dm_node,
                                         -residual_fn, residual_tn)

    @classmethod
    def _register_thermal_equations(cls, net, branch_pit, node_pit, sys_idx, registry):
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

        register_branch_thermal_equation(sys_idx, registry, branch_idx, fn, dfb_dt, dfb_dtout, fb)
        register_node_thermal_equation(sys_idx, registry, branch_idx, tn, dfnt_dt, dfnt_dtout, fnt)

    @classmethod
    def extract_results(cls, net, options, branch_results, mode):
        """Class method to extract pipeflow results from the internal structure into the results table.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :param options: pipeflow options
        :type options: dict
        :param branch_results: important branch results
        :type branch_results: dict
        :param mode: simulation mode
        :type mode: str
        :return: No Output.
        :rtype: None
        """
        required_results_hyd, required_results_ht = standard_branch_wo_internals_result_lookup(net)

        extract_branch_results_without_internals(net, branch_results, required_results_hyd,
                                                 required_results_ht, cls.table_name(), mode)

    @classmethod
    def get_result_table(cls, net):
        if get_fluid(net).is_gas:
            output = ["p_from_bar", "p_to_bar",
                      "t_from_k", "t_to_k", "t_outlet_k", "mdot_from_kg_per_s", "mdot_to_kg_per_s",
                      "vdot_norm_m3_per_s", "normfactor_from",
                      "normfactor_to"]
        else:
            output = ["p_from_bar", "p_to_bar", "t_from_k", "t_to_k", "t_outlet_k",
                      "mdot_from_kg_per_s", "mdot_to_kg_per_s", "vdot_m3_per_s"]
        return output, True
