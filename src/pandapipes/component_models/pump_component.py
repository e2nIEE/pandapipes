# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from operator import itemgetter

import numpy as np
from numpy import dtype

from pandapipes.component_models import standard_branch_wo_internals_result_lookup
from pandapipes.component_models.abstract_models.branch_wo_internals_models import \
    BranchWOInternalsComponent
from pandapipes.component_models.component_toolbox import (
    build_pit_entries,
    get_component_array,
    get_std_type_lookup,
    get_hydraulic_options,
    get_thermal_options,
    register_branch_node_mass_balance,
    register_branch_node_thermal_balance,
)
from pandapipes.component_models.junction_component import Junction
from pandapipes.constants import NORMAL_TEMPERATURE, NORMAL_PRESSURE, R_UNIVERSAL, P_CONVERSION
from pandapipes.idx_branch import IdxBranch
from pandapipes.idx_node import IdxNode
from pandapipes.pf.derivative_calculation import (
    calculate_derivatives_hydraulic, calculate_derivatives_branch_thermal,
)
from pandapipes.pf.internals_toolbox import get_from_nodes_corrected, get_to_nodes_corrected, branch_area
from pandapipes.pf.pipeflow_setup import get_fluid, get_net_option, get_lookup
from pandapipes.pf.result_extraction import extract_branch_results_without_internals
from pandapipes.pf.system_index import ComponentEquations, HydVarEq, PitEntries, ThermVarEq

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class Pump(BranchWOInternalsComponent):
    """Pump component that lifts pressure according to a characteristic curve."""

    STD_TYPE = 0

    internal_cols = 1

    @classmethod
    def table_name(cls):
        return "pump"

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
    def get_component_input(cls):
        return [("name", dtype(object)),
                ("from_junction", "u4"),
                ("to_junction", "u4"),
                ("std_type", dtype(object)),
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
        d_val = 0.1
        area_val = d_val ** 2 * np.pi / 4
        mdotinit = 0.1 * area_val * get_fluid(net).get_density(NORMAL_TEMPERATURE)
        registry.add(PitEntries(*build_pit_entries(
            rows, [IdxBranch.MDOTINIT], [mdotinit],
        )))

    @classmethod
    def create_component_array(cls, net, component_pits):
        tbl = net[cls.table_name()]
        if len(tbl):
            pump_array = np.zeros(shape=(len(tbl), cls.internal_cols), dtype=np.float64)

            std_types_lookup = get_std_type_lookup(net, cls.table_name())
            std_type, pos = np.where(net[cls.table_name()]['std_type'].values
                                     == std_types_lookup[:, np.newaxis])
            pump_array[pos, cls.STD_TYPE] = std_type
            component_pits[cls.table_name()] = pump_array

    @classmethod
    def register_hydraulic_equations(cls, net, branch_pit, node_pit, sys_idx, registry) -> None:
        f, t = get_lookup(net, "branch", "from_to_active_hydraulics")[cls.table_name()]
        branch_idx = np.arange(f, t, dtype=np.int32)
        if not len(branch_idx):
            return

        b_pit = branch_pit[f:t]
        cls._compute_pl(net, b_pit, node_pit)

        df_dm, df_dp, df_dp1, df_dm_node, load, load_fn, load_tn = (
            calculate_derivatives_hydraulic(net, b_pit, node_pit, get_hydraulic_options(net))
        )

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
    def _compute_pl(cls, net, b_pit, node_pit):
        """Compute pressure lift from pump characteristic and write it into b_pit[:, PL].

        get_component_array(net, cls.table_name()) is filtered by the same active_hydraulics
        mask, over the same table row range, as b_pit itself (see register_hydraulic_equations
        above and get_component_array's own only_active filtering) - so its rows are already
        aligned 1:1 with b_pit's rows without needing any extra index. Indexing it via
        IdxBranch.ELEMENT_IDX (a pandas index *label*) instead of positionally was a bug: that
        array is built positionally (row i = i-th row of net[table_name()]), so a label-based
        index silently breaks as soon as the table's index isn't a contiguous 0..n-1 range (e.g.
        after dropping a row and adding a new one).
        """
        pump_array = get_component_array(net, cls.table_name())
        idx = pump_array[:, cls.STD_TYPE].astype(np.int32)
        std_types = get_std_type_lookup(net, cls.table_name())[idx]

        from_nodes = b_pit[:, IdxBranch.FROM_NODE].astype(np.int32)
        fluid = get_fluid(net)
        area = branch_area(b_pit)
        v_mps = b_pit[:, IdxBranch.MDOTINIT] / area / fluid.get_density(NORMAL_TEMPERATURE)
        if fluid.is_gas:
            p_from = node_pit[from_nodes, IdxNode.PAMB] + node_pit[from_nodes, IdxNode.PINIT]
            t_from = node_pit[from_nodes, IdxNode.TINIT]
            normfactor = (NORMAL_PRESSURE * t_from
                          * fluid.get_compressibility(p_from, t_from)
                          / (p_from * NORMAL_TEMPERATURE))
            v_from = v_mps * normfactor
        else:
            v_from = v_mps

        vol = v_from * area
        if len(std_types):
            fcts = itemgetter(*std_types)(net['std_types']['pump'])
            fcts = [fcts] if not isinstance(fcts, tuple) else fcts
            b_pit[:, IdxBranch.PL] = np.array(list(map(lambda f, v: f.get_pressure(v), fcts, vol)))

    @classmethod
    def register_thermal_equations(cls, net, branch_pit, node_pit, sys_idx, registry) -> None:
        f, t = get_lookup(net, "branch", "from_to_active_heat_transfer")[cls.table_name()]
        branch_idx = np.arange(f, t, dtype=np.int32)
        if not len(branch_idx):
            return
        branch_pit_old = net["_active_old_pit"]["branch"]
        fnt, dfnt_dt, dfnt_dtout, fb, dfb_dt, dfb_dtout = (
            calculate_derivatives_branch_thermal(net, branch_pit[f:t], node_pit,
                                          branch_pit_old[f:t], get_thermal_options(net))
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
        calc_compr_pow = get_net_option(net, 'calc_compression_power')

        if get_fluid(net).is_gas:
            output = ["deltap_bar",
                      "p_from_bar", "p_to_bar",
                      "t_from_k", "t_to_k", "t_outlet_k", "mdot_from_kg_per_s", "mdot_to_kg_per_s",
                      "vdot_norm_m3_per_s", "normfactor_from", "normfactor_to"]
        else:
            output = ["deltap_bar", "p_from_bar", "p_to_bar", "t_from_k",
                      "t_to_k", "t_outlet_k", "mdot_from_kg_per_s", "mdot_to_kg_per_s", "vdot_m3_per_s"]
        if calc_compr_pow:
            output += ["compr_power_mw"]

        return output, True

    @classmethod
    def extract_results(cls, net, options, branch_results, mode):
        required_results_hyd, required_results_ht = standard_branch_wo_internals_result_lookup(net)
        required_results_hyd.extend([("deltap_bar", "pl")])

        extract_branch_results_without_internals(net, branch_results, required_results_hyd,
                                                 required_results_ht, cls.table_name(), mode)

        calc_compr_pow = options['calc_compression_power']
        if calc_compr_pow:
            f, t = get_lookup(net, "branch", "from_to")[cls.table_name()]
            from_nodes = branch_results["from_nodes"][f:t]

            res_table = net["res_" + cls.table_name()]
            if net.fluid.is_gas:
                p_from = branch_results["p_abs_from"][f:t]
                p_to = branch_results["p_abs_to"][f:t]
                t0 = net["_pit"]["node"][from_nodes, IdxNode.TINIT]
                mf_sum_int = branch_results["mf_from"][f:t]
                compr = get_fluid(net).get_compressibility(p_from, t0)
                try:
                    molar_mass = net.fluid.get_molar_mass()
                except UserWarning:
                    logger.error('Molar mass is missing in your fluid. Before you are able to '
                                 'retrieve the compression power make sure that the molar mass is'
                                 ' defined')
                else:
                    r_spec = 1e3 * R_UNIVERSAL / molar_mass
                    cp = net.fluid.get_heat_capacity(t0)
                    cv = cp - r_spec
                    k = cp / cv
                    w_real_isentr = (k / (k - 1)) * r_spec * compr * t0 * \
                                    (np.divide(p_to, p_from) ** ((k - 1) / k) - 1)
                    res_table['compr_power_mw'].values[:] = \
                        w_real_isentr * np.abs(mf_sum_int) / 1e6
            else:
                vf_sum_int = branch_results["vf"][f:t]
                pl = branch_results["pl"][f:t]
                res_table['compr_power_mw'].values[:] = pl * P_CONVERSION * vf_sum_int / 1e6
