# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np

from pandapipes.component_models.abstract_models.branch_wo_internals_models import BranchWOInternalsComponent
from pandapipes.component_models.component_toolbox import build_pit_entries, standard_branch_wo_internals_result_lookup
from pandapipes.idx_branch import IdxBranch
from pandapipes.idx_node import IdxNode
from pandapipes.pf.pipeflow_setup import get_fluid, get_lookup
from pandapipes.pf.internals_toolbox import get_from_nodes_corrected
from pandapipes.pf.result_extraction import extract_branch_results_without_internals
from pandapipes.pf.system_index import PitEntries, PitWriteMode

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class CirculationPump(BranchWOInternalsComponent):

    # columns for internal array
    TYPE_P = 0

    internal_cols = 1

    @classmethod
    def table_name(cls):
        raise NotImplementedError

    @classmethod
    def active_identifier(cls):
        raise NotImplementedError

    @classmethod
    def from_to_node_cols(cls):
        return "return_junction", "flow_junction"

    @classmethod
    def get_connected_node_type(cls):
        from pandapipes.component_models.junction_component import Junction
        return Junction

    @classmethod
    def get_component_input(cls):
        raise NotImplementedError

    @classmethod
    def create_component_array(cls, net, component_pits):
        tbl = net[cls.table_name()]
        cp_array = np.zeros(shape=(len(tbl), cls.internal_cols), dtype=np.float64)
        cp_array[:, cls.TYPE_P] = np.isin(tbl.type.values, ["p", "pt"])
        component_pits[cls.table_name()] = cp_array

    @classmethod
    def register_pit_node_entries(cls, net, node_pit, registry) -> None:
        tbl = net[cls.table_name()]
        active_mask = tbl[cls.active_identifier()].values
        circ_pump_tbl = tbl[active_mask]
        if not len(circ_pump_tbl):
            return

        _, tn_col = cls.from_to_node_cols()
        junction = circ_pump_tbl[tn_col].values
        types = circ_pump_tbl.type.values
        p_values = circ_pump_tbl.p_flow_bar.values

        junction_lookup = get_lookup(net, "node", "index")[cls.get_connected_node_type().table_name()]
        mask_p = np.isin(types, ["p", "pt"])
        index_p = junction_lookup[junction[mask_p]]

        registry.add_override(PitEntries(*build_pit_entries(
            index_p,
            [IdxNode.PINIT, IdxNode.NODE_TYPE, IdxNode.NODE_TYPE_T],
            [p_values[mask_p], float(IdxNode.P), float(IdxNode.GE)],
        ), mode=PitWriteMode.UNIQUE))

    @classmethod
    def register_pit_branch_entries(cls, net, branch_pit, node_pit, registry) -> None:
        super().register_pit_branch_entries(net, branch_pit, node_pit, registry)

    @classmethod
    def _toutinit_vals(cls, net, to_junctions, junction_table_name):
        tbl = net[cls.table_name()]
        mask_t = np.isin(tbl.type.values, ["pt", "t"])
        return np.where(
            mask_t,
            tbl.t_flow_k.values,
            net[junction_table_name].loc[to_junctions, "tfluid_k"].values,
        )

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
        """Function that extracts certain results.

        :param mode:
        :type mode:
        :param branch_results:
        :type branch_results:
        :param net: The pandapipes network
        :type net: pandapipesNet
        :param options:
        :type options:
        :return: No Output.
        """
        node_pit = net['_pit']['node']
        branch_pit = net['_pit']['branch']
        branch_lookups = get_lookup(net, "branch", "from_to")
        f, t = branch_lookups[cls.table_name()]

        mask = (branch_pit[f:t, IdxBranch.MDOTINIT] < 0) & ~np.isclose(branch_pit[f:t, IdxBranch.MDOTINIT], 0)
        if np.any(mask):
            raise UserWarning(r'Your grid is badly modelled and would lead to a direction change in circulation pump %s'
                              % str(net[cls.table_name()].index[mask].tolist()))

        required_results_hyd, required_results_ht = standard_branch_wo_internals_result_lookup(net)

        extract_branch_results_without_internals(net, branch_results, required_results_hyd, required_results_ht,
                                                 cls.table_name(), mode)

        res_table = net["res_" + cls.table_name()]

        from_nodes = get_from_nodes_corrected(branch_pit[f:t])
        t_from = node_pit[from_nodes, IdxNode.TINIT]
        tout = branch_pit[f:t, IdxBranch.TOUTINIT]
        res_table['deltat_k'].values[:] = t_from - tout

        fluid = get_fluid(net)

        cp_i = fluid.get_heat_capacity(t_from)
        cp_i1 = fluid.get_heat_capacity(tout)
        cp = (cp_i + cp_i1) / 2

        mass = branch_pit[f:t, IdxBranch.MDOTINIT]
        res_table['qext_w'].values[:] = mass * cp * (tout - t_from)
