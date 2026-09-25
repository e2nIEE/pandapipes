# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype

from pandapipes.component_models.abstract_models.node_element_models import NodeElementComponent
from pandapipes.component_models.component_toolbox import build_pit_entries, get_component_array
from pandapipes.pf.pipeflow_setup import get_lookup
from pandapipes.idx_node import IdxNode
from pandapipes.pf.system_index import ComponentEquations, EqWriteMode, PitEntries, PitWriteMode, HydVarEq, ThermVarEq

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class ExtGrid(NodeElementComponent):
    """External grid component acting as the network's slack node for pressure and temperature."""

    # columns for internal array
    JUNCTION = 0
    TYPE_P = 1
    TYPE_T = 2
    P_BAR = 3
    T_K = 4
    IN_SERVICE = 5

    internal_cols = 6

    @classmethod
    def table_name(cls):
        return "ext_grid"

    @classmethod
    def active_identifier(cls):
        return "in_service"

    @classmethod
    def sign(cls):
        return -1.

    @classmethod
    def get_connected_node_type(cls):
        from pandapipes.component_models.junction_component import Junction
        return Junction

    @classmethod
    def get_connected_junction(cls, net):
        junction = net[cls.table_name()].junction
        return junction

    @classmethod
    def get_node_col(cls):
        return "junction"

    @classmethod
    def get_component_input(cls):
        """Get component input.

        :return:
        :rtype:
        """
        return [("name", dtype(object)),
                ("junction", "u4"),
                ("p_bar", "f8"),
                ("t_k", "f8"),
                ("in_service", "bool"),
                ('type', dtype(object))]

    @classmethod
    def create_component_array(cls, net, component_pits):
        tbl = net[cls.table_name()]
        eg_array = np.zeros(shape=(len(tbl), cls.internal_cols), dtype=np.float64)
        eg_array[:, cls.JUNCTION] = tbl.junction.values
        eg_array[:, cls.TYPE_P] = np.isin(tbl.type.values, ["p", "pt"])
        eg_array[:, cls.TYPE_T] = np.isin(tbl.type.values, ["t", "pt"])
        eg_array[:, cls.P_BAR] = tbl.p_bar.values
        eg_array[:, cls.T_K] = tbl.t_k.values
        eg_array[:, cls.IN_SERVICE] = tbl.in_service.values
        component_pits[cls.table_name()] = eg_array

    @classmethod
    def register_pit_node_entries(cls, net, node_pit, registry) -> None:
        ext_grids = net[cls.table_name()]
        ext_grids = ext_grids[ext_grids[cls.active_identifier()].values]
        if not len(ext_grids):
            return

        junction = ext_grids[cls.get_node_col()].values
        types = ext_grids.type.values
        junction_lookup = get_lookup(net, "node", "index")[cls.get_connected_node_type().table_name()]
        mask_p = np.isin(types, ["p", "pt"])
        mask_t = np.isin(types, ["t", "pt"])
        index_p = junction_lookup[junction[mask_p]]
        index_t = junction_lookup[junction[mask_t]]

        registry.add_override(PitEntries(*build_pit_entries(
            index_p,
            [IdxNode.PINIT, IdxNode.NODE_TYPE],
            [ext_grids.p_bar.values[mask_p], float(IdxNode.P)],
        ), mode=PitWriteMode.MEAN))
        registry.add_override(PitEntries(*build_pit_entries(
            index_t,
            [IdxNode.TINIT, IdxNode.NODE_TYPE_T],
            [ext_grids.t_k.values[mask_t], float(IdxNode.T)],
        ), mode=PitWriteMode.MEAN))
        # COUNT_VAR_MASS_SLACK is a "does a genuine mass slack exist at this node at all" marker, not an
        # exclusively-owned value - UNIQUE would make it impossible for any other slack-capable
        # component to ever ALSO mark the same node (a hard conflict error, even though both sides
        # would agree on the same value). ADDITIVE lets any number of contributors coexist; the
        # only reader (register_circ_pump_slack_equations) checks "== 0" / "!= 0", so an
        # accumulated value like 2. from two co-located ext_grids is still read correctly as
        # "yes, a real slack is here" - there's no need to clamp/OR it down to exactly 1.
        registry.add(PitEntries(*build_pit_entries(
            index_p,
            [IdxNode.COUNT_VAR_MASS_SLACK],
            [1.]),
            mode=PitWriteMode.ADDITIVE))

    @classmethod
    def register_hydraulic_equations(cls, net, branch_pit, node_pit, sys_idx, registry):
        # register only for nodes that actually have an active ext_grid row - NOT every P-type
        # node in the system (a circ_pump also marks its own flow junction as NODE_TYPE=P purely
        # to anchor a pressure reference; that node is none of ExtGrid's business - it's handled
        # by register_circ_pump_slack_equations instead, using the COUNT_VAR_MASS_SLACK flag
        # written below to know whether a real ext_grid also sits there). ext_grid ALWAYS
        # provides genuine mass-slack capability - no COUNT_VAR_MASS_SLACK check needed on this side.
        eg_array = get_component_array(net, cls.table_name(), only_active=False)
        if not len(eg_array):
            return
        p_mask = eg_array[:, cls.IN_SERVICE].astype(bool) & eg_array[:, cls.TYPE_P].astype(bool)
        if not np.any(p_mask):
            return

        # "index_active_hydraulics" (not the plain "index" lookup!) maps onto the ACTIVE/reduced
        # pit register_hydraulic_equations operates on here - the plain lookup is for the full
        # pit, used by register_pit_node_entries before reduction; using it here would index into
        # the wrong (larger) array and either crash or silently hit the wrong node. -1 means
        # disconnected (dropped from the active pit) - skip those, same as register_thermal_equations.
        junction_lookup = get_lookup(net, "node", "index_active_hydraulics")[
            cls.get_connected_node_type().table_name()]
        # one entry per ext_grid ROW - deliberately NOT deduplicated by node (see below: multiple
        # ext_grids at the same node each contribute their own additive share to MDOTSLACKINIT)
        eg_nodes = junction_lookup[eg_array[p_mask, cls.JUNCTION].astype(np.int32)]
        eg_nodes = eg_nodes[eg_nodes != -1]
        if not len(eg_nodes):
            return

        # variables - MDOTSLACKINIT/SLACK are indexed by raw node index too (like PINIT/NODE),
        # no rank-within-slack_nodes translation needed (see HydraulicSystemIndex)
        p_col     = sys_idx.idx(HydVarEq.PINIT,         eg_nodes)
        slack_col = sys_idx.idx(HydVarEq.MDOTSLACKINIT, eg_nodes)

        # equation position slack
        slack_eq = sys_idx.idx(HydVarEq.SLACK, eg_nodes)

        # system matrix slack: pressure fix — δPINIT = 0. Registered once per ext_grid ROW (not
        # deduplicated by node) with MEAN: several ext_grids at the same junction all target the
        # same row, and MEAN lets them coexist there peacefully (also with a circ_pump's own
        # pressure fix, if co-located) instead of UNIQUE's exclusive-ownership conflict check.
        rows_slack = slack_eq.astype(np.int32)
        cols_slack = p_col.astype(np.int32)
        data_slack = np.ones(len(slack_eq), dtype=np.float64)
        load_rows_slack = slack_eq.astype(np.int32)
        load_slack = np.zeros(len(slack_eq), dtype=np.float64)

        # equation position node
        n_eq = sys_idx.idx(HydVarEq.NODE, eg_nodes)

        # system matrix node: MDOTSLACKINIT participates in mass balance - free to absorb
        # whatever residual the rest of the network leaves over, exactly the point of a real
        # ext_grid (unlike a circ_pump's own anchor node, see CirculationPump). Also registered
        # once per ext_grid ROW (not deduplicated): N ext_grids at the same node each add their
        # own +1 coefficient to that SAME row, so the row's total coefficient becomes N and
        # Newton solves directly for MDOTSLACKINIT = (whatever the rest of the network leaves
        # over) / N - each ext_grid's own share, with no separate averaging step needed in
        # extract_results.
        rows_node = n_eq.astype(np.int32)
        cols_node = slack_col.astype(np.int32)
        data_node = np.ones(len(n_eq), dtype=np.float64)
        load_rows_node = n_eq.astype(np.int32)
        load_node = node_pit[eg_nodes, IdxNode.MDOTSLACKINIT].astype(np.float64)

        registry.add(ComponentEquations(
            rows=rows_slack,
            cols=cols_slack,
            data=data_slack,
            load_rows=load_rows_slack,
            load_data=load_slack,
            mode=EqWriteMode.MEAN,
        ))

        registry.add(ComponentEquations(
            rows=rows_node,
            cols=cols_node,
            data=data_node,
            load_rows=load_rows_node,
            load_data=load_node,
        ))

    @classmethod
    def register_thermal_equations(cls, net, branch_pit, node_pit, sys_idx, registry):
        eg_array = get_component_array(net, cls.table_name(), only_active=False)
        if not len(eg_array):
            return
        t_mask = eg_array[:, cls.IN_SERVICE].astype(bool) & eg_array[:, cls.TYPE_T].astype(bool)
        if not np.any(t_mask):
            return

        junction_lookup = get_lookup(net, "node", "index_active_heat_transfer")[
            cls.get_connected_node_type().table_name()
        ]
        ext_nodes = junction_lookup[eg_array[t_mask, cls.JUNCTION].astype(np.int32)]
        ext_nodes = ext_nodes[ext_nodes != -1]  # drop disconnected, sort to match infeed_nodes order

        if not len(ext_nodes):
            return

        infeed_mask = node_pit[:, IdxNode.INFEED].astype(bool)
        infeed_nodes = np.where(infeed_mask)[0].astype(np.int32)

        if not len(infeed_nodes):
            return

        # variables
        t_col = sys_idx.idx(ThermVarEq.TINIT, ext_nodes)

        # equation position node
        n_eq = sys_idx.idx(ThermVarEq.NODE, infeed_nodes)

        # system matrix node
        rows_node = n_eq.astype(np.int32)
        cols_node = t_col.astype(np.int32)
        data_node = np.ones(len(n_eq), dtype=np.float64)
        load_rows_node = n_eq.astype(np.int32)
        load_node = np.zeros(len(n_eq), dtype=np.float64)

        registry.add_override(ComponentEquations(
            rows=rows_node,
            cols=cols_node,
            data=data_node,
            load_rows=load_rows_node,
            load_data=load_node,
            mode=EqWriteMode.MEAN,
        ))

    @classmethod
    def get_result_table(cls, net):
        """Get the result table columns.

        :param net: The pandapipes network
        :type net: pandapipesNet
        :return: (columns, all_float) - the column names and whether they are all float type. Only
                if False, returns columns as tuples also specifying the dtypes
        :rtype: (list, bool)
        """
        return ["mdot_kg_per_s"], True

    @classmethod
    def extract_results(cls, net, options, branch_results, mode):
        """Function that extracts certain results.

        :param branch_results:
        :type branch_results:
        :param net: The pandapipes network
        :type net: pandapipesNet
        :param options:
        :type options:
        :param mode:
        :type mode:
        :return: No Output.
        """
        ext_grids = net[cls.table_name()]

        if len(ext_grids) == 0:
            return

        res_table = net["res_" + cls.table_name()]

        node_pit = net["_pit"]["node"]

        p_grids = np.isin(ext_grids.type.values, ["p", "pt"]) & ext_grids.in_service.values
        junction = cls.get_connected_junction(net).values
        # get indices in internal structure for junctions in ext_grid tables which are "active"
        eg_nodes = get_lookup(net, "node", "index")[cls.get_connected_node_type().table_name()][
            junction[p_grids]]

        # positive results mean that the ext_grid feeds in, negative means that the ext grid
        # extracts (like a load). MDOTSLACKINIT already IS this ext_grid's own share (see
        # register_hydraulic_equations: N co-located ext_grids each add their own +1 coefficient
        # to the same row, so Newton solves directly for the per-instance value) - no separate
        # averaging needed here.
        res_table["mdot_kg_per_s"].values[p_grids] = cls.sign() * node_pit[eg_nodes, IdxNode.MDOTSLACKINIT]
