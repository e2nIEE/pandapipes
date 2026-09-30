# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from numpy import dtype

from pandapipes.component_models.component_toolbox import get_component_array
from pandapipes.component_models.junction_component import Junction
from pandapipes.component_models.pump_component import Pump
from pandapipes.idx_branch import IdxBranch
from pandapipes.idx_node import IdxNode


class Compressor(Pump):
    """Compressor component that lifts pressure by a fixed pressure ratio."""

    PRESSURE_RATIO = 0

    internal_cols = 1

    @classmethod
    def table_name(cls):
        return "compressor"

    @classmethod
    def get_connected_node_type(cls):
        return Junction

    @classmethod
    def get_component_input(cls):
        return [("name", dtype(object)),
                ("from_junction", "u4"),
                ("to_junction", "u4"),
                ("pressure_ratio", "f8"),
                ("in_service", 'bool')]

    @classmethod
    def create_component_array(cls, net, component_pits):
        tbl = net[cls.table_name()]
        compr_array = np.zeros(shape=(len(tbl), cls.internal_cols), dtype=np.float64)
        compr_array[:, cls.PRESSURE_RATIO] = net[cls.table_name()].pressure_ratio.values
        component_pits[cls.table_name()] = compr_array

    @classmethod
    def _compute_pl(cls, net, b_pit, node_pit):
        """Compute pressure lift from pressure_ratio and write into b_pit[:, PL].

        See Pump._compute_pl's docstring: compr_array is already row-aligned with b_pit through
        get_component_array's own active_hydraulics filtering, so no separate index is needed.
        """
        compr_array = get_component_array(net, cls.table_name())
        pressure_ratio = compr_array[:, cls.PRESSURE_RATIO]
        from_nodes = b_pit[:, IdxBranch.FROM_NODE].astype(np.int32)
        p_from = node_pit[from_nodes, IdxNode.PAMB] + node_pit[from_nodes, IdxNode.PINIT]
        pl_abs = p_from * pressure_ratio - p_from
        pl_abs[b_pit[:, IdxBranch.MDOTINIT] < 0] = 0.0
        b_pit[:, IdxBranch.PL] = pl_abs
