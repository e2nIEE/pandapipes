# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from pandapipes.idx import IndexMeta


class IdxNode(metaclass=IndexMeta):
    # node types
    P = 1  # Reference node, pressure is fixed
    L = 2  # All other nodes
    T = 3  # Reference node with fixed temperature, otherwise 0
    PC = 4  # Controlled node with fixed pressure p
    GE = 5

    # node indices
    TABLE_IDX = 0  # number of the table that this node belongs to
    ELEMENT_IDX = 1  # index of the element that this node belongs to (within the given table)
    NODE_TYPE = 2  # junction type
    NODE_TYPE_T = 3
    ACTIVE = 4
    HEIGHT = 5
    PAMB = 6  # Ambient pressure in [bar]
    LOAD = 7
    LOAD_T = 8  # Heat power drawn in [W]
    INFEED = 9

    PINIT = 10
    TINIT = 11
    MDOTSLACKINIT = 12
    # a real ext_grid sits here - MDOTSLACKINIT may absorb residual mass; otherwise (e.g. a
    # circ_pump's own pressure-anchor node) it must be 0
    COUNT_VAR_MASS_SLACK = 13

    node_cols = 14
