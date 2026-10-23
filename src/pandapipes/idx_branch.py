# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from pandapipes.idx import IndexMeta


class IdxBranch(metaclass=IndexMeta):
    # branch types
    PC = 1  # Pressure controller branch

    # branch indices
    TABLE_IDX = 0  # number of the table that this branch belongs to
    ELEMENT_IDX = 1  # index of the element that this branch belongs to (within the given table)
    BRANCH_TYPE = 2  # branch type relevant for the pressure controller
    DIRECTED = 3
    FROM_NODE = 4  # f, from bus number
    TO_NODE = 5  # t, to bus number
    ACTIVE = 6
    LENGTH = 7  # Pipe length in [m]
    D = 8  # Diameter in [m]
    DO = 9 # Outer Diameter in [m]
    K = 10  # Pipe roughness in [m]
    RE = 11 # Reynolds number
    LAMBDA = 12  # Lambda
    LOSS_COEFFICIENT = 13
    ALPHA = 14  # Slot for heat transfer coefficient
    QEXT = 15  # heat input into the branch [W]
    TEXT = 16 # temperature of surrounding [K]
    PL = 17 # Pressure lift [bar]
    TL = 18 # Temperature lift [K]

    MDOTINIT = 19  # mass in  [m/s]
    TOUTINIT = 20  # Internal slot for outlet pipe temperature

    FROM_NODE_T_SWITCHED = 21 # flag to indicate if the from and to node are switched in the thermal calculation
    FLOW_RETURN_CONNECT = 22 # Make sure that return and flow side are connected to the central pump, respectively

    DP_FRICT_LOSS = 23

    branch_cols = 24
