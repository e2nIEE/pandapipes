# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

# branch types
PC = 1  # Pressure controller branch

# branch indices
TABLE_IDX = 0  # number of the table that this branch belongs to
ELEMENT_IDX = 1  # index of the element that this branch belongs to (within the given table)
BRANCH_TYPE = 2  # branch type relevant for the pressure controller
FROM_NODE = 3  # f, from bus number
TO_NODE = 4  # t, to bus number
ACTIVE = 5
LENGTH = 6  # Pipe length in [m]
D = 7  # Inner Diameter in [m]
DO = 8  # Outer Diameter in [m]
AREA = 9  # Area in [m²]
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
MDOTINIT_T = 20
FROM_NODE_T_SWITCHED = 21 # flag to indicate if the from and to node are switched in the thermal calculation
TOUTINIT = 22  # Internal slot for outlet pipe temperature
FLOW_RETURN_CONNECT = 23 # Make sure that return and flow side are connected to the central pump, respectively

JAC_DERIV_DM = 24  # Slot for the derivative by mass
JAC_DERIV_DP = 25  # Slot for the derivative by pressure from_node
JAC_DERIV_DP1 = 26  # Slot for the derivative by pressure to_node
JAC_DERIV_DM_NODE = 27  # Slot for the derivative by mass for the nodes connected to branch
LOAD_VEC_BRANCHES = 28  # Slot for the load vector for the branches
LOAD_VEC_NODES_FROM = 29  # Slot for the load vector of the from nodes connected to branch
LOAD_VEC_NODES_TO = 30  # Slot for the load vector of the to nodes connected to branch

JAC_DERIV_DT = 31
JAC_DERIV_DTOUT = 32
JAC_DERIV_DT_NODE = 33  # Slot for the node equation derivative of T for the nodes branch is connected to
JAC_DERIV_DTOUT_NODE = 34  # Slot for the node equation derivative of T for the corresponding branch
LOAD_VEC_BRANCHES_T = 35
LOAD_VEC_NODES_FROM_T = 36 # Slot for the load vector of the from nodes connected to branch
LOAD_VEC_NODES_TO_T = 37 # Slot for the load vector of the to nodes connected to branch

T_OUT_OLD = 38

branch_cols = 39
