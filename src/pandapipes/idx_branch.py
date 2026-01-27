# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

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
AREA = 10  # Area in [m²]
K = 11  # Pipe roughness in [m]
RE = 12 # Reynolds number
LAMBDA = 13  # Lambda
LOSS_COEFFICIENT = 14
ALPHA = 15  # Slot for heat transfer coefficient
QEXT = 16  # heat input into the branch [W]
TEXT = 17 # temperature of surrounding [K]
PL = 18 # Pressure lift [bar]
TL = 19 # Temperature lift [K]

MDOTINIT = 20  # mass in  [m/s]
MDOTINIT_T = 21
FROM_NODE_T_SWITCHED = 22 # flag to indicate if the from and to node are switched in the thermal calculation
TOUTINIT = 23  # Internal slot for outlet pipe temperature
FLOW_RETURN_CONNECT = 24 # Make sure that return and flow side are connected to the central pump, respectively

JAC_DERIV_DM = 25  # Slot for the derivative by mass
JAC_DERIV_DP = 26  # Slot for the derivative by pressure from_node
JAC_DERIV_DP1 = 27  # Slot for the derivative by pressure to_node
JAC_DERIV_DM_NODE = 28  # Slot for the derivative by mass for the nodes connected to branch
LOAD_VEC_BRANCHES = 29  # Slot for the load vector for the branches
LOAD_VEC_NODES_FROM = 30  # Slot for the load vector of the from nodes connected to branch
LOAD_VEC_NODES_TO = 31  # Slot for the load vector of the to nodes connected to branch

JAC_DERIV_DT = 32
JAC_DERIV_DTOUT = 33
JAC_DERIV_DT_NODE = 34  # Slot for the node equation derivative of T for the nodes branch is connected from
JAC_DERIV_DTOUT_NODE = 35  # Slot for the node equation derivative of T for the corresponding branch
LOAD_VEC_BRANCHES_T = 36
LOAD_VEC_NODES_FROM_T = 37 # Slot for the load vector of the from nodes connected to branch
LOAD_VEC_NODES_TO_T = 38 # Slot for the load vector of the to nodes connected to branch

T_OUT_OLD = 39

branch_cols = 40
