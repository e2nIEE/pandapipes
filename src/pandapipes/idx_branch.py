# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

# branch types
CIRC = 1  # Circ pump branch
PC = 2  # Pressure controller branch

# branch indices
TABLE_IDX = 0  # number of the table that this branch belongs to
ELEMENT_IDX = 1  # index of the element that this branch belongs to (within the given table)
BRANCH_TYPE = 2  # branch type relevant for the pressure controller
PUMP_TYPE = 3
FROM_NODE = 4  # f, from bus number
TO_NODE = 5  # t, to bus number
ACTIVE = 6
LENGTH = 7  # Pipe length in [m]
D = 8  # Diameter in [m]
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

JAC_DERIV_DM = 23  # Slot for the derivative by mass
JAC_DERIV_DP = 24  # Slot for the derivative by pressure from_node
JAC_DERIV_DP1 = 25  # Slot for the derivative by pressure to_node
JAC_DERIV_DM_NODE = 26  # Slot for the derivative by mass for the nodes connected to branch
LOAD_VEC_BRANCHES = 27  # Slot for the load vector for the branches
LOAD_VEC_NODES_FROM = 28  # Slot for the load vector of the from nodes connected to branch
LOAD_VEC_NODES_TO = 29  # Slot for the load vector of the to nodes connected to branch


JAC_DERIV_DT = 30
JAC_DERIV_DTOUT = 31
JAC_DERIV_DT_NODE = 32  # Slot for the node equation derivative of T for the nodes branch is connected to
JAC_DERIV_DTOUT_NODE = 33  # Slot for the node equation derivative of T for the corresponding branch
LOAD_VEC_BRANCHES_T = 34
LOAD_VEC_NODES_FROM_T = 35 # Slot for the load vector of the from nodes connected to branch
LOAD_VEC_NODES_TO_T = 36 # Slot for the load vector of the to nodes connected to branch

branch_cols = 37
