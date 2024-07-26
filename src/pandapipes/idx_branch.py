# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

# branch types
# no types defined

# branch indices
TABLE_IDX = 0  # number of the table that this branch belongs to
ELEMENT_IDX = 1  # index of the element that this branch belongs to (within the given table)
FROM_NODE = 2  # f, from bus number
TO_NODE = 3  # t, to bus number
ACTIVE = 4
LENGTH = 5  # Pipe length in [m]
D = 6  # Diameter in [m]
AREA = 7  # Area in [m²]
K = 8  # Pipe roughness in [m]
MDOTINIT = 9  # mass in  [m/s]
RE = 10 # Reynolds number
LAMBDA = 11  # Lambda
JAC_DERIV_DM = 12  # Slot for the derivative by mass
JAC_DERIV_DP = 13  # Slot for the derivative by pressure from_node
JAC_DERIV_DP1 = 14  # Slot for the derivative by pressure to_node
LOAD_VEC_BRANCHES = 15  # Slot for the load vector for the branches
JAC_DERIV_DM_NODE = 16  # Slot for the derivative by mass for the nodes connected to branch
LOAD_VEC_NODES = 17  # Slot for the load vector of the nodes connected to branch
LOSS_COEFFICIENT = 18
ALPHA = 19  # Slot for heat transfer coefficient
JAC_DERIV_DT = 20
JAC_DERIV_DTOUT = 21
LOAD_VEC_BRANCHES_T = 22
TOUTINIT = 23  # Internal slot for outlet pipe temperature
JAC_DERIV_DT_NODE = 24  # Slot for the derivative fpr T for the nodes connected to branch
LOAD_VEC_NODES_T = 25
MDOTINIT_T = 26
FROM_NODE_T = 27
TO_NODE_T = 28
QEXT = 29  # heat input in [W]
TEXT = 30
PL = 31
TL = 32 # Temperature lift [K]
BRANCH_TYPE = 33  # branch type relevant for the pressure controller

branch_cols = 34
