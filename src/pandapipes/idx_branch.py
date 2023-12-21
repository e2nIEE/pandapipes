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
RHO = 8  # Density in [kg/m^3
ETA = 9  # Dynamic viscosity in [Pas]
K = 10  # Pipe roughness in [m]
VINIT = 11  # velocity in  [m/s]
RE = 12 # Reynolds number
LAMBDA = 13  # Lambda
JAC_DERIV_DV = 14  # Slot for the derivative by velocity
JAC_DERIV_DP = 15  # Slot for the derivative by pressure from_node
JAC_DERIV_DP1 = 16  # Slot for the derivative by pressure to_node
LOAD_VEC_BRANCHES = 17  # Slot for the load vector for the branches
JAC_DERIV_DV_NODE = 18  # Slot for the derivative by velocity for the nodes connected to branch
LOAD_VEC_NODES = 19  # Slot for the load vector of the nodes connected to branch
LOSS_COEFFICIENT = 20
CP = 21  # Slot for fluid heat capacity values
ALPHA = 22  # Slot for heat transfer coefficient
JAC_DERIV_DT = 23
JAC_DERIV_DT1 = 24
LOAD_VEC_BRANCHES_T = 25
TOUTINIT = 26  # Internal slot for outlet pipe temperature
JAC_DERIV_DT_NODE = 27  # Slot for the derivative fpr T for the nodes connected to branch
LOAD_VEC_NODES_T = 28
VINIT_T = 29
FROM_NODE_T = 30
TO_NODE_T = 31
QEXT = 32  # heat input in [W]
TEXT = 33
PL = 34
TL = 35 # Temperature lift [K]
BRANCH_TYPE = 36  # branch type relevant for the pressure controller

branch_cols = 37
