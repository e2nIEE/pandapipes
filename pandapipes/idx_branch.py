# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

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
TINIT = 11  # Temperature in [K]
VINIT = 12  # velocity in  [m/s]
RE = 13  # Reynolds number
LAMBDA = 14  # Lambda
JAC_DERIV_DV = 15  # Slot for the derivative by velocity
JAC_DERIV_DP = 16  # Slot for the derivative by pressure from_node
JAC_DERIV_DP1 = 17  # Slot for the derivative by pressure to_node
LOAD_VEC_BRANCHES = 18  # Slot for the load vector for the branches
JAC_DERIV_DV_NODE = 19  # Slot for the derivative by velocity for the nodes connected to branch
LOAD_VEC_NODES = 20  # Slot for the load vector of the nodes connected to branch
LOSS_COEFFICIENT = 21
CP = 22  # Slot for fluid heat capacity values
ALPHA = 23  # Slot for heat transfer coefficient
JAC_DERIV_DT = 24
JAC_DERIV_DT1 = 25
LOAD_VEC_BRANCHES_T = 26
T_OUT = 27  # Internal slot for outlet pipe temperature
JAC_DERIV_DT_NODE = 28  # Slot for the derivative fpr T for the nodes connected to branch
LOAD_VEC_NODES_T = 29
VINIT_T = 30
FROM_NODE_T = 31
TO_NODE_T = 32
QEXT = 33  # heat input in [W]
TEXT = 34
STD_TYPE = 35
PL = 36
TL = 37  # Temperature lift [K]

branch_cols = 38
