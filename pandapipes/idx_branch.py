# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
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
JAC_DERIV_DV = 15  # Slot for the derivative by velocity # df_dv
JAC_DERIV_DP = 16  # Slot for the derivative by pressure from_node # df_dp
JAC_DERIV_DP1 = 17  # Slot for the derivative by pressure to_node # df_dp1
LOAD_VEC_BRANCHES = 18  # Slot for the load vector for the branches : pressure difference between nodes (Bar) #load_vec
JAC_DERIV_DV_NODE = 19  # Slot for the derivative by velocity for the nodes connected to branch #df_dv_nodes
LOAD_VEC_NODES = 20  # Slot for the load vector of the nodes connected to branch : mass_flow (kg_s) # load_vec_nodes
LOSS_COEFFICIENT = 21
CP = 22  # Slot for fluid heat capacity at constant pressure : cp = (J/kg.K)
ALPHA = 23  # Slot for heat transfer coefficient: U = (W/m2.K)
JAC_DERIV_DT = 24 # Slot for the derivative by temperature from_node # df_dt
JAC_DERIV_DT1 = 25 # Slot for the derivative by temperature to_node # df_dt1
LOAD_VEC_BRANCHES_T = 26
T_OUT = 27  # Internal slot for outlet pipe temperature
JAC_DERIV_DT_NODE = 28  # Slot for the derivative for T for the nodes connected to branch
LOAD_VEC_NODES_T = 29
VINIT_T = 30
FROM_NODE_T = 31
TO_NODE_T = 32
QEXT = 33  # heat input in [W]
TEXT = 34
STD_TYPE = 35
PL = 36   # Pressure lift [bar]
TL = 37  # Temperature lift [K]
BRANCH_TYPE = 38  # branch type relevant for the pressure controller
PRESSURE_RATIO = 39  # boost ratio for compressors with proportional pressure lift
T_OUT_OLD = 40
Kv_max = 41  # dynamic valve flow characteristics
DESIRED_MV = 42  # Final Control Element (FCE) Desired Manipulated Value percentage opened
ACTUAL_POS = 43  # Final Control Element (FCE) Actual Position Value percentage opened

branch_cols = 44