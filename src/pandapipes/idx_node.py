# Copyright (c) 2020-2025 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

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
PAMB = 6 # Ambient pressure in [bar]
LOAD = 7
LOAD_T = 8  # Heat power drawn in [W]
EXT_GRID_OCCURENCE = 9
EXT_GRID_OCCURENCE_T = 10
INFEED = 11
VAR_MASS_SLACK = 12 #required as slack do not necesseraly allow mass different from zero

PINIT = 13
MDOTSLACKINIT = 14
TINIT = 15

JAC_DERIV_MSL = 16

JAC_DERIV_DT_SLACK = 17
JAC_DERIV_DT_LOAD = 18

TINIT_OLD = 19

node_cols = 20