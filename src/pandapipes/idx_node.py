# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

# node types
P = 1  # Reference node, pressure is fixed
L = 2  # All other nodes
T = 10  # Reference node with fixed temperature, otherwise 0
PC = 20  # Controlled node with fixed pressure p
NONE = 3  # None

# node indices
TABLE_IDX = 0  # number of the table that this node belongs to
ELEMENT_IDX = 1  # index of the element that this node belongs to (within the given table)
NODE_TYPE = 2  # junction type
ACTIVE = 3
PINIT = 4
LOAD = 5
HEIGHT = 6
TINIT = 7
PAMB = 8 # Ambient pressure in [bar]
LOAD_T = 9  # Heat power drawn in [W]
NODE_TYPE_T = 10
EXT_GRID_OCCURENCE = 11
EXT_GRID_OCCURENCE_T = 12

node_cols = 13
