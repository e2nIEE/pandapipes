# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

# define bus types
P = 1  # Reference node, pressure is fixed
L = 2  # All other nodes
T = 10  # Reference node with fixed temperature, otherwise 0
NONE = 3  # None

# define the indices
TABLE_IDX = 0  # number of the table that this node belongs to
ELEMENT_IDX = 1  # index of the element that this node belongs to (within the given table)
NODE_TYPE = 2  # junction type
ACTIVE = 3
RHO = 4  # Density in [kg/m^3]
PINIT = 5
LOAD = 6
HEIGHT = 7
TINIT = 8
PAMB = 9  # Ambient pressure in [bar]
LOAD_T = 10  # Heat power drawn in [W]
NODE_TYPE_T = 11
EXT_GRID_OCCURENCE = 12
EXT_GRID_OCCURENCE_T = 13

node_cols = 14
