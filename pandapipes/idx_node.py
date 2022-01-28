# Copyright (c) 2020-2021 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.



def idx_node(net):
    idx_node = dict()
    # define bus types
    idx_node['P'] = 1  # Reference node, pressure is fixed
    idx_node['L'] = 2  # All other nodes
    idx_node['T'] = 10  # Reference node with fixed temperature, otherwise 0
    idx_node['PC'] = 20  # Controlled node with fixed pressure p
    idx_node['NONE'] = 3  # None

    # define the indices
    idx_node['TABLE_IDX'] = 0  # number of the table that this node belongs to
    idx_node['ELEMENT_IDX'] = 1  # index of the element that this node belongs to (within the given table)
    idx_node['NODE_TYPE'] = 2  # junction type
    idx_node['ACTIVE'] = 3
    idx_node['PINIT'] = 4
    idx_node['HEIGHT'] = 5
    idx_node['TINIT'] = 6
    idx_node['PAMB'] = 7  # Ambient pressure in [bar]
    idx_node['LOAD_T'] = 8  # Heat power drawn in [W]
    idx_node['NODE_TYPE_T'] = 9
    idx_node['EXT_GRID_OCCURENCE'] = 10
    idx_node['EXT_GRID_OCCURENCE_T'] = 11
    idx_node['FLUID'] = 12

    idx_node['RHO'] = 13  # Density in [kg/m^3]
    idx_node['LOAD'] = 14
    idx_node['SLACK'] = 15
    counter = 15

    for key in net._fluid:
        counter += 1
        idx_node['RHO_' + key] = counter
        counter += 1
        idx_node['LOAD__' + key] = counter
        counter += 1
        idx_node['W_' + key] = counter

    idx_node['node_cols'] = counter + 1
    net['_idx_node'] = idx_node