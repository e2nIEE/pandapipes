# Copyright (c) 2020-2021 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.


def idx_node_element(net):
    idx_node_element = dict()
    idx_node_element['P'] = 1  # Reference node, pressure is fixed
    idx_node_element['T'] = 2   # Reference node, temperature is fixed
    idx_node_element['TABLE_IDX'] = 0  # number of the table that this node belongs to
    idx_node_element['ELEMENT_IDX'] = 1  # index of the element that this node belongs to (within the given table)
    idx_node_element['NODE_ELEMENT_TYPE'] = 2
    idx_node_element['JUNCTION'] = 3
    idx_node_element['MINIT'] = 4
    idx_node_element['ACTIVE'] = 5
    counter = 6

    for key in net._fluid:
        counter += 1
        idx_node_element[key + '_W'] = counter

    node_element_cols = counter + 1

    idx_node_element['node_element_cols'] = node_element_cols

    net['_idx_node_element'] = idx_node_element
