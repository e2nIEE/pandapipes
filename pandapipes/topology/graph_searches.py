# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import networkx as nx
import pandas as pd
from pandapipes.topology.create_graph import create_nxgraph


def calc_distance_to_junction(net, junction, notravjunctions=None, nogojunctions=None,
                              weight="weight"):
    """
    Calculates the shortest distance between a source junction and all junctions connected to it.

     INPUT:
        **net** (pandapipesNet) - Variable that contains a pandapipes network.

        **junction** (integer) - Index of the source junction.


     OPTIONAL:
        **nogojunctions** (integer/list, None) - nogojunctions are not being considered

        **notravjunctions** (integer/list, None) - lines connected to these junctions are not being
                                              considered
        **weight** (string, None) – Edge data key corresponding to the edge weight

     OUTPUT:
        **dist** - Returns a pandas series with containing all distances to the source junction
                   in km. If weight=None dist is the topological distance (int).

     EXAMPLE:
         import pandapipes.topology as top

         dist = top.calc_distance_to_junction(net, 5)

    """
    g = create_nxgraph(net, nogojunctions=nogojunctions,
                       notravjunctions=notravjunctions)
    dist = nx.single_source_dijkstra_path_length(g, junction, weight=weight)
    return pd.Series(dist)


def calc_minimum_distance_to_junctions(net, junctions, notravjunctions=None, nogojunctions=None,
                                       weight="weight"):
    """
    Calculates the shortest distance between multiple source junctions and all junctions connected \
    to it.

     INPUT:
        **net** (pandapipesNet) - Variable that contains a pandapipes network.

        **junction** (integer) - Index of the source junction.


     OPTIONAL:
        **nogojunctions** (integer/list, None) - nogojunctions are not being considered

        **notravjunctions** (integer/list, None) - lines connected to these junctions are not being
                                              considered
        **weight** (string, None) – Edge data key corresponding to the edge weight

     OUTPUT:
        **dist** - Returns a pandas series with containing all distances to the source junction
                   in km. If weight=None dist is the topological distance (int).

     EXAMPLE:
         import pandapipes.topology as top

         dist = top.calc_distance_to_junction(net, 5)

    """
    mg = create_nxgraph(net, notravjunctions=notravjunctions,
                        nogojunctions=nogojunctions, weight=weight)
    junctions = set(junctions)
    junction = junctions.pop()
    mg.add_edges_from([(junction, y, {"weight": 0}) for y in junctions])
    return pd.Series(nx.single_source_dijkstra_path_length(mg, junction))


if __name__ == '__main__':
    import pandapipes.networks as nw
    net = nw.gas_meshed_delta()
    dist = calc_minimum_distance_to_junctions(net, net.ext_grid.junction.values)
