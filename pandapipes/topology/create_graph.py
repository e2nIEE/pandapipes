# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import networkx as nx

from pandapower.topology.create_graph import get_edge_table, add_edges, init_par

try:
    import pplog as logging
except ImportError:
    import logging

INDEX = 0
F_JUNCTION = 1
T_JUNCTION = 2

WEIGHT = 0

logger = logging.getLogger(__name__)


def create_nxgraph(net, include_pipes=True, include_valves=True, include_pumps=True,
                   nogojunctions=None, notravjunctions=None, multi=True,
                   include_out_of_service=False):

    if multi:
        mg = nx.MultiGraph()
    else:
        mg = nx.Graph()

    if hasattr(net, "pipe"):
        pipe = get_edge_table(net, "pipe", include_pipes)
        if pipe is not None:
            indices, parameter, in_service = init_par(pipe)
            indices[:, F_JUNCTION] = pipe.from_junction.values
            indices[:, T_JUNCTION] = pipe.to_junction.values
            parameter[:, WEIGHT] = pipe.length_km.values
            add_edges(mg, indices, parameter, in_service, net, "pipe")

    if hasattr(net, "valve"):
        valve = get_edge_table(net, "valve", include_valves)
        if valve is not None:
            indices, parameter, in_service = init_par(valve)
            indices[:, F_JUNCTION] = valve.from_junction.values
            indices[:, T_JUNCTION] = valve.to_junction.values
            add_edges(mg, indices, parameter, in_service, net, "valve")

    if hasattr(net, "pump"):
        pump = get_edge_table(net, "pump", include_pumps)
        if pump is not None:
            indices, parameter, in_service = init_par(pump)
            indices[:, F_JUNCTION] = pump.from_junction.values
            indices[:, T_JUNCTION] = pump.to_junction.values
            add_edges(mg, indices, parameter, in_service, net, "pump")

    # add all junctions that were not added when creating branches
    if len(mg.nodes()) < len(net.junction.index):
        for b in set(net.junction.index) - set(mg.nodes()):
            mg.add_node(b)

    # remove nogojunctions
    if nogojunctions is not None:
        for b in nogojunctions:
            mg.remove_node(b)

    # remove the edges pointing away of notravjunctions
    if notravjunctions is not None:
        for b in notravjunctions:
            for i in list(mg[b].keys()):
                try:
                    del mg[b][i]  # networkx versions < 2.0
                except:
                    del mg._adj[b][i]  # networkx versions 2.0

    # remove out of service junctions
    if not include_out_of_service:
        for b in net.junction.index[~net.junction.in_service.values]:
            mg.remove_node(b)

    return mg