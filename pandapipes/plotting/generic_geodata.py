# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.


import copy

import pandas as pd
from pandapipes.component_models import ExtGrid, Pipe, Sink, Source, Junction
from pandapower.plotting.generic_geodata import coords_from_igraph

try:
    import igraph

    IGRAPH_INSTALLED = True
except ImportError:
    IGRAPH_INSTALLED = False

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def build_igraph_from_ppipes(net):
    """
    This function uses the igraph library to create an igraph graph for a given pandapipes network.
    Pipes and valves are respected.
    Performance vs. networkx: https://graph-tool.skewed.de/performance

    :param net: The pandapipes network
    :type net: pandapipesNet
    :return: The returned values are:
        - g - The igraph graph
        - meshed - a flag that states whether the graph is meshed
        - roots - the root vertex indices

    :Example:
        graph, meshed, roots = build_igraph_from_pp(net)
    """

    try:
        import igraph
    except (DeprecationWarning, ImportError):
        raise ImportError("Please install python-igraph")
    g = igraph.Graph(directed=True)
    g.add_vertices(net.junction.shape[0])
    g.vs["label"] = net.junction.index.tolist()
    pp_junction_mapping = dict(list(zip(net.junction.index,
                                        list(range(net.junction.index.shape[0])))))

    for lix in net.pipe.index:
        fb, tb = net.pipe.at[lix, "from_junction"], net.pipe.at[lix, "to_junction"]
        g.add_edge(pp_junction_mapping[fb], pp_junction_mapping[tb],
                   weight=net.pipe.at[lix, "length_km"])

    for comp in net['component_list']:
        if comp in [Source, Sink, ExtGrid, Pipe, Junction]:
            continue
        else:
            for comp_data in net[comp.table_name()].itertuples():
                g.add_edge(pp_junction_mapping[comp_data.from_junction],
                           pp_junction_mapping[comp_data.to_junction], weight=0.001)

    meshed = False
    for i in range(1, net.junction.shape[0]):
        if len(g.get_all_shortest_paths(0, i, mode="ALL")) > 1:
            meshed = True
            break

    roots = [pp_junction_mapping[s] for s in net.ext_grid.junction.values]
    return g, meshed, roots  # g, (not g.is_dag())


def create_generic_coordinates(net, mg=None, library="igraph"):
    """
    This function will add arbitrary geo-coordinates for all junctions based on an analysis of
    branches and rings. It will remove out of service junctions/pipes from the net. The coordinates
    will be created either by igraph or by using networkx library.

    :param net: pandapipes network
    :type net: pandapipesNet
    :param mg: Existing networkx multigraph, if available. Convenience to save computation time.
    :type mg: networkx.Graph
    :param library: "igraph" to use igraph package or "networkx" to use networkx package
    :type library: str
    :return: net - pandapipes network with added geo coordinates for the buses

    :Example:
        net = create_generic_coordinates(net)

    ..note:
        The networkx implementation is currently not working, as a proper networkx graph creation\
        is not yet implemented in pandapipes. **Coming soon!**
    """
    if "junction_geodata" in net and net.junction_geodata.shape[0]:
        logger.warning("Please delete all geodata. This function cannot be used with pre-existing "
                       "geodata.")
        return
    if "junction_geodata" not in net or net.junction_geodata is None:
        net.junction_geodata = pd.DataFrame(columns=["x", "y"])

    gnet = copy.deepcopy(net)
    gnet.junction = gnet.junction[gnet.junction.in_service]
    gnet.pipe = gnet.pipe[gnet.pipe.in_service]

    if library == "igraph":
        if not IGRAPH_INSTALLED:
            raise UserWarning("The library igraph is selected for plotting, but not installed "
                              "correctly.")
        graph, meshed, roots = build_igraph_from_ppipes(gnet)
        coords = coords_from_igraph(graph, roots, meshed)
    elif library == "networkx":
        logger.warning("The networkx implementation is not working currently!")
        return
        # if mg is None:
        #     nxg = top.create_nxgraph(gnet)
        # else:
        #     nxg = copy.deepcopy(mg)
        # coords = coords_from_nxgraph(nxg)
    else:
        raise ValueError("Unknown library %s - chose 'igraph' or 'networkx'" % library)

    net.junction_geodata.x = coords[1]
    net.junction_geodata.y = coords[0]
    net.junction_geodata.index = gnet.junction.index
    return net
