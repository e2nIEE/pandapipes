# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from pandapower.plotting.generic_geodata import coords_from_igraph, \
    _prepare_geodata_table, _get_element_mask_from_nodes, _igraph_meshed

from pandapipes.component_models.abstract_models.branch_models import BranchComponent
from pandapipes.component_models.abstract_models.circulation_pump import CirculationPump

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def build_igraph_from_ppipes(net, junctions=None):
    """
    This function uses the igraph library to create an igraph graph for a given pandapipes network.
    Pipes and valves are respected.
    Performance vs. networkx: https://graph-tool.skewed.de/performance

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param junctions: subset of junctions that should be part of the graph
    :type junctions: list
    :return: The returned values are:
        - g - The igraph graph
        - meshed - a flag that states whether the graph is meshed
        - roots - the root vertex indices

    :Example:
        graph, meshed, roots = build_igraph_from_pp(net)
    """

    try:
        import igraph as ig
    except (DeprecationWarning, ImportError):
        raise ImportError("Please install python-igraph with "
                          "`pip install python-igraph` or "
                          "`conda install python-igraph` "
                          "or from https://www.lfd.uci.edu/~gohlke/pythonlibs")
    g = ig.Graph(directed=True)
    junction_index = net.junction.index if junctions is None else np.array(junctions)
    nr_junctions = len(junction_index)
    g.add_vertices(nr_junctions)
    g.vs["label"] = list(junction_index)
    pp_junction_mapping = dict(list(zip(junction_index, list(range(nr_junctions)))))

    mask = _get_element_mask_from_nodes(net, "pipe", ["from_junction", "to_junction"], junctions)
    for pipe in net.pipe[mask].itertuples():
        g.add_edge(pp_junction_mapping[pipe.from_junction], pp_junction_mapping[pipe.to_junction],
                   weight=pipe.length_km)

    for comp in net['component_list']:
        if not isinstance(comp, BranchComponent):
            continue
        fjc, tjc = comp.from_to_node_cols()
        mask = _get_element_mask_from_nodes(net, comp.table_name(), [fjc, tjc], junctions)
        for comp_data in net[comp.table_name()][mask].itertuples():
            g.add_edge(pp_junction_mapping[comp_data[fjc]], pp_junction_mapping[comp_data[tjc]],
                       weight=0.001)

    meshed = _igraph_meshed(g)
    roots = [pp_junction_mapping[s] for s in net.ext_grid.junction.values if s in junction_index]
    return g, meshed, roots  # g, (not g.is_dag())


def create_generic_coordinates(net, mg=None, library="igraph", geodata_table="junction_geodata",
                               junctions=None, overwrite=False):
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
    _prepare_geodata_table(net, geodata_table, overwrite)

    if library == "igraph":
        graph, meshed, roots = build_igraph_from_ppipes(net, junctions=junctions)
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

    if len(coords):
        net[geodata_table].x = coords[1]
        net[geodata_table].y = coords[0]
        net[geodata_table].index = net.junction.index if junctions is None else junctions
    return net
