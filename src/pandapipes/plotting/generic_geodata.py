# Copyright (c) 2020-2025 by Fraunhofer Institute for Energy Economics
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


def build_igraph_from_ppipes(net, junctions=None, weight_column_lookup="length_km",
                             edge_factories_override=None, additional_edge_factories=None,
                             exclude_branch_elements=(), ignore_in_service_branch_elements=()):
    """
    This function uses the igraph library to create an igraph graph for a given pandapipes network.
    Any branch component is respected.
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
        raise ImportError("Please install igraph with "
                          "`pip install igraph` or "
                          "`conda install igraph` "
                          "or from https://www.lfd.uci.edu/~gohlke/pythonlibs")
    g = ig.Graph(directed=True)
    junction_index = net.junction.index.to_numpy() if junctions is None else np.array(junctions)
    nr_junctions = len(junction_index)
    g.add_vertices(nr_junctions)
    g.vs["label"] = list(junction_index)
    pp_junction_mapping = dict(list(zip(junction_index, list(range(nr_junctions)))))

    for comp in net['component_list']:
        tbl = comp.table_name()
        if tbl in exclude_branch_elements:
            continue
        if edge_factories_override is not None and tbl in edge_factories_override:
            edge_factories_override[tbl](net, g, pp_junction_mapping, junction_index)
            continue
        if not issubclass(comp, BranchComponent) or net[tbl].empty:
            continue
        fjc, tjc = comp.from_to_node_cols()
        mask = _get_element_mask_from_nodes(net, tbl, [fjc, tjc], junctions)
        if tbl not in ignore_in_service_branch_elements:
            active_col = comp.active_identifier()
            mask &= net[tbl][active_col].to_numpy()
        fj = net[tbl][fjc].to_numpy()[mask]
        tj = net[tbl][tjc].to_numpy()[mask]
        if len(fj) == 0:
            continue
        if isinstance(weight_column_lookup, dict):
            weight_col = weight_column_lookup.get(tbl, "length_km")
        else:
            weight_col = weight_column_lookup
        if weight_col in net[tbl].columns:
            weights = net[tbl][weight_col].to_numpy()[mask]
        else:
            weights = np.ones(len(fj)) * 0.001

        g.add_edges([[pp_junction_mapping[fj], pp_junction_mapping[tj]] for (fj, tj) in zip(fj, tj)],
                    {"weight": weights})

    if additional_edge_factories is not None:
        for tbl, edge_factory in additional_edge_factories.items():
            edge_factory(net, g, pp_junction_mapping, junction_index)

    meshed = _igraph_meshed(g)
    roots = [pp_junction_mapping[s] for s in net.ext_grid.junction.values if s in junction_index]
    return g, meshed, roots  # g, (not g.is_dag())


def create_generic_coordinates(net, mg=None, library="igraph", geodata_table="junction_geodata",
                               junctions=None, overwrite=False, **kwargs):
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
        graph, meshed, roots = build_igraph_from_ppipes(net, junctions=junctions, **kwargs)
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
        net[geodata_table].loc[:, 'x'] = coords[1]
        net[geodata_table].loc[:, 'y'] = coords[0]
        net[geodata_table].index = net.junction.index if junctions is None else junctions
    return net
