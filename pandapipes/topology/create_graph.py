# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import networkx as nx
import numpy as np

from pandapower.topology.create_graph import add_edges, get_edge_table

from pandapipes.component_models import Pipe, Valve, Pump

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
                   nogojunctions=None, notravjunctions=None, multi=True, neglect_in_service=False):

    if multi:
        mg = nx.MultiGraph()
    else:
        mg = nx.Graph()

    if neglect_in_service is False:
        neglect_in_service_comps = []
    elif neglect_in_service is True:
        neglect_in_service_comps = [comp.table_name() for comp in net.component_list]
    else:
        neglect_in_service_comps = list(neglect_in_service)

    for comp, weight_col, include_comp in zip([Pipe, Valve, Pump], ["length_km", None, None],
                                              [include_pipes, include_valves, include_pumps]):
        if comp in net.component_list:
            add_branch_component(comp, mg, net, include_comp, neglect_in_service_comps, weight_col)

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
    if "junction" not in neglect_in_service_comps:
        for b in net.junction.index[~net.junction.in_service.values]:
            mg.remove_node(b)

    return mg


def add_branch_component(comp, mg, net, include_comp, neglect_in_service_comps, weight_col):
    table_name = comp.table_name()
    tab = get_edge_table(net, table_name, include_comp)

    if tab is not None:
        in_service_name = comp.active_identifier()
        from_col, to_col = comp.from_to_node_cols()
        indices, parameter, in_service = init_par(tab, table_name in neglect_in_service_comps,
                                                  in_service_name)
        indices[:, F_JUNCTION] = tab[from_col].values
        indices[:, T_JUNCTION] = tab[to_col].values
        if weight_col is not None:
            parameter[:, WEIGHT] = net[table_name][weight_col].values
        add_edges(mg, indices, parameter, in_service, net, table_name)


def init_par(tab, neglect_in_service=False, in_service_col="in_service"):
    n = tab.shape[0]
    indices = np.zeros((n, 3), dtype=np.int)
    indices[:, INDEX] = tab.index
    parameters = np.zeros((n, 1), dtype=np.float)

    if neglect_in_service:
        return indices, parameters, np.ones(n, dtype=np.bool)
    elif in_service_col in tab:
        return indices, parameters, tab[in_service_col].values.copy()
    else:
        return indices, parameters
