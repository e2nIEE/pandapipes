# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import networkx as nx
import numpy as np
from pandapipes.component_models.abstract_models.branch_models import BranchComponent
from pandapower.topology.create_graph import add_edges, get_edge_table

try:
    import pplog as logging
except ImportError:
    import logging

INDEX = 0
F_JUNCTION = 1
T_JUNCTION = 2

WEIGHT = 0

logger = logging.getLogger(__name__)


def get_col_value(net, table_name, column_name):
    return net[table_name][column_name].values


def create_nxgraph(net, include_pipes=True, respect_status_pipes=True,
                   weighting_pipes=(get_col_value, ("pipe", "length_km")), include_valves=True,
                   respect_status_valves=True, weighting_valves=None, include_pumps=True,
                   respect_status_pumps=True, weighting_pumps=None, respect_status_junctions=True,
                   nogojunctions=None, notravjunctions=None, multi=True,
                   respect_status_branches_all=None, **kwargs):

    if multi:
        mg = nx.MultiGraph()
    else:
        mg = nx.Graph()

    branch_params = {K: v for k, v in kwargs.items() if any(
        k.startswith(par) for par in ["include", "respect_status", "weighting"])}
    loc = locals()
    branch_params.update({"%s_%s" % (par, bc): loc.get("%s_%s" % (par, bc))
                          for par in ["include", "respect_status", "weighting"]
                          for bc in ["pipes", "valves", "pumps"]})
    for comp in net.component_list:
        if not issubclass(comp, BranchComponent):
            continue
        table_name = comp.table_name()
        include_comp = branch_params.get("include_%ss" % table_name, True)
        respect_status = branch_params.get("respect_status_%ss" % table_name, True) \
            if respect_status_branches_all not in [True, False] else respect_status_branches_all
        # some formulation to add weight
        weight_getter = branch_params.get("weighting_%ss" % table_name, None)
        add_branch_component(comp, mg, net, table_name, include_comp, respect_status, weight_getter)

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
    if respect_status_junctions:
        for b in net.junction.index[~net.junction.in_service.values]:
            mg.remove_node(b)

    return mg


def add_branch_component(comp, mg, net, table_name, include_comp, respect_status, weight_getter):
    tab = get_edge_table(net, table_name, include_comp)

    if tab is not None:
        in_service_name = comp.active_identifier()
        from_col, to_col = comp.from_to_node_cols()
        indices, parameter, in_service = init_par(tab, respect_status, in_service_name)
        indices[:, F_JUNCTION] = tab[from_col].values
        indices[:, T_JUNCTION] = tab[to_col].values
        if weight_getter is not None:
            parameter[:, WEIGHT] = weight_getter[0](net, *weight_getter[1])
        add_edges(mg, indices, parameter, in_service, net, table_name)


def init_par(tab, respect_status=True, in_service_col="in_service"):
    n = tab.shape[0]
    indices = np.zeros((n, 3), dtype=np.int)
    indices[:, INDEX] = tab.index
    parameters = np.zeros((n, 1), dtype=np.float)

    if not respect_status:
        return indices, parameters, np.ones(n, dtype=np.bool)
    elif in_service_col in tab:
        return indices, parameters, tab[in_service_col].values.copy()
    else:
        return indices, parameters
