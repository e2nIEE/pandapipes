# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import networkx as nx
import numpy as np
from pandapower.topology.create_graph import add_edges, get_edge_table

from pandapipes.component_models.abstract_models.branch_models import BranchComponent

try:
    import pandaplan.core.pplog as logging
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
                   weighting_pipes=(get_col_value, ("pipe", "length_km")),
                   include_valves=True, respect_status_valves=True,
                   weighting_valves=None,
                   include_compressors=True, respect_status_compressors=True,
                   weighting_compressors=None,
                   include_mass_circ_pumps=True, respect_status_mass_circ_pumps=True,
                   weighting_mass_circ_pumps=None,
                   include_pressure_circ_pumps=True, respect_status_pressure_circ_pumps=True,
                   weighting_pressure_circ_pumps=None,
                   include_press_controls=True, respect_status_press_controls=True,
                   weighting_press_controls=None,
                   include_pumps=True, respect_status_pumps=True,
                   weighting_pumps=None,
                   include_flow_controls=True, respect_status_flow_controls=True,
                   weighting_flow_controls=None,
                   respect_status_junctions=True, nogojunctions=None, notravjunctions=None, multi=True,
                   respect_status_branches_all=None, **kwargs):
    """
    Converts a pandapipes network into a NetworkX graph, which is a simplified representation of a
    network's topology, reduced to nodes and edges. Junctions are being represented by nodes, edges
    represent physical connections between junctions (typically pipes or pumps).

    :param net: The pandapipes network to be converted
    :type net: pandapipesNet
    :param include_pipes: Flag whether pipes should be included in the graph, OR: list of pipes to\
        be included explicitly.
    :type include_pipes: bool, iterable, default True
    :param respect_status_pipes: Flag whether the "in_service" column shall be considered and out\
        of service pipes neglected.
    :type respect_status_pipes: bool, default True
    :param weighting_pipes: Function that defines how the weighting of the pipes is defined. \
        Parameter of shape (function, (list of arguments)). If None, weight is set to 0.
    :type weighting_pipes: list, tuple, default (:func:`get_col_value`, ("pipe", "length_km"))
    :param include_valves: Flag whether valves should be included in the graph, OR: list of valves\
        to be included explicitly.
    :type include_valves: bool, iterable, default True
    :param respect_status_valves: Flag whether the "opened" column shall be considered and out\
        of service valves neglected.
    :type respect_status_valves: bool, default True
    :param weighting_valves: Function that defines how the weighting of the valves is defined. \
        Parameter of shape (function, (list of arguments)). If None, weight is set to 0.
    :type weighting_valves: list, tuple, default None
    :param include_pumps: Flag whether pumps should be included in the graph, OR: list of pumps to\
        be included explicitly.
    :type include_pumps: bool, iterable, default True
    :param respect_status_pumps: Flag whether the "in_service" column shall be considered and out\
        of service pumps neglected.
    :type respect_status_pumps: bool, default True
    :param weighting_pumps: Function that defines how the weighting of the pumps is defined. \
        Parameter of shape (function, (list of arguments)). If None, weight is set to 0.
    :type weighting_pumps: list, tuple, default None
    :param respect_status_junctions: Flag whether the "in_service" column shall be considered and\
        out of service junctions neglected.
    :type respect_status_junctions: bool, default True
    :param nogojunctions: nogojunctions are not being considered in the graph
    :type nogojunctions: iterable, default None
    :param notravjunctions: edges connected to these junctions are not being considered in the graph
    :type notravjunctions: iterable, default None
    :param multi: True: The function generates a NetworkX MultiGraph, which allows multiple \
        parallel edges between nodes
        False: NetworkX Graph (no multiple parallel edges)
    :type multi: bool, default True
    :param respect_status_branches_all: Flag for overriding the status consideration for all branch\
        elements (pipes, valves, pumps etc.). If None, will not be considered.
    :type respect_status_branches_all: bool, default None
    :param kwargs: Additional keyword arguments, especially useful to address inclusion of branch\
        components that are not in the default components (pipes, valves, pumps). It is always \
        possible to add "include_xy", "respect_status_xy" or "weighting_xy" arguments for \
        additional components
    :return: mg - the required NetworkX graph

    ..note: By default, all branch components are represented as edges in the graph. I.e. tables of\
            every branch component will be included if not stated otherwise. The status\
            (in_service) will always be considered, unless stated explicitly. The weighting by \
            default is 0, but can be changed with the help of a function, as it is done with pipes.
    """
    if multi:
        mg = nx.MultiGraph()
    else:
        mg = nx.Graph()

    branch_params = {k: v for k, v in kwargs.items() if any(
        k.startswith(par) for par in ["include", "respect_status", "weighting"])}
    loc = locals()
    branch_params.update({"%s_%s" % (par, bc): loc.get("%s_%s" % (par, bc))
                          for par in ["include", "respect_status", "weighting"]
                          for bc in ["pipes", "valves", "pumps", "press_controls",
                                     "mass_circ_pumps", "press_circ_pumps", "valve_pipes"]})
    for comp in net.component_list:
        if not issubclass(comp, BranchComponent):
            continue
        table_name = comp.table_name()
        include_kw = "%ss" % table_name
        if table_name.startswith("circ_pump"):
            include_kw = table_name.split("circ_pump")[-1][1:] + "_circ_pumps"
        include_comp = branch_params.get("include_%s" % include_kw, True)
        respect_status = branch_params.get("respect_status_%s" % include_kw, True) \
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
    indices = np.zeros((n, 3), dtype=int)
    indices[:, INDEX] = tab.index
    parameters = np.zeros((n, 1), dtype=float)

    if not respect_status:
        return indices, parameters, np.ones(n, dtype=bool)
    elif in_service_col in tab:
        return indices, parameters, tab[in_service_col].values.copy()
    else:
        return indices, parameters
