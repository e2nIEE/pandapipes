# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.



import matplotlib.pyplot as plt

from pandapipes.plotting.plotting_toolbox import get_collection_sizes, get_collection_sizes_pps
from pandapipes.plotting.collections import create_junction_collection, create_pipe_collection, \
     create_valve_collection, create_source_collection, \
     create_heat_exchanger_collection, create_sink_collection, create_pump_collection, create_house_collection, create_hp_collection
from pandapipes.plotting.generic_geodata import create_generic_coordinates
from pandapower.plotting import draw_collections
from itertools import chain
import numpy as np
from pandapower.plotting.collections import create_bus_collection, create_line_collection, create_trafo_collection, create_sgen_collection, create_ext_grid_collection
import types
import pandapower.non_git.house_control as hctrl
import pandapower.non_git.heat_pump as pumpctrl


try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def simple_plot(net, respect_valves=False, pipe_width=2.0, junction_size=1.0, ext_grid_size=1.0,
                plot_sinks=False, plot_sources=False, sink_size=1.0, source_size=1.0,
                valve_size=1.0, pump_size=1.0, heat_exchanger_size=1.0, scale_size=True, junction_color="r",
                pipe_color='silver', ext_grid_color='orange', valve_color='silver',
                pump_color='silver', heat_exchanger_color='silver', library="igraph", show_plot=True,
                ax=None, **kwargs):  # pragma: no cover
    """
    Plots a pandapipes network as simple as possible. If no geodata is available, artificial
    geodata is generated. For advanced plotting see the tutorial

    :param net: The pandapipes format network.
    :type net: pandapipesNet
    :param respect_valves: Respect valves if artificial geodata is created. \
            .. note:: This Flag is ignored if plot_line_switches is True
    :type respect_valves: bool default False
    :param pipe_width: width of pipes
    :type pipe_width: float, default 5.0
    :param junction_size: Relative size of junctions to plot. The value junction_size is multiplied\
            with mean_distance_between_buses, which equals the distance between the max geoocord\
            and the min divided by 200: \
            mean_distance_between_buses = sum((net['bus_geodata'].max() \
                                               - net['bus_geodata'].min()) / 200)
    :type junction_size: float, default 1.0
    :param ext_grid_size: Relative size of ext_grids to plot. See bus sizes for details. Note: \
            ext_grids are plottet as rectangles
    :type ext_grid_size: float, default 1.0
    :param plot_sinks: Flag to decide whether sink symbols should be drawn.
    :type plot_sinks: bool, default False
    :param plot_sources: Flag to decide whether source symbols should be drawn.
    :type plot_sources: bool, default False
    :param sink_size: Relative size of sinks to plot.
    :type sink_size: float, default 1.0
    :param source_size: Relative size of sources to plot.
    :type source_size: float, default 1.0
    :param valve_size: Relative size of valves to plot.
    :type valve_size: float, default 1.0
    :param heat_exchanger_size:
    :type heat_exchanger_size:
    :param scale_size: Flag if junction_size, ext_grid_size, valve_size- and distance will be \
            scaled with respect to grid mean distances
    :type scale_size: bool, default True
    :param junction_color: Junction Color. See also matplotlib or seaborn documentation on how to\
            choose colors.
    :type junction_color: str, tuple, default "r"
    :param pipe_color: Pipe Color.
    :type pipe_color: str, tuple, default "silver"
    :param ext_grid_color: External Grid Color.
    :type ext_grid_color: str, tuple, default "orange"
    :param library: library name to create generic coordinates (case of missing geodata). Choose\
            "igraph" to use igraph package or "networkx" to use networkx package.
    :type library: str, default "igraph"
    :param show_plot: If True, show plot at the end of plotting
    :type show_plot: bool, default True
    :param ax: matplotlib axis to plot to
    :type ax: object, default None
    :return: ax - axes of figure
    """
    collections = create_simple_collections(net, respect_valves, pipe_width, junction_size,
                                            ext_grid_size, plot_sinks, plot_sources, sink_size,
                                            source_size, valve_size, pump_size, heat_exchanger_size,
                                            scale_size, junction_color, pipe_color, ext_grid_color,
                                            valve_color, pump_color, heat_exchanger_color,
                                            library, as_dict=False, **kwargs)
    ax = draw_collections(collections, ax=ax)

    if show_plot:
        plt.show()
    return ax



def create_simple_collections_pps(net, line_width=5.0, bus_size=1.0,
                              ext_grid_size=1.5, plot_loads=False, plot_sgen=False,
                              load_size=1.0, sgen_size=1.0,
                               scale_size=True, bus_color="r",
                              line_color='blue', ext_grid_color='orange',
                              library="igraph", as_dict=True, **kwargs):
    """
    Plots a pandapipes network as simple as possible. If no geodata is available, artificial
    geodata is generated. For advanced plotting see the tutorial

    :param net: The pandapipes format network.
    :type net: pandapipesNet
    :param respect_valves: Respect valves if artificial geodata is created. \
            .. note:: This Flag is ignored if plot_line_switches is True
    :type respect_valves: bool default False
    :param pipe_width: width of pipes
    :type pipe_width: float, default 5.0
    :param junction_size: Relative size of junctions to plot. The value junction_size is multiplied\
            with mean_distance_between_buses, which equals the distance between the max geoocord\
            and the min divided by 200: \
            mean_distance_between_buses = sum((net['bus_geodata'].max() \
                                               - net['bus_geodata'].min()) / 200)
    :type junction_size: float, default 1.0
    :param ext_grid_size: Relative size of ext_grids to plot. See bus sizes for details. Note: \
            ext_grids are plottet as rectangles
    :type ext_grid_size: float, default 1.0
    :param plot_sinks: Flag to decide whether sink symbols should be drawn.
    :type plot_sinks: bool, default False
    :param plot_sources: Flag to decide whether source symbols should be drawn.
    :type plot_sources: bool, default False
    :param sink_size: Relative size of sinks to plot.
    :type sink_size: float, default 1.0
    :param source_size: Relative size of sources to plot.
    :type source_size: float, default 1.0
    :param valve_size: Relative size of valves to plot.
    :type valve_size: float, default 1.0
    :param heat_exchanger_size:
    :type heat_exchanger_size:
    :param scale_size: Flag if junction_size, ext_grid_size, valve_size- and distance will be \
            scaled with respect to grid mean distances
    :type scale_size: bool, default True
    :param junction_color: Junction Color. See also matplotlib or seaborn documentation on how to\
            choose colors.
    :type junction_color: str, tuple, default "r"
    :param pipe_color: Pipe Color.
    :type pipe_color: str, tuple, default "silver"
    :param ext_grid_color: External Grid Color.
    :type ext_grid_color: str, tuple, default "orange"
    :param library: library name to create generic coordinates (case of missing geodata). Choose\
            "igraph" to use igraph package or "networkx" to use networkx package. **NOTE**: \
            Currently the networkx implementation is not working!
    :type library: str, default "igraph"
    :param as_dict: flag whether to return dictionary for network components or just a list
    :type as_dict: bool, default True
    :return: collections - list of simple collections for the given network
    """
    # don't hide lines if switches are plotted

    # create geocoord if none are available
    if len(net.bus_geodata) == 0 and len(net.line_geodata) == 0:
        logger.warning("No or insufficient geodata available --> Creating artificial coordinates." +
                       " This may take some time")
        create_generic_coordinates(net, library=library)


    if scale_size:
        # if scale_size -> calc size from distance between min and max geocoord
        sizes = get_collection_sizes_pps(net, bus_size, ext_grid_size, load_size, sgen_size)
        bus_size = sizes["bus"]
        ext_grid_size = sizes["ext_grid"]
        sgen_size = sizes["sgen"]
        load_size = sizes["load"]


    # create junction collections to plot
    bus_coll = create_bus_collection(net, net.bus.index, size=bus_size,
                                               color=bus_color, zorder=10)

    # if bus geodata is available, but no line geodata
    use_bus_geodata = len(net.line_geodata) == 0

    plot_lines = net.line[net.line.in_service].index

    # create line collections
    color = list()
    for i in range(len(plot_lines)):
        if i !=4:
            color.append((0,0,1))
        else:
            color.append((51/255, 204/255, 51/255))
    line_coll = create_line_collection(net, plot_lines, color=color, linewidths=line_width,
                                       use_bus_geodata=use_bus_geodata)


    collections = {"bus": bus_coll, "line": line_coll}

    # create ext_grid collections
    eg_bus_with_geo_coordinates = set(net.ext_grid.bus.values) \
                                        & set(net.bus_geodata.index)
    # if len(eg_bus_with_geo_coordinates) > 0:
    #     eg_coll = create_node_collection(
    #         net, eg_bus_with_geo_coordinates, patch_type="rect", size=ext_grid_size,
    #         color=ext_grid_color, zorder=11)
    #     collections["ext_grid"] = eg_coll

    eg_coll = create_ext_grid_collection(net, size=ext_grid_size)
    collections["ext_grid"] = eg_coll

    if 'sgen' in net and plot_sgen and len(net.sgen) > 0:
        sgen_colls = create_sgen_collection(net, size=sgen_size)
        collections["sgen"] = sgen_colls

    print(net)
    if 'trafo' in net  and len(net.trafo) > 0:
        trafo_colls = create_trafo_collection(net, size=load_size, color='blue')
        collections["trafo"] = trafo_colls

   # if 'load' in net and plot_loads and len(net.load) > 0:
    #    load_colls = create_load_collection(net, size=load_size)
   #     collections["load"] = load_colls

    if 'additional_collections' in kwargs:
        collections["additional"] = list()
        for collection in kwargs.pop('additional_collections'):
            collections["additional"].extend(collection)

    if as_dict:
        return collections
    return list(chain.from_iterable([list(c) if hasattr(c, "__iter__") else [c]
                                     for c in collections.values()]))





def create_simple_collections(net, controller, respect_valves=False, pipe_width=7.0, junction_size=1.0,
                              ext_grid_size=2.0, plot_sinks=False, plot_sources=False,
                              sink_size=1.0, source_size=1.0, valve_size=1.0, pump_size=1.0,
                              heat_exchanger_size=1.0, scale_size=True, junction_color="r",
                              pipe_color='silver', ext_grid_color='orange', valve_color='silver',
                              pump_color='silver', heat_exchanger_color='silver',
                              library="igraph", as_dict=True, **kwargs):
    """
    Plots a pandapipes network as simple as possible. If no geodata is available, artificial
    geodata is generated. For advanced plotting see the tutorial

    :param net: The pandapipes format network.
    :type net: pandapipesNet
    :param respect_valves: Respect valves if artificial geodata is created. \
            .. note:: This Flag is ignored if plot_line_switches is True
    :type respect_valves: bool default False
    :param pipe_width: width of pipes
    :type pipe_width: float, default 5.0
    :param junction_size: Relative size of junctions to plot. The value junction_size is multiplied\
            with mean_distance_between_buses, which equals the distance between the max geoocord\
            and the min divided by 200: \
            mean_distance_between_buses = sum((net['bus_geodata'].max() \
                                               - net['bus_geodata'].min()) / 200)
    :type junction_size: float, default 1.0
    :param ext_grid_size: Relative size of ext_grids to plot. See bus sizes for details. Note: \
            ext_grids are plottet as rectangles
    :type ext_grid_size: float, default 1.0
    :param plot_sinks: Flag to decide whether sink symbols should be drawn.
    :type plot_sinks: bool, default False
    :param plot_sources: Flag to decide whether source symbols should be drawn.
    :type plot_sources: bool, default False
    :param sink_size: Relative size of sinks to plot.
    :type sink_size: float, default 1.0
    :param source_size: Relative size of sources to plot.
    :type source_size: float, default 1.0
    :param valve_size: Relative size of valves to plot.
    :type valve_size: float, default 1.0
    :param heat_exchanger_size:
    :type heat_exchanger_size:
    :param scale_size: Flag if junction_size, ext_grid_size, valve_size- and distance will be \
            scaled with respect to grid mean distances
    :type scale_size: bool, default True
    :param junction_color: Junction Color. See also matplotlib or seaborn documentation on how to\
            choose colors.
    :type junction_color: str, tuple, default "r"
    :param pipe_color: Pipe Color.
    :type pipe_color: str, tuple, default "silver"
    :param ext_grid_color: External Grid Color.
    :type ext_grid_color: str, tuple, default "orange"
    :param library: library name to create generic coordinates (case of missing geodata). Choose\
            "igraph" to use igraph package or "networkx" to use networkx package. **NOTE**: \
            Currently the networkx implementation is not working!
    :type library: str, default "igraph"
    :param as_dict: flag whether to return dictionary for network components or just a list
    :type as_dict: bool, default True
    :return: collections - list of simple collections for the given network
    """
    # don't hide lines if switches are plotted

    if isinstance(net,list):
        power_net = net[0]
        dh_net = net[1]

    # create geocoord if none are available
    if len(dh_net.junction_geodata) == 0 and len(dh_net.pipe_geodata) == 0:
        logger.warning("No or insufficient geodata available --> Creating artificial coordinates." +
                       " This may take some time")
        create_generic_coordinates(dh_net, library=library)

    if scale_size:
        # if scale_size -> calc size from distance between min and max geocoord
        sizes = get_collection_sizes(dh_net, junction_size, ext_grid_size, sink_size, source_size,
                                     valve_size, pump_size, heat_exchanger_size)
        junction_size = sizes["junction"]
        ext_grid_size = sizes["ext_grid"]
        source_size = sizes["source"]
        sink_size = sizes["sink"]
        valve_size = sizes["valve"]
        pump_size = sizes["pump"]
        heat_exchanger_size = sizes["heat_exchanger"]

    # create junction collections to plot
    junction_coll = create_junction_collection(dh_net, dh_net.junction.index, size=junction_size,
                                               color=junction_color, zorder=10)

    # if bus geodata is available, but no line geodata
    use_junction_geodata = len(dh_net.pipe_geodata) == 0

    plot_lines = dh_net.pipe[dh_net.pipe.in_service].index

    # create line collections
    pipe_coll = create_pipe_collection(dh_net, plot_lines, color=pipe_color, linewidths=pipe_width,
                                       use_junction_geodata=use_junction_geodata)
    collections = {"junction": junction_coll, "pipe": pipe_coll}

    # create ext_grid collections
    eg_junctions_with_geo_coordinates = set(dh_net.ext_grid.junction.values) \
                                        & set(dh_net.junction_geodata.index)
    if len(eg_junctions_with_geo_coordinates) > 0:
        eg_coll = create_junction_collection(
            dh_net, eg_junctions_with_geo_coordinates, patch_type="rect", size=ext_grid_size,
            color=ext_grid_color, zorder=11)
        collections["ext_grid"] = eg_coll

    if 'source' in net and plot_sources and len(net.source) > 0:
        source_colls = create_source_collection(net, size=source_size)
        collections["source"] = source_colls

    if 'sink' in dh_net and plot_sinks and len(dh_net.sink) > 0:
        sink_colls = create_sink_collection(dh_net, size=sink_size)
        collections["sink"] = sink_colls

    if 'valve' in net:
        valve_colls = create_valve_collection(dh_net, size=valve_size, linewidths=pipe_width, color=valve_color,
                                              respect_valves=respect_valves)
        collections["valve"] = valve_colls

    if 'pump' in dh_net:
        pump_colls = create_pump_collection(dh_net, size=pump_size, linewidths=pipe_width, color=pump_color)
        collections["pump"] = pump_colls

    # if 'circ_pump_mass' in dh_net:
    #     circ_pump_colls = create_pump_collection(dh_net, table_name='circ_pump_mass',
    #                                         size=pump_size, linewidths=pipe_width, color=pump_color)
    #     collections["circ_pump_mass"] = circ_pump_colls

    if 'circ_pump_pressure' in dh_net:
        circ_pump_colls = create_pump_collection(dh_net, table_name='circ_pump_pressure',
                                            size=pump_size, linewidths=pipe_width, color=pump_color)
        collections["circ_pump_pressure"] = circ_pump_colls

    if 'controller' in dh_net:
        for i in range(len(controller)):
            if isinstance(controller[i], hctrl.HouseControl):
                cpc = create_house_collection(power_net, dh_net, controller, size=pump_size,
                                                       linewidths = pipe_width, color = heat_exchanger_color)
                collections["house"] = cpc
            elif isinstance(controller[i],pumpctrl.HeatPumpControl):
                hpc = create_hp_collection(power_net, dh_net, controller, size=pump_size,
                                              linewidths=pipe_width, color=heat_exchanger_color)
                collections["hp"] = hpc




   # if 'heat_exchanger' in dh_net:
    #    hxc = create_heat_exchanger_collection(dh_net, size=heat_exchanger_size, linewidths=pipe_width,
    #                                           color=heat_exchanger_color)
    #    collections["heat_exchanger"] = hxc

    if 'additional_collections' in kwargs:
        collections["additional"] = list()
        for collection in kwargs.pop('additional_collections'):
            collections["additional"].extend(collection)

    if as_dict:
        return collections
    return list(chain.from_iterable([list(c) if hasattr(c, "__iter__") else [c]
                                     for c in collections.values()]))



