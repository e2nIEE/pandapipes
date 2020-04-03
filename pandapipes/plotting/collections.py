# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
from pandapower.plotting.collections import _create_node_collection, \
    _create_node_element_collection, _create_line2d_collection, _create_complex_branch_collection, \
    add_cmap_to_collection, coords_from_node_geodata
from pandapower.plotting.patch_makers import load_patches, ext_grid_patches
from pandapipes.plotting.patch_makers import valve_patches, source_patches, heat_exchanger_patches, \
    pump_patches
from pandapower.plotting.plotting_toolbox import get_index_array

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def create_junction_collection(net, junctions=None, size=5, patch_type="circle", color=None,
                               z=None, cmap=None, norm=None, infofunc=None, picker=False,
                               junction_geodata=None, cbar_title="Junction Pressure [bar]",
                               **kwargs):
    """
    Creates a matplotlib patch collection of pandapipes junctions.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param junctions: The junctions for which the collections are created.
                    If None, all junctions in the network are considered.
    :type junctions: list, default None
    :param size: patch size
    :type size: int, default 5
    :param patch_type: patch type, can be\
        - "circle" or "ellipse" for an ellipse (cirlces are just ellipses with the same width \
            + height)\
        - "rect" or "rectangle" for a rectangle\
        - "poly<n>" for a polygon with n edges
    :type patch_type: str, default "circle"
    :param color: color or list of colors for every element
    :type color: iterable, float, default None
    :param z: array of magnitudes for colormap. Used in case of given cmap. If None,\
        net.res_junction.p_bar is used.
    :type z: array, default None
    :param cmap: colormap for the patch colors
    :type cmap: matplotlib colormap object, default None
    :param norm:  matplotlib norm object
    :type norm: matplotlib norm object, default None
    :param infofunc: infofunction for the patch element
    :type infofunc: function, default None
    :param picker: picker argument passed to the patch collection
    :type picker: bool, default False
    :param junction_geodata: coordinates to use for plotting. If None, net["junction_geodata"] is\
        used
    :type junction_geodata: pandas.DataFrame, default None
    :param cbar_title: colormap bar title in case of given cmap
    :type cbar_title: str, default "Junction Pressure [bar]"
    :param kwargs: keyword arguments are passed to the patch function and the patch maker
    :return: pc (matplotlib collection object) - patch collection
    """
    junctions = get_index_array(junctions, net.junction.index)
    if len(junctions) == 0:
        return None
    if junction_geodata is None:
        junction_geodata = net["junction_geodata"]

    junctions_with_geo = junctions[np.isin(junctions, junction_geodata.index.values)]
    if len(junctions_with_geo) < len(junctions):
        logger.warning("The following junctions cannot be displayed, as there is no geodata "
                       "available: %s" % (set(junctions) - set(junctions_with_geo)))

    coords = list(zip(junction_geodata.loc[junctions_with_geo, "x"].values,
                      junction_geodata.loc[junctions_with_geo, "y"].values))

    infos = [infofunc(junc) for junc in junctions_with_geo] if infofunc is not None else []

    pc = _create_node_collection(junctions_with_geo, coords, size, patch_type, color, picker, infos,
                                 **kwargs)

    if cmap is not None:
        if z is None:
            z = net.res_junction.p_bar.loc[junctions_with_geo]
        add_cmap_to_collection(pc, cmap, norm, z, cbar_title)

    return pc


def create_pipe_collection(net, pipes=None, pipe_geodata=None, junction_geodata=None,
                           use_junction_geodata=False, infofunc=None, cmap=None, norm=None,
                           picker=False, z=None, cbar_title="Pipe Loading [%]", clim=None,
                           **kwargs):
    """
    Creates a matplotlib pipe collection of pandapipes pipes.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param pipes: The pipes for which the collections are created. If None, all pipes
            in the network are considered.
    :type pipes: list, default None
    :param pipe_geodata: coordinates to use for plotting. If None, net["pipe_geodata"] is used
    :type pipe_geodata: pandas.DataFrame, default None
    :param junction_geodata: coordinates to use for plotting in case of use_junction_geodata=True.\
        If None, net["junction_geodata"] is used
    :type junction_geodata: pandas.DataFrame, default None
    :param use_junction_geodata: Defines whether junction or pipe geodata are used.
    :type use_junction_geodata: bool, default False
    :param infofunc: infofunction for the line element
    :type infofunc: function, default None
    :param cmap: colormap for the line colors
    :type cmap: matplotlib norm object, default None
    :param norm: matplotlib norm object
    :type norm: matplotlib norm object, default None
    :param picker: picker argument passed to the patch collection
    :type picker: bool, default False
    :param z: array of pipe loading magnitudes for colormap. Used in case of given cmap. If None,\
        net.res_pipe.loading_percent is used.
    :type z: array, default None
    :param cbar_title: colormap bar title in case of given cmap
    :type cbar_title: str, default "Pipe Loading [%]"
    :param clim: setting the norm limits for image scaling
    :type clim: tuple of floats, default None
    :param kwargs: keyword arguments are passed to the patch function and the patch maker
    :return: lc (matplotlib line collection)- line collection for pipes
    """
    if use_junction_geodata is False and net.pipe_geodata.empty:
        # if bus geodata is available, but no line geodata
        logger.warning("use_junction_geodata is automatically set to True, since net.pipe_geodata "
                       "is empty.")
        use_junction_geodata = True

    pipes = get_index_array(pipes, net.pipe.index)
    if len(pipes) == 0:
        return None

    if use_junction_geodata:
        coords, pipes_with_geo = coords_from_node_geodata(
            pipes, net.pipe.from_junction.loc[pipes].values, net.pipe.to_junction.loc[pipes].values,
            junction_geodata if junction_geodata is not None else net["junction_geodata"], "pipe",
            "Junction")
    else:
        if pipe_geodata is None:
            pipe_geodata = net.pipe_geodata
        pipes_with_geo = pipes[np.isin(pipes, pipe_geodata.index.values)]
        coords = list(pipe_geodata.loc[pipes_with_geo, "coords"])
        pipes_without_geo = set(pipes) - set(pipes_with_geo)
        if pipes_without_geo:
            logger.warning("Could not plot pipes %s. Pipe geodata is missing for those pipes!"
                           % pipes_without_geo)

    if len(pipes_with_geo) == 0:
        return None

    infos = [infofunc(pipe) for pipe in pipes_with_geo] if infofunc else []

    lc = _create_line2d_collection(coords, pipes_with_geo, infos=infos, picker=picker, **kwargs)

    if cmap is not None:
        if z is None:
            z = net.res_pipe.v_mean_m_per_s.loc[pipes_with_geo]
        add_cmap_to_collection(lc, cmap, norm, z, cbar_title, clim)

    return lc


def create_sink_collection(net, sinks=None, size=1., infofunc=None, picker=False,
                           orientation=(np.pi*5/6), **kwargs):
    """
    Creates a matplotlib patch collection of pandapipes sinks.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param sinks: The sinks for which the collections are created. If None, all sinks
                        connected to junctions that have junction_geodata entries are considered.
    :type sinks: list, default None
    :param size: patch size
    :type size: float, default 1
    :param infofunc: infofunction for the patch element
    :type infofunc: function, default None
    :param picker: picker argument passed to the patch collection
    :type picker: bool, default False
    :param orientation: orientation of sink collection. pi is directed downwards, increasing values\
        lead to clockwise direction changes.
    :type orientation: float, default np.pi
    :param kwargs: key word arguments are passed to the patch function
    :return: sink_pc - patch collection
             sink_lc - line collection
    """
    sinks = get_index_array(sinks, net.sink.index)
    if len(sinks) == 0:
        return None
    infos = [infofunc(i) for i in range(len(sinks))] if infofunc is not None else []
    node_coords = net.junction_geodata.loc[
        net.sink.loc[sinks, "junction"].values, ['x', 'y']].values
    sink_pc, sink_lc = _create_node_element_collection(
        node_coords, load_patches, size=size, infos=infos, orientation=orientation,
        picker=picker, **kwargs)
    return sink_pc, sink_lc


def create_source_collection(net, sources=None, size=1., infofunc=None, picker=False,
                             orientation=(np.pi*7/6), **kwargs):
    """
    Creates a matplotlib patch collection of pandapipes sources.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param sources: The sources for which the collections are created. If None, all sources
                        connected to junctions that have junction_geodata entries are considered.
    :type sources: list, default None
    :param size: patch size
    :type size: float, default 1.
    :param infofunc: infofunction for the patch element
    :type infofunc: function, default None
    :param picker: picker argument passed to the patch collection
    :type picker: bool, default False
    :param orientation: orientation of source collection. pi is directed downwards, increasing\
        values lead to clockwise direction changes.
    :type orientation: float, default np.pi
    :param kwargs: key word arguments are passed to the patch function
    :return: source_pc - patch collection
             source_lc - line collection
    """
    sources = get_index_array(sources, net.sink.index)
    if len(sources) == 0:
        return None
    infos = [infofunc(i) for i in range(len(sources))] if infofunc is not None else []
    node_coords = net.junction_geodata.loc[:, ["x", "y"]].values[net.source.loc[sources,
                                                                                "junction"].values]
    source_pc, source_lc = _create_node_element_collection(
        node_coords, source_patches, size=size, infos=infos, orientation=orientation,
        picker=picker, repeat_infos=(1, 3), **kwargs)
    return source_pc, source_lc


def create_ext_grid_collection(net, size=1., infofunc=None, orientation=0, picker=False,
                               ext_grids=None, ext_grid_junctions=None, **kwargs):
    """
    Creates a matplotlib patch collection of pandapipes ext_grid. Parameters
    ext_grids, ext_grid_junctions can be used to specify, which ext_grids the collection should be
    created for.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param size: patch size
    :type size: float, default 1.
    :param infofunc: infofunction for the patch element
    :type infofunc: function, default None
    :param orientation: orientation of ext_grid collection. 0 is directed upwards,
                        increasing values lead to clockwise direction changes.
    :type orientation: float, default 0
    :param picker: picker argument passed to the patch collection
    :type picker: bool, default False
    :param ext_grids: The ext_grids for which the collections are created. If None, all ext_grids
                        which have the entry coords in ext_grid_geodata are considered.
    :type ext_grids: list, default None
    :param ext_grid_junctions: junctions to be used as ext_grid locations
    :type ext_grid_junctions: np.ndarray, default None
    :param kwargs: key word arguments are passed to the patch function
    :return: ext_grid1 - patch collection
             ext_grid2 - patch collection
    """
    ext_grids = get_index_array(ext_grids, net.ext_grid.index)
    if ext_grid_junctions is None:
        ext_grid_junctions = net.ext_grid.junction.loc[ext_grids].values
    else:
        if len(ext_grids) != len(ext_grid_junctions):
            raise ValueError("Length mismatch between chosen ext_grids and ext_grid_junctions.")
    infos = [infofunc(ext_grid_idx) for ext_grid_idx in ext_grids] if infofunc is not None else []

    node_coords = net.junction_geodata.loc[ext_grid_junctions, ["x", "y"]].values
    ext_grid_pc, ext_grid_lc = _create_node_element_collection(
        node_coords, ext_grid_patches, size=size, infos=infos, orientation=orientation,
        picker=picker, hatch="XXX", **kwargs)
    return ext_grid_pc, ext_grid_lc


def create_heat_exchanger_collection(net, hex=None, size=5., junction_geodata=None, color='k',
                                     infofunc=None, picker=False, **kwargs):
    """
    Creates a matplotlib patch collection of pandapipes junction-junction heat_exchangers.
    Heat_exchangers are plotted in the center between two junctions with a "helper" line
    (dashed and thin) being drawn  between the junctions as well.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param size: patch size
    :type size: float, default 2.
    :param helper_line_style: Line style of the "helper" line being plotted between two junctions
                                connected by a junction-junction heat_exchanger
    :type helper_line_style: str, default ":"
    :param helper_line_size: Line width of the "helper" line being plotted between two junctions
                                connected by a junction-junction heat_exchanger
    :type helper_line_size: float, default 1.
    :param helper_line_color: Line color of the "helper" line being plotted between two junctions
                                connected by a junction-junction valve
    :type helper_line_color: str, default "gray"
    :param orientation: orientation of heat_exchanger collection. pi is directed downwards,
                    increasing values lead to clockwise direction changes.
    :type orientation: float, default np.pi/2
    :param kwargs: Key word arguments are passed to the patch function
    :return: heat_exchanger, helper_lines
    :rtype: tuple of patch collections
    """
    hex = get_index_array(hex, net.heat_exchanger.index)
    hex_table = net.heat_exchanger.loc[hex]

    coords, hex_with_geo = coords_from_node_geodata(
        hex, hex_table.from_junction.values, hex_table.to_junction.values,
        junction_geodata if junction_geodata is not None else net["junction_geodata"],
        "heat_exchanger", "Junction")

    if len(hex_with_geo) == 0:
        return None

    linewidths = kwargs.pop("linewidths", 2.)
    linewidths = kwargs.pop("linewidth", linewidths)
    linewidths = kwargs.pop("lw", linewidths)

    infos = list(np.repeat([infofunc(i) for i in range(len(hex_with_geo))], 2)) \
        if infofunc is not None else []

    lc, pc = _create_complex_branch_collection(coords, heat_exchanger_patches, size, infos,
                                               picker=picker, linewidths=linewidths,
                                               patch_facecolor=color, line_color=color,
                                               **kwargs)

    return lc, pc


def create_valve_collection(net, valves=None, size=5., junction_geodata=None, color='k',
                            infofunc=None, picker=False, fill_closed=True,
                            respect_valves=False, **kwargs):
    """
    Creates a matplotlib patch collection of pandapipes junction-junction valves. Valves are
    plotted in the center between two junctions with a "helper" line (dashed and thin) being drawn
    between the junctions as well.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param valves: The valves for which the collections are created. If None, all valves which have\
        enries in the respective junction geodata will be plotted.
    :type valves: list, default None
    :param size: patch size
    :type size: float, default 5.
    :param junction_geodata: coordinates to use for plotting. If None, net["junction_geodata"] is \
        used
    :type junction_geodata: pandas.DataFrame, default None
    :param colors: color or list of colors for every valve
    :type colors: iterable, float, default None
    :param infofunc: infofunction for the patch element
    :type infofunc: function, default None
    :param picker: picker argument passed to the patch collection
    :type picker: bool, default False
    :param fill_closed: If True, valves with parameter opened == False will be filled and those\
        with opened == True will have a white facecolor. Vice versa if False.
    :type fill_closed: bool, default True
    :param kwargs: key word arguments are passed to the patch function
    :return: lc - line collection
             pc - patch collection
    """
    valves = get_index_array(
        valves, net.valve[net.valve.opened.values].index if respect_valves else net.valve.index)

    valve_table = net.valve.loc[valves]

    coords, valves_with_geo = coords_from_node_geodata(
        valves, valve_table.from_junction.values, valve_table.to_junction.values,
        junction_geodata if junction_geodata is not None else net["junction_geodata"], "valve",
        "Junction")

    if len(valves_with_geo) == 0:
        return None

    linewidths = kwargs.pop("linewidths", 2.)
    linewidths = kwargs.pop("linewidth", linewidths)
    linewidths = kwargs.pop("lw", linewidths)

    infos = list(np.repeat([infofunc(i) for i in range(len(valves_with_geo))], 2)) \
        if infofunc is not None else []
    filled = valve_table["opened"].values
    if fill_closed:
        filled = ~filled
    lc, pc = _create_complex_branch_collection(coords, valve_patches, size, infos,
                                               picker=picker, linewidths=linewidths, filled=filled,
                                               patch_facecolor=color, line_color=color,
                                               **kwargs)

    return lc, pc


def create_pump_collection(net, pumps=None, table_name='pump', size=5., junction_geodata=None,
                           color='k', infofunc=None, picker=False, **kwargs):
    """
    Creates a matplotlib patch collection of pandapipes junction-junction valves. Valves are
    plotted in the center between two junctions with a "helper" line (dashed and thin) being drawn
    between the junctions as well.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param valves: The valves for which the collections are created. If None, all valves which have\
        enries in the respective junction geodata will be plotted.
    :type valves: list, default None
    :param size: patch size
    :type size: float, default 5.
    :param junction_geodata: coordinates to use for plotting. If None, net["junction_geodata"] is \
        used
    :type junction_geodata: pandas.DataFrame, default None
    :param colors: color or list of colors for every valve
    :type colors: iterable, float, default None
    :param infofunc: infofunction for the patch element
    :type infofunc: function, default None
    :param picker: picker argument passed to the patch collection
    :type picker: bool, default False
    :param fill_closed: If True, valves with parameter opened == False will be filled and those\
        with opened == True will have a white facecolor. Vice versa if False.
    :type fill_closed: bool, default True
    :param kwargs: key word arguments are passed to the patch function
    :return: lc - line collection
             pc - patch collection
    """
    pumps = get_index_array(pumps, net[table_name].index)
    pump_table = net[table_name].loc[pumps]

    coords, pumps_with_geo = coords_from_node_geodata(
        pumps, pump_table.from_junction.values, pump_table.to_junction.values,
        junction_geodata if junction_geodata is not None else net["junction_geodata"], "pump",
        "Junction")

    if len(pumps_with_geo) == 0:
        return None

    linewidths = kwargs.pop("linewidths", 2.)
    linewidths = kwargs.pop("linewidth", linewidths)
    linewidths = kwargs.pop("lw", linewidths)

    infos = list(np.repeat([infofunc(i) for i in range(len(pumps_with_geo))], 2)) \
        if infofunc is not None else []
    lc, pc = _create_complex_branch_collection(coords, pump_patches, size, infos,
                                               picker=picker, linewidths=linewidths,
                                               patch_edgecolor=color, line_color=color,
                                               **kwargs)

    return lc, pc
