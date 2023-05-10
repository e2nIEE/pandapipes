# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from functools import partial

import numpy as np
import pandas as pd
from pandapower.plotting.collections import _create_complex_branch_collection, \
    add_cmap_to_collection, coords_from_node_geodata
from pandapower.plotting.plotting_toolbox import get_index_array

from pandapipes.plotting.patch_makers import valve_patches

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def create_valve_pipe_collection(net, valve_pipes=None, valve_pipe_geodata=None, junction_geodata=None,
                                 use_junction_geodata=False, infofunc=None, fill_closed=True,
                                 respect_valves=False, size=5., cmap=None, norm=None,
                                 picker=False, z=None, cbar_title="Pipe Loading  [%]", clim=None, **kwargs):
    """
    Creates a matplotlib patch collection of pandapipes junction-junctiion valve_pipes.
    Valve_pipes are plotted in the center between two junctions with a "helper" line
    (dashed and thin) being drawn between the junctions as well.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param size: patch size
    :type size: float, default 5.
    :param helper_line_style: Line style of the "helper" line being plotted between two junctions
                                connected by a junction-junction valve_pipe
    :type helper_line_style: str, default ":"
    :param helper_line_size: Line width of the "helper" line being plotted between two junctions
                                connected by a junction-junction valve_pipe
    :type helper_line_size:  float, default 1.
    :param helper_line_color: Line color of the "helper" line being plotted between two junctions
                                connected by a junction-junction valve_pipe
    :type helper_line_color: str, default "gray"
    :param orientation: orientation of valve_pipe collection. pi is directed downwards,
                    increasing values lead to clockwise direction changes.
    :type orientation: float, default np.pi/2
    :param kwargs: Key word arguments are passed to the patch function
    :return: valves, helper_lines
    :rtype: tuple of patch collections
    """
    if use_junction_geodata is False and net.valve_pipe_geodata.empty:
        # if bus geodata is available, but no line geodata
        logger.warning("use_junction_geodata is automatically set to True, since net.valve_pipe_geodata "
                       "is empty.")
        use_junction_geodata = True
    valve_pipes = get_index_array(valve_pipes, net.valve[net.valve_pipe.opened.values].index if \
        respect_valves else net.valve_pipe.index)
    if len(valve_pipes) == 0:
        return None

    if use_junction_geodata:
        coords, valve_pipes_with_geo = coords_from_node_geodata(
            valve_pipes, net.valve_pipe.from_junction.loc[valve_pipes].values,
            net.valve_pipe.to_junction.loc[valve_pipes].values,
            junction_geodata if junction_geodata is not None else net["junction_geodata"], "valve_pipe",
            "Junction")
    else:
        if valve_pipe_geodata is None:
            valve_pipe_geodata = net.valve_pipe_geodata
        valve_pipes_with_geo = valve_pipes[np.isin(valve_pipes, valve_pipe_geodata.index.values)]
        coords = list(valve_pipe_geodata.loc[valve_pipes_with_geo, "coords"])
        valve_pipes_without_geo = set(valve_pipes) - set(valve_pipes_with_geo)
        if valve_pipes_without_geo:
            logger.warning("Could not plot valve-pipes %s. Geodata is missing for those valve-pipes!"
                           % valve_pipes_without_geo)

    if len(valve_pipes_with_geo) == 0:
        return None

    infos = [infofunc(pipe) for pipe in valve_pipes_with_geo] if infofunc else []

    colors = kwargs.pop("color", "k")
    linewidths = kwargs.pop("linewidths", 2.)
    linewidths = kwargs.pop("linewidth", linewidths)
    linewidths = kwargs.pop("lw", linewidths)
    patch_edgecolor = kwargs.pop("patch_edgecolor", colors)
    line_color = kwargs.pop("line_color", colors)

    filled = net.valve_pipe.loc[valve_pipes, "opened"].values
    if fill_closed:
        filled = ~filled

    patch = partial(valve_patches, valve_position=4, linestyle=':')
    lc, pc = _create_complex_branch_collection(
        coords, patch, size, infos, picker=picker, linewidths=linewidths, filled=filled,
        patch_edgecolor=patch_edgecolor, line_color=line_color, **kwargs)

    if cmap is not None:
        if z is None:
            z = net.res_valve_pipe.v_mean_m_per_s.loc[valve_pipes_with_geo]
        elif isinstance(z, pd.Series):
            z = z.loc[valve_pipes_with_geo]
        add_cmap_to_collection(lc, cmap, norm, z, cbar_title, clim=clim)

    return lc, pc

