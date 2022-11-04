
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import RegularPolygon

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def create_valve_pipe_collection(net, size=5., helper_line_style=':', helper_line_size=1.,
                                 helper_line_color="gray", orientation=np.pi/2, **kwargs):
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
    valves = net.valve_pipe.index
    color = kwargs.pop("color", "k")
    valve_patches = []
    line_patches = []
    r_triangle = size * 1
    ang = orientation if hasattr(orientation, '__iter__') else [orientation] * \
        net.valve_pipe.shape[0]
    for i, valve in enumerate(valves):
        from_junction = net.valve_pipe.from_junction.loc[valve]
        to_junction = net.valve_pipe.to_junction.loc[valve]
        if from_junction not in net.junction_geodata.index or to_junction not in \
                net.junction_geodata.index:
            logger.warning("Junction coordinates for valve %s not found, skipped valve!" % valve)
            continue
        # switch bus and target coordinates
        pos_fj = net.junction_geodata.loc[from_junction, ["x", "y"]].values.astype(np.float64)
        pos_tj = net.junction_geodata.loc[to_junction, ["x", "y"]].values.astype(np.float64)
        # position of switch symbol
        vec = pos_tj - pos_fj
        angle = np.arctan2(vec[1], vec[0])
        pos_val = pos_fj + vec * 0.5 if not np.allclose(pos_tj, pos_fj) else pos_tj
        mp_circ = pos_val
        # rotation of switch symbol
        # color switch by state
        col = color if net.valve_pipe.opened.loc[valve] else "white"
        # create switch patch (switch size is respected to center the switch on the line)
        # norm = np.array([-vec[1], vec[0]])
        mp_tri1 = mp_circ + vec * r_triangle * 0.9
        mp_tri2 = mp_circ - vec * r_triangle * 0.9
        valve_patches.append(RegularPolygon(
            mp_tri1, numVertices=3, radius=r_triangle, orientation=angle+ang[i], facecolor=col,
            edgecolor=color))
        valve_patches.append(RegularPolygon(
            mp_tri2, numVertices=3, radius=r_triangle, orientation=angle-ang[i], facecolor=col,
            edgecolor=color))
        # apply rotation
        # add to collection lists
        line_patches.append([pos_fj.tolist(), pos_tj.tolist()])
    # create collections and return
    valves = PatchCollection(valve_patches, match_original=True, **kwargs)
    helper_lines = LineCollection(line_patches, linestyles=helper_line_style,
                                  linewidths=helper_line_size, colors=helper_line_color)
    return valves, helper_lines
