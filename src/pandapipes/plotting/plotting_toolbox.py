# Copyright (c) 2020-2025 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def get_collection_sizes(net, junction_size=1.0, ext_grid_size=1.0, sink_size=1.0, source_size=1.0,
                         valve_size=2.0, pump_size=1.0, heat_exchanger_size=1.0,
                         pressure_control_size=1.0, compressor_size=1.0, flow_control_size=1.0, heat_consumer_size=1.0):
    """
    Calculates the size for most collection types according to the distance between min and max
    geocoord so that the collections fit the plot nicely

    .. note: This is implemented because if you would choose a fixed values (e.g.\
        junction_size = 0.2), the size could be too small for large networks and vice versa

    :param net: pandapower network for which to create plot
    :type net: pandapowerNet
    :param junction_size: relative junction size
    :type junction_size: float, default 1.
    :param ext_grid_size: relative external grid size
    :type ext_grid_size: float, default 1.
    :param sink_size: relative sink size
    :type sink_size: float, default 1.
    :param source_size: relative source size
    :type source_size: float, default 1.
    :param valve_size: relative valve size
    :type valve_size: float, default 2.
    :param heat_exchanger_size: relative heat exchanger size
    :type heat_exchanger_size: float, default 1.
    :param heat_consumer_size: relative heat consumer size
    :type heat_consumer_size: float, default 1.
    :return: sizes (dict) - dictionary containing all scaled sizes
    """
    mean_distance_between_junctions = sum(
        (np.max(net['junction_geodata'].loc[:, ["x", "y"]].to_numpy(), axis=0)
         - np.min(net['junction_geodata'].loc[:, ["x", "y"]].to_numpy(), axis=0))
        / 200
    )

    sizes = {
        "junction": junction_size * mean_distance_between_junctions,
        "ext_grid": ext_grid_size * mean_distance_between_junctions * 2,
        "valve": valve_size * mean_distance_between_junctions * 2,
        "sink": sink_size * mean_distance_between_junctions * 2,
        "source": source_size * mean_distance_between_junctions * 2,
        "heat_exchanger": heat_exchanger_size * mean_distance_between_junctions * 8,
        "pump": pump_size * mean_distance_between_junctions * 8,
        "pressure_control": pressure_control_size * mean_distance_between_junctions * 8,
        "compressor": compressor_size * mean_distance_between_junctions * 8,
        "flow_control": flow_control_size * mean_distance_between_junctions * 2,
        "heat_consumer": heat_consumer_size * mean_distance_between_junctions*2
    }

    return sizes


def coords_from_node_geodata(element_indices, from_nodes, to_nodes, node_geodata, table_name,
                             node_name="Bus", ignore_zero_length=True):
    """
    Auxiliary function to get the node coordinates for a number of branches with respective from
    and to nodes. The branch elements for which there is no geodata available are not included in
    the final list of coordinates.

    :param element_indices: Indices of the branch elements for which to find node geodata
    :type element_indices: iterable
    :param from_nodes: Indices of the starting nodes
    :type from_nodes: iterable
    :param to_nodes: Indices of the ending nodes
    :type to_nodes: iterable
    :param node_geodata: Dataframe containing x and y coordinates of the nodes
    :type node_geodata: pd.DataFrame
    :param table_name: Name of the table that the branches belong to (only for logging)
    :type table_name: str
    :param node_name: Name of the node type (only for logging)
    :type node_name: str, default "Bus"
    :param ignore_zero_length: States if branches should be left out, if their length is zero, i.e.\
        from_node_coords = to_node_coords
    :type ignore_zero_length: bool, default True
    :return: Return values are:\
        - coords (list) - list of branch coordinates of shape (N, (2, 2))\
        - elements_with_geo (set) - the indices of branch elements for which coordinates wer found\
            in the node geodata table
    """
    have_geo = np.isin(from_nodes, node_geodata.index.values) \
        & np.isin(to_nodes, node_geodata.index.values)
    elements_with_geo = np.array(element_indices)[have_geo]
    fb_with_geo, tb_with_geo = from_nodes[have_geo], to_nodes[have_geo]
    coords = [[(x_from, y_from), (x_to, y_to)] for x_from, y_from, x_to, y_to
              in np.concatenate([node_geodata.loc[fb_with_geo, ["x", "y"]].values,
                                 node_geodata.loc[tb_with_geo, ["x", "y"]].values], axis=1)
              if not ignore_zero_length or not (x_from == x_to and y_from == y_to)]
    elements_without_geo = set(element_indices) - set(elements_with_geo)
    if len(elements_without_geo) > 0:
        logger.warning("No coords found for %s %s. %s geodata is missing for those %s!"
                       % (table_name + "s", elements_without_geo, node_name, table_name + "s"))
    return coords, elements_with_geo
