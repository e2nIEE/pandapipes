# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from pandapower.plotting.geo import _node_geometries_from_geodata, \
    _transform_node_geometry_to_geodata, _branch_geometries_from_geodata, \
    _transform_branch_geometry_to_coords, _convert_xy_epsg


def convert_gis_to_geodata(net, node_geodata=True, branch_geodata=True):
    """
    Extracts information on bus and line geodata from the geometries of a geopandas geodataframe.

    :param net: The net for which to convert the geodata
    :type net: pandapowerNet
    :param node_geodata: flag if to extract x and y values for bus geodata
    :type node_geodata: bool, default True
    :param branch_geodata: flag if to extract coordinates values for line geodata
    :type branch_geodata: bool, default True
    :return: No output.
    """
    if node_geodata:
        _transform_node_geometry_to_geodata(net.junction_geodata)
    if branch_geodata:
        _transform_branch_geometry_to_coords(net.pipe_geodata)


def convert_geodata_to_gis(net, epsg=31467, node_geodata=True, branch_geodata=True):
    """
    Transforms the bus and line geodata of a net into a geopandaas geodataframe with the respective
    geometries.

    :param net: The net for which to convert the geodata
    :type net: pandapowerNet
    :param epsg: current epsg projection
    :type epsg: int, default 4326 (= WGS84)
    :param node_geodata: flag if to transform the bus geodata table
    :type node_geodata: bool, default True
    :param branch_geodata: flag if to transform the line geodata table
    :type branch_geodata: bool, default True
    :return: No output.
    """
    if node_geodata:
        net["bus_geodata"] = _node_geometries_from_geodata(net["bus_geodata"], epsg)
    if branch_geodata:
        net["line_geodata"] = _branch_geometries_from_geodata(net["line_geodata"], epsg)
    net["gis_epsg_code"] = epsg


def convert_epsg_bus_geodata(net, epsg_in=4326, epsg_out=31467):
    """
    Converts bus geodata in net from epsg_in to epsg_out

    :param net: The pandapower network
    :type net: pandapowerNet
    :param epsg_in: current epsg projection
    :type epsg_in: int, default 4326 (= WGS84)
    :param epsg_out: epsg projection to be transformed to
    :type epsg_out: int, default 31467 (= Gauss-Krüger Zone 3)
    :return: net - the given pandapower network (no copy!)
    """
    net['bus_geodata'].loc[:, "x"], net['bus_geodata'].loc[:, "y"] = _convert_xy_epsg(
        net['bus_geodata'].loc[:, "x"], net['bus_geodata'].loc[:, "y"], epsg_in, epsg_out)
    return net
