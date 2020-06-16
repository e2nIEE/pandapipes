# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import json
import os
import pickle

try:
    from fiona.crs import from_epsg
    from geopandas import GeoDataFrame, GeoSeries
    from shapely.geometry import Point, LineString

    GEOPANDAS_INSTALLED = True
except ImportError:
    GEOPANDAS_INSTALLED = False

from pandapipes.pandapipes_net import pandapipesNet
from pandapower.io_utils import PPJSONEncoder, PPJSONDecoder, to_dict_with_coord_transform, \
    get_raw_data_from_pickle, transform_net_with_df_and_geo
from pandapipes.io.io_utils import ppipes_hook, isinstance_partial


def to_pickle(net, filename):
    """
    Saves a pandapipes Network with the pickle library.

    :param net: The pandapipes Network to save.
    :type net:  pandapipesNet
    :param filename: The absolute or relative path to the output file or a writable file-like object
    :type filename: str, file-object
    :return: No output.

    :Example:
        >>> pandapipes.to_pickle(net, os.path.join("C:", "example_folder", "example1.p"))  # absolute path
        >>> pandapipes.to_pickle(net, "example2.p")  # relative path

    """
    if hasattr(filename, 'write'):
        pickle.dump(dict(net), filename, protocol=2)
        return
    if not filename.endswith(".p"):
        raise Exception("Please use .p to save pandapipes networks!")
    save_net = to_dict_with_coord_transform(net, ["junction_geodata"], ["pipe_geodata"])

    with open(filename, "wb") as f:
        pickle.dump(save_net, f, protocol=2)  # use protocol 2 for py2 / py3 compatibility


def to_json(net, filename=None):
    """
    Saves a pandapipes Network in JSON format. The index columns of all pandas DataFrames will be
    saved in ascending order. net elements which name begins with "_" (internal elements) will not
    be saved. Std types will also not be saved.

    :param net: The pandapipes Network to save.
    :type net: pandapipesNet
    :param filename: The absolute or relative path to the output file or a writable file-like \
            object. If None, a JSON string is returned.
    :type filename: str, file-object, default None
    :return: JSON string of the Network (only if filename is None)

    :Example:

        >>> pandapipes.to_json(net, "example.json")

    """
    if filename is None:
        return json.dumps(net, cls=PPJSONEncoder, indent=2, isinstance_func=isinstance_partial)
    if hasattr(filename, 'write'):
        json.dump(net, fp=filename, cls=PPJSONEncoder, indent=2, isinstance_func=isinstance_partial)
    else:
        with open(filename, "w") as fp:
            json.dump(net, fp=fp, cls=PPJSONEncoder, indent=2, isinstance_func=isinstance_partial)


def from_pickle(filename):
    """
    Load a pandapipes format Network from pickle file.

    :param filename: The absolute or relative path to the input file or file-like object
    :type filename: str, file-object
    :return: net - The pandapipes Network which was saved as pickle
    :rtype: pandapipesNet

    :Example:

        >>> net1 = pandapipes.from_pickle(os.path.join("C:", "example_folder", "example1.p"))
        >>> net2 = pandapipes.from_pickle("example2.p") #relative path

    """
    net = pandapipesNet(get_raw_data_from_pickle(filename))
    transform_net_with_df_and_geo(net, ["junction_geodata"], ["pipe_geodata"])
    return net


def from_json(filename):
    """
    Load a pandapipes network from a JSON file or string.
    The index of the returned network is not necessarily in the same order as the original network.
    Index columns of all pandas DataFrames are sorted in ascending order.

    :param filename: The absolute or relative path to the input file or file-like object
    :type filename: str, file-object
    :return: net - The pandapipes network that was saved as JSON
    :rtype: pandapipesNet

    :Example:

        >>> net = pandapipes.from_json("example.json")

    """
    if hasattr(filename, 'read'):
        net = json.load(filename, cls=PPJSONDecoder, object_hook=ppipes_hook)
    elif not os.path.isfile(filename):
        raise UserWarning("File {} does not exist!!".format(filename))
    else:
        with open(filename) as fp:
            net = json.load(fp, cls=PPJSONDecoder, object_hook=ppipes_hook)
    return net


def from_json_string(json_string):
    """
    Load a pandapipes network from a JSON string.
    The index of the returned network is not necessarily in the same order as the original network.
    Index columns of all pandas DataFrames are sorted in ascending order.

    :param json_string: The JSON string representation of the network
    :type json_string: str
    :return: net - The pandapipes network that was contained in the JSON string
    :rtype: pandapipesNet

    :Example:

        >>> net = pandapipes.from_json_string(json_str)

    """
    net = json.loads(json_string, cls=PPJSONDecoder, object_hook=ppipes_hook)
    return net
