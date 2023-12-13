# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os
import warnings
import re

import pandas as pd
from pandapipes import pp_dir
from pandapipes.std_types.std_type_class import get_data, PumpStdType

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def create_std_type(net, component, std_type_name, typedata, overwrite=False, check_required=True):
    """
    Create a new standard type for a specific component with the given data.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param component: type of component ("pipe" or "pump")
    :type component: str
    :param std_type_name: name of the standard type as string
    :type std_type_name: str
    :param typedata: dictionary containing type data or standard type object
    :type typedata: dict, StdType
    :param overwrite: if True, overwrites the standard type if it already exists in the net
    :type overwrite: bool, default False
    :param check_required: if True, checks for required std_type entries
    :type check_required: bool, default True
    """
    if check_required:
        if component == "pipe":
            required = ["inner_diameter_mm"]
        else:
            if component in ["pump"]:
                required = []
            else:
                raise ValueError("Unkown component type %s" % component)
        for par in required:
            if par not in typedata:
                raise UserWarning("%s is required as %s type parameter" % (par, component))
    if "std_types" not in net:
        net.update({"std_types": {component: {std_type_name: typedata}}})
    elif component not in net.std_types:
        std_types = net.std_types
        std_types.update({component: {std_type_name: typedata}})
        net.std_types = std_types
    elif not overwrite and std_type_name in net["std_types"][component]:
        raise (ValueError(
            "%s is already in net.std_types['%s']. Set overwrite=True if you want to change values!"
            % (std_type_name, component)))
    else:
        std_types = net.std_types[component]
        std_types.update({std_type_name: typedata})
        net.std_types[component] = std_types


def create_std_types(net, component, type_dict, overwrite=False):
    """
    Create several new standard types for a specific component with the given data.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param component: type of component ("pipe" or "pump")
    :type component: str
    :param type_dict: dictionary containing type data with names of the standard types as keys
    :type type_dict: dict
    :param overwrite: if True, overwrites standard types that already exist in the net
    :type overwrite: bool, default False
    """
    for std_type_name, typedata in type_dict.items():
        create_std_type(net, component, std_type_name, typedata, overwrite)


def copy_std_types(to_net, from_net, component, overwrite=False):
    """
    Transfers all standard types of one network to another.

    :param to_net: The pandapipes network to which the standard types are copied
    :type to_net: pandapipesNet
    :param from_net: The pandapipes network from which the standard types are taken
    :type from_net: pandapipesNet
    :param component: "pipe" or "pump"
    :type component: str
    :param overwrite: if True, overwrites standard types which already exist in to_net
    :type overwrite: bool, default True
    """
    for name, typdata in from_net.std_types[component].items():
        if overwrite is False:
            try:
                create_std_type(to_net, component, name, typdata, overwrite=overwrite)
            except ValueError as e:
                if re.search("Set overwrite=True if you want to change values", str(e)) is None:
                    raise e
        else:
            create_std_type(to_net, component, name, typdata, overwrite=overwrite)


def load_std_type(net, name, component):
    """
    Loads standard type data from the data base. Issues a warning if
    stdtype is unknown.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param name: name of the standard type as string
    :type name: str
    :param component: type of component ("pipe" or "pump")
    :type component: str
    :return: typedata - dictionary containing type data
    :rtype: dict
    """
    if component not in net.std_types:
        raise UserWarning("Unknown std_type component %s" % component)
    library = net.std_types[component]
    if name not in library:
        raise UserWarning("Unknown standard %s type %s" % (component, name))
    return library[name]


def std_type_exists(net, name, component):
    """
    Checks if a standard type exists.

    :param net: The pandapipes network
    :type net: pandapipesNet
    :param name: name of the standard type as string
    :type name: str
    :param component: type of component ("pipe" or "pump")
    :type component: str
    :return: exists - True if standard type exists, False otherwise
    :rtype: bool
    """
    library = net.std_types[component]
    return name in library


def delete_std_type(net, name, component):
    """
    Deletes standard type parameters from database.

    :param net: pandapipes Network
    :type net: pandapipesNet
    :param name: name of the standard type as string
    :type name: str
    :param component: type of component ("pipe" or "pump")
    :type component: str
    """
    library = net.std_types[component]
    if name in library:
        del library[name]
    else:
        raise UserWarning("Unknown standard %s type %s" % (component, name))


def available_std_types(net, component):
    """
    Returns all standard types available for this network as a table.

    :param net: pandapipes Network
    :type net: pandapipesNet
    :param component: type of component ("pipe" or "pump")
    :type component: str
    :return: typedata - table of standard type parameters
    :rtype: pd.DataFrame
    """
    if component == "pump":
        std_types = pd.Series(net.std_types[component])
    else:
        std_types = pd.DataFrame(net.std_types[component]).T
    try:
        return std_types.infer_objects()
    except AttributeError:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return std_types.convert_objects()


def change_std_type(net, cid, name, component):
    """
    Changes the type of a given component in pandapower. Changes only parameter that are given
    for the type.

    :param net: pandapipes network
    :type net: pandapipesNet
    :param cid: component index (either pipe or pump index)
    :type cid: int
    :param name: name of the new standard type
    :type name: str
    :param component: type of component ("pipe" or "pump")
    :type component: str
    :return:
    :rtype:
    """
    type_param = load_std_type(net, name, component)
    table = net[component]
    for column in table.columns:
        if column in type_param:
            table.at[cid, column] = type_param[column]
    table.at[cid, "std_type"] = name


def create_pump_std_type(net, name, pump_object, overwrite=False):
    """
    Create a new pump stdandard type object and add it to the pump standard types in net.

    :param net: The pandapipes network to which the standard type is added.
    :type net: pandapipesNet
    :param name: name of the created standard type
    :type name: str
    :param pump_object: pump standard type object
    :type pump_object: PumpStdType
    :param overwrite: if True, overwrites the standard type if it already exists in the net
    :type overwrite: bool, default False
    :return:
    :rtype:
    """
    if not isinstance(pump_object, PumpStdType):
        raise ValueError("pump needs to be of PumpStdType")

    create_std_type(net, "pump", name, pump_object, overwrite)


def add_basic_std_types(net):
    """

    :param net: pandapipes network in which the standard types should be added
    :type net: pandapipesNet

    """
    pump_files = os.listdir(os.path.join(pp_dir, "std_types", "library", "Pump"))
    for pump_file in pump_files:
        pump_name = str(pump_file.split(".")[0])
        pump = PumpStdType.from_path(pump_name, os.path.join(pp_dir, "std_types", "library", "Pump",
                                                             pump_file))
        create_pump_std_type(net, pump_name, pump, True)

    pipe_file = os.path.join(pp_dir, "std_types", "library", "Pipe.csv")
    data = get_data(pipe_file, "pipe").to_dict()
    create_std_types(net, "pipe", data, True)
