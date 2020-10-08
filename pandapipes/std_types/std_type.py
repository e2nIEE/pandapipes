# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
from pandapipes import pp_dir
from pandapipes.std_types.std_type_toolbox import get_data, get_p_v_values, regression_function
from pandapower.io_utils import JSONSerializableClass

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class StdType(JSONSerializableClass):
    """

    """

    def __init__(self, name, type):
        """

        :param name: name of the standard type object
        :type name: str
        :param type: the specific standard type
        :type type: str
        """
        super(StdType, self).__init__()
        self.name = name
        self.type = type


class PumpStdType(StdType):

    def __init__(self, name, reg_par):
        """

        :param name: Name of the pump object
        :type name: str
        :param reg_par: If the parameters of a regression function are already determined they \
                can be directly be set by initializing a pump object
        :type reg_par: List of floats
        """
        super(PumpStdType, self).__init__(name, 'pump')
        self.reg_par = reg_par

    def get_pressure(self, vdot_m3_per_s):
        """
        Calculate the pressure lift based on a polynomial from a regression.

        It is ensured that the pressure lift is always >= 0. For reverse flows, bypassing is
        assumed.

        :param vdot_m3_per_s: Volume flow rate of a fluid in [m^3/s]. Abs() will be applied.
        :type vdot_m3_per_s: float
        :return: This function returns the corresponding pressure to the given volume flow rate \
                in [bar]
        :rtype: float
        """
        n = np.arange(len(self.reg_par), 0, -1)
        # no reverse flow - for vdot < 0, assume bypassing
        if vdot_m3_per_s < 0:
            logger.debug("Reverse flow observed in a %s pump. "
                         "Bypassing without pressure change is assumed" % str(self.name))
            return 0
        # no negative pressure lift - bypassing always allowed:
        # /1 to ensure float format:
        p = max(0, sum(self.reg_par * (vdot_m3_per_s/1 * 3600) ** (n - 1)))
        return p

    @classmethod
    def from_path(cls, name, path):
        """

        :param name: Name of the pump object
        :type name: str
        :param path: Path where the CSV file, defining a pump object, is stored
        :type path: str
        :return: An object of the pump standard type class
        :rtype: PumpStdType
        """
        p_values, v_values, degree = get_p_v_values(path)
        reg_par = regression_function(p_values, v_values, degree)
        return cls(name, reg_par)

    @classmethod
    def from_list(cls, name, p_values, v_values, degree):
        reg_par = regression_function(p_values, v_values, degree)
        return cls(name, reg_par)


def add_basic_std_types(net):
    """

    :param net: pandapipes network in which the standard types should be added
    :type net: pandapipesNet

    """
    pump_files = os.listdir(os.path.join(pp_dir, 'std_types', 'library', 'Pump'))
    for pump_file in pump_files:
        pump_name = str(pump_file.split('.')[0])
        pump = PumpStdType.from_path(pump_name, os.path.join(pp_dir, 'std_types', 'library', 'Pump',
                                                             pump_file))
        add_pump_std_type(net, pump_name, pump, True)

    pipe_file = os.path.join(pp_dir, 'std_types', 'library', 'Pipe.csv')
    data = get_data(pipe_file, 'pipe').to_dict()
    add_std_types(net, 'pipe', data, True)


def load_std_type(net, name, element):
    """
    Loads standard type data from the data base. Issues a warning if
    stdtype is unknown.

    INPUT:
        **net** - The pandapipes network

        **name** - name of the standard type as string

        **element** - "pipe"

    OUTPUT:
        **typedata** - dictionary containing type data
    """
    library = net.std_type[element]
    if name in library:
        return library[name]
    else:
        raise UserWarning("Unknown standard %s type %s" % (element, name))


def add_pump_std_type(net, name, pump_object, overwrite=False):
    """

    :param net:
    :type net:
    :param name:
    :type name:
    :param pump_object:
    :type pump_object:
    :param overwrite:
    :type overwrite:
    :return:
    :rtype:
    """
    if not isinstance(pump_object, PumpStdType):
        raise ValueError('pump needs to be of Pump_Std_Type')

    add_std_type(net, 'pump', name, pump_object, overwrite)


def add_std_type(net, std_type_category, component_name, component_object, overwrite=False):
    """

    :param net:
    :type net:
    :param std_type_category:
    :type std_type_category:
    :param component_name:
    :type component_name:
    :param component_object:
    :type component_object:
    :param overwrite:
    :type overwrite:
    :return:
    :rtype:
    """
    if not 'std_type' in net:
        net.update({'std_type': {std_type_category: {component_name: component_object}}})
    elif not std_type_category in net.std_type:
        std_types = net.std_type
        std_types.update({std_type_category: {component_name: component_object}})
        net.std_type = std_types
    elif not overwrite and component_name in net['std_type'][std_type_category]:
        raise (ValueError(
            '%s is already in net.std_type["%s"]. Set overwrite = True if you want to change values!'
            % (component_name, std_type_category)))
    else:
        std_types = net.std_type[std_type_category]
        std_types.update({component_name: component_object})
        net.std_type[std_type_category] = std_types


def add_std_types(net, std_type_category, component_dictionary, overwrite=False):
    """

    :param net:
    :type net:
    :param std_type_category:
    :type std_type_category:
    :param component_name:
    :type component_name:
    :param component_object:
    :type component_object:
    :param overwrite:
    :type overwrite:
    :return:
    :rtype:
    """
    if not 'std_type' in net:
        net.update({'std_type': {std_type_category: component_dictionary}})
    elif not std_type_category in net.std_type:
        std_types = net.std_type
        std_types.update({std_type_category: component_dictionary})
        net.std_type = std_types
    elif not overwrite:
        std_types = net.std_type[std_type_category]
        std_types.update({k: v for k, v in component_dictionary.items() if k not in std_types})
        net.std_type[std_type_category] = std_types
    else:
        net.std_type[std_type_category].update(component_dictionary)
