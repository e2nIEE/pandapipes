# Copyright (c) 2020-2021 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os
import pandas as pd
import numpy as np
from pandapipes import logger
from pandapower.io_utils import JSONSerializableClass


class StdType(JSONSerializableClass):
    """

    """

    def __init__(self, name, component):
        """

        :param name: name of the standard type object
        :type name: str
        :param component: the specific standard type
        :type component: str
        """
        super(StdType, self).__init__()
        self.name = name
        self.component = component

    @classmethod
    def from_dict(cls, d):
        obj = super().from_dict(d)
        if hasattr(obj, "type") and not hasattr(obj, "component"):
            setattr(obj, "component", getattr(obj, "type"))
            delattr(obj, "type")
        return obj


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
        self._pressure_list = None
        self._flowrate_list = None
        self._reg_polynomial_degree = 2

    def update_pump(self, pressure_list, flowrate_list, reg_polynomial_degree):
        reg_par = regression_function(pressure_list, flowrate_list, reg_polynomial_degree)
        self.reg_par = reg_par
        self._pressure_list = pressure_list
        self._flowrate_list = flowrate_list
        self._reg_polynomial_degree = reg_polynomial_degree

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
        p = max(0, sum(self.reg_par * (vdot_m3_per_s / 1 * 3600) ** (n - 1)))
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
        pump_st = cls(name, reg_par)
        pump_st._pressure_list = p_values
        pump_st._flowrate_list = v_values
        pump_st._reg_polynomial_degree = degree
        return pump_st

    @classmethod
    def from_list(cls, name, p_values, v_values, degree):
        reg_par = regression_function(p_values, v_values, degree)
        pump_st = cls(name, reg_par)
        pump_st._pressure_list = p_values
        pump_st._flowrate_list = v_values
        pump_st._reg_polynomial_degree = degree
        return pump_st


def get_data(path, std_type_category):
    """
    get_data.

    :param path:
    :type path:
    :param std_type_category:
    :type std_type_category:
    :return:
    :rtype:
    """
    if std_type_category == 'pump':
        path = os.path.join(path)
        data = pd.read_csv(path, sep=';', dtype=np.float64)
    elif std_type_category == 'pipe':
        data = pd.read_csv(path, sep=';', index_col=0).T
    else:
        raise AttributeError('std_type_category %s not implemented yet' % std_type_category)
    return data


def get_p_v_values(path):
    """

    :param path:
    :type path:
    :return:
    :rtype:
    """
    data = get_data(path, 'pump')
    p_values = data.values[:, 0]
    v_values = data.values[:, 1]
    degree = data.values[0, 2]
    return p_values, v_values, degree


def regression_function(p_values, v_values, degree):
    """
    Regression function...

    :param p_values:
    :type p_values:
    :param v_values:
    :type v_values:
    :param degree:
    :type degree:
    :return:
    :rtype:
    """
    if not int(degree) == degree:
        raise UserWarning("The polynomial degree has to be an integer, but %s was given. "
                          "It will be rounded down now." % str(degree))
    z = np.polyfit(v_values, p_values, degree)
    reg_par = z
    return reg_par
