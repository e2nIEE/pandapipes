# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

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


class InterpolationStdType(StdType):

    def __init__(self, name, component, int_fct):
        """
        The interpolation standrad type object interpolates and extrapolates between the given values

        :param name: Name of the interpolation standard type object
        :type name: str
        :param component: The specific interpolation standard type
        :type component: str
        :param int_fct: Defines the interpolation function
        :type int_fct: fct
        :return: An object of the interpolation standard type class
        :rtype: InterpolationStdType
        """
        super(InterpolationStdType, self).__init__(name, component)
        self.int_fct = int_fct
        self._x_list = None
        self._y_list = None
        self._fill_value = 'extrapolate'

    def update_std_type(self, x_list, y_list, fill_value='extrapolate'):
        int_fct = interpolation_function(x_list, y_list, fill_value)
        self.int_fct = int_fct
        self._x_list = x_list
        self._y_list = y_list
        self._fill_value = fill_value

    @classmethod
    def init_std_type(cls, int_st, x_values, y_values, fill_value='extrapolate'):
        int_st._x_list = x_values
        int_st._y_list = y_values
        int_st._fill_value = fill_value

    @classmethod
    def from_path(cls, name, path):
        raise NotImplementedError

    @classmethod
    def from_list(cls, name, x_values, y_values, fill_value='extrapolate'):
        raise NotImplementedError

    @classmethod
    def _from_path(cls, path):
        data = cls.load_data(path)
        x_values, y_values = _retrieve_data(data)
        int_fct = interpolation_function(x_values, y_values)
        return int_fct, x_values, y_values

    @classmethod
    def _from_list(cls, x_values, y_values, fill_value='extrapolate'):
        int_fct = interpolation_function(x_values, y_values, fill_value)
        return int_fct, x_values, y_values, fill_value

    @classmethod
    def load_data(cls, path):
        raise NotImplementedError


class RegressionStdType(StdType):

    def __init__(self, name, component, reg_par):
        """
        The regression standrad type object creates a regression based on the given data and regression parameters

        :param name: Name of the regression object
        :type name: str
        :param component: The specific regression standard type
        :type component: str
        :param reg_par: The determined regression parameters based on the given data and polynominal degree
        :type reg_par: list
        :return: An object of the regression standard type class
        :rtype: RegressionStdType
        """

        super(RegressionStdType, self).__init__(name, component)
        self.reg_par = reg_par
        self._x_values = None
        self._y_values = None
        self._reg_polynomial_degree = 2

    def update_std_type(self, x_values, y_values, reg_polynomial_degree):
        reg_par = regression_function(x_values, y_values, reg_polynomial_degree)
        self.reg_par = reg_par
        self._x_values = x_values
        self._y_values = y_values
        self._reg_polynomial_degree = reg_polynomial_degree

    @classmethod
    def init_std_type(cls, reg_st, x_values, y_values, degree):
        reg_st._x_values = x_values
        reg_st._y_values = y_values
        reg_st._reg_polynomial_degree = degree

    @classmethod
    def from_path(cls, name, path):
        raise NotImplementedError

    @classmethod
    def from_list(cls, name, x_values, y_values, degree):
        raise NotImplementedError

    @classmethod
    def _from_path(cls, path):
        data = cls.load_data(path)
        x_values, y_values, degree = _retrieve_data(data)
        reg_par = regression_function(x_values, y_values, degree[0])
        return reg_par, x_values, y_values, degree[0]

    @classmethod
    def _from_list(cls, x_values, y_values, degree):
        reg_par = regression_function(x_values, y_values, degree)
        return reg_par, x_values, y_values, degree

    @classmethod
    def load_data(cls, path):
        raise NotImplementedError


class PumpStdType(RegressionStdType):

    def __init__(self, name, reg_par):
        """
        Creates a concrete pump std type. The class is a child class of the RegressionStdType, therefore, the here
        derived values are calculated based on a previously performed regression. The regression parameters need to
        be passed or alternatively, can be determined through the here defined class methods.

        :param name: Name of the pump object
        :type name: str
        :param reg_par: If the parameters of a regression function are already determined they \
                can be directly be set by initializing a pump object
        :type reg_par: List of floats
        """
        super(PumpStdType, self).__init__(name, 'pump', reg_par)

    def get_pressure(self, vdot_m3_per_s):
        """
        Calculate the pressure lift based on a polynomial from a regression.

        It is ensured that the pressure lift is always >= 0. For reverse flows, bypassing is
        assumed.

        :param vdot_m3_per_s: Volume flow rate of a fluid in [m^3/s].
        :type vdot_m3_per_s: float, array-like
        :return: This function returns the corresponding pressure to the given volume flow rate \
                in [bar]
        :rtype: float
        """
        # no reverse flow - for vdot < 0, assume bypassing
        n = np.arange(len(self.reg_par), 0, -1)
        if np.iterable(vdot_m3_per_s):
            results = np.zeros(len(vdot_m3_per_s), dtype=float)
            if any(vdot_m3_per_s < 0):
                logger.debug("Reverse flow observed in a %s pump. "
                             "Bypassing without pressure change is assumed" % str(self.name))
            mask = vdot_m3_per_s >= 0
            # no negative pressure lift - bypassing always allowed:
            results[mask] = np.where(mask, sum(self.reg_par * (vdot_m3_per_s[mask] * 3600) ** (n - 1)), 0)
        else:
            if vdot_m3_per_s < 0:
                logger.debug("Reverse flow observed in a %s pump. "
                             "Bypassing without pressure change is assumed" % str(self.name))
                results = 0
            else:
                results = max(0, sum(self.reg_par * (vdot_m3_per_s * 3600) ** (n - 1)))
        return results

    @classmethod
    def from_path(cls, name, path):
        reg_par, x_values, y_values, degree = cls._from_path(path)
        reg_st = cls(name, reg_par)
        cls.init_std_type(reg_st, x_values, y_values, degree)
        return reg_st

    @classmethod
    def from_list(cls, name, x_values, y_values, degree):
        reg_par, x_values, y_values, degree = cls._from_list(x_values, y_values, degree)
        reg_st = cls(name, reg_par)
        cls.init_std_type(reg_st, x_values, y_values, degree)

    @classmethod
    def load_data(cls, path):
        """
        load_data.

        :param path:
        :type path:
        :return:
        :rtype:
        """
        path = os.path.join(path)
        data = pd.read_csv(path, sep=';', dtype=np.float64)
        return data


def regression_function(x_values, y_values, degree):
    """
    Regression function: performs a regression based on the given x-, y-values and the polynominal degree.

    :param x_values: given data on x-axis
    :type x_values: array_like
    :param y_values: given data on y-axis
    :type y_values: array_like
    :param degree: polynominal degree
    :type degree: int
    :return: polynominal coefficients
    :rtype: ndarray
    """
    if not int(degree) == degree:
        raise UserWarning("The polynomial degree has to be an integer, but %s was given. "
                          "It will be rounded down now." % str(degree))
    return np.polyfit(x_values, y_values, degree)


def interpolation_function(x_values, y_values, fill_value='extrapolate'):
    """
    interpolation function: performs an interpolation based on the given x- and y-values.

    :param x_values: given data on x-axis
    :type x_values: array_like
    :param y_values: given data on y-axis
    :type y_values: array_like
    :param degree: how to handle missing data
    :type degree: str
    :return: interpolation function
    :rtype: fct
    """
    return interp1d(x_values, y_values, fill_value=fill_value)


def _retrieve_data(loaded_data):
    data_list = [loaded_data.values[:, i] for i in range(np.shape(loaded_data)[1])]
    return data_list

def get_data(path, std_type_category):
    """
    retrieve data

    :param path: path the data can be retrieved from
    :type path: str
    :param std_type_category:
    :type std_type_category:
    :return: data in a pd.DataFrame
    :rtype: pd.DataFrame
    """
    if std_type_category == 'pump':
        return PumpStdType.load_data(path)
    elif std_type_category == 'pipe':
        return pd.read_csv(path, sep=';', index_col=0).T
    else:
        raise AttributeError('std_type_category %s not implemented yet' % std_type_category)
