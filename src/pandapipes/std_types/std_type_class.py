# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from pandapipes import logger
from pandapower.io_utils import JSONSerializableClass
from pandapipes.pandapipes_net import Sector



class StdType(JSONSerializableClass):
    """Base class for a standard type object."""

    def __init__(self, name, component, sector=Sector.ALL):
        """Initialize a standard type object.

        :param name: name of the standard type object
        :type name: str
        :param component: the specific standard type
        :type component: str
        :param sector: sector the Std Type is assigned to
        :type sector: Sector, default Sector.ALL
        """
        super(StdType, self).__init__()
        self.name = name
        self.component = component
        self.sector = sector

    @classmethod
    def from_dict(cls, d):
        obj = super().from_dict(d)
        if hasattr(obj, "type") and not hasattr(obj, "component"):
            setattr(obj, "component", getattr(obj, "type"))
            delattr(obj, "type")
        return obj


class InterpolationStdType(StdType):

    def __init__(self, name, component, int_fct, sector=Sector.ALL):
        """The interpolation standrad type object interpolates and extrapolates between the given values.

        :param name: Name of the interpolation standard type object
        :type name: str
        :param component: The specific interpolation standard type
        :type component: str
        :param int_fct: Defines the interpolation function
        :type int_fct: fct
        :param sector: sector the Std Type is assigned to
        :type sector: Sector, default Sector.ALL
        :return: An object of the interpolation standard type class
        :rtype: InterpolationStdType
        """
        super(InterpolationStdType, self).__init__(name, component, sector)
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

    def __init__(self, name, component, reg_par, sector=Sector.ALL):
        """The regression standrad type object creates a regression based on the given data and regression parameters.

        :param name: Name of the regression object
        :type name: str
        :param component: The specific regression standard type
        :type component: str
        :param reg_par: The determined regression parameters based on the given data and polynominal degree
        :type reg_par: list
        :param sector: sector the Std Type is assigned to
        :type sector: Sector, default Sector.ALL
        :return: An object of the regression standard type class
        :rtype: RegressionStdType
        """
        super(RegressionStdType, self).__init__(name, component, sector)
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
        x_values, y_values, degree, sector = _retrieve_data(data)
        reg_par = regression_function(x_values.astype(float), y_values.astype(float), float(degree[0]))
        return reg_par, x_values, y_values, degree[0], sector[0]

    @classmethod
    def _from_list(cls, x_values, y_values, degree):
        reg_par = regression_function(x_values, y_values, degree)
        return reg_par, x_values, y_values, degree

    @classmethod
    def load_data(cls, path):
        raise NotImplementedError


class PumpStdType(RegressionStdType):

    def __init__(self, name, reg_par, sector=Sector.ALL):
        """Create a concrete pump std type.

        The class is a child class of the RegressionStdType, therefore, the here
        derived values are calculated based on a previously performed regression. The regression parameters need to
        be passed or alternatively, can be determined through the here defined class methods.

        :param name: Name of the pump object
        :type name: str
        :param sector: sector the pump is assigned to
        :type sector: Sector, default Sector.ALL
        :param reg_par: If the parameters of a regression function are already determined they \
                can be directly be set by initializing a pump object
        :type reg_par: List of floats
        """
        super(PumpStdType, self).__init__(name, 'pump', reg_par, sector)

    def get_pressure(self, vdot_m3_per_s):
        """Calculate the pressure lift based on a polynomial from a regression.

        It is ensured that the pressure lift is always >= 0. For reverse flows, bypassing is
        assumed.

        :param vdot_m3_per_s: Volume flow rate of a fluid in [m^3/s].
        :type vdot_m3_per_s: float, array-like
        :return: This function returns the corresponding pressure to the given volume flow rate \
                in [bar]
        :rtype: float
        """
        vdot, lift, active = self._evaluate_curve(vdot_m3_per_s)
        results = np.where(active, lift, 0.)
        return results if np.iterable(vdot_m3_per_s) else float(results)

    def get_pressure_derivative(self, vdot_m3_per_s):
        """Calculate the derivative of the pressure lift with respect to the volume flow rate.

        Consistent with get_pressure: the derivative is 0 wherever get_pressure is cut off, i.e.
        for reverse flows (bypassing) and where the polynomial would yield a negative lift.

        :param vdot_m3_per_s: Volume flow rate of a fluid in [m^3/s].
        :type vdot_m3_per_s: float, array-like
        :return: The derivative of the pressure lift in [bar / (m^3/s)]
        :rtype: float, array-like
        """
        vdot, _, active = self._evaluate_curve(vdot_m3_per_s)
        # the polynomial is defined in m^3/h
        derivative = 3600 * np.polyval(np.polyder(self.reg_par), np.maximum(vdot, 0) * 3600)
        results = np.where(active, derivative, 0.)
        return results if np.iterable(vdot_m3_per_s) else float(results)

    def _evaluate_curve(self, vdot_m3_per_s):
        """Evaluate the regression polynomial and mark where the pump actually lifts pressure.

        For vdot < 0, bypassing without pressure change is assumed; a negative lift from the
        polynomial is cut off as well.

        :param vdot_m3_per_s: Volume flow rate of a fluid in [m^3/s].
        :type vdot_m3_per_s: float, array-like
        :return: (vdot, lift, active) - volume flow as array, uncut polynomial lift in [bar] and
                 the mask where the lift is applied
        :rtype: tuple(np.ndarray)
        """
        vdot = np.asarray(vdot_m3_per_s, dtype=float)
        if np.any(vdot < 0):
            logger.debug("Reverse flow observed in a %s pump. "
                         "Bypassing without pressure change is assumed" % str(self.name))
        # the polynomial is defined in m^3/h
        lift = np.polyval(self.reg_par, np.maximum(vdot, 0) * 3600)
        return vdot, lift, (vdot >= 0) & (lift > 0)

    @classmethod
    def from_path(cls, name, path):
        reg_par, x_values, y_values, degree, sector = cls._from_path(path)
        reg_st = cls(name, reg_par, sector)
        cls.init_std_type(reg_st, x_values, y_values, degree)
        return reg_st

    @classmethod
    def from_list(cls, name, x_values, y_values, degree, sector=Sector.ALL):
        reg_par, x_values, y_values, degree = cls._from_list(x_values, y_values, degree)
        pump_st = cls(name, reg_par, sector)
        cls.init_std_type(pump_st, x_values, y_values, degree)
        return pump_st

    @classmethod
    def load_data(cls, path):
        """load_data.

        :param path:
        :type path:
        :return:
        :rtype:
        """
        path = os.path.join(path)
        data = pd.read_csv(path, sep=';')
        return data


def regression_function(x_values, y_values, degree):
    """Regression function: performs a regression based on the given x-, y-values and the polynominal degree.

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
    """interpolation function: performs an interpolation based on the given x- and y-values.

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
    """retrieve data.

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
