# Copyright (c) 2020-2021 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from pandapipes import logger
from pandapower.io_utils import JSONSerializableClass
from scipy.interpolate import interp2d

try:
    import plotly.graph_objects as go
    PLOTLY_INSTALLED = True
except ImportError:
    PLOTLY_INSTALLED = False


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


class DynPumpStdType(RegressionStdType):
    def __init__(self, name, interp2d_fct):
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
        super(DynPumpStdType, self).__init__(name, 'dynamic_pump', interp2d_fct)
        self.interp2d_fct = interp2d_fct
        # y_values = self._head_list = None
        # x_values = self._flowrate_list = None
        self._speed_list = None
        self._efficiency = None
        self._individual_curves = None


    def get_m_head(self, vdot_m3_per_s, speed):
        """
        Calculate the head (m) lift based on 2D linear interpolation function.

        It is ensured that the head (m) lift is always >= 0. For reverse flows, bypassing is
        assumed.

        :param vdot_m3_per_s: Volume flow rate of a fluid in [m^3/s]. Abs() will be applied.
        :type vdot_m3_per_s: float
        :return: This function returns the corresponding pressure to the given volume flow rate \
                in [bar]
        :rtype: float
        """
        # no reverse flow - for vdot < 0, assume bypassing
        if vdot_m3_per_s < 0:
            logger.debug("Reverse flow observed in a %s pump. "
                         "Bypassing without pressure change is assumed" % str(self.name))
            return 0
        # no negative pressure lift - bypassing always allowed:
        # /1 to ensure float format:
        m_head = self.interp2d_fct((vdot_m3_per_s / 1 * 3600), speed)
        return m_head

    def plot_pump_curve(self):
        if not PLOTLY_INSTALLED:
            logger.error("You need to install plotly to plot the pump curve.")
        fig = go.Figure(go.Surface(
            contours={
                "x": {"show": True, "start": 1.5, "end": 2, "size": 0.04, "color": "white"},
                "z": {"show": True, "start": 0.5, "end": 0.8, "size": 0.05}
            },
            x=self._x_values,
            y=self._speed_list,
            z=self._y_values))
        fig.update_xaxes = 'flow'
        fig.update_yaxes = 'speed'
        fig.update_layout(scene=dict(
            xaxis_title='x: Flow (m3/h)',
            yaxis_title='y: Speed (%)',
            zaxis_title='z: Head (m)'),
            title='Pump Curve', autosize=False,
            width=400, height=400,
        )
        return fig


    @classmethod
    def from_folder(cls, name, dyn_path):
        pump_st = None
        individual_curves = {}
        # Compile dictionary of dataframes from file path
        x_flow_max = 0
        speed_list = []
        degree = []

        for file_name in os.listdir(dyn_path):
            key_name = file_name[0:file_name.find('.')]
            individual_curves[key_name] = get_data(os.path.join(dyn_path, file_name), 'pump')
            speed_list.append(individual_curves[key_name].speed_pct[0])

            if max(individual_curves[key_name].Vdot_m3ph) > x_flow_max:
                x_flow_max = max(individual_curves[key_name].Vdot_m3ph)

        if individual_curves:
            flow_list = np.linspace(0, x_flow_max, 10)
            head_list = np.zeros([len(speed_list), len(flow_list)])

            for idx, key in enumerate(individual_curves):
                # create individual poly equations for each curve and append to (z)_head_list
                reg_par = np.polyfit(individual_curves[key].Vdot_m3ph.values, individual_curves[key].Head_m.values,
                                     individual_curves[key].degree.values[0])
                degree.append(individual_curves[key].degree.values[0])
                n = np.arange(len(reg_par), 0, -1)
                head_list[idx::] = [max(0, sum(reg_par * x ** (n - 1))) for x in flow_list]

            # Sorting the speed and head list results:
            head_list_sorted = np.zeros([len(speed_list), len(flow_list)])
            speed_sorted = sorted(speed_list)
            for idx_s, val_s in enumerate(speed_sorted):
                for idx_us, val_us in enumerate(speed_list):
                    if val_s == val_us:  # find sorted value in unsorted list
                        head_list_sorted[idx_s, :] = head_list[idx_us, :]


            # interpolate 2d function to determine head (m) from specified flow and speed variables
            if min(degree) == 0:
                interpolate_kind = 'linear'
            else:
                interpolate_kind = 'cubic'

            interp2d_fct = interp2d(flow_list, speed_list, head_list, kind=interpolate_kind, fill_value='0')

            pump_st = cls(name, interp2d_fct)
            pump_st._x_values = flow_list
            pump_st._speed_list = speed_sorted # speed_list
            pump_st._y_values = head_list_sorted # head_list
            pump_st._individual_curves = individual_curves

        return pump_st

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

class ValveStdType(RegressionStdType):
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
        super(ValveStdType, self).__init__(name, 'dynamic_valve', reg_par)

    def get_relative_flow(self, relative_travel):
        """
        Calculate the pressure lift based on a polynomial from a regression.

        It is ensured that the pressure lift is always >= 0. For reverse flows, bypassing is
        assumed.

        :param relative_travel: Relative valve travel (opening).
        :type relative_travel: float
        :return: This function returns the corresponding relative flow coefficient to the given valve travel
        :rtype: float
        """
        if self._reg_polynomial_degree == 0:
            # Compute linear interpolation of the std type
            f = np.interp(relative_travel, self._x_values, self._y_values)
            return f

        else:
            n = np.arange(len(self.reg_par), 0, -1)
            if relative_travel < 0:
                logger.debug("Issue with dynamic valve travel dimensions."
                             "Issue here" % str(self.name))
                return 0
            # no negative pressure lift - bypassing always allowed:
            # /1 to ensure float format:
            f = max(0, sum(self.reg_par * relative_travel ** (n - 1)))

            return f
    @classmethod
    def from_path(cls, name, path):
        reg_par, x_values, y_values, degree = cls._from_path(path)
        reg_st = cls(name, reg_par)
        cls.init_std_type(reg_st, x_values, y_values, degree)
        return reg_st

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
            results[mask] = \
                np.where(mask, np.sum(self.reg_par * (vdot_m3_per_s[mask][:, None] * 3600) ** (n - 1), axis=1), 0)
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
        pump_st = cls(name, reg_par)
        cls.init_std_type(pump_st, x_values, y_values, degree)
        return pump_st

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
    elif std_type_category == 'dynamic_valve':
        return ValveStdType.load_data(path)
    elif std_type_category == 'pipe':
        return pd.read_csv(path, sep=';', index_col=0).T
    else:
        raise AttributeError('std_type_category %s not implemented yet' % std_type_category)
