# Copyright (c) 2020-2021 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os
import pandas as pd
import numpy as np
from pandapipes import logger
from pandapower.io_utils import JSONSerializableClass
from scipy.interpolate import interp2d
import plotly.graph_objects as go

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


class DynPumpStdType(StdType):

    def __init__(self, name, interp2d_fct):
        """

        :param name: Name of the pump object
        :type name: str
        :param reg_par: If the parameters of a regression function are already determined they \
                can be directly be set by initializing a pump object
        :type reg_par: List of floats
        """
        super(DynPumpStdType, self).__init__(name, 'dynamic_pump')
        self.interp2d_fct = interp2d_fct
        self._head_list = None
        self._flowrate_list = None
        self._speed_list = None
        self._efficiency = None
        self._individual_curves = None
        self._reg_polynomial_degree = 2

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

        fig = go.Figure(go.Surface(
            contours={
                "x": {"show": True, "start": 1.5, "end": 2, "size": 0.04, "color": "white"},
                "z": {"show": True, "start": 0.5, "end": 0.8, "size": 0.05}
            },
            x=self._flowrate_list,
            y=self._speed_list,
            z=self._head_list))
        fig.update_xaxes = 'flow'
        fig.update_yaxes = 'speed'
        fig.update_layout(scene=dict(
            xaxis_title='x: Flow (m3/h)',
            yaxis_title='y: Speed (%)',
            zaxis_title='z: Head (m)'),
            title='Pump Curve', autosize=False,
            width=1000, height=1000,
        )
        #fig.show()

        return fig #self._flowrate_list, self._speed_list, self._head_list


    @classmethod
    def from_folder(cls, name, dyn_path):
        pump_st = None
        individual_curves = {}
        # Compile dictionary of dataframes from file path
        x_flow_max = 0
        speed_list = []

        for file_name in os.listdir(dyn_path):
            key_name = file_name[0:file_name.find('.')]
            individual_curves[key_name] = get_data(os.path.join(dyn_path, file_name), 'dyn_pump')
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
            interp2d_fct = interp2d(flow_list, speed_list, head_list, kind='cubic', fill_value='0')

            pump_st = cls(name, interp2d_fct)
            pump_st._flowrate_list = flow_list
            pump_st._speed_list = speed_sorted # speed_list
            pump_st._head_list = head_list_sorted # head_list
            pump_st._individual_curves = individual_curves

        return pump_st




class ValveStdType(StdType):
    def __init__(self, name, reg_par):
        """

        :param name: Name of the dynamic valve object
        :type name: str
        :param reg_par: If the parameters of a regression function are already determined they \
                can be directly be set by initializing a valve object
        :type reg_par: List of floats
        """
        super(ValveStdType, self).__init__(name, 'dynamic_valve')
        self.reg_par = reg_par
        self._relative_flow_list = None
        self._relative_travel_list = None
        self._reg_polynomial_degree = 2

    def update_valve(self, travel_list, flow_list, reg_polynomial_degree):
        reg_par = regression_function(travel_list, flow_list, reg_polynomial_degree)
        self.reg_par = reg_par
        self._relative_travel_list = travel_list
        self._relative_flow_list = flow_list
        self._reg_polynomial_degree = reg_polynomial_degree

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
            # Compute linear interpolation
            f = linear_interpolation(relative_travel, self._relative_travel_list, self._relative_flow_list)
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
        """

        :param name: Name of the valve object
        :type name: str
        :param path: Path where the CSV file, defining a valve object, is stored
        :type path: str
        :return: An object of the valve standard type class
        :rtype: ValveStdType
        """
        f_values, h_values, degree = get_f_h_values(path)
        reg_par = regression_function(f_values, h_values, degree)
        valve_st = cls(name, reg_par)
        valve_st._relative_flow_list = f_values
        valve_st._relative_travel_list = h_values
        valve_st._reg_polynomial_degree = degree
        return valve_st

    @classmethod
    def from_list(cls, name, f_values, h_values, degree):
        reg_par = regression_function(f_values, h_values, degree)
        valve_st = cls(name, reg_par)
        valve_st._relative_flow_list = f_values
        valve_st._relative_travel_list = h_values
        valve_st._reg_polynomial_degree = degree
        return valve_st

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
    elif std_type_category == 'valve':
        path = os.path.join(path)
        data = pd.read_csv(path, sep=';', dtype=np.float64)
    elif std_type_category == 'dyn_pump':
        path = os.path.join(path)
        data = pd.read_csv(path, sep=';', dtype=np.float64)

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

def get_f_h_values(path):
    """

    :param path:
    :type path:
    :return:
    :rtype:
    """
    data = get_data(path, 'valve')
    f_values = data.values[:, 0]
    h_values = data.values[:, 1]
    degree = data.values[0, 2]
    return f_values, h_values, degree


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


def linear_interpolation(x, xp, fp):
    # provides linear interpolation of points
    z = np.interp(x, xp, fp)
    return z
