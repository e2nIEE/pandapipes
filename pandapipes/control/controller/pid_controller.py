# -*- coding: utf-8 -*-

# Copyright (c) 2016-2022 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from pandapower.control.basic_controller import Controller
from pandapower.toolbox import _detect_read_write_flag, write_to_net
from pandapipes.control.controller.collecting_controller import CollectorController
import math, numpy as np
try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class PidControl(Controller):
    """
    Class representing a PID time series controller for a specified element and variable.

    """

    def __init__(self, net, fc_element, fc_variable, fc_element_index, pv_max, pv_min, auto=True, dir_reversed=False,
                 process_variable=None, process_element=None, process_element_index=None, cv_scaler=1,
                 Kp=1, Ti=5, Td=0, mv_max=100.00, mv_min=20.00, profile_name=None, ctrl_typ='std',
                 data_source=None, scale_factor=1.0, in_service=True, recycle=True, order=-1, level=-1,
                 drop_same_existing_ctrl=False, matching_params=None,
                 initial_run=False, **kwargs):
        # just calling init of the parent
        if matching_params is None:
            matching_params = {"fc_element": fc_element, "fc_variable": fc_variable,
                               "fc_element_index": fc_element_index}
        super().__init__(net, in_service=in_service, recycle=recycle, order=order, level=level,
                         drop_same_existing_ctrl=drop_same_existing_ctrl,
                         matching_params=matching_params, initial_run=initial_run,
                         **kwargs)

        self.__dict__.update(kwargs)
        #self.kwargs = kwargs

        # data source for time series values
        self.data_source = data_source
        # ids of sgens or loads
        self.fc_element_index = fc_element_index
        # control element type
        self.fc_element = fc_element
        self.ctrl_values = None

        self.profile_name = profile_name
        self.scale_factor = scale_factor
        self.applied = False
        self.write_flag, self.fc_variable = _detect_read_write_flag(net, fc_element, fc_element_index, fc_variable)
        self.set_recycle(net)

        # PID config
        self.Kp = Kp
        self.Ti = Ti
        self.Td = Td
        self.MV_max = mv_max
        self.MV_min = mv_min
        self.PV_max = pv_max
        self.PV_min = pv_min
        self.prev_mv = net[fc_element][fc_variable].loc[fc_element_index]
        self.prev_mvlag = net[fc_element][fc_variable].loc[fc_element_index]
        self.prev_act_pos = net[fc_element][fc_variable].loc[fc_element_index]
        self.prev_error = 0
        self.dt = 1
        self.dir_reversed = dir_reversed
        self.gain_effective = ((self.MV_max-self.MV_min)/(self.PV_max - self.PV_min)) * Kp
        # selected pv value
        self.process_element = process_element
        self.process_variable = process_variable
        self.process_element_index = process_element_index
        self.cv_scaler = cv_scaler
        self.cv = net[self.process_element][self.process_variable].loc[self.process_element_index] * cv_scaler
        self.sp = 0
        self.pv = 0
        self.prev_sp = 0
        self.prev_cv = net[self.process_element][self.process_variable].loc[self.process_element_index] * cv_scaler
        self.ctrl_typ = ctrl_typ
        self.diffgain = 1 # must be between 1 and 10
        self.diff_part = 0
        self.prev_diff_out = 0
        self.auto = auto

        super().set_recycle(net)

    def pid_control(self, error_value):
        """
        Algorithm 1: External Reset PID controller

        """
        # External Reset PID

        diff_component = np.divide(self.Td, self.Td + self.dt * self.diffgain)
        self.diff_part = diff_component * (self.prev_diff_out + self.diffgain * (error_value - self.prev_error))

        g_ain = (error_value * (1 + self.diff_part)) * self.gain_effective

        a = np.divide(self.dt, self.Ti + self.dt)

        mv_lag = (1 - a) * self.prev_mvlag + a * self.prev_mv

        mv_lag = np.clip(mv_lag, self.MV_min, self.MV_max)

        mv = g_ain + mv_lag

        # MV Saturation
        mv = np.clip(mv, self.MV_min, self.MV_max)

        self.prev_diff_out = self.diff_part
        self.prev_error = error_value
        self.prev_mvlag = mv_lag
        self.prev_mv = mv

        return mv


    def time_step(self, net, time):
        """
        Get the values of the element from data source
        Write to pandapower net by calling write_to_net()
        If ConstControl is used without a data_source, it will reset the controlled values to the initial values,
        preserving the initial net state.
        """
        self.applied = False
        self.dt = net['_options']['dt']


        self.pv = net[self.process_element][self.process_variable].loc[self.process_element_index]

        self.cv = self.pv * self.cv_scaler

        if type(self.data_source) is float:
            self.sp = self.data_source
        else:
            self.sp = self.data_source.get_time_step_value(time_step=time,
                                                           profile_name=self.profile_name,
                                                           scale_factor=self.scale_factor)


        if self.auto:
            # PID is in Automatic Mode
            # self.values is the set point we wish to make the output
            if not self.dir_reversed:
                # error= SP-PV
                error_value = self.sp - self.cv
            else:
                # error= SP-PV
                error_value = self.cv - self.sp

            # TODO: hysteresis band
            # if error < 0.01 : error = 0
            desired_mv = self.pid_control(error_value)

        else:
            # Write data source directly to controlled variable
            desired_mv = self.sp

        self.ctrl_values = desired_mv

        # Write desired_mv to the network

        if self.ctrl_typ == "over_ride":
            CollectorController.write_to_ctrl_collector(net, self.fc_element, self.fc_element_index,
                                                        self.fc_variable, self.ctrl_values, self.selector_typ,
                                                        self.write_flag)
        else: #self.ctrl_typ == "std":
            # write_to_net(net, self.ctrl_element, self.ctrl_element_index, self.ctrl_variable,
            # self.ctrl_values, self.write_flag)
            # Write the desired MV value to results for future plotting
            write_to_net(net, self.fc_element, self.fc_element_index, self.fc_variable, self.ctrl_values,
                         self.write_flag)


    def is_converged(self, net):
        """
        Actual implementation of the convergence criteria: If controller is applied, it can stop
        """
        return self.applied

    def control_step(self, net):
        """
        Set applied to True, which means that the values set in time_step have been included in the load flow calculation.
        """
        self.applied = True

    def __str__(self):
        return super().__str__() + " [%s.%s]" % (self.fc_element, self.fc_variable)
