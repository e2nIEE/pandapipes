# -*- coding: utf-8 -*-

# Copyright (c) 2016-2022 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from pandapipes.control.controller.pid_controller import PidControl
from pandapower.toolbox import _detect_read_write_flag, write_to_net

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class DifferentialControl(PidControl):
    """
    Class representing a PID- differential time series controller for specified elements and variables.

    """

    def __init__(self, net, fc_element, fc_variable, fc_element_index, pv_max, pv_min, prev_mv=100, integral=0, dt=1,
                 dir_reversed=False, process_variable_1=None, process_element_1=None, process_element_index_1=None,
                 process_variable_2=None, process_element_2=None, process_element_index_2=None, cv_scaler=1,
                 kp=1, ki=0.05, Ti=5, Td=0, kd=0, mv_max=100.00, mv_min=20.00, profile_name=None,
                 data_source=None, scale_factor=1.0, in_service=True, recycle=True, order=-1, level=-1,
                 drop_same_existing_ctrl=False, matching_params=None,
                 initial_run=False, pass_element=None, pass_variable=None, pass_element_index=None, **kwargs):
        # just calling init of the parent
        if matching_params is None:
            matching_params = {"element": fc_element, "variable": fc_variable,
                               "element_index": fc_element_index}
        super().__init__(net, in_service=in_service, recycle=recycle, order=order, level=level,
                         drop_same_existing_ctrl=drop_same_existing_ctrl,
                         matching_params=matching_params, initial_run=initial_run,
                         **kwargs)

        # data source for time series values
        self.data_source = data_source
        self.ctrl_variable = fc_variable
        self.ctrl_element_index = fc_element_index
        # element type
        self.ctrl_element = fc_element
        self.values = None

        self.profile_name = profile_name
        self.scale_factor = scale_factor
        self.applied = False
        self.write_flag, self.variable = _detect_read_write_flag(net, fc_element, fc_element_index, fc_variable)
        self.set_recycle(net)

        # PID config
        self.Kp = kp
        self.Kc = 1
        self.Ki = ki
        self.Ti = Ti
        self.Td = Td
        self.Kd = kd
        self.MV_max = mv_max
        self.MV_min = mv_min
        self.PV_max = pv_max
        self.PV_min = pv_min
        self.integral = integral
        self.prev_mv = prev_mv
        self.prev_error = 0
        self.dt = dt
        self.dir_reversed = dir_reversed
        self.gain_effective = ((self.MV_max - self.MV_min) / (self.PV_max - self.PV_min)) * self.Kp
        # selected pv value
        self.process_element_1 = process_element_1
        self.process_variable_1 = process_variable_1
        self.process_element_index_1 = process_element_index_1
        self.process_element_2 = process_element_2
        self.process_variable_2 = process_variable_2
        self.process_element_index_2 = process_element_index_2
        self.cv_scaler = cv_scaler
        self.cv = net[self.ctrl_element][self.process_variable].loc[self.process_element_index]
        self.prev_cv = 0
        self.prev2_cv = 0

        super().set_recycle(net)

    def time_step(self, net, time):
        """
        Get the values of the element from data source
        Write to pandapower net by calling write_to_net()
        If ConstControl is used without a data_source, it will reset the controlled values to the initial values,
        preserving the initial net state.
        """
        self.applied = False
        # Differential calculation
        pv_1 = net[self.process_element_1][self.process_variable_1].loc[self.process_element_index_1]
        pv_2 = net[self.process_element_2][self.process_variable_2].loc[self.process_element_index_2]
        pv = pv_1 - pv_2

        self.cv = pv * self.cv_scaler
        sp = self.data_source.get_time_step_value(time_step=time,
                                                  profile_name=self.profile_name,
                                                  scale_factor=self.scale_factor)

        # self.values is the set point we wish to make the output
        if not self.dir_reversed:
            # error= SP-PV
            error_value = sp - self.cv
        else:
            # error= SP-PV
            error_value = self.cv - sp

        # TODO: hysteresis band
        # if error < 0.01 : error = 0
        mv = self.pid_control(error_value.values)

        self.values = mv
        # here we write the mv value to the network controlled element
        write_to_net(net, self.element, self.element_index, self.variable, self.values, self.write_flag)

    def __str__(self):
        return super().__str__() + " [%s.%s]" % (self.element, self.variable)


