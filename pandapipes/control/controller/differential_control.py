# -*- coding: utf-8 -*-

# Copyright (c) 2016-2022 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import numpy as np
from pandapower.auxiliary import _detect_read_write_flag, write_to_net

from pandapipes.control.controller.collecting_controller import CollectorController
from pandapipes.control.controller.pid_controller import PidControl

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class DifferentialControl(PidControl):
    """
    Class representing a PID- differential time series controller for specified elements and variables.

    """

    def __init__(self, net, fc_element, fc_variable, fc_element_index, pv_max, pv_min, sp_max, sp_min, auto=True,
                 direct_acting=False,
                 process_variable_1=None, process_element_1=None, process_element_index_1=None,
                 process_variable_2=None, process_element_2=None, process_element_index_2=None,
                 cv_scaler=1, Kp=1, Ti=5, Td=0, mv_max=100.00, mv_min=20.00, diff_gain= 1, sp_profile_name=None, man_profile_name=None,
                 ctrl_typ='std',  data_source=None, sp_scale_factor=1.0, in_service=True, recycle=True, order=-1, level=-1,
                 drop_same_existing_ctrl=False, matching_params=None,
                 initial_run=False, **kwargs):
        # just calling init of the parent
        if matching_params is None:
            matching_params = {"fc_element": fc_element, "fc_variable": fc_variable,
                               "fc_element_index": fc_element_index}
        super(PidControl, self).__init__(net, in_service=in_service, recycle=recycle, order=order, level=level,
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

        self.sp_profile_name = sp_profile_name
        self.sp_scale_factor = sp_scale_factor
        self.SP_max = sp_max
        self.SP_min = sp_min
        self.man_profile_name = man_profile_name
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
        self.direct_acting = direct_acting
        self.gain_effective = ((self.MV_max-self.MV_min)/(self.PV_max - self.PV_min)) * Kp
        # selected pv value
        # selected pv value
        self.process_element_1 = process_element_1
        self.process_variable_1 = process_variable_1
        self.process_element_index_1 = process_element_index_1
        self.process_element_2 = process_element_2
        self.process_variable_2 = process_variable_2
        self.process_element_index_2 = process_element_index_2
        self.cv_scaler = cv_scaler
        self.cv = (net[self.process_element_1][self.process_variable_1].loc[self.process_element_index_1] - \
                  net[self.process_element_2][self.process_variable_2].loc[self.process_element_index_2]) * self.cv_scaler
        self.sp = 0
        self.man_sp = 0
        self.pv = 0
        self.prev_sp = 0
        self.prev_cv = (net[self.process_element_1][self.process_variable_1].loc[self.process_element_index_1]
                        - net[self.process_element_2][self.process_variable_2].loc[self.process_element_index_2]) \
                       * self.cv_scaler
        self.ctrl_typ = ctrl_typ
        self.diffgain = diff_gain # must be between 1 and 10
        self.diff_part = 0
        self.prev_diff_out = 0
        self.auto = auto

        super().set_recycle(net)

    def time_step(self, net, time):
        """
        Get the values of the element from data source
        Write to pandapower net by calling write_to_net()
        If ConstControl is used without a data_source, it will reset the controlled values to the initial values,
        preserving the initial net state.
        """
        self.applied = False
        self.dt = net['_options']['dt']
        # Differential calculation
        pv_1 = net[self.process_element_1][self.process_variable_1].loc[self.process_element_index_1]
        pv_2 = net[self.process_element_2][self.process_variable_2].loc[self.process_element_index_2]
        self.pv = pv_1 - pv_2

        self.cv = self.pv * self.cv_scaler


        if self.auto:
            # PID is in Automatic Mode
            if type(self.sp_data_source) is float:
                self.sp = self.sp_data_source
            elif type(self.sp_data_source) is int:
                self.sp = np.float64(self.sp_data_source)
            else:
                self.sp = self.sp_data_source.get_time_step_value(time_step=time,
                                                               profile_name=self.sp_profile_name,
                                                               scale_factor=self.sp_scale_factor)
            # Clip set point and ensure within allowed operation ranges
            self.sp = np.clip(self.sp, self.SP_min, self.SP_max)

            # PID Controller Action:
            if not self.direct_acting:
                # Reverse acting
                # positive error which increases output
                error_value = self.sp - self.cv
            else:
                # Direct acting
                # negative error that decreases output
                error_value = self.cv - self.sp

            # TODO: hysteresis band
            # if error < 0.01 : error = 0
            desired_mv = self.pidConR_control(error_value)

        else:
            # Get Manual set point from data source:
            if type(self.man_data_source) is float:
                self.man_sp = self.man_data_source
            else:
                self.man_sp = self.man_data_source.get_time_step_value(time_step=time,
                                                                       profile_name=self.man_profile_name,
                                                                       scale_factor=1)
            desired_mv = np.clip(self.man_sp, self.MV_min, self.MV_max)

        self.ctrl_values = desired_mv

        # Write desired_mv to the logic controller
        if hasattr(self, "logic_element"):
            if self.logic_element is not None:  #
                self.logic_element_index.__setattr__(self.logic_variable, self.ctrl_values)
            else:
                raise NotImplementedError("logic_element for " + str(self.logic_element_index) +
                                          ' is not set correctly')
        # Write desired_mv to the network
        elif self.ctrl_typ == "over_ride":
            CollectorController.write_to_ctrl_collector(net, self.fc_element, self.fc_element_index,
                                                        self.fc_variable, self.ctrl_values, self.selector_typ,
                                                        self.write_flag)
        else:  # self.ctrl_typ == "std":
            # Write the desired MV value to results for future plotting
            write_to_net(net, self.fc_element, self.fc_element_index, self.fc_variable, self.ctrl_values,
                         self.write_flag)

    def __str__(self):
        return super().__str__() + " [%s.%s]" % (self.fc_element, self.fc_variable)


