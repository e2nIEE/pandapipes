# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import pandapipes as pp
import numpy
from pandapower.control.basic_controller import Controller

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class NonReturnValveController(Controller):
    """
    Controller for implementing a non-return valve.

    :param net: The net in which the controller resides
    :type net: pandapipesNet
    :param element_index: IDs of controlled valves
    :type element_index: int[]
    :param in_service: Indicates if the controller is currently in_service
    :type in_service: bool, default True
    :param recycle: Re-use of internal-data
    :type recycle: bool, default True
    :param drop_same_existing_ctrl: Indicates if already existing controllers of the same type and with the same matching parameters (e.g. at same element) should be dropped
    :type drop_same_existing_ctrl: bool, default False
    :param kwargs: Parameters for pipeflow
    :type kwargs: dict

    :Example:
        >>> kwargs = {'stop_condition': 'tol', 'iter': 100, 'tol_p': 1e-7, 'tol_v': 1e-7, 'friction_model': 'colebrook',
        >>>           'mode': 'hydraulics', 'only_update_hydraulic_matrix': False}
        >>> NonReturnValveController(net, element_index=[0, 1, 3], **kwargs)
        >>> run_control(net)

    """

    def __init__(self, net, element_index, profile_name=None,
                 scale_factor=1.0, in_service=True, recycle=True, order=0, level=0,
                 drop_same_existing_ctrl=False, set_q_from_cosphi=False, matching_params=None, initial_pipeflow=False,
                 **kwargs):

        if matching_params is None:
            matching_params = {"element_index": element_index}

        # just calling init of the parent
        super().__init__(net, in_service=in_service, recycle=recycle, order=order, level=level,
                         drop_same_existing_ctrl=drop_same_existing_ctrl,
                         matching_params=matching_params, initial_powerflow=initial_pipeflow,
                         **kwargs)

        self.matching_params = {"element_index": element_index}
        if numpy.isscalar(element_index):
            self.element_index = [element_index]
        else:
            self.element_index = element_index
        self.values = None
        self.profile_name = profile_name
        self.scale_factor = scale_factor
        self.initial_pipeflow = initial_pipeflow
        self.kwargs = kwargs
        self.v_m_per_s = []  # current flow velocities at valves
        self.opened = []  # remember original user-defined values of opened

        if set_q_from_cosphi:
            logger.error("Parameter set_q_from_cosphi deprecated!")
            raise ValueError

    def initialize_control(self):
        """
        First calculation of a pipeflow. \n
        Saving the user-defined values, determine valves with negative flow velocities,
        set opened to False for these.
        """
        pp.pipeflow(self.net, self.kwargs)

        self.opened = self.net.valve.loc[self.element_index, "opened"]

        j = 0
        for i in self.element_index:
            self.v_m_per_s.append(self.net.res_valve.loc[i, "v_mean_m_per_s"])

            if self.net.valve.loc[i, "opened"] and self.v_m_per_s[j] < 0:
                # use the element indices, where opened = True, otherwise NaN would be in self.v_m_per_s
                self.net.valve.loc[i, "opened"] = False
            j += 1

    def is_converged(self):
        """
        Convergence Condition: If all flow velocities at the non-return valves are >= 0 or opened equal False. \n
        Resetting the variable opened to user defaults.
        """

        for i in range(len(self.element_index)):
            if self.net.valve.loc[self.element_index[i], "opened"] and self.v_m_per_s[i] < 0:
                return False

        self.net.valve.loc[self.element_index, "opened"] = self.opened
        return True

    def control_step(self):
        """
        Check whether negative flow velocities are still present at non-return valves.
        """
        pp.pipeflow(self.net, self.kwargs)

        j = 0
        for i in self.element_index:
            self.v_m_per_s.append(self.net.res_valve.loc[i, "v_mean_m_per_s"])

            if self.net.valve.loc[i, "opened"] and self.v_m_per_s[j] < 0:
                # use the element indices, where opened = True, otherwise NaN would be in self.v_m_per_s
                self.net.valve.loc[i, "opened"] = False
            j += 1
