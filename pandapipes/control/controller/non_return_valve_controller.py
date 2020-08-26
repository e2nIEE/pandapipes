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
                 drop_same_existing_ctrl=False, matching_params=None, initial_run=False,
                 **kwargs):

        if matching_params is None:
            matching_params = {"element_index": element_index}

        # just calling init of the parent
        super().__init__(net, in_service=in_service, recycle=recycle, order=order, level=level,
                         drop_same_existing_ctrl=drop_same_existing_ctrl,
                         matching_params=matching_params, initial_run=initial_run,
                         **kwargs)

        self.matching_params = {"element_index": element_index}
        if numpy.isscalar(element_index):
            self.element_index = [element_index]
        else:
            self.element_index = element_index
        self.values = None
        self.profile_name = profile_name
        self.scale_factor = scale_factor
        self.initial_run = initial_run
        self.kwargs = kwargs
        self.v_m_per_s = []  # current flow velocities at valves
        self.opened = []  # remember original user-defined values of opened

    def initialize_control(self):
        """
        Saving the user-defined values and adapt types.
        """
        self.opened = self.net.valve.loc[self.element_index, "opened"]
        self.net.valve.loc[self.element_index, "opened"] = True
        self.net.valve.loc[self.element_index, "type"] = "non-return valve"

    def is_converged(self):
        """
        Convergence Condition: If all flow velocities at the non-return valves are >= 0 or opened equal False.
        """
        if numpy.array(self.v_m_per_s).size == 0:
            return False

        if numpy.array(self.net.valve.loc[self.element_index, "opened"]).any() and \
                numpy.array(self.v_m_per_s).any() < 0:
            return False

        return True

    def control_step(self):
        """
        Check whether negative flow velocities are present at non-return valves,
        set opened to False for these.
        """
        pp.pipeflow(self.net, self.kwargs)

        self.v_m_per_s = numpy.array(self.net.res_valve.loc[self.element_index, "v_mean_m_per_s"])

        ind_opened = numpy.where(self.net.valve.loc[self.element_index, "opened"])
        # use the element indices, where opened = True, otherwise NaN would be in self.v_m_per_s

        ind_negative_v = numpy.where(self.v_m_per_s[ind_opened[0]] < 0)

        self.net.valve.loc[numpy.array(self.element_index)[ind_negative_v[0]], "opened"] = False

    def finalize_control(self):
        """
        Resetting the variable opened to user defaults.
        """
        self.net.valve.loc[self.element_index, "opened"] = self.opened
