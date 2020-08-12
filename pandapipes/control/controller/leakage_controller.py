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


class LeakageController(Controller):
    """
    Leakage Controller

    :param net: The net in which the controller resides
    :type net: pandapipesNet
    :param element: Element (first only "pipe", "valve", "heat_exchanger")
    :type element: string
    :param element_index: IDs of controlled elements
    :type element_index: int[]
    :param output_area_m2: Size of the leakage in m^2
    :type output_area_m2: float[]
    :param in_service: Indicates if the controller is currently in_service
    :type in_service: bool, default True
    :param recycle: Re-use of internal-data
    :type recycle: bool, default True
    :param drop_same_existing_ctrl: Indicates if already existing controllers of the same type and with
    the same matching parameters (e.g. at same element) should be dropped
    :type drop_same_existing_ctrl: bool, default False

    """

    def __init__(self, net, element, element_index, output_area_m2, profile_name=None,
                 scale_factor=1.0, in_service=True, recycle=True, order=0, level=0,
                 drop_same_existing_ctrl=False, set_q_from_cosphi=False, matching_params=None, initial_pipeflow=False,
                 **kwargs):

        if element not in ["pipe", "valve", "heat_exchanger"]:
            raise Exception("Only 'pipe', 'valve' or 'heat_exchanger' is allowed as element.")

        if matching_params is None:
            matching_params = {"element": element, "element_index": element_index}

        # just calling init of the parent
        super().__init__(net, in_service=in_service, recycle=recycle, order=order, level=level,
                         drop_same_existing_ctrl=drop_same_existing_ctrl,
                         matching_params=matching_params, initial_powerflow=initial_pipeflow,
                         **kwargs)

        self.matching_params = {"element": element, "element_index": element_index}
        if numpy.isscalar(element_index):
            self.element_index = [element_index]
            self.output_area_m2 = [output_area_m2]
        else:
            self.element_index = element_index
            self.output_area_m2 = output_area_m2
        self.element = element
        self.values = None
        self.profile_name = profile_name
        self.scale_factor = scale_factor
        self.initial_pipeflow = initial_pipeflow
        self.kwargs = kwargs
        self.rho_kg_per_m3 = []  # densities for the calculation of leakage mass flows
        self.v_m_per_s = []  # current flow velocities at pipes
        self.mass_flow_kg_per_s_init = []  # initial/ previous leakage mass flows
        self.mass_flow_kg_per_s = []  # current leakage mass flows
        self.leakage_index = []  # index of the sinks for leakages

        if set_q_from_cosphi:
            logger.error("Parameter set_q_from_cosphi deprecated!")
            raise ValueError

    def initialize_control(self):
        """

        """
        pp.pipeflow(self.net, self.kwargs)

        for i in range(len(self.element_index)):
            self.init_values(self.element_index[i])

            index = pp.create_sink(self.net, self.net[self.element].loc[self.element_index[i], "to_junction"],
                                   mdot_kg_per_s=0, name="leakage"+str(i))
            self.leakage_index.append(index)
            self.mass_flow_kg_per_s_init.append(0)

    def init_values(self, index):
        """

        """
        self.v_m_per_s.append(self.net["res_"+self.element].loc[index, "v_mean_m_per_s"])

        temp_1 = self.net.res_junction.loc[self.net[self.element].loc[index, "from_junction"], "t_k"]
        temp_2 = self.net.res_junction.loc[self.net[self.element].loc[index, "to_junction"], "t_k"]
        self.rho_kg_per_m3.append((self.net.fluid.get_density(temp_1) + self.net.fluid.get_density(temp_2)) / 2)

    def is_converged(self):
        """
        Convergence Condition: Difference between mass flows smaller than 1e-5 = 0.00001
        """
        if not self.mass_flow_kg_per_s:
            return False

        for i in range(len(self.element_index)):
            if abs(self.mass_flow_kg_per_s_init[i] - self.mass_flow_kg_per_s[i]) > 1e-5:
                return False

        self.net.sink = self.net.sink.drop(labels=self.leakage_index)
        self.net.res_sink = self.net.res_sink.drop(labels=self.leakage_index)

        return True

    def control_step(self):
        """

        """
        pp.pipeflow(self.net, self.kwargs)
        self.mass_flow_kg_per_s = []
        self.v_m_per_s = []
        self.rho_kg_per_m3 = []

        for i in range(len(self.element_index)):
            self.init_values(self.element_index[i])

            self.net.sink.loc[self.leakage_index[i], "mdot_kg_per_s"] = self.rho_kg_per_m3[i] * self.v_m_per_s[i] * \
                                                                        self.output_area_m2[i]
            self.mass_flow_kg_per_s.append(self.net.sink.loc[self.leakage_index[i], "mdot_kg_per_s"])

        self.mass_flow_kg_per_s_init = self.mass_flow_kg_per_s
