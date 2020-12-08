# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import pandapipes as pp
import numpy
from pandapower.control.basic_controller import Controller
from pandapipes.component_models import Pipe, Valve, HeatExchanger

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class LeakageController(Controller):
    """
    Controller to consider a leak with an outlet area at pipes, valves, heat exchangers or junctions.

    :param net: The net in which the controller resides
    :type net: pandapipesNet
    :param element: Element (first only "pipe", "valve", "heat_exchanger", "junction")
    :type element: string
    :param element_index: IDs of controlled elements
    :type element_index: int[]
    :param output_area_m2: Size of the leakage in m^2
    :type output_area_m2: float[]
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
        >>> LeakageController(net, element='pipe', element_index=0, output_area_m2=1, **kwargs)
        >>> run_control(net)

    """

    def __init__(self, net, element, element_index, output_area_m2, profile_name=None,
                 scale_factor=1.0, in_service=True, recycle=True, order=0, level=0,
                 drop_same_existing_ctrl=False, matching_params=None, initial_run=True,
                 **kwargs):

        if element not in ["pipe", "valve", "heat_exchanger", "junction"]:
            raise Exception("Only 'pipe', 'valve', 'heat_exchanger' or 'junction' is allowed as element.")

        if matching_params is None:
            matching_params = {"element": element, "element_index": element_index}

        # just calling init of the parent
        super().__init__(net, in_service=in_service, recycle=recycle, order=order, level=level,
                         drop_same_existing_ctrl=drop_same_existing_ctrl,
                         matching_params=matching_params, initial_run=initial_run,
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
        self.initial_run = initial_run
        self.kwargs = kwargs
        self.rho_kg_per_m3 = numpy.array([])  # densities for the calculation of leakage mass flows
        self.v_m_per_s = numpy.full(len(self.element_index), numpy.nan, float)  # current flow velocities at pipes
        self.mass_flow_kg_per_s_init = []  # initial/ previous leakage mass flows
        self.mass_flow_kg_per_s = []  # current leakage mass flows
        self.leakage_index = []  # index of the sinks for leakages
        self.branches = []  # branch components in current net

    def initialize_control(self):
        """
        Define the initial values and create the sinks representing the leaks.
        """
        for i in range(len(self.element_index)):
            if self.element == "junction":
                index = pp.create_sink(self.net, self.element_index[i],
                                       mdot_kg_per_s=0, name="leakage" + str(i))
            else:
                index = pp.create_sink(self.net, self.net[self.element].loc[self.element_index[i], "to_junction"],
                                       mdot_kg_per_s=0, name="leakage" + str(i))
            self.leakage_index.append(index)
            self.mass_flow_kg_per_s_init.append(0)

        if self.element == "junction":
            # determine branch components in the network, necessary for further calculations
            branches = {Pipe: "pipe", Valve: "valve", HeatExchanger: "heat_exchanger"}

            for key in branches:
                if (numpy.array(self.net.component_list) == key).any():
                    self.branches.append(branches[key])

    def init_values(self):
        """
        Initialize the flow velocity and density for the individual control steps.
        """
        if self.element == "junction":
            # identify the branches connected to a junction leakage to obtain the flow velocity
            for i in range(len(self.element_index)):
                for branch in self.branches:
                    ind = numpy.where(numpy.array(self.net[branch]["to_junction"]) == self.element_index[i])

                    if ind[0].size != 0:
                        if ind[0].size == 1:
                            index = ind[0]
                        else:
                            index = ind[0][0]
                        self.v_m_per_s[i] = abs(self.net["res_" + branch].loc[index, "v_mean_m_per_s"])
                        break

                    ind = numpy.where(numpy.array(self.net[branch]["from_junction"]) == self.element_index[i])

                    if ind[0].size != 0:
                        if ind[0].size == 1:
                            index = ind[0]
                        else:
                            index = ind[0][0]
                        self.v_m_per_s[i] = abs(self.net["res_" + branch].loc[index, "v_mean_m_per_s"])
                        break

            if (self.v_m_per_s == numpy.nan).any():
                raise Exception("One or more junctions are only connected to a pump. Leakage calculation "
                                "not yet possible here.")

            temp = self.net.res_junction.loc[self.element_index, "t_k"]
            self.rho_kg_per_m3 = self.net.fluid.get_density(temp)

        else:
            self.v_m_per_s = numpy.array(self.net["res_" + self.element].loc[self.element_index, "v_mean_m_per_s"])

            temp_1 = self.net.res_junction.loc[self.net[self.element].loc[self.element_index, "from_junction"], "t_k"]
            temp_2 = self.net.res_junction.loc[self.net[self.element].loc[self.element_index, "to_junction"], "t_k"]
            self.rho_kg_per_m3 = (self.net.fluid.get_density(temp_1) + self.net.fluid.get_density(temp_2)) / 2

    def is_converged(self):
        """
        Convergence Condition: Difference between mass flows smaller than 1e-5 = 0.00001
        """
        if numpy.array(self.mass_flow_kg_per_s).size == 0:
            return False

        if abs(numpy.subtract(self.mass_flow_kg_per_s, self.mass_flow_kg_per_s_init)).all() > 1e-5:
            return False

        return True

    def control_step(self):
        """
        Calculate the new mass flow for each leak using the density, flow velocity and outlet area.
        """
        pp.pipeflow(self.net, self.kwargs)
        self.mass_flow_kg_per_s = []
        self.v_m_per_s = numpy.full(len(self.element_index), numpy.nan, float)
        self.rho_kg_per_m3 = numpy.array([])

        self.init_values()

        self.net.sink.loc[self.leakage_index, "mdot_kg_per_s"] = self.rho_kg_per_m3 * self.v_m_per_s * \
                                                                 numpy.array(self.output_area_m2)
        self.mass_flow_kg_per_s = self.net.sink.loc[self.leakage_index, "mdot_kg_per_s"]

        self.mass_flow_kg_per_s_init = self.mass_flow_kg_per_s

    def finalize_control(self):
        """
        Delete the sinks that represented the leaks.
        """
        self.net.sink = self.net.sink.drop(labels=self.leakage_index)
        self.net.res_sink = self.net.res_sink.drop(labels=self.leakage_index)
