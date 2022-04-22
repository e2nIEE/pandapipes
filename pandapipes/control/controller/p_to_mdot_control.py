import copy

import numpy as np

from pandapipes.pipeflow_setup import get_lookup
from pandapipes.properties.fluids import get_mixture_higher_heating_value, get_mixture_lower_heating_value, get_fluid
from pandapower.control import ConstControl
from pandapower.toolbox import write_to_net


class PtoMdotController(ConstControl):

    def __init__(self, net, element, element_index, profile_name=None, data_source=None,
                 efficiency_factor=1.05, calorific_value='lower',
                 scale_factor=1.0, in_service=True, recycle=True, order=-1, level=-1,
                 drop_same_existing_ctrl=False, matching_params=None,
                 initial_run=True, **kwargs):
        # just calling init of the parent
        if matching_params is None:
            matching_params = {"element": element, "variable": 'mdot_kg_per_s',
                               "element_index": element_index}
        super().__init__(net, element, 'mdot_kg_per_s', element_index, profile_name, data_source,
                         scale_factor, in_service, recycle, order, level,
                         drop_same_existing_ctrl, matching_params, initial_run, **kwargs)
        if calorific_value not in ['lower', 'higher']: raise (
            AttributeError("Choose either 'higher' or 'lower' calorific value "))
        self.eff = efficiency_factor
        self.cal = calorific_value

    def time_step(self, net, time):
        self.mf = None
        self.applied = False
        if self.data_source is None:
            self.values = net[self.element][self.variable].loc[self.element_index]
        else:
            self.values = self.data_source.get_time_step_value(time_step=time,
                                                               profile_name=self.profile_name,
                                                               scale_factor=self.scale_factor)
            if len(net._fluid) == 1:
                fluid = net._fluid[0]
                cal = get_fluid(net, fluid).get_higher_heating_value() if self.cal == 'higher' else \
                    get_fluid(net, fluid).get_lower_heating_value()
                self.values *= 1 / self.eff / cal / 3.6  # from W (/1000) and kWh (*3600) to kg / s
                self.values = self.values.astype(float)
            else:
                w = get_lookup(net, 'node', 'w')
                node_pit = net._pit['node']
                index = get_lookup(net, 'node', "index")['junction'][net.sink.junction.values]
                mf = node_pit[index, :][:, w]

                cal = get_mixture_higher_heating_value(net, mf) \
                    if self.cal == 'higher' else get_mixture_lower_heating_value(net, mf)
                self.values_mdot = copy.deepcopy(self.values)
                self.p_to_mdot(cal)
        if self.values is not None:
            write_to_net(net, self.element, self.element_index, self.variable, self.values, self.write_flag)

    def p_to_mdot(self, heating_value):
        self.values = self.values_mdot * 1 / self.eff / heating_value / 3.6
        self.values = self.values.astype(float)

    def control_step(self, net):
        w = get_lookup(net, 'node', 'w')
        node_pit = net._pit['node']
        if len(net._fluid) == 1:
            self.applied = True
        elif self.values is None:
            self.applied = True
        elif (self.mf is None) or np.any((self.mf - node_pit[:, w]) >= 10 ** -3):
            w = get_lookup(net, 'node', 'w')
            node_pit = net._pit['node']
            index = get_lookup(net, 'node', "index")['junction'][net.sink.junction.values]
            mf = node_pit[index, :][:, w]

            cal = get_mixture_higher_heating_value(net, mf) \
                if self.cal == 'higher' else get_mixture_lower_heating_value(net, mf)
            self.p_to_mdot(cal)
            write_to_net(net, self.element, self.element_index, self.variable, self.values, self.write_flag)
            self.mf = node_pit[:, w]
        else:
            self.applied = True
