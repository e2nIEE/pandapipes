from pandapower.control import ConstControl
from pandapipes.properties.fluids import get_fluid
from pandapower.control.basic_controller import Controller
from pandas.errors import InvalidIndexError

import pandapower as ppower
class P2HControlMultiEnergy(Controller):
    def __init__(self, multinet, element_index_power, element_index_heat, efficiency,
                 name_power_net='power', name_heat_net='heat',
                 in_service=True, order=0, level=0,
                 drop_same_existing_ctrl=False, initial_run=True, **kwargs):
        super().__init__(multinet, in_service, order, level,
                         drop_same_existing_ctrl=drop_same_existing_ctrl, initial_run=initial_run,
                         **kwargs)

        self.elm_idx_power = element_index_power
        self.elm_idx_heat = element_index_heat
        self.name_net_power = name_power_net
        self.name_net_heat = name_heat_net
        self.efficiency = efficiency
        self.qext_w = None
        self.fluid = get_fluid(multinet['nets'][name_heat_net])
        self.applied = False

    def initialize_control(self, multinet):
        self.applied = False

    def get_all_net_names(self):
        return [self.name_net_power, self.name_net_heat]

    def control_step(self, multinet):
        ppower.runpp(multinet['nets'][self.name_net_power])

        try:
            power_load = \
                multinet['nets'][self.name_net_power].res_load.at[self.elm_idx_power, 'p_mw']
        except (ValueError, TypeError, InvalidIndexError):
            power_load = \
                multinet['nets'][self.name_net_power].res_load.loc[self.elm_idx_power, 'p_mw'].values
        self.qext_w = - (power_load * self.conversion_factor_mw_to_w() * self.efficiency)

        self.write_to_net(multinet)
        self.applied = True

    def write_to_net(self, multinet):
        try:
            multinet['nets'][self.name_net_heat].heat_exchanger.at[self.elm_idx_heat, 'qext_w'] \
                = self.qext_w
        except (ValueError, TypeError, InvalidIndexError):
            multinet['nets'][self.name_net_heat].heat_exchanger.loc[self.elm_idx_heat,
                                                           'qext_w'] = self.qext_w

    def is_converged(self, multinet):
        return self.applied

    def conversion_factor_mw_to_w(self):
        return 1e6


class H2PControlMultiEnergy(Controller):
    def __init__(self, multinet, element_index_power, element_index_heat, efficiency,
                 name_power_net='power', name_heat_net='heat', element_type_power="sgen",
                 in_service=True, order=0,
                 level=0, drop_same_existing_ctrl=False, initial_run=True,
                 calc_heat_from_power=False, **kwargs):
        super().__init__(multinet, in_service, order, level,
                         drop_same_existing_ctrl=drop_same_existing_ctrl, initial_run=initial_run,
                         **kwargs)

        self.elm_idx_power = element_index_power
        self.elm_idx_heat = element_index_heat
        self.elm_type_power = element_type_power
        self.name_net_power = name_power_net
        self.name_net_heat = name_heat_net
        self.efficiency = efficiency
        self.qext_w = None
        self.fluid = get_fluid(multinet['nets'][name_heat_net])
        self.el_power_led = calc_heat_from_power
        self.applied = False

    def initialize_control(self, multinet):
        self.applied = False

    def get_all_net_names(self):
        return [self.name_net_heat, self.name_net_power]

    def control_step(self, multinet):
        if self.el_power_led:
            try:
                power_gen = multinet['nets'][self.name_net_power][self.elm_type_power].at[
                    self.elm_idx_power, 'p_mw'] * multinet['nets'][self.name_net_power][
                    self.elm_type_power].at[self.elm_idx_power, 'scaling']

            except (ValueError, TypeError, InvalidIndexError):
                power_gen = multinet['nets'][self.name_net_power][self.elm_type_power].loc[
                                self.elm_idx_power, 'p_mw'].values[:] \
                            * multinet['nets'][self.name_net_power][self.elm_type_power].loc[
                                self.elm_idx_power, 'scaling'].values[:]

            self.heat_cons = power_gen / (self.conversion_factor_w_to_mw() * self.efficiency)

        else:
            try:
                heat_heat_exchanger = \
                    multinet['nets'][self.name_net_heat].heat_exchanger.at[self.elm_idx_heat, 'qext_w']

            except (ValueError, TypeError, InvalidIndexError):
                heat_heat_exchanger = multinet['nets'][self.name_net_heat].heat_exchanger.loc[self.elm_idx_heat,
                                                                        'qext_w'].values[:]

            self.power_gen = heat_heat_exchanger * self.conversion_factor_w_to_mw() * self.efficiency

        self.write_to_net(multinet)
        self.applied = True

    def write_to_net(self, multinet):
        if self.el_power_led:
            try:
                multinet['nets'][self.name_net_heat].heat_exchanger.at[self.elm_idx_heat,
                                                            'qext_w'] = self.heat_cons
            except (ValueError, TypeError, InvalidIndexError):
                multinet['nets'][self.name_net_heat].heat_exchanger.loc[self.elm_idx_heat,
                                                             'qext_w'] = self.heat_cons
        else:
            try:
                multinet['nets'][self.name_net_power][self.elm_type_power].at[
                    self.elm_idx_power, 'p_mw'] = self.power_gen
            except (ValueError, TypeError, InvalidIndexError):
                multinet['nets'][self.name_net_power][self.elm_type_power].loc[
                    self.elm_idx_power, 'p_mw'] = self.power_gen

    def is_converged(self, multinet):
        return self.applied

    def conversion_factor_w_to_mw(self):
        return 1 / 1e6
