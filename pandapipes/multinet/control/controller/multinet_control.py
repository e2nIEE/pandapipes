# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from pandapower.control import ConstControl
from pandapipes import get_fluid
from pandapower.control.basic_controller import Controller


class P2GControlMultiEnergy(Controller):
    def __init__(self, multinet, element_index_power, element_index_gas, efficiency,
                 name_power_net='power', name_gas_net='gas',
                 in_service=True, order=0, level=0,
                 drop_same_existing_ctrl=False, initial_run=True, **kwargs):
        super().__init__(multinet, in_service, order, level,
                         drop_same_existing_ctrl=drop_same_existing_ctrl, initial_run=initial_run,
                         **kwargs)
        self.elm_idx_power = element_index_power
        self.elm_idx_gas = element_index_gas
        self.name_net_power = name_power_net
        self.name_net_gas = name_gas_net
        self.efficiency = efficiency
        self.mdot_kg_per_s = None
        self.fluid = get_fluid(multinet['nets'][name_gas_net])
        self.fluid_calorific_value = self.fluid.get_property('hhv')
        self.initial_run = initial_run
        self.applied = False

    def initialize_control(self, multinet):
        self.applied = False

    def get_all_net_names(self):
        return [self.name_net_power, self.name_net_gas]

    def control_step(self, multinet):
        try:
            power_load = multinet['nets'][self.name_net_power].load.at[self.elm_idx_power, 'p_mw']
        except (ValueError, TypeError):
            power_load = multinet['nets'][self.name_net_power].load.loc[self.elm_idx_power, 'p_mw'].values
        self.mdot_kg_per_s = power_load * self.conversion_factor_mw_to_kgps() * self.efficiency
        self.write_to_net(multinet)
        self.applied = True

    def write_to_net(self, multinet):
        try:
            multinet['nets'][self.name_net_gas].source.at[self.elm_idx_gas, 'mdot_kg_per_s'] \
                = self.mdot_kg_per_s
        except (ValueError, TypeError):
            multinet['nets'][self.name_net_gas].source.loc[self.elm_idx_gas,
                                                           'mdot_kg_per_s'].values[:] = self.mdot_kg_per_s

    def is_converged(self, multinet):
        return self.applied

    def conversion_factor_mw_to_kgps(self):
        return 1e3 / (self.fluid_calorific_value * 3600)


class G2PControlMultiEnergy(Controller):
    """Gas to power conversion"""

    def __init__(self, multinet, element_index_power, element_index_gas, efficiency,
                 name_power_net='power', name_gas_net='gas', element_type_power="sgen",
                 in_service=True, order=0,
                 level=0, drop_same_existing_ctrl=False, initial_run=True,
                 calc_gas_from_power=False, **kwargs):
        super().__init__(multinet, in_service, order, level,
                         drop_same_existing_ctrl=drop_same_existing_ctrl, initial_run=initial_run,
                         **kwargs)
        self.elm_idx_power = element_index_power
        self.elm_idx_gas = element_index_gas
        self.elm_type_power = element_type_power
        self.name_net_power = name_power_net
        self.name_net_gas = name_gas_net
        self.efficiency = efficiency
        self.mdot_kg_per_s = None
        self.fluid = get_fluid(multinet['nets'][name_gas_net])
        self.fluid_calorific_value = self.fluid.get_property('hhv')
        self.initial_run = initial_run
        self.el_power_led = calc_gas_from_power
        self.applied = False

    def initialize_control(self, multinet):
        self.applied = False

    def get_all_net_names(self):
        return [self.name_net_gas, self.name_net_power]

    def control_step(self, multinet):
        if self.el_power_led:
            try:
                power_gen = multinet['nets'][self.name_net_power][self.elm_type_power].at[
                    self.elm_idx_power, 'p_mw']
            except (ValueError, TypeError):
                power_gen = multinet['nets'][self.name_net_power][self.elm_type_power].loc[
                                self.elm_idx_power, 'p_mw'].values[:]

            self.gas_cons = power_gen / (self.conversion_factor_kgps_to_mw() * self.efficiency)

        else:
            try:
                gas_sink = multinet['nets'][self.name_net_gas].sink.at[self.elm_idx_gas,
                                                                       'mdot_kg_per_s']
            except (ValueError, TypeError):
                gas_sink = multinet['nets'][self.name_net_gas].sink.loc[self.elm_idx_gas,
                                                                        'mdot_kg_per_s'].values[:]

            self.power_gen = gas_sink * self.conversion_factor_kgps_to_mw() * self.efficiency

        self.write_to_net(multinet)
        self.applied = True

    def write_to_net(self, multinet):
        if self.el_power_led:
            try:
                multinet['nets'][self.name_net_gas].sink.at[self.elm_idx_gas,
                                                            'mdot_kg_per_s'] = self.gas_cons
            except (ValueError, TypeError):
                multinet['nets'][self.name_net_gas].sink.loc[self.elm_idx_gas,
                                                             'mdot_kg_per_s'] = self.gas_cons
        else:
            try:
                multinet['nets'][self.name_net_power][self.elm_type_power].at[
                    self.elm_idx_power, 'p_mw'] = self.power_gen
            except (ValueError, TypeError):
                multinet['nets'][self.name_net_power][self.elm_type_power].loc[
                    self.elm_idx_power, 'p_mw'] = self.power_gen

    def is_converged(self, multinet):
        return self.applied

    def conversion_factor_kgps_to_mw(self):
        return self.fluid_calorific_value * 3600 / 1e3


class GasToGasConversion(Controller):
    """
    This controller represents a gas conversion unit (e.g. steam methane reformer or
    methanization) and couples two pandapipes-gas networks.
    Requires a corresponding sink and source in the 'gas_net_from' and 'gas_net_to', respectively.
    """

    def __init__(self, multinet, element_index_from, element_index_to, efficiency,
                 name_gas_net_from='gas1', name_gas_net_to='gas2', in_service=True, order=0,
                 level=0, drop_same_existing_ctrl=False, initial_run=True, **kwargs):
        super().__init__(multinet, in_service, order, level,
                         drop_same_existing_ctrl=drop_same_existing_ctrl, initial_run=initial_run,
                         **kwargs)

        self.element_index_from = element_index_from
        self.element_index_to = element_index_to
        self.name_net_from = name_gas_net_from
        self.name_net_to = name_gas_net_to
        self.efficiency = efficiency
        self.gas1_calorific_value = get_fluid(multinet['nets'][name_gas_net_from]).get_property('hhv')
        self.gas2_calorific_value = get_fluid(multinet['nets'][name_gas_net_to]).get_property('hhv')
        self.initial_run = initial_run
        self.applied = False

    def initialize_control(self, multinet):
        self.applied = False

    def get_all_net_names(self):
        return [self.name_net_from, self.name_net_to]

    def control_step(self, multinet):
        try:
            gas_in = multinet['nets'][self.name_net_from].sink.at[self.element_index_from, 'mdot_kg_per_s']
        except (ValueError, TypeError):
            gas_in = multinet['nets'][self.name_net_from].sink.loc[self.element_index_from, 'mdot_kg_per_s'].values

        self.mdot_kg_per_s_out = gas_in * self.conversion_factor_gas1_to_gas2() * self.efficiency
        self.write_to_net(multinet)
        self.applied = True

    def write_to_net(self, multinet):
        try:
            multinet['nets'][self.name_net_to].source.at[self.element_index_to, 'mdot_kg_per_s'] \
                = self.mdot_kg_per_s_out
        except (ValueError, TypeError):
            multinet['nets'][self.name_net_to].source.loc[self.element_index_to,
                                                          'mdot_kg_per_s'].values[:] = self.mdot_kg_per_s_out

    def is_converged(self, multinet):
        return self.applied

    def conversion_factor_gas1_to_gas2(self):
        """ideal conversion with energy conservation"""
        return self.gas1_calorific_value / self.gas2_calorific_value


def coupled_p2g_const_control(multinet, element_index_power, element_index_gas, p2g_efficiency,
                              name_power_net='power', name_gas_net='gas', profile_name=None,
                              data_source=None, scale_factor=1.0, in_service=True,
                              order=(0, 1), level=0,
                              drop_same_existing_ctrl=False, set_q_from_cosphi=False,
                              matching_params=None, initial_run=False, **kwargs):
    net_power = multinet['nets'][name_power_net]

    const = ConstControl(
        net_power, element='load', variable='p_mw', element_index=element_index_power,
        profile_name=profile_name, data_source=data_source, scale_factor=scale_factor,
        in_service=in_service, order=order[0], level=level,
        drop_same_existing_ctrl=drop_same_existing_ctrl,
        set_q_from_cosphi=set_q_from_cosphi, matching_params=matching_params,
        initial_run=initial_run, **kwargs)

    p2g = P2GControlMultiEnergy(multinet, element_index_power, element_index_gas, p2g_efficiency,
                                name_power_net, name_gas_net,
                                in_service, order[1], level,
                                drop_same_existing_ctrl=drop_same_existing_ctrl,
                                initial_run=initial_run, **kwargs)
    return const, p2g


def coupled_g2p_const_control(multinet, element_index_power, element_index_gas, g2p_efficiency=0.6,
                              name_power_net='power', name_gas_net='gas', element_type_power="sgen",
                              profile_name=None, data_source=None, scale_factor=1.0,
                              in_service=True, order=(0, 1), level=0, drop_same_existing_ctrl=False,
                              matching_params=None, initial_run=False, **kwargs):
    net_gas = multinet['nets'][name_gas_net]

    const = ConstControl(
        net_gas, element='sink', variable='mdot_kg_per_s', element_index=element_index_gas,
        profile_name=profile_name, data_source=data_source,
        scale_factor=scale_factor,
        in_service=in_service, order=order[0], level=level,
        drop_same_existing_ctrl=drop_same_existing_ctrl, matching_params=matching_params,
        initial_run=initial_run, **kwargs)
    g2p = G2PControlMultiEnergy(multinet, element_index_power, element_index_gas, g2p_efficiency,
                                name_power_net, name_gas_net, element_type_power,
                                in_service, order[1], level,
                                drop_same_existing_ctrl=drop_same_existing_ctrl,
                                initial_run=initial_run, **kwargs)
    return const, g2p
