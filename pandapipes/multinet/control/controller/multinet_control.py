# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

from pandapower.control import ConstControl
from pandapipes.properties.fluids import get_fluid
from pandapower.control.basic_controller import Controller
from pandas.errors import InvalidIndexError


class P2GControlMultiEnergy(Controller):

    """
    A controller to be used in a multinet. Converts power consumption to gas production.

    This controller couples a power network (from pandapower) and a gas network (from
    pandapipes) that are stored in a multinet. Requires one or multiple 'load' elements in the
    power net and as many corresponding 'source' elements in the gas net. It reads the power load
    values for given 'load' elements, applies the efficiency factor and unit conversions and
    writes the resulting gas mass flow to 'source' elements in the gas net.
    It is stored in the controller-DataFrame of the multinet (multinet.controller).
    It is run by run_control_multinet.run_control() or within
    run_time_series_multinet.run_timeseries().

    :param multinet: pandapipes-Mulitnet that includes the power and gas network with load and
                     source elements
    :type multinet: attrdict (pandapipes.MultiNet)
    :param element_index_power: Index of one or more load elements in the power net from which
                                the power consumption will be read. For each load element,
                                a corresponding source element has to be provided in
                                element_index_gas.
    :type element_index_power: int or iterable of integers
    :param element_index_gas: Index of one or more source elements in the gas net to which the
                              calculated mass flow will be written. For each source element,
                              a corresponing el. load element has to be provided in
                              element_index_power.
    :type element_index_gas: int or iterable of integers
    :param efficiency: constant efficiency factor (default: based on HHV)
    :type efficiency: float
    :param name_power_net: Key name to find the power net in multinet['nets']
    :type name_power_net: str
    :param name_gas_net: Key name to find the gas net in multinet['nets']
    :type name_gas_net: str
    :param in_service: Indicates if the controllers are currently in_service
    :type in_service: bool
    :param order: within the same level, controllers with lower order are called before controllers
                  with numerical higher order
    :type order: real
    :param level: level to which the controller belongs. Low level is called before higher level.
                  Respective run function for the nets are called at least once per level.
    :type level: real
    :param drop_same_existing_ctrl: Indicates if already existing controllers of the same type and
                                    with the same matching parameters (e.g. at same element) should
                                    be dropped
    :type drop_same_existing_ctrl: bool
    :param initial_run: Whether a power and pipe flow should be run before the control step is
                        applied or not
    :type initial_run: bool
    :param kwargs: optional additional controller arguments that were implemented by users
    :type kwargs: any
    """
    def __init__(self, multinet, element_index_power, element_index_gas, efficiency,
                 name_power_net='power', name_gas_net='gas',
                 in_service=True, order=0, level=0,
                 drop_same_existing_ctrl=False, initial_run=True, **kwargs):
        """
        see class docstring
        """
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
        self.applied = False

    def initialize_control(self, multinet):
        self.applied = False

    def get_all_net_names(self):
        return [self.name_net_power, self.name_net_gas]

    def control_step(self, multinet):
        try:
            power_load = \
                multinet['nets'][self.name_net_power].load.at[self.elm_idx_power, 'p_mw'] \
                * multinet['nets'][self.name_net_power].load.at[self.elm_idx_power, 'scaling']
        except (ValueError, TypeError, InvalidIndexError):
            power_load = \
                multinet['nets'][self.name_net_power].load.loc[self.elm_idx_power, 'p_mw'].values \
                * multinet['nets'][self.name_net_power].load.loc[self.elm_idx_power,
                                                                 'scaling'].values
        self.mdot_kg_per_s = power_load * self.conversion_factor_mw_to_kgps() * self.efficiency
        self.write_to_net(multinet)
        self.applied = True

    def write_to_net(self, multinet):
        try:
            multinet['nets'][self.name_net_gas].source.at[self.elm_idx_gas, 'mdot_kg_per_s'] \
                = self.mdot_kg_per_s
        except (ValueError, TypeError, InvalidIndexError):
            multinet['nets'][self.name_net_gas].source.loc[self.elm_idx_gas,
                                                           'mdot_kg_per_s'] = self.mdot_kg_per_s

    def is_converged(self, multinet):
        return self.applied

    def conversion_factor_mw_to_kgps(self):
        return 1e3 / (self.fluid_calorific_value * 3600)


class G2PControlMultiEnergy(Controller):

    """
    A controller to be used in a multinet. Connects power generation and gas consumption.

    This controller couples a gas network (from pandapipes) and a power network (from
    pandapower) that are stored in a multinet. Requires one or multiple 'sink' elements in the gas
    net and as many corresponding 'sgen'/'gen' elements in the power net.
    If 'calc_gas_from_power' is False (default), it reads the gas mass consumption values
    of given 'sink' elements, applies the efficiency factor and unit conversions and writes the
    resulting power output to 'sgen' (default) or 'gen' elements in the power net.
    If 'calc_gas_from_power' is True, it reads the power output of
    given 'sgen' (default) or 'gen' elements, calculates the corresponding gas consumption by
    appling the efficiency factor and unit conversions, and writes the resulting gas consumption
    mass flow to the given 'sink' elements in the gas net.
    It is stored in the controller-DataFrame of the multinet (multinet.controller).
    It is run by run_control_multinet.run_control() or within
    run_time_series_multinet.run_timeseries().

    :param multinet: pandapipes-Mulitnet that includes the power and gas network with sgen/gen and
                     sink elements
    :type multinet: attrdict (pandapipes.MultiNet)
    :param element_index_power: Index of one or more elements in the power net from which
                                the power generation will be read from or written to (either
                                'sgen' or 'gen' elements, as defined by element_type_power).
                                For each entry, a corresponding gas sink element has to be
                                provided in element_index_gas.
    :type element_index_power: int or iterable of integers
    :param element_index_gas: Index of one or more sink elements in the gas net from which the
                              G2P units' gas consumption (mass flow) is read from or written to.
                              For each sink element, a corresponding sgen/gen element has to be
                              provided in element_index_power.
    :type element_index_gas: int or iterable of integers
    :param efficiency: constant efficiency factor (default: based on HHV)
    :type efficiency: float
    :param name_power_net: Key name to find the power net in multinet['nets']
    :type name_power_net: str
    :param name_gas_net: Key name to find the gas net in multinet['nets']
    :type name_gas_net: str
    :param element_type_power: type of the corresponding power generation units, either 'sgen' or \
        'gen'
    :type element_type_power: str
    :param in_service: Indicates if the controllers are currently in_service
    :type in_service: bool
    :param order: within the same level, controllers with lower order are called before controllers
                  with numerical higher order
    :type order: int or float
    :param level: level to which the controller belongs. Low level is called before higher level.
                  Respective run function for the nets are called at least once per level.
    :type level: int or float
    :param drop_same_existing_ctrl: Indicates if already existing controllers of the same type and
                                    with the same matching parameters (e.g. at same element) should
                                    be dropped
    :type drop_same_existing_ctrl: bool
    :param initial_run: Whether a power and pipe flow should be run before the control step is
                        applied or not
    :type initial_run: bool
    :param calc_gas_from_power: (default: False) If False, the power output will be calculated on
                                the basis of the sink's gas consumption. If True, the gas
                                consumption will be calculated on the basis of the generator's power
                                output.
    :type calc_gas_from_power: bool
    :param kwargs: optional additional controller arguments that were implemented by users
    :type kwargs: any
    """

    def __init__(self, multinet, element_index_power, element_index_gas, efficiency,
                 name_power_net='power', name_gas_net='gas', element_type_power="sgen",
                 in_service=True, order=0,
                 level=0, drop_same_existing_ctrl=False, initial_run=True,
                 calc_gas_from_power=False, **kwargs):
        """
        see class docstring
        """
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
                    self.elm_idx_power, 'p_mw'] * multinet['nets'][self.name_net_power][
                    self.elm_type_power].at[self.elm_idx_power, 'scaling']

            except (ValueError, TypeError, InvalidIndexError):
                power_gen = multinet['nets'][self.name_net_power][self.elm_type_power].loc[
                                self.elm_idx_power, 'p_mw'].values[:] \
                            * multinet['nets'][self.name_net_power][self.elm_type_power].loc[
                                self.elm_idx_power, 'scaling'].values[:]

            self.gas_cons = power_gen / (self.conversion_factor_kgps_to_mw() * self.efficiency)

        else:
            try:
                gas_sink = \
                    multinet['nets'][self.name_net_gas].sink.at[self.elm_idx_gas, 'mdot_kg_per_s'] \
                    * multinet['nets'][self.name_net_gas].sink.at[self.elm_idx_gas, 'scaling']

            except (ValueError, TypeError, InvalidIndexError):
                gas_sink = multinet['nets'][self.name_net_gas].sink.loc[self.elm_idx_gas,
                                                                        'mdot_kg_per_s'].values[:] \
                           * multinet['nets'][self.name_net_gas].sink.loc[self.elm_idx_gas,
                                                                          'scaling'].values[:]

            self.power_gen = gas_sink * self.conversion_factor_kgps_to_mw() * self.efficiency

        self.write_to_net(multinet)
        self.applied = True

    def write_to_net(self, multinet):
        if self.el_power_led:
            try:
                multinet['nets'][self.name_net_gas].sink.at[self.elm_idx_gas,
                                                            'mdot_kg_per_s'] = self.gas_cons
            except (ValueError, TypeError, InvalidIndexError):
                multinet['nets'][self.name_net_gas].sink.loc[self.elm_idx_gas,
                                                             'mdot_kg_per_s'] = self.gas_cons
        else:
            try:
                multinet['nets'][self.name_net_power][self.elm_type_power].at[
                    self.elm_idx_power, 'p_mw'] = self.power_gen
            except (ValueError, TypeError, InvalidIndexError):
                multinet['nets'][self.name_net_power][self.elm_type_power].loc[
                    self.elm_idx_power, 'p_mw'] = self.power_gen

    def is_converged(self, multinet):
        return self.applied

    def conversion_factor_kgps_to_mw(self):
        return self.fluid_calorific_value * 3600 / 1e3


class GasToGasConversion(Controller):

    """
    A controller to be used in a multinet with two gas nets that have different gases.

    This controller represents a gas conversion unit (e.g. methanization or steam methane reformer)
    and couples two pandapipes-gas networks that are stored together in a multinet.
    Requires one or multiple sinks in one net ('gas_net_from') and as many corresponding sources
    in the other net ('gas_net_to').
    It reads the gas consumption values for given 'sink' elements in one gas net, applies the
    efficiency factor and writes the resulting gas mass flow to 'source' elements in the other
    gas net.
    It is stored in the controller-DataFrame of the multinet (multinet.controller).
    It is run by run_control_multinet.run_control() or within
    run_time_series_multinet.run_timeseries().

    :param multinet: pandapipes-Mulitnet that includes the gas networks that will be coupled with \
                     sink and source elements
    :type multinet: attrdict (pandapipes.MultiNet)
    :param element_index_from: Index of one or more sink elements in the name_gas_net_from from \
                               which the gas consumption will be read
    :type element_index_from: int or iterable of integers
    :param element_index_to: Index of one or more source elements in the name_gas_net_to to which \
                             the calculated mass flow will be written
    :type element_index_to: int or iterable of integers
    :param efficiency: constant efficiency factor (default: based on HHV)
    :type efficiency: float
    :param name_gas_net_from: Key name to find the gas net from which gas is consumed in \
                              multinet['nets']
    :type name_gas_net_from: str
    :param name_gas_net_to: Key name to find the gas net in which gas is fed into in \
                            multinet['nets']
    :type name_gas_net_to: str
    :param in_service: Indicates if the controllers are currently in_service
    :type in_service: bool
    :param order: within the same level, controllers with lower order are called before controllers
                  with numerical higher order
    :type order: int or float
    :param level: level to which the controller belongs. Low level is called before higher level.
                  Respective run function for the nets are called at least once per level.
    :type level: int or float
    :param drop_same_existing_ctrl: Indicates if already existing controllers of the same type and
                                    with the same matching parameters (e.g. at same element) should
                                    be dropped
    :type drop_same_existing_ctrl: bool
    :param initial_run: Whether a power and pipe flow should be run before the control step is
                        applied or not
    :type initial_run: bool
    :param kwargs: optional additional controller arguments that were implemented by users
    :type kwargs: any
    """

    def __init__(self, multinet, element_index_from, element_index_to, efficiency,
                 name_gas_net_from='gas1', name_gas_net_to='gas2', in_service=True, order=0,
                 level=0, drop_same_existing_ctrl=False, initial_run=True, **kwargs):
        """
        see class docstring
        """
        super().__init__(multinet, in_service, order, level,
                         drop_same_existing_ctrl=drop_same_existing_ctrl, initial_run=initial_run,
                         **kwargs)

        self.element_index_from = element_index_from
        self.element_index_to = element_index_to
        self.name_net_from = name_gas_net_from
        self.name_net_to = name_gas_net_to
        self.efficiency = efficiency
        self.gas1_calorific_value = get_fluid(
            multinet['nets'][name_gas_net_from]).get_property('hhv')
        self.gas2_calorific_value = get_fluid(
            multinet['nets'][name_gas_net_to]).get_property('hhv')
        self.applied = False

    def initialize_control(self, multinet):
        self.applied = False

    def get_all_net_names(self):
        return [self.name_net_from, self.name_net_to]

    def control_step(self, multinet):
        try:
            gas_in = multinet['nets'][self.name_net_from].sink.at[self.element_index_from,
                                                                  'mdot_kg_per_s'] \
                     * multinet['nets'][self.name_net_from].sink.at[self.element_index_from,
                                                                    'scaling']

        except (ValueError, TypeError, InvalidIndexError):
            gas_in = multinet['nets'][self.name_net_from].sink.loc[self.element_index_from,
                                                                   'mdot_kg_per_s'].values \
                     * multinet['nets'][self.name_net_from].sink.loc[self.element_index_from,
                                                                     'scaling'].values

        self.mdot_kg_per_s_out = gas_in * self.conversion_factor_gas1_to_gas2() * self.efficiency
        self.write_to_net(multinet)
        self.applied = True

    def write_to_net(self, multinet):
        try:
            multinet['nets'][self.name_net_to].source.at[self.element_index_to, 'mdot_kg_per_s'] \
                = self.mdot_kg_per_s_out
        except (ValueError, TypeError, InvalidIndexError):
            multinet['nets'][self.name_net_to].source.loc[self.element_index_to,
                                                          'mdot_kg_per_s'] = self.mdot_kg_per_s_out

    def is_converged(self, multinet):
        return self.applied

    def conversion_factor_gas1_to_gas2(self):
        """Ideal conversion with energy conservation."""
        return self.gas1_calorific_value / self.gas2_calorific_value


def coupled_p2g_const_control(multinet, element_index_power, element_index_gas, p2g_efficiency,
                              name_power_net='power', name_gas_net='gas', profile_name=None,
                              data_source=None, scale_factor=1.0, in_service=True,
                              order=(0, 1), level=0, drop_same_existing_ctrl=False,
                              matching_params=None, initial_run=False, **kwargs):
    """
    Creates a ConstController (load values) and a P2G Controller (corresponding gas mass flows).

    The ConstController updates load values of a given electric load in accordance to the profile
    given in the datasource.
    The P2GControlMultiEnergy-controller couples a power network (from pandapower) and a gas
    network (from pandapipes). It reads the power load values that were updated by the
    ConstController, applies the efficiency factor and unit conversions and writes the resulting
    gas mass flow to 'source' elements in the gas net.
    The ConstController is stored in the power net inside the multinet.
    The P2GControlMultiEnergy is stored in the controller-DataFrame of the multinet
    (multinet.controller).
    Both controllers are run by run_control_multinet.run_control() or within
    run_time_series_multinet.run_timeseries().

    :param multinet: pandapipes-Mulitnet that includes the power and gas network with load and
                     source elements
    :type multinet: attrdict (pandapipes.MultiNet)
    :param element_index_power: Index of one or more load elements in the power net to which
                                the load values from the data source will be written
    :type element_index_power: int or iterable of integers
    :param element_index_gas: Index of one or more source elements in the gas net to which the
                              corresponding calculated mass flow will be written
    :param p2g_efficiency: constant efficiency factor (default: based on HHV)
    :type p2g_efficiency: float
    :param name_power_net: Key name to find the power net in multinet['nets']
    :type name_power_net: str
    :param name_gas_net: Key name to find the gas net in multinet['nets']
    :type name_gas_net: str
    :param profile_name: The profile names of the elements in the data source
    :type profile_name: str
    :param data_source: The data source that provides profile data
    :type data_source: object
    :param scale_factor: Scaling factor for time series input values
    :type scale_factor: real
    :param in_service: Indicates if the controller is currently in_service
    :type in_service: bool
    :param order: within the same level, controllers with lower order are called before controllers
                  with numerical higher order. Default: (0, 1) -> ConstController updates values
                  before P2G controller calculates mass flow
    :type order: tuple of integers
    :param level: level to which the controllers belong. Low level is called before higher level.
                  Respective run function for the nets are called at least once per level.
    :type level: int or float
    :param drop_same_existing_ctrl: Indicates if already existing controllers of the same type and
                                    with the same matching parameters (e.g. at same element) should
                                    be dropped
    :type drop_same_existing_ctrl: bool
    :param matching_params: is required to check if same controller already exists (dropping or \
        logging)
    :type matching_params: dict
    :param initial_run: Whether a power and pipe flow should be run before the control step is
                        applied or not
    :type initial_run: bool
    :param kwargs: optional additional controller arguments that were implemented by users
    :type kwargs:
    :return: (ID of the ConstController, ID of P2G controller)
    :rtype:
    """
    net_power = multinet['nets'][name_power_net]

    const = ConstControl(
        net_power, element='load', variable='p_mw', element_index=element_index_power,
        profile_name=profile_name, data_source=data_source, scale_factor=scale_factor,
        in_service=in_service, order=order[0], level=level,
        drop_same_existing_ctrl=drop_same_existing_ctrl, matching_params=matching_params,
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
                              power_led=False, in_service=True, order=(0, 1), level=0,
                              drop_same_existing_ctrl=False, matching_params=None,
                              initial_run=False, **kwargs):
    """
    Creates a ConstController (gas consumption) and a G2P Controller (corresponding power output).

    The ConstController updates gas consumption values of a given sink element in accordance to
    the profile given in the datasource.
    The G2PControlMultiEnergy-controller couples a gas network (from pandapipes) and a power
    network (from pandapower). It reads the gas consumption values that were updated by the
    ConstController, applies the efficiency factor and unit conversions and writes the resulting
    power output to 'sgen' or 'gen' elements in the power net.
    The ConstController is stored in the gas net inside the multinet.
    The G2PControlMultiEnergy is stored in the controller-DataFrame of the multinet itself
    (multinet.controller).
    Both controllers are run by run_control_multinet.run_control() or within
    run_time_series_multinet.run_timeseries().

    :param multinet: pandapipes-Mulitnet that includes the power and gas network with load and
                     source elements
    :type multinet: attrdict (pandapipes.MultiNet)
    :param element_index_power: Index of one or more sgen/gen elements in the power net to which
                                the power generation values from the data source will be written
    :type element_index_power: int or iterable of integers
    :param element_index_gas: Index of one or more sink elements in the gas net to which the
                              corresponding calculated mass flow will be written
    :param g2p_efficiency: constant efficiency factor (default: based on HHV)
    :type g2p_efficiency: float
    :param name_power_net: Key name to find the power net in multinet['nets']
    :type name_power_net: str
    :param name_gas_net: Key name to find the gas net in multinet['nets']
    :type name_gas_net: str
    :param element_type_power: type of the corresponding power generation units, either 'sgen' or \
        'gen'
    :type element_type_power: str
    :param profile_name: The profile names of the elements in the data source
    :type profile_name: str
    :param data_source: The data source that provides profile data
    :type data_source: object
    :param scale_factor: Scaling factor for time series input values
    :type scale_factor: real
    :param in_service: Indicates if the controllers are currently in_service
    :type in_service: bool
    :param order: within the same level, controllers with lower order are called before controllers
                  with numerical higher order. Default: (0, 1) -> ConstController updates values
                  before G2P controller calculates power output
    :type order: tuple of integers
    :param level: level to which the controllers belong. Low level is called before higher level.
                  Respective run function for the nets are called at least once per level.
    :type level: real
    :param drop_same_existing_ctrl: Indicates if already existing controllers of the same type and
                                    with the same matching parameters (e.g. at same element) should
                                    be dropped
    :type drop_same_existing_ctrl: bool
    :param matching_params: is required to check if same controller already exists (dropping or \
        logging)
    :type matching_params: dict
    :param initial_run: Whether a power and pipe flow should be run before the control step is
                        applied or not
    :type initial_run: bool
    :param kwargs: optional additional controller arguments that were implemented by users
    :type kwargs:
    :return: (ID of the ConstController, ID of G2P controller)
    :rtype:
    """
    if power_led:
        net_power = multinet['nets'][name_power_net]

        const = ConstControl(
            net_power, element='sgen', variable='p_mw', element_index=element_index_power,
            profile_name=profile_name, data_source=data_source, scale_factor=scale_factor,
            in_service=in_service, order=order[0], level=level,
            drop_same_existing_ctrl=drop_same_existing_ctrl, matching_params=matching_params,
            initial_run=initial_run, **kwargs)
    else:
        net_gas = multinet['nets'][name_gas_net]
        const = ConstControl(
            net_gas, element='sink', variable='mdot_kg_per_s', element_index=element_index_gas,
            profile_name=profile_name, data_source=data_source,
            scale_factor=scale_factor, in_service=in_service, order=order[0], level=level,
            drop_same_existing_ctrl=drop_same_existing_ctrl, matching_params=matching_params,
            initial_run=initial_run, **kwargs)
    g2p = G2PControlMultiEnergy(multinet, element_index_power, element_index_gas, g2p_efficiency,
                                name_power_net, name_gas_net, element_type_power,
                                in_service, order[1], level,
                                drop_same_existing_ctrl=drop_same_existing_ctrl,
                                initial_run=initial_run, calc_gas_from_power=power_led, **kwargs)
    return const, g2p
