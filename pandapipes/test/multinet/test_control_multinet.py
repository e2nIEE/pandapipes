# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import copy

import numpy as np
import pandapower
import pytest
from pandapower import networks as e_nw
from pandapower.control.controller.const_control import ConstControl

import pandapipes
from pandapipes import networks as g_nw
from pandapipes.multinet.control.controller.multinet_control import P2GControlMultiEnergy, \
    G2PControlMultiEnergy, GasToGasConversion, coupled_p2g_const_control
from pandapipes.multinet.control.run_control_multinet import run_control
from pandapipes.multinet.create_multinet import create_empty_multinet, add_nets_to_multinet
from pandapipes.test import runpp_with_mark, pipeflow_with_mark


@pytest.fixture
def get_gas_example():
    net_gas = g_nw.gas_meshed_square()
    pandapipes.create_fluid_from_lib(net_gas, "hgas", overwrite=True)
    net_gas.sink.drop(index=0, inplace=True)
    net_gas.junction.pn_bar = 30
    net_gas.ext_grid.p_bar = 30
    net_gas.pipe.diameter_m = 0.8

    return net_gas


@pytest.fixture
def get_power_example_simple():
    net_power = e_nw.example_simple()
    net_power.sgen.drop(index=0, inplace=True)
    net_power.load.drop(index=0, inplace=True)
    net_power.gen.drop(index=0, inplace=True)

    return net_power


def test_p2g_single(get_gas_example, get_power_example_simple):
    """ coupling of a single element in the power and gas net each with one MulitEnergyController"""
    # get the nets
    net_gas = copy.deepcopy(get_gas_example)
    net_power = copy.deepcopy(get_power_example_simple)
    assert pandapipes.get_fluid(net_gas).name == "hgas"

    # set up multinet
    mn = create_empty_multinet("test_p2g")
    add_nets_to_multinet(mn, power=net_power, gas=net_gas)

    # add components to represent P2G unit
    p_p2g_el = 50
    p2g_id_el = pandapower.create_load(net_power, 6, p_mw=p_p2g_el, name="power to gas consumption")
    p2g_id_gas = pandapipes.create_source(net_gas, 1, 0, name="power to gas feed in")

    # add coupling controller
    eta = 0.5
    P2GControlMultiEnergy(mn, p2g_id_el, p2g_id_gas, efficiency=eta)

    run_control(mn)

    # nets must not be changed
    assert mn.nets["power"] == net_power
    assert mn.nets["gas"] == net_gas

    # check P2G result
    assert net_gas.source.at[p2g_id_gas, "mdot_kg_per_s"] == \
           net_gas.res_source.at[p2g_id_gas, "mdot_kg_per_s"]
    assert np.isclose(net_gas.source.at[p2g_id_gas, "mdot_kg_per_s"],
                      (p_p2g_el / (net_gas.fluid.get_property('hhv') * 3.6)) * eta)
    assert net_power.load.at[p2g_id_el, "p_mw"] == p_p2g_el  # has to be still the same

    # check scaling functionality
    scaling_factor = 0.5
    net_power.load.loc[p2g_id_el, 'scaling'] = scaling_factor
    run_control(mn)
    assert np.isclose(net_gas.source.at[p2g_id_gas, "mdot_kg_per_s"],
                      p_p2g_el * scaling_factor / (net_gas.fluid.get_property('hhv') * 3.6) * eta)
    assert net_power.load.at[p2g_id_el, "p_mw"] == p_p2g_el  # has to be still the same


def test_g2p_single(get_gas_example, get_power_example_simple):
    """ coupling of a single element in the power and gas net each with one MulitEnergyController"""
    # get the nets
    net_gas = copy.deepcopy(get_gas_example)
    net_power = copy.deepcopy(get_power_example_simple)
    assert pandapipes.get_fluid(net_gas).name == "hgas"

    # set up multinet
    mn = create_empty_multinet("test_g2p")
    add_nets_to_multinet(mn, power=net_power, gas=net_gas)

    # add components to represent G2P unit
    gas_cons_kg_per_s = 0.5
    g2p_id_gas = pandapipes.create_sink(net_gas, 1, mdot_kg_per_s=gas_cons_kg_per_s,
                                        name="gas to power consumption")
    g2p_id_el = pandapower.create_sgen(net_power, 6, 0, name="gas to power feed in")

    # add coupling controller
    eta = 0.4
    G2PControlMultiEnergy(mn, g2p_id_el, g2p_id_gas, efficiency=eta, element_type_power="sgen")

    run_control(mn)

    # nets must not be changed
    assert mn.nets["power"] == net_power
    assert mn.nets["gas"] == net_gas

    # check G2P result
    assert net_power.sgen.at[g2p_id_el, "p_mw"] == \
           net_power.res_sgen.at[g2p_id_el, "p_mw"]
    assert np.isclose(net_power.sgen.at[g2p_id_el, "p_mw"],
                      gas_cons_kg_per_s * net_gas.fluid.get_property("hhv") * 3600 / 1000 * eta)
    assert net_gas.sink.at[g2p_id_gas, "mdot_kg_per_s"] == gas_cons_kg_per_s

    # check scaling functionality
    scaling_factor = 0.5
    net_gas.sink.loc[g2p_id_gas, 'scaling'] = scaling_factor
    run_control(mn)
    assert np.isclose(net_power.sgen.at[g2p_id_el, "p_mw"],
                      gas_cons_kg_per_s * scaling_factor * net_gas.fluid.get_property("hhv")
                      * 3.6 * eta)
    assert net_gas.sink.at[g2p_id_gas, "mdot_kg_per_s"] == gas_cons_kg_per_s


def test_g2g_single(get_gas_example):
    """gas-to-gas = hgas (methane) to Hydrogen conversion"""
    # get the nets
    net_gas1 = copy.deepcopy(get_gas_example)
    pandapipes.create_fluid_from_lib(net_gas1, "hgas", overwrite=True)

    net_gas2 = copy.deepcopy(get_gas_example)
    pandapipes.create_fluid_from_lib(net_gas2, "hydrogen", overwrite=True)

    # set up multinet
    mn = create_empty_multinet("test_g2g")
    add_nets_to_multinet(mn, hgas_net=net_gas1, hydrogen_net=net_gas2)

    # add components to represent G2P unit
    gas1_cons_kg_per_s = 0.5
    g2g_id_cons = pandapipes.create_sink(net_gas1, 1, mdot_kg_per_s=gas1_cons_kg_per_s,
                                         name="SMR consumption")
    g2g_id_prod = pandapipes.create_source(net_gas2, 1, 0, name="SMR production")

    # add coupling controller
    eta = 0.65
    GasToGasConversion(mn, g2g_id_cons, g2g_id_prod, efficiency=eta, name_gas_net_from='hgas_net',
                       name_gas_net_to='hydrogen_net')

    run_control(mn)

    fluid1 = pandapipes.get_fluid(net_gas1)
    fluid2 = pandapipes.get_fluid(net_gas2)

    # nets must not be changed
    assert mn.nets["hgas_net"] == net_gas1
    assert mn.nets["hydrogen_net"] == net_gas2

    # check G2G result
    assert net_gas1.sink.at[g2g_id_cons, "mdot_kg_per_s"] == \
           net_gas1.res_sink.at[g2g_id_cons, "mdot_kg_per_s"]
    assert net_gas1.sink.at[g2g_id_cons, "mdot_kg_per_s"] == gas1_cons_kg_per_s
    assert np.isclose(net_gas2.source.at[g2g_id_prod, "mdot_kg_per_s"],
                      (gas1_cons_kg_per_s * fluid1.all_properties["hhv"].value
                       / fluid2.all_properties["hhv"].value) * eta)

    # check scaling functionality
    scaling_factor = 0.5
    net_gas1.sink.loc[g2g_id_cons, 'scaling'] = scaling_factor
    run_control(mn)
    assert np.isclose(net_gas2.source.at[g2g_id_prod, "mdot_kg_per_s"],
                      (gas1_cons_kg_per_s * scaling_factor * fluid1.all_properties["hhv"].value
                       / fluid2.all_properties["hhv"].value) * eta)


def test_p2g_multiple(get_gas_example, get_power_example_simple):
    """ coupling of multiple elements with one MulitEnergyController"""
    # get the nets
    net_gas = copy.deepcopy(get_gas_example)
    net_power = copy.deepcopy(get_power_example_simple)
    assert pandapipes.get_fluid(net_gas).name == "hgas"

    # set up multinet
    mn = create_empty_multinet("test_p2g")
    add_nets_to_multinet(mn, power=net_power, gas=net_gas)

    # dummy component for offset in load/source indices:
    _ = pandapower.create_load(net_power, 0, p_mw=0.01)
    no_p2g = pandapipes.create_sources(net_gas, [0, 3], mdot_kg_per_s=0.001)
    # add components to represent P2G unit
    p_p2g_el = 10
    p2g_ids_el = pandapower.create_loads(net_power, range(1, 6), p_mw=p_p2g_el,
                                         name="power to gas consumption")
    p2g_ids_gas = pandapipes.create_sources(net_gas, range(5), 0, name="power to gas feed in")

    # add coupling controller
    eta = 0.5
    P2GControlMultiEnergy(mn, p2g_ids_el, p2g_ids_gas, efficiency=eta)

    # run control should read/write values with .loc
    run_control(mn)

    # nets must not be changed
    assert mn.nets["power"] == net_power
    assert mn.nets["gas"] == net_gas

    # check P2G result
    assert np.all(net_gas.source.loc[p2g_ids_gas, "mdot_kg_per_s"] ==
                  net_gas.res_source.loc[p2g_ids_gas, "mdot_kg_per_s"])
    assert np.allclose(net_gas.source.loc[p2g_ids_gas, "mdot_kg_per_s"],
                       (p_p2g_el / (net_gas.fluid.get_property('hhv') * 3.6)) * eta)
    assert np.all(net_gas.source.loc[no_p2g, "mdot_kg_per_s"] == 0.001)
    assert np.all(net_power.load.loc[p2g_ids_el, "p_mw"] == p_p2g_el)  # has to be still the same

    # check scaling functionality
    scaling_factor = 0.5
    net_power.load.loc[p2g_ids_el, 'scaling'] = scaling_factor
    run_control(mn)

    assert np.allclose(net_gas.source.loc[p2g_ids_gas, "mdot_kg_per_s"],
                       p_p2g_el * scaling_factor / (net_gas.fluid.get_property('hhv') * 3.6) * eta)


def test_g2p_multiple(get_gas_example, get_power_example_simple):
    """ coupling of multiple elements with one MulitEnergyController"""
    # get the nets
    net_gas = copy.deepcopy(get_gas_example)
    net_power = copy.deepcopy(get_power_example_simple)
    fluid = pandapipes.get_fluid(net_gas)
    assert fluid.name == "hgas"

    # set up multinet
    mn = create_empty_multinet("test_g2p")
    add_nets_to_multinet(mn, power=net_power, gas=net_gas)

    # dummy component for offset in load/source indices:
    _ = pandapower.create_sgen(net_power, 0, p_mw=0.01)
    no_g2p = pandapipes.create_sinks(net_gas, [0, 3], mdot_kg_per_s=0.001)

    # add components to represent G2P unit
    gas_cons_kg_per_s = 0.5
    g2p_ids_gas = pandapipes.create_sinks(net_gas, range(1, 3), mdot_kg_per_s=gas_cons_kg_per_s,
                                          name="gas to power consumption")
    g2p_ids_el = pandapower.create_sgens(net_power, range(4, 6), 0, name="gas to power feed in")

    # add coupling controller
    eta = 0.4
    G2PControlMultiEnergy(mn, g2p_ids_el, g2p_ids_gas, efficiency=eta, element_type_power="sgen")

    run_control(mn)

    # nets must not be changed
    assert mn.nets["power"] == net_power
    assert mn.nets["gas"] == net_gas

    # check G2P result
    assert np.all(net_power.sgen.loc[g2p_ids_el, "p_mw"] ==
                  net_power.res_sgen.loc[g2p_ids_el, "p_mw"])
    assert np.allclose(net_power.sgen.loc[g2p_ids_el, "p_mw"],
                       gas_cons_kg_per_s * fluid.all_properties["hhv"].value * 3600 / 1000 * eta)
    assert np.all(net_gas.sink.loc[g2p_ids_gas, "mdot_kg_per_s"] == gas_cons_kg_per_s)
    assert np.all(net_gas.sink.loc[no_g2p, "mdot_kg_per_s"] == 0.001)

    # check scaling functionality
    scaling_factor = 0.5
    net_gas.sink.loc[g2p_ids_gas, 'scaling'] = scaling_factor
    run_control(mn)

    assert np.allclose(net_power.sgen.loc[g2p_ids_el, "p_mw"],
                       gas_cons_kg_per_s * scaling_factor * fluid.all_properties["hhv"].value
                       * 3.6 * eta)


def test_g2g_multiple(get_gas_example):
    """ coupling of multiple elements in two gas grids with one MulitEnergyController
        gas-to-gas = e.g. hgas (methane) to Hydrogen conversion (SMR)"""
    # get the nets
    net_gas1 = copy.deepcopy(get_gas_example)
    pandapipes.create_fluid_from_lib(net_gas1, "hgas", overwrite=True)

    net_gas2 = copy.deepcopy(get_gas_example)
    pandapipes.create_fluid_from_lib(net_gas2, "hydrogen", overwrite=True)

    # set up multinet
    mn = create_empty_multinet("test_g2g")
    add_nets_to_multinet(mn, hgas_net=net_gas1, hydrogen_net=net_gas2)

    # dummy component for offset in sink/source indices:
    _ = pandapipes.create_sink(net_gas1, 0, mdot_kg_per_s=0.001)
    no_g2g = pandapipes.create_sources(net_gas2, [0, 3], mdot_kg_per_s=0.0314)

    # add components to represent G2P unit
    gas1_cons_kg_per_s = 0.05
    g2g_ids_cons = pandapipes.create_sinks(net_gas1, range(1, 4), mdot_kg_per_s=gas1_cons_kg_per_s,
                                           name="SMR consumption")
    g2g_ids_prod = pandapipes.create_sources(net_gas2, [0, 2, 5], 0, name="SMR production")

    # add coupling controller
    eta = 0.65
    GasToGasConversion(mn, g2g_ids_cons, g2g_ids_prod, efficiency=eta, name_gas_net_from='hgas_net',
                       name_gas_net_to='hydrogen_net')

    run_control(mn)

    fluid1 = pandapipes.get_fluid(net_gas1)
    fluid2 = pandapipes.get_fluid(net_gas2)

    # nets must not be changed
    assert mn.nets["hgas_net"] == net_gas1
    assert mn.nets["hydrogen_net"] == net_gas2

    # check G2G result
    assert np.all(net_gas1.sink.loc[g2g_ids_cons, "mdot_kg_per_s"] ==
                  net_gas1.res_sink.loc[g2g_ids_cons, "mdot_kg_per_s"])
    assert np.all(net_gas1.sink.loc[g2g_ids_cons, "mdot_kg_per_s"] == gas1_cons_kg_per_s)
    assert np.all(net_gas2.source.loc[no_g2g, "mdot_kg_per_s"] == 0.0314)
    assert np.allclose(net_gas2.source.loc[g2g_ids_prod, "mdot_kg_per_s"],
                       gas1_cons_kg_per_s * fluid1.all_properties["hhv"].value
                       / fluid2.all_properties["hhv"].value * eta)

    # check scaling functionality
    scaling_factor = 0.5
    net_gas1.sink.loc[g2g_ids_cons, 'scaling'] = scaling_factor
    run_control(mn)
    assert np.allclose(net_gas2.source.loc[g2g_ids_prod, "mdot_kg_per_s"],
                       gas1_cons_kg_per_s * scaling_factor * fluid1.all_properties["hhv"].value
                       / fluid2.all_properties["hhv"].value * eta)


def test_const_p2g_control(get_gas_example, get_power_example_simple):
    net_gas = get_gas_example
    net_power = get_power_example_simple

    flow_gas = 0.003
    pandapipes.create_source(net_gas, 5, flow_gas)
    pandapipes.create_sink(net_gas, 3, flow_gas)
    pandapipes.create_sink(net_gas, 4, flow_gas)

    power_load = 0.004
    pandapower.create_load(net_power, 6, power_load)
    pandapower.create_load(net_power, 5, power_load)
    pandapower.create_sgen(net_power, 4, power_load)

    mn = create_empty_multinet('coupled net')

    add_nets_to_multinet(mn, power=net_power, gas=net_gas)

    _, p2g = coupled_p2g_const_control(mn, 0, 0, 0.6, initial_run=True)
    ConstControl(net_gas, 'sink', 'mdot_kg_per_s', [0, 1])
    ConstControl(net_power, 'load', 'p_mw', 1)
    ConstControl(net_power, 'sgen', 'p_mw', 0)

    run_control(mn)

    assert np.all(net_power.res_load.p_mw.values == power_load)
    assert np.all(net_gas.res_sink.values == flow_gas)
    assert net_gas.source.mdot_kg_per_s.values == power_load * p2g.conversion_factor_mw_to_kgps() \
           * p2g.efficiency


def test_run_control_wo_controller(get_gas_example, get_power_example_simple):
    net_gas = get_gas_example
    net_power = get_power_example_simple

    flow_gas = 0.003
    pandapipes.create_source(net_gas, 5, flow_gas)
    pandapipes.create_sink(net_gas, 3, flow_gas)
    pandapipes.create_sink(net_gas, 4, flow_gas)

    power_load = 0.004
    pandapower.create_load(net_power, 6, power_load)
    pandapower.create_load(net_power, 5, power_load)
    pandapower.create_sgen(net_power, 4, power_load)

    mn = create_empty_multinet('coupled net')

    add_nets_to_multinet(mn, power=net_power, gas=net_gas)

    run_control(mn)


def test_p2g_single_run_parameter(get_gas_example, get_power_example_simple):
    """ coupling of a single element in the power and gas net each with one MulitEnergyController"""
    # get the nets
    net_gas = copy.deepcopy(get_gas_example)
    net_power = copy.deepcopy(get_power_example_simple)
    assert pandapipes.get_fluid(net_gas).name == "hgas"

    # set up multinet
    mn = create_empty_multinet("test_p2g")
    add_nets_to_multinet(mn, power=net_power, gas=net_gas)

    # add components to represent P2G unit
    p_p2g_el = 50
    p2g_id_el = pandapower.create_load(net_power, 6, p_mw=p_p2g_el, name="power to gas consumption")
    p2g_id_gas = pandapipes.create_source(net_gas, 1, 0, name="power to gas feed in")

    # add coupling controller
    eta = 0.5
    P2GControlMultiEnergy(mn, p2g_id_el, p2g_id_gas, efficiency=eta)

    run_control(mn, ctrl_variables={"nets": {"power": {"run": runpp_with_mark},
                                             "gas": {"run": pipeflow_with_mark}}})

    assert net_power["mark"] == "runpp"
    assert net_gas["mark"] == "pipeflow"


if __name__ == '__main__':
    pytest.main(['-xs', __file__])
