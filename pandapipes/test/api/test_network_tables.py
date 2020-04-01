# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import copy

import pandapipes
from pandapipes.component_models import Sink, Source, Pump, \
    HeatExchanger, Valve, CirculationPumpPressure, CirculationPumpMass


def test_default_input_tables():
    net = pandapipes.create_empty_network()

    junction_input = list(copy.deepcopy(net.junction.columns))
    pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    junction_input_create = list(net.junction.columns)
    assert junction_input == junction_input_create, "Input does not equal Table in create-function"

    junction_input = list(copy.deepcopy(net.junction.columns))
    pandapipes.create_junction(net, pn_bar=5, tfluid_k=283, hallo=2)
    junction_input_create = list(net.junction.columns)
    assert junction_input != junction_input_create, "Input equals create-table, but they shouldn't"

    pipe_input = list(copy.deepcopy(net.pipe.columns))
    pandapipes.create_pipe_from_parameters(net, 0, 1, 6, diameter_m=0.2, k_mm=.1, sections=6,
                                           alpha_w_per_m2k=5)
    pipe_input_create = list(net.pipe.columns)
    assert pipe_input == pipe_input_create, "Input does not equal Table in create-function"

    ext_grid_input = list(copy.deepcopy(net.ext_grid.columns))
    pandapipes.create_ext_grid(net, 0, p_bar=5, t_k=330, type="pt")
    ext_grid_input_create = list(net.ext_grid.columns)
    assert ext_grid_input == ext_grid_input_create, "Input does not equal Table in create-function"


def test_additional_tables():
    net = pandapipes.create_empty_network()
    pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)

    pandapipes.add_new_component(net, Sink)
    sink_input = list(copy.deepcopy(net.sink.columns))
    pandapipes.create_sink(net, 1, mdot_kg_per_s=1)
    sink_input_create = list(net.sink.columns)
    assert sink_input == sink_input_create, "Input does not equal create-table"

    pandapipes.add_new_component(net, Source)
    source_input = list(copy.deepcopy(net.source.columns))
    pandapipes.create_source(net, 1, mdot_kg_per_s=1)
    source_input_create = list(net.source.columns)
    assert source_input == source_input_create, "Input does not equal create-table"

    pandapipes.add_new_component(net, Pump)
    pump_input = list(copy.deepcopy(net.pump.columns))
    pandapipes.create_pump_from_parameters(net, 0, 1, 'P4', [6.1, 5.8, 4], [0, 19, 83], 2)
    pump_input_create = list(net.pump.columns)
    assert pump_input == pump_input_create, "Input does not equal create-table"

    pandapipes.add_new_component(net, CirculationPumpMass)
    pumpcircmass_input = list(copy.deepcopy(net.circ_pump_mass.columns))
    pandapipes.create_circ_pump_const_mass_flow(net, 0, 1, 5, 5, 300, type='pt')
    pumpcircmass_input_create = list(net.circ_pump_mass.columns)
    assert pumpcircmass_input == pumpcircmass_input_create, "Input does not equal create-table"

    pandapipes.add_new_component(net, CirculationPumpPressure)
    pumpcircpres_input = list(copy.deepcopy(net.circ_pump_pressure.columns))
    pandapipes.create_circ_pump_const_pressure(net, 0, 1, 5, 2, 300, type='pt')
    pumpcircpres_input_create = list(net.circ_pump_pressure.columns)
    assert pumpcircpres_input == pumpcircpres_input_create, "Input does not equal create-table"

    pandapipes.add_new_component(net, Valve)
    valve_input = list(copy.deepcopy(net.valve.columns))
    pandapipes.create_valve(net, 0, 1, diameter_m=0.1, opened=False)
    valve_input_create = list(net.valve.columns)
    assert valve_input == valve_input_create, "Input does not equal create-table"

    pandapipes.add_new_component(net, HeatExchanger)
    hex_input = list(copy.deepcopy(net.heat_exchanger.columns))
    pandapipes.create_heat_exchanger(net, 0, 1, 0.2, qext_w=20000)
    hex_input_create = list(net.heat_exchanger.columns)
    assert hex_input == hex_input_create, "Input does not equal create-table"
