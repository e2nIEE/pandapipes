# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
import copy

import pytest
import pandapipes


@pytest.fixture
def create_empty_net():
    return pandapipes.create_empty_network()


def test_create_network():
    net = pandapipes.create_empty_network(fluid=3)
    try:
        pandapipes.get_fluid(net)
        assert False, "Shouldn't have any fluid!"
    except UserWarning:
        assert True


def test_create_junction(create_empty_net):
    net = copy.deepcopy(create_empty_net)
    pandapipes.create_junction(net, 1, 293, index=8)
    try:
        pandapipes.create_junction(net, 1, 293, index=8)
        assert False, "Shouldn't make junction!"
    except UserWarning:
        assert True
    try:
        pandapipes.create_junction(net, 1, 293, geodata=(1, 2, 3))
        assert False, "Shouldn't make junction!"
    except UserWarning:
        assert True


def test_create_sink(create_empty_net):
    net = copy.deepcopy(create_empty_net)
    pandapipes.create_junction(net, 1, 293, index=8)
    pandapipes.create_junction(net, 1, 293, index=9)
    pandapipes.create_sink(net, 9, mdot_kg_per_s=0.1, index=2)

    try:
        pandapipes.create_sink(net, junction=10, mdot_kg_per_s=0.1)
        assert False, "Shouldn't make sink!"
    except UserWarning:
        assert True
    try:
        pandapipes.create_sink(net, junction=9, mdot_kg_per_s=0.1, index=2)
        assert False, "Shouldn't make sink!"
    except UserWarning:
        assert True


def test_create_source(create_empty_net):
    net = copy.deepcopy(create_empty_net)
    pandapipes.create_junction(net, 1, 293, index=8)
    pandapipes.create_junction(net, 1, 293, index=9)
    pandapipes.create_source(net, 9, mdot_kg_per_s=0.1, index=2)

    try:
        pandapipes.create_source(net, junction=10, mdot_kg_per_s=0.1)
        assert False, "Shouldn't make source!"
    except UserWarning:
        assert True
    try:
        pandapipes.create_source(net, junction=9, mdot_kg_per_s=0.1, index=2)
        assert False, "Shouldn't make source!"
    except UserWarning:
        assert True


def test_create_ext_grid(create_empty_net):
    net = copy.deepcopy(create_empty_net)
    pandapipes.create_junction(net, 1, 293, index=8)
    pandapipes.create_junction(net, 1, 293, index=9)
    pandapipes.create_ext_grid(net, 9, p_bar=1, t_k=295, index=2)

    try:
        pandapipes.create_ext_grid(net, junction=10, p_bar=1, t_k=295)
        assert False, "Shouldn't make ext_grid!"
    except UserWarning:
        assert True
    try:
        pandapipes.create_ext_grid(net, junction=9, p_bar=1, t_k=295, index=2)
        assert False, "Shouldn't make ext_grid!"
    except UserWarning:
        assert True


def test_create_heat_exchanger(create_empty_net):
    net = copy.deepcopy(create_empty_net)
    pandapipes.create_junction(net, 1, 293, index=8, geodata=(0, 1))
    pandapipes.create_junction(net, 1, 293, index=9, geodata=(2, 2))
    pandapipes.create_heat_exchanger(net, 8, 9, 0.3, qext_w=200, index=2)

    try:
        pandapipes.create_heat_exchanger(net, 8, 10, 0.3, qext_w=200)
        assert False, "Shouldn't make heat exchanger!"
    except UserWarning:
        assert True
    try:
        pandapipes.create_heat_exchanger(net, 8, 9, 0.3, qext_w=200, index=2)
        assert False, "Shouldn't make heat exchanger!"
    except UserWarning:
        assert True


def test_create_pipe(create_empty_net):
    net = copy.deepcopy(create_empty_net)
    pandapipes.create_junction(net, 1, 293, index=8, geodata=(0, 1))
    pandapipes.create_junction(net, 1, 293, index=9, geodata=(2, 2))
    pandapipes.create_pipe(net, 8, 9, "80_GGG", 0.3, index=2, geodata=[(0, 1), (1, 1), (2, 2)])

    try:
        pandapipes.create_pipe(net, 8, 9, "80_GGG", 0.3, index=2)
        assert False, "Shouldn't make pipe, index exists!"
    except UserWarning:
        assert True
    try:
        pandapipes.create_pipe(net, 8, 10, "80_GGG", 0.3)
        assert False, "Shouldn't make pipe, non-existent to_junction!"
    except UserWarning:
        assert True
    try:
        pandapipes.create_pipe(net, 8, 9, "blah", 0.3)
        assert False, "Shouldn't make pipe, non-existent std_type!"
    except UserWarning:
        assert True

    net2 = pandapipes.create_empty_network(fluid="hgas", add_stdtypes=False)
    pandapipes.create_junction(net2, 1, 293, index=8, geodata=(0, 1))
    pandapipes.create_junction(net2, 1, 293, index=9, geodata=(2, 2))
    try:
        pandapipes.create_pipe(net2, 8, 9, "80_GGG", 0.3)
        assert False, "Shouldn't make pipe, no std_types in net!"
    except UserWarning:
        assert True


def test_create_pipe_from_parameters(create_empty_net):
    net = copy.deepcopy(create_empty_net)
    pandapipes.create_junction(net, 1, 293, index=8, geodata=(0, 1))
    pandapipes.create_junction(net, 1, 293, index=9, geodata=(2, 2))
    pandapipes.create_pipe_from_parameters(net, 8, 9, 0.3, 0.4, index=2,
                                           geodata=[(0, 1), (1, 1), (2, 2)])

    try:
        pandapipes.create_pipe_from_parameters(net, 8, 9, 0.3, 0.4, index=2)
        assert False, "Shouldn't make pipe, index exists!"
    except UserWarning:
        assert True
    try:
        pandapipes.create_pipe_from_parameters(net, 8, 10, 0.3, 0.4)
        assert False, "Shouldn't make pipe, non-existent to_junction!"
    except UserWarning:
        assert True
    try:
        pandapipes.create_pipe_from_parameters(net, 8, 9, 0.3, 0.4, std_type="blah")
        assert False, "Shouldn't make pipe, wrong statement!"
    except UserWarning:
        assert True


def test_create_valve(create_empty_net):
    net = copy.deepcopy(create_empty_net)
    pandapipes.create_junction(net, 1, 293, index=8, geodata=(0, 1))
    pandapipes.create_junction(net, 1, 293, index=9, geodata=(2, 2))
    pandapipes.create_valve(net, 8, 9, 0.4, True, index=2)

    try:
        pandapipes.create_valve(net, 8, 9, 0.4, True, index=2)
        assert False, "Shouldn't make valve, index exists!"
    except UserWarning:
        assert True
    try:
        pandapipes.create_valve(net, 8, 10, 0.4, True)
        assert False, "Shouldn't make valve, non-existent to_junction!"
    except UserWarning:
        assert True
    try:
        pandapipes.create_valve(net, 8, 9, 0.4, True, geodata=[(0, 1), (1, 1), (2, 2)])
        assert False, "Shouldn't make valve, geodata not possible!"
    except ValueError:
        assert True


def test_create_pump(create_empty_net):
    net = copy.deepcopy(create_empty_net)
    pandapipes.create_junction(net, 1, 293, index=8, geodata=(0, 1))
    pandapipes.create_junction(net, 1, 293, index=9, geodata=(2, 2))
    pandapipes.create_pump(net, 8, 9, "P1", index=2)

    try:
        pandapipes.create_pump(net, 8, 9, "P1", index=2)
        assert False, "Shouldn't make pump, index exists!"
    except UserWarning:
        assert True
    try:
        pandapipes.create_pump(net, 8, 10, "P1")
        assert False, "Shouldn't make pump, non-existent to_junction!"
    except UserWarning:
        assert True
    try:
        pandapipes.create_pump(net, 8, 9, "blah")
        assert False, "Shouldn't make pump, non-existent std_type!"
    except UserWarning:
        assert True
    try:
        pandapipes.create_pump(net, 8, 9, "P1", geodata=[(0, 1), (1, 1), (2, 2)])
        assert False, "Shouldn't make pump, geodata not possible!"
    except ValueError:
        assert True

    net2 = pandapipes.create_empty_network(fluid="hgas", add_stdtypes=False)
    pandapipes.create_junction(net2, 1, 293, index=8, geodata=(0, 1))
    pandapipes.create_junction(net2, 1, 293, index=9, geodata=(2, 2))
    try:
        pandapipes.create_pump(net2, 8, 9, "P1")
        assert False, "Shouldn't make pump, no std_types in net!"
    except UserWarning:
        assert True


def test_create_pump_from_parameters(create_empty_net):
    net = copy.deepcopy(create_empty_net)
    pandapipes.create_junction(net, 1, 293, index=8, geodata=(0, 1))
    pandapipes.create_junction(net, 1, 293, index=9, geodata=(2, 2))
    pandapipes.create_pump_from_parameters(net, 8, 9, "pump1", pressure_list=[0, 1, 2, 3],
                                           flowrate_list=[0, 1, 2, 3], regression_degree=1, index=2)

    try:
        pandapipes.create_pump_from_parameters(net, 8, 9, "pump1", pressure_list=[0, 1, 2, 3],
                                               flowrate_list=[0, 1, 2, 3], regression_degree=1,
                                               index=2)
        assert False, "Shouldn't make pump, index exists!"
    except UserWarning:
        assert True
    try:
        pandapipes.create_pump_from_parameters(net, 8, 10, "pump1", pressure_list=[0, 1, 2, 3],
                                               flowrate_list=[0, 1, 2, 3], regression_degree=1,
                                               index=2)
        assert False, "Shouldn't make pump, non-existent to_junction!"
    except UserWarning:
        assert True
    try:
        pandapipes.create_pump_from_parameters(net, 8, 9, "pump1", pressure_list=[0, 1, 2, 3],
                                               flowrate_list=[0, 1, 2, 3], regression_degree=1,
                                               geodata=[(0, 1), (1, 1), (2, 2)])
        assert False, "Shouldn't make pump, geodata not possible!"
    except ValueError:
        assert True


if __name__ == '__main__':
    pytest.main(["test_create.py"])
