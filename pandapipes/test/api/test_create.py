# Copyright (c) 2020-2021 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import copy

import pytest
import pandapipes
import numpy as np


@pytest.fixture
def create_empty_net():
    return pandapipes.create_empty_network()


def test_create_network():
    net = pandapipes.create_empty_network(fluid=3)
    with pytest.raises(AttributeError):
        pandapipes.get_fluid(net)


def test_create_junction(create_empty_net):
    net = copy.deepcopy(create_empty_net)
    pandapipes.create_junction(net, 1, 293, index=8)
    with pytest.raises(UserWarning):
        pandapipes.create_junction(net, 1, 293, index=8)
    with pytest.raises(UserWarning):
        pandapipes.create_junction(net, 1, 293, geodata=(1, 2, 3))


def test_create_sink(create_empty_net):
    net = copy.deepcopy(create_empty_net)
    pandapipes.create_junction(net, 1, 293, index=8)
    pandapipes.create_junction(net, 1, 293, index=9)
    pandapipes.create_sink(net, 9, mdot_kg_per_s=0.1, index=2)

    with pytest.raises(UserWarning):
        pandapipes.create_sink(net, junction=10, mdot_kg_per_s=0.1)
    with pytest.raises(UserWarning):
        pandapipes.create_sink(net, junction=9, mdot_kg_per_s=0.1, index=2)


def test_create_source(create_empty_net):
    net = copy.deepcopy(create_empty_net)
    pandapipes.create_junction(net, 1, 293, index=8)
    pandapipes.create_junction(net, 1, 293, index=9)
    pandapipes.create_source(net, 9, mdot_kg_per_s=0.1, index=2)

    with pytest.raises(UserWarning):
        pandapipes.create_source(net, junction=10, mdot_kg_per_s=0.1)
    with pytest.raises(UserWarning):
        pandapipes.create_source(net, junction=9, mdot_kg_per_s=0.1, index=2)


def test_create_ext_grid(create_empty_net):
    net = copy.deepcopy(create_empty_net)
    pandapipes.create_junction(net, 1, 293, index=8)
    pandapipes.create_junction(net, 1, 293, index=9)
    pandapipes.create_ext_grid(net, 9, p_bar=1, t_k=295, index=2)

    with pytest.raises(UserWarning):
        pandapipes.create_ext_grid(net, junction=10, p_bar=1, t_k=295)
    with pytest.raises(UserWarning):
        pandapipes.create_ext_grid(net, junction=9, p_bar=1, t_k=295, index=2)


def test_create_heat_exchanger(create_empty_net):
    net = copy.deepcopy(create_empty_net)
    pandapipes.create_junction(net, 1, 293, index=8, geodata=(0, 1))
    pandapipes.create_junction(net, 1, 293, index=9, geodata=(2, 2))
    pandapipes.create_heat_exchanger(net, 8, 9, 0.3, qext_w=200, index=2)

    with pytest.raises(UserWarning):
        pandapipes.create_heat_exchanger(net, 8, 10, 0.3, qext_w=200)
    with pytest.raises(UserWarning):
        pandapipes.create_heat_exchanger(net, 8, 9, 0.3, qext_w=200, index=2)


def test_create_pipe(create_empty_net):
    net = copy.deepcopy(create_empty_net)
    pandapipes.create_junction(net, 1, 293, index=8, geodata=(0, 1))
    pandapipes.create_junction(net, 1, 293, index=9, geodata=(2, 2))
    pandapipes.create_pipe(net, 8, 9, "80_GGG", 0.3, index=2, geodata=[(0, 1), (1, 1), (2, 2)])

    with pytest.raises(UserWarning):
        pandapipes.create_pipe(net, 8, 9, "80_GGG", 0.3, index=2)
    with pytest.raises(UserWarning):
        pandapipes.create_pipe(net, 8, 10, "80_GGG", 0.3)
    with pytest.raises(UserWarning):
        pandapipes.create_pipe(net, 8, 9, "blah", 0.3)

    net2 = pandapipes.create_empty_network(fluid="hgas", add_stdtypes=False)
    pandapipes.create_junction(net2, 1, 293, index=8, geodata=(0, 1))
    pandapipes.create_junction(net2, 1, 293, index=9, geodata=(2, 2))
    with pytest.raises(UserWarning):
        pandapipes.create_pipe(net2, 8, 9, "80_GGG", 0.3)


def test_create_pipe_from_parameters(create_empty_net):
    net = copy.deepcopy(create_empty_net)
    pandapipes.create_junction(net, 1, 293, index=8, geodata=(0, 1))
    pandapipes.create_junction(net, 1, 293, index=9, geodata=(2, 2))
    pandapipes.create_pipe_from_parameters(net, 8, 9, 0.3, 0.4, index=2,
                                           geodata=[(0, 1), (1, 1), (2, 2)])

    with pytest.raises(UserWarning):
        pandapipes.create_pipe_from_parameters(net, 8, 9, 0.3, 0.4, index=2)
    with pytest.raises(UserWarning):
        pandapipes.create_pipe_from_parameters(net, 8, 10, 0.3, 0.4)
    with pytest.raises(UserWarning):
        pandapipes.create_pipe_from_parameters(net, 8, 9, 0.3, 0.4, std_type="blah")


def test_create_valve(create_empty_net):
    net = copy.deepcopy(create_empty_net)
    pandapipes.create_junction(net, 1, 293, index=8, geodata=(0, 1))
    pandapipes.create_junction(net, 1, 293, index=9, geodata=(2, 2))
    pandapipes.create_valve(net, 8, 9, 0.4, True, index=2)

    with pytest.raises(UserWarning):
        pandapipes.create_valve(net, 8, 9, 0.4, True, index=2)
    with pytest.raises(UserWarning):
        pandapipes.create_valve(net, 8, 10, 0.4, True)
    with pytest.raises(ValueError):
        pandapipes.create_valve(net, 8, 9, 0.4, True, geodata=[(0, 1), (1, 1), (2, 2)])


def test_create_pump(create_empty_net):
    net = copy.deepcopy(create_empty_net)
    pandapipes.create_junction(net, 1, 293, index=8, geodata=(0, 1))
    pandapipes.create_junction(net, 1, 293, index=9, geodata=(2, 2))
    pandapipes.create_pump(net, 8, 9, "P1", index=2)

    with pytest.raises(UserWarning):
        pandapipes.create_pump(net, 8, 9, "P1", index=2)
    with pytest.raises(UserWarning):
        pandapipes.create_pump(net, 8, 10, "P1")
    with pytest.raises(UserWarning):
        pandapipes.create_pump(net, 8, 9, "blah")
    with pytest.raises(ValueError):
        pandapipes.create_pump(net, 8, 9, "P1", geodata=[(0, 1), (1, 1), (2, 2)])

    net2 = pandapipes.create_empty_network(fluid="hgas", add_stdtypes=False)
    pandapipes.create_junction(net2, 1, 293, index=8, geodata=(0, 1))
    pandapipes.create_junction(net2, 1, 293, index=9, geodata=(2, 2))
    with pytest.raises(UserWarning):
        pandapipes.create_pump(net2, 8, 9, "P1")


def test_create_pump_from_parameters(create_empty_net):
    net = copy.deepcopy(create_empty_net)
    pandapipes.create_junction(net, 1, 293, index=8, geodata=(0, 1))
    pandapipes.create_junction(net, 1, 293, index=9, geodata=(2, 2))
    pandapipes.create_pump_from_parameters(net, 8, 9, "pump1", pressure_list=[0, 1, 2, 3],
                                           flowrate_list=[0, 1, 2, 3], reg_polynomial_degree=1,
                                           index=2)

    with pytest.raises(UserWarning):
        pandapipes.create_pump_from_parameters(net, 8, 9, "pump1", pressure_list=[0, 1, 2, 3],
                                               flowrate_list=[0, 1, 2, 3], reg_polynomial_degree=1,
                                               index=2)
    with pytest.raises(UserWarning):
        pandapipes.create_pump_from_parameters(net, 8, 10, "pump1", pressure_list=[0, 1, 2, 3],
                                               flowrate_list=[0, 1, 2, 3], reg_polynomial_degree=1,
                                               index=2)
    with pytest.raises(ValueError):
        pandapipes.create_pump_from_parameters(net, 8, 9, "pump1", pressure_list=[0, 1, 2, 3],
                                               flowrate_list=[0, 1, 2, 3], reg_polynomial_degree=1,
                                               geodata=[(0, 1), (1, 1), (2, 2)])


def test_create_junctions(create_empty_net):
    net = copy.deepcopy(create_empty_net)
    # standard
    pandapipes.create_junctions(net, 3, 1, 293)
    # with geodata
    j2 = pandapipes.create_junctions(net, 3, 1, 293, geodata=(10, 20))
    # with geodata as array
    geodata = np.array([[10, 20], [20, 30], [30, 40]])
    j3 = pandapipes.create_junctions(net, 3, 1, 293, geodata=geodata)

    assert len(net.junction) == 9
    assert len(net.junction_geodata) == 6

    for i in j2:
        assert net.junction_geodata.at[i, 'x'] == 10
        assert net.junction_geodata.at[i, 'y'] == 20

    assert (net.junction_geodata.loc[j3, ['x', 'y']].values == geodata).all()
    assert (net.junction.pn_bar.values == 1).all()

    # no way of creating junctions with not matching shape
    with pytest.raises(ValueError):
        pandapipes.create_junctions(net, 2, 1, 293, geodata=geodata)


def test_create_pipes_from_parameters(create_empty_net):
    # standard
    net = copy.deepcopy(create_empty_net)
    j1 = pandapipes.create_junction(net, 3, 273)
    j2 = pandapipes.create_junction(net, 3, 273)
    pandapipes.create_pipes_from_parameters(net, [j1, j1], [j2, j2], 2, 0.2, sections=[1, 4])
    assert len(net.pipe) == 2
    assert len(net.pipe_geodata) == 0
    assert sum(net.pipe.sections) == 5
    assert len(set(net.pipe.length_km)) == 1

    # with geodata
    net = copy.deepcopy(create_empty_net)
    j1 = pandapipes.create_junction(net, 3, 273)
    j2 = pandapipes.create_junction(net, 3, 273)
    p = pandapipes.create_pipes_from_parameters(
        net, [j1, j1], [j2, j2], [1.5, 3], 0.5,
        geodata=[[(1, 1), (2, 2), (3, 3)], [(1, 1), (1, 2)]])

    assert len(net.pipe) == 2
    assert len(net.pipe_geodata) == 2
    assert net.pipe_geodata.at[p[0], "coords"] == [(1, 1), (2, 2), (3, 3)]
    assert net.pipe_geodata.at[p[1], "coords"] == [(1, 1), (1, 2)]

    # setting params as single value
    net = copy.deepcopy(create_empty_net)
    j1 = pandapipes.create_junction(net, 3, 273)
    j2 = pandapipes.create_junction(net, 3, 273)
    p = pandapipes.create_pipes_from_parameters(
        net, [j1, j1], [j2, j2], length_km=5, diameter_m=0.8, in_service=False,
        geodata=[(10, 10), (20, 20)], name="test", k_mm=0.01, loss_coefficient=0.3, sections=2,
        alpha_w_per_m2k=0.1, text_k=273, qext_w=0.01)

    assert len(net.pipe) == 2
    assert len(net.pipe_geodata) == 2
    assert net.pipe.length_km.at[p[0]] == 5
    assert net.pipe.length_km.at[p[1]] == 5
    assert net.pipe.at[p[0], "in_service"] == False  # is actually <class 'numpy.bool_'>
    assert net.pipe.at[p[1], "in_service"] == False  # is actually <class 'numpy.bool_'>
    assert net.pipe_geodata.at[p[0], "coords"] == [(10, 10), (20, 20)]
    assert net.pipe_geodata.at[p[1], "coords"] == [(10, 10), (20, 20)]
    assert net.pipe.at[p[0], "name"] == "test"
    assert net.pipe.at[p[1], "name"] == "test"
    assert net.pipe.at[p[0], "k_mm"] == 0.01
    assert net.pipe.at[p[1], "k_mm"] == 0.01
    assert net.pipe.at[p[0], "loss_coefficient"] == 0.3
    assert net.pipe.at[p[1], "loss_coefficient"] == 0.3
    assert net.pipe.at[p[0], "diameter_m"] == 0.8
    assert net.pipe.at[p[1], "diameter_m"] == 0.8
    assert net.pipe.at[p[0], "sections"] == 2
    assert net.pipe.at[p[1], "sections"] == 2
    assert net.pipe.at[p[0], "alpha_w_per_m2k"] == 0.1
    assert net.pipe.at[p[1], "alpha_w_per_m2k"] == 0.1
    assert net.pipe.at[p[0], "text_k"] == 273
    assert net.pipe.at[p[1], "text_k"] == 273
    assert net.pipe.at[p[0], "qext_w"] == 0.01
    assert net.pipe.at[p[1], "qext_w"] == 0.01

    # setting params as array
    net = copy.deepcopy(create_empty_net)
    j1 = pandapipes.create_junction(net, 3, 273)
    j2 = pandapipes.create_junction(net, 3, 273)
    p = pandapipes.create_pipes_from_parameters(
        net, [j1, j1], [j2, j2], length_km=[1, 5], diameter_m=[0.8, 0.7],
        in_service=[True, False],
        geodata=[[(10, 10), (20, 20)], [(100, 10), (200, 20)]], name=["p1", "p2"],
        k_mm=[0.01, 0.02], loss_coefficient=[0.3, 0.5], sections=[1, 2],
        alpha_w_per_m2k=[0.1, 0.2], text_k=[273, 274], qext_w=[0.01, 0.02])

    assert len(net.pipe) == 2
    assert len(net.pipe_geodata) == 2
    assert net.pipe.at[p[0], "length_km"] == 1
    assert net.pipe.at[p[1], "length_km"] == 5
    assert net.pipe.at[p[0], "in_service"] == True  # is actually <class 'numpy.bool_'>
    assert net.pipe.at[p[1], "in_service"] == False  # is actually <class 'numpy.bool_'>
    assert net.pipe_geodata.at[p[0], "coords"] == [(10, 10), (20, 20)]
    assert net.pipe_geodata.at[p[1], "coords"] == [(100, 10), (200, 20)]
    assert net.pipe.at[p[0], "name"] == "p1"
    assert net.pipe.at[p[1], "name"] == "p2"
    assert net.pipe.at[p[0], "diameter_m"] == 0.8
    assert net.pipe.at[p[1], "diameter_m"] == 0.7
    assert net.pipe.at[p[0], "k_mm"] == 0.01
    assert net.pipe.at[p[1], "k_mm"] == 0.02
    assert net.pipe.at[p[0], "loss_coefficient"] == 0.3
    assert net.pipe.at[p[1], "loss_coefficient"] == 0.5
    assert net.pipe.at[p[0], "alpha_w_per_m2k"] == 0.1
    assert net.pipe.at[p[1], "alpha_w_per_m2k"] == 0.2
    assert net.pipe.at[p[0], "sections"] == 1
    assert net.pipe.at[p[1], "sections"] == 2
    assert net.pipe.at[p[0], "text_k"] == 273
    assert net.pipe.at[p[1], "text_k"] == 274
    assert net.pipe.at[p[0], "qext_w"] == 0.01
    assert net.pipe.at[p[1], "qext_w"] == 0.02


def test_create_pipes_from_parameters_raise_except(create_empty_net):
    net = copy.deepcopy(create_empty_net)
    j1 = pandapipes.create_junction(net, 3, 273)
    j2 = pandapipes.create_junction(net, 3, 273)
    j3 = pandapipes.create_junction(net, 3, 273)

    with pytest.raises(UserWarning, match=r"trying to attach to non existing junctions"):
        pandapipes.create_pipes_from_parameters(
            net, [1, 3], [4, 5], length_km=5, diameter_m=0.8, in_service=False,
            geodata=[(10, 10), (20, 20)], name="test", k_mm=0.01, loss_coefficient=0.3, sections=2,
            alpha_w_per_m2k=0.1, text_k=273, qext_w=0.01)

    pandapipes.create_pipes_from_parameters(
        net, [j1, j1], [j2, j3], length_km=5, diameter_m=0.8, in_service=False,
        geodata=[(10, 10), (20, 20)], name="test", k_mm=0.01, loss_coefficient=0.3, sections=2,
        alpha_w_per_m2k=0.1, text_k=273, qext_w=0.01, index=[0, 1])
    with pytest.raises(UserWarning, match=r"with indexes \[0 1\] already exist"):
        pandapipes.create_pipes_from_parameters(
            net, [j1, j1], [j2, j3], length_km=5, diameter_m=0.8, in_service=False,
            geodata=[(10, 10), (20, 20)], name="test", k_mm=0.01, loss_coefficient=0.3, sections=2,
            alpha_w_per_m2k=0.1, text_k=273, qext_w=0.01, index=[0, 1])


def test_create_pipes(create_empty_net):
    # standard
    net = copy.deepcopy(create_empty_net)
    j1 = pandapipes.create_junction(net, 3, 273)
    j2 = pandapipes.create_junction(net, 3, 273)
    pandapipes.create_pipes(net, [j1, j1], [j2, j2], "80_GGG", 2, sections=[1, 4])
    assert len(net.pipe) == 2
    assert len(net.pipe_geodata) == 0
    assert sum(net.pipe.sections) == 5
    assert np.all(net.pipe.std_type == ["80_GGG"] * 2)
    assert len(set(net.pipe.length_km)) == 1

    # with geodata
    net = copy.deepcopy(create_empty_net)
    j1 = pandapipes.create_junction(net, 3, 273)
    j2 = pandapipes.create_junction(net, 3, 273)
    p = pandapipes.create_pipes(net, [j1, j1], [j2, j2], "80_GGG", [1.5, 3],
                                geodata=[[(1, 1), (2, 2), (3, 3)], [(1, 1), (1, 2)]])

    assert len(net.pipe) == 2
    assert len(net.pipe_geodata) == 2
    assert net.pipe_geodata.at[p[0], "coords"] == [(1, 1), (2, 2), (3, 3)]
    assert net.pipe_geodata.at[p[1], "coords"] == [(1, 1), (1, 2)]

    # setting params as single value
    net = copy.deepcopy(create_empty_net)
    j1 = pandapipes.create_junction(net, 3, 273)
    j2 = pandapipes.create_junction(net, 3, 273)
    p = pandapipes.create_pipes(
        net, [j1, j1], [j2, j2], std_type="80_GGG", length_km=5, in_service=False,
        geodata=[(10, 10), (20, 20)], name="test", k_mm=0.01, loss_coefficient=0.3, sections=2,
        alpha_w_per_m2k=0.1, text_k=273, qext_w=0.01)

    assert len(net.pipe) == 2
    assert len(net.pipe_geodata) == 2
    assert net.pipe.length_km.at[p[0]] == 5
    assert net.pipe.length_km.at[p[1]] == 5
    assert net.pipe.at[p[0], "in_service"] == False  # is actually <class 'numpy.bool_'>
    assert net.pipe.at[p[1], "in_service"] == False  # is actually <class 'numpy.bool_'>
    assert net.pipe_geodata.at[p[0], "coords"] == [(10, 10), (20, 20)]
    assert net.pipe_geodata.at[p[1], "coords"] == [(10, 10), (20, 20)]
    assert net.pipe.at[p[0], "name"] == "test"
    assert net.pipe.at[p[1], "name"] == "test"
    assert net.pipe.at[p[0], "std_type"] == "80_GGG"
    assert net.pipe.at[p[1], "std_type"] == "80_GGG"
    assert net.pipe.at[p[0], "k_mm"] == 0.01
    assert net.pipe.at[p[1], "k_mm"] == 0.01
    assert net.pipe.at[p[0], "loss_coefficient"] == 0.3
    assert net.pipe.at[p[1], "loss_coefficient"] == 0.3
    assert net.pipe.at[p[0], "diameter_m"] == 0.086
    assert net.pipe.at[p[1], "diameter_m"] == 0.086
    assert net.pipe.at[p[0], "sections"] == 2
    assert net.pipe.at[p[1], "sections"] == 2
    assert net.pipe.at[p[0], "alpha_w_per_m2k"] == 0.1
    assert net.pipe.at[p[1], "alpha_w_per_m2k"] == 0.1
    assert net.pipe.at[p[0], "text_k"] == 273
    assert net.pipe.at[p[1], "text_k"] == 273
    assert net.pipe.at[p[0], "qext_w"] == 0.01
    assert net.pipe.at[p[1], "qext_w"] == 0.01

    # setting params as array
    net = copy.deepcopy(create_empty_net)
    j1 = pandapipes.create_junction(net, 3, 273)
    j2 = pandapipes.create_junction(net, 3, 273)
    p = pandapipes.create_pipes(
        net, [j1, j1], [j2, j2], std_type="80_GGG", length_km=[1, 5], in_service=[True, False],
        geodata=[[(10, 10), (20, 20)], [(100, 10), (200, 20)]], name=["p1", "p2"],
        k_mm=[0.01, 0.02], loss_coefficient=[0.3, 0.5], sections=[1, 2],
        alpha_w_per_m2k=[0.1, 0.2], text_k=[273, 274], qext_w=[0.01, 0.02])

    assert len(net.pipe) == 2
    assert len(net.pipe_geodata) == 2
    assert net.pipe.at[p[0], "length_km"] == 1
    assert net.pipe.at[p[1], "length_km"] == 5
    assert net.pipe.at[p[0], "in_service"] == True  # is actually <class 'numpy.bool_'>
    assert net.pipe.at[p[1], "in_service"] == False  # is actually <class 'numpy.bool_'>
    assert net.pipe_geodata.at[p[0], "coords"] == [(10, 10), (20, 20)]
    assert net.pipe_geodata.at[p[1], "coords"] == [(100, 10), (200, 20)]
    assert net.pipe.at[p[0], "name"] == "p1"
    assert net.pipe.at[p[1], "name"] == "p2"
    assert net.pipe.at[p[0], "std_type"] == "80_GGG"
    assert net.pipe.at[p[1], "std_type"] == "80_GGG"
    assert net.pipe.at[p[0], "diameter_m"] == 0.086
    assert net.pipe.at[p[1], "diameter_m"] == 0.086
    assert net.pipe.at[p[0], "k_mm"] == 0.01
    assert net.pipe.at[p[1], "k_mm"] == 0.02
    assert net.pipe.at[p[0], "loss_coefficient"] == 0.3
    assert net.pipe.at[p[1], "loss_coefficient"] == 0.5
    assert net.pipe.at[p[0], "alpha_w_per_m2k"] == 0.1
    assert net.pipe.at[p[1], "alpha_w_per_m2k"] == 0.2
    assert net.pipe.at[p[0], "sections"] == 1
    assert net.pipe.at[p[1], "sections"] == 2
    assert net.pipe.at[p[0], "text_k"] == 273
    assert net.pipe.at[p[1], "text_k"] == 274
    assert net.pipe.at[p[0], "qext_w"] == 0.01
    assert net.pipe.at[p[1], "qext_w"] == 0.02


def test_create_pipes_raise_except(create_empty_net):
    net = copy.deepcopy(create_empty_net)
    j1 = pandapipes.create_junction(net, 3, 273)
    j2 = pandapipes.create_junction(net, 3, 273)
    j3 = pandapipes.create_junction(net, 3, 273)

    with pytest.raises(UserWarning, match=r"trying to attach to non existing junctions"):
        pandapipes.create_pipes(
            net, [1, 3], [4, 5], std_type="80_GGG", length_km=5, in_service=False,
            geodata=[(10, 10), (20, 20)], name="test", k_mm=0.01, loss_coefficient=0.3, sections=2,
            alpha_w_per_m2k=0.1, text_k=273, qext_w=0.01)

    pandapipes.create_pipes(
        net, [j1, j1], [j2, j3], std_type="80_GGG", length_km=5, in_service=False,
        geodata=[(10, 10), (20, 20)], name="test", k_mm=0.01, loss_coefficient=0.3, sections=2,
        alpha_w_per_m2k=0.1, text_k=273, qext_w=0.01, index=[0, 1])
    with pytest.raises(UserWarning, match=r"with indexes \[0 1\] already exist"):
        pandapipes.create_pipes(
            net, [j1, j1], [j2, j3], std_type="80_GGG", length_km=5, in_service=False,
            geodata=[(10, 10), (20, 20)], name="test", k_mm=0.01, loss_coefficient=0.3, sections=2,
            alpha_w_per_m2k=0.1, text_k=273, qext_w=0.01, index=[0, 1])


def test_create_valves(create_empty_net):
    # standard
    net = copy.deepcopy(create_empty_net)
    j1 = pandapipes.create_junction(net, 3, 273)
    j2 = pandapipes.create_junction(net, 3, 273)
    pandapipes.create_valves(net, [j1, j1], [j2, j2], 0.2)
    assert len(net.valve) == 2
    assert len(set(net.valve.diameter_m)) == 1
    assert np.all(net.valve.diameter_m == [0.2, 0.2])

    # setting params as single value
    net = copy.deepcopy(create_empty_net)
    j1 = pandapipes.create_junction(net, 3, 273)
    j2 = pandapipes.create_junction(net, 3, 273)
    v = pandapipes.create_valves(
        net, [j1, j1], [j2, j2], diameter_m=0.8, opened=False, name="test", new_col=0.01,
        loss_coefficient=0.3, type="v")

    assert len(net.valve) == 2
    assert net.valve.at[v[0], "opened"] == False  # is actually <class 'numpy.bool_'>
    assert net.valve.at[v[1], "opened"] == False  # is actually <class 'numpy.bool_'>
    assert net.valve.at[v[0], "name"] == "test"
    assert net.valve.at[v[1], "name"] == "test"
    assert net.valve.at[v[0], "type"] == "v"
    assert net.valve.at[v[1], "type"] == "v"
    assert net.valve.at[v[0], "new_col"] == 0.01
    assert net.valve.at[v[1], "new_col"] == 0.01
    assert net.valve.at[v[0], "loss_coefficient"] == 0.3
    assert net.valve.at[v[1], "loss_coefficient"] == 0.3

    # setting params as array
    net = copy.deepcopy(create_empty_net)
    j1 = pandapipes.create_junction(net, 3, 273)
    j2 = pandapipes.create_junction(net, 3, 273)
    v = pandapipes.create_valves(
        net, [j1, j1], [j2, j2], diameter_m=[0.8, 0.7], opened=[True, False], name=["v1", "v2"],
        type=["va1", "va2"], loss_coefficient=[0.3, 0.5], new_col=[0.01, 1.9])

    assert len(net.valve) == 2
    assert net.valve.at[v[0], "opened"] == True  # is actually <class 'numpy.bool_'>
    assert net.valve.at[v[1], "opened"] == False  # is actually <class 'numpy.bool_'>
    assert net.valve.at[v[0], "name"] == "v1"
    assert net.valve.at[v[1], "name"] == "v2"
    assert net.valve.at[v[0], "type"] == "va1"
    assert net.valve.at[v[1], "type"] == "va2"
    assert net.valve.at[v[0], "diameter_m"] == 0.8
    assert net.valve.at[v[1], "diameter_m"] == 0.7
    assert net.valve.at[v[0], "new_col"] == 0.01
    assert net.valve.at[v[1], "new_col"] == 1.9
    assert net.valve.at[v[0], "loss_coefficient"] == 0.3
    assert net.valve.at[v[1], "loss_coefficient"] == 0.5

    # setting index explicitly
    net = copy.deepcopy(create_empty_net)
    j1 = pandapipes.create_junction(net, 3, 273)
    j2 = pandapipes.create_junction(net, 3, 273)
    v = pandapipes.create_valves(
        net, [j1, j1], [j2, j2], diameter_m=[0.8, 0.7], opened=[True, False], name=["v1", "v2"],
        type=["va1", "va2"], loss_coefficient=[0.3, 0.5], new_col=[0.01, 1.9], index=[1, 5])

    assert len(net.valve) == 2
    assert np.all(v == [1, 5])
    assert np.all(net.valve.index == [1, 5])


def test_create_valves_raise_except(create_empty_net):
    net = copy.deepcopy(create_empty_net)
    j1 = pandapipes.create_junction(net, 3, 273)
    j2 = pandapipes.create_junction(net, 3, 273)
    j3 = pandapipes.create_junction(net, 3, 273)

    with pytest.raises(UserWarning, match=r"trying to attach to non existing junctions"):
        pandapipes.create_valves(net, [1, 3], [4, 5], diameter_m=0.8, opened=False, name="test",
                                 loss_coefficient=0.3)

    pandapipes.create_valves(net, [j1, j1], [j2, j3], diameter_m=0.8, opened=False, name="test",
                             loss_coefficient=0.3, index=[0, 1])
    with pytest.raises(UserWarning, match=r"with indexes \[0 1\] already exist"):
        pandapipes.create_valves(net, [j1, j1], [j2, j3], diameter_m=0.8, opened=False,
                                 name="test", loss_coefficient=0.3, index=[0, 1])


def test_create_sinks(create_empty_net):
    net = copy.deepcopy(create_empty_net)
    # standard
    j1 = pandapipes.create_junction(net, 3, 273)
    j2 = pandapipes.create_junction(net, 3, 273)
    j3 = pandapipes.create_junction(net, 3, 273)
    pandapipes.create_sinks(
        net, junctions=[j1, j2, j3], mdot_kg_per_s=[0, 0.1, 0.2], scaling=[1., 1., 0.5],
        name=["sink%d" % s for s in range(3)], new_col=[1, 3, 5])

    assert (net.sink.junction.at[0] == j1)
    assert (net.sink.junction.at[1] == j2)
    assert (net.sink.junction.at[2] == j3)
    assert (net.sink.mdot_kg_per_s.at[0] == 0)
    assert (net.sink.mdot_kg_per_s.at[1] == 0.1)
    assert (net.sink.mdot_kg_per_s.at[2] == 0.2)
    assert (net.sink.scaling.at[0] == 1)
    assert (net.sink.scaling.at[1] == 1)
    assert (net.sink.scaling.at[2] == 0.5)
    assert (all(net.sink.in_service.values == True))
    assert (all(net.sink.type.values == "sink"))
    assert (all(net.sink.new_col.values == [1, 3, 5]))


def test_create_sinks_raise_except(create_empty_net):
    net = copy.deepcopy(create_empty_net)
    # standard
    j1 = pandapipes.create_junction(net, 3, 273)
    j2 = pandapipes.create_junction(net, 3, 273)
    j3 = pandapipes.create_junction(net, 3, 273)

    with pytest.raises(UserWarning, match=r"Cannot attach to junctions \{3, 4, 5\}, they do not "
                                          r"exist"):
        pandapipes.create_sinks(net, junctions=[3, 4, 5], mdot_kg_per_s=[0, 0.1, 0.2],
                                scaling=[1., 1., 0.5], name=["sink%d" % s for s in range(3)],
                                new_col=[1, 3, 5])

    sg = pandapipes.create_sinks(net, junctions=[j1, j2, j3], mdot_kg_per_s=[0, 0.1, 0.2],
                                 scaling=[1., 1., 0.5], name=["sink%d" % s for s in range(3)],
                                 new_col=[1, 3, 5])
    with pytest.raises(UserWarning, match=r"Sinks with indexes \[0 1 2\] already exist."):
        pandapipes.create_sinks(net, junctions=[j1, j2, j3], mdot_kg_per_s=[0, 0.1, 0.2],
                                scaling=[1., 1., 0.5], name=["sink%d" % s for s in range(3)],
                                new_col=[1, 3, 5], index=sg)


def test_create_sources(create_empty_net):
    net = copy.deepcopy(create_empty_net)
    # standard
    j1 = pandapipes.create_junction(net, 3, 273)
    j2 = pandapipes.create_junction(net, 3, 273)
    j3 = pandapipes.create_junction(net, 3, 273)
    pandapipes.create_sources(
        net, junctions=[j1, j2, j3], mdot_kg_per_s=[0, 0.1, 0.2], scaling=[1., 1., 0.5],
        name=["source%d" % s for s in range(3)], new_col=[1, 3, 5])

    assert (net.source.junction.at[0] == j1)
    assert (net.source.junction.at[1] == j2)
    assert (net.source.junction.at[2] == j3)
    assert (net.source.mdot_kg_per_s.at[0] == 0)
    assert (net.source.mdot_kg_per_s.at[1] == 0.1)
    assert (net.source.mdot_kg_per_s.at[2] == 0.2)
    assert (net.source.scaling.at[0] == 1)
    assert (net.source.scaling.at[1] == 1)
    assert (net.source.scaling.at[2] == 0.5)
    assert (all(net.source.in_service.values == True))
    assert (all(net.source.type.values == "source"))
    assert (all(net.source.new_col.values == [1, 3, 5]))


def test_create_sources_raise_except(create_empty_net):
    net = copy.deepcopy(create_empty_net)
    # standard
    j1 = pandapipes.create_junction(net, 3, 273)
    j2 = pandapipes.create_junction(net, 3, 273)
    j3 = pandapipes.create_junction(net, 3, 273)

    with pytest.raises(UserWarning, match=r"Cannot attach to junctions \{3, 4, 5\}, they do not "
                                          r"exist"):
        pandapipes.create_sources(net, junctions=[3, 4, 5], mdot_kg_per_s=[0, 0.1, 0.2],
                                  scaling=[1., 1., 0.5], name=["source%d" % s for s in range(3)],
                                  new_col=[1, 3, 5])

    sg = pandapipes.create_sources(net, junctions=[j1, j2, j3], mdot_kg_per_s=[0, 0.1, 0.2],
                                   scaling=[1., 1., 0.5], name=["source%d" % s for s in range(3)],
                                   new_col=[1, 3, 5])
    with pytest.raises(UserWarning, match=r"Sources with indexes \[0 1 2\] already exist."):
        pandapipes.create_sources(net, junctions=[j1, j2, j3], mdot_kg_per_s=[0, 0.1, 0.2],
                                  scaling=[1., 1., 0.5], name=["source%d" % s for s in range(3)],
                                  new_col=[1, 3, 5], index=sg)


if __name__ == '__main__':
    pytest.main(["test_create.py"])
