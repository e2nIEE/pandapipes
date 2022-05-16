# Copyright (c) 2020-2022 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import copy

import numpy as np
import pytest

import pandapipes as pp
from pandapipes import networks as nets_pps
from pandapipes.create import create_empty_network, create_junction, create_ext_grid, create_sink, create_source, \
    create_pipe_from_parameters, create_valve
from pandapipes.test.pipeflow_internals.test_inservice import create_test_net
from pandapipes.pipeflow_setup import get_lookup
from pandapipes.properties.fluids import FluidPropertyConstant, Fluid, _add_fluid_to_net


def test_one_node_net():
    """

    :return:
    :rtype:
    """

    net = create_empty_network()
    j = create_junction(net, 1, 298.15)
    create_ext_grid(net, j, 1, 298.15, fluid='water')
    create_sink(net, j, 0.01)
    create_source(net, j, 0.02, fluid='water')
    pp.pipeflow(net)

    assert np.isclose(net.res_ext_grid.values + net.res_sink.values - net.res_source.values, 0)

    net = create_empty_network()
    j = create_junction(net, 1, 298.15)
    create_ext_grid(net, j, 1, 298.15, fluid='lgas')
    create_sink(net, j, 0.01)
    create_source(net, j, 0.02, fluid='lgas')
    pp.pipeflow(net)

    assert np.isclose(net.res_ext_grid.values + net.res_sink.values - net.res_source.values, 0)


def simple_fluid(net):
    fluid_name = 'fluid1'
    dens = FluidPropertyConstant(0.1)
    visc = FluidPropertyConstant(0.01)
    heat = FluidPropertyConstant(10)
    mass = FluidPropertyConstant(1)
    higc = FluidPropertyConstant(2)
    lowc = FluidPropertyConstant(1)
    derc = FluidPropertyConstant(0)
    comp = FluidPropertyConstant(0.001)
    fluid1 = Fluid(fluid_name, 'gas', density=dens, viscosity=visc, heat_capacity=heat, molar_mass=mass,
                   der_compressibility=derc, compressibility=comp, hhv=higc, lhv=lowc)
    _add_fluid_to_net(net, fluid1)

    fluid_name = 'fluid2'
    dens = FluidPropertyConstant(0.2)
    visc = FluidPropertyConstant(0.02)
    heat = FluidPropertyConstant(20)
    mass = FluidPropertyConstant(2)
    higc = FluidPropertyConstant(4)
    lowc = FluidPropertyConstant(2)
    derc = FluidPropertyConstant(0)
    comp = FluidPropertyConstant(0.002)
    fluid2 = Fluid(fluid_name, 'gas', density=dens, viscosity=visc, heat_capacity=heat, molar_mass=mass,
                   der_compressibility=derc, compressibility=comp, hhv=higc, lhv=lowc)
    _add_fluid_to_net(net, fluid2)

    fluid_name = 'fluid3'
    dens = FluidPropertyConstant(0.3)
    visc = FluidPropertyConstant(0.03)
    heat = FluidPropertyConstant(30)
    mass = FluidPropertyConstant(3)
    higc = FluidPropertyConstant(6)
    lowc = FluidPropertyConstant(3)
    derc = FluidPropertyConstant(0)
    comp = FluidPropertyConstant(0.003)
    fluid3 = Fluid(fluid_name, 'gas', density=dens, viscosity=visc, heat_capacity=heat, molar_mass=mass,
                   der_compressibility=derc, compressibility=comp, hhv=higc, lhv=lowc)
    _add_fluid_to_net(net, fluid3)


def same_fluid_twice_defined(net):
    fluid_name = 'fluid1'
    dens = FluidPropertyConstant(0.1)
    visc = FluidPropertyConstant(0.01)
    heat = FluidPropertyConstant(10)
    mass = FluidPropertyConstant(1)
    higc = FluidPropertyConstant(2)
    lowc = FluidPropertyConstant(1)
    derc = FluidPropertyConstant(0)
    comp = FluidPropertyConstant(0.001)
    fluid1 = Fluid(fluid_name, 'gas', density=dens, viscosity=visc, heat_capacity=heat, molar_mass=mass,
                   der_compressibility=derc, compressibility=comp, hhv=higc, lhv=lowc)
    _add_fluid_to_net(net, fluid1)

    fluid_name = 'fluid2'
    dens = FluidPropertyConstant(0.1)
    visc = FluidPropertyConstant(0.01)
    heat = FluidPropertyConstant(10)
    mass = FluidPropertyConstant(1)
    higc = FluidPropertyConstant(2)
    lowc = FluidPropertyConstant(1)
    derc = FluidPropertyConstant(0)
    comp = FluidPropertyConstant(0.001)
    fluid2 = Fluid(fluid_name, 'gas', density=dens, viscosity=visc, heat_capacity=heat, molar_mass=mass,
                   der_compressibility=derc, compressibility=comp, hhv=higc, lhv=lowc)
    _add_fluid_to_net(net, fluid2)


def test_two_fluids_grid_simple_gases():
    """

    :return:
    :rtype:
    """
    import logging
    logger = logging.getLogger()
    logger.setLevel("DEBUG")
    logger.debug('external grid')

    net = create_empty_network()
    j1 = create_junction(net, 1, 298.15, index=50)
    j2 = create_junction(net, 1, 298.15, index=52)
    simple_fluid(net)

    create_ext_grid(net, j1, 1, 298.15, fluid='fluid2', index=102)
    create_sink(net, j2, 0.5)
    create_source(net, j1, 0.3, fluid='fluid1')
    create_pipe_from_parameters(net, j1, j2, 1, 10, np.pi)
    pp.pipeflow(net, tol_p=1e-4, tol_v=1e-4, tol_m=1e-10, tol_w=1e-6, iter=400)
    w = get_lookup(net, 'node', 'w')

    print(net._pit['node'][:, w])
    print(net._internal_results)
    assert np.isclose(net.res_ext_grid.values.sum() + net.res_sink.values.sum() - net.res_source.values.sum(), 0)


def test_three_fluids_grid_simple_gases():
    """

    :return:
    :rtype:
    """
    net = create_empty_network()
    j1 = create_junction(net, 1, 298.15, index=50)
    j2 = create_junction(net, 1, 298.15, index=52)
    simple_fluid(net)

    create_ext_grid(net, j1, 1, 298.15, fluid='fluid1', index=102)
    create_sink(net, j2, 0.5)
    create_source(net, j1, 0.1, fluid='fluid2')
    create_source(net, j1, 0.2, fluid='fluid3')
    create_pipe_from_parameters(net, j1, j2, 1, 10, np.pi)
    pp.pipeflow(net, tol_p=1e-4, tol_v=1e-4, tol_m=1e-10, tol_w=1e-6, iter=400)
    w = get_lookup(net, 'node', 'w')

    print(net._pit['node'][:, w])
    print(net._internal_results)
    assert np.isclose(net.res_ext_grid.values.sum() + net.res_sink.values.sum() - net.res_source.values.sum(), 0)


def test_two_fluids_grid_simple():
    """

    :return:
    :rtype:
    """
    net = create_empty_network()
    j1 = create_junction(net, 1, 298.15, index=50)
    j2 = create_junction(net, 1, 298.15, index=52)
    create_ext_grid(net, j1, 1, 298.15, fluid='lgas', index=102)
    create_sink(net, j2, 0.08)
    create_source(net, j1, 0.03, fluid='hydrogen')
    create_pipe_from_parameters(net, j1, j2, 0.5, 0.1, 0.01)
    pp.pipeflow(net, tol_p=1e-4, tol_v=1e-4, tol_m=1e-10, tol_w=1e-6, iter=400)
    w = get_lookup(net, 'node', 'w')
    print(net._pit['node'][:, w])
    print(net._internal_results)
    assert np.isclose(net.res_ext_grid.values.sum() + net.res_sink.values.sum() - net.res_source.values.sum(), 0)


def test_three_fluids_grid_simple():
    """

    :return:
    :rtype:
    """
    net = create_empty_network()
    j1 = create_junction(net, 1, 298.15, index=50)
    j2 = create_junction(net, 1, 298.15, index=52)
    create_ext_grid(net, j1, 1, 298.15, fluid='lgas', index=102)
    create_sink(net, j2, 0.08)
    create_source(net, j1, 0.03, fluid='hydrogen')
    create_source(net, j1, 0.01, fluid='hgas')
    create_pipe_from_parameters(net, j1, j2, 1, np.pi, 0.01)
    pp.pipeflow(net, tol_p=1e-4, tol_v=1e-4, tol_m=1e-10, tol_w=1e-6, iter=400)
    w = get_lookup(net, 'node', 'w')
    print(net._pit['node'][:, w])
    print(net._internal_results)
    assert np.isclose(net.res_ext_grid.values.sum() + net.res_sink.values.sum() - net.res_source.values.sum(), 0)


def test_four_fluids_grid_simple():
    """

    :return:
    :rtype:
    """
    net = create_empty_network()
    j1 = create_junction(net, 1, 298.15, index=50)
    j2 = create_junction(net, 1, 298.15, index=52)
    create_ext_grid(net, j1, 1, 298.15, fluid='lgas', index=102)
    create_sink(net, j2, 0.08)
    create_source(net, j1, 0.03, fluid='hydrogen')
    create_source(net, j1, 0.01, fluid='hgas')
    create_source(net, j1, 0.01, fluid='butane')
    create_pipe_from_parameters(net, j1, j2, 1, np.pi, 0.01)
    pp.pipeflow(net, tol_p=1e-4, tol_v=1e-4, tol_m=1e-10, tol_w=1e-6, iter=400)
    w = get_lookup(net, 'node', 'w')
    print(net._pit['node'][:, w])
    print(net._internal_results)
    assert np.isclose(net.res_ext_grid.values.sum() + net.res_sink.values.sum() - net.res_source.values.sum(), 0)


def test_two_fluids_two_pipes_grid_simple():
    """

    :return:
    :rtype:
    """

    net = create_empty_network()
    j1 = create_junction(net, 1, 298.15, index=50)
    j2 = create_junction(net, 1, 298.15, index=52)
    j3 = create_junction(net, 1, 298.15, index=54)
    create_ext_grid(net, j1, 1, 298.15, fluid='lgas', index=102)
    create_sink(net, j3, 0.08)
    create_source(net, j1, 0.03, fluid='hydrogen')
    create_pipe_from_parameters(net, j2, j1, 0.01, np.pi, 0.01)
    create_pipe_from_parameters(net, j3, j2, 0.01, np.pi, 0.01)
    pp.pipeflow(net, tol_p=1e-4, tol_v=1e-4, tol_m=1e-10, tol_w=1e-6, iter=400)
    w = get_lookup(net, 'node', 'w')
    print(net._pit['node'][:, w])
    print(net._internal_results)
    assert np.isclose(net.res_ext_grid.values.sum() + net.res_sink.values.sum() - net.res_source.values.sum(), 0)


def test_multiple_fluids_grid_line_ascending():
    """

    :return:
    :rtype:
    """
    net = create_empty_network()
    j1 = create_junction(net, 1, 298.15, index=50)
    j2 = create_junction(net, 1, 298.15, index=52)
    j3 = create_junction(net, 1, 298.15, index=54)
    create_ext_grid(net, j1, 1, 298.15, fluid='lgas', index=102)
    create_sink(net, j3, 0.08)
    create_source(net, j1, 0.03, fluid='hydrogen')
    create_source(net, j1, 0.01, fluid='butane')
    # create_source(net, j1, 0.01, fluid='hgas')
    create_pipe_from_parameters(net, j1, j2, 0.01, np.pi, 0.01)
    create_pipe_from_parameters(net, j2, j3, 0.01, np.pi, 0.01)
    pp.pipeflow(net, tol_p=1e-4, tol_v=1e-4, tol_m=1e-10, tol_w=1e-6, iter=400)
    w = get_lookup(net, 'node', 'w')
    print(net._pit['node'][:, w])
    print(net._internal_results)
    assert np.isclose(net.res_ext_grid.values.sum() + net.res_sink.values.sum() - net.res_source.values.sum(), 0)


def test_multiple_fluids_grid():
    """

    :return:
    :rtype:
    """

    net = create_empty_network()
    j1 = create_junction(net, 1, 298.15, index=50)
    j2 = create_junction(net, 1, 298.15, index=52)
    j3 = create_junction(net, 1, 298.15, index=54)
    create_ext_grid(net, j1, 1, 298.15, fluid='lgas', index=102)
    create_sink(net, j3, 0.08)
    create_source(net, j2, 0.03, fluid='hydrogen')
    create_source(net, j1, 0.01, fluid='butane')
    create_source(net, j1, 0.01, fluid='hgas')
    create_pipe_from_parameters(net, j2, j1, 0.01, np.pi, 0.01)
    create_pipe_from_parameters(net, j3, j2, 0.01, np.pi, 0.01)
    pp.pipeflow(net, tol_p=1e-4, tol_v=1e-4, tol_m=1e-10, tol_w=1e-6, iter=400)
    w = get_lookup(net, 'node', 'w')
    print(net._pit['node'][:, w])
    print(net._internal_results)
    assert np.isclose(net.res_ext_grid.values.sum() + net.res_sink.values.sum() - net.res_source.values.sum(), 0)


def test_multiple_fluids_grid_mesehd_valve():
    """

    :return:
    :rtype:
    """

    net = create_empty_network()
    j1 = create_junction(net, 1, 298.15, index=50)
    j2 = create_junction(net, 1, 298.15, index=52)
    j3 = create_junction(net, 1, 298.15, index=54)
    j4 = create_junction(net, 1, 298.15, index=55)
    create_ext_grid(net, j1, 1, 298.15, fluid='lgas', index=102)
    create_sink(net, j4, 0.08)
    create_source(net, j1, 0.03, fluid='hydrogen')
    create_pipe_from_parameters(net, j1, j2, 0.01, np.pi, 0.01)
    create_pipe_from_parameters(net, j2, j3, 0.01, np.pi, 0.01)
    create_pipe_from_parameters(net, j3, j4, 0.01, np.pi, 0.01)
    create_pipe_from_parameters(net, j4, j1, 0.01, np.pi, 0.01)
    pp.pipeflow(net, tol_p=1e-4, tol_v=1e-4, tol_m=1e-10, tol_w=1e-6, iter=400)
    w = get_lookup(net, 'node', 'w')
    print(net._pit['node'][:, w])
    print(net._internal_results)
    assert np.isclose(net.res_ext_grid.values.sum() + net.res_sink.values.sum() - net.res_source.values.sum(), 0)


def test_multiple_fluids_grid_source():
    """

    :return:
    :rtype:
    """
    net = create_empty_network()
    j1 = create_junction(net, 1, 298.15, index=50)
    j2 = create_junction(net, 1, 298.15, index=52)
    j3 = create_junction(net, 1, 298.15, index=54)
    create_ext_grid(net, j1, 1, 298.15, fluid='lgas', index=102)
    create_sink(net, j3, 0.08)
    create_source(net, j2, 0.03, fluid='hydrogen')
    create_pipe_from_parameters(net, j1, j2, 1, np.pi, 0.01)
    create_pipe_from_parameters(net, j2, j3, 1, np.pi, 0.01)
    pp.pipeflow(net, tol_p=1e-4, tol_v=1e-4, tol_m=1e-10, tol_w=1e-6, iter=400)
    w = get_lookup(net, 'node', 'w')
    print(net._pit['node'][:, w])
    print(net._internal_results)
    assert np.isclose(net.res_ext_grid.values.sum() + net.res_sink.values.sum() - net.res_source.values.sum(), 0)


def test_multiple_fluids_grid_feed_back():
    """

    :return:
    :rtype:
    """
    net = create_empty_network()
    j1 = create_junction(net, 1, 298.15, index=50)
    j2 = create_junction(net, 1, 298.15, index=52)
    create_ext_grid(net, j1, 1, 298.15, fluid='lgas', index=102)
    create_sink(net, j2, 0.08)
    create_source(net, j1, 0.1, fluid='hydrogen')
    create_pipe_from_parameters(net, j1, j2, 1, np.pi, 0.01)
    pp.pipeflow(net, tol_p=1e-4, tol_v=1e-4, tol_m=1e-10, tol_w=1e-6, iter=400)
    w = get_lookup(net, 'node', 'w')
    print(net._pit['node'][:, w])
    print(net._internal_results)
    assert np.isclose(net.res_ext_grid.values.sum() + net.res_sink.values.sum() - net.res_source.values.sum(), 0)


def test_multiple_fluids_feeder():
    """

    :return:
    :rtype:
    """

    net = create_empty_network()
    j1 = create_junction(net, 1, 298.15, index=50)
    j2 = create_junction(net, 1, 298.15, index=52)
    j3 = create_junction(net, 1, 298.15, index=54)
    j4 = create_junction(net, 1, 298.15, index=56)
    j5 = create_junction(net, 1, 298.15, index=58)
    j6 = create_junction(net, 1, 298.15, index=60)
    j7 = create_junction(net, 1, 298.15, index=62)
    j8 = create_junction(net, 1, 298.15, index=64)
    j9 = create_junction(net, 1, 298.15, index=66)

    create_ext_grid(net, j1, 1, 298.15, fluid='lgas', index=102)
    create_source(net, j1, 0.1, fluid='hydrogen')
    create_sink(net, j3, 0.08)
    create_sink(net, j4, 0.08)
    create_sink(net, j5, 0.08)
    create_sink(net, j6, 0.08)
    create_sink(net, j8, 0.08)
    create_sink(net, j9, 0.08)

    create_pipe_from_parameters(net, j1, j2, 1, np.pi, 0.01)
    create_pipe_from_parameters(net, j2, j3, 1, np.pi, 0.01)
    create_pipe_from_parameters(net, j2, j4, 1, np.pi, 0.01)
    create_pipe_from_parameters(net, j4, j5, 1, np.pi, 0.01)
    create_pipe_from_parameters(net, j1, j6, 1, np.pi, 0.01)
    create_valve(net, j7, j1, np.pi)
    create_valve(net, j9, j1, np.pi)
    create_pipe_from_parameters(net, j8, j7, 1, np.pi, 0.01)

    pp.pipeflow(net, tol_p=1e-4, tol_v=1e-4, tol_m=1e-10, tol_w=1e-6, iter=400)
    w = get_lookup(net, 'node', 'w')
    print(net._pit['node'][:, w])
    print(net._internal_results)
    assert np.isclose(net.res_ext_grid.values.sum() + net.res_sink.values.sum() - net.res_source.values.sum(), 0)


def test_multiple_fluids_grid_valve():
    """

    :return:
    :rtype:
    """

    net = create_empty_network()
    j1 = create_junction(net, 1, 298.15, index=50)
    j2 = create_junction(net, 1, 298.15, index=52)
    j3 = create_junction(net, 1, 298.15, index=54)
    j4 = create_junction(net, 1, 298.15, index=55)
    create_ext_grid(net, j1, 1, 298.15, fluid='lgas', index=102)
    create_sink(net, j4, 0.08)
    create_source(net, j1, 0.03, fluid='hydrogen')
    create_pipe_from_parameters(net, j1, j2, 0.01, np.pi, 0.01)
    create_valve(net, j3, j2, 0.01)
    create_pipe_from_parameters(net, j3, j4, 0.01, np.pi, 0.01)
    create_pipe_from_parameters(net, j4, j1, 0.01, np.pi, 0.01)
    pp.pipeflow(net, tol_p=1e-4, tol_v=1e-4, tol_m=1e-10, tol_w=1e-6, iter=400)
    w = get_lookup(net, 'node', 'w')
    print(net._pit['node'][:, w])
    print(net._internal_results)
    assert np.isclose(net.res_ext_grid.values.sum() + net.res_sink.values.sum() - net.res_source.values.sum(), 0)


def test_two_node_net_with_two_different_fluids():
    """

    :return:
    :rtype:
    """

    net = create_empty_network()
    j1 = create_junction(net, 1, 298.15, index=51)
    j2 = create_junction(net, 1, 298.15, index=52)
    create_ext_grid(net, j1, 1, 298.15, fluid='hgas', index=100)
    # create_ext_grid(net, j1, 1, 298.15, fluid='lgas', index=101)
    create_source(net, j1, 0.03, fluid='hydrogen', index=100)
    create_source(net, j1, 0.01, fluid='lgas', index=101)
    create_source(net, j1, 0.01, fluid='methane', index=102)
    create_source(net, j1, 0.01, fluid='propane', index=103)
    create_source(net, j1, 0.01, fluid='butane', index=104)
    create_source(net, j1, 0.01, fluid='hgas', index=105)
    create_sink(net, j2, 0.01, index=100)
    create_pipe_from_parameters(net, j1, j2, 0.01, np.pi, 0.01)
    pp.pipeflow(net, tol_p=1e-4, tol_v=1e-4, tol_m=1e-8, tol_w=1e-8, iter=1000)
    w = get_lookup(net, 'node', 'w')
    print(net._pit['node'][:, w])
    print(net._internal_results)
    assert np.isclose(net.res_ext_grid.values.sum() + net.res_sink.values.sum() - net.res_source.values.sum(), 0)


def test_two_node_net():
    """

    :return:
    :rtype:
    """

    net = create_empty_network()
    j = create_junction(net, 1, 298.15)
    create_ext_grid(net, j, 1, 298.15, fluid='water')
    create_sink(net, j, 0.01)
    create_source(net, j, 0.02, fluid='water')
    j = create_junction(net, 1, 298.15)
    create_ext_grid(net, j, 1, 298.15, fluid='water')
    create_sink(net, j, 0.01)
    create_source(net, j, 0.02, fluid='water')
    pp.pipeflow(net)

    assert np.all(np.isclose(net.res_ext_grid.values + net.res_sink.values - net.res_source.values, np.zeros((2, 1))))

    net = create_empty_network()
    j = create_junction(net, 1, 298.15)
    create_ext_grid(net, j, 1, 298.15, fluid='lgas')
    create_sink(net, j, 0.01)
    create_source(net, j, 0.02, fluid='lgas')
    j = create_junction(net, 1, 298.15)
    create_ext_grid(net, j, 1, 298.15, fluid='lgas')
    create_sink(net, j, 0.01)
    create_source(net, j, 0.02, fluid='lgas')
    pp.pipeflow(net)

    assert np.all(np.isclose(net.res_ext_grid.values + net.res_sink.values - net.res_source.values, np.zeros((2, 1))))


def test_random_net_and_one_node_net(create_test_net):
    """

    :param create_test_net:
    :type create_test_net:
    :return:
    :rtype:
    """

    net = copy.deepcopy(create_test_net)

    pp.create_fluid_from_lib(net, "water")

    j = create_junction(net, 1, 298.15)
    net.ext_grid.fluid = 'water'
    create_ext_grid(net, j, 1, 298.15, fluid='water')
    create_sink(net, j, 0.01)
    create_source(net, j, 0.02, fluid='water')
    pp.pipeflow(net)

    net = copy.deepcopy(create_test_net)

    pp.create_fluid_from_lib(net, "lgas")

    j = create_junction(net, 1, 298.15)
    net.ext_grid.fluid = 'lgas'
    create_ext_grid(net, j, 1, 298.15, fluid='lgas')
    create_sink(net, j, 0.01)
    create_source(net, j, 0.02, fluid='lgas')
    pp.pipeflow(net)

    assert np.isclose(net.res_ext_grid.values[-1] + net.res_sink.values[-1] - net.res_source.values[-1], 0)


def test_multiple_fluids_sink_source():
    net = pp.create_empty_network()
    same_fluid_twice_defined(net)
    j1 = pp.create_junction(net, 1, 273)
    j2 = pp.create_junction(net, 1, 273)
    j3 = pp.create_junction(net, 1, 273)
    pp.create_ext_grid(net, j1, 1, 273, 'fluid1')
    pp.create_pipe_from_parameters(net, j1, j2, 1, 0.1, 0.1)
    pp.create_pipe_from_parameters(net, j2, j3, 1, 0.1, 0.1)
    pp.create_sink(net, j3, 0.05)
    pp.create_source(net, j2, 0.01, 'fluid2')
    pp.create_source(net, j3, 0.02, 'fluid2')
    pp.pipeflow(net, iter=100)
    print(net._internal_results)
    assert all(net.res_junction.w_fluid1.values == [1., 0.666667, 0.4])


def test_schutterwald_hydrogen():
    net = nets_pps.schutterwald()
    pp.create_sources(net, [5, 168, 193], 6.6e-3, 'hydrogen')
    pp.pipeflow(net, iter=100)
    print(net._internal_results)
    print(np.min(np.abs(net.res_pipe)))


if __name__ == "__main__":
    pytest.main([r'pandapipes/test/api/test_special_networks.py'])
