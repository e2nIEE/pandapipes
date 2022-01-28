# Copyright (c) 2020-2021 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import copy

import numpy as np
import pandapipes as pp
import pytest
from pandapipes.create import create_empty_network, create_junction, create_ext_grid, create_sink, create_source, \
    create_pipe_from_parameters
from pandapipes.test.pipeflow_internals.test_inservice import create_test_net


def test_one_node_net():
    """

    :return:
    :rtype:
    """

    net = create_empty_network()
    j = create_junction(net, 1, 298.15)
    create_ext_grid(net, j, 1, 298.15, fluid='water')
    create_sink(net, j, 0.01)
    create_source(net, j, 0.02)
    pp.pipeflow(net)

    assert np.isclose(net.res_ext_grid.values + net.res_sink.values - net.res_source.values, 0)

    net = create_empty_network()
    j = create_junction(net, 1, 298.15)
    create_ext_grid(net, j, 1, 298.15, fluid='lgas')
    create_sink(net, j, 0.01)
    create_source(net, j, 0.02)
    pp.pipeflow(net)

    assert np.isclose(net.res_ext_grid.values + net.res_sink.values - net.res_source.values, 0)


def test_two_node_net_with_two_different_fluids():
    """

    :return:
    :rtype:
    """

    net = create_empty_network()
    j = create_junction(net, 1, 298.15, index=50)
    j1 = create_junction(net, 1, 298.15, index=51)
    j2 = create_junction(net, 1, 298.15, index=52)
    create_ext_grid(net, j2, 1, 298.15, fluid='hgas', index=100)
    create_ext_grid(net, j, 1, 298.15, fluid='hgas', index=101)
    create_sink(net, j1, 0.2)
    create_source(net, j, 0.02, fluid='hydrogen')
    create_source(net, j, 0.02, fluid='hydrogen')
    create_source(net, j, 0.02, fluid='lgas')
    create_source(net, j2, 0.02, fluid='lgas')
    create_pipe_from_parameters(net, j, j1, 0.01, 0.1, 0.01)
    pp.pipeflow(net, tol_p= 1e-4, tol_v= 1e-4, iter=400)
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
    create_source(net, j, 0.02)
    j = create_junction(net, 1, 298.15)
    create_ext_grid(net, j, 1, 298.15, fluid='water')
    create_sink(net, j, 0.01)
    create_source(net, j, 0.02)
    pp.pipeflow(net)

    assert np.all(np.isclose(net.res_ext_grid.values + net.res_sink.values - net.res_source.values, np.zeros((2, 1))))

    net = create_empty_network()
    j = create_junction(net, 1, 298.15)
    create_ext_grid(net, j, 1, 298.15, fluid='lgas')
    create_sink(net, j, 0.01)
    create_source(net, j, 0.02)
    j = create_junction(net, 1, 298.15)
    create_ext_grid(net, j, 1, 298.15, fluid='lgas')
    create_sink(net, j, 0.01)
    create_source(net, j, 0.02)
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
    create_source(net, j, 0.02)
    pp.pipeflow(net)

    net = copy.deepcopy(create_test_net)

    pp.create_fluid_from_lib(net, "lgas")

    j = create_junction(net, 1, 298.15)
    net.ext_grid.fluid = 'lgas'
    create_ext_grid(net, j, 1, 298.15, fluid='lgas')
    create_sink(net, j, 0.01)
    create_source(net, j, 0.02)
    pp.pipeflow(net)

    assert np.isclose(net.res_ext_grid.values[-1] + net.res_sink.values[-1] - net.res_source.values[-1], 0)


if __name__ == "__main__":
    pytest.main([r'pandapipes/test/api/test_special_networks.py'])
