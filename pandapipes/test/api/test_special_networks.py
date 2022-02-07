# Copyright (c) 2020-2022 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import copy

import numpy as np
import pandapipes as pp
import pytest
from pandapipes.create import create_empty_network, create_junction, create_ext_grid, create_sink, create_source
from pandapipes.test.pipeflow_internals.test_inservice import create_test_net


def test_one_node_net():
    """

    :return:
    :rtype:
    """

    net = create_empty_network(fluid='water')
    j = create_junction(net, 1, 298.15)
    create_ext_grid(net, j, 1, 298.15)
    create_sink(net, j, 0.01)
    create_source(net, j, 0.02)
    pp.pipeflow(net)

    assert np.isclose(net.res_ext_grid.values + net.res_sink.values - net.res_source.values, 0)

    net = create_empty_network(fluid='lgas')
    j = create_junction(net, 1, 298.15)
    create_ext_grid(net, j, 1, 298.15)
    create_sink(net, j, 0.01)
    create_source(net, j, 0.02)
    pp.pipeflow(net)

    assert np.isclose(net.res_ext_grid.values + net.res_sink.values - net.res_source.values, 0)


def test_two_node_net():
    """

    :return:
    :rtype:
    """

    net = create_empty_network(fluid='water')
    j = create_junction(net, 1, 298.15)
    create_ext_grid(net, j, 1, 298.15)
    create_sink(net, j, 0.01)
    create_source(net, j, 0.02)
    j = create_junction(net, 1, 298.15)
    create_ext_grid(net, j, 1, 298.15)
    create_sink(net, j, 0.01)
    create_source(net, j, 0.02)
    pp.pipeflow(net)

    assert np.all(np.isclose(net.res_ext_grid.values + net.res_sink.values - net.res_source.values, np.zeros((2, 1))))

    net = create_empty_network(fluid='lgas')
    j = create_junction(net, 1, 298.15)
    create_ext_grid(net, j, 1, 298.15)
    create_sink(net, j, 0.01)
    create_source(net, j, 0.02)
    j = create_junction(net, 1, 298.15)
    create_ext_grid(net, j, 1, 298.15)
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
    create_ext_grid(net, j, 1, 298.15)
    create_sink(net, j, 0.01)
    create_source(net, j, 0.02)
    pp.pipeflow(net)

    net = copy.deepcopy(create_test_net)

    pp.create_fluid_from_lib(net, "lgas")

    j = create_junction(net, 1, 298.15)
    create_ext_grid(net, j, 1, 298.15)
    create_sink(net, j, 0.01)
    create_source(net, j, 0.02)
    pp.pipeflow(net)

    assert np.isclose(net.res_ext_grid.values[-1] + net.res_sink.values[-1] - net.res_source.values[-1], 0)


if __name__ == "__main__":
    pytest.main([r'pandapipes/test/api/test_special_networks.py'])
