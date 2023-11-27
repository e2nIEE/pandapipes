# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import copy

import numpy as np
import pytest

import pandapipes
from pandapipes.create import create_empty_network, create_junction, create_ext_grid, create_sink, \
    create_source
from pandapipes.test.pipeflow_internals.test_inservice import create_test_net
from pandapipes.test.test_toolbox import create_net_changed_indices


@pytest.mark.parametrize("use_numba", [True, False])
def test_one_node_net(use_numba):
    """

    :return:
    :rtype:
    """

    net = create_empty_network(fluid='water')
    j = create_junction(net, 1, 298.15)
    create_ext_grid(net, j, 1, 298.15)
    create_sink(net, j, 0.01)
    create_source(net, j, 0.02)
    pandapipes.pipeflow(net, use_numba=use_numba)

    assert np.isclose(net.res_ext_grid.values + net.res_sink.values - net.res_source.values, 0)

    net = create_empty_network(fluid='lgas')
    j = create_junction(net, 1, 298.15)
    create_ext_grid(net, j, 1, 298.15)
    create_sink(net, j, 0.01)
    create_source(net, j, 0.02)
    pandapipes.pipeflow(net, use_numba=use_numba)

    assert np.isclose(net.res_ext_grid.values + net.res_sink.values - net.res_source.values, 0)


@pytest.mark.parametrize("use_numba", [True, False])
def test_two_node_net(use_numba):
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
    pandapipes.pipeflow(net, use_numba=use_numba)

    assert np.all(np.isclose(net.res_ext_grid.values + net.res_sink.values - net.res_source.values,
                             np.zeros((2, 1))))

    net = create_empty_network(fluid='lgas')
    j = create_junction(net, 1, 298.15)
    create_ext_grid(net, j, 1, 298.15)
    create_sink(net, j, 0.01)
    create_source(net, j, 0.02)
    j = create_junction(net, 1, 298.15)
    create_ext_grid(net, j, 1, 298.15)
    create_sink(net, j, 0.01)
    create_source(net, j, 0.02)
    pandapipes.pipeflow(net, use_numba=use_numba)

    assert np.all(np.isclose(net.res_ext_grid.values + net.res_sink.values - net.res_source.values,
                             np.zeros((2, 1))))


@pytest.mark.parametrize("use_numba", [True, False])
def test_random_net_and_one_node_net(create_test_net, use_numba):
    """

    :param create_test_net:
    :type create_test_net:
    :return:
    :rtype:
    """

    net = copy.deepcopy(create_test_net)

    pandapipes.create_fluid_from_lib(net, "water")

    j = create_junction(net, 1, 298.15)
    create_ext_grid(net, j, 1, 298.15)
    create_sink(net, j, 0.01)
    create_source(net, j, 0.02)
    pandapipes.pipeflow(net, use_numba=use_numba)

    net = copy.deepcopy(create_test_net)

    pandapipes.create_fluid_from_lib(net, "lgas")

    j = create_junction(net, 1, 298.15)
    create_ext_grid(net, j, 1, 298.15)
    create_sink(net, j, 0.01)
    create_source(net, j, 0.02)
    pandapipes.pipeflow(net, use_numba=use_numba)

    assert np.isclose(
        net.res_ext_grid.values[-1] + net.res_sink.values[-1] - net.res_source.values[-1], 0)


@pytest.mark.xfail(reason="The test net is not set up properly.")
def test_wild_indexing(create_net_changed_indices):
    net = copy.deepcopy(create_net_changed_indices)

    pandapipes.pipeflow(net)
    assert net["converged"]


if __name__ == "__main__":
    pytest.main([r'pandapipes/test/api/test_special_networks.py'])
