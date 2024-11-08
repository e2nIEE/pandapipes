# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import copy

import numpy as np
import pytest

import pandapipes
from pandapipes.create import create_empty_network, create_junction, create_ext_grid, create_sink, create_source
from pandapipes.pf.pipeflow_setup import get_net_option
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

    max_iter_hyd = 2 if use_numba else 2
    pandapipes.pipeflow(net, use_numba=use_numba, max_iter_hyd=max_iter_hyd)

    assert np.isclose(net.res_ext_grid.values + net.res_sink.values - net.res_source.values, 0)

    net = create_empty_network(fluid='lgas')
    j = create_junction(net, 1, 298.15)
    create_ext_grid(net, j, 1, 298.15)
    create_sink(net, j, 0.01)
    create_source(net, j, 0.02)
    pandapipes.pipeflow(net, use_numba=use_numba, max_iter_hyd=max_iter_hyd)

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
    max_iter_hyd = 2 if use_numba else 2
    pandapipes.pipeflow(net, use_numba=use_numba, max_iter_hyd=max_iter_hyd)

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
    pandapipes.pipeflow(net, use_numba=use_numba, max_iter_hyd=max_iter_hyd)

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
    max_iter_hyd = 3 if use_numba else 3
    pandapipes.pipeflow(net, use_numba=use_numba, max_iter_hyd=max_iter_hyd)

    net = copy.deepcopy(create_test_net)

    pandapipes.create_fluid_from_lib(net, "lgas")

    j = create_junction(net, 1, 298.15)
    create_ext_grid(net, j, 1, 298.15)
    create_sink(net, j, 0.01)
    create_source(net, j, 0.02)
    pandapipes.pipeflow(net, use_numba=use_numba, max_iter_hyd=max_iter_hyd)

    assert np.isclose(
        net.res_ext_grid.values[-1] + net.res_sink.values[-1] - net.res_source.values[-1], 0)


@pytest.mark.xfail(reason="The test net is not set up properly.")
def test_wild_indexing(create_net_changed_indices):
    net = copy.deepcopy(create_net_changed_indices)
    pandapipes.pipeflow(net)
    assert net.converged


def test_valve_flow_contrl_heat_consumer():
    net = pandapipes.create_empty_network(fluid='water')
    j1, j2, j3, j4, j5, j6, j7, j8, j9 = pandapipes.create_junctions(net, 9, 5, 400)
    pandapipes.create_circ_pump_const_pressure(net, j9, j1, 5, 2, 400)
    pandapipes.create_valves(net, [j1, j4, j6, j8], [j2, j5, j7, j9], 0.1)
    pandapipes.create_flow_control(net, j3, j2, 10)
    pandapipes.create_heat_exchanger(net, j8, j3, -5000)
    pandapipes.create_pipes_from_parameters(net, [j2, j7], [j4, j8], 0.1, 0.1)
    pandapipes.create_heat_consumer(net, j5, j6, 10000, 10)
    net.valve.loc[0, 'opened'] = False
    pandapipes.pipeflow(net, mode='sequential')

    amb_temp = get_net_option(net, 'ambient_temperature')

    assert np.all(np.isclose(net.res_junction.loc[[j9, j8, j7, j6, j3], 'p_bar'].values, 3.))
    assert np.all(np.isnan(net.res_junction.loc[[j5, j4, j2], 'p_bar']).values)
    assert np.all(np.isclose(net.res_junction.loc[[j1], 'p_bar'].values, 5.))
    assert np.all(np.isclose(net.res_junction.t_k.values[1:], amb_temp))

    net.valve.loc[0, 'opened'] = True
    net.valve.loc[3, 'opened'] = False
    pandapipes.pipeflow(net, mode='sequential')

    assert np.all(np.isclose(net.res_junction.loc[[j9], 'p_bar'].values, 3.))
    assert np.all(np.isnan(net.res_junction.loc[[j8, j7, j6, j3], 'p_bar']).values)
    assert np.all(np.isclose(net.res_junction.loc[[j1, j2, j4, j5], 'p_bar'].values, 5.))
    assert np.all(np.isclose(net.res_junction.t_k.values[1:], amb_temp))

    net.valve.loc[3, 'opened'] = True
    net.valve.loc[1, 'opened'] = False
    pandapipes.pipeflow(net, mode='sequential')

    assert np.all(np.isclose(net.res_junction.loc[[j9, j8, j7, j6, j3], 'p_bar'].values, 3.))
    assert np.all(np.isnan(net.res_junction.loc[[j5], 'p_bar']).values)
    assert np.all(np.isclose(net.res_junction.loc[[j1, j2, j4], 'p_bar'].values, 5.))
    assert np.all(np.isclose(net.res_junction.t_k.values[[j4, j5, j6, j7]], amb_temp))

    net.valve.loc[1, 'opened'] = True
    net.valve.loc[2, 'opened'] = False
    pandapipes.pipeflow(net, mode='sequential')

    assert np.all(np.isclose(net.res_junction.loc[[j9, j8, j7, j3], 'p_bar'].values, 3.))
    assert np.all(np.isnan(net.res_junction.loc[[j6], 'p_bar']).values)
    assert np.all(np.isclose(net.res_junction.loc[[j1, j2, j4, j5], 'p_bar'].values, 5.))
    assert np.all(np.isclose(net.res_junction.t_k.values[[j4, j5, j6, j7]], amb_temp))

    net.valve.loc[1, 'opened'] = False
    pandapipes.pipeflow(net, mode='sequential')

    assert np.all(np.isnan(net.res_junction.p_bar.values[4:6]))

    net.valve.loc[:, 'opened'] = False
    pandapipes.pipeflow(net, mode='sequential')

    assert np.all(np.isnan(net.res_junction.p_bar.values[1:-1]))


if __name__ == "__main__":
    pytest.main([r'pandapipes/test/api/test_special_networks.py'])
