# Copyright (c) 2020-2025 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import copy

import numpy as np
import pytest

import pandapipes
from pandapipes.create import create_empty_network, create_junction, create_ext_grid, create_sink, create_source
from pandapipes.pf.pipeflow_setup import get_net_option, PipeflowNotConverged
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


@pytest.mark.parametrize("use_numba", [True, False])
def test_valve_flow_contrl_heat_consumer(use_numba):
    net = pandapipes.create_empty_network(fluid='water')
    j1f, j2f, j3f, j4f, j1r, j2r, j3r, j4r, j5 = pandapipes.create_junctions(net, 9, 5, 400)
    pandapipes.create_circ_pump_const_pressure(net, j4r, j1f, 5, 2, 400)
    pandapipes.create_valves(net, [j1f, j3f, j1r, j3r], [j2f, j4f, j2r, j4r], 0.1)
    pandapipes.create_flow_control(net, j2f, j5, 10)
    pandapipes.create_heat_exchanger(net, j5, j3r, 5000)
    pandapipes.create_pipes_from_parameters(net, [j2f, j2r], [j3f, j3r], 0.1, 0.1)
    pandapipes.create_heat_consumer(net, j4f, j1r, 10000, 10)

    net.valve.loc[0, 'opened'] = False
    with pytest.raises(PipeflowNotConverged):
        pandapipes.pipeflow(net, mode='sequential', use_numba=use_numba)

    amb_temp = get_net_option(net, 'ambient_temperature')

    pandapipes.pipeflow(net, mode='hydraulics', use_numba=use_numba)

    assert np.all(np.isclose(net.res_junction.loc[[j1r, j2r, j3r, j4r, j5], 'p_bar'].values, 3.))
    assert np.all(np.isnan(net.res_junction.loc[[j2f, j3f, j4f], 'p_bar']).values)
    assert np.all(np.isclose(net.res_junction.loc[[j1f], 'p_bar'].values, 5.))

    net.valve.loc[0, 'opened'] = True
    net.valve.loc[3, 'opened'] = False
    with pytest.raises(PipeflowNotConverged):
        pandapipes.pipeflow(net, mode='sequential', use_numba=use_numba)

    pandapipes.pipeflow(net, mode='hydraulics', use_numba=use_numba)

    assert np.all(np.isclose(net.res_junction.loc[[j4r], 'p_bar'].values, 3.))
    assert np.all(np.isnan(net.res_junction.loc[[j5, j1r, j2r, j3r], 'p_bar']).values)
    assert np.all(np.isclose(net.res_junction.loc[[j1f, j2f, j3f, j4f], 'p_bar'].values, 5.))

    net.valve.loc[3, 'opened'] = True
    net.valve.loc[1, 'opened'] = False
    pandapipes.pipeflow(net, mode='sequential', use_numba=use_numba)

    assert np.all(np.isclose(net.res_junction.loc[[j4r, j3r, j2r, j1r, j5], 'p_bar'].values, 3.))
    assert np.all(np.isnan(net.res_junction.loc[[j4f], 'p_bar']).values)
    assert np.all(np.isclose(net.res_junction.loc[[j1f, j2f, j3f], 'p_bar'].values, 5.))
    assert np.all(np.isclose(net.res_junction.t_k.values[[j3f, j4f, j1r, j2r]], amb_temp))

    net.valve.loc[1, 'opened'] = True
    net.valve.loc[2, 'opened'] = False
    pandapipes.pipeflow(net, mode='sequential', use_numba=use_numba)

    assert np.all(np.isclose(net.res_junction.loc[[j4r, j3r, j2r, j5], 'p_bar'].values, 3.))
    assert np.all(np.isnan(net.res_junction.loc[[j1r], 'p_bar']).values)
    assert np.all(np.isclose(net.res_junction.loc[[j1f, j2f, j3f, j4f], 'p_bar'].values, 5.))
    assert np.all(np.isclose(net.res_junction.t_k.values[[j3f, j4f, j1r, j2r]], amb_temp))

    net.valve.loc[1, 'opened'] = False
    pandapipes.pipeflow(net, mode='sequential', use_numba=use_numba)

    assert np.all(np.isnan(net.res_junction.loc[[j1r, j4f], 'p_bar'].values))

    net.valve.loc[:, 'opened'] = False
    with pytest.raises(PipeflowNotConverged):
        pandapipes.pipeflow(net, mode='sequential', use_numba=use_numba)

    assert np.all(np.isnan(net.res_junction.p_bar.values[1:-1]))

    net.heat_consumer.loc[0, 'controlled_mdot_kg_per_s'] = 11
    net.valve.loc[:, 'opened'] = True
    net.valve.loc[3, 'opened'] = False
    pandapipes.create_circ_pump_const_pressure(net, j3r, j1f, 5, 2, 400)

    pandapipes.pipeflow(net, mode='sequential', use_numba=use_numba)



if __name__ == "__main__":
    pytest.main([r'pandapipes/test/api/test_special_networks.py'])
