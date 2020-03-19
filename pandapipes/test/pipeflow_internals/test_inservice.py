# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import copy

import pandapipes
import pytest
from pandapipes.pipeflow import logger as pf_logger
import numpy as np
from pandapipes.pipeflow_setup import get_lookup

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
pf_logger.setLevel(logging.WARNING)


@pytest.fixture
def create_test_net():
    """

    :return:
    :rtype:
    """
    net = pandapipes.create_empty_network()
    j1 = pandapipes.create_junction(net, 1, 293.15)
    j2 = pandapipes.create_junction(net, 1, 293.15)
    j3 = pandapipes.create_junction(net, 1, 293.15, in_service=False)
    j4 = pandapipes.create_junction(net, 1, 293.15)
    j5 = pandapipes.create_junction(net, 1, 293.15)
    j6 = pandapipes.create_junction(net, 1, 293.15)
    j7 = pandapipes.create_junction(net, 1, 293.15, in_service=False)

    pandapipes.create_ext_grid(net, j1, 1, 285.15, type="pt")

    pandapipes.create_pipe_from_parameters(net, j1, j2, 0.1, 0.1, sections=1, alpha_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net, j2, j3, 0.1, 0.1, sections=2, alpha_w_per_m2k=5,
                                           in_service=False)
    pandapipes.create_pipe_from_parameters(net, j4, j6, 0.1, 0.1, sections=2, alpha_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net, j6, j7, 0.1, 0.1, sections=1, alpha_w_per_m2k=5,
                                           in_service=False)
    pandapipes.create_pipe_from_parameters(net, j1, j5, 0.1, 0.1, sections=2, alpha_w_per_m2k=5)

    pandapipes.create_valve(net, j1, j4, 0.1)
    pandapipes.create_valve(net, j4, j5, 0.1, opened=False)

    pandapipes.create_sink(net, j2, mdot_kg_per_s=0.1)
    pandapipes.create_sink(net, j3, mdot_kg_per_s=0.1)
    pandapipes.create_sink(net, j5, mdot_kg_per_s=0.1)
    pandapipes.create_sink(net, j6, mdot_kg_per_s=0.1)
    pandapipes.create_sink(net, j7, mdot_kg_per_s=0.1)

    return net


@pytest.fixture
def complex_heat_connectivity_grid():
    net = pandapipes.create_empty_network()
    pandapipes.create_fluid_from_lib(net, "water")

    j1 = pandapipes.create_junction(net, 1, 320.15, index=1)
    j2 = pandapipes.create_junction(net, 1, 320.15, index=2)
    j3 = pandapipes.create_junction(net, 1, 320.15, index=3)
    j4 = pandapipes.create_junction(net, 1, 320.15, index=4, in_service=False)
    j5 = pandapipes.create_junction(net, 1, 320.15, index=5)
    j6 = pandapipes.create_junction(net, 1, 320.15, index=6)
    j7 = pandapipes.create_junction(net, 1, 320.15, index=7)
    j8 = pandapipes.create_junction(net, 1, 320.15, index=8)
    j9 = pandapipes.create_junction(net, 1, 320.15, index=9)
    j10 = pandapipes.create_junction(net, 1, 320.15, index=10)

    pandapipes.create_ext_grid(net, j1, 1, 320.15, type="p", index=5)
    pandapipes.create_ext_grid(net, j7, 1, 320.15, type="t", index=2)
    pandapipes.create_ext_grid(net, j10, 1, 320.15, type="pt", index=1)

    pandapipes.create_pipe_from_parameters(net, j1, j2, 0.1, 0.1, alpha_w_per_m2k=5, index=3)
    pandapipes.create_pipe_from_parameters(net, j1, j3, 0.1, 0.1, alpha_w_per_m2k=5, index=4)
    pandapipes.create_pipe_from_parameters(net, j2, j4, 0.1, 0.1, alpha_w_per_m2k=5,
                                           in_service=False, index=5)
    pandapipes.create_pipe_from_parameters(net, j3, j5, 0.1, 0.1, alpha_w_per_m2k=5,
                                           in_service=False, index=7)
    pandapipes.create_pipe_from_parameters(net, j6, j7, 0.1, 0.1, alpha_w_per_m2k=5, index=9)
    pandapipes.create_pipe_from_parameters(net, j5, 8, 0.1, 0.1, alpha_w_per_m2k=5,
                                           in_service=False, index=8)
    pandapipes.create_pipe_from_parameters(net, j8, j10, 0.1, 0.1, alpha_w_per_m2k=5, index=1)
    pandapipes.create_pipe_from_parameters(net, j9, j10, 0.1, 0.1, alpha_w_per_m2k=5, index=2)

    pandapipes.create_valve(net, j5, j6, 0.1, index=10)
    pandapipes.create_valve(net, j4, j5, 0.1, opened=False, index=12)

    pandapipes.create_sink(net, j3, mdot_kg_per_s=0.1, index=3)
    pandapipes.create_sink(net, j4, mdot_kg_per_s=0.1, index=4)
    pandapipes.create_sink(net, j7, mdot_kg_per_s=0.2, index=5)
    pandapipes.create_sink(net, j9, mdot_kg_per_s=0.1, index=7)
    pandapipes.create_sink(net, j8, mdot_kg_per_s=0.1, index=1)
    pandapipes.create_source(net, j5, mdot_kg_per_s=0.1, index=7)
    pandapipes.create_source(net, j2, mdot_kg_per_s=0.05, index=2)

    return net


def test_inservice_gas(create_test_net):
    """

    :param create_test_net:
    :type create_test_net:
    :return:
    :rtype:
    """
    net = copy.deepcopy(create_test_net)

    pandapipes.create_fluid_from_lib(net, "lgas")

    pandapipes.pipeflow(net, iter=100, tol_p=1e-7, tol_v=1e-7, friction_model="nikuradse")

    assert np.all(np.isnan(net.res_pipe.loc[~net.pipe.in_service]))
    assert np.all(np.isnan(net.res_valve.loc[~net.valve.opened]))
    assert np.all(np.isnan(net.res_junction.loc[~net.junction.in_service]))

    oos_sinks = np.isin(net.sink.junction.values, net.junction.index[~net.junction.in_service]) \
                | ~net.sink.in_service.values
    assert np.all(np.isnan(net.res_sink.loc[oos_sinks]))

    assert not np.any(np.isnan(net.res_pipe.v_mean_m_per_s.loc[net.pipe.in_service].values))
    assert not np.any(np.isnan(net.res_valve.v_mean_m_per_s.loc[net.valve.opened].values))
    assert not np.any(np.isnan(net.res_junction.p_bar.loc[net.junction.in_service].values))
    assert not np.any(np.isnan(net.res_sink.loc[~oos_sinks]))

    assert np.allclose(net.res_ext_grid.mdot_kg_per_s.sum(), -net.res_sink.mdot_kg_per_s.sum(),
                       rtol=1e-10, atol=0)


def test_inservice_water(create_test_net):
    """

    :param create_test_net:
    :type create_test_net:
    :return:
    :rtype:
    """
    net = copy.deepcopy(create_test_net)

    pandapipes.create_fluid_from_lib(net, "water")

    pandapipes.pipeflow(net, iter=100, tol_p=1e-7, tol_v=1e-7, friction_model="nikuradse")

    assert np.all(np.isnan(net.res_pipe.loc[~net.pipe.in_service]))
    assert np.all(np.isnan(net.res_valve.loc[~net.valve.opened]))
    assert np.all(np.isnan(net.res_junction.loc[~net.junction.in_service]))

    oos_sinks = np.isin(net.sink.junction.values, net.junction.index[~net.junction.in_service]) \
                | ~net.sink.in_service.values
    assert np.all(np.isnan(net.res_sink.loc[oos_sinks]))

    assert not any(np.isnan(net.res_pipe.v_mean_m_per_s.loc[net.pipe.in_service].values))
    assert not any(np.isnan(net.res_valve.v_mean_m_per_s.loc[net.valve.opened].values))
    assert not any(np.isnan(net.res_junction.p_bar.loc[net.junction.in_service].values))
    assert not np.any(np.isnan(net.res_sink.loc[~oos_sinks]))

    assert np.allclose(net.res_ext_grid.mdot_kg_per_s.sum(), -net.res_sink.mdot_kg_per_s.sum(),
                       rtol=1e-10, atol=0)


def test_connectivity_hydraulic(create_test_net):
    """

    :param create_test_net:
    :type create_test_net:
    :return:
    :rtype:
    """
    net = copy.deepcopy(create_test_net)

    net.junction.in_service = True
    net.pipe.in_service = [True, False, False, True, True]

    pandapipes.create_fluid_from_lib(net, "water")

    pandapipes.pipeflow(net, iter=100, tol_p=1e-7, tol_v=1e-7, friction_model="nikuradse")

    assert np.all(np.isnan(net.res_junction.loc[[2, 5, 6], :]))
    assert np.all(np.isnan(net.res_pipe.loc[[1, 2, 3], :]))
    assert not np.any(np.isnan(net.res_junction.loc[[0, 1, 3, 4], :]))
    assert not np.any(np.isnan(net.res_pipe.loc[[0, 4],
                                                ["v_mean_m_per_s", "p_from_bar", "p_to_bar"]]))
    assert not np.any(np.isnan(net.res_sink.loc[[0, 2], "mdot_kg_per_s"]))
    assert np.all(np.isnan(net.res_sink.loc[[1, 3, 4], "mdot_kg_per_s"]))

    assert np.allclose(net.res_ext_grid.mdot_kg_per_s.sum(), -net.res_sink.mdot_kg_per_s.sum(),
                       rtol=1e-10, atol=0)

    active_branches = get_lookup(net, "branch", "active")
    active_nodes = get_lookup(net, "node", "active")

    assert np.all(active_nodes == np.array([True, True, False, True, True, False, False, False,
                                            False, True]))
    assert np.all(active_branches == np.array([True, False, False, False, False, False, True,
                                               True, True, False]))


def test_connectivity_heat1(complex_heat_connectivity_grid):
    net = copy.deepcopy(complex_heat_connectivity_grid)
    pandapipes.pipeflow(net, mode="all", check_connectivity=True)

    assert set(net.res_junction.loc[net.res_junction.p_bar.notnull()].index) == {8, 9, 10}
    assert set(net.res_junction.loc[net.res_junction.p_bar.isnull()].index) \
           == set(net.junction.index) - {8, 9, 10}
    assert set(net.res_pipe.loc[net.res_pipe.v_mean_m_per_s.notnull()].index) == {1, 2}
    assert set(net.res_pipe.loc[net.res_pipe.v_mean_m_per_s.isnull()].index) \
           == set(net.pipe.index) - {1, 2}
    assert set(net.res_valve.loc[net.res_valve.v_mean_m_per_s.notnull()].index) == set()
    assert set(net.res_valve.loc[net.res_valve.v_mean_m_per_s.isnull()].index) \
           == set(net.valve.index)
    assert set(net.res_sink.loc[net.res_sink.mdot_kg_per_s.isnull()].index) == {3, 4, 5}
    assert set(net.res_sink.loc[net.res_sink.mdot_kg_per_s.notnull()].index) == \
           set(net.sink.index) - {3, 4, 5}
    assert set(net.res_source.loc[net.res_source.mdot_kg_per_s.isnull()].index) == \
           set(net.source.index)
    assert set(net.res_source.loc[net.res_source.mdot_kg_per_s.notnull()].index) == set()

    assert np.allclose(net.res_ext_grid.mdot_kg_per_s.sum(),
                       -net.res_sink.mdot_kg_per_s.sum() + net.res_source.mdot_kg_per_s.sum(),
                       rtol=1e-10, atol=0)


def test_connectivity_heat2(complex_heat_connectivity_grid):
    net = copy.deepcopy(complex_heat_connectivity_grid)
    net.pipe.at[7, "in_service"] = True
    pandapipes.pipeflow(net, mode="all", check_connectivity=True)

    assert set(net.res_junction.loc[net.res_junction.p_bar.isnull()].index) == {4}
    assert set(net.res_junction.loc[net.res_junction.p_bar.notnull()].index) \
           == set(net.junction.index) - {4}
    assert set(net.res_pipe.loc[net.res_pipe.v_mean_m_per_s.notnull()].index) \
           == set(net.pipe.loc[net.pipe.in_service].index)
    assert set(net.res_pipe.loc[net.res_pipe.v_mean_m_per_s.isnull()].index) \
           == set(net.pipe.loc[~net.pipe.in_service].index)
    assert set(net.res_valve.loc[net.res_valve.v_mean_m_per_s.notnull()].index) == {10}
    assert set(net.res_valve.loc[net.res_valve.v_mean_m_per_s.isnull()].index) \
           == set(net.valve.index) - {10}
    assert set(net.res_sink.loc[net.res_sink.mdot_kg_per_s.isnull()].index) == {4}
    assert set(net.res_sink.loc[net.res_sink.mdot_kg_per_s.notnull()].index) == \
           set(net.sink.index) - {4}
    assert set(net.res_source.loc[net.res_source.mdot_kg_per_s.isnull()].index) == set()
    assert set(net.res_source.loc[net.res_source.mdot_kg_per_s.notnull()].index) == \
           set(net.source.index)

    assert np.allclose(net.res_ext_grid.mdot_kg_per_s.sum(),
                       -net.res_sink.mdot_kg_per_s.sum() + net.res_source.mdot_kg_per_s.sum(),
                       rtol=1e-10, atol=0)


def test_connectivity_heat3(complex_heat_connectivity_grid):
    net = copy.deepcopy(complex_heat_connectivity_grid)
    net.pipe.at[5, "in_service"] = True
    net.valve.at[12, "opened"] = True
    net.junction.at[5, "in_service"] = False
    pandapipes.pipeflow(net, mode="all", check_connectivity=True)

    assert set(net.res_junction.loc[net.res_junction.p_bar.isnull()].index) == set()
    assert set(net.res_junction.loc[net.res_junction.p_bar.notnull()].index) \
           == set(net.junction.index)
    assert set(net.res_pipe.loc[net.res_pipe.v_mean_m_per_s.notnull()].index) \
           == set(net.pipe.loc[net.pipe.in_service].index)
    assert set(net.res_pipe.loc[net.res_pipe.v_mean_m_per_s.isnull()].index) \
           == set(net.pipe.loc[~net.pipe.in_service].index)
    assert set(net.res_valve.loc[net.res_valve.v_mean_m_per_s.isnull()].index) == set()
    assert set(net.res_valve.loc[net.res_valve.v_mean_m_per_s.notnull()].index) \
           == set(net.valve.index)
    assert set(net.res_sink.loc[net.res_sink.mdot_kg_per_s.isnull()].index) == set()
    assert set(net.res_sink.loc[net.res_sink.mdot_kg_per_s.notnull()].index) == set(net.sink.index)
    assert set(net.res_source.loc[net.res_source.mdot_kg_per_s.isnull()].index) == set()
    assert set(net.res_source.loc[net.res_source.mdot_kg_per_s.notnull()].index) == \
           set(net.source.index)

    assert np.allclose(net.res_ext_grid.mdot_kg_per_s.sum(),
                       -net.res_sink.mdot_kg_per_s.sum() + net.res_source.mdot_kg_per_s.sum(),
                       rtol=1e-10, atol=0)


@pytest.mark.xfail(
    reason="reduced_pit does not reindex the reduced nodes correctly, will be fixed.")
def test_exclude_unconnected_junction():
    """
    test if unconnected junctions that do not have the highest index are excluded correctly
    (pipeflow fails if reduced_pit does not reindex the reduced nodes correctly)
    :return:
    :rtype:
    """
    net = pandapipes.create_empty_network(fluid="lgas")

    j1 = pandapipes.create_junction(net, pn_bar=1.05, tfluid_k=293.15, name="Junction 1")
    _ = pandapipes.create_junction(net, pn_bar=1.05, tfluid_k=293.15, name="unconnected junction")
    j3 = pandapipes.create_junction(net, pn_bar=1.05, tfluid_k=293.15, name="Junction 3")

    pandapipes.create_ext_grid(net, junction=j1, p_bar=1.1, t_k=293.15)
    pandapipes.create_sink(net, junction=j3, mdot_kg_per_s=0.045)
    pandapipes.create_pipe_from_parameters(net, from_junction=j1, to_junction=j3, length_km=0.1,
                                           diameter_m=0.05)
    pandapipes.pipeflow(net)
    assert net.converged


if __name__ == "__main__":
    pytest.main([r'pandapipes/test/pipeflow_internals/test_inservice.py'])
