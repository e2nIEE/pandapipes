# Copyright (c) 2020-2025 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import copy

import numpy as np
import pytest

import pandapipes
from pandapipes.pf.pipeflow_setup import get_lookup
from pandapipes.pipeflow import PipeflowNotConverged
from pandapipes.pipeflow import logger as pf_logger

try:
    import pandaplan.core.pplog as logging
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

    pandapipes.create_pipe_from_parameters(net, j1, j2, 0.1, 0.1, sections=1, u_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net, j2, j3, 0.1, 0.1, sections=2, u_w_per_m2k=5,
                                           in_service=False)
    pandapipes.create_pipe_from_parameters(net, j4, j6, 0.1, 0.1, sections=2, u_w_per_m2k=5)
    pandapipes.create_pipe_from_parameters(net, j6, j7, 0.1, 0.1, sections=1, u_w_per_m2k=5,
                                           in_service=False)
    pandapipes.create_pipe_from_parameters(net, j1, j5, 0.1, 0.1, sections=2, u_w_per_m2k=5)

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

    pandapipes.create_pipe_from_parameters(net, j1, j2, 0.1, 0.1, u_w_per_m2k=5, index=3)
    pandapipes.create_pipe_from_parameters(net, j1, j3, 0.1, 0.1, u_w_per_m2k=5, index=4)
    pandapipes.create_pipe_from_parameters(net, j2, j4, 0.1, 0.1, u_w_per_m2k=5,
                                           in_service=False, index=5)
    pandapipes.create_pipe_from_parameters(net, j3, j5, 0.1, 0.1, u_w_per_m2k=5,
                                           in_service=False, index=7)
    pandapipes.create_pipe_from_parameters(net, j6, j7, 0.1, 0.1, u_w_per_m2k=5, index=9)
    pandapipes.create_pipe_from_parameters(net, j5, j8, 0.1, 0.1, u_w_per_m2k=5,
                                           in_service=False, index=8)
    pandapipes.create_pipe_from_parameters(net, j8, j10, 0.1, 0.1, u_w_per_m2k=5, index=1)
    pandapipes.create_pipe_from_parameters(net, j9, j10, 0.1, 0.1, u_w_per_m2k=5, index=2)

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


@pytest.fixture
def create_mixed_indexing_grid():
    net = pandapipes.create_empty_network()
    pandapipes.create_fluid_from_lib(net, "lgas")
    pandapipes.create_junctions(net, 11, 1.0, 283.15, index=[1, 3, 5, 10, 12, 14, 9, 8, 7, 6, 15])
    pandapipes.create_ext_grid(net, 1, 1.0, 283.15)
    pandapipes.create_ext_grid(net, 5, 1.0, 283.15)
    pandapipes.create_pipes_from_parameters(
        net, [1, 5, 3, 14, 14, 8], [3, 3, 10, 6, 9, 7], 0.5, 0.12, sections=[1, 1, 1, 2, 3, 1],
        index=[0, 3, 10, 7, 6, 8])
    pandapipes.create_valves(net, [3, 10, 6], [14, 12, 15], 0.2, index=[3, 5, 2])
    pandapipes.create_pressure_control(net, 9, 8, 8, 0.7)
    pandapipes.create_sinks(net, [10, 6, 15, 7], 0.1, index=[3, 5, 1, 2],
                            in_service=[True, False, True, True])
    pandapipes.create_source(net, 12, 0.05, index=2)
    pandapipes.create_source(net, 9, 0.06, index=4, in_service=False)
    return net


@pytest.fixture
def create_net_wo_ext_grid():
    net = pandapipes.create_empty_network("net", add_stdtypes=False)
    pandapipes.create_fluid_from_lib(net, "hgas", overwrite=True)
    pandapipes.create_junction(net, index=3, pn_bar=16, tfluid_k=283, height_m=0,
                               name="Junction 3", in_service=True,
                               type="junction", geodata=(0, 0))
    pandapipes.create_junction(net, index=9, pn_bar=16, tfluid_k=283, height_m=0,
                               name="Junction 9", in_service=True,
                               type="junction", geodata=(1, 0))
    pandapipes.create_junction(net, index=10, pn_bar=16, tfluid_k=283, height_m=0,
                               name="Junction 10", in_service=True,
                               type="junction", geodata=(2, 0))
    pandapipes.create_pipe_from_parameters(net, 9, 10, length_km=1, diameter_m=0.03, k_mm=.1, sections=10,
                                           u_w_per_m2k=1, name="Pipe 6")
    pandapipes.create_sink(net, 9, mdot_kg_per_s=0.01, name="Sink 3")
    pandapipes.create_source(net, junction=10, mdot_kg_per_s=0.04, name="Source 3")
    pandapipes.create_compressor(net, from_junction=9, to_junction=3, pressure_ratio=1.1,
                                 name="Compressor 0", index=None, in_service=True)
    return net


@pytest.mark.parametrize("use_numba", [True, False])
def test_inservice_gas(create_test_net, use_numba):
    """

    :param create_test_net:
    :type create_test_net:
    :return:
    :rtype:
    """
    net = copy.deepcopy(create_test_net)

    pandapipes.create_fluid_from_lib(net, "lgas")

    max_iter_hyd = 3 if use_numba else 3
    pandapipes.pipeflow(net, max_iter_hyd=max_iter_hyd,
                        tol_p=1e-7, tol_m=1e-7, friction_model="nikuradse",
                        use_numba=use_numba)

    assert np.all(np.isnan(net.res_pipe.loc[~net.pipe.in_service, :].values))
    assert np.all(np.isnan(net.res_valve.loc[~net.valve.opened, :].values))
    assert np.all(np.isnan(net.res_junction.p_bar.loc[~net.junction.in_service].values))

    oos_sinks = np.isin(net.sink.junction.values, net.junction.index[~net.junction.in_service]) \
                | ~net.sink.in_service.values
    assert np.all(np.isnan(net.res_sink.loc[oos_sinks, :].values))

    assert not np.any(np.isnan(net.res_pipe.v_mean_m_per_s.loc[net.pipe.in_service].values))
    assert not np.any(np.isnan(net.res_valve.v_mean_m_per_s.loc[net.valve.opened].values))
    assert not np.any(np.isnan(net.res_junction.p_bar.loc[net.junction.in_service].values))
    assert not np.any(np.isnan(net.res_sink.loc[~oos_sinks, :].values))

    assert np.allclose(net.res_ext_grid.mdot_kg_per_s.sum(), -net.res_sink.mdot_kg_per_s.sum(),
                       rtol=1e-10, atol=0)


@pytest.mark.parametrize("use_numba", [True, False])
def test_inservice_water(create_test_net, use_numba):
    """

    :param create_test_net:
    :type create_test_net:
    :return:
    :rtype:
    """
    net = copy.deepcopy(create_test_net)

    pandapipes.create_fluid_from_lib(net, "water")

    max_iter_hyd = 3 if use_numba else 3
    pandapipes.pipeflow(net, max_iter_hyd=max_iter_hyd,
                        tol_p=1e-7, tol_m=1e-7, friction_model="nikuradse",
                        use_numba=use_numba)

    assert np.all(np.isnan(net.res_pipe.loc[~net.pipe.in_service, :].values))
    assert np.all(np.isnan(net.res_valve.loc[~net.valve.opened, :].values))
    assert np.all(np.isnan(net.res_junction.p_bar.loc[~net.junction.in_service].values))

    oos_sinks = np.isin(net.sink.junction.values, net.junction.index[~net.junction.in_service]) \
                | ~net.sink.in_service.values
    assert np.all(np.isnan(net.res_sink.loc[oos_sinks, :].values))

    assert not any(np.isnan(net.res_pipe.v_mean_m_per_s.loc[net.pipe.in_service].values))
    assert not any(np.isnan(net.res_valve.v_mean_m_per_s.loc[net.valve.opened].values))
    assert not any(np.isnan(net.res_junction.p_bar.loc[net.junction.in_service].values))
    assert not np.any(np.isnan(net.res_sink.loc[~oos_sinks, :].values))

    assert np.allclose(net.res_ext_grid.mdot_kg_per_s.sum(), -net.res_sink.mdot_kg_per_s.sum(),
                       rtol=1e-10, atol=0)


@pytest.mark.parametrize("use_numba", [True, False])
def test_connectivity_hydraulic(create_test_net, use_numba):
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

    max_iter_hyd = 3 if use_numba else 3
    pandapipes.pipeflow(net, max_iter_hyd=max_iter_hyd, tol_p=1e-7, tol_m=1e-7, friction_model="nikuradse",
                        use_numba=use_numba)

    assert np.all(np.isnan(net.res_junction.p_bar.loc[[2, 5, 6]].values))
    assert np.all(np.isnan(net.res_pipe.loc[[1, 2, 3], :].values))
    assert not np.any(np.isnan(net.res_junction.loc[[0, 1, 3, 4], :].values))
    assert not np.any(np.isnan(net.res_pipe.loc[[0, 4], ["v_mean_m_per_s", "p_from_bar", "p_to_bar"]].values))
    assert not np.any(np.isnan(net.res_sink.loc[[0, 2], "mdot_kg_per_s"].values))
    assert np.all(np.isnan(net.res_sink.loc[[1, 3, 4], "mdot_kg_per_s"].values))

    assert np.allclose(net.res_ext_grid.mdot_kg_per_s.sum(), -net.res_sink.mdot_kg_per_s.sum(),
                       rtol=1e-10, atol=0)

    active_branches = get_lookup(net, "branch", "active_hydraulics")
    active_nodes = get_lookup(net, "node", "active_hydraulics")

    assert np.all(active_nodes == np.array([True, True, False, True, True, False, False, False,
                                            False, True]))
    assert np.all(active_branches == np.array([True, False, False, False, False, False, True,
                                               True, True, False]))


@pytest.mark.parametrize("use_numba", [True, False])
def test_connectivity_hydraulic2(create_test_net, use_numba):
    net = copy.deepcopy(create_test_net)

    net.junction.in_service = True
    net.pipe.in_service = True
    net.valve.opened = True

    pandapipes.create_fluid_from_lib(net, "water")

    max_iter_hyd = 7 if use_numba else 7
    pandapipes.pipeflow(net, max_iter_hyd=max_iter_hyd, tol_p=1e-7, tol_m=1e-7, friction_model="nikuradse",
                        use_numba=use_numba)

    pandapipes.create_junction(net, 1., 293.15)
    pandapipes.create_junction(net, 1., 293.15)
    j = pandapipes.create_junction(net, 1., 293.15)
    pandapipes.create_junction(net, 1, 293.15)
    pandapipes.create_junction(net, 1, 293.15)

    pandapipes.create_pipe_from_parameters(net, 0, j, 0.1, 0.1)

    pandapipes.create_sink(net, j, 0.1)

    pandapipes.pipeflow(net, max_iter_hyd=max_iter_hyd, tol_p=1e-7, tol_m=1e-7, friction_model="nikuradse",
                        use_numba=use_numba)

    active_branches = get_lookup(net, "branch", "active_hydraulics")
    active_nodes = get_lookup(net, "node", "active_hydraulics")

    assert np.all(active_nodes == np.array([True, True, True, True, True, True, True, False,
                                            False, True, False, False, True, True, True]))
    assert np.all(active_branches)

    assert not np.all(np.isnan(net.res_junction.p_bar.loc[[0, 1, 2, 3, 4, 5, 9]].values))
    assert not np.all(np.isnan(net.res_pipe.values))
    assert np.all(np.isnan(net.res_junction.p_bar.loc[[7, 8, 10, 11]].values))

    with pytest.raises(PipeflowNotConverged):
        pandapipes.pipeflow(net, max_iter_hyd=max_iter_hyd, tol_p=1e-7, tol_m=1e-7, friction_model="nikuradse",
                            use_numba=use_numba, check_connectivity=False)


@pytest.mark.parametrize("use_numba", [True, False])
def test_connectivity_heat1(complex_heat_connectivity_grid, use_numba):
    net = copy.deepcopy(complex_heat_connectivity_grid)

    max_iter_hyd = 3 if use_numba else 3
    max_iter_therm = 3 if use_numba else 3
    pandapipes.pipeflow(net, max_iter_hyd=max_iter_hyd, max_iter_therm=max_iter_therm,
                        mode='sequential', check_connectivity=True, use_numba=use_numba)

    oos_juncs_hyd = {4, 5, 6, 7}
    oos_pipe_hyd = {5, 7, 8, 9}
    oos_sink_hyd = {4, 5}
    oos_source_hyd = {7}

    assert set(net.res_junction.loc[net.res_junction.p_bar.notnull()].index) == \
           set(net.junction.index) - oos_juncs_hyd
    assert set(net.res_junction.loc[net.res_junction.p_bar.isnull()].index) == oos_juncs_hyd

    assert set(net.res_pipe.loc[net.res_pipe.v_mean_m_per_s.notnull()].index) == \
           set(net.pipe.index) - oos_pipe_hyd
    assert set(net.res_pipe.loc[net.res_pipe.v_mean_m_per_s.isnull()].index) == oos_pipe_hyd

    assert set(net.res_valve.loc[net.res_valve.v_mean_m_per_s.notnull()].index) == set()
    assert set(net.res_valve.loc[net.res_valve.v_mean_m_per_s.isnull()].index) \
           == set(net.valve.index)

    assert set(net.res_sink.loc[net.res_sink.mdot_kg_per_s.isnull()].index) == oos_sink_hyd
    assert set(net.res_sink.loc[net.res_sink.mdot_kg_per_s.notnull()].index) == \
           set(net.sink.index) - oos_sink_hyd

    assert set(net.res_source.loc[net.res_source.mdot_kg_per_s.isnull()].index) == oos_source_hyd
    assert (set(net.res_source.loc[net.res_source.mdot_kg_per_s.notnull()].index) ==
            set(net.source.index) - oos_source_hyd)

    assert np.allclose(net.res_ext_grid.mdot_kg_per_s.sum(),
                       -net.res_sink.mdot_kg_per_s.sum() + net.res_source.mdot_kg_per_s.sum(),
                       rtol=1e-10, atol=0)


@pytest.mark.parametrize("use_numba", [True, False])
def test_connectivity_heat2(complex_heat_connectivity_grid, use_numba):
    net = copy.deepcopy(complex_heat_connectivity_grid)
    net.pipe.at[7, "in_service"] = True

    max_iter_hyd = 3 if use_numba else 3
    max_iter_therm = 5 if use_numba else 5
    pandapipes.pipeflow(net, max_iter_hyd=max_iter_hyd, max_iter_therm=max_iter_therm,
                        mode='sequential', check_connectivity=True, use_numba=use_numba)

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


@pytest.mark.parametrize("use_numba", [True, False])
def test_connectivity_heat3(complex_heat_connectivity_grid, use_numba):
    net = copy.deepcopy(complex_heat_connectivity_grid)
    net.pipe.at[5, "in_service"] = True
    net.valve.at[12, "opened"] = True
    net.junction.at[5, "in_service"] = False
    max_iter_hyd = 3 if use_numba else 3
    max_iter_therm = 4 if use_numba else 4
    pandapipes.pipeflow(net, max_iter_hyd=max_iter_hyd, max_iter_therm=max_iter_therm,
                        mode='sequential', check_connectivity=True, use_numba=use_numba)

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


@pytest.mark.parametrize("use_numba", [True, False])
def test_connectivity_heat4(complex_heat_connectivity_grid, use_numba):
    net = copy.deepcopy(complex_heat_connectivity_grid)

    net.pipe.loc[[7, 8], 'in_service'] = True
    j_new = pandapipes.create_junction(net, 1, 320.15)
    pandapipes.create_pipe_from_parameters(net, 8, j_new, 0.1, 0.1, u_w_per_m2k=5)

    net2 = copy.deepcopy(net)

    max_iter_hyd = 10 if use_numba else 10
    max_iter_therm = 6 if use_numba else 6
    pandapipes.pipeflow(net, max_iter_hyd=max_iter_hyd, max_iter_therm=max_iter_therm,
                        mode='sequential', check_connectivity=True, use_numba=use_numba)
    pandapipes.pipeflow(net2, max_iter_hyd=max_iter_hyd, max_iter_therm=max_iter_therm,
                        mode='sequential', check_connectivity=False, use_numba=use_numba)

    assert pandapipes.nets_equal(net, net2, check_only_results=True)


@pytest.mark.parametrize("use_numba", [True, False])
def test_connectivity_heat5(complex_heat_connectivity_grid, use_numba):
    net = copy.deepcopy(complex_heat_connectivity_grid)
    net.pipe.loc[[7, 8], 'in_service'] = True

    j_from, j_to = pandapipes.create_junctions(net, 2, 1, 320.15)

    pandapipes.create_pipe_from_parameters(net, j_from, j_to, 0.1, 0.1, u_w_per_m2k=5)
    pandapipes.create_sink(net, j_to, 0.1)
    pandapipes.create_ext_grid(net, j_from, 1, 320.15)

    net.ext_grid.loc[2, 'in_service'] = False
    net.ext_grid.loc[1, 'type'] = 'p'

    max_iter_hyd = 10 if use_numba else 10
    max_iter_therm = 3 if use_numba else 3
    pandapipes.pipeflow(net, max_iter_hyd=max_iter_hyd, max_iter_therm=max_iter_therm,
                        check_connectivity=True, mode='sequential', use_numba=use_numba)

    with pytest.raises(PipeflowNotConverged):
        pandapipes.pipeflow(net, max_iter_hyd=max_iter_hyd, max_iter_therm=max_iter_therm,
                            check_connectivity=False, mode='sequential', use_numba=use_numba)


@pytest.mark.parametrize("use_numba", [True, False])
def test_exclude_unconnected_junction(use_numba):
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
    max_iter_hyd = 4 if use_numba else 4
    pandapipes.pipeflow(net, max_iter_hyd=max_iter_hyd, use_numba=use_numba)
    assert net.converged


special_tables = {"valve": "opened"}


def get_oos(net, tbl, oos_ind=()):
    is_name = special_tables.get(tbl, "in_service")
    return ~net[tbl][is_name].values | net[tbl].index.isin(oos_ind)


def get_oos_node_elem(net, tbl, oosj=()):
    return get_oos(net, tbl) | net[tbl].junction.isin(oosj)


def get_oos_branch(net, tbl, oosj=()):
    return get_oos(net, tbl) | net[tbl].from_junction.isin(oosj) | net[tbl].to_junction.isin(oosj)


def get_col_slice_null(tbl):
    if tbl == "junction":
        return "p_bar"
    return slice(None)


all_tbls_funcs = {"junction": get_oos, "pipe": get_oos_branch, "sink": get_oos_node_elem,
                  "source": get_oos_node_elem, "ext_grid": get_oos_node_elem,
                  "press_control": get_oos_branch}


def check_mass_flows(net):
    return np.isclose(net.res_ext_grid.mdot_kg_per_s.sum(),
                      net.res_source.loc[~get_oos(net, "source")].mdot_kg_per_s.sum() -
                      net.res_sink.loc[~get_oos(net, "sink")].mdot_kg_per_s.sum())


@pytest.mark.parametrize("use_numba", [True, False])
def test_mixed_indexing_oos1(create_mixed_indexing_grid, use_numba):
    net = copy.deepcopy(create_mixed_indexing_grid)

    max_iter_hyd = 3 if use_numba else 3
    pandapipes.pipeflow(net, mode="hydraulics", max_iter_hyd=max_iter_hyd, use_numba=use_numba)
    assert all(np.all(net["res_" + tbl].loc[~oos_func(net, tbl)].notnull()) for tbl, oos_func
               in all_tbls_funcs.items())
    assert all(np.all(net["res_" + tbl].loc[oos_func(net, tbl)].isnull()) for tbl, oos_func
               in all_tbls_funcs.items())
    assert check_mass_flows(net)

    max_iter_hyd = 4 if use_numba else 4
    net.pipe.at[3, "in_service"] = False
    pandapipes.pipeflow(net, mode="hydraulics", max_iter_hyd=max_iter_hyd, use_numba=use_numba)
    assert all(np.all(net["res_" + tbl].loc[~oos_func(net, tbl)].notnull()) for tbl, oos_func
               in all_tbls_funcs.items())
    assert all(np.all(net["res_" + tbl].loc[oos_func(net, tbl)].isnull()) for tbl, oos_func
               in all_tbls_funcs.items())
    assert check_mass_flows(net)

    max_iter_hyd = 4 if use_numba else 4
    pandapipes.pipeflow(net, mode="hydraulics", max_iter_hyd=max_iter_hyd,
                        use_numba=use_numba, check_connectivity=False)
    assert all(np.all(net["res_" + tbl].loc[~oos_func(net, tbl)].notnull()) for tbl, oos_func
               in all_tbls_funcs.items())
    assert all(np.all(net["res_" + tbl].loc[oos_func(net, tbl)].isnull()) for tbl, oos_func
               in all_tbls_funcs.items())
    assert check_mass_flows(net)


@pytest.mark.parametrize("use_numba", [True, False])
def test_mixed_indexing_oos2(create_mixed_indexing_grid, use_numba):
    net = copy.deepcopy(create_mixed_indexing_grid)
    net.pipe.at[10, "in_service"] = False
    oos_juncs = [10, 12]

    max_iter_hyd = 3 if use_numba else 3
    with pytest.raises(PipeflowNotConverged):
        pandapipes.pipeflow(net, mode="hydraulics",
                            max_iter_hyd=max_iter_hyd,
                            use_numba=use_numba, check_connectivity=False)

    pandapipes.pipeflow(net, mode="hydraulics", max_iter_hyd=max_iter_hyd,
                        use_numba=use_numba, check_connectivity=True)
    assert all(np.all(net["res_" + tbl].loc[~oos_func(net, tbl, oos_juncs)].notnull())
               for tbl, oos_func in all_tbls_funcs.items())
    assert all(np.all(net["res_" + tbl].loc[oos_func(net, tbl, oos_juncs),
    get_col_slice_null(tbl)].isnull())
               for tbl, oos_func in all_tbls_funcs.items())
    assert check_mass_flows(net)


@pytest.mark.parametrize("use_numba", [True, False])
def test_mixed_indexing_oos3(create_mixed_indexing_grid, use_numba):
    net = copy.deepcopy(create_mixed_indexing_grid)
    net.pipe.at[7, "in_service"] = False
    oos_juncs = [6, 15]

    max_iter_hyd = 3 if use_numba else 3
    with pytest.raises(PipeflowNotConverged):
        pandapipes.pipeflow(net, max_iter_hyd=max_iter_hyd,
                            mode="hydraulics", use_numba=use_numba, check_connectivity=False)

    pandapipes.pipeflow(net, mode="hydraulics", max_iter_hyd=max_iter_hyd,
                        use_numba=use_numba, check_connectivity=True)
    assert all(np.all(net["res_" + tbl].loc[~oos_func(net, tbl, oos_juncs)].notnull())
               for tbl, oos_func in all_tbls_funcs.items())
    assert all(np.all(net["res_" + tbl].loc[oos_func(net, tbl, oos_juncs),
    get_col_slice_null(tbl)].isnull())
               for tbl, oos_func in all_tbls_funcs.items())
    assert check_mass_flows(net)


@pytest.mark.parametrize("use_numba", [True, False])
def test_mixed_indexing_oos4(create_mixed_indexing_grid, use_numba):
    net = copy.deepcopy(create_mixed_indexing_grid)
    net.valve.at[2, "opened"] = False
    oos_juncs = [15]

    max_iter_hyd = 3 if use_numba else 3
    with pytest.raises(PipeflowNotConverged):
        pandapipes.pipeflow(net, max_iter_hyd=max_iter_hyd,
                            mode="hydraulics", use_numba=use_numba, check_connectivity=False)

    pandapipes.pipeflow(net, mode="hydraulics", max_iter_hyd=max_iter_hyd,
                        use_numba=use_numba, check_connectivity=True)
    assert all(np.all(net["res_" + tbl].loc[~oos_func(net, tbl, oos_juncs)].notnull())
               for tbl, oos_func in all_tbls_funcs.items())
    assert all(np.all(net["res_" + tbl].loc[oos_func(net, tbl, oos_juncs),
    get_col_slice_null(tbl)].isnull())
               for tbl, oos_func in all_tbls_funcs.items())
    assert check_mass_flows(net)


@pytest.mark.parametrize("use_numba", [True, False])
def test_mixed_indexing_oos5(create_mixed_indexing_grid, use_numba):
    net = copy.deepcopy(create_mixed_indexing_grid)
    net.pipe.at[6, "in_service"] = False
    oos_juncs = [9, 8, 7]

    max_iter_hyd = 3 if use_numba else 3
    with pytest.raises(PipeflowNotConverged):
        pandapipes.pipeflow(net, max_iter_hyd=max_iter_hyd,
                            mode="hydraulics", use_numba=use_numba, check_connectivity=False)

    pandapipes.pipeflow(net, mode="hydraulics", max_iter_hyd=max_iter_hyd,
                        use_numba=use_numba, check_connectivity=True)
    assert all(np.all(net["res_" + tbl].loc[~oos_func(net, tbl, oos_juncs)].notnull())
               for tbl, oos_func in all_tbls_funcs.items())
    assert all(np.all(net["res_" + tbl].loc[oos_func(net, tbl, oos_juncs),
    get_col_slice_null(tbl)].isnull())
               for tbl, oos_func in all_tbls_funcs.items())
    assert check_mass_flows(net)


@pytest.mark.parametrize("use_numba", [True, False])
def test_mixed_indexing_oos6(create_mixed_indexing_grid, use_numba):
    net = copy.deepcopy(create_mixed_indexing_grid)
    net.valve.at[3, "opened"] = False
    oos_juncs = [14, 6, 15, 9, 8, 7]

    max_iter_hyd = 3 if use_numba else 3
    with pytest.raises(PipeflowNotConverged):
        pandapipes.pipeflow(net, max_iter_hyd=max_iter_hyd,
                            mode="hydraulics", use_numba=use_numba, check_connectivity=False)

    pandapipes.pipeflow(net, mode="hydraulics", max_iter_hyd=max_iter_hyd,
                        use_numba=use_numba, check_connectivity=True)
    assert all(np.all(net["res_" + tbl].loc[~oos_func(net, tbl, oos_juncs)].notnull())
               for tbl, oos_func in all_tbls_funcs.items())
    assert all(np.all(net["res_" + tbl].loc[oos_func(net, tbl, oos_juncs),
    get_col_slice_null(tbl)].isnull())
               for tbl, oos_func in all_tbls_funcs.items())
    assert check_mass_flows(net)

    net.ext_grid.at[1, "in_service"] = False
    pandapipes.pipeflow(net, mode="hydraulics", use_numba=use_numba,
                        max_iter_hyd=max_iter_hyd, check_connectivity=True)
    assert all(np.all(net["res_" + tbl].loc[~oos_func(net, tbl, oos_juncs)].notnull())
               for tbl, oos_func in all_tbls_funcs.items())
    assert all(np.all(net["res_" + tbl].loc[oos_func(net, tbl, oos_juncs),
    get_col_slice_null(tbl)].isnull())
               for tbl, oos_func in all_tbls_funcs.items())
    assert check_mass_flows(net)


@pytest.mark.parametrize("use_numba", [True, False])
def test_pipeflow_all_oos(create_net_wo_ext_grid, use_numba):
    net = create_net_wo_ext_grid
    ex1 = pandapipes.create_ext_grid(net, junction=3, t_k=300)
    ex2 = pandapipes.create_ext_grid(net, junction=3, p_bar=1)
    max_iter_hyd = 9 if use_numba else 9
    with pytest.raises(PipeflowNotConverged):
        net.ext_grid.at[ex2, 'in_service'] = False
        pandapipes.pipeflow(net, max_iter_hyd=max_iter_hyd, tol_p=1e-7, tol_m=1e-7, friction_model="nikuradse",
                            use_numba=use_numba, check_connectivity=True)
    assert ~net.converged
    net.ext_grid.at[ex1, 'in_service'] = False
    net.ext_grid.at[ex2, 'in_service'] = True

    pandapipes.pipeflow(net, max_iter_hyd=max_iter_hyd, tol_p=1e-7, tol_m=1e-7, friction_model="nikuradse",
                        use_numba=use_numba, check_connectivity=True)
    assert not np.all(np.isnan(net.res_junction.p_bar.values))
    assert net.converged

    with pytest.raises(PipeflowNotConverged):
        pandapipes.pipeflow(net, mode='sequential', max_iter_hyd=max_iter_hyd,
                            tol_p=1e-7, tol_m=1e-7, friction_model="nikuradse",
                            use_numba=use_numba, check_connectivity=True)
    assert ~net.converged


if __name__ == "__main__":
    pytest.main([r'pandapipes/test/pipeflow_internals/test_inservice.py'])
