# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
import os

import numpy as np
import pandas as pd
import pytest

import pandapipes
from pandapipes.test import data_path


@pytest.mark.parametrize("use_numba", [True, False])
def test_circulation_pump_constant_pressure(use_numba):
    """
        :return:
        :rtype:
        """
    net = pandapipes.create_empty_network("net", add_stdtypes=False)

    j1 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j2 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j3 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j4 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)

    pandapipes.create_pipe_from_parameters(net, j1, j2, k_mm=1., length_km=0.43380,
                                           diameter_m=0.1022)
    pandapipes.create_pipe_from_parameters(net, j3, j4, k_mm=1., length_km=0.26370,
                                           diameter_m=0.1022)
    pandapipes.create_circ_pump_const_pressure(net, j4, j1, 5, 2, 300, type='pt')
    pandapipes.create_heat_exchanger(net, j2, j3, 0.1, qext_w=200000)
    pandapipes.create_sink(net, j1, 2)
    pandapipes.create_source(net, j4, 2)

    pandapipes.create_fluid_from_lib(net, "water", overwrite=True)

    max_iter_hyd = 8 if use_numba else 8
    max_iter_therm = 4 if use_numba else 4
    pandapipes.pipeflow(net, max_iter_hyd=max_iter_hyd, max_iter_therm=max_iter_therm,
                        stop_condition="tol", friction_model="nikuradse",
                        mode="all", transient=False, nonlinear_method="automatic",
                        tol_p=1e-4, tol_m=1e-4, use_numba=use_numba)

    data = pd.read_csv(os.path.join(data_path, "test_circ_pump_pressure.csv"), sep=';')

    res_junction = net.res_junction
    res_pipe = net.res_pipe.v_mean_m_per_s.values
    res_pump = net.res_circ_pump_pressure

    p_diff = np.abs(1 - res_junction.p_bar.values / data['p'].dropna().values)
    t_diff = np.abs(1 - res_junction.t_k.values / data['t'].dropna().values)
    v_diff = np.abs(1 - res_pipe / data['v'].dropna().values)
    mdot_diff = np.abs(1 - res_pump['mdot_flow_kg_per_s'].values / data['mdot'].dropna().values)
    deltap_diff = np.abs(1 - res_pump['deltap_bar'].values / data['deltap'].dropna().values)

    assert np.all(p_diff < 0.01)
    assert np.all(t_diff < 0.01)
    assert np.all(v_diff < 0.01)
    assert np.all(mdot_diff < 0.01)
    assert np.all(deltap_diff < 0.01)

@pytest.mark.parametrize("use_numba", [True, False])
def test_circ_pump_pressure_return_flow(use_numba):
    net = pandapipes.create_empty_network(fluid="water")

    j0 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=293.15, name="junction 0")
    j1 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=293.15, name="junction 1")
    j2 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=293.15, name="junction 2")
    j3 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=293.15, name="junction 3")

    pandapipes.create_circ_pump_const_pressure(net, return_junction=j3, flow_junction=j0, p_setpoint_bar=7.5,
                                       plift_bar=5, setpoint='flow', t_flow_k=273.15 + 35)

    pandapipes.create_heat_exchanger(net, from_junction=j1, to_junction=j2, diameter_m=200e-3, qext_w=100000)

    pandapipes.create_pipe_from_parameters(net, from_junction=j0, to_junction=j1, length_km=1,
                                   diameter_m=200e-3, k_mm=.1, alpha_w_per_m2k=10, sections=5, text_k=283)
    pandapipes.create_pipe_from_parameters(net, from_junction=j2, to_junction=j3, length_km=1,
                                   diameter_m=200e-3, k_mm=.1, alpha_w_per_m2k=10, sections=5, text_k=283)
    max_iter_hyd = 10 if use_numba else 10
    max_iter_therm = 4 if use_numba else 4
    pandapipes.pipeflow(net, max_iter_hyd=max_iter_hyd, max_iter_therm=max_iter_therm,
                        stop_condition="tol", friction_model="nikuradse",
                        mode="all", transient=False, nonlinear_method="automatic",
                        tol_p=1e-4, tol_m=1e-4, use_numba=use_numba)
    data = pd.read_csv(os.path.join(data_path, "test_circ_pump_pressure_return_flow.csv"), sep=';')

    res_junction = net.res_junction
    res_pipe = net.res_pipe.v_mean_m_per_s.values
    res_pump = net.res_circ_pump_pressure

    p_diff = np.abs(1 - res_junction.p_bar.values / data['p'].dropna().values)
    t_diff = np.abs(1 - res_junction.t_k.values / data['t'].dropna().values)
    v_diff = np.abs(1 - res_pipe / data['v'].dropna().values)
    mdot_diff = np.abs(1 - res_pump['mdot_flow_kg_per_s'].values / data['mdot'].dropna().values)
    deltap_diff = np.abs(1 - res_pump['deltap_bar'].values / data['deltap'].dropna().values)

    assert np.all(p_diff < 0.01)
    assert np.all(t_diff < 0.01)
    assert np.all(v_diff < 0.01)
    assert np.all(mdot_diff < 0.01)
    assert np.all(deltap_diff < 0.01)

@pytest.mark.parametrize("use_numba", [True, False])
def test_circ_pump_pressure_return_and_flow_setpoint(use_numba):
    """
    :param use_numba:
    :type use_numba:
    :return:
    :rtype:
    """
    net = pandapipes.create_empty_network(fluid="water")
    #first closed loop
    j0 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=293.15, name="junction 0")
    j1 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=293.15, name="junction 1")
    j2 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=293.15, name="junction 2")
    j3 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=293.15, name="junction 3")

    pandapipes.create_circ_pump_const_pressure(net, return_junction=j3, flow_junction=j0, p_setpoint_bar=7.5,
                                       plift_bar=5, setpoint='flow', t_flow_k=273.15 + 35)


    pandapipes.create_heat_exchanger(net, from_junction=j1, to_junction=j2, diameter_m=200e-3, qext_w=100000)

    pandapipes.create_pipe_from_parameters(net, from_junction=j0, to_junction=j1, length_km=1,
                                   diameter_m=200e-3, k_mm=.1, alpha_w_per_m2k=10, sections=5, text_k=283)
    pandapipes.create_pipe_from_parameters(net, from_junction=j2, to_junction=j3, length_km=1,
                                   diameter_m=200e-3, k_mm=.1, alpha_w_per_m2k=10, sections=5, text_k=283)
    # second closed loop
    j4 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=293.15, name="junction 0")
    j5 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=293.15, name="junction 1")
    j6 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=293.15, name="junction 2")
    j7 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=293.15, name="junction 3")

    pandapipes.create_circ_pump_const_pressure(net, return_junction=j7, flow_junction=j4, p_setpoint_bar=5,
                                       plift_bar=5, setpoint='return', t_flow_k=273.15 + 35)


    pandapipes.create_heat_exchanger(net, from_junction=j5, to_junction=j6, diameter_m=200e-3, qext_w=100000)

    pandapipes.create_pipe_from_parameters(net, from_junction=j4, to_junction=j5, length_km=1,
                                   diameter_m=200e-3, k_mm=.1, alpha_w_per_m2k=10, sections=5, text_k=283)
    pandapipes.create_pipe_from_parameters(net, from_junction=j6, to_junction=j7, length_km=1,
                                   diameter_m=200e-3, k_mm=.1, alpha_w_per_m2k=10, sections=5, text_k=283)

    max_iter_hyd = 10 if use_numba else 10
    max_iter_therm = 4 if use_numba else 4
    pandapipes.pipeflow(net, max_iter_hyd=max_iter_hyd, max_iter_therm=max_iter_therm,
                        stop_condition="tol", friction_model="nikuradse",
                        mode="all", transient=False, nonlinear_method="automatic",
                        tol_p=1e-4, tol_m=1e-4, use_numba=use_numba)

    data = pd.read_csv(os.path.join(data_path, "test_circ_pump_pressure_return_and_flow_setpoint.csv"), sep=';')

    res_junction = net.res_junction
    res_pipe = net.res_pipe.v_mean_m_per_s.values
    res_pump = net.res_circ_pump_pressure

    p_diff = np.abs(1 - res_junction.p_bar.values / data['p'].dropna().values)
    t_diff = np.abs(1 - res_junction.t_k.values / data['t'].dropna().values)
    v_diff = np.abs(1 - res_pipe / data['v'].dropna().values)
    mdot_diff = np.abs(1 - res_pump['mdot_flow_kg_per_s'].values / data['mdot'].dropna().values)
    deltap_diff = np.abs(1 - res_pump['deltap_bar'].values / data['deltap'].dropna().values)

    assert np.all(p_diff < 0.01)
    assert np.all(t_diff < 0.01)
    assert np.all(v_diff < 0.01)
    assert np.all(mdot_diff < 0.01)
    assert np.all(deltap_diff < 0.01)