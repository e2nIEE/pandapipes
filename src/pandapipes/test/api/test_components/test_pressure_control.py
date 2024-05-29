# Copyright (c) 2020-2024 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import pandas as pd
import pytest

import pandapipes
from pandapipes.test import data_path

@pytest.mark.parametrize("use_numba", [True, False])
def test_pressure_control_from_measurement_parameters(use_numba):
    """
        :return:
        :rtype:
        """
    net = pandapipes.create_empty_network("net", add_stdtypes=False)

    j1 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j2 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j3 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j4 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)

    pandapipes.create_pipe_from_parameters(net, j2, j3, k_mm=1., length_km=5.,
                                           diameter_m=0.1022)
    pandapipes.create_pipe_from_parameters(net, j3, j4, k_mm=1., length_km=10.,
                                           diameter_m=0.1022)
    pandapipes.create_pressure_control(net, j1, j2, j4, 20.)
    pandapipes.create_ext_grid(net, j1, 32, 283.15, type="p")
    pandapipes.create_sink(net, j4, 0.5)

    pandapipes.create_fluid_from_lib(net, "lgas", overwrite=True)

    max_iter_hyd = 4 if use_numba else 4
    pandapipes.pipeflow(net, stop_condition="tol", max_iter_hyd=max_iter_hyd, friction_model="nikuradse",
                        mode="hydraulics", transient=False, nonlinear_method="automatic",
                        tol_p=1e-4, tol_m=1e-4, use_numba=use_numba)

    data = pd.read_csv(os.path.join(data_path, "test_pressure_control.csv"), sep=';')

    res_junction = net.res_junction.p_bar.values
    res_pipe = net.res_pipe.v_mean_m_per_s.values

    p_diff = np.abs(1 - res_junction / data['p'].dropna().values)
    v_diff = np.abs(1 - res_pipe / data['v'].dropna().values)

    assert np.all(p_diff < 0.01)
    assert np.all(v_diff < 0.01)


def test_2pressure_controller_controllability():
    net = pandapipes.create_empty_network("net", add_stdtypes=False)

    j1 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j2 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j3 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j4 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j5 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)

    pandapipes.create_pipe_from_parameters(net, j2, j3, k_mm=1., length_km=5.,
                                           diameter_m=0.1022)
    pandapipes.create_pipe_from_parameters(net, j3, j4, k_mm=1., length_km=10.,
                                           diameter_m=0.1022)
    pandapipes.create_pressure_control(net, j4, j5, j5, 20.)
    pandapipes.create_pressure_control(net, j1, j2, j5, 20.)

    assert len(net.press_control == 1)


def test_pctrl_line():
    net = pandapipes.create_empty_network(fluid='hgas')
    j1, j2 = pandapipes.create_junctions(net, 2, 3, 300)
    pandapipes.create_ext_grid(net, j1, 2)
    pandapipes.create_sink(net, j2, 4)
    pandapipes.create_pressure_control(net, j1, j2, j2, 3, loss_coefficient=0.5)
    pandapipes.pipeflow(net, use_numba=False)
    assert np.isclose(net.res_junction.at[j2, 'p_bar'], 1.6556170)


def test_pctrl_reverse():
    net = pandapipes.create_empty_network(fluid='hgas')
    j1, j2 = pandapipes.create_junctions(net, 2, 3, 300)
    pandapipes.create_ext_grid(net, j1, 5)
    pandapipes.create_source(net, j2, 4)
    pandapipes.create_pressure_control(net, j1, j2, j2, 3, loss_coefficient=0.5)
    pandapipes.pipeflow(net, use_numba=False)
    assert np.isnan(net.res_junction.at[j2, 'p_bar'])


def test_pctrl_default():
    net = pandapipes.create_empty_network(fluid='hgas')
    j1, j2 = pandapipes.create_junctions(net, 2, 3, 300)
    pandapipes.create_ext_grid(net, j1, 5)
    pandapipes.create_sink(net, j2, 4)
    pandapipes.create_pressure_control(net, j1, j2, j2, 3, loss_coefficient=0.5)
    pandapipes.pipeflow(net, use_numba=False)
    assert net.res_junction.at[j2, 'p_bar'] == 3

def test_pctrl_max_mdot():
    net = pandapipes.create_empty_network(fluid='water')
    j1, j2, j3, j4 = pandapipes.create_junctions(net, 4, 3, 300)
    pandapipes.create_circ_pump_const_pressure(net, j4, j1, 10, 8)
    pandapipes.create_pipes_from_parameters(net, [j1, j2, j3], [j2, j4,  j4], 0.1, 0.1)
    pandapipes.create_pressure_control(net, j2, j3, j3, 3, loss_coefficient=0.5, max_mdot_kg_per_s=10)
    pandapipes.pipeflow(net, use_numba=False)

    assert net.res_press_control.mdot_from_kg_per_s.values == 10
