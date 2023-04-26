# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import pandas as pd
import pytest

import pandapipes
from pandapipes.test.pipeflow_internals import internals_data_path


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
    pandapipes.create_ext_grid(net, j1, 5, 283.15, type="p")
    pandapipes.create_sink(net, j4, 0.5)

    pandapipes.create_fluid_from_lib(net, "lgas", overwrite=True)

    pandapipes.pipeflow(net, stop_condition="tol", iter=3, friction_model="nikuradse",
                        mode="hydraulics", transient=False, nonlinear_method="automatic",
                        tol_p=1e-4, tol_v=1e-4, use_numba=use_numba)

    data = pd.read_csv(os.path.join(internals_data_path, "test_pressure_control.csv"), sep=';')

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
