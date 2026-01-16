# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
import os

import numpy as np
import pandas as pd
import pytest

import pandapipes
from pandapipes.test import data_path

@pytest.mark.parametrize("use_numba", [True, False])
def test_valve(use_numba):
    """

        :return:
        :rtype:
        """
    net = pandapipes.create_empty_network("net", add_stdtypes=True)

    j0 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15, index=5)
    j1 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15, index=3)
    j2 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15, index=6)
    j3 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15, index=9)
    j4 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15, index=20)
    j5 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15, index=45)
    j6 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15, index=4)
    j7 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15, index=8)

    pandapipes.create_ext_grid(net, j0, 5, 283.15, type="p")

    pandapipes.create_pipe_from_parameters(net, j0, j1, diameter_m=.1, k_mm=1, length_km=1.)
    pandapipes.create_pipe_from_parameters(net, j3, j4, diameter_m=.1, k_mm=1, length_km=.5)
    pandapipes.create_pipe_from_parameters(net, j2, j4, diameter_m=.1, k_mm=1, length_km=.5)
    pandapipes.create_pipe_from_parameters(net, j5, j4, diameter_m=.1, k_mm=1, length_km=.35)
    pandapipes.create_pipe_from_parameters(net, j1, j6, diameter_m=.1, k_mm=1, length_km=.1,
                                           loss_coefficient=9000)
    pandapipes.create_pipe_from_parameters(net, j1, j7, diameter_m=.1, k_mm=1, length_km=.1,
                                           loss_coefficient=9000)

    pandapipes.create_valve(net, j6, j2, et='ju', diameter_m=0.1, opened=False)
    pandapipes.create_valve(net, j7, j3, et='ju', diameter_m=0.1, opened=True)

    pandapipes.create_sink(net, j5, 0.11667)

    pandapipes.create_fluid_from_lib(net, "lgas", overwrite=True)

    max_iter_hyd = 8 if use_numba else 7
    pandapipes.pipeflow(net, stop_condition="tol", max_iter_hyd=max_iter_hyd, friction_model="nikuradse",
                        mode="hydraulics", transient=False, nonlinear_method="automatic",
                        tol_p=1e-8, tol_m=1e-8, use_numba=use_numba)

    data = pd.read_csv(os.path.join(data_path, "test_valve.csv"), sep=';')
    data_p = data['p'].dropna(inplace=False)
    data_v = data['v'].dropna(inplace=False)

    res_junction = net.res_junction.p_bar.values
    res_pipe = net.res_pipe.v_mean_m_per_s.values
    zeros = np.isclose(res_pipe, 0, atol=1e-12)
    test_zeros = np.isclose(data_v.values, 0, atol=1e-12)
    check_zeros = zeros == test_zeros

    assert np.all(check_zeros)

    p_diff = np.abs(1 - res_junction / data_p[data_p != 0].values)
    v_diff = np.abs(1 - res_pipe[~zeros] / data_v[~test_zeros].values)

    assert np.all(p_diff < 0.01)
    assert np.all(v_diff < 0.01)


@pytest.mark.parametrize("use_numba", [True, False])
def test_valve_to_pipe(use_numba):
    """

        :return:
        :rtype:
        """
    net = pandapipes.create_empty_network("net", add_stdtypes=True)

    j0 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15, index=7)
    j1 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15, index=20)
    j2 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15, index=5)
    j3 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15, index=3)

    pandapipes.create_ext_grid(net, j0, 5, 283.15, type="p", index=12)

    p0 = pandapipes.create_pipe_from_parameters(net, j0, j1, diameter_m=.1, k_mm=1, length_km=1., index=10)
    p1 = pandapipes.create_pipe_from_parameters(net, j2, j1, diameter_m=.1, k_mm=1, length_km=1., sections=5, index=3)

    pandapipes.create_valve(net, j1, p0, et='pi', diameter_m=0.1, opened=True, loss_coefficient=1000, index=5)
    pandapipes.create_valve(net, j2, p1, et='pi', diameter_m=0.1, opened=False, loss_coefficient=2000, index=6)

    pandapipes.create_valve(net, j2, p1, et='pi', diameter_m=0.1, opened=True, loss_coefficient=500, index=9)
    pandapipes.create_valve(net, j1, p0, et='pi', diameter_m=0.1, opened=True, loss_coefficient=2000, index=3)

    p2 = pandapipes.create_pipe_from_parameters(net, j3, j2, diameter_m=.1, k_mm=1, length_km=1., sections=10, index=5)

    pandapipes.create_valve(net, j3, p2, et='pi', diameter_m=0.1, opened=True, loss_coefficient=5000, index=11)

    pandapipes.create_sink(net, j3, 0.11667)

    pandapipes.create_fluid_from_lib(net, "lgas", overwrite=True)

    max_iter_hyd = 100 if use_numba else 100
    pandapipes.pipeflow(net, stop_condition="tol", max_iter_hyd=max_iter_hyd, friction_model="nikuradse",
                        mode="hydraulics", transient=False, nonlinear_method="automatic",
                        tol_p=1e-8, tol_m=1e-8, use_numba=use_numba)

    net2 = pandapipes.create_empty_network("net2", add_stdtypes=True)

    j0_2 = pandapipes.create_junction(net2, pn_bar=5, tfluid_k=283.15, index=5)
    j1_2 = pandapipes.create_junction(net2, pn_bar=5, tfluid_k=283.15, index=3)
    j2_2 = pandapipes.create_junction(net2, pn_bar=5, tfluid_k=283.15, index=12)
    j3_2 = pandapipes.create_junction(net2, pn_bar=5, tfluid_k=283.15, index=7)
    j4_2 = pandapipes.create_junction(net2, pn_bar=5, tfluid_k=283.15, index=1)
    j5_2 = pandapipes.create_junction(net2, pn_bar=5, tfluid_k=283.15, index=13)
    j6_2 = pandapipes.create_junction(net2, pn_bar=5, tfluid_k=283.15, index=6)

    pandapipes.create_ext_grid(net2, j0_2, 5, 283.15, type="p")

    pandapipes.create_pipe_from_parameters(net2, j0_2, j1_2, diameter_m=.1, k_mm=1, length_km=1., index=10)
    pandapipes.create_pipe_from_parameters(net2, j3_2, j2_2, diameter_m=.1, k_mm=1, length_km=1., sections=5, index=3)

    pandapipes.create_valve(net2, j2_2, j1_2, et='ju', diameter_m=0.1, opened=True, loss_coefficient=1000)
    pandapipes.create_valve(net2, j4_2, j3_2, et='ju', diameter_m=0.1, opened=False, loss_coefficient=2000)

    pandapipes.create_valve(net2, j4_2, j3_2, et='ju', diameter_m=0.1, opened=True, loss_coefficient=500)
    pandapipes.create_valve(net2, j2_2, j1_2, et='ju', diameter_m=0.1, opened=True, loss_coefficient=2000)

    pandapipes.create_pipe_from_parameters(net2, j5_2, j4_2, diameter_m=.1, k_mm=1, length_km=1., sections=10, index=5)

    pandapipes.create_valve(net2, j6_2, j5_2, et='ju', diameter_m=0.1, opened=True, loss_coefficient=5000)

    pandapipes.create_sink(net2, j6_2, 0.11667)

    pandapipes.create_fluid_from_lib(net2, "lgas", overwrite=True)

    max_iter_hyd = 100 if use_numba else 100
    pandapipes.pipeflow(net2, stop_condition="tol", max_iter_hyd=max_iter_hyd, friction_model="nikuradse",
                        mode="hydraulics", transient=False, nonlinear_method="automatic",
                        tol_p=1e-8, tol_m=1e-8, use_numba=use_numba)

    assert np.all(np.isclose(net.res_pipe.values, net2.res_pipe.values, equal_nan=True))
    assert np.all(np.isclose(net.res_junction.loc[j0], net2.res_junction.loc[j0_2], equal_nan=True))
    assert np.all(np.isclose(net.res_junction.loc[j1], net2.res_junction.loc[j2_2], equal_nan=True))
    assert np.all(np.isclose(net.res_junction.loc[j2], net2.res_junction.loc[j4_2], equal_nan=True))
    assert np.all(np.isclose(net.res_sink.values, net2.res_sink.values, equal_nan=True))
    assert np.all(np.isclose(net.res_valve.values, net2.res_valve.values, equal_nan=True))
    assert np.all(np.isclose(net.res_ext_grid.values, net2.res_ext_grid.values, equal_nan=True))


@pytest.mark.parametrize("use_numba", [True, False])
def test_valve_to_pipe2(use_numba):
    net = pandapipes.create_empty_network("net", add_stdtypes=True)

    j0 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15, index=7)
    j1 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15, index=20)
    j2 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15, index=5)
    j3 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15, index=3)
    j4 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15, index=1)

    pandapipes.create_ext_grid(net, j0, 5, 283.15, type="p", index=12)

    p0 = pandapipes.create_pipe_from_parameters(net, j0, j1, diameter_m=.1, k_mm=1, length_km=1., index=10)
    p1 = pandapipes.create_pipe_from_parameters(net, j2, j1, diameter_m=.1, k_mm=1, length_km=1., sections=5, index=3)

    pandapipes.create_valve(net, j1, p0, et='pi', diameter_m=0.1, opened=True, loss_coefficient=1000, index=5)
    pandapipes.create_valve(net, j1, p0, et='pi', diameter_m=0.1, opened=True, loss_coefficient=2000, index=3)

    pandapipes.create_valve(net, j3, j2, et='ju', diameter_m=0.1, opened=False, loss_coefficient=2000)
    pandapipes.create_valve(net, j3, j2, et='ju', diameter_m=0.1, opened=True, loss_coefficient=500)

    p2 = pandapipes.create_pipe_from_parameters(net, j4, j3, diameter_m=.1, k_mm=1, length_km=1., sections=10, index=5)

    pandapipes.create_valve(net, j4, p2, et='pi', diameter_m=0.1, opened=True, loss_coefficient=5000, index=11)

    pandapipes.create_sink(net, j4, 0.11667)

    pandapipes.create_fluid_from_lib(net, "lgas", overwrite=True)

    max_iter_hyd = 100 if use_numba else 100
    pandapipes.pipeflow(net, stop_condition="tol", max_iter_hyd=max_iter_hyd, friction_model="nikuradse",
                        mode="hydraulics", transient=False, nonlinear_method="automatic",
                        tol_p=1e-8, tol_m=1e-8, use_numba=use_numba)

    net2 = pandapipes.create_empty_network("net2", add_stdtypes=True)

    j0_2 = pandapipes.create_junction(net2, pn_bar=5, tfluid_k=283.15, index=5)
    j1_2 = pandapipes.create_junction(net2, pn_bar=5, tfluid_k=283.15, index=3)
    j2_2 = pandapipes.create_junction(net2, pn_bar=5, tfluid_k=283.15, index=12)
    j3_2 = pandapipes.create_junction(net2, pn_bar=5, tfluid_k=283.15, index=7)
    j4_2 = pandapipes.create_junction(net2, pn_bar=5, tfluid_k=283.15, index=1)
    j5_2 = pandapipes.create_junction(net2, pn_bar=5, tfluid_k=283.15, index=13)
    j6_2 = pandapipes.create_junction(net2, pn_bar=5, tfluid_k=283.15, index=6)

    pandapipes.create_ext_grid(net2, j0_2, 5, 283.15, type="p")

    pandapipes.create_pipe_from_parameters(net2, j0_2, j1_2, diameter_m=.1, k_mm=1, length_km=1., index=10)
    pandapipes.create_pipe_from_parameters(net2, j3_2, j2_2, diameter_m=.1, k_mm=1, length_km=1., sections=5, index=3)

    pandapipes.create_valve(net2, j2_2, j1_2, et='ju', diameter_m=0.1, opened=True, loss_coefficient=1000)
    pandapipes.create_valve(net2, j2_2, j1_2, et='ju', diameter_m=0.1, opened=True, loss_coefficient=2000)

    pandapipes.create_valve(net2, j4_2, j3_2, et='ju', diameter_m=0.1, opened=False, loss_coefficient=2000)
    pandapipes.create_valve(net2, j4_2, j3_2, et='ju', diameter_m=0.1, opened=True, loss_coefficient=500)

    pandapipes.create_pipe_from_parameters(net2, j5_2, j4_2, diameter_m=.1, k_mm=1, length_km=1., sections=10, index=5)

    pandapipes.create_valve(net2, j6_2, j5_2, et='ju', diameter_m=0.1, opened=True, loss_coefficient=5000)

    pandapipes.create_sink(net2, j6_2, 0.11667)

    pandapipes.create_fluid_from_lib(net2, "lgas", overwrite=True)

    max_iter_hyd = 100 if use_numba else 100
    pandapipes.pipeflow(net2, stop_condition="tol", max_iter_hyd=max_iter_hyd, friction_model="nikuradse",
                        mode="hydraulics", transient=False, nonlinear_method="automatic",
                        tol_p=1e-8, tol_m=1e-8, use_numba=use_numba)

    assert np.all(np.isclose(net.res_pipe.values, net2.res_pipe.values, equal_nan=True))
    assert np.all(np.isclose(net.res_junction.loc[j0], net2.res_junction.loc[j0_2], equal_nan=True))
    assert np.all(np.isclose(net.res_junction.loc[j1], net2.res_junction.loc[j2_2], equal_nan=True))
    assert np.all(np.isclose(net.res_junction.loc[j3], net2.res_junction.loc[j4_2], equal_nan=True))
    assert np.all(np.isclose(net.res_sink.values, net2.res_sink.values, equal_nan=True))
    assert np.all(np.isclose(net.res_valve.values, net2.res_valve.values, equal_nan=True))
    assert np.all(np.isclose(net.res_ext_grid.values, net2.res_ext_grid.values, equal_nan=True))

if __name__ == '__main__':
    test_valve_to_pipe(False)