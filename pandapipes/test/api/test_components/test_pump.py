# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import pandapipes
import os
import numpy as np
import pandas as pd
from pandapipes.test.pipeflow_internals import internals_data_path


def test_pump_from_measurement_parameteres():
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
    pandapipes.create_ext_grid(net, j1, 5, 283.15, type="p")
    pandapipes.create_pump_from_parameters(net, j2, j3, 'P1', [6.1, 5.8, 4], [0, 19, 83], 2)
    pandapipes.create_sink(net, j4, 0.02333)

    pandapipes.create_fluid_from_lib(net, "lgas", overwrite=True)

    pandapipes.pipeflow(net, stop_condition="tol", iter=3, friction_model="nikuradse",
                        mode="hydraulics", transient=False, nonlinear_method="automatic",
                        tol_p=1e-4,
                        tol_v=1e-4)

    data = pd.read_csv(os.path.join(internals_data_path, "test_pump.csv"), sep=';')

    res_junction = net.res_junction.p_bar.values
    res_pipe = net.res_pipe.v_mean_m_per_s.values

    p_diff = np.abs(1 - res_junction / data['p'].dropna().values)
    v_diff = np.abs(1 - res_pipe / data['v'].dropna().values)

    assert np.all(p_diff < 0.01)
    assert np.all(v_diff < 0.01)


def test_pump_from_regression_parameteres():
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
    pandapipes.create_ext_grid(net, j1, 5, 283.15, type="p")
    pandapipes.create_pump_from_parameters(net, j2, j3, 'P1',
                                           regression_parameters=[-1.48620799e-04, -1.29656785e-02,
                                                                  6.10000000e+00])
    pandapipes.create_sink(net, j4, 0.02333)

    pandapipes.create_fluid_from_lib(net, "lgas", overwrite=True)

    pandapipes.pipeflow(net, stop_condition="tol", iter=3, friction_model="nikuradse",
                        mode="hydraulics", transient=False, nonlinear_method="automatic",
                        tol_p=1e-4,
                        tol_v=1e-4)

    data = pd.read_csv(os.path.join(internals_data_path, "test_pump.csv"), sep=';')

    res_junction = net.res_junction.p_bar.values
    res_pipe = net.res_pipe.v_mean_m_per_s.values

    p_diff = np.abs(1 - res_junction / data['p'].dropna().values)
    v_diff = np.abs(1 - res_pipe / data['v'].dropna().values)

    assert np.all(p_diff < 0.01)
    assert np.all(v_diff < 0.01)


def test_pump_from_std_type():
    """

        :return:
        :rtype:
        """
    net = pandapipes.create_empty_network("net", add_stdtypes=True)

    j1 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j2 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j3 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j4 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)

    pandapipes.create_pipe(net, j1, j2, std_type='125_PE_80_SDR_11', k_mm=1., length_km=0.43380)
    pandapipes.create_pipe(net, j3, j4, std_type='125_PE_80_SDR_11', k_mm=1., length_km=0.26370)
    pandapipes.create_ext_grid(net, j1, 5, 283.15, type="p")
    pandapipes.create_pump(net, j2, j3, std_type='P1')
    pandapipes.create_sink(net, j4, 0.02333)

    pandapipes.create_fluid_from_lib(net, "lgas", overwrite=True)

    pandapipes.pipeflow(net, stop_condition="tol", iter=3, friction_model="nikuradse",
                        mode="hydraulics", transient=False, nonlinear_method="automatic",
                        tol_p=1e-4,
                        tol_v=1e-4)

    data = pd.read_csv(os.path.join(internals_data_path, "test_pump.csv"), sep=';')

    res_junction = net.res_junction.p_bar.values
    res_pipe = net.res_pipe.v_mean_m_per_s.values

    p_diff = np.abs(1 - res_junction / data['p'].dropna().values)
    v_diff = np.abs(1 - res_pipe / data['v'].dropna().values)

    assert np.all(p_diff < 0.01)
    assert np.all(v_diff < 0.01)
