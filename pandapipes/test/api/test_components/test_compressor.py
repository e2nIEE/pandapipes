# Copyright (c) 2020-2021 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import pandapipes
import os
import pytest
import numpy as np
import pandas as pd

from pandapipes.component_models import p_correction_height_air
from pandapipes.test.pipeflow_internals import internals_data_path


def test_compressor_boost_ratio():

    net = pandapipes.create_empty_network("net", add_stdtypes=True, fluid="hgas")

    j = pandapipes.create_junctions(net, 6, pn_bar=5, tfluid_k=283.15)
    j1, j2, j3, j4, j5, j6 = j

    pandapipes.create_pipe_from_parameters(net, j1, j2, length_km=0.43380, diameter_m=0.1022)
    pandapipes.create_pipe_from_parameters(net, j3, j4, length_km=0.26370, diameter_m=0.1022)
    pandapipes.create_ext_grid(net, j1, 5, 283.15, type="p")
    pandapipes.create_sink(net, j6, 0.02333)

    br1 = 1.5
    br3 = 1.1
    c1 = pandapipes.create_compressor_br(net, j2, j3, boost_ratio=br1)
    c2 = pandapipes.create_compressor_br(net, j5, j4, boost_ratio=br1)  # reverse flow -> bypass
    c3 = pandapipes.create_compressor_br(net, j5, j6, boost_ratio=br3)

    pandapipes.pipeflow(net)
    net.res_junction["abs_p_bar"] = net.res_junction.p_bar + \
                                         p_correction_height_air(net.junction.height_m)

    assert np.isclose(net.res_junction.at[j3, "abs_p_bar"],
                      br1 * net.res_junction.at[j2, "abs_p_bar"])
    assert np.isclose(net.res_junction.at[j6, "abs_p_bar"],
                      br3 * net.res_junction.at[j5, "abs_p_bar"])
    assert np.isclose(net.res_junction.at[j5, "p_bar"], net.res_junction.at[j4, "p_bar"]), \
        "pressure lift on rev. flow should be 0"
    assert np.isclose(net.res_junction.at[j5, "abs_p_bar"], net.res_junction.at[j4, "abs_p_bar"]), \
        "pressure lift on rev. flow should be 0"
    assert np.isclose(net.res_compressor.at[c1, "deltap_bar"],
                      net.res_junction.at[j2, "abs_p_bar"] * (br1 - 1))
    assert np.isclose(net.res_compressor.at[c3, "deltap_bar"],
                      net.res_junction.at[j5, "abs_p_bar"] * (br3 - 1))
    assert np.isclose(net.res_compressor.at[c2, "deltap_bar"], 0.0), \
       "pressure lift on rev. flow should be 0"

    br_new = 1.3
    net.compressor.loc[c1, "boost_ratio"] = br_new

    pandapipes.pipeflow(net)
    net.res_junction["abs_p_bar"] = net.res_junction.p_bar + \
                                    p_correction_height_air(net.junction.height_m)

    assert np.isclose(net.res_junction.at[j3, "abs_p_bar"], br_new * net.res_junction.at[j2, "abs_p_bar"])
    assert np.isclose(net.res_compressor.at[c1, "deltap_bar"],
                      net.res_junction.at[j2, "abs_p_bar"] * (br_new - 1))
    assert np.isclose(net.res_junction.at[j5, "abs_p_bar"], net.res_junction.at[j4, "abs_p_bar"])
    assert np.isclose(net.res_compressor.at[c2, "deltap_bar"], 0.0), \
        "pressure lift on rev. flow should be 0"


#
#
# def test_pump_from_measurement_parameteres():
#     """
#         :return:
#         :rtype:
#         """
#     net = pandapipes.create_empty_network("net", add_stdtypes=False)
#
#     j1 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
#     j2 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
#     j3 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
#     j4 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
#
#     pandapipes.create_pipe_from_parameters(net, j1, j2, k_mm=1., length_km=0.43380,
#                                            diameter_m=0.1022)
#     pandapipes.create_pipe_from_parameters(net, j3, j4, k_mm=1., length_km=0.26370,
#                                            diameter_m=0.1022)
#     pandapipes.create_ext_grid(net, j1, 5, 283.15, type="p")
#     pandapipes.create_pump_from_parameters(net, j2, j3, 'P1', [6.1, 5.8, 4], [0, 19, 83], 2)
#     pandapipes.create_sink(net, j4, 0.02333)
#
#     pandapipes.create_fluid_from_lib(net, "lgas", overwrite=True)
#
#    c, stop_condition="tol", iter=3, friction_model="nikuradse",
#                         mode="hydraulics", transient=False, nonlinear_method="automatic",
#                         tol_p=1e-4,
#                         tol_v=1e-4)
#
#     data = pd.read_csv(os.path.join(internals_data_path, "test_pump.csv"), sep=';')
#
#     res_junction = net.res_junction.p_bar.values
#     res_pipe = net.res_pipe.v_mean_m_per_s.values
#
#     p_diff = np.abs(1 - res_junction / data['p'].dropna().values)
#     v_diff = np.abs(1 - res_pipe / data['v'].dropna().values)
#
#     assert np.all(p_diff < 0.01)
#     assert np.all(v_diff < 0.01)
#
#
# def test_pump_from_regression_parameteres():
#     """
#
#         :return:
#         :rtype:
#         """
#     net = pandapipes.create_empty_network("net", add_stdtypes=False)
#
#     j1 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
#     j2 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
#     j3 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
#     j4 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
#
#     pandapipes.create_pipe_from_parameters(net, j1, j2, k_mm=1., length_km=0.43380,
#                                            diameter_m=0.1022)
#     pandapipes.create_pipe_from_parameters(net, j3, j4, k_mm=1., length_km=0.26370,
#                                            diameter_m=0.1022)
#     pandapipes.create_ext_grid(net, j1, 5, 283.15, type="p")
#     pandapipes.create_pump_from_parameters(net, j2, j3, 'P1',
#                                            poly_coefficents=[-1.48620799e-04, -1.29656785e-02,
#                                                              6.10000000e+00])
#     pandapipes.create_sink(net, j4, 0.02333)
#
#     pandapipes.create_fluid_from_lib(net, "lgas", overwrite=True)
#
#     pandapipes.pipeflow(net, stop_condition="tol", iter=3, friction_model="nikuradse",
#                         mode="hydraulics", transient=False, nonlinear_method="automatic",
#                         tol_p=1e-4,
#                         tol_v=1e-4)
#
#     data = pd.read_csv(os.path.join(internals_data_path, "test_pump.csv"), sep=';')
#
#     res_junction = net.res_junction.p_bar.values
#     res_pipe = net.res_pipe.v_mean_m_per_s.values
#
#     p_diff = np.abs(1 - res_junction / data['p'].dropna().values)
#     v_diff = np.abs(1 - res_pipe / data['v'].dropna().values)
#
#     assert np.all(p_diff < 0.01)
#     assert np.all(v_diff < 0.01)
#
#
# def test_pump_from_std_type():
#     """
#
#         :return:
#         :rtype:
#         """
#     net = pandapipes.create_empty_network("net", add_stdtypes=True)
#
#     j1 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
#     j2 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
#     j3 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
#     j4 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
#
#     pandapipes.create_pipe(net, j1, j2, std_type='125_PE_80_SDR_11', k_mm=1., length_km=0.43380)
#     pandapipes.create_pipe(net, j3, j4, std_type='125_PE_80_SDR_11', k_mm=1., length_km=0.26370)
#     pandapipes.create_ext_grid(net, j1, 5, 283.15, type="p")
#     pandapipes.create_pump(net, j2, j3, std_type='P1')
#     pandapipes.create_sink(net, j4, 0.02333)
#
#     pandapipes.create_fluid_from_lib(net, "lgas", overwrite=True)
#
#     pandapipes.pipeflow(net, stop_condition="tol", iter=3, friction_model="nikuradse",
#                         mode="hydraulics", transient=False, nonlinear_method="automatic",
#                         tol_p=1e-4,
#                         tol_v=1e-4)
#
#     data = pd.read_csv(os.path.join(internals_data_path, "test_pump.csv"), sep=';')
#
#     res_junction = net.res_junction.p_bar.values
#     res_pipe = net.res_pipe.v_mean_m_per_s.values
#
#     p_diff = np.abs(1 - res_junction / data['p'].dropna().values)
#     v_diff = np.abs(1 - res_pipe / data['v'].dropna().values)
#
#     assert np.all(p_diff < 0.01)
#     assert np.all(v_diff < 0.01)
#
# def test_pump_bypass_on_reverse_flow():
#     """
#     reverse flow = no pressure lift
#         :return:
#         :rtype:
#         """
#     net = pandapipes.create_empty_network("net", add_stdtypes=True)
#
#     j1 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
#     j2 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
#     j3 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
#     j4 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
#
#     pandapipes.create_pipe(net, j1, j2, std_type='125_PE_80_SDR_11', k_mm=1., length_km=10)
#     pandapipes.create_pipe(net, j3, j4, std_type='125_PE_80_SDR_11', k_mm=1., length_km=12)
#     pandapipes.create_ext_grid(net, j1, 5, 283.15, type="p")
#     pandapipes.create_pump(net, j2, j3, std_type='P1')
#     pandapipes.create_source(net, j4, 0.02333)
#
#     pandapipes.create_fluid_from_lib(net, "hgas", overwrite=True)
#
#     pandapipes.pipeflow(net, stop_condition="tol", iter=3, friction_model="nikuradse",
#                         mode="hydraulics", transient=False, nonlinear_method="automatic",
#                         tol_p=1e-4, tol_v=1e-4)
#
#     assert net.res_pump.deltap_bar.isin([0]).all()
#     assert np.isclose(net.res_junction.loc[1, "p_bar"], net.res_junction.loc[2, "p_bar"])
#
# def test_pump_bypass_high_vdot():
#     """
#     High flow: pressure lift not <0, always >=0
#         :return:
#         :rtype:
#         """
#     net = pandapipes.create_empty_network("net", add_stdtypes=True)
#
#     j1 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
#     j2 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
#     j3 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
#     j4 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
#
#     pandapipes.create_pipe(net, j1, j2, std_type='2000_ST<16', k_mm=0.1, length_km=0.1)
#     pandapipes.create_pipe(net, j3, j4, std_type='2000_ST<16', k_mm=0.1, length_km=0.1)
#     pandapipes.create_ext_grid(net, j1, 5, 283.15, type="p")
#     pandapipes.create_pump(net, j2, j3, std_type='P1')
#     pandapipes.create_sink(net, j4, 1000)
#
#     pandapipes.create_fluid_from_lib(net, "hgas", overwrite=True)
#
#     pandapipes.pipeflow(net, stop_condition="tol", iter=30, friction_model="nikuradse",
#                         mode="hydraulics", transient=False, nonlinear_method="automatic",
#                         tol_p=1e-4, tol_v=1e-4)
#
#     assert net.res_pump.deltap_bar.isin([0]).all()
#     assert np.isclose(net.res_junction.loc[1, "p_bar"], net.res_junction.loc[2, "p_bar"])


if __name__ == '__main__':
    n = pytest.main(["test_compressor.py"])