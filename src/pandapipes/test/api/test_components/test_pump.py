# Copyright (c) 2020-2025 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import pandas as pd
import pytest

import pandapipes
from pandapipes.test import data_path


@pytest.mark.parametrize("use_numba", [True, False])
def test_pump_from_measurement_parameteres(use_numba):
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

    max_iter_hyd = 3 if use_numba else 3
    pandapipes.pipeflow(net, stop_condition="tol", max_iter_hyd=max_iter_hyd, friction_model="nikuradse",
                        mode="hydraulics", transient=False, nonlinear_method="automatic",
                        tol_p=1e-4, tol_m=1e-4, use_numba=use_numba)

    data = pd.read_csv(os.path.join(data_path, "test_pump.csv"), sep=';')

    res_junction = net.res_junction.p_bar.values
    res_pipe = net.res_pipe.v_mean_m_per_s.values

    p_diff = np.abs(1 - res_junction / data['p'].dropna().values)
    v_diff = np.abs(1 - res_pipe / data['v'].dropna().values)

    assert np.all(p_diff < 0.01)
    assert np.all(v_diff < 0.01)


@pytest.mark.parametrize("use_numba", [True, False])
def test_pump_from_regression_parameteres(use_numba):
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
                                           poly_coefficents=[-1.48620799e-04, -1.29656785e-02,
                                                             6.10000000e+00])
    pandapipes.create_sink(net, j4, 0.02333)

    pandapipes.create_fluid_from_lib(net, "lgas", overwrite=True)

    max_iter_hyd = 3 if use_numba else 3
    pandapipes.pipeflow(net, stop_condition="tol", max_iter_hyd=max_iter_hyd, friction_model="nikuradse",
                        mode="hydraulics", transient=False, nonlinear_method="automatic",
                        tol_p=1e-4, tol_m=1e-4, use_numba=use_numba)

    data = pd.read_csv(os.path.join(data_path, "test_pump.csv"), sep=';')

    res_junction = net.res_junction.p_bar.values
    res_pipe = net.res_pipe.v_mean_m_per_s.values

    p_diff = np.abs(1 - res_junction / data['p'].dropna().values)
    v_diff = np.abs(1 - res_pipe / data['v'].dropna().values)

    assert np.all(p_diff < 0.01)
    assert np.all(v_diff < 0.01)


@pytest.mark.parametrize("use_numba", [True, False])
def test_pump_from_std_type(use_numba):
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

    max_iter_hyd = 3 if use_numba else 3
    pandapipes.pipeflow(net, stop_condition="tol", max_iter_hyd=max_iter_hyd, friction_model="nikuradse",
                        mode="hydraulics", transient=False, nonlinear_method="automatic",
                        tol_p=1e-4, tol_m=1e-4, use_numba=use_numba)

    data = pd.read_csv(os.path.join(data_path, "test_pump.csv"), sep=';')

    res_junction = net.res_junction.p_bar.values
    res_pipe = net.res_pipe.v_mean_m_per_s.values

    p_diff = np.abs(1 - res_junction / data['p'].dropna().values)
    v_diff = np.abs(1 - res_pipe / data['v'].dropna().values)

    assert np.all(p_diff < 0.01)
    assert np.all(v_diff < 0.01)


@pytest.mark.parametrize("use_numba", [True, False])
def test_pump_bypass_on_reverse_flow(use_numba):
    """
    reverse flow = no pressure lift
        :return:
        :rtype:
        """
    net = pandapipes.create_empty_network("net", add_stdtypes=True)

    j1 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j2 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j3 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j4 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)

    pandapipes.create_pipe(net, j1, j2, std_type='125_PE_80_SDR_11', k_mm=1., length_km=10)
    pandapipes.create_pipe(net, j3, j4, std_type='125_PE_80_SDR_11', k_mm=1., length_km=12)
    pandapipes.create_ext_grid(net, j1, 5, 283.15, type="p")
    pandapipes.create_pump(net, j2, j3, std_type='P1')
    pandapipes.create_source(net, j4, 0.02333)

    pandapipes.create_fluid_from_lib(net, "hgas", overwrite=True)

    max_iter_hyd = 4 if use_numba else 4
    pandapipes.pipeflow(net, stop_condition="tol", max_iter_hyd=max_iter_hyd, friction_model="nikuradse",
                        mode="hydraulics", transient=False, nonlinear_method="automatic",
                        tol_p=1e-4, tol_m=1e-4, use_numba=use_numba)

    assert net.res_pump.deltap_bar.isin([0]).all()
    assert np.isclose(net.res_junction.loc[1, "p_bar"], net.res_junction.loc[2, "p_bar"])


@pytest.mark.parametrize("use_numba", [True, False])
def test_pump_bypass_high_vdot(use_numba):
    """
    High flow: pressure lift not <0, always >=0
        :return:
        :rtype:
        """
    net = pandapipes.create_empty_network("net", add_stdtypes=True)

    j1 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j2 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j3 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j4 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)

    pandapipes.create_pipe(net, j1, j2, std_type='2000_ST<16', k_mm=0.1, length_km=0.1)
    pandapipes.create_pipe(net, j3, j4, std_type='2000_ST<16', k_mm=0.1, length_km=0.1)
    pandapipes.create_ext_grid(net, j1, 5, 283.15, type="p")
    pandapipes.create_pump(net, j2, j3, std_type='P1')
    pandapipes.create_sink(net, j4, 1000)

    pandapipes.create_fluid_from_lib(net, "hgas", overwrite=True)

    max_iter_hyd = 5 if use_numba else 5
    pandapipes.pipeflow(net, stop_condition="tol", max_iter_hyd=max_iter_hyd, friction_model="nikuradse",
                        mode="hydraulics", transient=False, nonlinear_method="automatic",
                        tol_p=1e-4, tol_m=1e-4, use_numba=use_numba)

    assert net.res_pump.deltap_bar.isin([0]).all()
    assert np.isclose(net.res_junction.loc[1, "p_bar"], net.res_junction.loc[2, "p_bar"])


@pytest.mark.parametrize("use_numba", [True, False])
def test_compression_power(use_numba):
    # based on example by "oporras"
    from pandapipes.component_models import R_UNIVERSAL
    from pandapipes.idx_node import PAMB

    height_asl_m = 2842
    net = pandapipes.create_empty_network(fluid="methane")

    j0 = pandapipes.create_junction(net, pn_bar=1.05, tfluid_k=293.15, height_m=height_asl_m)
    j1 = pandapipes.create_junction(net, pn_bar=1.05, tfluid_k=293.15, height_m=height_asl_m)
    j2 = pandapipes.create_junction(net, pn_bar=1.05, tfluid_k=293.15, height_m=height_asl_m+10)
    j3 = pandapipes.create_junction(net, pn_bar=1.05, tfluid_k=293.15, height_m=height_asl_m+10)

    _ = pandapipes.create_pipe_from_parameters(net, from_junction=j0, to_junction=j1, length_km=0.1,
                                               diameter_m=0.05)
    _ = pandapipes.create_pipe_from_parameters(net, from_junction=j2, to_junction=j3, length_km=0.5,
                                               diameter_m=0.05)

    _ = pandapipes.create_pump(net, from_junction=j1, to_junction=j2, std_type="P2", name="Pump1")

    _ = pandapipes.create_ext_grid(net, junction=j0, p_bar=4, t_k=293.15)
    _ = pandapipes.create_sink(net, junction=j3, mdot_kg_per_s=0.05)

    max_iter_hyd = 4 if use_numba else 4
    pandapipes.pipeflow(net, max_iter_hyd=max_iter_hyd,
                        use_numba=use_numba)

    # Local ambiental (atmospheric) pressure
    p_amb_bar_j1 = net["_pit"]['node'][1][PAMB]
    p_amb_bar_j2 = net["_pit"]['node'][2][PAMB]

    # Isentropic power for the compression
    R_spec = R_UNIVERSAL * 1e3 / pandapipes.get_fluid(net).get_molar_mass()
    cp = pandapipes.get_fluid(net).get_heat_capacity(293.15)
    cv = cp - R_spec
    k = cp / cv
    pressure_ratio = ((net.res_pump.p_to_bar[0] + p_amb_bar_j2) /
                      (net.res_pump.p_from_bar[0] + p_amb_bar_j1))
    compr = pandapipes.get_fluid(net).get_compressibility(net.res_pump.p_from_bar[0] + + p_amb_bar_j1)
    pow_pump_MW = (net.res_pump.mdot_from_kg_per_s[0] * (k / (k - 1)) * R_spec *
                   compr * net.res_pump.t_from_k[0] * (pressure_ratio ** ((k - 1) / k) - 1) / 1e6)
    assert np.isclose(pow_pump_MW[0], net.res_pump.compr_power_mw[0])


if __name__ == '__main__':
    n = pytest.main(["test_pump.py"])
