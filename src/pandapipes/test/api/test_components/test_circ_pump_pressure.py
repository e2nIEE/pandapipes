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
    pandapipes.create_heat_exchanger(net, j2, j3, qext_w=200000)
    pandapipes.create_sink(net, j1, 2)
    pandapipes.create_source(net, j4, 2)

    pandapipes.create_fluid_from_lib(net, "water", overwrite=True)

    max_iter_hyd = 8 if use_numba else 8
    max_iter_therm = 7 if use_numba else 7
    pandapipes.pipeflow(net, max_iter_hyd=max_iter_hyd, max_iter_therm=max_iter_therm,
                        stop_condition="tol", friction_model="nikuradse",
                        mode='sequential', transient=False, nonlinear_method="automatic",
                        tol_p=1e-4, tol_m=1e-4, use_numba=use_numba)

    data = pd.read_csv(os.path.join(data_path, "test_circ_pump_pressure.csv"), sep=';')

    res_junction = net.res_junction
    res_pipe = net.res_pipe.v_mean_m_per_s.values
    res_pump = net.res_circ_pump_pressure

    p_diff = np.abs(1 - res_junction.p_bar.values / data['p'].dropna().values)
    t_diff = np.abs(1 - res_junction.t_k.values / data['t'].dropna().values)
    v_diff = np.abs(1 - res_pipe / data['v'].dropna().values)
    mdot_diff = np.abs(1 - res_pump['mdot_from_kg_per_s'].values / data['mdot'].dropna().values)
    deltap_diff = np.abs(
        1 - (res_pump['p_to_bar'].values - res_pump['p_from_bar'].values) / data['deltap'].dropna().values)

    assert np.all(p_diff < 0.01)
    assert np.all(t_diff < 0.01)
    assert np.all(v_diff < 0.01)
    assert np.all(mdot_diff < 0.01)
    assert np.all(deltap_diff < 0.01)
