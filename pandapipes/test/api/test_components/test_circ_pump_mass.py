# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import pandapipes
import os
import numpy as np
import pandas as pd
from pandapipes.test.pipeflow_internals import internals_data_path


def test_circulation_pump_constant_mass():
    """
        :return:
        :rtype:
        """
    net = pandapipes.create_empty_network("net", add_stdtypes=False)

    j1 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j2 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j3 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j4 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)

    pandapipes.create_pipe_from_parameters(net, j1, j2, k=0.1, length_km=0.43380, diameter_m=0.1022)
    pandapipes.create_pipe_from_parameters(net, j3, j4, k=0.1, length_km=0.26370, diameter_m=0.1022)
    pandapipes.create_circ_pump_const_mass_flow(net, j1, j4, 5, 5, 300, type='pt')
    pandapipes.create_heat_exchanger(net, j2, j3, 0.1, qext_w=200000)
    pandapipes.create_sink(net, j1, 2)
    pandapipes.create_source(net, j4, 2)

    pandapipes.create_fluid_from_lib(net, "water", overwrite=True)

    pandapipes.pipeflow(net, stop_condition="tol", iter=10, friction_model="nikuradse",
                        mode="all", transient=False, nonlinear_method="automatic",
                        tol_p=1e-4, tol_v=1e-4)

    data = pd.read_csv(os.path.join(internals_data_path, "test_circ_pump_mass.csv"), sep=';')

    res_junction = net.res_junction
    res_pipe = net.res_pipe.v_mean_m_per_s.values
    res_pump = net.res_circ_pump_mass

    p_diff = np.abs(1 - res_junction.p_bar.values / data['p'].dropna().values)
    t_diff = np.abs(1 - res_junction.t_k.values / data['t'].dropna().values)
    v_diff = np.abs(1 - res_pipe / data['v'].dropna().values)
    mdot_diff = np.abs(1 - res_pump['mdot_kg_per_s'].values / data['mdot'].dropna().values)
    deltap_diff = np.abs(1 - res_pump['deltap_bar'].values / data['deltap'].dropna().values)

    assert np.all(p_diff < 0.01)
    assert np.all(t_diff < 0.01)
    assert np.all(v_diff < 0.01)
    assert np.all(mdot_diff < 0.01)
    assert np.all(deltap_diff < 0.01)
