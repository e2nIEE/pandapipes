# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import pandapipes
import os
import pandas as pd
import numpy as np
from pandapipes.test.pipeflow_internals import internals_data_path


def test_valve():
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

    pandapipes.create_pipe_from_parameters(net, j0, j1, diameter_m=.1, k=0.1, length_km=1.)
    pandapipes.create_pipe_from_parameters(net, j3, j4, diameter_m=.1, k=0.1, length_km=.5)
    pandapipes.create_pipe_from_parameters(net, j2, j4, diameter_m=.1, k=0.1, length_km=.5)
    pandapipes.create_pipe_from_parameters(net, j5, j4, diameter_m=.1, k=0.1, length_km=.35)
    pandapipes.create_pipe_from_parameters(net, j1, j6, diameter_m=.1, k=0.1, length_km=.1,
                                           loss_coefficient=9000)
    pandapipes.create_pipe_from_parameters(net, j1, j7, diameter_m=.1, k=0.1, length_km=.1,
                                           loss_coefficient=9000)

    pandapipes.create_valve(net, j6, j2, diameter_m=0.1, opened=False)
    pandapipes.create_valve(net, j7, j3, diameter_m=0.1, opened=True)

    pandapipes.create_sink(net, j5, 0.11667)

    pandapipes.create_fluid_from_lib(net, "lgas", overwrite=True)

    pandapipes.pipeflow(net, stop_condition="tol", iter=10, friction_model="nikuradse",
                        mode="hydraulics", transient=False, nonlinear_method="automatic",
                        tol_p=1e-4,
                        tol_v=1e-4)

    data = pd.read_csv(os.path.join(internals_data_path, "test_valve.csv"), sep=';')
    data['p'].dropna(inplace=True)
    data['v'].dropna(inplace=True)

    res_junction = net.res_junction.p_bar.values
    res_pipe = net.res_pipe.v_mean_m_per_s.values
    zeros = res_pipe == 0
    test_zeros = data['v'].values == 0
    check_zeros = zeros == test_zeros

    assert np.all(check_zeros)

    p_diff = np.abs(1 - res_junction / data['p'][data['p'] != 0].values)
    v_diff = np.abs(1 - res_pipe[res_pipe != 0] / data['v'][data['v'] != 0].values)

    assert np.all(p_diff < 0.01)
    assert np.all(v_diff < 0.01)
