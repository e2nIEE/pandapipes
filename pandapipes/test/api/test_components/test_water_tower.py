# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import pandas as pd

import pandapipes as pp
from pandapipes.test.pipeflow_internals import internals_data_path


def test_reservoir():
    """

        :rtype:
        """
    net = pp.create_empty_network(fluid="water")

    junction1 = pp.create_junction(net, pn_bar=1.0, tfluid_k=293.15, name="Connection to Water Tower")
    junction2 = pp.create_junction(net, pn_bar=1.0, tfluid_k=293.15, name="Junction 2")

    pp.create_water_tower(net, junction1, height_m=30, name="Water Tower")

    pp.create_pipe_from_parameters(net, from_junction=junction1, to_junction=junction2, length_km=10, diameter_m=0.075,
                                   name="Pipe 1")

    pp.create_sink(net, junction=junction2, mdot_kg_per_s=0.545, name="Sink 1")

    pp.pipeflow(net, stop_condition="tol", iter=3, friction_model="nikuradse",
                mode="hydraulics", transient=False, nonlinear_method="automatic",
                tol_p=1e-4,
                tol_v=1e-4)

    data = pd.read_csv(os.path.join(internals_data_path, "test_water_tower.csv"), sep=';')

    res_junction = net.res_junction.p_bar.values
    res_pipe = net.res_pipe.v_mean_m_per_s.values

    p_diff = np.abs(1 - res_junction / data['p'].dropna().values)
    v_diff = np.abs(1 - res_pipe / data['v'].dropna().values)

    assert np.all(p_diff < 0.01)
    assert np.all(v_diff < 0.01)
