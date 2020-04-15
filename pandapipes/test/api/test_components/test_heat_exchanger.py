# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import pandapipes
import pandas as pd
import os
from pandapipes.test.pipeflow_internals import internals_data_path
import numpy as np


def test_heat_exchanger():
    """

        :return:
        :rtype:
        """
    net = pandapipes.create_empty_network("net", add_stdtypes=False)
    d = 75e-3

    pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    pandapipes.create_heat_exchanger(net, 0, 1, d, qext_w=20000)
    pandapipes.create_ext_grid(net, 0, p_bar=5, t_k=330, type="pt")
    pandapipes.create_sink(net, 1, mdot_kg_per_s=1)

    pandapipes.create_fluid_from_lib(net, "water", overwrite=True)

    pandapipes.pipeflow(net, stop_condition="tol", iter=3, friction_model="nikuradse",
                        mode="all", transient=False, nonlinear_method="automatic", tol_p=1e-4,
                        tol_v=1e-4)

    data = pd.read_csv(os.path.join(internals_data_path, "heat_exchanger_test.csv"), sep=';',
                       header=0, keep_default_na=False)
    temp_an = data["T1"]

    t_pan = net.res_junction.t_k

    temp_diff = np.abs(1 - t_pan / temp_an)

    assert np.all(temp_diff < 0.01)
