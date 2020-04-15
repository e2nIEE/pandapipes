# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import pandapipes as pp
import pandas as pd
from pandapipes.component_models import Pipe
from pandapipes.idx_branch import VINIT
from pandapipes.idx_node import PINIT, TINIT
from pandapipes.pipeflow_setup import get_lookup
from pandapipes.component_models.junction_component import Junction
from pandapipes.test import test_path

data_path = os.path.join(test_path, "pipeflow_internals", "data")


def test_hydraulic_only():
    """

    :return:
    :rtype:
    """
    net = pp.create_empty_network("net")
    d = 75e-3
    pp.create_junction(net, pn_bar=5, tfluid_k=283)
    pp.create_junction(net, pn_bar=5, tfluid_k=283)
    pp.create_pipe_from_parameters(net, 0, 1, 6, diameter_m=d, k_mm=.1, sections=1,
                                   alpha_w_per_m2k=5)
    pp.create_ext_grid(net, 0, p_bar=5, t_k=330, type="pt")
    pp.create_sink(net, 1, mdot_kg_per_s=1)

    pp.create_fluid_from_lib(net, "water", overwrite=True)

    pp.pipeflow(net, stop_condition="tol", iter=70, friction_model="nikuradse",
                transient=False, nonlinear_method="automatic", tol_p=1e-4,
                tol_v=1e-4)

    data = pd.read_csv(os.path.join(data_path, "hydraulics.csv"), sep=';', header=0,
                       keep_default_na=False)

    node_pit = net["_pit"]["node"]
    branch_pit = net["_pit"]["branch"]

    v_an = data.loc[0, "pv"]
    p_an = data.loc[1:3, "pv"]

    p_pandapipes = node_pit[:, PINIT]
    v_pandapipes = branch_pit[:, VINIT]

    p_diff = np.abs(1 - p_pandapipes / p_an)
    v_diff = np.abs(v_pandapipes - v_an)

    assert np.all(p_diff < 0.01)
    assert (np.all(v_diff < 0.05))


def test_heat_only():
    net = pp.create_empty_network("net")
    d = 75e-3
    pp.create_junction(net, pn_bar=5, tfluid_k=283)
    pp.create_junction(net, pn_bar=5, tfluid_k=283)
    pp.create_pipe_from_parameters(net, 0, 1, 6, diameter_m=d, k_mm=.1, sections=6,
                                   alpha_w_per_m2k=5)
    pp.create_ext_grid(net, 0, p_bar=5, t_k=330, type="pt")
    pp.create_sink(net, 1, mdot_kg_per_s=1)

    pp.create_fluid_from_lib(net, "water", overwrite=True)

    pp.pipeflow(net, stop_condition="tol", iter=70, friction_model="nikuradse",
                nonlinear_method="automatic", mode="all")

    ntw = pp.create_empty_network("net")
    d = 75e-3
    pp.create_junction(ntw, pn_bar=5, tfluid_k=283)
    pp.create_junction(ntw, pn_bar=5, tfluid_k=283)
    pp.create_pipe_from_parameters(ntw, 0, 1, 6, diameter_m=d, k_mm=.1, sections=6,
                                   alpha_w_per_m2k=5)
    pp.create_ext_grid(ntw, 0, p_bar=5, t_k=330, type="pt")
    pp.create_sink(ntw, 1, mdot_kg_per_s=1)

    pp.create_fluid_from_lib(ntw, "water", overwrite=True)

    pp.pipeflow(ntw, stop_condition="tol", iter=50, friction_model="nikuradse",
                nonlinear_method="automatic", mode="hydraulics")

    p = ntw._pit["node"][:, 5]
    v = ntw._pit["branch"][:, 12]
    u = np.concatenate((p, v))

    pp.pipeflow(ntw, sol_vec=u, stop_condition="tol", iter=50, friction_model="nikuradse",
                nonlinear_method="automatic", mode="heat")

    T_net = net.res_junction.t_k
    T_ntw = ntw.res_junction.t_k

    T_diff = np.abs(1 - T_net / T_ntw)

    assert np.all(T_diff < 0.01)
