# Copyright (c) 2020-2023 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import copy
import os

import numpy as np
import pandas as pd
import pytest

import pandapipes
from pandapipes.idx_branch import VINIT
from pandapipes.idx_node import PINIT
from pandapipes.test import test_path

data_path = os.path.join(test_path, "pipeflow_internals", "data")


@pytest.fixture
def simple_test_net():
    net = pandapipes.create_empty_network("net")
    d = 75e-3
    pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    pandapipes.create_pipe_from_parameters(net, 0, 1, 6, diameter_m=d, k_mm=.1, sections=1,
                                           alpha_w_per_m2k=5)
    pandapipes.create_ext_grid(net, 0, p_bar=5, t_k=330, type="pt")
    pandapipes.create_sink(net, 1, mdot_kg_per_s=1)

    pandapipes.create_fluid_from_lib(net, "water", overwrite=True)

    return net


@pytest.mark.parametrize("use_numba", [True, False])
def test_hydraulic_only(simple_test_net, use_numba):
    """

    :return:
    :rtype:
    """
    net = copy.deepcopy(simple_test_net)
    pandapipes.pipeflow(net, stop_condition="tol", iter=70, friction_model="nikuradse",
                        transient=False, nonlinear_method="automatic", tol_p=1e-4, tol_v=1e-4,
                        use_numba=use_numba)

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


@pytest.mark.parametrize("use_numba", [True, False])
def test_heat_only(use_numba):
    net = pandapipes.create_empty_network("net")
    d = 75e-3
    pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    pandapipes.create_pipe_from_parameters(net, 0, 1, 6, diameter_m=d, k_mm=.1, sections=6,
                                           alpha_w_per_m2k=5)
    pandapipes.create_ext_grid(net, 0, p_bar=5, t_k=330, type="pt")
    pandapipes.create_sink(net, 1, mdot_kg_per_s=1)

    pandapipes.create_fluid_from_lib(net, "water", overwrite=True)

    pandapipes.pipeflow(net, stop_condition="tol", iter=70, friction_model="nikuradse",
                        nonlinear_method="automatic", mode="all", use_numba=use_numba)

    ntw = pandapipes.create_empty_network("net")
    d = 75e-3
    pandapipes.create_junction(ntw, pn_bar=5, tfluid_k=283)
    pandapipes.create_junction(ntw, pn_bar=5, tfluid_k=283)
    pandapipes.create_pipe_from_parameters(ntw, 0, 1, 6, diameter_m=d, k_mm=.1, sections=6,
                                           alpha_w_per_m2k=5)
    pandapipes.create_ext_grid(ntw, 0, p_bar=5, t_k=330, type="pt")
    pandapipes.create_sink(ntw, 1, mdot_kg_per_s=1)

    pandapipes.create_fluid_from_lib(ntw, "water", overwrite=True)

    pandapipes.pipeflow(ntw, stop_condition="tol", iter=50, friction_model="nikuradse",
                        nonlinear_method="automatic", mode="hydraulics", use_numba=use_numba)

    p = ntw._pit["node"][:, 5]
    v = ntw._pit["branch"][:, 12]
    u = np.concatenate((p, v))

    pandapipes.pipeflow(ntw, sol_vec=u, stop_condition="tol", iter=50, friction_model="nikuradse",
                        nonlinear_method="automatic", mode="heat", use_numba=use_numba)

    temp_net = net.res_junction.t_k
    temp_ntw = ntw.res_junction.t_k

    temp_diff = np.abs(1 - temp_net / temp_ntw)

    assert np.all(temp_diff < 0.01)
