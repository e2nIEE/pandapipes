# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import copy
import os

import numpy as np
import pandas as pd
import pytest

import pandapipes
from pandapipes.constants import NORMAL_TEMPERATURE
from pandapipes.idx_branch import MDOTINIT, AREA
from pandapipes.idx_node import PINIT

from pandapipes.properties import get_fluid
from pandapipes.test import data_path
from pandapipes.networks import schutterwald_gas, schutterwald_heat


@pytest.fixture
def simple_test_net():
    net = pandapipes.create_empty_network("net")
    d = 75
    pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    pandapipes.create_pipe_from_parameters(net, 0, 1, 6, inner_diameter_mm=d, k_mm=.1, sections=1,
                                           u_w_per_m2k=5)
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

    max_iter_hyd = 3 if use_numba else 3
    pandapipes.pipeflow(net, max_iter_hyd=max_iter_hyd,
                        stop_condition="tol", friction_model="nikuradse",
                        transient=False, nonlinear_method="automatic", tol_p=1e-4, tol_m=1e-4,
                        use_numba=use_numba)

    data = pd.read_csv(os.path.join(data_path, "hydraulics.csv"), sep=';', header=0,
                       keep_default_na=False)

    node_pit = net["_pit"]["node"]
    branch_pit = net["_pit"]["branch"]

    v_an = data.loc[0, "pv"]
    p_an = data.loc[1:3, "pv"]

    p_pandapipes = node_pit[:, PINIT]
    fluid = get_fluid(net)
    v_pandapipes = branch_pit[:, MDOTINIT] / branch_pit[:, AREA] / fluid.get_density(NORMAL_TEMPERATURE)

    p_diff = np.abs(1 - p_pandapipes / p_an)
    v_diff = np.abs(v_pandapipes - v_an)

    assert np.all(p_diff < 0.01)
    assert (np.all(v_diff < 0.05))


@pytest.mark.parametrize("use_numba", [True, False])
def test_heat_only(use_numba):
    net = pandapipes.create_empty_network("net")
    d = 75
    pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    pandapipes.create_junction(net, pn_bar=5, tfluid_k=283)
    pandapipes.create_pipe_from_parameters(net, 0, 1, 6, inner_diameter_mm=d, k_mm=.1, sections=6,
                                           u_w_per_m2k=5)
    pandapipes.create_ext_grid(net, 0, p_bar=5, t_k=330, type="pt")
    pandapipes.create_sink(net, 1, mdot_kg_per_s=1)

    pandapipes.create_fluid_from_lib(net, "water", overwrite=True)

    max_iter_hyd = 3 if use_numba else 3
    max_iter_therm = 4 if use_numba else 4
    pandapipes.pipeflow(net, max_iter_hyd=max_iter_hyd, max_iter_therm=max_iter_therm,
                        stop_condition="tol", friction_model="nikuradse",
                        nonlinear_method="automatic", mode='sequential', use_numba=use_numba)

    ntw = pandapipes.create_empty_network("net")
    d = 75
    pandapipes.create_junction(ntw, pn_bar=5, tfluid_k=283)
    pandapipes.create_junction(ntw, pn_bar=5, tfluid_k=283)
    pandapipes.create_pipe_from_parameters(ntw, 0, 1, 6, inner_diameter_mm=d, k_mm=.1, sections=6,
                                           u_w_per_m2k=5)
    pandapipes.create_ext_grid(ntw, 0, p_bar=5, t_k=330, type="pt")
    pandapipes.create_sink(ntw, 1, mdot_kg_per_s=1)

    pandapipes.create_fluid_from_lib(ntw, "water", overwrite=True)

    max_iter_hyd = 3 if use_numba else 3
    pandapipes.pipeflow(ntw, max_iter_hyd=max_iter_hyd, stop_condition="tol", friction_model="nikuradse",
                        nonlinear_method="automatic", mode="hydraulics", use_numba=use_numba)

    p = ntw._pit["node"][:, PINIT]
    m = ntw._pit["branch"][:, MDOTINIT]
    u = np.concatenate((p, m))

    max_iter_therm = 4 if use_numba else 4
    pandapipes.pipeflow(ntw, max_iter_therm=max_iter_therm,
                        sol_vec=u, stop_condition="tol", friction_model="nikuradse",
                        nonlinear_method="automatic", mode="heat", use_numba=use_numba)

    temp_net = net.res_junction.t_k
    temp_ntw = ntw.res_junction.t_k

    temp_diff = np.abs(1 - temp_net / temp_ntw)

    assert np.all(temp_diff < 0.01)

@pytest.mark.parametrize("net", [schutterwald_gas, schutterwald_heat])
@pytest.mark.parametrize("use_numba", [True, False])
def test_diam_opt_v(net, use_numba):
    net = net()
    net.pipe.v_max_m_per_s = 0.2
    iteration = 0
    tol_v = np.ones(len(net.pipe))
    while np.any(np.abs(tol_v) >= 0.001) and (iteration <= 100):
        pandapipes.pipeflow(net, diam_opt=True)
        zero_mask = np.isclose(net.res_pipe.d_opt_m.values, 0, atol = 0.01)
        net.pipe.loc[net.pipe.index[~zero_mask], 'diameter_m'] = net.res_pipe.loc[net.pipe.index[~zero_mask], 'd_opt_m']
        tol_v = np.abs(net.res_pipe.loc[~zero_mask, 'v_mean_m_per_s']) - 0.2
        iteration += 1
    print(iteration)
    assert np.allclose(np.abs(net.res_pipe.loc[net.pipe.index[~zero_mask], 'v_mean_m_per_s']), 0.2, atol=0.001)