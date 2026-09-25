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
from pandapipes.idx_branch import IdxBranch
from pandapipes.idx_node import IdxNode

from pandapipes.properties import get_fluid
from pandapipes.test import data_path


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

    p_pandapipes = node_pit[:, IdxNode.PINIT]
    fluid = get_fluid(net)
    v_pandapipes = branch_pit[:, IdxBranch.MDOTINIT] / (np.pi * (branch_pit[:, IdxBranch.D] / 2) ** 2) / fluid.get_density(NORMAL_TEMPERATURE)

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

    max_iter_therm = 4 if use_numba else 4
    pandapipes.pipeflow(ntw, max_iter_therm=max_iter_therm,
                        stop_condition="tol", friction_model="nikuradse",
                        nonlinear_method="automatic", mode="heat", use_numba=use_numba)

    temp_net = net.res_junction.t_k
    temp_ntw = ntw.res_junction.t_k

    temp_diff = np.abs(1 - temp_net / temp_ntw)

    assert np.all(temp_diff < 0.01)


def test_bidirectional_automatic_damping_no_crash():
    """Regression test: BidirectionalCalculation.VARS/TOLS/PITS/COLS used to list only 4 entries
    (mdot, p, TOUT, T) while solve_bidirectional() actually returns 5 variables' worth of data, in
    order (mdot, p, mdotslack, Tout, T) - Calculation.run()'s positional un-interleaving then
    silently paired 'TOUT' with mdotslack's values/branch pit/TOUTINIT column and 'T' with Tout's
    values, dropping the real T pair entirely.

    MDOTSLACKINIT (the residual mass each ext_grid absorbs) only actually changes between
    iterations - and so only actually triggers automatic damping's error-increased check for that
    mismatched slot - with more than one ext_grid to balance mass flow between; a single-ext_grid
    net never exercises this path since mdotslack then stays at/near 0 throughout. With two
    ext_grids and nonlinear_method="automatic", a damping-fallback write for that mismatched
    'TOUT' slot instead wrote mdotslack's (small, node-range) index/shape into the branch pit,
    raising IndexError or ValueError (shape mismatch) depending on how the node/branch pit sizes
    happened to compare on the network at hand - both symptoms of the same misalignment."""
    net = pandapipes.create_empty_network("net", add_stdtypes=False, fluid="water")
    j = pandapipes.create_junctions(net, 6, pn_bar=5, tfluid_k=300)
    for a, b in [(0, 1), (1, 4), (4, 5)]:
        pandapipes.create_pipe_from_parameters(net, j[a], j[b], length_km=1, inner_diameter_mm=80)
    for eg in [0, 5]:
        pandapipes.create_ext_grid(net, j[eg], p_bar=5, t_k=300, type="pt")
    for s in [1, 4]:
        pandapipes.create_sink(net, j[s], mdot_kg_per_s=1)

    pandapipes.pipeflow(net, mode="bidirectional", nonlinear_method="automatic",
                        max_iter_bidirect=20)

    assert net.converged
