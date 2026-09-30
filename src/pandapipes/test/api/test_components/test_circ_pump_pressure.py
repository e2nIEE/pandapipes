# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
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
                                           inner_diameter_mm=102.2)
    pandapipes.create_pipe_from_parameters(net, j3, j4, k_mm=1., length_km=0.26370,
                                           inner_diameter_mm=102.2)
    pandapipes.create_circ_pump_const_pressure(net, j4, j1, 5, 2, 300, type='pt')
    pandapipes.create_heat_exchanger(net, j2, j3, qext_w=200000, inner_diameter_mm=100)
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


def _build_circ_pump_with_colocated_ext_grid(n_ext_grids):
    """A circ pump loop (j1..j4) plus an extra branch off the pump's own flow junction (j1) with
    a sink demand (1.5 kg/s) that only a real ext_grid can supply - forces MDOTSLACKINIT at j1 to
    be genuinely nonzero, so a diluted/wrong split is actually observable (with only the loop's
    own sink/source, which cancel out exactly, MDOTSLACKINIT stays at 0 regardless of any
    dilution and the bug is invisible). ``n_ext_grids`` real ext_grids are co-located directly at
    j1, the pump's own NODE_TYPE=P anchor junction."""
    net = pandapipes.create_empty_network("net", add_stdtypes=False)
    j1 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j2 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j3 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j4 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j5 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    pandapipes.create_pipe_from_parameters(net, j1, j2, k_mm=1., length_km=0.4338, inner_diameter_mm=102.2)
    pandapipes.create_pipe_from_parameters(net, j3, j4, k_mm=1., length_km=0.2637, inner_diameter_mm=102.2)
    pandapipes.create_pipe_from_parameters(net, j1, j5, k_mm=1., length_km=0.3, inner_diameter_mm=80)
    pandapipes.create_circ_pump_const_pressure(net, j4, j1, 5, 2, 300, type='pt')
    pandapipes.create_heat_exchanger(net, j2, j3, qext_w=200000, inner_diameter_mm=100)
    pandapipes.create_sink(net, j1, 2)
    pandapipes.create_source(net, j4, 2)
    pandapipes.create_sink(net, j5, 1.5)
    for _ in range(n_ext_grids):
        pandapipes.create_ext_grid(net, j1, p_bar=5, t_k=300, type="p")
    pandapipes.create_fluid_from_lib(net, "water", overwrite=True)
    return net


@pytest.mark.parametrize("n_ext_grids", [1, 2])
def test_circulation_pump_colocated_ext_grid_reports_full_share(n_ext_grids):
    """Regression test: register_circ_pump_slack_equations used to add its own +1
    Jacobian coefficient to the MDOTSLACKINIT node-balance row/column even at nodes where a real
    ext_grid is also present (COUNT_VAR_MASS_SLACK != 0) - despite its own docstring saying this case
    should be skipped. Since ExtGrid.extract_results reads MDOTSLACKINIT directly as "this
    ext_grid's own share" of an N-way split among however many components register a +1 there,
    the circ pump's extra, unneeded registration silently diluted that split to N+1, understating
    every co-located ext_grid's reported mdot_kg_per_s - without corrupting the rest of the
    hydraulic solution (pressures/branch flows elsewhere are unaffected either way, since Newton
    still balances the AGGREGATE mass injected at that node regardless of how many entities share
    credit for it - only the per-instance split was wrong)."""
    required_total_mdot = 1.5  # the sink at j5 - the only demand a real ext_grid can supply here
    net = _build_circ_pump_with_colocated_ext_grid(n_ext_grids)

    pandapipes.pipeflow(net, max_iter_hyd=25, stop_condition="tol", friction_model="nikuradse",
                        mode='hydraulics', nonlinear_method="automatic", tol_p=1e-8, tol_m=1e-8)

    assert net.converged
    # each of the n_ext_grids splits the required total evenly - the circ pump's own anchor node
    # must not count as an extra (n_ext_grids + 1)-th sharer
    expected_each = required_total_mdot / n_ext_grids
    assert np.allclose(-net.res_ext_grid.mdot_kg_per_s.values, expected_each, atol=1e-6)
    assert np.isclose(-net.res_ext_grid.mdot_kg_per_s.sum(), required_total_mdot, atol=1e-6)
