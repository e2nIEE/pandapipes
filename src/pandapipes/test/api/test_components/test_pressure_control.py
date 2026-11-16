# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import pandas as pd
import pytest

import pandapipes
from pandapipes.io.convert_format import DEFAULT_PRESS_CONTROL_DIAMETER_MM, convert_format
from pandapipes.test import data_path

@pytest.mark.parametrize("use_numba", [True, False])
def test_pressure_control_from_measurement_parameters(use_numba):
    """
        :return:
        :rtype:
        """
    net = pandapipes.create_empty_network("net", add_stdtypes=False)

    j1 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j2 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j3 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j4 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)

    pandapipes.create_pipe_from_parameters(net, j2, j3, k_mm=1., length_km=5.,
                                           inner_diameter_mm=102.2)
    pandapipes.create_pipe_from_parameters(net, j3, j4, k_mm=1., length_km=10.,
                                           inner_diameter_mm=102.2)
    pandapipes.create_pressure_control(net, j1, j2, j4, 20.)
    pandapipes.create_ext_grid(net, j1, 32, 283.15, type="p")
    pandapipes.create_sink(net, j4, 0.5)

    pandapipes.create_fluid_from_lib(net, "lgas", overwrite=True)

    max_iter_hyd = 4 if use_numba else 4
    pandapipes.pipeflow(net, stop_condition="tol", max_iter_hyd=max_iter_hyd, friction_model="nikuradse",
                        mode="hydraulics", transient=False, nonlinear_method="automatic",
                        tol_p=1e-4, tol_m=1e-4, use_numba=use_numba)

    data = pd.read_csv(os.path.join(data_path, "test_pressure_control.csv"), sep=';')

    res_junction = net.res_junction.p_bar.values
    res_pipe = net.res_pipe.v_mean_m_per_s.values

    p_diff = np.abs(1 - res_junction / data['p'].dropna().values)
    v_diff = np.abs(1 - res_pipe / data['v'].dropna().values)

    assert np.all(p_diff < 0.01)
    assert np.all(v_diff < 0.01)


def test_2pressure_controller_controllability():
    net = pandapipes.create_empty_network("net", add_stdtypes=False)

    j1 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j2 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j3 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j4 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j5 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)

    pandapipes.create_pipe_from_parameters(net, j2, j3, k_mm=1., length_km=5.,
                                           diameter_m=0.1022)
    pandapipes.create_pipe_from_parameters(net, j3, j4, k_mm=1., length_km=10.,
                                           diameter_m=0.1022)
    pandapipes.create_pressure_control(net, j4, j5, j5, 20.)
    pandapipes.create_pressure_control(net, j1, j2, j5, 20.)

    assert len(net.press_control == 1)


def test_non_working_distance_control():
    net = pandapipes.create_empty_network("net", add_stdtypes=False, fluid="hgas")

    j0 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j1 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j2 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j3 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j4 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j5 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)

    pandapipes.create_pipe_from_parameters(net, j0, j1, k_mm=1., length_km=2.,
                                           inner_diameter_mm=102.2)
    pandapipes.create_pipe_from_parameters(net, j2, j3, k_mm=1., length_km=5.,
                                           inner_diameter_mm=102.2)
    pandapipes.create_pipe_from_parameters(net, j4, j5, k_mm=1., length_km=1.,
                                           inner_diameter_mm=102.2)

    pandapipes.create_pressure_control(net, j1, j2, j5, 1.)
    assert len(net.press_control) == 0
    pandapipes.create_pressure_control(net, j1, j2, j5, 1., check_controllability=False)
    assert len(net.press_control) == 1
    pandapipes.create_ext_grid(net, j0, 12, type="p")

    with pytest.raises(UserWarning) as e:
        pandapipes.pipeflow(net)
        assert "The following controlled junction(s) were identified as disconnected" in str(e.value)


@pytest.mark.parametrize("use_numba", [True, False])
def test_pressure_control_after_index_gap(use_numba):
    """
    PressureControlComponent._register_hydraulic_equations used to look up
    control_active/in_service/controlled_junction/controlled_p_bar via
    net[table_name].values[tbl_idx], where tbl_idx came from IdxBranch.ELEMENT_IDX - the
    pandas *index label* of the press_control row, not its position in net.press_control.
    .values is positional, so as soon as the table's index isn't 0..n-1 anymore (e.g.
    after dropping a press_control and adding a replacement, since pandas keeps counting
    new row labels upward instead of reusing the freed one), the lookup goes out of
    bounds or silently picks up a different press_control's set point.
    """
    net = pandapipes.create_empty_network("net", add_stdtypes=False, fluid="lgas")

    j0, j1, j2, j3, j4 = [pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
                          for _ in range(5)]

    pandapipes.create_ext_grid(net, j0, 32, 283.15, type="p")

    pc_a = pandapipes.create_pressure_control(net, j0, j1, j1, 20.)
    pandapipes.create_pipe_from_parameters(net, j1, j2, k_mm=1., length_km=5.,
                                           inner_diameter_mm=102.2)
    pandapipes.create_sink(net, j2, 0.5)

    pandapipes.create_pressure_control(net, j0, j3, j3, 15.)
    pandapipes.create_pipe_from_parameters(net, j3, j4, k_mm=1., length_km=5.,
                                           inner_diameter_mm=102.2)
    pandapipes.create_sink(net, j4, 0.3)

    # same 2 press_controls/topology/set points as above, just re-labeled: drop the first
    # press_control and add an equivalent replacement -> index becomes [1, 2] instead of
    # [0, 1]
    net.press_control.drop(index=[pc_a], inplace=True)
    pandapipes.create_pressure_control(net, j0, j1, j1, 20.)
    assert net.press_control.index.tolist() == [1, 2]

    max_iter_hyd = 4 if use_numba else 4
    pandapipes.pipeflow(net, max_iter_hyd=max_iter_hyd, use_numba=use_numba)

    assert np.isclose(net.res_junction.at[j1, "p_bar"], 20.)
    assert np.isclose(net.res_junction.at[j3, "p_bar"], 15.)


@pytest.mark.parametrize("use_numba", [True, False])
@pytest.mark.parametrize("control_active", [True, False])
@pytest.mark.parametrize("inner_diameter_mm", [50., 100.])
def test_pressure_control_diameter_and_loss(use_numba, control_active, inner_diameter_mm):
    """The inner diameter is the reference area of the loss coefficient.

    While the control is active it has no influence at all - the set point alone determines the
    outlet pressure. Once the control is switched off, the pressure drop is the one of a lumped
    loss over that diameter, which is what pins the diameter down to a specific value here: two
    different diameters have to give two different results.
    """
    net = pandapipes.create_empty_network("net", add_stdtypes=False, fluid="water")
    t_k, p_in_bar, p_set_bar, mdot, loss_coefficient = 293.15, 5., 4., 2., 10.

    j0, j1 = [pandapipes.create_junction(net, pn_bar=p_in_bar, tfluid_k=t_k) for _ in range(2)]
    pandapipes.create_ext_grid(net, j0, p_in_bar, t_k, type="p")
    pandapipes.create_sink(net, j1, mdot)
    pandapipes.create_pressure_control(net, j0, j1, j1, p_set_bar,
                                       control_active=control_active,
                                       loss_coefficient=loss_coefficient,
                                       inner_diameter_mm=inner_diameter_mm)

    pandapipes.pipeflow(net, mode="hydraulics", use_numba=use_numba)

    if control_active:
        expected_p_bar = p_set_bar
    else:
        rho = net.fluid.get_density(t_k)
        area_m2 = np.pi * (inner_diameter_mm / 1000.) ** 2 / 4
        velocity = mdot / (rho * area_m2)
        expected_p_bar = p_in_bar - loss_coefficient * rho * velocity ** 2 / (2 * 1e5)

    assert net.res_junction.at[j1, "p_bar"] == pytest.approx(expected_p_bar, abs=1e-7)


def test_pressure_control_diameter_from_old_format():
    """Nets stored before the diameter existed get the documented 100 mm fallback."""
    net = pandapipes.create_empty_network("net", add_stdtypes=False, fluid="water")

    j0, j1 = [pandapipes.create_junction(net, pn_bar=5., tfluid_k=293.15) for _ in range(2)]
    pandapipes.create_pressure_control(net, j0, j1, j1, 4., inner_diameter_mm=42.)

    net.press_control.drop(columns="inner_diameter_mm", inplace=True)
    net.format_version = "0.14.0"
    convert_format(net)

    assert net.press_control.at[0, "inner_diameter_mm"] == DEFAULT_PRESS_CONTROL_DIAMETER_MM
