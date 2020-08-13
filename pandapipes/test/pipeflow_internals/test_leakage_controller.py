# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import pandapipes
import pytest
from pandapipes.control import LeakageController, run_control


def test_one_pipe_one_leakage():
    net = pandapipes.create_empty_network("net", fluid="water", add_stdtypes=True)

    j0 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=293.15)
    j1 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=293.15)

    pandapipes.create_ext_grid(net, j0, p_bar=5, t_k=293.15, type="pt")

    pandapipes.create_sink(net, j1, mdot_kg_per_s=1)

    pandapipes.create_pipe_from_parameters(net, j0, j1, diameter_m=0.75, k_mm=0.1, length_km=2)

    kwargs = {'stop_condition': 'tol', 'iter': 100, 'tol_p': 1e-7, 'tol_v': 1e-7, 'friction_model': 'colebrook',
              'mode': 'hydraulics', 'only_update_hydraulic_matrix': False}

    LeakageController(net, element='pipe', element_index=0, output_area_m2=1, **kwargs)

    run_control(net)


def test_two_pipes_two_leakages():
    net = pandapipes.create_empty_network("net", fluid="water", add_stdtypes=True)

    j0 = pandapipes.create_junction(net, pn_bar=3, tfluid_k=293.15)
    j1 = pandapipes.create_junction(net, pn_bar=3, tfluid_k=293.15)
    j2 = pandapipes.create_junction(net, pn_bar=3, tfluid_k=293.15)

    pandapipes.create_ext_grid(net, j0, p_bar=3, t_k=293.15, type="pt")

    pandapipes.create_sink(net, j1, mdot_kg_per_s=1)
    pandapipes.create_sink(net, j2, mdot_kg_per_s=0.5)

    pandapipes.create_pipe_from_parameters(net, j0, j1, diameter_m=0.75, k_mm=0.1, length_km=2)
    pandapipes.create_pipe_from_parameters(net, j1, j2, diameter_m=0.6, k_mm=0.1, length_km=3)

    kwargs = {'stop_condition': 'tol', 'iter': 100, 'tol_p': 1e-7, 'tol_v': 1e-7, 'friction_model': 'colebrook',
              'mode': 'hydraulics', 'only_update_hydraulic_matrix': False}

    LeakageController(net, element='pipe', element_index=[0, 1], output_area_m2=[1, 2.5], **kwargs)

    run_control(net)


def test_one_valve_one_leakage():
    net = pandapipes.create_empty_network("net", fluid="water", add_stdtypes=True)

    j0 = pandapipes.create_junction(net, pn_bar=2, tfluid_k=293.15)
    j1 = pandapipes.create_junction(net, pn_bar=2, tfluid_k=293.15)

    pandapipes.create_ext_grid(net, j0, p_bar=2, t_k=293.15, type="pt")

    pandapipes.create_sink(net, j1, mdot_kg_per_s=0.5)

    pandapipes.create_valve(net, j0, j1, diameter_m=0.1, opened=True)

    kwargs = {'stop_condition': 'tol', 'iter': 100, 'tol_p': 1e-7, 'tol_v': 1e-7, 'friction_model': 'nikuradse',
              'mode': 'hydraulics', 'only_update_hydraulic_matrix': False}

    LeakageController(net, element='valve', element_index=0, output_area_m2=0.5, **kwargs)

    run_control(net)


def test_one_heat_exchanger_one_leakage():
    net = pandapipes.create_empty_network("net", fluid="water", add_stdtypes=True)

    j0 = pandapipes.create_junction(net, pn_bar=1, tfluid_k=293.15)
    j1 = pandapipes.create_junction(net, pn_bar=1, tfluid_k=293.15)

    pandapipes.create_ext_grid(net, j0, p_bar=1, t_k=350, type="pt")

    pandapipes.create_sink(net, j1, mdot_kg_per_s=0.5)

    pandapipes.create_heat_exchanger(net, j0, j1, diameter_m=0.8, qext_w=20000)

    kwargs = {'stop_condition': 'tol', 'iter': 100, 'tol_p': 1e-7, 'tol_v': 1e-7, 'friction_model': 'colebrook',
              'mode': 'all', 'only_update_hydraulic_matrix': False}

    LeakageController(net, element='heat_exchanger', element_index=0, output_area_m2=0.1, **kwargs)

    run_control(net)


if __name__ == "__main__":
    pytest.main([r'pandapipes/test/pipeflow_internals/test_leakage_controller.py'])