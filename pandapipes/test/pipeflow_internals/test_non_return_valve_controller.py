# Copyright (c) 2020 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import pandapipes
import pytest
from pandapipes.control import NonReturnValveController, run_control


def test_nrv():
    net = pandapipes.create_empty_network("net", fluid="water", add_stdtypes=True)

    j0 = pandapipes.create_junction(net, pn_bar=3, tfluid_k=293.15)
    j1 = pandapipes.create_junction(net, pn_bar=3, tfluid_k=293.15)
    j2 = pandapipes.create_junction(net, pn_bar=3, tfluid_k=293.15)
    j3 = pandapipes.create_junction(net, pn_bar=3, tfluid_k=283.15)
    j4 = pandapipes.create_junction(net, pn_bar=3, tfluid_k=283.15)

    pandapipes.create_ext_grid(net, j0, p_bar=3, t_k=293.15, type="pt")

    pandapipes.create_sink(net, j1, mdot_kg_per_s=1)
    pandapipes.create_sink(net, j4, mdot_kg_per_s=0.5)

    pandapipes.create_source(net, j2, mdot_kg_per_s=0.05)
    pandapipes.create_source(net, j3, mdot_kg_per_s=1)

    pandapipes.create_pipe_from_parameters(net, j1, j0, diameter_m=0.75, k_mm=0.1, length_km=15)
    pandapipes.create_pipe_from_parameters(net, j4, j2, diameter_m=0.1, k_mm=0.1, length_km=10)

    pandapipes.create_valve(net, j4, j3, diameter_m=0.1, opened=True)
    pandapipes.create_valve(net, j1, j3, diameter_m=0.07, opened=True)
    pandapipes.create_valve(net, j1, j2, diameter_m=0.05, opened=True)
    pandapipes.create_valve(net, j3, j0, diameter_m=0.01, opened=True)

    kwargs = {'stop_condition': 'tol', 'iter': 100, 'tol_p': 1e-7, 'tol_v': 1e-7, 'friction_model': 'colebrook',
              'mode': 'hydraulics', 'only_update_hydraulic_matrix': False}

    NonReturnValveController(net, element_index=[1, 3], **kwargs)

    run_control(net)


if __name__ == "__main__":
    pytest.main([r'pandapipes/test/pipeflow_internals/test_non_return_valve_controller.py'])
