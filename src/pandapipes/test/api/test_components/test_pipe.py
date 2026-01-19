# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
import numpy as np
import pytest

import pandapipes
from pandapipes.properties.fluids import _add_fluid_to_net
import copy


@pytest.mark.parametrize("use_numba", [True, False])
def test_pipe_velocity_results(use_numba):
    """
        This test verifies the entries in the result table for a pipe network with pipes consisting
        of more than one section. The basic idea is that the computation is first done with only one
        section per pipe. Afterwards, the same network is calculated with more nodes. if everything
        works correctly, the entries in the result table for the velocities (from, to) should be the
        same.

        A T-junction is used for the test setup
        :return:
        :rtype:
        """
    net = pandapipes.create_empty_network("net", add_stdtypes=False)
    d = 209.1e-3
    pandapipes.create_junction(net, pn_bar=51, tfluid_k=285.15)
    pandapipes.create_junction(net, pn_bar=51, tfluid_k=285.15)
    pandapipes.create_junction(net, pn_bar=51, tfluid_k=285.15)
    pandapipes.create_junction(net, pn_bar=51, tfluid_k=285.15)
    pandapipes.create_pipe_from_parameters(net, 0, 1, 6.0, d, k_mm=.5, sections=1)
    pandapipes.create_pipe_from_parameters(net, 1, 2, 6.0, d, k_mm=.5, sections=1)
    pandapipes.create_pipe_from_parameters(net, 1, 3, 6.0, d, k_mm=.5, sections=1)
    pandapipes.create_ext_grid(net, 0, p_bar=51 - 1.01325, t_k=285.15, type="pt")
    pandapipes.create_sink(net, 2, mdot_kg_per_s=0.82752 * 45000 / 3600 / 3)
    pandapipes.create_sink(net, 3, mdot_kg_per_s=0.82752 * 45000 / 3600 / 2)
    _add_fluid_to_net(net, pandapipes.create_constant_fluid(
        name="natural_gas", fluid_type="gas", viscosity=11.93e-6, heat_capacity=2185,
        compressibility=1, der_compressibility=0, density=0.82752
    ))
    max_iter_hyd = 5 if use_numba else 5
    pandapipes.pipeflow(
        net, stop_condition="tol", max_iter_hyd=max_iter_hyd, friction_model="nikuradse",
        transient=False, nonlinear_method="automatic", tol_p=1e-5, tol_m=1e-5, use_numba=use_numba
    )

    v_1_sec_from = net.res_pipe.v_from_m_per_s
    v_1_sec_to = net.res_pipe.v_from_m_per_s

    net = pandapipes.create_empty_network("net", add_stdtypes=False)
    d = 209.1e-3
    pandapipes.create_junction(net, pn_bar=51, tfluid_k=285.15)
    pandapipes.create_junction(net, pn_bar=51, tfluid_k=285.15)
    pandapipes.create_junction(net, pn_bar=51, tfluid_k=285.15)
    pandapipes.create_junction(net, pn_bar=51, tfluid_k=285.15)
    pandapipes.create_pipe_from_parameters(net, 0, 1, 6.0, d, k_mm=.5, sections=3)
    pandapipes.create_pipe_from_parameters(net, 1, 2, 6.0, d, k_mm=.5, sections=4)
    pandapipes.create_pipe_from_parameters(net, 1, 3, 6.0, d, k_mm=.5, sections=2)
    pandapipes.create_ext_grid(net, 0, p_bar=51 - 1.01325, t_k=285.15, type="pt")
    pandapipes.create_sink(net, 2, mdot_kg_per_s=0.82752 * 45000 / 3600 / 3)
    pandapipes.create_sink(net, 3, mdot_kg_per_s=0.82752 * 45000 / 3600 / 2)
    _add_fluid_to_net(net, pandapipes.create_constant_fluid(
        name="natural_gas", fluid_type="gas", viscosity=11.93e-6, heat_capacity=2185,
        compressibility=1, der_compressibility=0, density=0.82752
    ))
    pandapipes.pipeflow(net, stop_condition="tol", max_iter_hyd=max_iter_hyd,
                        friction_model="nikuradse",
                        transient=False, nonlinear_method="automatic", tol_p=1e-5, tol_m=1e-5,
                        use_numba=use_numba)

    v_n_sec_from = net.res_pipe.v_from_m_per_s
    v_n_sec_to = net.res_pipe.v_from_m_per_s

    diff_from = v_1_sec_from - v_n_sec_from
    diff_to = v_1_sec_to - v_n_sec_to

    assert np.all(np.abs(diff_from) < 1e-9)
    assert np.all(np.abs(diff_to) < 1e-9)



@pytest.mark.parametrize("use_numba", [True, False])
def test_pipe_sections_temperature(use_numba):
    """
        This test verifies the results of the temperatures for a pipe network with pipes consisting
        of more than one section. The basic idea is that the computation is first done with only one
        section per pipe. Afterwards, the same network is calculated with more nodes. if everything
        works correctly, the entries in the result table for the velocities (from, to) should be the
        same.

        A T-junction is used for the test setup
        :return:
        :rtype:
        """
    net = pandapipes.create_empty_network("net", add_stdtypes=False)
    d = 209.1e-3
    pandapipes.create_junction(net, pn_bar=5, tfluid_k=285.15)
    pandapipes.create_junction(net, pn_bar=5, tfluid_k=285.15)
    pandapipes.create_pipe_from_parameters(net, 0, 1, 6.0, d, k_mm=.5, sections=1, u_w_per_m2k=5, text_k=285.15)
    pandapipes.create_ext_grid(net, 0, p_bar=5, t_k=385.15, type="pt")
    pandapipes.create_sink(net, 1, mdot_kg_per_s=2)
    pandapipes.create_fluid_from_lib(net, "water")

    net2 = copy.deepcopy(net)
    net2.pipe.sections = 100

    max_iter_hyd = 3 if use_numba else 3
    max_iter_therm = 3 if use_numba else 3
    pf_args = {
        "mode": "sequential",
        "max_iter_hyd": max_iter_hyd,
        "max_iter_therm": max_iter_therm,
        "use_numba": use_numba
    }
    pandapipes.pipeflow(net, **pf_args)
    pandapipes.pipeflow(net2, **pf_args)

    assert np.allclose(
        net.res_junction.p_bar.to_numpy(), net2.res_junction.p_bar.to_numpy(), rtol=5e-4, atol=1e-5
    )
    assert np.allclose(
        net.res_junction.t_k.to_numpy(), net2.res_junction.t_k.to_numpy(), rtol=5e-4, atol=1e-5
    )


@pytest.fixture
def create_net_3_juncs():
    net = pandapipes.create_empty_network()
    pandapipes.create_junctions(net, 3, 3, 273)
    return net


def test_namechange_pipe_from_parameters(create_net_3_juncs):
    net = copy.deepcopy(create_net_3_juncs)
    length_km = 1
    diameter_m = 0.01
    alpha = 5
    with pytest.warns(DeprecationWarning):
        pandapipes.create_pipe_from_parameters(net, 0, 1, length_km, diameter_m,
                                               alpha_w_per_m2k=alpha)
        assert net.pipe.u_w_per_m2k.values == alpha


def test_namechange_pipes_from_parameters(create_net_3_juncs):
    net = copy.deepcopy(create_net_3_juncs)
    length_km = 1
    diameter_m = 0.01
    alpha = [5, 3]
    with pytest.warns(DeprecationWarning):
        pandapipes.create_pipes_from_parameters(net, [0, 1], [1, 2], length_km, diameter_m,
                                                alpha_w_per_m2k=alpha)
        assert net.pipe.u_w_per_m2k.values.tolist() == alpha
    net = copy.deepcopy(create_net_3_juncs)
    u = [1, 7]
    with pytest.warns(DeprecationWarning):
        pandapipes.create_pipes_from_parameters(net, [0, 1], [1, 2], length_km, diameter_m,
                                                alpha_w_per_m2k=alpha, u_w_per_m2k=np.array(u))
        assert net.pipe.u_w_per_m2k.values.tolist() == u


@pytest.mark.parametrize("use_numba", [True, False])
def test_zero_flow_pipes(use_numba):
    net = pandapipes.create_empty_network("net", add_stdtypes=False)
    d = 209.1e-3
    pandapipes.create_junction(net, pn_bar=5, tfluid_k=285.15)
    pandapipes.create_junction(net, pn_bar=5, tfluid_k=285.15)
    pandapipes.create_junction(net, pn_bar=5, tfluid_k=285.15)
    pandapipes.create_pipe_from_parameters(net, 0, 1, 6.0, d, k_mm=.5, sections=1, u_w_per_m2k=5, text_k=285.15)
    pandapipes.create_pipe_from_parameters(net, 1, 2, 6.0, d, k_mm=.5, sections=1, u_w_per_m2k=5, text_k=285.15)
    pandapipes.create_ext_grid(net, 0, p_bar=5, t_k=385.15, type="pt")
    pandapipes.create_sink(net, 1, mdot_kg_per_s=2)
    pandapipes.create_fluid_from_lib(net, "water")
    pandapipes.pipeflow(net, mode='bidirectional', iter=5, use_numba=use_numba)
    assert net.res_pipe.t_outlet_k.at[1] == 293.15
    assert net.res_junction.t_k.at[2] == 293.15