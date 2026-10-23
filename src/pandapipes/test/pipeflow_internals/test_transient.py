# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import pytest

import pandapipes
from pandapipes.idx_node import IdxNode
from pandapipes.pf.pipeflow_setup import PipeflowNotConverged, is_transient_warm_start

T_INIT = 333.15
T_SUPPLY = 363.15
T_RETURN = 313.15


def _run_transient_steps(net, steps, dt, callback=None, **kwargs):
    """Run consecutive transient pipeflows the same way run_timeseries does."""
    for step in steps:
        pandapipes.pipeflow(net, transient=True, dt=dt, simulation_time_step=step, **kwargs)
        if callback is not None:
            callback(net, step)


def _heating_loop():
    """Closed district heating loop whose junctions start colder than the supply temperature."""
    net = pandapipes.create_empty_network(fluid="water")
    j_flow, j_cons_in, j_cons_out, j_return = [
        pandapipes.create_junction(net, pn_bar=5, tfluid_k=T_INIT) for _ in range(4)]
    pandapipes.create_pipe_from_parameters(net, j_flow, j_cons_in, length_km=0.2,
                                           inner_diameter_mm=80, k_mm=0.05, sections=10,
                                           u_w_per_m2k=0., text_k=283.15)
    pandapipes.create_heat_consumer(net, j_cons_in, j_cons_out, qext_w=200e3, treturn_k=T_RETURN)
    pandapipes.create_pipe_from_parameters(net, j_cons_out, j_return, length_km=0.5,
                                           inner_diameter_mm=80, k_mm=0.05, sections=10,
                                           u_w_per_m2k=0., text_k=283.15)
    pandapipes.create_circ_pump_const_pressure(net, j_return, j_flow, p_flow_bar=5, plift_bar=2,
                                               t_flow_k=T_SUPPLY, type="pt")
    return net, j_cons_in


@pytest.mark.parametrize("use_numba", [True, False])
def test_transient_hydraulics_follow_simulated_temperatures(use_numba):
    """In sequential mode, each step's hydraulics must use the previous step's temperatures.

    The consumer's mass flow mdot = Q / (cp * (T_in - T_return)) has to follow the warming supply
    temperature (lagging by one step) instead of staying at the value of the initial
    junction temperatures.
    """
    net, j_cons_in = _heating_loop()
    t_in, mdot = [], []

    def collect(net, _):
        t_in.append(net.res_junction.t_k.at[j_cons_in])
        mdot.append(net.res_heat_consumer.mdot_from_kg_per_s.iat[0])

    _run_transient_steps(net, range(80), dt=60, callback=collect, mode="sequential",
                         use_numba=use_numba)

    fluid = net.fluid
    cp = (fluid.get_heat_capacity(t_in[-2]) + fluid.get_heat_capacity(T_RETURN)) / 2
    expected_mdot = 200e3 / (cp * (t_in[-2] - T_RETURN))
    assert t_in[-1] == pytest.approx(T_SUPPLY, abs=0.1)
    assert mdot[-1] == pytest.approx(expected_mdot, rel=1e-3)
    assert mdot[-1] < 0.5 * mdot[0]


@pytest.mark.parametrize("use_numba", [True, False])
def test_transient_front_travel_time(use_numba):
    """A temperature step has to reach the end of a loss-free pipe after length / velocity."""
    length_m, d_m, velocity = 2000., 0.1, 1.
    dt = 50
    net = pandapipes.create_empty_network(fluid="water")
    j_in = pandapipes.create_junction(net, pn_bar=5, tfluid_k=T_INIT)
    j_out = pandapipes.create_junction(net, pn_bar=5, tfluid_k=T_INIT)
    pandapipes.create_ext_grid(net, j_in, p_bar=5, t_k=T_SUPPLY, type="pt")
    pandapipes.create_pipe_from_parameters(net, j_in, j_out, length_km=length_m / 1000,
                                           inner_diameter_mm=d_m * 1000, k_mm=0.05,
                                           sections=40, u_w_per_m2k=0., text_k=283.15)
    mdot = net.fluid.get_density(T_INIT) * np.pi * d_m ** 2 / 4 * velocity
    pandapipes.create_sink(net, j_out, mdot_kg_per_s=mdot)

    t_out = []
    _run_transient_steps(net, range(100), dt=dt, mode="sequential", use_numba=use_numba,
                         callback=lambda net, _: t_out.append(net.res_junction.t_k.at[j_out]))

    # step 0 already advances the initial state by dt
    times = (np.arange(len(t_out)) + 1) * dt
    t_out = np.array(t_out)
    t_half = times[np.argmax(t_out >= (T_INIT + T_SUPPLY) / 2)]
    assert t_half == pytest.approx(length_m / velocity, rel=0.05)
    assert t_out[0] == pytest.approx(T_INIT, abs=0.01)
    assert t_out[-1] == pytest.approx(T_SUPPLY, abs=0.1)


def test_transient_state_kept_after_divergence():
    """A non-converged step must not reset the thermal state to the initial junction values."""
    net, _ = _heating_loop()
    _run_transient_steps(net, range(5), dt=60, mode="sequential")
    t_before_failure = net["_pit"]["node"][:, IdxNode.TINIT].copy()
    assert not np.allclose(t_before_failure, T_INIT)

    with pytest.raises(PipeflowNotConverged):
        pandapipes.pipeflow(net, mode="sequential", transient=True, dt=60, simulation_time_step=5,
                            max_iter_therm=1)

    pandapipes.pipeflow(net, mode="sequential", transient=True, dt=60, simulation_time_step=6)
    assert net.converged
    t_old_col = net["_lookups"]["node_old_pit_cols"][IdxNode.TINIT]
    assert np.allclose(net["_old_pit"]["node"][:, t_old_col], t_before_failure)


@pytest.mark.parametrize("transient, step, converged, expected", [
    (True, 3, True, True),
    (True, 0, True, False),
    (True, 3, False, False),
    (False, 3, True, False),
])
def test_is_transient_warm_start(transient, step, converged, expected):
    net = pandapipes.create_empty_network(fluid="water")
    net["_options"] = {"transient": transient, "simulation_time_step": step}
    net.converged = converged
    assert is_transient_warm_start(net) is expected


if __name__ == "__main__":
    pytest.main([__file__])
