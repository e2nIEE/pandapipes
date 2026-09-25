# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

"""
``net.component_list`` normally holds every registered component class (see
``pandapipes_net.get_basic_all_components``), regardless of whether the net actually uses that
component - a net with no valves still carries ``Valve`` in its component_list, its
``register_hydraulic_equations``/``create_pit_branch_entries``/etc. just iterate over zero rows
and contribute nothing.

This checks that assumption holds: for a variety of networks exercising most component types
(pipe, valve, pump, compressor, flow control, pressure control, mass storage, heat exchanger,
heat consumer, both circulation pump variants), pruning component_list down to only the
components that actually have rows in that net must not change a single computed result compared
to running with the full, default component_list.
"""

import copy

import pandas as pd
import pytest

import pandapipes
import pandapipes.networks.simple_gas_networks as gas_nw
import pandapipes.networks.simple_water_networks as water_nw


def _used_component_list(net):
    """Every component in ``net.component_list`` that actually has rows in this net."""
    return [comp for comp in net.component_list if len(net[comp.table_name()]) > 0]


def _assert_results_match(net_full, net_pruned):
    """Compare every result table that has rows in ``net_full`` against ``net_pruned`` - tables
    with zero rows are skipped rather than compared, since an unused component's result table may
    not even exist in the pruned net (it's only ever created once that component's own
    ``extract_results`` runs) while the full net still carries an empty placeholder for it; that
    difference is a bookkeeping artifact, not a result difference."""
    res_tables = [
        name for name, table in net_full.items()
        if name.startswith("res_") and isinstance(table, pd.DataFrame) and len(table) > 0
    ]
    assert res_tables, "the full run produced no result rows at all - nothing to compare"
    for name in res_tables:
        assert name in net_pruned, f"{name} has rows in the full run but doesn't exist at all " \
                                   f"in the pruned-component-list run"
        pd.testing.assert_frame_equal(net_full[name], net_pruned[name], check_exact=False,
                                      atol=1e-9, rtol=1e-9)


def _run_full_vs_pruned(net, pipeflow_kwargs):
    net_full = copy.deepcopy(net)
    net_pruned = copy.deepcopy(net)
    net_pruned.component_list = _used_component_list(net_pruned)
    # every net here is built with the full default component_list, so pruning must actually
    # remove something, or the test isn't exercising anything
    assert len(net_pruned.component_list) < len(net_full.component_list)

    pandapipes.pipeflow(net_full, **pipeflow_kwargs)
    pandapipes.pipeflow(net_pruned, **pipeflow_kwargs)

    assert net_full.converged
    assert net_pruned.converged
    _assert_results_match(net_full, net_pruned)


def _net_compressor():
    """Junction, Pipe, ExtGrid, Sink, Compressor (with one reverse-flow bypass compressor)."""
    net = pandapipes.create_empty_network("net", add_stdtypes=True, fluid="hgas")
    j1, j2, j3, j4, j5, j6 = pandapipes.create_junctions(net, 6, pn_bar=5, tfluid_k=283.15)
    pandapipes.create_pipe_from_parameters(net, j1, j2, length_km=0.4338, inner_diameter_mm=102.2)
    pandapipes.create_pipe_from_parameters(net, j3, j4, length_km=0.2637, inner_diameter_mm=102.2)
    pandapipes.create_ext_grid(net, j1, 5, 283.15, type="p")
    pandapipes.create_sink(net, j6, 0.02333)
    pandapipes.create_compressor(net, j2, j3, pressure_ratio=1.5)
    pandapipes.create_compressor(net, j5, j4, pressure_ratio=1.5)
    pandapipes.create_compressor(net, j5, j6, pressure_ratio=1.1)
    return net, dict(max_iter_hyd=10)


def _net_flow_control_heat_exchanger():
    """Junction, Pipe, ExtGrid, Sink, HeatExchanger, FlowControlComponent (one active, one not)."""
    net = pandapipes.create_empty_network("net", add_stdtypes=True, fluid="water")
    j1, j2, j3, j4, j5, j6, j7, j8 = pandapipes.create_junctions(net, 8, pn_bar=5, tfluid_k=360)
    pandapipes.create_pipes_from_parameters(
        net, [j1, j2, j4, j7], [j2, j5, j8, j4], 0.2, 100, k_mm=0.1, u_w_per_m2k=20., text_k=280)
    pandapipes.create_heat_exchanger(net, j3, j4, 0.1, 50000, 1)
    pandapipes.create_heat_exchanger(net, j6, j7, 0.1, 50000, 1)
    pandapipes.create_flow_control(net, j2, j3, 2)
    pandapipes.create_flow_control(net, j5, j6, 2, control_active=False)
    pandapipes.create_ext_grid(net, j1, p_bar=5, t_k=360, type="pt")
    pandapipes.create_sink(net, j8, 3)
    return net, dict(max_iter_hyd=10, max_iter_therm=10, mode='sequential')


def _net_pressure_control():
    """Junction, Pipe, ExtGrid, Sink, PressureControlComponent."""
    net = pandapipes.create_empty_network("net", add_stdtypes=False)
    j1 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j2 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j3 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j4 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    pandapipes.create_pipe_from_parameters(net, j2, j3, k_mm=1., length_km=5., inner_diameter_mm=102.2)
    pandapipes.create_pipe_from_parameters(net, j3, j4, k_mm=1., length_km=10., inner_diameter_mm=102.2)
    pandapipes.create_pressure_control(net, j1, j2, j4, 20.)
    pandapipes.create_ext_grid(net, j1, 32, 283.15, type="p")
    pandapipes.create_sink(net, j4, 0.5)
    pandapipes.create_fluid_from_lib(net, "lgas", overwrite=True)
    return net, dict(stop_condition="tol", max_iter_hyd=10, friction_model="nikuradse",
                     mode="hydraulics", transient=False, nonlinear_method="automatic",
                     tol_p=1e-4, tol_m=1e-4)


def _net_mass_storage():
    """Junction, Pipe, ExtGrid, MassStorage (one charging, one discharging)."""
    net = pandapipes.create_empty_network("net", add_stdtypes=True, fluid="water")
    j1, j2, j3 = pandapipes.create_junctions(net, 3, pn_bar=2, tfluid_k=283.15)
    pandapipes.create_pipe_from_parameters(net, j1, j2, length_km=1, diameter_m=0.5)
    pandapipes.create_pipe_from_parameters(net, j2, j3, length_km=1, diameter_m=0.5)
    pandapipes.create_ext_grid(net, j1, 2, 283.15, type="p")
    pandapipes.create_mass_storage(net, j2, 0.1)
    pandapipes.create_mass_storage(net, j3, -0.2)
    return net, dict(max_iter_hyd=10)


def _net_circ_pump_mass():
    """Junction, Pipe, CirculationPumpMass, HeatExchanger, Sink, Source."""
    net = pandapipes.create_empty_network("net", add_stdtypes=False)
    j1 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j2 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j3 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    j4 = pandapipes.create_junction(net, pn_bar=5, tfluid_k=283.15)
    pandapipes.create_pipe_from_parameters(net, j1, j2, k_mm=1., length_km=0.4338, inner_diameter_mm=102.2)
    pandapipes.create_pipe_from_parameters(net, j3, j4, k_mm=1., length_km=0.2637, inner_diameter_mm=102.2)
    pandapipes.create_circ_pump_const_mass_flow(net, j4, j1, 5, 5, 300, type='pt')
    pandapipes.create_heat_exchanger(net, j2, j3, qext_w=200000, inner_diameter_mm=100)
    pandapipes.create_sink(net, j1, 2)
    pandapipes.create_source(net, j4, 2)
    pandapipes.create_fluid_from_lib(net, "water", overwrite=True)
    return net, dict(max_iter_hyd=10, max_iter_therm=10, stop_condition="tol",
                     friction_model="nikuradse", mode='sequential', transient=False,
                     nonlinear_method="automatic", tol_p=1e-4, tol_m=1e-4)


def _net_heat_consumer_circ_pump_pressure():
    """Junction, Pipe, CirculationPumpPressure, HeatConsumer (two, on a flow/return loop)."""
    net = pandapipes.create_empty_network("net", add_stdtypes=False, fluid="water")
    juncs = pandapipes.create_junctions(net, 6, pn_bar=5, tfluid_k=283.15,
                                        system=["flow"] * 3 + ["return"] * 3)
    pandapipes.create_pipes_from_parameters(
        net, juncs[[0, 1, 3, 4]], juncs[[1, 2, 4, 5]], k_mm=0.1, length_km=1,
        inner_diameter_mm=102.2, system=["flow"] * 2 + ["return"] * 2, u_w_per_m2k=10, text_k=273.15)
    pandapipes.create_circ_pump_const_pressure(net, juncs[-1], juncs[0], 5, 2, 400, type='pt')
    pandapipes.create_heat_consumer(net, juncs[1], juncs[4], controlled_mdot_kg_per_s=3, qext_w=150000)
    pandapipes.create_heat_consumer(net, juncs[2], juncs[3], controlled_mdot_kg_per_s=2, qext_w=75000)
    return net, dict(mode='sequential')


NETWORK_BUILDERS = {
    # bundled STANET/example networks - real, previously-validated topologies
    "water_district_grid": lambda: (water_nw.water_district_grid(),
                                    dict(mode='hydraulics', max_iter_hyd=20)),
    "water_meshed_2valves": lambda: (water_nw.water_meshed_2valves(results_from="stanet"),
                                     dict(mode='hydraulics', max_iter_hyd=20)),
    "gas_meshed_pumps": lambda: (gas_nw.gas_meshed_pumps(), dict(mode='hydraulics', max_iter_hyd=20)),
    "gas_versatility": lambda: (gas_nw.gas_versatility(), dict(mode='hydraulics', max_iter_hyd=20)),
    # hand-built networks for component types none of the bundled examples exercise
    "compressor": _net_compressor,
    "flow_control_heat_exchanger": _net_flow_control_heat_exchanger,
    "pressure_control": _net_pressure_control,
    "mass_storage": _net_mass_storage,
    "circ_pump_mass": _net_circ_pump_mass,
    "heat_consumer_circ_pump_pressure": _net_heat_consumer_circ_pump_pressure,
}


@pytest.mark.parametrize("network_name", list(NETWORK_BUILDERS.keys()))
def test_pruned_component_list_matches_full(network_name):
    """A net's component_list normally contains every registered component class regardless of
    whether the net uses it - removing the ones with zero rows must not change any result."""
    net, pipeflow_kwargs = NETWORK_BUILDERS[network_name]()
    _run_full_vs_pruned(net, pipeflow_kwargs)


if __name__ == "__main__":
    pytest.main([__file__])
