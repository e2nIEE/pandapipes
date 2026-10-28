# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import pytest

import pandapipes as pp
from pandapipes.explain.knowledge.loader import load_relevant
from pandapipes.explain.llm import prompt as llm_prompt
from pandapipes.pf.pipeflow_setup import PipeflowNotConverged


def _base_gas_net():
    net = pp.create_empty_network(fluid="lgas")
    junctions = [
        pp.create_junction(net, pn_bar=1.05, tfluid_k=293.15, height_m=0)
        for _ in range(6)
    ]
    pp.create_ext_grid(net, junction=junctions[0], p_bar=1.1, t_k=293.15)
    for from_j, to_j in zip(junctions[:-1], junctions[1:]):
        pp.create_pipe_from_parameters(
            net,
            from_j,
            to_j,
            length_km=0.1,
            inner_diameter_mm=300,
            k_mm=0.1,
            u_w_per_m2k=0,
        )
    pp.create_sink(net, junction=junctions[-1], mdot_kg_per_s=0.1)
    return net


def _base_water_net():
    net = pp.create_empty_network(fluid="water")
    j0 = pp.create_junction(net, pn_bar=4.0, tfluid_k=353.15, height_m=0)
    j1 = pp.create_junction(net, pn_bar=4.0, tfluid_k=353.15, height_m=0)
    j2 = pp.create_junction(net, pn_bar=4.0, tfluid_k=353.15, height_m=0)
    pp.create_ext_grid(net, junction=j0, p_bar=4.0, t_k=353.15)
    pp.create_pipe_from_parameters(net, j0, j1, length_km=0.1, inner_diameter_mm=100, k_mm=0.1)
    pp.create_pipe_from_parameters(net, j1, j2, length_km=0.1, inner_diameter_mm=100, k_mm=0.1)
    pp.create_sink(net, junction=j2, mdot_kg_per_s=0.1)
    return net


def _context_for(net):
    try:
        raise PipeflowNotConverged("The hydraulic calculation did not converge to a solution.")
    except PipeflowNotConverged as exc:
        diagnostics = llm_prompt.extract_diagnostics(exc.__traceback__, type(exc).__name__, str(exc))
        stats = llm_prompt.extract_net_stats(exc.__traceback__)
        knowledge = load_relevant(type(exc).__name__, str(exc), context=f"{stats}\n{diagnostics}")
        messages = llm_prompt.build(
            error_type=type(exc).__name__,
            error_msg=str(exc),
            user_contexts=[],
            lib_contexts=[],
            knowledge=knowledge,
            net_stats=stats,
            diagnostics=diagnostics,
        )
    return diagnostics, "\n\n".join(message["content"] for message in messages)


def _no_fluid_net():
    net = pp.create_empty_network()
    j0 = pp.create_junction(net, pn_bar=1.0, tfluid_k=293.15)
    j1 = pp.create_junction(net, pn_bar=1.0, tfluid_k=293.15)
    pp.create_ext_grid(net, junction=j0, p_bar=1.0, t_k=293.15)
    pp.create_pipe_from_parameters(net, j0, j1, length_km=0.1, inner_diameter_mm=300)
    pp.create_sink(net, junction=j1, mdot_kg_per_s=0.1)
    return net


def _missing_ext_grid_net():
    net = _base_gas_net()
    net.ext_grid.drop(net.ext_grid.index, inplace=True)
    return net


def _unsupplied_junction_net():
    net = pp.create_empty_network(fluid="lgas")
    j0 = pp.create_junction(net, 1.0, 293.15)
    j1 = pp.create_junction(net, 1.0, 293.15)
    j2 = pp.create_junction(net, 1.0, 293.15)
    pp.create_ext_grid(net, j0, p_bar=1.1, t_k=293.15)
    pp.create_pipe_from_parameters(net, j0, j1, length_km=0.1, inner_diameter_mm=300)
    pp.create_sink(net, j2, mdot_kg_per_s=0.1)
    return net


def _with_closed_valve():
    net = _base_gas_net()
    pp.create_valve(net, junction=2, element=3, et="ju", inner_diameter_mm=100, opened=False)
    return net


def _with_bad_valve_reference():
    net = _base_gas_net()
    pp.create_valve(net, junction=2, element=3, et="ju", inner_diameter_mm=100, opened=True)
    net.valve.loc[0, "element"] = 999
    return net


def _with_compressor_ratio():
    net = _base_gas_net()
    pp.create_compressor(net, from_junction=0, to_junction=1, pressure_ratio=10)
    return net


def _with_circ_pump_mass():
    net = _base_water_net()
    pp.create_circ_pump_const_mass_flow(
        net,
        return_junction=1,
        flow_junction=2,
        p_flow_bar=4,
        mdot_flow_kg_per_s=1000,
        t_flow_k=353.15,
    )
    return net


def _closed_loop_const_pressure_net():
    net = pp.create_empty_network(fluid="water")
    j0 = pp.create_junction(net, pn_bar=4.0, tfluid_k=343.15, height_m=0)
    j1 = pp.create_junction(net, pn_bar=4.0, tfluid_k=343.15, height_m=0)
    j2 = pp.create_junction(net, pn_bar=4.0, tfluid_k=343.15, height_m=0)
    j3 = pp.create_junction(net, pn_bar=4.0, tfluid_k=343.15, height_m=0)
    pp.create_circ_pump_const_pressure(
        net,
        return_junction=j0,
        flow_junction=j1,
        p_flow_bar=4.0,
        plift_bar=1.5,
        t_flow_k=343.15,
    )
    pp.create_pipe_from_parameters(net, j1, j2, length_km=0.1, inner_diameter_mm=100, k_mm=0.1)
    pp.create_heat_consumer(net, from_junction=j2, to_junction=j3, qext_w=10000, treturn_k=323.15)
    pp.create_pipe_from_parameters(net, j3, j0, length_km=0.1, inner_diameter_mm=100, k_mm=0.1)
    return net


SCENARIOS = [
    ("missing_fluid", _no_fluid_net, "no fluid is defined"),
    ("missing_ext_grid", _missing_ext_grid_net, "net.ext_grid is empty"),
    ("unsupplied_junction", _unsupplied_junction_net, "unsupplied junctions have no path to an ext_grid"),
    ("extreme_height", lambda: _mutate(_base_gas_net(), "junction", 3, "height_m", 1e6), "net.junction.loc[3, 'height_m'] = 1000000.0"),
    ("huge_sink", lambda: _mutate(_base_gas_net(), "sink", 0, "mdot_kg_per_s", 10000.0), "net.sink.loc[0, 'mdot_kg_per_s'] = 10000.0"),
    ("negative_sink", lambda: _mutate(_base_gas_net(), "sink", 0, "mdot_kg_per_s", -1.0), "net.sink.loc[0, 'mdot_kg_per_s'] = -1.0"),
    ("nan_sink", lambda: _mutate(_base_gas_net(), "sink", 0, "mdot_kg_per_s", np.nan), "net.sink.loc[0, 'mdot_kg_per_s'] = nan"),
    ("huge_source", lambda: _add_source(_base_gas_net(), 2, 10000.0), "net.source.loc[0, 'mdot_kg_per_s'] = 10000.0"),
    ("tiny_pipe_diameter", lambda: _mutate(_base_gas_net(), "pipe", 0, "inner_diameter_mm", 0.001), "net.pipe.loc[0, 'inner_diameter_mm'] = 0.001"),
    ("zero_pipe_diameter", lambda: _mutate(_base_gas_net(), "pipe", 0, "inner_diameter_mm", 0.0), "net.pipe.loc[0, 'inner_diameter_mm'] = 0.0"),
    ("huge_pipe_length", lambda: _mutate(_base_gas_net(), "pipe", 0, "length_km", 10000.0), "net.pipe.loc[0, 'length_km'] = 10000.0"),
    ("zero_pipe_length", lambda: _mutate(_base_gas_net(), "pipe", 0, "length_km", 0.0), "net.pipe.loc[0, 'length_km'] = 0.0"),
    ("invalid_pipe_to_junction", lambda: _mutate(_base_gas_net(), "pipe", 0, "to_junction", 999), "net.pipe.loc[0, 'to_junction'] = 999"),
    ("invalid_sink_junction", lambda: _mutate(_base_gas_net(), "sink", 0, "junction", 999), "net.sink.loc[0, 'junction'] = 999"),
    ("invalid_ext_grid_junction", lambda: _mutate(_base_gas_net(), "ext_grid", 0, "junction", 999), "net.ext_grid.loc[0, 'junction'] = 999"),
    ("bad_junction_pressure", lambda: _mutate(_base_gas_net(), "junction", 0, "pn_bar", 0.0), "net.junction.loc[0, 'pn_bar'] = 0.0"),
    ("bad_junction_temperature", lambda: _mutate(_base_gas_net(), "junction", 0, "tfluid_k", 0.0), "net.junction.loc[0, 'tfluid_k'] = 0.0"),
    ("bad_ext_grid_pressure", lambda: _mutate(_base_gas_net(), "ext_grid", 0, "p_bar", 0.0), "net.ext_grid.loc[0, 'p_bar'] = 0.0"),
    ("bad_ext_grid_temperature", lambda: _mutate(_base_gas_net(), "ext_grid", 0, "t_k", 0.0), "net.ext_grid.loc[0, 't_k'] = 0.0"),
    ("high_pipe_roughness", lambda: _mutate(_base_gas_net(), "pipe", 0, "k_mm", 10.0), "net.pipe.loc[0, 'k_mm'] = 10.0"),
    ("closed_valve", _with_closed_valve, "net.valve.loc[0, 'opened'] = False"),
    ("bad_valve_reference", _with_bad_valve_reference, "net.valve.loc[0, 'element'] = 999"),
    ("high_heat_transfer", lambda: _mutate(_base_water_net(), "pipe", 0, "u_w_per_m2k", 100.0), "net.pipe.loc[0, 'u_w_per_m2k'] = 100.0"),
    ("high_compressor_ratio", _with_compressor_ratio, "net.compressor.loc[0, 'pressure_ratio'] = 10.0"),
    ("high_circ_pump_mass", _with_circ_pump_mass, "net.circ_pump_mass.loc[0, 'mdot_flow_kg_per_s'] = 1000.0"),
]


def _mutate(net, table, idx, column, value):
    net[table].loc[idx, column] = value
    return net


def _add_source(net, junction, mdot):
    pp.create_source(net, junction=junction, mdot_kg_per_s=mdot)
    return net


@pytest.mark.parametrize("name, factory, expected", SCENARIOS)
def test_explain_prompt_context_contains_specific_root_cause(name, factory, expected):
    diagnostics, prompt_context = _context_for(factory())
    assert expected in diagnostics, name
    assert expected in prompt_context, name


@pytest.mark.parametrize(
    "name, expected_snippet",
    [
        ("extreme_height", "Extreme elevations dominate hydrostatic pressure"),
        ("huge_sink", "Wrong magnitude / units"),
        ("tiny_pipe_diameter", "Unrealistically small but positive pipe diameter"),
        ("invalid_pipe_to_junction", "Non-existent junction index"),
        ("huge_pipe_length", "Unrealistically large pipe length"),
        ("missing_fluid", "No fluid defined on the net"),
        ("closed_valve", "Closed valve isolating a section"),
        ("high_compressor_ratio", "Pump / compressor issues"),
    ],
)
def test_knowledge_injection_includes_specific_sections(name, expected_snippet):
    scenarios = {scenario_name: factory for scenario_name, factory, _ in SCENARIOS}
    _, prompt_context = _context_for(scenarios[name]())
    assert "## Diagnostic Checklist" in prompt_context
    assert expected_snippet in prompt_context, name


def test_diagnostic_formatter_accepts_numpy_bool_results():
    formatted = llm_prompt._format_diagnostic_results(
        {
            "pipe_diameter": np.bool_(True),
            "junction_height": np.bool_(True),
            "heat_transfer_coefficient": np.bool_(False),
        }
    )

    assert "  - pipe_diameter: pipeflow converged after increasing very small pipe diameters." in formatted
    assert "  - junction_height: pipeflow converged after flattening all junction heights to 0 m." in formatted
    assert not any("heat_transfer_coefficient" in line for line in formatted)


def test_inactive_pressure_controls_diagnostic_is_formatted():
    formatted = llm_prompt._format_diagnostic_results({"inactive_pressure_controls": np.bool_(True)})

    assert formatted == [
        "  - inactive_pressure_controls: pipeflow converged after all active pressure controls "
        "were deactivated. The pressure-control configuration is therefore a strong candidate "
        "for the non-convergence."
    ]


def test_closed_loop_const_pressure_pump_is_not_reported_as_missing_ext_grid():
    diagnostics, prompt_context = _context_for(_closed_loop_const_pressure_net())

    assert "active circ_pump_const_pressure" in diagnostics
    assert "net.ext_grid is empty, so there is no pressure reference" not in diagnostics
    assert "no ext_grid — there is no pressure reference" not in prompt_context
    assert "Never suggest adding an ext_grid solely because \"Ext grids: 0\"" in prompt_context


def test_selected_diagnostic_checks_exist_upstream():
    """Guard: explain wires specific pandapipes.diagnostic checks; renames upstream must fail loudly."""
    import inspect

    from pandapipes.diagnostic import diagnostic_functions as df

    source = inspect.getsource(llm_prompt._run_selected_diagnostic_checks)
    imported = source.split("from pandapipes.diagnostic.diagnostic_functions import (")[1].split(")")[0]
    names = [n.strip().rstrip(",") for n in imported.split() if n.strip()]
    assert names
    missing = [n for n in names if not hasattr(df, n)]
    assert not missing, f"pandapipes.diagnostic no longer provides: {missing}"


def test_valve_configuration_result_is_formatted():
    lines = llm_prompt._format_diagnostic_results(
        {"valve_configuration": {"all_open": True, "all_closed": False}}
    )
    assert any("opening all valves" in line for line in lines), lines
    assert not any("closing all valves" in line for line in lines), lines


def test_missing_pressure_reference_result_is_formatted():
    lines = llm_prompt._format_diagnostic_results({"missing_pressure_reference": True})
    assert any("missing_pressure_reference" in line for line in lines), lines
