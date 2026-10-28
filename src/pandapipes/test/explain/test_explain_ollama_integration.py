# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

import os
import re

import pytest

import pandapipes as pp
from pandapipes.explain.core.config import get_config
from pandapipes.explain.knowledge.loader import load_relevant
from pandapipes.explain.llm import client as llm_client
from pandapipes.explain.llm import prompt as llm_prompt
from pandapipes.pf.pipeflow_setup import PipeflowNotConverged
from pandapipes.test.explain.test_explain_diagnostic_scenarios import SCENARIOS, _context_for

pytestmark = pytest.mark.skipif(
    os.environ.get("PANDAPIPES_EXPLAIN_RUN_OLLAMA") != "1",
    reason="Set PANDAPIPES_EXPLAIN_RUN_OLLAMA=1 to run slow local Ollama integration tests.",
)


def _needle_terms(expected):
    match = re.search(r"net\.([a-z_]+)\.loc\[[^]]+], '([^']+)']", expected)
    if match:
        return [match.group(1), match.group(2)]
    if "no fluid" in expected:
        return ["fluid"]
    if "ext_grid is empty" in expected:
        return ["ext_grid"]
    if "unsupplied junctions" in expected:
        return ["unsupplied", "ext_grid"]
    return [expected.split()[0]]


@pytest.mark.parametrize("name, factory, expected", SCENARIOS)
def test_ollama_response_mentions_specific_root_cause(name, factory, expected):
    if not llm_client.is_available():
        pytest.skip("Ollama server is not reachable.")

    configured_model = os.environ.get("PANDAPIPES_EXPLAIN_OLLAMA_MODEL")
    if configured_model:
        get_config().model = configured_model

    diagnostics, prompt_context = _context_for(factory())
    assert expected in prompt_context

    cfg = get_config()
    messages = [
        {
            "role": "system",
            "content": llm_prompt._SYSTEM,
        },
        {
            "role": "user",
            "content": prompt_context,
        },
    ]
    response = llm_client.chat(messages)
    assert response, f"Ollama model {cfg.model!r} returned no response for {name}"

    lowered = response.lower()
    missing = [term for term in _needle_terms(expected) if term.lower() not in lowered]
    assert not missing, f"{name}: missing {missing} in response:\n{response}\n\nDiagnostics:\n{diagnostics}"


def _heat_transfer_context():
    net = pp.create_empty_network(fluid="water")
    j0 = pp.create_junction(net, pn_bar=4.0, tfluid_k=353.15, height_m=0)
    j1 = pp.create_junction(net, pn_bar=4.0, tfluid_k=353.15, height_m=0)
    j2 = pp.create_junction(net, pn_bar=4.0, tfluid_k=353.15, height_m=0)

    pp.create_ext_grid(net, junction=j0, p_bar=4.0, t_k=353.15)
    pp.create_pipe_from_parameters(
        net,
        j0,
        j1,
        length_km=0.5,
        inner_diameter_mm=100,
        k_mm=0.1,
        sections=5,
        u_w_per_m2k=100.0,
        text_k=273.15,
    )
    pp.create_pipe_from_parameters(
        net,
        j1,
        j2,
        length_km=0.5,
        inner_diameter_mm=100,
        k_mm=0.1,
        sections=5,
        u_w_per_m2k=1.0,
        text_k=273.15,
    )
    pp.create_sink(net, junction=j2, mdot_kg_per_s=0.1)

    try:
        raise PipeflowNotConverged("The heat transfer calculation did not converge to a solution.")
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
    return diagnostics, messages


def test_ollama_response_mentions_heat_transfer_coefficient_for_heat_nonconvergence():
    if not llm_client.is_available():
        pytest.skip("Ollama server is not reachable.")

    configured_model = os.environ.get("PANDAPIPES_EXPLAIN_OLLAMA_MODEL")
    if configured_model:
        get_config().model = configured_model

    diagnostics, messages = _heat_transfer_context()
    prompt_context = "\n\n".join(message["content"] for message in messages)
    assert "net.pipe.loc[0, 'u_w_per_m2k'] = 100.0" in prompt_context

    response = llm_client.chat(messages)
    assert response, f"Ollama model {get_config().model!r} returned no response for heat transfer case"

    lowered = response.lower()
    missing = [term for term in ("heat", "u_w_per_m2k") if term not in lowered]
    assert not missing, f"heat_transfer: missing {missing} in response:\n{response}\n\nDiagnostics:\n{diagnostics}"
