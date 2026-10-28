# Copyright (c) 2020-2026 by Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel, and University of Kassel. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.

"""
CI-safe tests for pandapipes.explain.

None of these tests need Ollama: the LLM is reached only by the final
``client.chat`` HTTP call, which is either mocked or never reached (the
dispatcher early-returns when the server is unavailable). Everything else
— hook install, traceback parsing, knowledge matching, prompt building —
is pure Python and tested directly.
"""

import subprocess
import sys

import pytest

import pandapipes as pps
import pandapipes.explain as ppe
from pandapipes.explain.core import dispatcher
from pandapipes.explain.core import log_capture
from pandapipes.explain.core.config import get_config
from pandapipes.explain.knowledge.loader import load_relevant
from pandapipes.explain.llm import client as llm_client
from pandapipes.explain.llm import prompt as llm_prompt
from pandapipes.pf.pipeflow_setup import PipeflowNotConverged


def _faulty_net():
    """No ext_grid -> no pressure reference -> pp.pipeflow raises PipeflowNotConverged."""
    net = pps.create_empty_network(fluid="water")
    j1 = pps.create_junction(net, pn_bar=1.0, tfluid_k=293.15)
    j2 = pps.create_junction(net, pn_bar=1.0, tfluid_k=293.15)
    pps.create_pipe_from_parameters(net, j1, j2, length_km=0.1, inner_diameter_mm=100)
    pps.create_sink(net, j2, mdot_kg_per_s=0.5)
    return net


def _six_junction_gas_net():
    net = pps.create_empty_network(fluid="lgas")
    junctions = [
        pps.create_junction(net, pn_bar=1.05, tfluid_k=293.15, height_m=0)
        for _ in range(6)
    ]
    pps.create_ext_grid(net, junction=junctions[0], p_bar=1.1, t_k=293.15)
    for from_j, to_j in zip(junctions[:-1], junctions[1:]):
        pps.create_pipe_from_parameters(
            net, from_j, to_j, length_km=0.1, inner_diameter_mm=300, k_mm=0.1
        )
    pps.create_sink(net, junction=junctions[-1], mdot_kg_per_s=0.1)
    return net


def _diagnostics_from_net(net):
    try:
        raise PipeflowNotConverged("The hydraulic calculation did not converge to a solution.")
    except PipeflowNotConverged as exc:
        return llm_prompt.extract_diagnostics(exc.__traceback__, type(exc).__name__, str(exc))


@pytest.fixture(autouse=True)
def _restore_excepthook():
    """Each test must leave the global hook exactly as it found it."""
    saved = sys.excepthook
    yield
    ppe.disable()
    sys.excepthook = saved


def test_import_is_inert():
    # Run in a clean subprocess: importing the package must NOT install a hook
    # or flip the enabled flag (verifiable only before enable() is ever called).
    code = (
        "import sys; before = sys.excepthook;"
        "import pandapipes.explain;"
        "assert sys.excepthook is before, 'import changed sys.excepthook';"
        "from pandapipes.explain.core.config import get_config;"
        "assert get_config().enabled is False, 'import enabled the hook';"
        "print('INERT_OK')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "INERT_OK" in proc.stdout


def test_enable_disable_roundtrip():
    saved = sys.excepthook
    ppe.enable()
    assert sys.excepthook is not saved
    assert get_config().enabled is True
    ppe.disable()
    assert sys.excepthook is saved
    assert get_config().enabled is False


def test_enable_is_idempotent():
    # R9: a second enable() must not re-wrap the hook or lose the original.
    saved = sys.excepthook
    ppe.enable()
    hook_after_first = sys.excepthook
    ppe.enable()
    assert sys.excepthook is hook_after_first
    ppe.disable()
    assert sys.excepthook is saved


def test_knowledge_loader_matches_pipeflow():
    text = load_relevant("PipeflowNotConverged", "did not converge to a solution")
    assert text.strip(), "no knowledge matched PipeflowNotConverged"


def test_prompt_builds_and_stays_within_budget():
    net = _faulty_net()
    with pytest.raises(PipeflowNotConverged) as ei:
        pps.pipeflow(net)
    exc = ei.value
    tb = exc.__traceback__

    stats = llm_prompt.extract_net_stats(tb)
    # The dominant cause must be surfaced in the stats the model sees.
    assert "ext_grid" in stats

    messages = llm_prompt.build(
        error_type=type(exc).__name__,
        error_msg=str(exc),
        user_contexts=[],
        lib_contexts=[],
        knowledge=load_relevant(type(exc).__name__, str(exc)),
        net_stats=stats,
        diagnostics=llm_prompt.extract_diagnostics(tb, type(exc).__name__, str(exc)),
        terminal_logs="",
        traceback_text="",
    )
    assert [m["role"] for m in messages] == ["system", "user"]
    total = sum(len(m["content"]) for m in messages)
    assert total < 20000, f"prompt {total} chars exceeds R6 budget"


def test_diagnostics_surface_feedback_cases():
    cases = []

    net = _six_junction_gas_net()
    net.junction.loc[3, "height_m"] = 1e6
    cases.append((net, "net.junction.loc[3, 'height_m'] = 1000000.0"))

    net = _six_junction_gas_net()
    net.sink.loc[0, "mdot_kg_per_s"] = 10000.0
    cases.append((net, "net.sink.loc[0, 'mdot_kg_per_s'] = 10000.0"))

    net = _six_junction_gas_net()
    net.pipe.loc[0, "inner_diameter_mm"] = 0.001
    cases.append((net, "net.pipe.loc[0, 'inner_diameter_mm'] = 0.001"))

    net = _six_junction_gas_net()
    net.pipe.loc[0, "to_junction"] = 999
    cases.append((net, "net.pipe.loc[0, 'to_junction'] = 999"))

    net = _six_junction_gas_net()
    net.pipe.loc[0, "length_km"] = 10000.0
    cases.append((net, "net.pipe.loc[0, 'length_km'] = 10000.0"))

    for net, expected in cases:
        diagnostics = _diagnostics_from_net(net)
        assert expected in diagnostics


def test_warning_capture_includes_matrix_rank_warning():
    log_capture.clear()
    log_capture.install()
    try:
        import warnings
        warnings.warn("MatrixRankWarning: Matrix is exactly singular", RuntimeWarning)
        assert "MatrixRankWarning: Matrix is exactly singular" in log_capture.get_recent_output()
    finally:
        log_capture.uninstall()


def test_full_pipeline_prints_explanation_with_mocked_ollama(monkeypatch, capsys):
    # Replace the only network call: pretend Ollama is up and returns a canned answer.
    canned = (
        "(a) Cause - no ext_grid, so no pressure reference.\n"
        "(b) Evidence - Network Statistics shows Ext grids: 0\n"
        "(c) Fix - pps.create_ext_grid(net, junction=0, p_bar=5.0, t_k=293.15)"
    )
    monkeypatch.setattr(llm_client, "is_available", lambda: True)
    monkeypatch.setattr(llm_client, "model_available", lambda model: True)
    monkeypatch.setattr(llm_client, "chat", lambda messages: canned)

    net = _faulty_net()
    with pytest.raises(PipeflowNotConverged) as ei:
        pps.pipeflow(net)
    exc = ei.value
    dispatcher.handle(type(exc), exc, exc.__traceback__)

    err = capsys.readouterr().err
    assert "pandapipes-explain" in err
    assert "(a) Cause" in err and "(c) Fix" in err


def test_program_survives_when_ollama_unreachable():
    # R1: with the layer active and Ollama down, the ORIGINAL exception must
    # propagate unchanged and nothing else may be raised.
    net = _faulty_net()
    with pytest.raises(PipeflowNotConverged):
        with ppe.explain(ollama_host="http://127.0.0.1:59999"):
            pps.pipeflow(net)


def test_model_available_requires_exact_tag(monkeypatch):
    monkeypatch.setattr(llm_client, "list_models", lambda: ["llama3.1:70b", "llama3.2:3b", "mistral:latest"])
    assert not llm_client.model_available("llama3.1:8b")
    assert not llm_client.model_available("llama3")
    assert llm_client.model_available("llama3.1:70b")
    assert llm_client.model_available("mistral")


def test_missing_model_prints_pull_hint(monkeypatch, capsys):
    monkeypatch.setattr(llm_client, "is_available", lambda: True)
    monkeypatch.setattr(llm_client, "model_available", lambda model: False)
    monkeypatch.setattr(llm_client, "chat", lambda messages: pytest.fail("must not query a missing model"))

    net = _faulty_net()
    with pytest.raises(PipeflowNotConverged) as ei:
        pps.pipeflow(net)
    dispatcher.handle(type(ei.value), ei.value, ei.value.__traceback__)
    assert "ollama pull" in capsys.readouterr().err


def test_unrelated_exception_does_not_query_ollama(monkeypatch, tmp_path):
    import runpy

    script = tmp_path / "user_script.py"
    script.write_text("d = {}\nd['missing']\n")
    monkeypatch.setattr(llm_client, "is_available", lambda: pytest.fail("must not contact Ollama"))
    try:
        runpy.run_path(str(script))
    except KeyError as exc:
        # tb_next drops this test's own frame (it lives under src/pandapipes/), leaving user-script frames only.
        dispatcher.handle(type(exc), exc, exc.__traceback__.tb_next)


def test_context_manager_keeps_prior_global_enable():
    ppe.enable(model="custom:1b")
    try:
        with pytest.raises(ZeroDivisionError):
            with ppe.explain(ollama_host="http://127.0.0.1:59999"):
                1 / 0
        assert get_config().enabled
        assert get_config().model == "custom:1b"
    finally:
        ppe.disable()


def test_disable_does_not_clobber_replaced_stdout():
    import io

    ppe.enable()
    replacement = io.StringIO()
    original = sys.stdout
    sys.stdout = replacement
    try:
        ppe.disable()
        assert sys.stdout is replacement
    finally:
        sys.stdout = original


def test_client_rejects_non_http_urls():
    import pytest
    from pandapipes.explain.llm.client import _open

    with pytest.raises(ValueError):
        _open("file:///etc/passwd", 1)
