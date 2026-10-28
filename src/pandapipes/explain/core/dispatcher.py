"""
Dispatcher — the single orchestration point.

Receives an exception, builds context from traceback + knowledge,
assembles a prompt, sends it to Ollama, prints the explanation.
"""

from __future__ import annotations

import sys
import traceback as _tb
from types import TracebackType
from typing import Type

from pandapipes.explain.core.config import get_config
from pandapipes.explain.core import log_capture
from pandapipes.explain.context.traceback_parser import extract_context
from pandapipes.explain.knowledge.loader import load_relevant
from pandapipes.explain.llm import client as llm_client
from pandapipes.explain.llm import prompt as llm_prompt


def handle(
    exc_type: Type[BaseException],
    exc_value: BaseException,
    exc_tb: TracebackType | None,
) -> None:
    """
    Main entry point called by the exception hook.

    Parameters
    ----------
    exc_type : type
        Exception class.
    exc_value : BaseException
        Exception instance (carries the message).
    exc_tb : TracebackType | None
        Traceback object — used to locate source files.
    """
    cfg = get_config()

    # --- Extract source context from traceback ---
    contexts = extract_context(exc_tb, cfg.source_context_lines, cfg.target_package)
    user_contexts = [c for c in contexts if c.is_user_code]
    lib_contexts = [c for c in contexts if not c.is_user_code]

    error_type = exc_type.__name__
    error_msg = str(exc_value)

    # --- Extract network stats from traceback locals ---
    net_stats = llm_prompt.extract_net_stats(exc_tb)

    # --- Ignore errors unrelated to pandapipes (no pandapipes frame, no network in scope) ---
    if not lib_contexts and not net_stats:
        return

    # --- Check Ollama availability ---
    if not llm_client.is_available():
        _print_hint(
            "Ollama not running. Start it with: ollama serve\n"
            f"  Then make sure the model is pulled: ollama pull {cfg.model}"
        )
        return
    if not llm_client.model_available(cfg.model):
        _print_hint(f"Model '{cfg.model}' is not available locally. Pull it with: ollama pull {cfg.model}")
        return

    # --- Grab the tail of stdout/warnings captured before the error ---
    terminal_logs = log_capture.get_recent_output()

    # --- Run deterministic pandapipes diagnostics before asking the LLM ---
    diagnostics = llm_prompt.extract_diagnostics(exc_tb, error_type, error_msg)

    # --- Load relevant curated knowledge, using diagnostics as retrieval context ---
    knowledge = load_relevant(error_type, error_msg, context=f"{net_stats}\n{diagnostics}\n{terminal_logs}")

    # --- Format the full traceback as readable text (the call chain) ---
    try:
        traceback_text = "".join(_tb.format_tb(exc_tb)).strip()
    except Exception:
        traceback_text = ""

    # --- Build and send prompt ---
    messages = llm_prompt.build(
        error_type=error_type,
        error_msg=error_msg,
        user_contexts=user_contexts,
        lib_contexts=lib_contexts,
        knowledge=knowledge,
        net_stats=net_stats,
        diagnostics=diagnostics,
        terminal_logs=terminal_logs,
        traceback_text=traceback_text,
    )

    response = llm_client.chat(messages)

    if response:
        _print_explanation(response)
    else:
        _print_hint("Ollama returned no response. Check that the model is running correctly.")


def _print_explanation(text: str) -> None:
    """Print the LLM explanation to stderr with a clear visual boundary."""
    border = "─" * 64
    print(f"\n{border}", file=sys.stderr)
    print("  pandapipes-explain — LLM Diagnostic", file=sys.stderr)
    print(border, file=sys.stderr)
    print(text.strip(), file=sys.stderr)
    print(border, file=sys.stderr)


def _print_hint(msg: str) -> None:
    """Print a short hint when explanation is not possible."""
    print(f"\n[pandapipes-explain] {msg}", file=sys.stderr)
