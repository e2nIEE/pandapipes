"""
Ollama HTTP client — stdlib only (urllib), no third-party dependencies.

All functions return None / False on failure rather than raising,
so the dispatcher can degrade gracefully.
"""

from __future__ import annotations

import json
import sys
import urllib.request

from pandapipes.explain.core.config import get_config


def _open(url_or_request, timeout):
    """urlopen that only talks http(s): rejects file:/ and other schemes from a bad config."""
    url = url_or_request if isinstance(url_or_request, str) else url_or_request.full_url
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"Only http(s) URLs are allowed, got: {url}")
    return urllib.request.urlopen(url_or_request, timeout=timeout)  # nosec B310 - scheme checked above


def is_available() -> bool:
    """Return True if the Ollama server is reachable."""
    try:
        with _open(f"{get_config().ollama_host}/api/tags", 2):
            return True
    except Exception:
        return False


def list_models() -> list[str]:
    """Return names of locally available Ollama models."""
    try:
        with _open(f"{get_config().ollama_host}/api/tags", 5) as resp:
            data = json.load(resp)
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def model_available(model: str) -> bool:
    """Return True if exactly `model` is pulled in Ollama (no tag means ``:latest``)."""
    wanted = model if ":" in model else f"{model}:latest"
    return wanted in list_models()


def chat(messages: list[dict]) -> str | None:
    """
    Send a chat completion request to Ollama.

    Parameters
    ----------
    messages : list[dict]
        OpenAI-style message list: [{"role": ..., "content": ...}, ...]

    Returns
    -------
    str | None
        The assistant's reply, or None if the request failed.
    """
    cfg = get_config()
    payload = json.dumps(
        {
            "model": cfg.model,
            "messages": messages,
            "stream": False,
            # 8192 tokens fits our worst-case prompt (~3500 tokens) with room for the response.
            # llama3.2:3b defaults to 4096 in Ollama; this override is required.
            "options": {"num_ctx": 8192, "temperature": 0.2},
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        f"{cfg.ollama_host}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    # Timeout is generous (see cfg.request_timeout) so a cold model load on CPU
    # has time to finish; set it to None to wait indefinitely. Print a status
    # hint so the user knows we're querying.
    print("\n[pandapipes-explain] Querying Ollama...", file=sys.stderr)
    try:
        with _open(req, cfg.request_timeout) as resp:
            body = json.load(resp)
            return body["message"]["content"]
    except Exception:
        return None
