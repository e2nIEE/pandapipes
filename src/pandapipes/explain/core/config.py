"""Runtime configuration — single source of truth for all tunables."""

from dataclasses import dataclass


@dataclass
class ExplainConfig:
    """
    All runtime-configurable values in one place.

    Attributes
    ----------
    ollama_host : str
        Base URL of the Ollama server (must be running locally).
    model : str
        Ollama model name. Must be pulled: ``ollama pull <model>``.
    source_context_lines : int
        Number of lines above and below the error line to include in context.
    target_package : str
        "pandapipes" or "pandapower" — controls which library frames are captured.
    request_timeout : float | None
        Seconds to wait for the Ollama chat response before giving up.
        Generous by default to survive a cold model load on CPU; set to
        ``None`` to wait indefinitely.
    enabled : bool
        Internal flag — True when hook is active.
    """

    ollama_host: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    source_context_lines: int = 50
    target_package: str = "pandapipes"
    request_timeout: float | None = 300.0
    enabled: bool = False


# Module-level singleton — do not instantiate ExplainConfig elsewhere.
_config = ExplainConfig()


def get_config() -> ExplainConfig:
    """Return the global config instance."""
    return _config
