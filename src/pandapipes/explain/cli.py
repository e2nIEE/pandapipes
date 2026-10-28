"""
CLI entry point.

Commands
--------
pandapipes-explain status    — check Ollama connectivity and model availability
pandapipes-explain version   — print package version
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> None:
    args = (argv or sys.argv)[1:]

    if not args or args[0] in ("-h", "--help"):
        _print_help()
        return

    cmd = args[0]

    if cmd == "status":
        _cmd_status()
    elif cmd == "version":
        from pandapipes.explain import __version__
        print(f"pandapipes-explain {__version__}")
    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        _print_help()
        sys.exit(1)


def _cmd_status() -> None:
    from pandapipes.explain.llm.client import is_available, list_models, model_available
    from pandapipes.explain.core.config import get_config

    cfg = get_config()
    print(f"Ollama host : {cfg.ollama_host}")
    print(f"Model       : {cfg.model}")

    if is_available():
        print("Ollama      : ✓ running")
        models = list_models()
        if models:
            print("Models      :", ", ".join(models))
            if model_available(cfg.model):
                print(f"Target model: ✓ {cfg.model} is available")
            else:
                print(f"Target model: ✗ '{cfg.model}' not found — run: ollama pull {cfg.model}")
        else:
            print("Models      : (none pulled yet) — run: ollama pull", cfg.model)
    else:
        print("Ollama      : ✗ not running — start with: ollama serve")


def _print_help() -> None:
    print(
        "pandapipes-explain — LLM diagnostic layer for pandapipes\n"
        "\n"
        "Usage:\n"
        "  pandapipes-explain status    Check Ollama connectivity\n"
        "  pandapipes-explain version   Print version\n"
        "  pandapipes-explain --help    Show this message\n"
        "\n"
        "Setup:\n"
        "  1. Install Ollama: https://ollama.com/download\n"
        "  2. Pull a model:   ollama pull llama3.1:8b\n"
        "  3. Start Ollama:   ollama serve\n"
        "  4. In your code:   import pandapipes.explain; pandapipes.explain.enable()"
    )


if __name__ == "__main__":
    main()
