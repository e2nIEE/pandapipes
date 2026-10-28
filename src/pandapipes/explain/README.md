# pandapipes-explain

Plain-English explanations for pandapipes runtime errors, powered by a local LLM.
Zero cloud calls. Zero data leaves your machine.

When your simulation crashes with `PipeflowNotConverged` or a NumPy stacktrace,
this layer hands the error, your code, and the network's actual values to a
local model and prints **what went wrong, the evidence, and a minimal fix**.

---

## Quick start

```python
import pandapipes as pp
import pandapipes.explain

pandapipes.explain.enable()      # turn on the diagnostic hook

net = pp.create_empty_network(fluid="water")
# ... build your network ...
pp.pipeflow(net)                 # if this raises, you get a plain-English explanation
```

That's it. The original pandapipes error is never hidden — the explanation is
printed *after* it.

### Notebook / scoped use

```python
with pandapipes.explain.explain():
    pp.pipeflow(net)
```

---

## Install

```bash
pip install "pandapipes[explain]"
```

The `[explain]` extra adds **no Python dependencies** — everything talks to
Ollama over `urllib`. You only need to install Ollama itself (below).

---

## Install Ollama (one-time)

Ollama runs the LLM locally at `http://localhost:11434`.

### macOS
```bash
brew install ollama
ollama serve &           # start the server in the background
ollama pull llama3.1:8b  # download the default model (~5 GB)
```
Or download the installer from [ollama.com/download](https://ollama.com/download).

### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull llama3.1:8b
```

### Windows
1. Download the installer from [ollama.com/download](https://ollama.com/download)
2. Run it — Ollama starts automatically as a background service
3. In a terminal: `ollama pull llama3.1:8b`

### Verify
```bash
curl http://localhost:11434/api/tags
```
Should return JSON listing your installed models.

---

## Picking a model

The default is **`llama3.1:8b`** — a good balance of accuracy and speed for
diagnosing pandapipes errors. You can switch to any model Ollama supports:

```python
pandapipes.explain.enable(model="qwen2.5:14b")
```

Pull the model first with `ollama pull <name>`.

### Recommended models

| Model | Size | Good for |
|---|---|---|
| `llama3.2:3b` | ~2 GB | Low-RAM laptops, fastest responses |
| **`llama3.1:8b`** *(default)* | ~5 GB | Best general balance — recommended |
| `qwen2.5:7b` | ~4 GB | Strong reasoning, multilingual |
| `mistral:7b` | ~4 GB | Fast, solid baseline |
| `gemma3:12b` | ~8 GB | High-quality answers if you have the RAM |
| `phi4:14b` | ~9 GB | Strong on structured reasoning |
| `llama3.3:70b` | ~40 GB | Best quality — needs a workstation / GPU |
| `deepseek-r1:8b` | ~5 GB | Reasoning-focused (slower, more thorough) |

Browse the full catalogue at [ollama.com/library](https://ollama.com/library).
Anything in the library works — pull it, pass the name to `enable(model=...)`.

---

## Custom prompt (tell the LLM your preferences)

Want answers in a specific language, shorter fixes, or a custom focus?
Edit one variable.

**File:** `pandapipes/explain/llm/prompt.py`

```python
CUSTOM_PROMPT = ""   # ← put your instructions here
```

Anything you put here is appended to the system prompt under a `CUSTOM PROMPT:`
header on every request, and **overrides** the built-in rules when they conflict.

**Examples:**
```python
CUSTOM_PROMPT = "Answer in German."
CUSTOM_PROMPT = "Keep the Fix snippet under 5 lines."
CUSTOM_PROMPT = "Assume the user is a district-heating engineer — explain physical concepts in plain language."
```

Leave it as `""` to use the defaults.

---

## What it shows you

When a pandapipes call raises, you'll see:

1. The normal pandapipes traceback (untouched)
2. Below it, a block like:

```
(a) Cause — the network has no ext_grid, so there is no pressure reference
    (slack node) and the hydraulic solver cannot converge.
(b) Evidence — Network Statistics: `Ext grids: 0`
(c) Fix — add a pressure reference at the supply junction:
        pp.create_ext_grid(net, junction=j0, p_bar=5.0, t_k=293.15)
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| No explanation appears | Check `ollama serve` is running: `curl localhost:11434` |
| "model not found" | `ollama pull llama3.1:8b` |
| Explanation is slow (>30 s) | Use a smaller model, or run on a machine with more RAM |
| Want to turn it off | `pandapipes.explain.disable()` |

If the LLM is unreachable or returns junk, **your program is unaffected** —
you only see the original pandapipes error. The explainer is a silent passenger.

---

## Privacy

- No internet calls. Ollama runs at `localhost:11434`.
- No telemetry, no API keys, no accounts.
- Your code and network data never leave the machine.

---

## For developers / maintainers

### How it works

```
sys.excepthook (core/intercept.py)  ──►  core/dispatcher.handle()
                                            │
    1. llm/client.is_available()            │  Ollama reachable? else print a hint and stop
    2. context/traceback_parser             │  source lines around the failing frames
    3. llm/prompt.extract_net_stats         │  counts/values from the `net` found in frame locals
    4. llm/prompt._run_selected_diagnostic_checks
                                            │  runs checks from pandapipes.diagnostic (see below)
    5. core/log_capture                     │  recent stdout + captured warnings
    6. knowledge/loader.load_relevant       │  keyword-matched sections of knowledge/pack/*.md
    7. llm/prompt (system prompt + soul.md + rules.md)  ──►  llm/client.chat()  ──►  printed
```

| Path | Role |
|---|---|
| `__init__.py` | Public API: `enable()`, `disable()`, `explain()` context manager |
| `cli.py` | `pandapipes-explain` console script (`status`, `version`) |
| `core/config.py` | `ExplainConfig` singleton (host, model, timeout, ...) |
| `llm/prompt.py` | Prompt assembly, network statistics, deterministic pre-checks |
| `llm/client.py` | Ollama HTTP client (stdlib `urllib` only, no new dependencies) |
| `knowledge/pack/*.md` | Curated troubleshooting knowledge; edit these to improve answers |
| `soul.md`, `rules.md` | Persona and hard rules injected into the system prompt |

### Coupling to `pandapipes.diagnostic`

`llm/prompt.py::_run_selected_diagnostic_checks` imports specific check classes from
`pandapipes.diagnostic.diagnostic_functions` and registers them on a
`Diagnostic(add_default_functions=False)`. Results are rendered by
`_format_diagnostic_results`, keyed by the registration name.

If you rename/add a check upstream:

- update the import list and the `selected` list in `_run_selected_diagnostic_checks`,
- add a branch in `_format_diagnostic_results` for its result shape,
- `test_selected_diagnostic_checks_exist_upstream` fails if an imported check disappears.

An `ImportError` here is logged as a warning (logger `pandapipes.explain.llm.prompt`);
any other failure is logged at debug level and never breaks the user's program.

### Tests

```bash
pytest src/pandapipes/test/explain          # CI-safe, no Ollama needed (LLM calls are mocked)
```

Slow integration tests against a real local model are opt-in:

```bash
PANDAPIPES_EXPLAIN_RUN_OLLAMA=1 PANDAPIPES_EXPLAIN_OLLAMA_MODEL=llama3.1:8b \
    pytest src/pandapipes/test/explain/test_explain_ollama_integration.py
```

`src/pandapipes/test/explain/run_edge_case_probe.py` is a manual script for probing
model answers on edge cases; it is not collected by pytest.

### Known limitations / ideas

- Only Ollama is supported as backend; `llm/client.py` is the single place to add another.
- LLM output is non-deterministic; tests assert on the prompt/context, not on model text.
- Knowledge matching is keyword-based (`knowledge/loader.py`), not semantic.
