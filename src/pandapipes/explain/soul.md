# pandapipes-explain — Soul Document

## What This Is

An optional, zero-friction diagnostic layer that sits on top of pandapipes and
explains runtime errors in plain English using a local LLM. No internet required.
No data leaves the machine. No pandapipes internals are touched.

When a fluid-network engineer's simulation fails with `PipeflowNotConverged`, they
face a blank error message and a deep NumPy/SciPy stacktrace. This feature bridges
that gap: it turns the raw exception into a specific, actionable explanation that
points to the exact values in the user's code that caused the failure.

## Core Belief

**The error message already contains the answer — it just needs to be assembled.**

The Python traceback gives us:
- The exact file and line number that called pandapipes
- The exact file and line number inside pandapipes where the exception was raised
- Every local variable in every frame (including the `net` object itself)

No indexing, no embeddings, no AST, no pre-processing. We assemble what the
traceback already knows, add a curated knowledge snippet, and hand it to the LLM.

## Design Principles

### 1. Never crash the user's program
The entire explain pipeline runs inside `try/except`. If Ollama is down, the model
is wrong, the network can't be read, or anything else fails — the user sees only
the original pandapipes error. We are a silent passenger, not a co-pilot.

### 2. Zero mandatory dependencies
The feature installs with `pip install pandapipes[explain]`. The extra dependency
list is empty — Ollama is a separate system-level install, not a Python package.
All HTTP is done with `urllib` from the stdlib.

### 3. Local-first, offline-capable
Everything runs on the user's machine. No API keys, no rate limits, no cloud.
The LLM (Ollama + llama3.1:8b or better) runs at localhost:11434. Engineers at
gas/water utilities and research institutes often work on air-gapped networks.

### 4. Context-driven, not heuristic-driven
We don't pattern-match error messages. We show the LLM what a senior engineer
would look at: the user's code, the network object's actual values (junctions,
pipes, sinks, sources, ext_grids, the fluid), the pandapipes source that raised
the error, and what the terminal printed before it failed. The LLM does the
reasoning; we do the assembly.

### 5. Honest about limitations
A small model will sometimes be wrong. The system prompt is written to minimise
hallucination (check the network stats, quote exact values, use pandapipes units,
don't recommend ext_grid if one already exists) but the user should always verify
suggestions. The tool is a first-responder, not a final arbiter.

### 6. Composable and portable
Works the same way whether pandapipes is pip-installed in site-packages, installed
in editable mode at a repo root, or conda-installed. The traceback's `frame.filename`
is always an absolute path — we read source from there directly. Because pandapipes
itself depends on pandapower, pandapower frames are treated as library reference
code, not the user's bug.

### 7. Non-intrusive hook design
`sys.excepthook` is captured at `install()` time (not at import time) so we don't
race with pytest, IPython, Sentry, or any other tool that wraps the hook. The
original hook is always called first. We add value on top, never instead of.

## What It Is Not

- **Not a debugger** — we can't step through code or inspect intermediate states.
- **Not cloud AI** — the LLM is local, small, and imprecise. Treat its suggestions
  as a starting point for investigation, not a final answer.
- **Not a pandapipes wrapper** — we never modify, monkey-patch, or wrap any
  pandapipes function. The library remains exactly as the authors intended it.
- **Not a connectivity checker** — pandapipes' own `check_connectivity` option and
  `pandapipes.topology.unsupplied_junctions()` catch pre-run topology problems.
  This tool explains the exception *after* a run fails.

## Intended Users

Fluid-network engineers and researchers who:
- Use pandapipes to model gas, water, or district-heating networks
- Are not necessarily Python experts
- Spend significant time debugging convergence failures and bad network models
- Work in environments where sending code to cloud APIs is not acceptable

## Future Portability

This feature is itself a port of pandapower-explain. The architecture is
target-package-agnostic via the `target_package` config parameter. Switching back
to pandapower (electrical power flow) or to any future sibling library requires only
changing the target and swapping the knowledge packs — the pipeline code is shared.
