# pandapipes-explain — Engineering Rules

These rules govern every code change to this feature. Deviations require explicit justification.

---

## R1 — Never crash the user's program

Every function that touches the explain pipeline MUST be wrapped in `try/except Exception`.
This includes the hook, dispatcher, prompt builder, LLM client, and knowledge loader.
If anything in our code raises, the user sees only the original pandapipes error.
The `except` block must be silent (no print, no logging) unless we are in debug mode.

```python
# Correct pattern in any explain function:
try:
    result = our_function(...)
except Exception:
    pass  # never propagate
```

---

## R2 — Capture excepthook at install time, never at import time

`_saved_excepthook = sys.excepthook` must occur inside `install()`, not at module level.
This prevents racing with pytest, Sentry, IPython, and other tools that install their own
hooks between `import pandapipes.explain` and `pandapipes.explain.enable()`.

---

## R3 — Only stdout is captured; stderr is excluded

The log capture tee wraps only `sys.stdout`. stderr is excluded to prevent the Python
traceback (written to stderr by the default excepthook) from appearing a second time
in the LLM's "Terminal Output" section.

---

## R4 — `row.get()` is forbidden on pandas Series

`pandas.Series.get()` was removed in pandas 2.0. Always use:
```python
value = row[key] if key in row.index else default
```
or the helper `_get(row, key, default)` already defined in `prompt.py`.

---

## R5 — No mandatory Python dependencies

`pip install pandapipes[explain]` must not pull in any additional Python packages.
All HTTP calls use `urllib` (stdlib). Ollama is a system-level requirement, not a
Python dependency. If a feature needs a new package, it must be opt-in or removed.

---

## R6 — Token budget must fit 8192 num_ctx

Total prompt characters must stay under ~19 400 chars worst-case (~4 850 tokens) to leave
room for the model's response within the 8192 num_ctx limit. Current section limits:

| Section | Limit |
|---|---|
| User file | 12 000 chars |
| Terminal logs | 2 000 chars |
| Library source | 2 000 chars |
| Knowledge | 800 chars |
| Full stack trace | 1 500 chars |
| Net stats | ~500 chars (uncapped, but bounded by network size) |
| System prompt | ~1 100 chars |

Adding a new section requires either a new limit or reducing an existing one.

---

## R7 — Path classification uses path components, not substrings

`"pandapipes" in path` would match `/home/user/pandapipes-tutorials/script.py`.
Always split on `/` and check membership in the parts set:
```python
parts = path.replace("\\", "/").split("/")
return "pandapipes" in parts
```
Because pandapipes depends on pandapower, `pandapower` is in `_LIBRARY_MARKERS` so its
frames are classified as library reference, not the user's code.

---

## R8 — The prompt orders sections by diagnostic value

The section order in `build()` is fixed and deliberate:
1. Error — always first (anchor)
2. Network Statistics — most diagnostic signal for fluid-network errors
3. Full Stack Trace — the call chain
4. User Code — full file when ≤ 12 000 chars, window otherwise
5. Terminal Output — stdout only (see R3)
6. Library Source — for reference, labeled "fix is NOT here"
7. Diagnostic Checklist — curated knowledge

Do not reorder sections without updating the system prompt and validating with
a representative set of failing networks.

---

## R9 — install() is idempotent; uninstall() resets fully

`install()` called twice must not double-register the IPython handler or save
a stale excepthook reference. Guard with `if _installed: return`.
`uninstall()` must reset `_saved_excepthook = None` and `_installed = False`.

---

## R10 — Knowledge pack files must use real pandapipes exception and API names

Every `## Keywords` section that covers a solver failure must include the exact Python
exception class name as it appears in `pandapipes/pf/pipeflow_setup.py`:
- `PipeflowNotConverged` (raised by `pipeflow()` on hydraulic/heat/bidirectional failure)

Knowledge packs may reference `pandapipes.diagnostic.diagnostic(net, report=False)`.
Do not reference pandapower diagnostics or electrical checks. Topology-specific packs may
also use `pandapipes.topology.unsupplied_junctions(net)` and the `check_connectivity`
pipeflow option.

Verify exception by running:
`grep -rn "class.*NotConverged" pandapipes/pf/pipeflow_setup.py`

---

## R11 — LLM timeout is 60 seconds

`urllib.request.urlopen(req, timeout=60)`. This is a blocking call on the main
thread. 60 s is sufficient for a small local model on CPU. Do not increase this — a hung
Ollama would freeze the user's process for that duration. Print a status line to
stderr before the call so the user knows what is happening.

---

## R12 — No comments that describe WHAT the code does

Comments must explain WHY: hidden constraints, non-obvious invariants, workarounds
for specific bugs. Never write `# loop through frames` or `# build prompt`. Good
names make those unnecessary.

---

## R13 — Context manager must not double-print the traceback

Inside `with explain():`, the context manager prints the traceback once. Before
re-raising, it must temporarily disable `get_config().enabled` so the re-raise
does not trigger `sys.excepthook` a second time (which would print the traceback
again and fire the dispatcher again).

---

## R14 — Use pandapipes units and components, never pandapower ones

This feature was ported from pandapower-explain. All diagnostic text, the system prompt,
the net-stats summary, and the knowledge packs must use pandapipes terminology:
- Components: junction, pipe, sink, source, ext_grid, pump, valve, press_control (NOT
  bus, line, load, gen, sgen, trafo).
- Units: pressure in bar, mass flow in kg/s, length in km, diameter in mm, temperature
  in K, height in m (NOT MW, MVAr, kV).
- Run function: `pp.pipeflow(net)` (NOT `pp.runpp(net)`).
- The ext_grid is a pressure/temperature reference (slack node), not an electrical slack.

---

## Checklist for code review

- [ ] All new functions in the explain pipeline are wrapped in `try/except`
- [ ] No `pandas.Series.get()` calls (use `_get()` helper)
- [ ] No new mandatory pip dependencies
- [ ] Total prompt chars stay within the R6 budget
- [ ] Knowledge pack keywords include `PipeflowNotConverged` where relevant
- [ ] Knowledge pack diagnostics references point to `pandapipes.diagnostic`, not pandapower
- [ ] `install()` is guarded against double-call
- [ ] `uninstall()` resets all state
- [ ] Only stdout and Python warnings (not full stderr tracebacks) are captured
- [ ] pandapipes components and units only — no leftover pandapower terms
