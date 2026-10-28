"""
Prompt builder — assembles the system + user message list sent to Ollama.

Token budget (chars ÷ 4 ≈ tokens, worst-case with 8192 num_ctx):
  System prompt  :  ~1100 chars  →  ~275 tokens
  Net stats      :   ~500 chars  →  ~125 tokens
  Traceback text :   ~600 chars  →  ~150 tokens
  User file      : ~12000 chars  → ~3000 tokens   (full file if ≤ 12 000 chars)
  Terminal logs  :  ~2000 chars  →  ~500 tokens
  Lib source     :  ~2000 chars  →  ~500 tokens
  Diagnostics    :  ~3500 chars  →  ~875 tokens
  Knowledge      :  ~1600 chars  →  ~400 tokens
  Overhead       :   ~400 chars  →  ~100 tokens
  Total worst    : ~19400 chars  → ~4850 tokens   (fits 8192 num_ctx with ~3300 tokens for response)
"""

from __future__ import annotations

import logging
import math

import pandapipes.topology as top
from pandapipes.explain.context.traceback_parser import SourceContext, read_full_source

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are a pandapipes expert diagnosing a fluid-network (gas/water/heat) simulation error.

You will be given these sections IN ORDER:
  1. Error — the exception class and message
  2. Network Statistics — actual numbers from the network object
  3. Diagnostic Findings — pandapipes preflight and hypothesis checks
  4. Full Stack Trace — the call chain that led to the exception
  5. User Code — the engineer's script (>> marks the exact error line)
  6. Terminal Output — what was printed before the crash
  7. pandapipes Source — internal library code (for reference only)
  8. Diagnostic Checklist — known patterns for this error type

BEFORE WRITING YOUR ANSWER, silently (do not output these steps):
  • Note the exact exception class and message from section 1 — this IS the crash that happened.
  • Treat Diagnostic Findings as stronger evidence than generic solver text.
  • In a closed-loop heating network, an active circ_pump_const_pressure
    (net.circ_pump_pressure with in_service=True) provides the pressure reference.
    In that case, absence of an ext_grid is not itself an error and should not
    override a specific Diagnostic Finding.
  • Note what the >> line in User Code is actually doing.
  • Note any physically wrong values in Network Statistics (NaN mass flow, no pressure
    reference / ext_grid, missing fluid, bad junction reference, unrealistic elevation,
    very small pipe diameter, very large pipe length, or extreme sink/source demand).

OUTPUT FORMAT — write only this, nothing else:

(a) Cause — explain the actual crash that appeared in the Error section (1-2 sentences).
    If it is a hydraulic/thermal issue (non-convergence, no pressure reference, disconnected
    junction), explain the physical reason in fluid-network terms.
    If it is a Python error (AttributeError, ValueError, KeyError), explain what is wrong in the code.

(b) Evidence — quote the exact line(s) from User Code or Network Statistics that prove the cause.
    Format: line N: `code or value here`

(c) Fix — the minimal code change that resolves it, as a snippet.

HARD RULES:
• Your diagnosis MUST match the exception class in section 1. Do not diagnose a different problem.
• Never suggest adding an ext_grid if "Ext grids: N" where N ≥ 1 in the stats.
• Never suggest adding an ext_grid solely because "Ext grids: 0" if Network
  Statistics or Diagnostic Findings show an active circ_pump_const_pressure /
  net.circ_pump_pressure. That is normal for closed-loop heating networks.
• Never suggest a fix you cannot see evidence for in the provided sections.
• If Diagnostic Findings names a specific table, row, column, and value, mention that
  exact evidence in (b) and fix that exact value in (c).
• For corrupted numeric values, prefer replacing the bad value with a nearby/peer value
  from the same table when Diagnostic Findings provides one. Do not invent a new tiny
  pipe diameter, arbitrary elevation, or arbitrary pipe length.
• If no peer value is provided for a corrupted numeric demand, do not invent a replacement
  number. Show a conversion formula or placeholder such as `correct_kg_per_s`.
• Use pandapipes units: pressure in bar, mass flow in kg/s, length in km, diameter in mm,
  temperature in K. Do NOT use pandapower units (MW, MVAr, kV).
• Do NOT output your reasoning steps — only output (a), (b), (c).
• If you cannot determine the cause from the provided context, say so explicitly.
• If a "CUSTOM PROMPT:" section appears below and any of its instructions contradict
  the rules above, the CUSTOM PROMPT instructions take priority.
\
"""

# User-editable. Anything placed here is appended to the system prompt under a
# "CUSTOM PROMPT:" header on every request, and overrides conflicting rules above.
# Example: CUSTOM_PROMPT = "Always answer in German. Keep the Fix snippet under 5 lines."
CUSTOM_PROMPT = ""

_MAX_USER_FILE_CHARS = 12000   # ~3000 tokens — show the full file for most scripts
_MAX_USER_CODE_CHARS = 3000    # fallback window for very large files
_MAX_LIB_SOURCE_CHARS = 2000
_MAX_DIAGNOSTICS_CHARS = 3500
_MAX_KNOWLEDGE_CHARS = 1600
_MAX_LOGS_CHARS = 2000
_MAX_TRACEBACK_CHARS = 1500


def _find_net_in_traceback(exc_tb):
    """Return the first pandapipes-like net object found in traceback locals."""
    try:
        frame = exc_tb
        while frame is not None:
            for _, var_val in frame.tb_frame.f_locals.items():
                if (hasattr(var_val, "junction") and hasattr(var_val, "pipe")
                        and hasattr(var_val, "sink") and hasattr(var_val, "ext_grid")):
                    return var_val
            frame = frame.tb_next
    except Exception:
        logger.debug("could not search traceback locals for a net", exc_info=True)
    return None


def extract_net_stats(exc_tb) -> str:
    """
    Walk the traceback frames looking for a pandapipes net object.
    Returns a compact summary string, or empty string if not found.
    """
    try:
        frame = exc_tb
        while frame is not None:
            for var_name, var_val in frame.tb_frame.f_locals.items():
                if var_val is _find_net_in_traceback(exc_tb):
                    return _summarise_net(var_name, var_val)
            frame = frame.tb_next
    except Exception:
        pass
    return ""


def extract_diagnostics(exc_tb, error_type: str = "", error_msg: str = "") -> str:
    """
    Run low-cost structural checks plus selected pandapipes diagnostic hypotheses.

    This produces deterministic evidence for small local models before the LLM sees the
    generic non-convergence message.
    """
    try:
        net = _find_net_in_traceback(exc_tb)
        if net is None:
            return ""

        lines = ["## Diagnostic Findings"]
        lines.extend(_preflight_findings(net))
        lines.extend(_run_selected_diagnostic_checks(net, error_type, error_msg))

        if len(lines) == 1:
            lines.append("- No specific preflight or diagnostic finding was detected.")
        return "\n".join(lines)
    except Exception:
        return ""


def _get(row, key: str, default="?"):
    """Safe column access on a pandas Series (compatible with pandas 2.0+)."""
    return row[key] if key in row.index else default


def _fluid_name(net) -> str:
    """Return the configured fluid name, or '(none set)' if no fluid is defined."""
    try:
        fluid = net.get("fluid", None) if hasattr(net, "get") else getattr(net, "fluid", None)
        if fluid is None:
            return "(none set)"
        return getattr(fluid, "name", str(fluid))
    except Exception:
        return "(unknown)"


def _summarise_net(var_name: str, net) -> str:
    """Build a plain-text summary of the pandapipes network."""
    try:
        fluid = _fluid_name(net)
        has_circ_pressure_ref = _has_active_circ_pump_pressure(net)
        lines = [f"## Network Statistics (variable: `{var_name}`)"]
        lines.append(f"  Fluid       : {fluid}")
        lines.append(f"  Junctions   : {len(net.junction)}")
        lines.append(f"  Pipes       : {len(net.pipe)}")
        lines.append(f"  Sinks       : {len(net.sink)}")
        lines.append(f"  Sources     : {len(net.source)}")
        if len(net.ext_grid) == 0 and has_circ_pressure_ref:
            lines.append(
                "  Ext grids   : 0  (ok for closed-loop heating because an active "
                "circ_pump_const_pressure provides the pressure reference)"
            )
        else:
            lines.append(f"  Ext grids   : {len(net.ext_grid)}  (pressure/temperature references — slack nodes)")
        for tbl, label in (("pump", "Pumps"), ("valve", "Valves"),
                           ("press_control", "Pressure controls"),
                           ("circ_pump_pressure", "Circ pressure pumps"),
                           ("circ_pump_mass", "Circ mass pumps"),
                           ("heat_exchanger", "Heat exchangers")):
            if tbl in net and not net[tbl].empty:
                lines.append(f"  {label:<12}: {len(net[tbl])}")

        # Explicit warnings — flagged at the top so the model cannot miss them
        # regardless of section ordering. These are the dominant causes of
        # PipeflowNotConverged in pandapipes.
        warnings = []
        if fluid in ("(none set)", "(unknown)"):
            warnings.append(
                "  *** WARNING: no fluid is defined on the net — pipeflow requires a fluid "
                "(e.g. create_fluid_from_lib(net, 'water') or pass fluid= to create_empty_network) ***"
            )
        if len(net.ext_grid) == 0 and not has_circ_pressure_ref:
            warnings.append(
                "  *** WARNING: no ext_grid — there is no pressure reference (slack node). "
                "Every network needs at least one ext_grid that fixes a pressure ***"
            )
        elif len(net.ext_grid) == 0 and has_circ_pressure_ref:
            warnings.append(
                "  *** NOTE: no ext_grid is present, but this closed-loop network has an active "
                "circ_pump_const_pressure, so do not diagnose missing ext_grid without stronger evidence ***"
            )
        for tbl in ("sink", "source"):
            df = net[tbl] if tbl in net else None
            if df is not None and not df.empty and 'mdot_kg_per_s' in df.columns:
                nan_rows = df[df['mdot_kg_per_s'].isna()]
                for _, row in nan_rows.iterrows():
                    warnings.append(
                        f"  *** WARNING: {tbl} '{_get(row,'name')}' has mdot_kg_per_s = NaN "
                        f"(missing/unset value) — this will crash the solver ***"
                    )
        if not net.pipe.empty:
            if 'inner_diameter_mm' in net.pipe.columns:
                bad_d = net.pipe[net.pipe['inner_diameter_mm'] <= 0]
                for _, row in bad_d.iterrows():
                    warnings.append(
                        f"  *** WARNING: pipe '{_get(row,'name')}' has inner_diameter_mm = "
                        f"{_get(row,'inner_diameter_mm')} (must be > 0) ***"
                    )
                small_d = net.pipe[(net.pipe['inner_diameter_mm'] > 0) & (net.pipe['inner_diameter_mm'] < 20)]
                for idx, row in small_d.head(5).iterrows():
                    warnings.append(
                        f"  *** WARNING: pipe index {idx} has very small inner_diameter_mm = "
                        f"{_get(row,'inner_diameter_mm')} mm — this can make the hydraulic matrix singular ***"
                    )
            if 'length_km' in net.pipe.columns:
                bad_l = net.pipe[net.pipe['length_km'] <= 0]
                for _, row in bad_l.iterrows():
                    warnings.append(
                        f"  *** WARNING: pipe '{_get(row,'name')}' has length_km = "
                        f"{_get(row,'length_km')} (must be > 0) ***"
                    )
                long_l = net.pipe[net.pipe['length_km'] > 50]
                for idx, row in long_l.head(5).iterrows():
                    warnings.append(
                        f"  *** WARNING: pipe index {idx} has very large length_km = "
                        f"{_get(row,'length_km')} km — check whether metres were entered as km ***"
                    )
        if 'height_m' in net.junction.columns and not net.junction.empty:
            height = net.junction['height_m']
            extreme_h = net.junction[height.abs() > 1000]
            for idx, row in extreme_h.head(5).iterrows():
                warnings.append(
                    f"  *** WARNING: junction index {idx} has extreme height_m = "
                    f"{_get(row,'height_m')} m — elevation affects hydrostatic pressure ***"
                )
            if not height.empty and height.max() - height.min() > 1000:
                warnings.append(
                    f"  *** WARNING: junction height range is {height.min()}..{height.max()} m; "
                    "large elevation differences can dominate pressure balance ***"
                )
        lines.extend(warnings)

        if not net.sink.empty:
            total_sink = net.sink.mdot_kg_per_s.sum()
            s_str = f"{total_sink:.4f}" if not math.isnan(total_sink) else "NaN"
            lines.append(f"  Total sink  : {s_str} kg/s (fluid withdrawn)")
            for _, row in net.sink.iterrows():
                name = _get(row, 'name')
                m = row.mdot_kg_per_s
                m_val = f"{m:.4f}" if not math.isnan(float(m)) else "NaN !!!"
                lines.append(f"    sink '{name}' @ junction {_get(row,'junction')}: {m_val} kg/s")

        if not net.source.empty:
            total_src = net.source.mdot_kg_per_s.sum()
            src_str = f"{total_src:.4f}" if not math.isnan(total_src) else "NaN"
            lines.append(f"  Total source: {src_str} kg/s (fluid fed in)")

        if not net.ext_grid.empty:
            for _, row in net.ext_grid.iterrows():
                name = _get(row, 'name')
                lines.append(
                    f"  ext_grid '{name}' @ junction {_get(row,'junction')}: "
                    f"p={_get(row,'p_bar')} bar, t={_get(row,'t_k')} K, type={_get(row,'type')}"
                )

        return "\n".join(lines)
    except Exception:
        return ""


def _preflight_findings(net) -> list[str]:
    lines: list[str] = []
    try:
        fluid = _fluid_name(net)
        if fluid in ("(none set)", "(unknown)"):
            lines.append("- CRITICAL: no fluid is defined on the net.")
        has_circ_pressure_ref = _has_active_circ_pump_pressure(net)
        if hasattr(net, "ext_grid") and net.ext_grid.empty and not has_circ_pressure_ref:
            lines.append("- CRITICAL: net.ext_grid is empty, so there is no pressure reference.")
        elif hasattr(net, "ext_grid") and net.ext_grid.empty and has_circ_pressure_ref:
            lines.append(
                "- INFO: net.ext_grid is empty, but an active circ_pump_const_pressure "
                "(net.circ_pump_pressure) provides the pressure reference for a closed-loop heating network."
            )

        lines.extend(_find_missing_junction_references(net))
        lines.extend(_find_unsupplied_junctions(net))
        lines.extend(_find_suspicious_pipe_values(net))
        lines.extend(_find_suspicious_junction_values(net))
        lines.extend(_find_suspicious_ext_grid_values(net))
        lines.extend(_find_suspicious_sink_source_values(net))
        lines.extend(_find_suspicious_component_values(net))
    except Exception:
        pass
    return lines


def _find_unsupplied_junctions(net) -> list[str]:
    lines: list[str] = []
    try:
        if hasattr(net, "ext_grid") and net.ext_grid.empty and _has_active_circ_pump_pressure(net):
            return lines
        unsupplied = sorted(int(j) for j in top.unsupplied_junctions(net))
        if unsupplied:
            shown = unsupplied[:10]
            suffix = "" if len(unsupplied) <= 10 else f" ... ({len(unsupplied)} total)"
            lines.append(
                f"- CRITICAL: unsupplied junctions have no path to an ext_grid: {shown}{suffix}."
            )
    except Exception:
        pass
    return lines


def _find_missing_junction_references(net) -> list[str]:
    lines: list[str] = []
    try:
        junction_index = set(net.junction.index)
        checks = {
            "pipe": ("from_junction", "to_junction"),
            "sink": ("junction",),
            "source": ("junction",),
            "ext_grid": ("junction",),
            "pump": ("from_junction", "to_junction"),
            "compressor": ("from_junction", "to_junction"),
            "press_control": ("from_junction", "to_junction", "controlled_junction"),
            "flow_control": ("from_junction", "to_junction"),
            "circ_pump_pressure": ("from_junction", "to_junction", "return_junction", "flow_junction"),
            "circ_pump_mass": ("from_junction", "to_junction", "return_junction", "flow_junction"),
            "heat_exchanger": ("from_junction", "to_junction"),
            "heat_consumer": ("from_junction", "to_junction"),
            "valve": ("junction", "from_junction", "to_junction"),
        }
        for table, columns in checks.items():
            if not hasattr(net, table):
                continue
            df = net[table]
            if df.empty:
                continue
            for col in columns:
                if col not in df.columns:
                    continue
                bad = df[~df[col].isin(junction_index)]
                for idx, row in bad.head(5).iterrows():
                    lines.append(
                        f"- CRITICAL: net.{table}.loc[{idx}, '{col}'] = {_get(row, col)} "
                        f"but only junction indices {_index_range_text(net.junction.index)} exist."
                    )

        if hasattr(net, "valve") and {"element", "et"}.issubset(net.valve.columns):
            pipe_index = set(net.pipe.index) if hasattr(net, "pipe") else set()
            for idx, row in net.valve.iterrows():
                if row["et"] == "ju" and row["element"] not in junction_index:
                    lines.append(
                        f"- CRITICAL: net.valve.loc[{idx}, 'element'] = {row['element']} "
                        "but et='ju' requires an existing junction index."
                    )
                elif row["et"] == "pi" and row["element"] not in pipe_index:
                    lines.append(
                        f"- CRITICAL: net.valve.loc[{idx}, 'element'] = {row['element']} "
                        "but et='pi' requires an existing pipe index."
                    )
    except Exception:
        pass
    return lines


def _find_suspicious_pipe_values(net) -> list[str]:
    lines: list[str] = []
    try:
        if not hasattr(net, "pipe") or net.pipe.empty:
            return lines
        if "inner_diameter_mm" in net.pipe.columns:
            peer_diameters = _peer_values(net.pipe, "inner_diameter_mm", lambda s: s >= 20)
            small = net.pipe[net.pipe.inner_diameter_mm < 20]
            for idx, row in small.head(8).iterrows():
                severity = "CRITICAL" if row.inner_diameter_mm <= 0 else "SUSPECT"
                lines.append(
                    f"- {severity}: net.pipe.loc[{idx}, 'inner_diameter_mm'] = "
                    f"{row.inner_diameter_mm} mm. Tiny diameters cause huge pressure losses and singular matrices."
                    f"{peer_diameters}"
                )
        if "length_km" in net.pipe.columns:
            peer_lengths = _peer_values(net.pipe, "length_km", lambda s: (s > 0) & (s <= 50))
            bad = net.pipe[(net.pipe.length_km <= 0) | (net.pipe.length_km > 50)]
            for idx, row in bad.head(8).iterrows():
                severity = "CRITICAL" if row.length_km <= 0 else "SUSPECT"
                lines.append(
                    f"- {severity}: net.pipe.loc[{idx}, 'length_km'] = {row.length_km} km. "
                    f"Check for m/km unit mistakes or an unrealistic hydraulic resistance.{peer_lengths}"
                )
        if "k_mm" in net.pipe.columns:
            rough = net.pipe[net.pipe.k_mm > 0.5]
            for idx, row in rough.head(5).iterrows():
                lines.append(
                    f"- SUSPECT: net.pipe.loc[{idx}, 'k_mm'] = {row.k_mm} mm; roughness is very high."
                )
    except Exception:
        pass
    return lines


def _find_suspicious_junction_values(net) -> list[str]:
    lines: list[str] = []
    try:
        if not hasattr(net, "junction") or net.junction.empty:
            return lines
        for col in ("pn_bar", "tfluid_k"):
            if col in net.junction.columns:
                bad = net.junction[net.junction[col] <= 0]
                for idx, row in bad.head(5).iterrows():
                    lines.append(f"- CRITICAL: net.junction.loc[{idx}, '{col}'] = {row[col]} is unphysical.")
        if "height_m" in net.junction.columns:
            heights = net.junction.height_m
            peer_heights = _peer_values(net.junction, "height_m", lambda s: s.abs() <= 1000)
            extreme = net.junction[heights.abs() > 1000]
            for idx, row in extreme.head(8).iterrows():
                lines.append(
                    f"- SUSPECT: net.junction.loc[{idx}, 'height_m'] = {row.height_m} m. "
                    f"This elevation can dominate hydrostatic pressure.{peer_heights}"
                )
            if len(heights) and heights.max() - heights.min() > 1000:
                lines.append(
                    f"- SUSPECT: junction height range is {heights.min()}..{heights.max()} m."
                )
    except Exception:
        pass
    return lines


def _find_suspicious_ext_grid_values(net) -> list[str]:
    lines: list[str] = []
    try:
        if not hasattr(net, "ext_grid") or net.ext_grid.empty:
            return lines
        for col in ("p_bar", "t_k"):
            if col in net.ext_grid.columns:
                bad = net.ext_grid[net.ext_grid[col].isna() | (net.ext_grid[col] <= 0)]
                for idx, row in bad.head(5).iterrows():
                    lines.append(f"- CRITICAL: net.ext_grid.loc[{idx}, '{col}'] = {row[col]} is unphysical.")
        if "p_bar" in net.ext_grid.columns:
            extreme = net.ext_grid[net.ext_grid.p_bar > 1000]
            for idx, row in extreme.head(5).iterrows():
                lines.append(f"- SUSPECT: net.ext_grid.loc[{idx}, 'p_bar'] = {row.p_bar} bar is extremely high.")
    except Exception:
        pass
    return lines


def _find_suspicious_sink_source_values(net) -> list[str]:
    lines: list[str] = []
    try:
        for table in ("sink", "source"):
            if not hasattr(net, table):
                continue
            df = net[table]
            if df.empty:
                continue
            if "mdot_kg_per_s" in df.columns:
                bad = df[df.mdot_kg_per_s.isna() | (df.mdot_kg_per_s < 0) | (df.mdot_kg_per_s > 100)]
                for idx, row in bad.head(8).iterrows():
                    value = row.mdot_kg_per_s
                    if math.isnan(value):
                        why = "is NaN"
                        severity = "CRITICAL"
                    elif value < 0:
                        why = "is negative; use the opposite element type instead"
                        severity = "CRITICAL"
                    else:
                        why = (
                            "is extremely large for kg/s; check kg/h or volumetric-flow conversion. "
                            "Recommended fix form: use the known correct demand, e.g. "
                            f"`net.{table}.loc[{idx}, 'mdot_kg_per_s'] = correct_kg_per_s`, "
                            "or if the source value is kg/h, divide it by 3600. Do not invent 100.0 kg/s"
                        )
                        severity = "SUSPECT"
                    lines.append(f"- {severity}: net.{table}.loc[{idx}, 'mdot_kg_per_s'] = {value} {why}.")
            if "scaling" in df.columns:
                bad_scaling = df[(df.scaling < 0) | (df.scaling > 100)]
                for idx, row in bad_scaling.head(5).iterrows():
                    lines.append(f"- SUSPECT: net.{table}.loc[{idx}, 'scaling'] = {row.scaling}.")
    except Exception:
        pass
    return lines


def _find_suspicious_component_values(net) -> list[str]:
    lines: list[str] = []
    try:
        if hasattr(net, "valve") and not net.valve.empty and "opened" in net.valve.columns:
            closed = net.valve[~net.valve.opened.astype(bool)]
            for idx, row in closed.head(8).iterrows():
                lines.append(f"- SUSPECT: net.valve.loc[{idx}, 'opened'] = {row.opened}; closed valves can isolate a subnet.")
        if hasattr(net, "pipe") and not net.pipe.empty and "u_w_per_m2k" in net.pipe.columns:
            high_u = net.pipe[net.pipe.u_w_per_m2k > 50]
            for idx, row in high_u.head(5).iterrows():
                lines.append(
                    f"- SUSPECT: net.pipe.loc[{idx}, 'u_w_per_m2k'] = {row.u_w_per_m2k} W/(m2 K), "
                    "which is an extremely high heat-transfer coefficient."
                )
        if hasattr(net, "compressor") and not net.compressor.empty and "pressure_ratio" in net.compressor.columns:
            high_ratio = net.compressor[net.compressor.pressure_ratio > 5]
            for idx, row in high_ratio.head(5).iterrows():
                lines.append(f"- SUSPECT: net.compressor.loc[{idx}, 'pressure_ratio'] = {row.pressure_ratio}.")
        if hasattr(net, "circ_pump_mass") and not net.circ_pump_mass.empty and "mdot_flow_kg_per_s" in net.circ_pump_mass.columns:
            high_mdot = net.circ_pump_mass[net.circ_pump_mass.mdot_flow_kg_per_s > 100]
            for idx, row in high_mdot.head(5).iterrows():
                lines.append(f"- SUSPECT: net.circ_pump_mass.loc[{idx}, 'mdot_flow_kg_per_s'] = {row.mdot_flow_kg_per_s}.")
        for table in ("pump", "compressor", "press_control", "flow_control", "circ_pump_pressure", "circ_pump_mass"):
            if not hasattr(net, table):
                continue
            df = net[table]
            if df.empty or "in_service" not in df.columns:
                continue
            off = df[~df.in_service.astype(bool)]
            for idx, row in off.head(5).iterrows():
                lines.append(f"- SUSPECT: net.{table}.loc[{idx}, 'in_service'] = {row.in_service}.")
    except Exception:
        pass
    return lines


def _peer_values(df, column: str, valid_mask_factory) -> str:
    try:
        valid = df.loc[valid_mask_factory(df[column]), column].dropna()
        if valid.empty:
            return ""
        values = sorted({float(v) for v in valid})
        shown = ", ".join(f"{v:g}" for v in values[:5])
        return f" Peer {column} values elsewhere in the same table: {shown}."
    except Exception:
        return ""


def _has_active_circ_pump_pressure(net) -> bool:
    try:
        if not hasattr(net, "circ_pump_pressure") or net.circ_pump_pressure.empty:
            return False
        df = net.circ_pump_pressure
        if "in_service" not in df.columns:
            return True
        return bool(df.in_service.astype(bool).any())
    except Exception:
        return False


def _run_selected_diagnostic_checks(net, error_type: str, error_msg: str) -> list[str]:
    lines: list[str] = []
    if error_type != "PipeflowNotConverged" and "converge" not in error_msg.lower():
        return lines
    try:
        from pandapipes.diagnostic.diagnostic import Diagnostic
        from pandapipes.diagnostic.diagnostic_functions import (
            InvalidValuesCheck,
            MissingNodeJunctionsCheck,
            MissingBranchJunctionsCheck,
            PipeLengthCheck,
            SinkSourceScalingCheck,
            PipeDiameterCheck,
            JunctionHeightCheck,
            PipeRoughnessCheck,
            HeatTransferCoefficientCheck,
            ValveConfigurationCheck,
            MissingPressureReferenceCheck,
            CompressorPressureRatioCheck,
            InactivePressureControlsCheck,
            CircPumpMassFlowCheck,
            default_argument_values,
        )

        diag = Diagnostic(add_default_functions=False)
        diag.kwargs.update(default_argument_values.copy())
        selected = [
            ("invalid_values", InvalidValuesCheck(), []),
            ("missing_node_junctions", MissingNodeJunctionsCheck(), []),
            ("missing_branch_junctions", MissingBranchJunctionsCheck(), []),
            ("pipe_length", PipeLengthCheck(), None),
            ("sink_source_scaling", SinkSourceScalingCheck(), None),
            ("pipe_diameter", PipeDiameterCheck(), None),
            ("junction_height", JunctionHeightCheck(), []),
            ("pipe_roughness", PipeRoughnessCheck(), None),
            ("heat_transfer_coefficient", HeatTransferCoefficientCheck(), None),
            ("valve_configuration", ValveConfigurationCheck(), []),
            ("missing_pressure_reference", MissingPressureReferenceCheck(), []),
            ("compressor_pressure_ratio", CompressorPressureRatioCheck(), None),
            ("inactive_pressure_controls", InactivePressureControlsCheck(), []),
            ("circ_pump_mass_flow", CircPumpMassFlowCheck(), None),
        ]
        for name, check, args in selected:
            diag.register_function(check, args, name)
        results = diag.diagnose_network(net, report=False, return_result_dict=True)
        formatted = _format_diagnostic_results(results)
        if formatted:
            lines.append("- pandapipes.diagnostic selected checks found:")
            lines.extend(formatted)
        if diag.diag_errors:
            for name, err in list(diag.diag_errors.items())[:3]:
                lines.append(f"- Diagnostic check '{name}' could not run cleanly: {type(err).__name__}: {err}")
    except ImportError:
        # The diagnostic API changed upstream (renamed/removed check). Do not hide this.
        logger.warning("pandapipes.explain: could not import pandapipes.diagnostic checks.", exc_info=True)
    except Exception:
        logger.debug("pandapipes.explain: diagnostic checks failed.", exc_info=True)
    return lines


def _format_diagnostic_results(results: dict) -> list[str]:
    lines: list[str] = []
    for name, result in results.items():
        if result is None or _is_false_result(result):
            continue
        if name == "invalid_values":
            for table, violations in result.items():
                for idx, col, value, restriction in violations[:8]:
                    lines.append(
                        f"  - invalid_values: net.{table}.loc[{idx}, '{col}'] = {value} "
                        f"violates {restriction}."
                    )
        elif name in ("missing_node_junctions", "missing_branch_junctions"):
            lines.append(f"  - {name}: {result}")
        elif name == "pipe_length" and _is_true_result(result):
            lines.append("  - pipe_length: pipeflow converged after replacing pipe lengths with a standard short length.")
        elif name == "sink_source_scaling" and isinstance(result, dict):
            positives = [key for key, value in result.items() if value]
            for key in positives:
                lines.append(f"  - sink_source_scaling: pipeflow converged after scaling {key} mass flows down.")
        elif name == "pipe_diameter" and _is_true_result(result):
            lines.append("  - pipe_diameter: pipeflow converged after increasing very small pipe diameters.")
        elif name == "junction_height" and _is_true_result(result):
            lines.append("  - junction_height: pipeflow converged after flattening all junction heights to 0 m.")
        elif name == "pipe_roughness":
            lines.append(f"  - pipe_roughness: {result}")
        elif name == "heat_transfer_coefficient" and _is_true_result(result):
            lines.append("  - heat_transfer_coefficient: pipeflow converged after reducing high pipe heat-transfer coefficients.")
        elif name == "valve_configuration" and isinstance(result, dict):
            if result.get("all_open"):
                lines.append("  - valve_configuration: pipeflow converged after opening all valves.")
            if result.get("all_closed"):
                lines.append("  - valve_configuration: pipeflow converged after closing all valves.")
        elif name == "missing_pressure_reference" and _is_true_result(result):
            lines.append(
                "  - missing_pressure_reference: no in-service ext_grid, circ_pump_pressure or "
                "circ_pump_mass of type 'p' or 'pt' fixes the pressure level."
            )
        elif name == "compressor_pressure_ratio" and _is_true_result(result):
            lines.append("  - compressor_pressure_ratio: at least one compressor pressure_ratio is above the configured limit.")
        elif name == "inactive_pressure_controls" and _is_true_result(result):
            lines.append(
                "  - inactive_pressure_controls: pipeflow converged after all active pressure controls "
                "were deactivated. The pressure-control configuration is therefore a strong candidate "
                "for the non-convergence."
            )
        elif name == "circ_pump_mass_flow" and _is_true_result(result):
            lines.append("  - circ_pump_mass_flow: pipeflow converged after reducing circ_pump_mass mdot_flow_kg_per_s.")
        elif _is_true_result(result):
            lines.append(f"  - {name}: hypothesis check returned True.")
        elif isinstance(result, dict) and any(bool(v) for v in result.values()):
            lines.append(f"  - {name}: {result}")
    return lines


def _is_true_result(result) -> bool:
    return _is_bool_scalar(result) and bool(result)


def _is_false_result(result) -> bool:
    return _is_bool_scalar(result) and not bool(result)


def _is_bool_scalar(result) -> bool:
    if isinstance(result, bool):
        return True
    result_type = type(result)
    return result_type.__module__ == "numpy" and result_type.__name__ in {"bool", "bool_"}


def _index_range_text(index) -> str:
    try:
        values = list(index)
        if not values:
            return "[]"
        return f"{min(values)}..{max(values)} (count {len(values)})"
    except Exception:
        return str(index)


def build(
    error_type: str,
    error_msg: str,
    user_contexts: list[SourceContext],
    lib_contexts: list[SourceContext],
    knowledge: str,
    net_stats: str = "",
    diagnostics: str = "",
    terminal_logs: str = "",
    traceback_text: str = "",
) -> list[dict]:
    """
    Assemble the messages list for the Ollama chat API.

    Returns [{role: system, content: ...}, {role: user, content: ...}].
    """
    parts: list[str] = []

    # 1. Error — always first, anchors all subsequent reasoning.
    parts.append(f"## Error\n**{error_type}**: {error_msg}")

    # 2. Network statistics — most diagnostic signal for fluid-network errors.
    if net_stats:
        parts.append(net_stats)

    # 3. Deterministic diagnostics — exact table evidence and selected
    #    what-if checks, before generic traceback/prompt context.
    if diagnostics:
        parts.append(_truncate(diagnostics, _MAX_DIAGNOSTICS_CHARS))

    # 4. Full stack trace — shows the call chain so the model understands
    #    which user line triggered which pandapipes internals.
    if traceback_text:
        tb = _truncate(traceback_text, _MAX_TRACEBACK_CHARS)
        parts.append(f"## Full Stack Trace\n```\n{tb}\n```")

    # 5. User code — full file when ≤ 12 000 chars (covers most scripts),
    #    large-window snippet otherwise. ">>" marks the exact error line.
    if user_contexts:
        ctx = user_contexts[-1]  # innermost user frame = closest to the error
        full = read_full_source(ctx.file_path, ctx.lineno, _MAX_USER_FILE_CHARS)
        if full:
            parts.append(
                f"## User Code (full file — `>>` marks the error line)\n"
                f"File: `{ctx.file_path}` | Line: {ctx.lineno} | Function: `{ctx.function_name}`\n"
                f"```python\n{full}\n```"
            )
        else:
            snippet = _truncate(ctx.source_snippet, _MAX_USER_CODE_CHARS)
            parts.append(
                f"## User Code (window ±50 lines around error)\n"
                f"File: `{ctx.file_path}` | Line: {ctx.lineno} | Function: `{ctx.function_name}`\n"
                f"```python\n{snippet}\n```"
            )

    # 6. Terminal stdout/warnings captured before the error.
    #    Stderr excluded — avoids embedding the traceback a second time.
    if terminal_logs:
        logs = _truncate(terminal_logs, _MAX_LOGS_CHARS)
        parts.append(f"## Terminal Output Before Error\n```\n{logs}\n```")

    # 7. Library source — deepest pandapipes frame.
    #    Absolute path from traceback resolves to wherever pandapipes is installed.
    if lib_contexts:
        ctx = lib_contexts[-1]
        snippet = _truncate(ctx.source_snippet, _MAX_LIB_SOURCE_CHARS)
        parts.append(
            f"## pandapipes Source (where exception was raised — the fix is NOT here)\n"
            f"File: `{ctx.file_path}` | Line: {ctx.lineno} | Function: `{ctx.function_name}`\n"
            f"```python\n{snippet}\n```"
        )

    # 8. Curated knowledge checklist for this error category.
    if knowledge:
        knowledge = _truncate(knowledge, _MAX_KNOWLEDGE_CHARS)
        parts.append(f"## Diagnostic Checklist (verify these likely causes)\n{knowledge}")

    user_content = "\n\n".join(parts)

    system_content = _SYSTEM
    if CUSTOM_PROMPT and CUSTOM_PROMPT.strip():
        system_content = f"{_SYSTEM}\nCUSTOM PROMPT:\n{CUSTOM_PROMPT.strip()}"

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def _truncate(text: str, max_chars: int) -> str:
    """Truncate to max_chars on a line boundary to avoid cutting mid-token."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit("\n", 1)[0] + "\n..."
