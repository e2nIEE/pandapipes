"""
Knowledge loader — reads curated Markdown files and returns relevant sections.

Matching strategy: keyword overlap between the error/diagnostic context and
markdown sections. No ML, no embeddings, instant.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Directory containing the curated .md files.
_PACK_DIR = Path(__file__).parent / "pack"

# Cache: filename → file content (loaded once on first use).
_cache: dict[str, str] = {}


def load_relevant(error_type: str, error_msg: str, max_chars: int = 1600, context: str = "") -> str:
    """
    Return a curated knowledge excerpt relevant to the given error.

    Parameters
    ----------
    error_type : str
        Exception class name, e.g. ``"PipeflowNotConverged"``.
    error_msg : str
        Exception message string.
    max_chars : int
        Truncate the returned excerpt to this many characters.
    context : str
        Optional diagnostic/network text used to retrieve more specific packs.

    Returns
    -------
    str
        Relevant knowledge text, or empty string if nothing matched.
    """
    _ensure_cache()

    # Build a set of keywords from the error — lowercase, split on non-alpha.
    raw_terms = f"{error_type} {error_msg} {context}".lower()
    keywords = set(re.split(r"[^a-z0-9]+", raw_terms)) - _STOP_WORDS

    required_files = _required_files(raw_terms)

    scored_sections = []
    for filename, content in _cache.items():
        file_score = _score(content.lower(), keywords)
        if filename in required_files:
            file_score += 8
        for heading, section_text in _iter_sections(content):
            if heading.lower() == "keywords":
                continue
            section_score = _score(f"{heading}\n{section_text}".lower(), keywords)
            score = section_score + min(file_score, 10)
            if filename in required_files:
                score += 12
            if score > 0:
                scored_sections.append((filename, heading, section_text, score))

    scored_sections.sort(key=lambda item: item[3], reverse=True)

    if not scored_sections:
        return ""

    excerpts: list[str] = []
    forced = _forced_sections(raw_terms)
    remaining = max_chars
    seen: set[tuple[str, str]] = set()
    for filename, heading, section_text in forced:
        if remaining <= 0:
            break
        key = (filename, heading)
        if key in seen:
            continue
        seen.add(key)
        text = _trim_section(section_text, max(250, min(550, remaining)))
        excerpts.append(f"### {filename} — {heading}\n{text}")
        remaining = max_chars - sum(len(item) + 2 for item in excerpts)

    for filename, heading, section_text, _ in scored_sections:
        if remaining <= 0:
            break
        key = (filename, heading)
        if key in seen:
            continue
        seen.add(key)

        text = _trim_section(section_text, max(250, min(650, remaining)))
        excerpts.append(f"### {filename} — {heading}\n{text}")
        remaining = max_chars - sum(len(item) + 2 for item in excerpts)
        if len(excerpts) >= 5:
            break

    text = "\n\n".join(excerpts)
    # Truncate cleanly on a line boundary.
    if len(text) > max_chars:
        text = text[:max_chars].rsplit("\n", 1)[0] + "\n..."

    return text


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "", "a", "an", "the", "is", "in", "at", "of", "to", "and",
    "or", "not", "no", "for", "with", "this", "that", "from",
    "by", "be", "are", "was", "were", "has", "have",
}


def _ensure_cache() -> None:
    """Load all .md files from the pack directory into memory (once)."""
    if _cache:
        return
    if not _PACK_DIR.exists():
        return
    for md_file in sorted(_PACK_DIR.glob("*.md")):
        try:
            _cache[md_file.name] = md_file.read_text(encoding="utf-8")
        except Exception:
            logger.debug("could not read knowledge file %s", md_file, exc_info=True)


def _score(content: str, keywords: set[str]) -> int:
    """Count how many keywords appear in `content`."""
    return sum(1 for kw in keywords if kw and len(kw) > 2 and kw in content)


def _iter_sections(content: str):
    """Yield (heading, text) chunks split by level-2 markdown headings."""
    current_heading = "Overview"
    current_lines: list[str] = []
    for line in content.splitlines():
        if line.startswith("## "):
            if current_lines:
                yield current_heading, "\n".join(current_lines).strip()
            current_heading = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        yield current_heading, "\n".join(current_lines).strip()


def _trim_section(section_text: str, budget: int) -> str:
    text = section_text.strip()
    if len(text) > budget:
        text = text[:budget].rsplit("\n", 1)[0] + "\n..."
    return text


def _forced_sections(raw_terms: str) -> list[tuple[str, str, str]]:
    forced_headings: list[tuple[str, str]] = []
    if "no fluid" in raw_terms or "fluid is defined" in raw_terms or "(none set)" in raw_terms:
        forced_headings.append(("04_fluid.md", "Cause 1: No fluid defined on the net"))
    if "inner_diameter_mm'] =" in raw_terms or 'inner_diameter_mm"] =' in raw_terms or "matrixrankwarning" in raw_terms:
        forced_headings.append(("02_invalid_values.md", "Cause 3: Unrealistically small but positive pipe diameter"))
    if "length_km'] =" in raw_terms or 'length_km"] =' in raw_terms or "pipe length" in raw_terms:
        forced_headings.append(("02_invalid_values.md", "Cause 4: Unrealistically large pipe length"))
    if "height_m'] =" in raw_terms or 'height_m"] =' in raw_terms or "elevation" in raw_terms:
        forced_headings.append(("01_convergence.md", "Cause 8: Extreme elevations dominate hydrostatic pressure"))
    if any(term in raw_terms for term in (
        "only junction indices", "violates existing_junction",
        "missing_branch_junctions", "missing_node_junctions",
        "to_junction'] =", 'to_junction"] =', "from_junction'] =", 'from_junction"] =',
        "junction'] =", 'junction"] =',
    )) or ("element'] =" in raw_terms and "et='ju'" in raw_terms):
        forced_headings.extend([
            ("02_invalid_values.md", "Cause 5: Non-existent junction index"),
            ("03_topology.md", "Cause 4: Pipe connected to a non-existent junction"),
        ])
    valve_markers = ("opened'] =", 'opened"] =', "closed valve", "valve_opening", "valve_configuration")
    if any(marker in raw_terms for marker in valve_markers):
        forced_headings.append(("03_topology.md", "Cause 2: Closed valve isolating a section"))
    if "pressure_ratio'] =" in raw_terms or 'pressure_ratio"] =' in raw_terms or "compressor_pressure_ratio:" in raw_terms:
        forced_headings.append(("06_components_and_other.md", "2. Pump / compressor issues"))
    if "u_w_per_m2k'] =" in raw_terms or 'u_w_per_m2k"] =' in raw_terms or "heat_transfer_coefficient:" in raw_terms:
        forced_headings.append(("06_components_and_other.md", "4. Pressure / flow control infeasibility"))
    if any(term in raw_terms for term in (
        "mdot_kg_per_s'] =", 'mdot_kg_per_s"] =', "sink_source_scaling:",
        "kg/h", "correct_kg_per_s", "extremely large for kg/s",
        "is negative; use the opposite element type", "is nan",
    )):
        forced_headings.extend([
            ("05_sink_and_source.md", "Cause 5: Wrong magnitude / units"),
            ("05_sink_and_source.md", "Cause 6: Unrealistic sink demand relative to the network"),
        ])

    sections = []
    for filename, wanted_heading in forced_headings:
        content = _cache.get(filename, "")
        for heading, section_text in _iter_sections(content):
            if heading == wanted_heading:
                sections.append((filename, heading, section_text))
                break
    return sections


def _required_files(raw_terms: str) -> set[str]:
    """Force domain packs whose sections are important for known diagnostic terms."""
    required = {"01_convergence.md"}
    if any(term in raw_terms for term in (
        "mdot", "sink", "source", "mass flow", "kg/h", "kg_per_s",
        "sink_source_scaling",
    )):
        required.add("05_sink_and_source.md")
    if any(term in raw_terms for term in (
        "diameter", "length_km", "height_m", "elevation", "junction",
        "reference", "index", "matrixrankwarning", "singular", "invalid_values",
    )):
        required.add("02_invalid_values.md")
    if any(term in raw_terms for term in (
        "unsupplied", "topology", "valve", "ext_grid", "connected",
        "missing_branch_junctions", "missing_node_junctions",
    )):
        required.add("03_topology.md")
    if any(term in raw_terms for term in ("fluid", "water", "gas", "lgas", "hgas")):
        required.add("04_fluid.md")
    if any(term in raw_terms for term in (
        "pump", "compressor", "pressure_ratio", "heat_transfer",
        "u_w_per_m2k", "circ_pump", "friction_model",
    )):
        required.add("06_components_and_other.md")
    return required
