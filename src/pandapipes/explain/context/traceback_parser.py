"""
Traceback parser — extracts user code frames and target-library frames.

Key insight: the Python traceback already contains the exact file path
and line number of every frame in the call stack. We read those files
directly — no indexing, no embeddings, no AST required.

Editable installs: with `pip install -e .`, pandapipes files live at the
repo root (not in site-packages). We detect them via package-name substring
matching in the path rather than relying solely on site-packages prefix.
"""

from __future__ import annotations

import site
import traceback as _tb
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType


@dataclass
class SourceContext:
    """
    One frame from the traceback with its surrounding source lines.

    Attributes
    ----------
    file_path : str
        Absolute path to the source file.
    lineno : int
        Line number where the frame is (1-indexed).
    function_name : str
        Name of the function at this frame.
    source_snippet : str
        Lines of source code around `lineno`, with line numbers prefixed.
    is_user_code : bool
        True if this frame is from the user's own code (not a library).
    """

    file_path: str
    lineno: int
    function_name: str
    source_snippet: str
    is_user_code: bool


def extract_context(
    exc_tb: TracebackType | None,
    context_lines: int = 30,
    target_package: str = "pandapipes",
) -> list[SourceContext]:
    """
    Walk the traceback and return SourceContext for every relevant frame.

    Relevant frames are:
    - User code (files NOT classified as library code)
    - target_package code (pandapipes or pandapower frames)

    All other library frames (numpy, scipy, etc.) are skipped.

    Parameters
    ----------
    exc_tb : TracebackType | None
        The traceback object from the exception.
    context_lines : int
        Number of lines above and below the error line to read.
    target_package : str
        Library name whose frames should be captured (e.g. "pandapipes").

    Returns
    -------
    list[SourceContext]
        Ordered from outermost (user code entry) to innermost (error origin).
    """
    if exc_tb is None:
        return []

    site_paths = _get_site_paths()
    frames = _tb.extract_tb(exc_tb)
    results: list[SourceContext] = []

    for frame in frames:
        path = frame.filename or ""

        # Synthetic / compiled frames have no readable source.
        if path in ("<string>", "<stdin>", "") or path.startswith("<"):
            continue

        is_target = _is_target_package_path(target_package, path)
        is_library = _is_library_path(path, site_paths) or (
            # Editable install: file lives at a repo root, not in site-packages.
            # Classify as library if the path contains a known third-party marker
            # but NOT the target package name — avoids false positives for repos
            # named "pandapipes-tutorials" etc. that the user controls.
            _is_editable_library(path, target_package)
        )

        # Skip frames that are library code AND not our target package.
        if is_library and not is_target:
            continue

        snippet = _read_context(path, frame.lineno, context_lines)
        results.append(
            SourceContext(
                file_path=path,
                lineno=frame.lineno,
                function_name=frame.name,
                source_snippet=snippet,
                is_user_code=not (is_library or is_target),
            )
        )

    return results


def read_full_source(path: str, error_line: int, max_chars: int = 8000) -> str:
    """
    Return the entire file (numbered, with a ``>>`` marker on the error line)
    if it fits under ``max_chars``. Otherwise return an empty string so the
    caller can fall back to a window.

    Returns a "[source not available]" placeholder for compiled/synthetic paths.
    """
    if not path or path.startswith("<"):
        return "[source not available — compiled or dynamic code]"
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
        if len(raw) > max_chars:
            return ""
        lines = raw.splitlines()
        numbered = [
            f"{'>>' if (i + 1) == error_line else '  '}{i + 1:4d}│ {line}"
            for i, line in enumerate(lines)
        ]
        return "\n".join(numbered)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_site_paths() -> tuple[str, ...]:
    """Return all site-packages paths as a tuple for fast prefix matching."""
    paths = list(site.getsitepackages())
    try:
        paths.append(site.getusersitepackages())
    except AttributeError:
        pass
    return tuple(str(p) for p in paths if p)


def _is_library_path(path: str, site_paths: tuple[str, ...]) -> bool:
    """Return True if the file lives inside any site-packages directory."""
    return any(path.startswith(sp) for sp in site_paths)


def _sep_in_path(package: str, path: str) -> bool:
    """
    Check if `package` appears as a path component (not just a substring).

    "/home/user/pandapipes/pandapipes/pipeflow.py" → True  (the package dir)
    "/home/user/pandapipes-tutorials/script.py" → False  (user repo)
    """
    parts = path.replace("\\", "/").split("/")
    return package in parts


def _is_target_package_path(package: str, path: str) -> bool:
    """
    Return True only for actual installed/editable package files.

    A checkout directory may itself be named "pandapipes"; that alone must not
    make every script inside the repo look like pandapipes library code.
    """
    parts = path.replace("\\", "/").split("/")
    for i, part in enumerate(parts):
        if part != package:
            continue
        previous = parts[i - 1] if i else ""
        if previous in {"src", "site-packages", "dist-packages"}:
            return True
        if previous == package and i + 1 < len(parts):
            return True
    return False


def _is_editable_library(path: str, target_package: str) -> bool:
    """
    Detect editable-install library frames: files that live outside
    site-packages but belong to a library (not the target package and
    not the user's code).

    Heuristic: if the path contains a known stdlib/third-party directory
    name component AND does not contain target_package as a path component,
    classify it as a library frame. pandapower is listed because pandapipes
    depends on it — its frames are library reference, not the user's bug.
    """
    _LIBRARY_MARKERS = {
        "numpy", "scipy", "pandas", "matplotlib", "networkx",
        "numba", "pandapower",
    }
    parts = set(path.replace("\\", "/").split("/"))
    return bool(parts & _LIBRARY_MARKERS) and not _is_target_package_path(target_package, path)


def _read_context(path: str, lineno: int, n: int) -> str:
    """
    Read ±n lines around `lineno` from `path`.

    Lines are prefixed with their line number for easy LLM reference:
        ``  42│ code here``

    Returns an empty string if the file cannot be read.
    """
    try:
        all_lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        lo = max(0, lineno - n - 1)
        hi = min(len(all_lines), lineno + n)
        numbered = [
            f"{i + lo + 1:4d}│ {line}"
            for i, line in enumerate(all_lines[lo:hi])
        ]
        return "\n".join(numbered)
    except Exception:
        return ""
