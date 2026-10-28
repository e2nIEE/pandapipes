"""
Terminal stdout capture — rolling buffer of everything written to stdout
before an exception is raised. Passed to the LLM so it can see print()
output, progress messages, and partial results that preceded the error.

Only stdout is captured — NOT stderr. Excluding stderr prevents the Python
traceback (written to stderr by the default excepthook) from appearing a
second time in the LLM's "Terminal Output" section.

The tee wrapper forwards writes to the original stream unchanged, so the
user sees their output normally. In parallel, each completed line is appended
to a bounded deque — old lines are dropped so memory stays flat.
"""

from __future__ import annotations

import sys
import threading
import warnings
from collections import deque
from typing import Deque

_MAX_LINES = 200
_MAX_CHARS = 6000

_buffer: Deque[str] = deque(maxlen=_MAX_LINES)
_lock = threading.Lock()

_installed = False
_orig_stdout = None
_orig_showwarning = None
_tee = None


class _Tee:
    """Forwards writes to an underlying stream AND captures complete lines."""

    def __init__(self, underlying):
        self._underlying = underlying
        self._pending = ""
        self._tee_lock = threading.Lock()

    def write(self, s):
        try:
            n = self._underlying.write(s)
        except Exception:
            n = len(s) if isinstance(s, str) else 0
        try:
            with self._tee_lock:
                self._pending += s
                while "\n" in self._pending:
                    line, self._pending = self._pending.split("\n", 1)
                    with _lock:
                        _buffer.append(line)
        except Exception:  # nosec B110 - never log from inside the stdout tee: it would recurse
            pass
        return n

    def flush(self):
        try:
            self._underlying.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._underlying, name)


def install() -> None:
    """Replace sys.stdout with a capturing tee and record Python warnings."""
    global _installed, _orig_stdout, _orig_showwarning, _tee
    if _installed:
        return
    _orig_stdout = sys.stdout
    _orig_showwarning = warnings.showwarning
    _tee = _Tee(_orig_stdout)
    sys.stdout = _tee
    warnings.showwarning = _capture_warning
    _installed = True


def uninstall() -> None:
    """Restore sys.stdout / warnings.showwarning, unless someone else replaced them since."""
    global _installed, _orig_stdout, _orig_showwarning, _tee
    if not _installed:
        return
    if sys.stdout is _tee:
        sys.stdout = _orig_stdout
    if _orig_showwarning is not None and warnings.showwarning is _capture_warning:
        warnings.showwarning = _orig_showwarning
    _orig_stdout = None
    _orig_showwarning = None
    _tee = None
    _installed = False


def get_recent_output(max_chars: int = _MAX_CHARS) -> str:
    """
    Return the captured stdout as a single string, trimmed from the front
    if over max_chars so the tail (closest to the error) is preserved.
    """
    with _lock:
        if not _buffer:
            return ""
        text = "\n".join(_buffer)
    if len(text) > max_chars:
        text = "... [older output truncated] ...\n" + text[-max_chars:]
    return text


def clear() -> None:
    """Empty the buffer — used between test runs to prevent cross-contamination."""
    with _lock:
        _buffer.clear()


def _capture_warning(message, category, filename, lineno, file=None, line=None):
    """Record warning text while preserving the user's normal warning display."""
    try:
        rendered = warnings.formatwarning(message, category, filename, lineno, line).strip()
        with _lock:
            _buffer.append(rendered)
    except Exception:
        pass
    try:
        if _orig_showwarning is not None:
            return _orig_showwarning(message, category, filename, lineno, file=file, line=line)
    except Exception:
        pass
    return None
