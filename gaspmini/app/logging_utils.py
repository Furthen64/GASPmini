# app/logging_utils.py
# Simple logging helpers for GASPmini v1.
# All output goes to stdout for easy debugging.

from __future__ import annotations

_log_callback = None  # Optional[Callable[[str], None]]


def set_log_callback(callback) -> None:
    """Redirect log output to a Qt widget or other sink."""
    global _log_callback
    _log_callback = callback


def log(message: str) -> None:
    """Print a general log message."""
    print(message)
    if _log_callback is not None:
        _log_callback(message)


def debug_log(message: str) -> None:
    """Print a debug message (always, caller decides when to call)."""
    print(f"[DEBUG] {message}")
    if _log_callback is not None:
        _log_callback(f"[DEBUG] {message}")
