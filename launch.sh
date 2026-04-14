#!/usr/bin/env bash
# launch.sh - GASPmini launcher for Ubuntu
# ─────────────────────────────────────────────────────────────────────────────
# EDIT THIS LINE to point at your Python 3.12 executable (or leave as-is to
# auto-detect python3.12 on your PATH):
PYTHON="python3.12"
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
VENV_PY="$VENV_DIR/bin/python"

# Verify the chosen Python exists
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: '$PYTHON' not found on PATH." >&2
    echo "Install it with:  sudo apt install python3.12 python3.12-venv" >&2
    echo "Or edit the PYTHON variable at the top of launch.sh." >&2
    exit 1
fi

PYTHON_VERSION="$("$PYTHON" --version 2>&1)"
echo "Using $PYTHON_VERSION from $(command -v "$PYTHON")"

# ── Rebuild .venv if it is missing or was created by a different Python ───────
REBUILD_VENV=0

if [[ ! -x "$VENV_PY" ]]; then
    echo ".venv not found — creating it..."
    REBUILD_VENV=1
else
    VENV_BASE_REAL="$($VENV_PY  -c 'import os, sys; print(os.path.realpath(getattr(sys, "_base_executable", sys.executable)))')"
    REQUESTED_REAL="$($PYTHON -c 'import os, sys; print(os.path.realpath(sys.executable))')"
    if [[ "$VENV_BASE_REAL" != "$REQUESTED_REAL" ]]; then
        echo ".venv base python ($VENV_BASE_REAL) differs from requested ($REQUESTED_REAL) — rebuilding..."
        REBUILD_VENV=1
    fi
fi

if [[ $REBUILD_VENV -eq 1 ]]; then
    if [[ -d "$VENV_DIR" ]]; then
        echo "Removing old .venv..."
        rm -rf "$VENV_DIR"
    fi
    "$PYTHON" -m venv "$VENV_DIR"
    echo ".venv created."
fi

# ── Upgrade pip and install / refresh dependencies only when needed ──────────
REFRESH_DEPS=$REBUILD_VENV

if [[ $REFRESH_DEPS -eq 0 ]]; then
    if ! "$VENV_PY" -c 'import importlib.util, sys; sys.exit(0 if importlib.util.find_spec("PySide6") and importlib.util.find_spec("pytest") else 1)' >/dev/null 2>&1; then
        echo "Dependencies missing from .venv — installing..."
        REFRESH_DEPS=1
    fi
fi

if [[ $REFRESH_DEPS -eq 1 ]]; then
    echo "Upgrading pip..."
    "$VENV_PY" -m pip install --upgrade pip --quiet

    echo "Installing dependencies..."
    "$VENV_PY" -m pip install PySide6 pytest --quiet
fi

echo "Dependencies OK."

# ── Launch the app ────────────────────────────────────────────────────────────
echo "Launching GASPmini..."
cd "$SCRIPT_DIR"
exec "$VENV_PY" main.py "$@"
