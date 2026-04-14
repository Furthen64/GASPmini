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
    VENV_PY_REAL="$("$VENV_PY"  -c 'import sys; print(sys.executable)')"
    REQUESTED_REAL="$("$PYTHON" -c 'import sys; print(sys.executable)')"
    if [[ "$VENV_PY_REAL" != "$REQUESTED_REAL" ]]; then
        echo ".venv python ($VENV_PY_REAL) differs from requested ($REQUESTED_REAL) — rebuilding..."
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

# ── Upgrade pip and install / refresh dependencies ────────────────────────────
echo "Upgrading pip..."
"$VENV_PY" -m pip install --upgrade pip --quiet

echo "Installing dependencies..."
"$VENV_PY" -m pip install PySide6 pytest --quiet

echo "Dependencies OK."

# ── Launch the app ────────────────────────────────────────────────────────────
echo "Launching GASPmini..."
cd "$SCRIPT_DIR"
exec "$VENV_PY" main.py "$@"
