# launch.ps1 - GASPmini launcher for Windows
# ─────────────────────────────────────────────────────────────────────────────
# EDIT THIS LINE to point at your Python 3.12 executable:
$PYTHON = "C:\Python312\python.exe"
# ─────────────────────────────────────────────────────────────────────────────

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir   = Join-Path $ScriptDir ".venv"
$VenvPy    = Join-Path $VenvDir "Scripts\python.exe"
$VenvPip   = Join-Path $VenvDir "Scripts\pip.exe"

# Verify the chosen Python exists
if (-not (Test-Path $PYTHON)) {
    Write-Error "Python not found at: $PYTHON"
    Write-Error "Edit the `$PYTHON variable at the top of launch.ps1."
    exit 1
}

$PythonVersion = & $PYTHON --version 2>&1
Write-Host "Using $PythonVersion from $PYTHON"

# ── Rebuild .venv if it is missing or was created by a different Python ───────
$RebuildVenv = $false

if (-not (Test-Path $VenvPy)) {
    Write-Host ".venv not found — creating it..."
    $RebuildVenv = $true
} else {
    # Check that the venv python matches the requested python
    $VenvPyReal    = (& $VenvPy -c "import sys; print(sys.executable)" 2>$null)
    $RequestedReal = (& $PYTHON -c "import sys; print(sys.executable)" 2>$null)
    if ($VenvPyReal -ne $RequestedReal) {
        Write-Host ".venv python ($VenvPyReal) differs from requested ($RequestedReal) — rebuilding..."
        $RebuildVenv = $true
    }
}

if ($RebuildVenv) {
    if (Test-Path $VenvDir) {
        Write-Host "Removing old .venv..."
        Remove-Item -Recurse -Force $VenvDir
    }
    & $PYTHON -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment."
        exit 1
    }
    Write-Host ".venv created."
}

# ── Upgrade pip and install / refresh dependencies ────────────────────────────
Write-Host "Upgrading pip..."
& $VenvPy -m pip install --upgrade pip --quiet

Write-Host "Installing dependencies..."
& $VenvPip install PySide6 pytest --quiet

if ($LASTEXITCODE -ne 0) {
    Write-Error "Dependency installation failed."
    exit 1
}

Write-Host "Dependencies OK."

# ── Launch the app ────────────────────────────────────────────────────────────
Write-Host "Launching GASPmini..."
Set-Location $ScriptDir
& $VenvPy main.py $args
