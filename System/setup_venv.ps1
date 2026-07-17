# SerenityPC Virtual Environment Setup Script
# This script automates the creation of a venv and installation of dependencies.

$ErrorActionPreference = "Stop"

Write-Host "--- SerenityPC: Environment Setup ---" -ForegroundColor Cyan

# 1. Check for Python
try {
    $pythonVersion = & python --version 2>&1
    Write-Host "Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Error "Python not found! Please install Python 3.11+ and add it to your PATH."
    exit 1
}

# Resolve the project root folder (parent of System/ directory where this script is located)
$rootDir = Split-Path $PSScriptRoot -Parent

# 2. Create Virtual Environment in the root folder
$venvPath = Join-Path $rootDir ".venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment in $venvPath..." -ForegroundColor Yellow
    # Create the virtual environment directly at the resolved root path
    & python -m venv $venvPath
} else {
    Write-Host "Virtual environment already exists in $venvPath." -ForegroundColor Green
}

$pythonExe = Join-Path $venvPath "Scripts\python.exe"

# 3. Upgrade pip and install wheel
Write-Host "Upgrading pip and installing wheel..." -ForegroundColor Yellow
& $pythonExe -m pip install --upgrade pip wheel --quiet

# 4. Run existing setup.py from the root folder
$setupScript = Join-Path $rootDir "setup.py"
Write-Host "Starting SerenityPC setup logic (compiling engine and installing requirements)..." -ForegroundColor Cyan
& $pythonExe $setupScript

Write-Host "`n--- Setup Complete! ---" -ForegroundColor Green
Write-Host "To start SerenityPC, you can now use run.bat or run:"
Write-Host "  .venv\Scripts\python.exe main.py" -ForegroundColor Gray
