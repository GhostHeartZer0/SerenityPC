@echo off
setlocal
cd /d "%~dp0"

:: Fast-path: Launch existing .venv directly without needing system PATH
if exist "%~dp0.venv\Scripts\python.exe" (
    if exist "%~dp0.venv\Scripts\activate.bat" (
        call "%~dp0.venv\Scripts\activate.bat"
    )
    "%~dp0.venv\Scripts\python.exe" "%~dp0main.py" %*
    if errorlevel 1 pause
    exit /b %errorlevel%
)

:: First-time setup fallback: Locate system python to build .venv
where python >nul 2>&1
if %errorlevel% equ 0 (
    set "PY_CMD=python"
) else (
    where py >nul 2>&1
    if %errorlevel% equ 0 (
        set "PY_CMD=py -3"
    ) else (
        echo ============================================================
        echo [!] ERROR: Python is not installed or not added to PATH!
        echo [!] Please download and install Python (64-bit) from:
        echo     https://www.python.org/downloads/
        echo [!] Make sure to check "Add Python to PATH" during installation.
        echo ============================================================
        pause
        exit /b 1
    )
)

echo [*] First run detected. Initializing setup...
%PY_CMD% "%~dp0setup.py"

if exist "%~dp0.venv\Scripts\python.exe" (
    if exist "%~dp0.venv\Scripts\activate.bat" (
        call "%~dp0.venv\Scripts\activate.bat"
    )
    "%~dp0.venv\Scripts\python.exe" "%~dp0main.py" %*
) else (
    %PY_CMD% "%~dp0main.py" %*
)

if errorlevel 1 pause


