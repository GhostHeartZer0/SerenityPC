@echo off
cd /d "%~dp0"

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

if not exist "%~dp0.venv\Scripts\activate.bat" (
    echo [*] First run detected. Initializing setup...
    %PY_CMD% "%~dp0setup.py"
)

if exist "%~dp0.venv\Scripts\activate.bat" (
    call "%~dp0.venv\Scripts\activate.bat"
    python "%~dp0main.py" %*
) else (
    %PY_CMD% "%~dp0main.py" %*
)


