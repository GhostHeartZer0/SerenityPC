@echo off
setlocal
cd /d "%~dp0.."

echo --- SerenityPC: IDE Run Fix ---
echo [!] Detected Windows Store Python stub at the top of your PATH.
echo [!] This often causes "can't run" issues in IDEs.
echo.
echo Running your script using the project's virtual environment...
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found. Please run setup_venv.bat first.
    pause
    exit /b 1
)

:: Run the script passed as an argument, or default to main.py
if "%~1"=="" (
    set TARGET=main.py
) else (
    set TARGET=%~1
)

echo [V] EXECUTING: %TARGET%
".venv\Scripts\python.exe" "%TARGET%"

if %ERRORLEVEL% neq 0 (
    echo.
    echo [!] Script exited with error code %ERRORLEVEL%
    pause
)
