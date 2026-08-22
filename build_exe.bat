@echo off
setlocal
cd /d "%~dp0"

echo [1/2] Checking dependencies...
call .venv\Scripts\python.exe -m pip install pyinstaller

echo [2/2] Building SerenityPC standalone application...
call .venv\Scripts\pyinstaller.exe SerenityPC.spec --clean -y

echo Build finished. Output directory: dist\SerenityPC
pause
