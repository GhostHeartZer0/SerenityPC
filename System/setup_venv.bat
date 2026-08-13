1@echo off
TITLE SerenityPC Setup Wrapper
echo Launching SerenityPC Environment Setup...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_venv.ps1"
pause
