@echo off
echo --- SerenityPC: Deep Clean Initiated ---
pip uninstall llama-cpp-python -y
del /s /q *.pyc
del /s /q *.log
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
echo [OK] Cache, Logs, and broken Backend purged.
pause