@echo off
cd /d "%~dp0"

if not exist "tmp" mkdir "tmp"
if not exist ".cuda_cache" mkdir ".cuda_cache"

:: Localize temp and cache variables to resolve Smart App Control (SAC) blocks
set TEMP=%~dp0tmp
set TMP=%~dp0tmp
set CUDA_CACHE_PATH=%~dp0.cuda_cache
set TORCH_EXTENSIONS_DIR=%~dp0tmp\torch_extensions
set TRITON_CACHE_DIR=%~dp0tmp\triton_cache

set PYTHONPATH=.venv\Lib\site-packages
pythonw.exe main.py
