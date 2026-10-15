========================================================================
SERENITY PC LOCAL LEGACY - QUICK SETUP GUIDE (GTX 1050 Ti & Legacy GPUs)
========================================================================

SYSTEM REQUIREMENTS:
 - Windows 10 (64-bit)
 - NVIDIA GPU (GTX 1050 Ti / sm_61 / sm_50 or newer)
 - Python installed

------------------------------------------------------------------------
STEP 1: INSTALL CUDA (If not already installed)
------------------------------------------------------------------------
 Run the CUDA 12.6 installer included in this package:
   Misc\CUDA\cuda_12.6.0_560.76_windows.exe

 (If you already have CUDA 12.x installed, you can skip this step.)

------------------------------------------------------------------------
STEP 2: RUN SETUP & LAUNCH
------------------------------------------------------------------------
 Double-click:
   run.bat

 What happens automatically:
  1. Creates `.venv` environment.
  2. Installs pre-compiled wheels instantly from `wheels/` (No long compile times!).
  3. Creates desktop shortcut.
  4. Launches SerenityPC main application.

------------------------------------------------------------------------
TROUBLESHOOTING & MANUAL SETUP
------------------------------------------------------------------------
 If you ever need to re-run setup or rebuild:
   python setup.py

