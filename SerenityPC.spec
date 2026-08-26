# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

# Define the path to the Users directory relative to the spec file location
Users = os.path.join(os.path.dirname(__name__), 'Users')

# 1. Data Files: Collect app asset folders and library data
datas = [
    ('System', 'System'),
    (Users, 'Users'), # Bundled Users directory
]
datas += collect_data_files('llama_cpp')

# 2. Binaries: Dynamic C/C++ runtime DLLs
binaries = collect_dynamic_libs('llama_cpp')

# 3. Hidden Imports: Capture deferred & dynamic modules from load_heavy_libraries()
hiddenimports = [
    # Core C/C++ backend
    'llama_cpp',
    'llama_cpp.llama',
    'llama_cpp._internals',
    'llama_cpp.llama_chat_format',
    'llama_cpp.llama_speculative',
    # Monitoring & Hardware
    'pynvml',
    'psutil',
    # Imaging, Video & GUI
    'PIL',
    'PIL.Image',
    'PIL.ImageTk',
    'PIL._tkinter_finder',
    'cv2',
    'windnd',
    # Root helper modules
    'serenity_resources',
]

# Recursively capture all modules within the System package
hiddenimports += collect_submodules('System')

# Optional modules: only pull essential submodules
for optional_mod in ['numpy', 'requests']:
    try:
        __import__(optional_mod)
        hiddenimports += collect_submodules(optional_mod)
    except ImportError:
        pass

# Heavy packages to exclude from build bundle
excludes = [
    'torch',
    'torchvision',
    'torchaudio',
    'bitsandbytes',
    'triton',
    'benchmarks',
    'pytest',
    'unittest',
    'test',
    'IPython',
    'matplotlib',
    'scipy',
    'pandas',
]

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='SerenityPC',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # Disabled to prevent DLL corruption and crash on load
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Set to True for stdout/stderr log capture
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)