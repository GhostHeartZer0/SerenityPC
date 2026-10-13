# -*- mode: python ; coding: utf-8 -*-

import os
import glob
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

# 1. Data Files: Collect app asset folders and library data
system_datas = []
for root, dirs, files in os.walk('System'):
    # Exclude duplicate Media assets, tests, and bytecode caches from embedded System bundle
    dirs[:] = [d for d in dirs if d not in ('Media', 'tests', '__pycache__')]
    for f in files:
        full_path = os.path.join(root, f)
        rel_dir = os.path.relpath(root, '.')
        system_datas.append((full_path, rel_dir))

datas = [
    ('Users', 'Users'),
    ('Media', 'Media'),
    ('Docs', 'Docs'),
    ('Tools', 'Tools'),
    ('serenity_resources.py', '.'),
] + system_datas

if os.path.exists('Models/For More Models....txt'):
    datas.append(('Models/For More Models....txt', 'Models'))

# Collect llama_cpp data files excluding development-only static libraries (*.lib)
datas += [d for d in collect_data_files('llama_cpp') if not d[0].endswith('.lib')]

# 2. Binaries: Dynamic C/C++ runtime DLLs
binaries = collect_dynamic_libs('llama_cpp')

# Add CUDA toolkit DLL search paths if present
cuda_base = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
if os.path.exists(cuda_base):
    cuda_dirs = sorted(glob.glob(os.path.join(cuda_base, "v*")), reverse=True)
    for c_dir in cuda_dirs:
        c_bin = os.path.join(c_dir, "bin")
        if os.path.exists(c_bin):
            for f in os.listdir(c_bin):
                if f.lower().startswith(('cublas', 'cudart')) and f.lower().endswith('.dll'):
                    binaries.append((os.path.join(c_bin, f), '.'))
            break

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

icon_path = 'System/serenity.ico' if os.path.exists('System/serenity.ico') else None
app_name = os.environ.get("APP_NAME", "SerenityPC Legacy v1.7.3").strip()
build_mode = os.environ.get("BUILD_MODE", "onedir").strip().lower()

if build_mode == "onefile":
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name=app_name,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=True,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=icon_path,
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=app_name,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=True,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=icon_path,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        upx_exclude=[],
        name=app_name,
    )