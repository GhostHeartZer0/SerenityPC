# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

datas = [
    ('System/Media', 'System/Media'),
    ('Media', 'Media'),
    ('Docs', 'Docs'),
    ('serenity_resources.py', '.'),
]
datas += collect_data_files('llama_cpp')
binaries = collect_dynamic_libs('llama_cpp')

# Add CUDA 12 toolkit DLL search paths if present
cuda_bin = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin'
if os.path.exists(cuda_bin):
    for dll in ['cudart64_12.dll', 'cublas64_12.dll', 'cublasLt64_12.dll']:
        dll_path = os.path.join(cuda_bin, dll)
        if os.path.exists(dll_path):
            binaries.append((dll_path, '.'))

a = Analysis(
    ['main.py'],
    pathex=['.', 'System'],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        'llama_cpp',
        'psutil',
        'pynvml',
        'PIL',
        'PIL.ImageTk',
        'numpy',
        'requests',
        'cryptography',
        'torch',
        'torch.autograd',
        'sounddevice',
        'pyttsx3',
        'pystray',
        'uvicorn',
        'fastapi',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
