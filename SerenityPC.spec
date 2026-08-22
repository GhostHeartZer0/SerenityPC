# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

datas = [
    ('System', 'System'),
    ('Users', 'Users'),
    ('Media', 'Media'),
    ('Docs', 'Docs'),
    ('Tools', 'Tools'),
    ('serenity_resources.py', '.'),
]

if os.path.exists('Models/For More Models....txt'):
    datas.append(('Models/For More Models....txt', 'Models'))

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
    pathex=['.', 'System', 'Tools'],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        'llama_cpp',
        'psutil',
        'pynvml',
        'PIL',
        'PIL.Image',
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
        'windnd',
        'cv2',
        'speech_recognition',
        'standard_aifc',
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'tkinter.scrolledtext',
        'tkinter.font',
        'tkinter.simpledialog',
        'System',
        'System.serenity_utils',
        'System.kv_manager',
        'System.tool_registry',
        'System.modular_registry',
        'System.markdown_engine',
        'System.settings_ui',
        'System.vault_manager',
        'System.network_guard',
        'System.stt_manager',
        'System.vision_handler',
        'System.synthesis_handler',
        'System.settings_manager',
        'System.diffusion_wrapper',
        'System.gguf_draft_model',
        'System.tri_attention_core',
        'serenity_resources',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

icon_path = 'System/serenity.ico' if os.path.exists('System/serenity.ico') else None

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
    icon=icon_path,
)
