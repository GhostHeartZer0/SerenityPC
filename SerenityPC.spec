# -*- mode: python ; coding: utf-8 -*-
import os
import re
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

app_version = "1.7.0"
changelog_path = os.path.join(os.path.dirname(os.path.abspath(SPEC)), "CHANGELOG.md") if 'SPEC' in globals() and SPEC else "CHANGELOG.md"
if os.path.exists(changelog_path):
    try:
        with open(changelog_path, "r", encoding="utf-8") as f:
            match = re.search(r"## Current Version:\s*\n+### Version\s+([^\s\n]+)", f.read())
            if match:
                app_version = match.group(1).strip()
    except Exception:
        pass

exe_name = f"SerenityPC Legacy v{app_version}"

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
    excludes=[
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
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

icon_path = 'System/serenity.ico' if os.path.exists('System/serenity.ico') else None

build_mode = os.environ.get("BUILD_MODE", "onedir").strip().lower()

if build_mode == "onefile":
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name=exe_name,
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
        name=exe_name,
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
        name=exe_name,
    )
