# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files
import os

datas = [
    ('Media', 'Media'),
    ('System/Media', 'System/Media'),
    ('Docs', 'Docs'),
    ('serenity_resources.py', '.'),
    ('System', 'System'),
]

# Hidden imports for packages that rely on dynamic dispatch or background threads
hiddenimports = [
    'tkinter',
    'tkinter.filedialog',
    'tkinter.messagebox',
    'tkinter.scrolledtext',
    'tkinter.simpledialog',
    'tkinter.ttk',
    'tkinter.font',
    'uvicorn',
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
    'fastapi',
    'pystray',
    'comtypes',
    'comtypes.client',
    'pyttsx3',
    'pyttsx3.drivers',
    'pyttsx3.drivers.sapi5',
    'sounddevice',
    'speech_recognition',
    'chromadb',
    'engineio.async_drivers.asgi',
    'PIL',
    'PIL.Image',
    'PIL.ImageTk',
    'cv2',
    'psutil',
    'torch',
    'transformers',
    'accelerate',
    'requests',
    'bs4',
    'cryptography',
]

# Collect dynamic data / binaries for heavy packages
packages_to_collect = [
    'chromadb', 
    'onnxruntime', 
    'pystray', 
    'pyttsx3', 
    'sounddevice', 
    'speech_recognition', 
    'transformers', 
    'accelerate', 
    'torch', 
    'cv2', 
    'PIL', 
    'llama_cpp'
]

for pkg in packages_to_collect:
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
        datas += pkg_datas
        hiddenimports += pkg_hidden
    except Exception as e:
        print(f"PyInstaller collect_all warning for {pkg}: {e}")

a = Analysis(
    ['main.py'],
    pathex=['.', 'System'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['unittest', 'test'],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SerenityPC',
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
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='SerenityPC',
)
