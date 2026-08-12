import sys
import subprocess
import os
import time
import traceback
import ctypes
import glob
import shutil

# --- Configuration ---
REQUIREMENTS_FILE = "requirements.txt"
# For 12700KF: 8 P-Cores / 16 threads
# For RTX 3050: Compute Architecture 8.6 (sm_86)
CUDA_ARCH_TARGET = "86"
# --- End Configuration ---

def get_cuda_path():
    base_install = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if os.path.exists(base_install):
        versions = glob.glob(os.path.join(base_install, "v*")) 
        if versions:
            return os.path.join(sorted(versions)[-1], "bin")
    return os.environ.get('CUDA_PATH') or os.environ.get('CUDA_HOME')

def inject_cuda_path(cuda_bin_path):
    if cuda_bin_path and os.path.isdir(cuda_bin_path) and hasattr(os, 'add_dll_directory'):
        try:
            os.add_dll_directory(cuda_bin_path)
            # Add the lib/x64 path where cublas lives
            lib_path = os.path.join(os.path.dirname(cuda_bin_path), "lib", "x64")
            if os.path.exists(lib_path) and os.path.isdir(lib_path):
                os.add_dll_directory(lib_path)
            print(f"  > [V] Apex Link Established: {cuda_bin_path}")
        except Exception as e:
            print(f"  > [!] DLL Error: {e}")

def find_vcvars():
    # 1. Try using vswhere.exe
    vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if os.path.exists(vswhere):
        try:
            out = subprocess.check_output(
                [vswhere, "-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
                text=True
            ).strip()
            if out:
                bat = os.path.join(out, "VC", "Auxiliary", "Build", "vcvars64.bat")
                if os.path.exists(bat):
                    return bat
        except Exception:
            pass

    # 2. Fallback to standard installation paths
    paths = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    ]
    for p in paths:
        if os.path.exists(p): return p
    return None

def check_python_environment():
    """Verify Python bitness, version, and system memory."""
    print("\n--- [ Pre-flight Python & System Verification ] ---")
    is_64bit = sys.maxsize > 2**32
    py_version = sys.version_info
    py_str = f"{py_version.major}.{py_version.minor}.{py_version.micro}"
    
    print(f"  > Python Version: {py_str} ({'64-bit' if is_64bit else '32-bit'})")
    
    status = True
    if not is_64bit:
        print("  ❌ CRITICAL: 32-bit Python detected! 64-bit Python is required for CUDA & native C++ extensions.")
        status = False
    
    if py_version < (3, 8):
        print("  ❌ WARNING: Python 3.8+ required. Installed version is outdated.")
        status = False
    elif py_version >= (3, 13):
        print("  ⚠️  WARNING: Python 3.13+ detected. Prebuilt wheels or C++ bindings may have compatibility issues.")
    else:
        print("  ✅ Python version & architecture compatible.")
        
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        total_ram_gb = stat.ullTotalPhys / (1024 ** 3)
        avail_ram_gb = stat.ullAvailPhys / (1024 ** 3)
        print(f"  > System RAM: {total_ram_gb:.1f} GB Total ({avail_ram_gb:.1f} GB Available)")
    except Exception as e:
        print(f"  > System RAM check skipped: {e}")
        
    return status

def check_cpu_vector_support():
    """Checks CPU AVX / AVX2 support to set optimal CMake flags."""
    avx_supported = True
    avx2_supported = True
    try:
        avx_supported = bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(17))
        avx2_supported = bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(40))
    except Exception:
        pass

    print(f"  > CPU Instruction Support: AVX={'Yes' if avx_supported else 'No'}, AVX2={'Yes' if avx2_supported else 'No'}")
    return {"avx": avx_supported, "avx2": avx2_supported}

def check_cuda_hardware_and_toolkit():
    """Verify CUDA Toolkit, GPU driver, and compute capability."""
    print("\n--- [ Pre-flight CUDA Hardware & Toolkit Check ] ---")
    cuda_path = get_cuda_path()
    
    nvcc_path = shutil.which("nvcc")
    if not nvcc_path and cuda_path:
        possible_nvcc = os.path.join(cuda_path, "nvcc.exe")
        if os.path.exists(possible_nvcc):
            nvcc_path = possible_nvcc

    if nvcc_path:
        try:
            out = subprocess.check_output([nvcc_path, "--version"], text=True, stderr=subprocess.STDOUT)
            for line in out.splitlines():
                if "release" in line.lower():
                    print(f"  > NVCC Output: {line.strip()}")
        except Exception as e:
            print(f"  ⚠️ Could not parse NVCC version: {e}")
    else:
        print("  ⚠️ CUDA Toolkit Compiler (nvcc) NOT found in PATH or standard installation directories.")

    gpu_info = {"detected": False, "name": None, "driver": None, "compute_cap": None}
    smi_path = shutil.which("nvidia-smi")
    if not smi_path:
        smi_matches = glob.glob(r"C:\Windows\System32\DriverStore\FileRepository\nv_dispi*\nvidia-smi.exe")
        if smi_matches:
            smi_path = smi_matches[0]

    if smi_path and os.path.exists(smi_path):
        try:
            out = subprocess.check_output(
                [smi_path, "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader"],
                text=True, stderr=subprocess.STDOUT
            )
            lines = [l.strip() for l in out.strip().splitlines() if l.strip()]
            for l in lines:
                parts = [p.strip() for p in l.split(",")]
                if len(parts) >= 3:
                    gpu_info["detected"] = True
                    gpu_info["name"], gpu_info["driver"], gpu_info["compute_cap"] = parts[0], parts[1], parts[2]
                    print(f"  > GPU Detected: {gpu_info['name']}")
                    print(f"  > Driver Version: {gpu_info['driver']}")
                    print(f"  > Compute Capability: {gpu_info['compute_cap']}")
        except Exception as e:
            print(f"  ⚠️ nvidia-smi execution failed: {e}")

    return gpu_info

def activate_local_venv():
    """Detects workspace .venv directory and auto-injects its Scripts directory, or targets global python if --global is passed."""
    if "--global" in sys.argv:
        print("  > [!] Target: Global System Python (via --global flag).")
        return sys.executable

    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(workspace_dir, ".venv")
    venv_scripts = os.path.join(venv_dir, "Scripts")
    venv_python = os.path.join(venv_scripts, "python.exe")

    if os.path.exists(venv_python):
        if os.path.normcase(os.path.abspath(sys.executable)) != os.path.normcase(os.path.abspath(venv_python)):
            print(f"  > [!] Automatically re-launching setup.py inside .venv (pass --global for system python)...")
            sys.exit(subprocess.call([venv_python] + sys.argv))
            
        os.environ["VIRTUAL_ENV"] = venv_dir
        if venv_scripts not in os.environ.get("PATH", ""):
            os.environ["PATH"] = venv_scripts + os.pathsep + os.environ.get("PATH", "")
        print(f"  ✅ Local Virtual Environment (.venv) active: {venv_dir}")
        return venv_python
    return sys.executable

def gather_missing_tools():
    """Auto-gathers missing build tools (CMake, Ninja, Git, CUDA)."""
    print("\n--- [ Auto-Gathering Missing Tools & Environment Resolution ] ---")
    active_py = activate_local_venv()

    if not shutil.which("cmake"):
        print("  [!] CMake not found in PATH. Attempting auto-installation via pip...")
        try:
            subprocess.call([sys.executable, "-m", "pip", "install", "cmake", "--quiet"])
            cmake_pkg_bin = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "cmake", "data", "bin")
            if os.path.exists(cmake_pkg_bin):
                os.environ["PATH"] = cmake_pkg_bin + os.pathsep + os.environ.get("PATH", "")
        except Exception as e:
            print(f"  ⚠️ Failed to auto-gather CMake: {e}")

    if not shutil.which("ninja"):
        print("  [!] Ninja not found in PATH. Attempting auto-installation via pip...")
        try:
            subprocess.call([sys.executable, "-m", "pip", "install", "ninja", "--quiet"])
            ninja_pkg_bin = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "ninja", "data", "bin")
            if os.path.exists(ninja_pkg_bin):
                os.environ["PATH"] = ninja_pkg_bin + os.pathsep + os.environ.get("PATH", "")
        except Exception as e:
            print(f"  ⚠️ Failed to auto-gather Ninja: {e}")

    if not shutil.which("git"):
        git_candidates = [
            r"C:\Program Files\Git\cmd\git.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Git\cmd\git.exe"),
        ]
        for candidate in git_candidates:
            if os.path.exists(candidate):
                os.environ["PATH"] = os.path.dirname(candidate) + os.pathsep + os.environ.get("PATH", "")
                break

    if not shutil.which("nvcc"):
        cuda_path = get_cuda_path()
        if cuda_path and os.path.exists(os.path.join(cuda_path, "nvcc.exe")):
            os.environ["PATH"] = cuda_path + os.pathsep + os.environ.get("PATH", "")

def check_build_environment():
    """Pre-flight check to verify required setup dependencies."""
    py_ok = check_python_environment()
    check_cpu_vector_support()
    check_cuda_hardware_and_toolkit()
    gather_missing_tools()
    
    vcvars_path = find_vcvars()
    cl_path = shutil.which("cl") or (vcvars_path if vcvars_path else None)

    print("\n--- [ Pre-flight Build Tools Check ] ---")
    checks = {
        "CMake": shutil.which("cmake"),
        "MSVC (cl.exe)": cl_path,
        "CUDA Compiler (nvcc)": shutil.which("nvcc"),
        "Git": shutil.which("git"),
        "Ninja": shutil.which("ninja")
    }
    
    all_passed = True
    for tool, path in checks.items():
        if path:
            print(f"✅ {tool}: Detected ({path})")
        else:
            print(f"❌ {tool}: NOT FOUND")
            if tool in ["CMake", "MSVC (cl.exe)", "CUDA Compiler (nvcc)"]:
                all_passed = False
    
    return py_ok and all_passed

def get_optimized_cmake_flags():
    """Construct target CMake flags for modern GGML, dynamic GPU architecture, and Gemma/Diffusion support."""
    cpu_caps = check_cpu_vector_support()
    gpu_info = check_cuda_hardware_and_toolkit()
    
    cuda_arch = CUDA_ARCH_TARGET
    ptx_range = "50-virtual;52-virtual;61-virtual;70-virtual;75-virtual;80-virtual;86-virtual;89-virtual;90-virtual"
    if gpu_info.get("detected") and gpu_info.get("compute_cap"):
        cap = gpu_info["compute_cap"].replace(".", "")
        if cap.isdigit():
            # Target local GPU (fast startup) + comprehensive PTX range (Universal JIT for sm_50 to latest)
            cuda_arch = f"{cap}-real;{ptx_range}"
            print(f"  > [AUTO-CONFIG] Target: sm_{cap} (Native SASS) + Universal PTX (sm_50 -> sm_90+ JIT for .whl portability)")
    else:
        # Fallback to universal PTX range if local GPU isn't detected
        cuda_arch = ptx_range
        print(f"  > [AUTO-CONFIG] GPU not cleanly detected. Using Universal PTX range (sm_50 -> sm_90+)")
        
    cmake_flags = [
        "-DGGML_CUDA=ON",
        "-DGGML_CUDA_FLASH_ATTENTION=ON",
        "-DGGML_CUDA_FA_ALL_QUANTS=ON",
        "-DLLAMA_MTP=ON",
        "-DLLAMA_DIFFUSION=ON",
        "-DLLAMA_TURBOQUANT=ON",
        "-DLLAMA_TRI_ATTENTION=ON",
        "-DLLAMA_TURBOVEC=ON",
        f'"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}"',
        "-DGGML_AVX512=OFF",
        "-DCMAKE_CXX_STANDARD=17",
        "-DCMAKE_CUDA_STANDARD=17",
        "-DCMAKE_CXX_FLAGS='/Zc:preprocessor'",
        "-DCMAKE_CUDA_FLAGS='-Xcompiler /Zc:preprocessor'"
    ]
    
    if not cpu_caps.get("avx2", True):
        cmake_flags.append("-DGGML_AVX2=OFF")
    if not cpu_caps.get("avx", True):
        cmake_flags.append("-DGGML_AVX=OFF")
        cmake_flags.append("-DGGML_FMA=OFF")
        
    return cmake_flags

def install_engine(cuda_path, use_source=False, force_pypi=False):
    """Unified installer optimized for local hardware, Gemma quants, and Muse/Diffusion binaries."""
    print(f"\n[ ! ] PHASE: Building Apex Engine for detected architecture...")
    
    cuda_root = os.path.dirname(cuda_path) if cuda_path else ""
    cmake_flags = get_optimized_cmake_flags()

    use_ninja = bool(shutil.which("ninja"))
    cmake_args = " ".join(cmake_flags)
    if cuda_root and not use_ninja:
        cmake_args += f' -T "cuda={cuda_root}"'

    env = os.environ.copy()
    env["CMAKE_ARGS"] = cmake_args
    env["SKBUILD_CMAKE_ARGS"] = cmake_args
    env["FORCE_CMAKE"] = "1"
    
    if use_ninja:
        env["CMAKE_GENERATOR"] = "Ninja"

    scripts_dir = os.path.join(os.path.dirname(sys.executable), "Scripts")
    if cuda_path:
        env["CUDA_PATH"] = cuda_root
        env["CUDA_HOME"] = cuda_root
        env["PATH"] = cuda_path + os.pathsep + scripts_dir + os.pathsep + env.get("PATH", "")
    else:
        env["PATH"] = scripts_dir + os.pathsep + env.get("PATH", "")

    vcvars = find_vcvars()
    local_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llama-cpp-python-src")
    
    if os.path.exists(local_src) and not force_pypi:
        print(f"  > [!] Local custom engine source detected: {local_src}")
        install_cmd = [sys.executable, "-m", "pip", "install", local_src, "--no-cache-dir", "--no-deps", "--upgrade", "--ignore-installed"]
    else:
        target_package = "llama-cpp-python"
        if use_source:
            target_package = "git+https://github.com/abetlen/llama-cpp-python.git"
        install_cmd = [sys.executable, "-m", "pip", "install", target_package, "--no-cache-dir", "--no-deps", "--upgrade"]

    try:
        if vcvars:
            import tempfile
            bat_path = os.path.join(tempfile.gettempdir(), "temp_install.bat")
            bat_content = f'call "{vcvars}"\n' + " ".join(f'"{x}"' if ' ' in x else x for x in install_cmd)
            with open(bat_path, "w") as f: f.write(bat_content)
            subprocess.check_call(["cmd.exe", "/c", bat_path], env=env)
            try: os.remove(bat_path)
            except: pass
        else:
            subprocess.check_call(install_cmd, env=env)

        # --- Build llama-diffusion-cli / Muse Glimmer backend from vendor source ---
        if os.path.exists(local_src) and not force_pypi:
            vendor_dir = os.path.join(local_src, "vendor", "llama.cpp")
            if os.path.exists(vendor_dir):
                print(f"\n  > [!] Building llama-diffusion-cli for Muse Glimmer workflows...")
                build_dir = os.path.join(vendor_dir, "build")
                
                cuda_arch_flag = next((f.strip('"') for f in cmake_flags if f.strip('"').startswith("-DCMAKE_CUDA_ARCHITECTURES=")), f"-DCMAKE_CUDA_ARCHITECTURES={CUDA_ARCH_TARGET}")
                cmake_gen_cmd = [
                    "cmake", "-S", vendor_dir, "-B", build_dir,
                    "-DGGML_CUDA=ON",
                    "-DSD_CUDA=ON",
                    cuda_arch_flag
                ]
                if cuda_root and not use_ninja:
                    cmake_gen_cmd.extend(["-T", f"cuda={cuda_root}"])
                
                cmake_build_cmd = ["cmake", "--build", build_dir, "-j", "--config", "Release", "--target", "llama-diffusion-cli"]
                
                def quote_arg(arg):
                    if ' ' in arg:
                        if arg.startswith("-T") and "=" in arg:
                            prefix, path = arg.split("=", 1)
                            return f'{prefix}="{path}"'
                        return f'"{arg}"'
                    return arg

                if vcvars:
                    bat_path = os.path.join(tempfile.gettempdir(), "temp_build_cli.bat")
                    bat_content = f'call "{vcvars}"\n' + " ".join(quote_arg(x) for x in cmake_gen_cmd) + "\n" + " ".join(quote_arg(x) for x in cmake_build_cmd)
                    with open(bat_path, "w") as f: f.write(bat_content)
                    subprocess.check_call(["cmd.exe", "/c", bat_path], env=env)
                    try: os.remove(bat_path)
                    except: pass
                else:
                    subprocess.check_call(cmake_gen_cmd, env=env)
                    subprocess.check_call(cmake_build_cmd, env=env)
                
                tools_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Tools")
                os.makedirs(tools_dir, exist_ok=True)
                    
                exe_src = os.path.join(build_dir, "bin", "Release", "llama-diffusion-cli.exe")
                exe_fallback = os.path.join(build_dir, "bin", "llama-diffusion-cli.exe")
                target_exe = os.path.join(tools_dir, "llama-diffusion-cli.exe")
                
                if os.path.exists(exe_src):
                    shutil.copy(exe_src, target_exe)
                    print(f"  > [V] Copied llama-diffusion-cli.exe to {target_exe}")
                elif os.path.exists(exe_fallback):
                    shutil.copy(exe_fallback, target_exe)
                    print(f"  > [V] Copied llama-diffusion-cli.exe to {target_exe}")

        return True
    except Exception as e:
        print(f"Install failed: {e}")
        return False

def update_core_environment_tools():
    """Updates pip, setuptools, and wheel."""
    print("\n--- [ Updating Pip & Core Build Tooling ] ---")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel", "--quiet"]
        )
        print("  ✅ Pip, setuptools, and wheel successfully updated.")
    except Exception as e:
        print(f"  ⚠️ Warning: Could not update core pip tooling: {e}")

def main():
    print("--- Serenity Apex: Hardware Initialization ---")
    
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    activate_local_venv()
    tmp_dir = os.path.join(workspace_dir, "tmp")
    cuda_cache_dir = os.path.join(workspace_dir, ".cuda_cache")
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(cuda_cache_dir, exist_ok=True)
    
    os.environ["TEMP"] = tmp_dir
    os.environ["TMP"] = tmp_dir
    os.environ["CUDA_CACHE_PATH"] = cuda_cache_dir
    os.environ["TORCH_EXTENSIONS_DIR"] = os.path.join(tmp_dir, "torch_extensions")
    os.environ["TRITON_CACHE_DIR"] = os.path.join(tmp_dir, "triton_cache")

    cuda_path = get_cuda_path()
    inject_cuda_path(cuda_path)

    check_build_environment()

    print("\n--- [ llama-cpp-python Stability Check ] ---")
    try:
        import llama_cpp
        if llama_cpp.llama_supports_gpu_offload():
            status_msg = "✅ Installed and stable (GPU acceleration verified)."
        else:
            status_msg = "⚠️  Installed, but running on CPU (no GPU offloading detected)."
    except Exception as e:
        status_msg = f"❌ Not installed or unstable ({e})."
        
    print("Status: " + status_msg)
    
    rebuild_local = "--rebuild" in sys.argv or "--source" in sys.argv
    auto_upgrade = "--upgrade-all" in sys.argv or "--latest" in sys.argv
    if rebuild_local:
        print("\n[!] Rebuild flag detected (--rebuild). Rebuilding engine from local source folder...")
        choice = "2"
    elif auto_upgrade:
        print("\n[!] Auto-upgrade flag detected (--upgrade-all / --latest). Upgrading engine...")
        choice = "5"
    else:
        print("\nSelect llama-cpp-python Option:")
        print(" [1] Keep existing installation")
        print(" [2] Rebuild/Install from local source folder (llama-cpp-python-src) with GPU acceleration")
        print(" [3] Install default version from PyPI")
        print(" [4] Skip engine setup entirely")
        print(" [5] Upgrade engine to LATEST git/PyPI release with GPU support & update dependencies")
        
        choice = input("Choice [1/2/3/4/5] (default: 1): ").strip() or "1"
        
    skip_reinstall = choice in ["1", "4"]
    force_pypi = choice in ["3", "5"]
    use_source_flag = choice == "5"

    if not skip_reinstall:
        print("[*] Uninstalling current llama-cpp-python...")
        subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "llama-cpp-python"])

        if choice != "5":
            build_choice = input("Select Build Method [1: Fast Build (Recommended), 2: Source Build]: ").strip() or "1"
            use_source_flag = (build_choice == "2")

        if install_engine(cuda_path, use_source=use_source_flag, force_pypi=force_pypi):
            print("\n[V] Engine rebuild complete. Verifying GPU support...")
            try:
                import llama_cpp
                if llama_cpp.llama_supports_gpu_offload():
                    print(" > [V] CUDA OFFLOAD ACTIVE.")
                else:
                    print(" > [X] ERROR: Engine built but CUDA is inactive.")
            except Exception as e:
                print(f" > [!] Verification failed: {e}")

    print("\n[STEP 4]: Finalizing Environment & Updating Core Tooling...")
    update_core_environment_tools()
    if os.path.exists(REQUIREMENTS_FILE):
        print("  > Syncing pip dependencies from requirements.txt...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_FILE])
            print("  ✅ Requirements successfully synced.")
        except subprocess.CalledProcessError as e:
            print(f"  ⚠️ Warning: Requirements sync encountered non-fatal issues: {e}")
    
    print("\n[STEP 5]: Setting up Web Automation Driver (Playwright)...")
    try:
        subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
        print("  > [V] Playwright Chromium Driver Ready.")
    except Exception as e:
        print(f"  > [!] Playwright Setup Failed: {e}")

    print("\n[STEP 6]: Creating Desktop Shortcuts...")
    try:
        shortcuts_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "System", "shortcuts.py")
        if os.path.exists(shortcuts_py):
            subprocess.check_call([sys.executable, shortcuts_py])
        
        setup_engine_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Live", "System", "setup_engine.py")
        if os.path.exists(setup_engine_py):
            subprocess.check_call([sys.executable, setup_engine_py])
    except Exception as e:
        print(f" > [!] Error creating shortcuts: {e}")

    if not any(flag in sys.argv for flag in ["--rebuild", "--upgrade-all", "--latest", "--source", "--non-interactive"]):
        input("\nPress Enter to finish...")
    print("\n✅ Serenity Apex Initialization Complete.")

if __name__ == "__main__":
    main()
