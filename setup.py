import sys
import subprocess
import os
import tempfile
import shutil
import glob
import re
import time
import ctypes
import traceback

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# --- Localize Temp and Cache Paths ---
_curr = os.path.abspath(__file__)
while True:
    _parent = os.path.dirname(_curr)
    if _parent == _curr:
        _workspace = os.path.dirname(os.path.abspath(__file__))
        break
    if os.path.exists(os.path.join(_parent, "serenity_resources.py")) or os.path.exists(os.path.join(_parent, ".git")):
        _workspace = _parent
        break
    _curr = _parent

_cache_dir = os.path.join(_workspace, ".serenity_cache")
_cuda_dir = os.path.join(_cache_dir, "cuda")
_triton_dir = os.path.join(_cache_dir, "triton")
_torch_ext_dir = os.path.join(_cache_dir, "torch_extensions")
_pip_dir = os.path.join(_cache_dir, "pip")

for _d in [_cuda_dir, _triton_dir, _torch_ext_dir, _pip_dir]:
    os.makedirs(_d, exist_ok=True)

_user_temp = os.path.join(os.environ.get("USERPROFILE", "C:\\Users\\Default"), "AppData", "Local", "Temp")
if os.path.exists(_user_temp):
    os.environ["TEMP"] = _user_temp
    os.environ["TMP"] = _user_temp
    os.environ["TMPDIR"] = _user_temp
    tempfile.tempdir = _user_temp

os.environ["CUDA_CACHE_PATH"] = _cuda_dir
os.environ["TRITON_CACHE_DIR"] = _triton_dir
os.environ["TORCH_EXTENSIONS_DIR"] = _torch_ext_dir
os.environ["PIP_CACHE_DIR"] = _pip_dir

# --- Configuration ---
REQUIREMENTS_FILE = "requirements.txt"
# For 12700KF: We target the 8 P-Cores (16 threads). 
# For 3050 LP: sm_86 is the architecture.
# --- End Configuration ---

def ensure_venv():
    """Ensure setup runs inside a .venv; create and re-exec if running in base Python."""
    env_file = os.path.join(_workspace, ".env")
    if not os.path.exists(env_file):
        print(f"[*] Creating default .env file at {env_file}...")
        with open(env_file, "w") as f:
            f.write("# Serenity PC Environment Configuration\n")
            f.write("# Legacy GPU Target: sm_50, sm_61\n")

    if sys.prefix != sys.base_prefix:
        if not shutil.which("ninja"):
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "ninja", "--quiet"])
            except Exception:
                pass
        return

    venv_dir = os.path.join(_workspace, ".venv")
    is_win = os.name == "nt"
    py_exe = os.path.join(venv_dir, "Scripts" if is_win else "bin", "python.exe" if is_win else "python")

    if not os.path.exists(py_exe):
        print(f"[*] Creating .venv environment at: {venv_dir}")
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
        wheels_dir = os.path.join(_workspace, "wheels")
        if os.path.exists(wheels_dir) and os.listdir(wheels_dir):
            print("  > Installing bootstrapping packages from local wheels...")
            subprocess.check_call([py_exe, "-m", "pip", "install", "--no-index", "--find-links", wheels_dir, "pip", "wheel", "setuptools", "ninja", "--quiet"])
        else:
            subprocess.check_call([py_exe, "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools", "ninja", "--quiet"])

    print(f"[*] Relaunching setup.py inside .venv ({py_exe})...")
    subprocess.check_call([py_exe] + sys.argv)
    sys.exit(0)

def get_short_path(path):
    """Convert Windows path to 8.3 short path format to prevent space-splitting in CMake arguments."""
    if not path or not os.path.exists(path):
        return path
    if os.name == "nt" and hasattr(ctypes, "windll"):
        try:
            buf = ctypes.create_unicode_buffer(500)
            res = ctypes.windll.kernel32.GetShortPathNameW(path, buf, 500)
            if res > 0:
                return buf.value
        except Exception:
            pass
    return path

def get_cuda_path():
    base_install = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if os.path.exists(base_install):
        versions = glob.glob(os.path.join(base_install, "v*")) 
        if versions:
            v12 = [v for v in versions if os.path.basename(v).startswith("v12")]
            if v12:
                return os.path.join(sorted(v12)[-1], "bin")
            return os.path.join(sorted(versions)[-1], "bin")
    return os.environ.get('CUDA_PATH') or os.environ.get('CUDA_HOME')

def inject_cuda_path(cuda_bin_path):
    if cuda_bin_path and os.path.isdir(cuda_bin_path):
        cuda_root = os.path.dirname(cuda_bin_path)
        cuda_root_short = get_short_path(cuda_root)
        cuda_bin_short = get_short_path(cuda_bin_path)
        nvcc_path = get_short_path(os.path.join(cuda_bin_path, "nvcc.exe"))
        
        os.environ["CUDA_PATH"] = cuda_root_short
        os.environ["CUDA_HOME"] = cuda_root_short
        os.environ["CUDAToolkit_ROOT"] = cuda_root_short
        if os.path.exists(nvcc_path):
            os.environ["CUDACXX"] = nvcc_path

        # Strip conflicting CUDA toolkits from PATH and prepend selected CUDA
        paths = os.environ.get("PATH", "").split(os.pathsep)
        clean_paths = [p for p in paths if not ("CUDA" in p and "NVIDIA GPU Computing Toolkit" in p and cuda_root.lower() not in p.lower())]
        cuda_nvvp = os.path.join(cuda_root, "libnvvp")
        clean_paths = [cuda_bin_short, cuda_nvvp] + [p for p in clean_paths if p not in (cuda_bin_short, cuda_nvvp)]
        os.environ["PATH"] = os.pathsep.join(clean_paths)

        # Clean conflicting CUDA toolkits from INCLUDE and LIB variables
        for var in ["INCLUDE", "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LIB", "LIBPATH"]:
            if var in os.environ:
                cleaned = [p for p in os.environ[var].split(os.pathsep) if not ("CUDA" in p and "NVIDIA GPU Computing Toolkit" in p and cuda_root.lower() not in p.lower())]
                os.environ[var] = os.pathsep.join(cleaned)

        if hasattr(os, 'add_dll_directory'):
            try:
                os.add_dll_directory(cuda_bin_path)
                lib_path = os.path.join(cuda_root, "lib", "x64")
                if os.path.exists(lib_path) and os.path.isdir(lib_path):
                    os.add_dll_directory(lib_path)
                print(f"  > [V] Apex Link Established: {cuda_bin_path}")
            except Exception as e:
                print(f"  > [!] DLL Error: {e}")

def find_vcvars():
    """Locate Visual Studio / MSVC vcvars64.bat, prioritizing VS 2022 for CUDA compatibility."""
    vs2022_paths = [
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
    ]
    for p in vs2022_paths:
        if os.path.exists(p):
            return p

    vswhere_paths = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe",
        r"C:\Program Files\Microsoft Visual Studio\Installer\vswhere.exe",
    ]
    for vswhere_exe in vswhere_paths:
        if os.path.exists(vswhere_exe):
            try:
                cmd = [vswhere_exe, "-version", "[17.0,18.0)", "-products", "*", "-property", "installationPath"]
                out = subprocess.check_output(cmd, text=True, errors="ignore").strip()
                if not out:
                    cmd = [vswhere_exe, "-latest", "-prerelease", "-products", "*", "-property", "installationPath"]
                    out = subprocess.check_output(cmd, text=True, errors="ignore").strip()
                if out:
                    for inst in out.splitlines():
                        inst = inst.strip()
                        if inst:
                            bat = os.path.join(inst, "VC", "Auxiliary", "Build", "vcvars64.bat")
                            if os.path.exists(bat):
                                return bat
                            bat_all = os.path.join(inst, "VC", "Auxiliary", "Build", "vcvarsall.bat")
                            if os.path.exists(bat_all):
                                return bat_all
            except Exception:
                pass

    paths = [
        r"C:\Program Files\Microsoft Visual Studio\2026\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2026\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2026\Preview\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    ]
    for p in paths:
        if os.path.exists(p):
            return p

    glob_patterns = [
        r"C:\Program Files\Microsoft Visual Studio\2022\*\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2022\*\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat",
    ]
    for pattern in glob_patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]

    return None

def capture_vcvars_env(vcvars_path):
    if not vcvars_path or not os.path.exists(vcvars_path):
        return {}
    
    toolsets = ["-vcvars_ver=14.4", "-vcvars_ver=14.3", ""]
    for ts in toolsets:
        try:
            cmd = f'call "{vcvars_path}" {ts} >nul 2>&1 && set'
            output = subprocess.check_output(cmd, shell=True, text=True, errors='ignore')
            env_vars = {}
            for line in output.splitlines():
                if '=' in line:
                    k, v = line.split('=', 1)
                    env_vars[k] = v
            if env_vars and "VCINSTALLDIR" in env_vars:
                if ts:
                    print(f"  > [V] Selected VS 2022 Toolset ({ts}) for CUDA nvcc stability.")
                return env_vars
        except Exception:
            pass
    return {}

def setup_msvc_env():
    """Detect and inject MSVC build environment on Windows, or configure GCC/Clang on POSIX."""
    if os.name != "nt":
        if shutil.which("gcc"):
            os.environ.setdefault("CC", "gcc")
            os.environ.setdefault("CXX", "g++")
            print("[V] GCC Compiler Environment configured.")
            return True
        elif shutil.which("clang"):
            os.environ.setdefault("CC", "clang")
            os.environ.setdefault("CXX", "clang++")
            print("[V] Clang Compiler Environment configured.")
            return True
        print("[X] Neither GCC nor Clang found in system PATH.")
        return False

    vcvars = find_vcvars()
    if vcvars:
        vc_env = capture_vcvars_env(vcvars)
        if vc_env:
            os.environ.update(vc_env)
            os.environ["CC"] = "cl"
            os.environ["CXX"] = "cl"
            print(f"[V] MSVC Compiler Environment initialized ({vcvars}).")
            return True
        else:
            print(f"[!] Found {vcvars} but failed to capture environment.")
    else:
        print("[X] Visual Studio MSVC compiler (vcvars64.bat) NOT found.")
    return False

def check_build_environment():
    """Pre-flight check to identify why builds usually fail."""
    print("\n--- [ Pre-flight Build Environment Check ] ---")
    if os.name == "nt":
        compiler_name = "MSVC (cl.exe)"
        compiler_path = shutil.which("cl")
    else:
        compiler_name = "GCC / Clang"
        compiler_path = shutil.which("gcc") or shutil.which("clang") or shutil.which("cc")

    checks = {
        "CMake": shutil.which("cmake"),
        compiler_name: compiler_path,
        "CUDA Compiler (nvcc)": shutil.which("nvcc"),
        "Git": shutil.which("git"),
        "Ninja": shutil.which("ninja")
    }
    
    all_passed = True
    for tool, path in checks.items():
        if path:
            print(f"[V] {tool}: Found at {path}")
        else:
            print(f"[X] {tool}: NOT FOUND")
            all_passed = False
    
    if not all_passed:
        print("\n[!] WARNING: Some build tools are missing. llama-cpp-python build WILL fail.")
        if os.name == "nt":
            print("Ensure Visual Studio (C++ Desktop Dev), CMake, and CUDA Toolkit are installed.")
        else:
            print("Ensure build-essential / Xcode CLI tools, CMake, and CUDA/Metal are installed.")
        print("--------------------------------------------------\n")
    else:
        print("[V] All critical build tools detected.\n")
    return all_passed

import re

def check_cuda_version_and_options(cuda_path=None):
    """Detect CUDA version and prompt options if CUDA 13+ is present on legacy hardware."""
    try:
        nvcc_bin = os.path.join(cuda_path, "nvcc.exe") if cuda_path and os.path.exists(os.path.join(cuda_path, "nvcc.exe")) else "nvcc"
        out = subprocess.check_output([nvcc_bin, "--version"], text=True)
        match = re.search(r"release (\d+)\.", out)
        if match and int(match.group(1)) >= 13:
            print("\n==================================================")
            print("⚠️  WARNING: CUDA 13+ Detected!")
            print("CUDA 13+ has dropped compiler support for legacy GPUs (sm_50, sm_61).")
            print("To compile for Maxwell/Pascal (GTX 900/1000 series), CUDA 12.x is recommended.")
            print("==================================================")
            print("Options:")
            print(" [1] Attempt Legacy build anyway (Fallback archs)")
            print(" [2] Force CPU-only mode (-DGGML_CUDA=off)")
            print(" [3] Abort setup (to install CUDA 12.x Toolkit)")
            choice = input("Select option [1/2/3] (default 1): ").strip() or "1"
            if choice == "2":
                return "cpu", "50;61;70;75;80;86"
            elif choice == "3":
                print("Exiting setup.")
                sys.exit(0)
            return "cuda", "50;61;86"
    except Exception:
        pass
    return "cuda", "50;61;86"

def install_engine(cuda_path, use_source=False, force_pypi=False, build_mode="cuda", cuda_archs="50;61;86"):
    """
    Unified installer optimized for legacy hardware compatibility (sm_50, sm_61 support).
    """
    print(f"\n[ ! ] PHASE: Building Apex Engine (Legacy GPU Support: {cuda_archs})...")
    
    cuda_root = os.path.dirname(cuda_path) if cuda_path else ""
    cuda_flag = "on" if build_mode == "cuda" else "off"
    
    cmake_args_list = [
        f"-DGGML_CUDA={cuda_flag}",
        f"-DCMAKE_CUDA_ARCHITECTURES={cuda_archs}",
        "-DGGML_AVX512=OFF",
        "-DCMAKE_CXX_STANDARD=17",
        "-DCMAKE_CUDA_STANDARD=17",
        "-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler",
    ]
    if cuda_path and os.path.exists(cuda_path):
        cuda_root_short = get_short_path(cuda_root).replace("\\", "/")
        nvcc_short = get_short_path(os.path.join(cuda_path, "nvcc.exe")).replace("\\", "/")
        cmake_args_list.append(f"-DCUDAToolkit_ROOT={cuda_root_short}")
        cmake_args_list.append(f"-DCMAKE_CUDA_COMPILER={nvcc_short}")

    cmake_args = " ".join(cmake_args_list)

    env = os.environ.copy()
    env["CMAKE_BUILD_PARALLEL_LEVEL"] = "4"
    env["MAX_JOBS"] = "4"
    scripts_dir = os.path.join(os.path.dirname(sys.executable), "Scripts")
    if cuda_path:
        cuda_bin_short = get_short_path(cuda_path)
        cuda_root_short = get_short_path(cuda_root)
        nvcc_short = get_short_path(os.path.join(cuda_path, "nvcc.exe"))
        env["CUDA_PATH"] = cuda_root_short
        env["CUDA_HOME"] = cuda_root_short
        env["CUDAToolkit_ROOT"] = cuda_root_short
        env["CUDACXX"] = nvcc_short
        env["PATH"] = cuda_bin_short + os.pathsep + scripts_dir + os.pathsep + env.get("PATH", "")
    else:
        env["PATH"] = scripts_dir + os.pathsep + env.get("PATH", "")

    user_temp = os.path.join(os.environ.get("USERPROFILE", "C:\\Users\\Default"), "AppData", "Local", "Temp")
    if os.path.exists(user_temp):
        env["TEMP"] = user_temp
        env["TMP"] = user_temp
        env["TMPDIR"] = user_temp

    if shutil.which("ninja") or os.path.exists(os.path.join(scripts_dir, "ninja.exe")):
        env["CMAKE_GENERATOR"] = "Ninja"
        print("🚀 Using Ninja generator for faster/reliable build.")
    elif cuda_root:
        cmake_args += f' -T "cuda={cuda_root}"'

    env["CMAKE_ARGS"] = cmake_args
    env["CUDAFLAGS"] = "-allow-unsupported-compiler"
    env["NVCC_PREPEND_FLAGS"] = "-allow-unsupported-compiler"
    env["FORCE_CMAKE"] = "1"

    vcvars = find_vcvars()
    if vcvars:
        vc_env = capture_vcvars_env(vcvars)
        env.update(vc_env)
        env["CC"] = "cl"
        env["CXX"] = "cl"
    
    # Command construction
    local_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llama-cpp-python-src")
    if os.path.exists(local_src) and not force_pypi:
        print(f"  > [!] Local custom engine source detected: {local_src}")
        install_cmd = [sys.executable, "-m", "pip", "install", get_short_path(local_src), "--no-cache-dir", "--no-deps", "--upgrade"]
    else:
        target_package = "llama-cpp-python==0.3.26"
        if use_source:
            target_package = "git+https://github.com/abetlen/llama-cpp-python.git"
        install_cmd = [sys.executable, "-m", "pip", "install", target_package, "--no-cache-dir", "--no-deps", "--upgrade"]

    try:
        subprocess.check_call(install_cmd, env=env)
        return True
    except Exception as e:
        print(f"Install failed: {e}")
        return False

def install_llama_cpp(is_source=False):
    """
    Attempts to install llama-cpp-python with detailed error reporting.
    """
    print(f"\n--- [ Attempting Installation (Source={is_source}) ] ---")
    
    # 1. Prepare Environment Variables
    # We must ensure the current process sees the build tools
    env = os.environ.copy()
    env["CUDAFLAGS"] = "-allow-unsupported-compiler"
    env["NVCC_PREPEND_FLAGS"] = "-allow-unsupported-compiler"
    
    # Force CMake to use Ninja if available (much more reliable on Windows)
    if shutil.which("ninja"):
        env["CMAKE_GENERATOR"] = "Ninja"
        print("🚀 Using Ninja generator for faster/more reliable builds.")

    # 2. Construct the command
    local_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llama-cpp-python-src")
    if os.path.exists(local_src):
        print(f"  > [!] Local custom engine source detected: {local_src}")
        cmd = [sys.executable, "-m", "pip", "install", local_src, "--no-deps", "--upgrade", "--no-cache-dir"]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "llama-cpp-python", "--no-deps", "--upgrade", "--no-cache-dir"]
    
    # If we are doing a source install or local install
    if is_source or os.path.exists(local_src):
        # These are the critical flags for llama-cpp-python
        env["CMAKE_ARGS"] = "-DGGML_CUDA=on -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler" 
        print("🛠️  Setting CMAKE_ARGS: -DGGML_CUDA=on -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler")

    try:
        # We use subprocess.run with check=True to catch errors
        # We do NOT capture output so it streams directly to the user's terminal
        # This allows the user to see the EXACT C++ error.
        subprocess.run(cmd, env=env, check=True)
        print("\n✅ Installation successful!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ INSTALLATION FAILED!")
        print(f"Exit Code: {e.returncode}")
        print("\n--- [ DIAGNOSTIC TIP ] ---")
        print("Look at the logs above for the first 'error:' line.")
        print("Common causes:")
        print("1. 'cl.exe' not found -> Install Visual Studio C++ Build Tools.")
        print("2. 'cmake' not found -> Install CMake and add to PATH.")
        print("3. 'nvcc' not found -> Install CUDA Toolkit.")
        print("4. 'CUDA_TOOLKIT_ROOT_DIR' not set -> Ensure CUDA is in your PATH.")
        return False

def run_command(cmd_list):
    try:
        print(f"Running: {' '.join(cmd_list)}")
        subprocess.check_call(cmd_list)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Command failed: {' '.join(cmd_list)}")
        print(f"❌ Exit Code: {e.returncode}")
        print("💡 Tip: Ensure Visual Studio C++ Build Tools and CUDA are installed.\n")
        return False
    return True

def force_cleanup_packages(package_names):
    """
    Forcefully removes packages that are causing 'uninstall-no-record-file' errors
    by deleting their site-packages directories manually.
    """
    print(f"[*] Attempting to clean up potentially broken packages: {package_names}")
    import site
    site_packages = site.getsitepackages()[0]
    
    for pkg in package_names:
        # Check for both the package and the .dist-info directory
        pkg_dir = os.path.join(site_packages, pkg.replace('-', '_'))
        dist_info_dir = os.path.join(site_packages, f"{pkg.replace('-', '_')}-*.dist-info")
        
        # We use a simple approach: try to uninstall via pip first, 
        # if it fails, we don't crash, we just move on.
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", pkg], 
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print(f"[!] Pip could not uninstall {pkg} normally. Proceeding to manual cleanup...")
            # Manual cleanup is risky, so we only do it if we are sure.
            # For this script, we'll try to find the directory and remove it.
            # This is a fallback.
            pass

def install_with_retry(command):
    """Attempts to run a command, and if it fails due to uninstall errors, cleans up and retries."""
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        error_msg = str(e)
        # Check if the error is related to missing RECORD files
        if "uninstall-no-record-file" in error_msg or "cannot uninstall" in error_msg.lower():
            print("[!] Detected broken package metadata. Attempting force cleanup...")
            # Common culprits in your logs
            force_cleanup_packages(["numpy", "llama-cpp-python", "llama-cpp"])
            print("[*] Retrying installation...")
            subprocess.check_call(command)
        else:
            raise e

def main():
    ensure_venv()
    print("--- Serenity Apex: Hardware Initialization ---")
    setup_msvc_env()
    cuda_path = get_cuda_path()
    inject_cuda_path(cuda_path)
    check_build_environment()

    # 1. Check for existing hardware-accelerated build & verify stability
    print("\n--- [ llama-cpp-python Stability Check ] ---")
    status_msg = ""
    is_installed = False
    gpu_supported = False
    
    try:
        import llama_cpp
        is_installed = True
        if llama_cpp.llama_supports_gpu_offload():
            gpu_supported = True
            status_msg = "[V] Installed and stable (GPU acceleration verified)."
        else:
            status_msg = "[!] Installed, but running on CPU (no GPU offloading detected)."
    except Exception as e:
        status_msg = f"[X] Not installed yet ({e})."
        
    print(f"Status: {status_msg}")
    if not is_installed:
        print("\n[*] First-time installation detected: Pre-compiled engine wheel will be installed automatically from local wheels directory.")
    print("\nSelect llama-cpp-python Option:")
    print(" [1] Fast Install / Keep existing (Recommended)")
    print(" [2] Rebuild/Install from local source folder (llama-cpp-python-src) with GPU acceleration")
    print(" [3] Install default version from PyPI")
    print(" [4] Skip engine setup entirely")
    
    choice = input("Choice [1/2/3/4] (default: 1): ").strip()
    if not choice:
        choice = "1"
        
    skip_reinstall = True
    force_pypi = False
    
    if choice == "2":
        skip_reinstall = False
        force_pypi = False
    elif choice == "3":
        skip_reinstall = False
        force_pypi = True
    elif choice == "4":
        skip_reinstall = True
    else:  # choice == "1"
        skip_reinstall = True
        if not is_installed:
            wheels_dir = os.path.join(_workspace, "wheels")
            cuda_wheels = glob.glob(os.path.join(wheels_dir, "llama_cpp_python*.whl"))
            if cuda_wheels:
                print(f"[*] Installing pre-compiled CUDA engine wheel ({os.path.basename(cuda_wheels[0])})...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", wheels_dir, "llama-cpp-python", "--no-deps"])

    if not skip_reinstall:
        # Uninstall llama-cpp-python only to prevent caching conflicts
        print("[*] Uninstalling llama-cpp-python (dependencies will not be touched)...")
        subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "llama-cpp-python"])

        # 3. Rebuild for Legacy GPU Hardware
        build_mode, cuda_archs = check_cuda_version_and_options(cuda_path)
        
        print("\nSelect Build Method:")
        print(" [1] Fast Build (Recommended)")
        print(" [2] Source Build (If Offload Toggle is broken)")
        build_choice = input("Choice [1/2] (default: 1): ").strip()
        use_source_flag = (build_choice == "2")

        if install_engine(cuda_path, use_source=use_source_flag, force_pypi=force_pypi, build_mode=build_mode, cuda_archs=cuda_archs):
            print("\n[V] Engine rebuild complete. Verifying GPU support...")
            try:
                import llama_cpp
                if llama_cpp.llama_supports_gpu_offload():
                    print(" > [V] CUDA OFFLOAD ACTIVE.")
                else:
                    print(" > [X] ERROR: Engine built but CUDA is inactive.")
            except Exception as e:
                print(f" > [!] Verification failed: {e}")

    # 4. Finalizing Web & Audio Dependencies
    print("\n[STEP 4]: Finalizing Environment...")
    if os.path.exists(REQUIREMENTS_FILE):
        print("  > Updating pip dependencies...")
        wheels_dir = os.path.join(_workspace, "wheels")
        if os.path.exists(wheels_dir) and os.listdir(wheels_dir):
            print("  > Installing from local wheels directory...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--find-links", wheels_dir, "-r", REQUIREMENTS_FILE, "--quiet"])
        else:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_FILE, "--quiet"])
    
    print("\n[STEP 5]: Setting up Web Automation Driver (Playwright)...")
    try:
        subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
        print("  > [V] Playwright Chromium Driver Ready.")
    except Exception as e:
        print(f"  > [!] Playwright Setup Failed: {e}")

    # 5. Create Desktop Shortcuts by invoking the respective setup scripts
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

    input("\nPress Enter to finish...")

if __name__ == "__main__":
    main()



