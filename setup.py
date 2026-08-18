import os
import sys
import shutil
import subprocess
import glob
import re

def get_cuda_version():
    """Extract CUDA major/minor version from nvcc and enforce CUDA >= 13.3."""
    try:
        out = subprocess.check_output(["nvcc", "--version"], text=True)
        match = re.search(r"release (\d+)\.(\d+)", out)
        if match:
            major, minor = int(match.group(1)), int(match.group(2))
            if (major, minor) < (13, 3):
                raise RuntimeError(f"CUDA {major}.{minor} detected. SerenityPC requires CUDA 13.3+ (no legacy supported).")
            return major, minor
    except Exception as e:
        print(f"[!] CUDA version check failed: {e}")
        return None, None
    return None, None

def get_safe_cuda_archs():
    """Detect local GPU compute cap or default to native for fast single-arch compilation."""
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], text=True)
        cap = out.strip().split("\n")[0].replace(".", "").strip()
        if cap and cap.isdigit():
            print(f"[*] Detected local GPU Compute Capability: sm_{cap}")
            return cap
    except Exception:
        pass
    return "native"

def kill_locking_processes():
    """Kill other Python processes locking DLL files in .venv (excluding self)."""
    print("[*] Clearing file locks on .venv DLLs...")
    if sys.platform != "win32":
        return
    try:
        curr_pid = str(os.getpid())
        # Get all python PIDs whose path contains .venv
        result = subprocess.run(
            ["wmic", "process", "where",
             "name='python.exe'",
             "get", "ProcessId,ExecutablePath", "/format:csv"],
            capture_output=True, text=True
        )
        for line in result.stdout.splitlines():
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            # csv columns: Node, ExecutablePath, ProcessId
            exe_path, pid = parts[1], parts[2]
            if ".venv" in exe_path and pid != curr_pid:
                subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True)
                print(f"  > Killed locked process PID {pid}")
    except Exception as e:
        print(f"[!] Process cleanup note: {e}")



def resolve_conflicts_and_cleanup():
    """Aggressive cleanup for orphaned packages and build caches."""
    kill_locking_processes()
    
    venv_site = os.path.join(".venv", "Lib", "site-packages")
    if os.path.exists(venv_site):
        print(f"[*] Sweeping {venv_site} for broken installations...")
        patterns = ["*llama*", "~*"]
        for pat in patterns:
            for item in glob.glob(os.path.join(venv_site, pat)):
                try:
                    if os.path.isdir(item):
                        shutil.rmtree(item)
                    else:
                        os.remove(item)
                    print(f"  > Removed stale: {item}")
                except Exception as e:
                    print(f"  [!] Could not remove {item}: {e}")

    # Clear CMake and pip build temp folders
    for temp_dir in ["tmp", "build", "_skbuild"]:
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"  > Cleaned build cache: {temp_dir}")
            except Exception as e:
                print(f"  [!] Failed removing {temp_dir}: {e}")

def get_msvc_env():
    """Locate and return MSVC environment dictionary via vcvarsall.bat."""
    if sys.platform != "win32":
        return {}
    vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    vcvars_candidates = []
    if os.path.exists(vswhere):
        try:
            out = subprocess.check_output([
                vswhere, "-latest", "-products", "*",
                "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property", "installationPath"
            ], text=True).strip()
            if out:
                vcvars_candidates.append(os.path.join(out, r"VC\Auxiliary\Build\vcvarsall.bat"))
        except Exception:
            pass
    vcvars_candidates.extend([
        r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvarsall.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat",
    ])
    for bat in vcvars_candidates:
        if os.path.exists(bat):
            try:
                cmd = f'cmd.exe /c "call \"{bat}\" x64 && set"'
                out = subprocess.check_output(cmd, shell=True, text=True)
                env = {}
                for line in out.splitlines():
                    if "=" in line:
                        k, v = line.split("=", 1)
                        env[k.upper()] = v
                return env
            except Exception:
                pass
    return {}

def configure_cuda_toolchain(env=None):
    """Isolate and configure latest CUDA toolchain and MSVC, stripping legacy CUDA and MinGW paths from env."""
    if env is None:
        env = os.environ.copy()

    vc_env = get_msvc_env()
    for k, v in vc_env.items():
        env[k] = v

    cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    chosen_cuda = None
    if os.path.exists(cuda_base):
        candidates = []
        for entry in os.listdir(cuda_base):
            if entry.startswith("v"):
                try:
                    ver_parts = tuple(map(int, entry[1:].split(".")))
                    candidates.append((ver_parts, os.path.join(cuda_base, entry)))
                except ValueError:
                    pass
        candidates.sort(reverse=True)
        if candidates:
            chosen_cuda = candidates[0][1]
            print(f"[*] Selected CUDA Toolkit: {chosen_cuda} (found {len(candidates)} installed versions)")

    if not chosen_cuda and "CUDA_PATH" in env and os.path.exists(env["CUDA_PATH"]):
        chosen_cuda = env["CUDA_PATH"]

    if chosen_cuda:
        env["CUDA_PATH"] = chosen_cuda
        env["CUDA_HOME"] = chosen_cuda
        nvcc_bin = os.path.join(chosen_cuda, "bin", "nvcc.exe")
        if os.path.exists(nvcc_bin):
            env["CUDACXX"] = nvcc_bin

        # Clean MinGW/GCC and legacy CUDA from PATH
        paths = env.get("PATH", "").split(os.pathsep)
        clean_paths = []
        cuda_bin = os.path.join(chosen_cuda, "bin")
        cuda_nvvp = os.path.join(chosen_cuda, "libnvvp")
        clean_paths.extend([cuda_bin, cuda_nvvp])

        for p in paths:
            p_lower = p.lower()
            if any(k in p_lower for k in ["w64devkit", "mingw", "msys"]):
                continue
            if "nvidia gpu computing toolkit" in p_lower and chosen_cuda.lower() not in p_lower:
                continue
            if p and p not in clean_paths:
                clean_paths.append(p)

        env["PATH"] = os.pathsep.join(clean_paths)

        # Clean legacy CUDA from INCLUDE and LIB variables
        for var in ["INCLUDE", "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LIB", "LIBPATH"]:
            if var in env:
                cleaned = [p for p in env[var].split(os.pathsep) if not ("CUDA" in p and "NVIDIA GPU Computing Toolkit" in p and chosen_cuda.lower() not in p.lower())]
                env[var] = os.pathsep.join(cleaned)

    return env, chosen_cuda

def build_local_llama(target_python=sys.executable):
    """Build local llama-cpp-python-src targeting detected native architecture."""
    resolve_conflicts_and_cleanup()
    
    src_dir = "llama-cpp-python-src"
    if not os.path.exists(src_dir):
        print(f"[!] Error: Local source directory '{src_dir}' not found.")
        return False

    # Check for submodule completeness
    cmake_check = os.path.join(src_dir, "vendor", "llama.cpp", "CMakeLists.txt")
    if not os.path.exists(cmake_check):
        print(f"[!] Error: '{src_dir}' is incomplete (vendor/llama.cpp/CMakeLists.txt not found).")
        print("    Run 'git submodule update --init --recursive' to initialize submodules before building.")
        return False

    # Ensure build dependencies in target python
    print("[*] Ensuring build dependencies (scikit-build-core, cmake, ninja)...")
    subprocess.run([target_python, "-m", "pip", "install", "--quiet", "scikit-build-core", "cmake", "ninja", "pyproject_metadata", "pathspec"])

    archs = get_safe_cuda_archs()
    print(f"[*] Target CUDA Architectures for NVCC: {archs}")

    env, cuda_dir = configure_cuda_toolchain()
    toolkit_arg = f' -DCUDAToolkit_ROOT="{cuda_dir}"' if cuda_dir else ""
    nvcc_bin = os.path.join(cuda_dir, "bin", "nvcc.exe") if cuda_dir else ""
    nvcc_arg = f' -DCMAKE_CUDA_COMPILER="{nvcc_bin}"' if (nvcc_bin and os.path.exists(nvcc_bin)) else ""

    env["CMAKE_ARGS"] = f"-DGGML_CUDA=on -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_CUDA_ARCHITECTURES={archs}{toolkit_arg}{nvcc_arg} -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler"
    env["CUDAFLAGS"] = "-allow-unsupported-compiler"
    env["NVCC_PREPEND_FLAGS"] = "-allow-unsupported-compiler"
    env["CMAKE_BUILD_PARALLEL_LEVEL"] = str(os.cpu_count() or 8)
    env["FORCE_CMAKE"] = "1"

    src_path = os.path.abspath(src_dir)
    cmd = [target_python, "-m", "pip", "install", "-v", "--no-cache-dir", "--no-build-isolation", "--force-reinstall", "--no-deps", src_path]
    print(f"[*] Executing parallel build ({env['CMAKE_BUILD_PARALLEL_LEVEL']} CPU threads): {' '.join(cmd)}")
    
    res = subprocess.run(cmd, env=env)
    
    # Ensure compiled DLLs are present in .venv site-packages lib directory
    built_lib = os.path.join(src_path, "llama_cpp", "lib")
    if os.path.exists(built_lib):
        venv_site_lib = os.path.join(os.path.dirname(os.path.dirname(target_python)), "Lib", "site-packages", "llama_cpp", "lib")
        if os.path.exists(os.path.dirname(venv_site_lib)):
            os.makedirs(venv_site_lib, exist_ok=True)
            for f in os.listdir(built_lib):
                shutil.copy2(os.path.join(built_lib, f), os.path.join(venv_site_lib, f))
            print(f"[*] Synchronized {len(os.listdir(built_lib))} compiled runtime libraries to {venv_site_lib}")

    return res.returncode == 0

def ensure_and_activate_venv():
    """Ensure .venv exists and activate it by setting VIRTUAL_ENV and updating PATH."""
    venv_dir = os.path.abspath(".venv")
    if not os.path.exists(venv_dir):
        print(f"[*] Creating virtual environment in {venv_dir}...")
        subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)

    scripts_dir = os.path.join(venv_dir, "Scripts") if sys.platform == "win32" else os.path.join(venv_dir, "bin")
    os.environ["VIRTUAL_ENV"] = venv_dir
    os.environ["PATH"] = scripts_dir + os.pathsep + os.environ.get("PATH", "")
    return os.path.join(scripts_dir, "python.exe" if sys.platform == "win32" else "python")

def menu():
    print("""
==================================================
   SerenityPC Developer Orchestrator (Personal)   
==================================================
 [1] Setup Global Environment
 [2] Setup .venv Environment (Full Build)
 [3] Prepare .venv for SETUPfile.py (Base deps only)
 [4] Update Everything (Deps + Local Llama Build)
 [5] Update Llama Only (Local Source Build)
 [6] Resolve Conflicts & Manual Cleanup
==================================================
    """)
    choice = input("Select option [1-6] (default 2): ").strip() or "2"

    
    if choice == "1":
        print("[*] Installing requirements to global environment...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        build_local_llama(sys.executable)

    elif choice == "2":
        py_venv = ensure_and_activate_venv()
        resolve_conflicts_and_cleanup()
        print("[*] Installing base requirements to .venv...")
        subprocess.run([py_venv, "-m", "pip", "install", "-r", "requirements.txt"])
        build_local_llama(py_venv)

    elif choice == "3":
        py_venv = ensure_and_activate_venv()
        print("[*] Setting up base .venv for SETUPfile.py testing...")
        subprocess.run([py_venv, "-m", "pip", "install", "-r", "requirements.txt"])
        print("[V] Base .venv configured. Ready for user-facing SETUPfile.py deployment test.")

    elif choice == "4":
        py_venv = ensure_and_activate_venv()
        resolve_conflicts_and_cleanup()
        subprocess.run([py_venv, "-m", "pip", "install", "--upgrade", "-r", "requirements.txt"])
        build_local_llama(py_venv)

    elif choice == "5":
        py_venv = ensure_and_activate_venv()
        build_local_llama(py_venv)

    elif choice == "6":
        resolve_conflicts_and_cleanup()
        print("[V] Manual cleanup completed successfully.")

if __name__ == "__main__":
    menu()

