import os
import sys
import subprocess
import shutil
import re

def ensure_venv():
    """Bootstrap and re-execute inside .venv if invoked globally."""
    venv_dir = os.path.abspath(".venv")
    if not os.path.exists(venv_dir):
        print("[*] Creating isolated virtual environment (.venv)...")
        subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
    
    scripts_dir = os.path.join(venv_dir, "Scripts") if sys.platform == "win32" else os.path.join(venv_dir, "bin")
    os.environ["VIRTUAL_ENV"] = venv_dir
    os.environ["PATH"] = scripts_dir + os.pathsep + os.environ.get("PATH", "")

    in_venv = sys.prefix != sys.base_prefix
    if not in_venv:
        py_bin = os.path.join(scripts_dir, "python.exe" if sys.platform == "win32" else "python")
        print("[*] Relaunching setup inside virtual environment...")
        subprocess.run([py_bin] + sys.argv)
        sys.exit(0)


def detect_gpu_arch():
    """Detect Compute Capability using nvidia-smi or default to native target."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], 
            text=True
        ).strip()
        if out:
            # Format CC (e.g. '8.6' -> '86')
            arch = out.split('\n')[0].replace('.', '').strip()
            return arch
    except Exception:
        pass
    
    # Fallback to sm_86 if detection fails
    return "86"

def detect_cuda_major_version():
    try:
        out = subprocess.check_output(["nvcc", "--version"], text=True)
        match = re.search(r"release (\d+)\.", out)
        if match:
            return int(match.group(1))
    except Exception:
        pass
    return 12  # Standard assumption if nvcc binary isn't in PATH

def get_safe_cuda_archs(cuda_major):
    """Filter CUDA architectures for modern stack (minimum sm_75)."""
    # Modern SerenityPC requires sm_75 (Turing) or higher
    return "75;80;86;89;90"

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

def install_llama_cpp_self_correcting():
    """Multi-pass resilient build loop for llama-cpp-python."""
    gpu_arch = detect_gpu_arch()
    cuda_major = detect_cuda_major_version()
    safe_archs = get_safe_cuda_archs(cuda_major)
    
    # Pre-install build tools
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "scikit-build-core", "cmake", "ninja", "pyproject_metadata", "pathspec"])

    # Filter architecture compatibility matrix (minimum sm_75 for modern stack)
    if int(gpu_arch) < 75:
        print(f"[!] Warning: Modern stack requires sm_75+. Overriding target sm_{gpu_arch} to sm_86.")
        gpu_arch = "86"

    src_dir = os.path.abspath("llama-cpp-python-src")
    cmake_check = os.path.join(src_dir, "vendor", "llama.cpp", "CMakeLists.txt")
    
    if os.path.exists(src_dir) and os.path.exists(cmake_check):
        pkg_target = src_dir
        extra_args = ["--no-build-isolation"]
    else:
        if os.path.exists(src_dir):
            print(f"[!] Warning: Local '{src_dir}' is missing submodules (vendor/llama.cpp/CMakeLists.txt).")
            print("    Falling back to PyPI llama-cpp-python>=0.3.26. Run 'git submodule update --init --recursive' to fix.")
        pkg_target = "llama-cpp-python>=0.3.26"
        extra_args = []

    base_env, cuda_dir = configure_cuda_toolchain()
    toolkit_arg = f' -DCUDAToolkit_ROOT="{cuda_dir}"' if cuda_dir else ""
    nvcc_bin = os.path.join(cuda_dir, "bin", "nvcc.exe") if cuda_dir else ""
    nvcc_arg = f' -DCMAKE_CUDA_COMPILER="{nvcc_bin}"' if (nvcc_bin and os.path.exists(nvcc_bin)) else ""

    # Clean any corrupted dist-info
    for d in os.listdir(os.path.join(sys.prefix, "Lib", "site-packages")):
        if "llama" in d.lower() and d.endswith(".dist-info"):
            target = os.path.join(sys.prefix, "Lib", "site-packages", d)
            if not os.path.exists(os.path.join(target, "RECORD")):
                shutil.rmtree(target, ignore_errors=True)

    passes = [
        {
            "name": f"Pass 1: Portable GPU Optimization (All Supported Archs: {safe_archs})",
            "env": {
                "CMAKE_ARGS": f"-DGGML_CUDA=on -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_CUDA_ARCHITECTURES={safe_archs}{toolkit_arg}{nvcc_arg} -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler",
                "CUDAFLAGS": "-allow-unsupported-compiler",
                "NVCC_PREPEND_FLAGS": "-allow-unsupported-compiler",
                "FORCE_CMAKE": "1"
            },
            "cmd": [sys.executable, "-m", "pip", "install", pkg_target, "--no-cache-dir", "--no-deps", "--force-reinstall"] + extra_args
        },
        {
            "name": f"Pass 2: Native GPU Optimization (sm_{gpu_arch} + CUDA)",
            "env": {
                "CMAKE_ARGS": f"-DGGML_CUDA=on -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_CUDA_ARCHITECTURES={gpu_arch}{toolkit_arg}{nvcc_arg} -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler",
                "CUDAFLAGS": "-allow-unsupported-compiler",
                "NVCC_PREPEND_FLAGS": "-allow-unsupported-compiler",
                "FORCE_CMAKE": "1"
            },
            "cmd": [sys.executable, "-m", "pip", "install", pkg_target, "--no-cache-dir", "--no-deps", "--force-reinstall"] + extra_args
        },
        {
            "name": "Pass 3: Generic CUDA Fallback",
            "env": {
                "CMAKE_ARGS": f"-DGGML_CUDA=on -DGGML_CUDA_FA_ALL_QUANTS=ON{toolkit_arg}{nvcc_arg} -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler",
                "CUDAFLAGS": "-allow-unsupported-compiler",
                "NVCC_PREPEND_FLAGS": "-allow-unsupported-compiler",
                "FORCE_CMAKE": "1"
            },
            "cmd": [sys.executable, "-m", "pip", "install", pkg_target, "--no-cache-dir", "--no-deps", "--force-reinstall"] + extra_args
        },
        {
            "name": "Pass 4: Pre-compiled Wheels Fallback (PyPI)",
            "env": {},
            "cmd": [sys.executable, "-m", "pip", "install", "llama-cpp-python>=0.3.26", "--prefer-binary"]
        }
    ]

    for p in passes:
        print(f"\n---> Attempting {p['name']}...")
        env = base_env.copy()
        env.update(p["env"])
        
        res = subprocess.run(p["cmd"], env=env)
        if res.returncode == 0:
            print(f"[V] Successfully installed llama-cpp-python via {p['name']}.")
            return True
        else:
            print(f"[!] {p['name']} failed. Rolling over to next compatibility profile...")

    print("[X] Critical: All self-correcting build passes failed.")
    return False

def main():
    ensure_venv()
    print("==================================================")
    print("       SerenityPC Automated System Installer      ")
    print("==================================================")
    
    print("\n[Step 1/2]: Syncing dependencies from requirements.txt...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    print("\n[Step 2/2]: Executing self-correcting engine setup...")
    success = install_llama_cpp_self_correcting()

    if success:
        print("\n[V] SerenityPC Setup finished successfully.")
    else:
        print("\n[X] Setup finished with errors. Review the build output above.")

if __name__ == "__main__":
    main()