import sys
import subprocess
import os
import time
import traceback
import ctypes
import glob

# --- Configuration ---
REQUIREMENTS_FILE = "requirements.txt"
CUDA_BIN_PATH = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
# --- End Configuration ---

def inject_cuda_path():
    """Forces Python 3.11 to see the CUDA 12.4 DLLs specifically."""
    if os.path.exists(CUDA_BIN_PATH):
        if hasattr(os, 'add_dll_directory'):
            try:
                os.add_dll_directory(CUDA_BIN_PATH)
                print(f"  > [V] System link established: {CUDA_BIN_PATH}")
            except Exception as e:
                print(f"  > [!] DLL Linking Error: {e}")
        os.environ["PATH"] = CUDA_BIN_PATH + os.pathsep + os.environ.get("PATH", "")
    else:
        print(f"  > [X] CRITICAL: Could not find CUDA 12.4 at {CUDA_BIN_PATH}")

def check_cuda_dlls():
    """Diagnostic check for the specific files required by the RTX 3050 engine."""
    print("\n--- Diagnostic: Checking Windows for CUDA 12.4 Core ---")
    inject_cuda_path()
    
    # These are the specific files llama-cpp-python looks for
    target_dlls = ["cublas64_12.dll", "cudart64_12.dll", "nvcuda.dll"]
    missing = []
    
    for dll_name in target_dlls:
        try:
            ctypes.WinDLL(dll_name)
            print(f"  > [V] Verified: {dll_name}")
        except OSError:
            print(f"  > [X] MISSING: {dll_name}")
            missing.append(dll_name)
            
    if missing:
        print("\n[!] WARNING: Python still cannot see your CUDA 12.4 files.")
        print("Ensure you have the 'CUDA Toolkit 12.4' installed correctly.")
        return False
    return True

def find_vcvars():
    """Locates the hidden Visual Studio developer environment script."""
    paths = [
        r"C:\Program Files\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\*\*\VC\Auxiliary\Build\vcvars64.bat"
    ]
    for p in paths:
        matches = glob.glob(p)
        if matches:
            return sorted(matches)[-1]  # Get the latest version installed
    return None

def check_engine_health():
    """Final hardware-level verification."""
    try:
        import llama_cpp
        
        is_gpu = False
        if hasattr(llama_cpp, 'llama_supports_gpu_offload'):
            is_gpu = llama_cpp.llama_supports_gpu_offload()
        elif hasattr(llama_cpp, 'llama_system_info'):
            info = llama_cpp.llama_system_info()
            info_str = info.decode('utf-8') if isinstance(info, bytes) else str(info)
            is_gpu = "CUDA = 1" in info_str
            
        return is_gpu
    except Exception:
        return False

def install_engine():
    print(f"\n[ ! ] PHASE: Compiling GPU Engine from Source (Gemma-3 Support)...")
    print("      >>> Grab a coffee. This will take 10 to 15 minutes. <<<")
    print("      >>> Do NOT close this window until it finishes.     <<<")
    
    build_env = os.environ.copy()
    
    # 1. Install Ninja to bypass Visual Studio's broken MSBuild CUDA targets
    try:
        print("      > Bootstrapping Ninja build system...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ninja", "cmake", "scikit-build-core"], env=build_env, stdout=subprocess.DEVNULL)
    except Exception as e:
        print(f"      > [!] Warning: Could not bootstrap Ninja: {e}")

    # 2. Locate the Visual Studio C++ Compiler Environment
    vcvars = find_vcvars()
    if not vcvars:
        print("\n[X] CRITICAL: Could not find Visual Studio C++ build environment (vcvars64.bat).")
        print("    Please open Visual Studio Installer and ensure 'Desktop development with C++' is checked.")
        return False
    print(f"      > Found MSVC Environment: {os.path.dirname(vcvars)}")

    # 3. Force CMake to use Ninja, point it to CUDA, and inject the fixes
    build_env["CMAKE_GENERATOR"] = "Ninja"
    build_env["CUDACXX"] = os.path.join(CUDA_BIN_PATH, "nvcc.exe")
    build_env["CUDA_PATH"] = os.path.dirname(CUDA_BIN_PATH)
    build_env["CUDA_HOME"] = os.path.dirname(CUDA_BIN_PATH)
    build_env["FORCE_CMAKE"] = "1"
    build_env["CMAKE_ARGS"] = "-DGGML_CUDA=ON -DLLAMA_CURL=OFF"
    
    # Added '-v' (verbose) so pip streams the compiler logs instead of looking frozen!
    pip_cmd = f'"{sys.executable}" -m pip install llama-cpp-python>=0.3.16 --no-cache-dir --force-reinstall --upgrade -v'
    full_cmd = f'call "{vcvars}" && {pip_cmd}'

    try:
        # shell=True is required so the environment variables transfer to the pip command
        subprocess.check_call(full_cmd, shell=True, env=build_env)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[X] ERROR: Compilation failed. Code {e.returncode}")
        print("\n[!] Check the red text above to see why the C++ compiler failed.")
        return False

def main():
    print("--- Serenity AI: Source Build Setup (Gemma-3 Target) ---")
    
    # 1. Path Diagnostics
    if not check_cuda_dlls():
        print("\n[ACTION REQUIRED]: DLLs not found. Installation may result in CUDA=0.")
        input("Press Enter to attempt compilation anyway...")

    # 2. Engine Refresh (Compile)
    print("\n[STEP 2]: Compiling AI Engine...")
    if install_engine():
        time.sleep(2)
        if check_engine_health():
            print("\n[V] SUCCESS: Serenity is hardware-accelerated (CUDA 12.4).")
            print("             Gemma-3 Architecture is ready.")
        else:
            print("\n[X] FAILED: Engine compiled, but CUDA reports 0.")
            print("    Python is still being blocked from accessing Program Files.")
    else:
        print("\n[X] Installation aborted.")
    
    # 3. Dependencies
    if os.path.exists(REQUIREMENTS_FILE):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_FILE, "--quiet"])

    print("\nSetup Process Finished.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n=======================================================")
        print("FATAL ERROR: Script Crashed.")
        traceback.print_exc()
        print("=======================================================\n")
    finally:
        input("\nPress Enter to exit...")