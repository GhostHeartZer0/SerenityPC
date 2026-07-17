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
# For 12700KF: We target the 8 P-Cores (16 threads). 
# For 3050 LP: sm_86 is the architecture.
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
            # Also add the lib/x64 path where cublas lives
            lib_path = os.path.join(os.path.dirname(cuda_bin_path), "lib", "x64")
            if os.path.exists(lib_path) and os.path.isdir(lib_path):
                os.add_dll_directory(lib_path)
            print(f"  > [V] Apex Link Established: {cuda_bin_path}")
        except Exception as e:
            print(f"  > [!] DLL Error: {e}")

def find_vcvars():
    paths = [
        r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
    ]
    for p in paths:
        if os.path.exists(p): return p
    return None

def check_build_environment():
    """Pre-flight check to identify why builds usually fail."""
    print("\n--- [ Pre-flight Build Environment Check ] ---")
    checks = {
        "CMake": shutil.which("cmake"),
        "MSVC (cl.exe)": shutil.which("cl"),
        "CUDA Compiler (nvcc)": shutil.which("nvcc"),
        "Git": shutil.which("git"),
        "Ninja": shutil.which("ninja")
    }
    
    all_passed = True
    for tool, path in checks.items():
        if path:
            print(f"✅ {tool}: Found at {path}")
        else:
            print(f"❌ {tool}: NOT FOUND")
            all_passed = False
    
    if not all_passed:
        print("\n⚠️  WARNING: Some build tools are missing. llama-cpp-python build WILL fail.")
        print("Ensure Visual Studio (C++ Desktop Dev), CMake, and CUDA Toolkit are installed.")
        print("--------------------------------------------------\n")
    else:
        print("✅ All critical build tools detected.\n")
    return all_passed

def install_engine(cuda_path, use_source=False, force_pypi=False):
    """
    Unified installer optimized for i7-12700KF + RTX 3050 LP.
    """
    print(f"\n[ ! ] PHASE: Building Apex Engine (P-Core Optimized)...")
    
    cuda_root = os.path.dirname(cuda_path) if cuda_path else ""
    
    # CRITICAL: We let CMake auto-detect or use native architecture.
    # We also disable AVX512 as it causes stability issues on hybrid architectures.
    cmake_args = (
        "-DGGML_CUDA=on "
        "-DGGML_AVX512=OFF "
        "-DCMAKE_CXX_STANDARD=17 "
        "-DCMAKE_CUDA_STANDARD=17 "
        "-DCMAKE_CXX_FLAGS='/Zc:preprocessor' "
        "-DCMAKE_CUDA_FLAGS='-Xcompiler /Zc:preprocessor'"
    )
    if cuda_root:
        cmake_args += f' -T "cuda={cuda_root}"'

    env = os.environ.copy()
    env["CMAKE_ARGS"] = cmake_args
    env["FORCE_CMAKE"] = "1"
    scripts_dir = os.path.join(os.path.dirname(sys.executable), "Scripts")
    if cuda_path:
        env["CUDA_PATH"] = cuda_root
        env["CUDA_HOME"] = cuda_root
        env["PATH"] = cuda_path + os.pathsep + scripts_dir + os.pathsep + env.get("PATH", "")
    else:
        env["PATH"] = scripts_dir + os.pathsep + env.get("PATH", "")

    vcvars = find_vcvars()
    
    # Command construction
    local_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llama-cpp-python-src")
    if os.path.exists(local_src) and not force_pypi:
        print(f"  > [!] Local custom engine source detected: {local_src}")
        install_cmd = [sys.executable, "-m", "pip", "install", local_src, "--no-cache-dir", "--no-deps", "--upgrade"]
    else:
        target_package = "llama-cpp-python==0.3.26"
        if use_source:
            target_package = "git+https://github.com/abetlen/llama-cpp-python.git"
        install_cmd = [sys.executable, "-m", "pip", "install", target_package, "--no-cache-dir", "--no-deps", "--upgrade"]

    try:
        if vcvars:
            # Wrap in VS context to ensure compiler is found in system temp dir
            import tempfile
            bat_path = os.path.join(tempfile.gettempdir(), "temp_install.bat")
            bat_content = f'call "{vcvars}"\n' + " ".join(install_cmd)
            with open(bat_path, "w") as f: f.write(bat_content)
            subprocess.check_call(["cmd.exe", "/c", bat_path], env=env)
            try:
                os.remove(bat_path)
            except:
                pass
        else:
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
        env["CMAKE_ARGS"] = "-DGGML_CUDA=on" 
        print("🛠️  Setting CMAKE_ARGS: -DGGML_CUDA=on")

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
    print("--- Serenity Apex: Hardware Initialization ---")
    cuda_path = get_cuda_path()
    inject_cuda_path(cuda_path)

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
            status_msg = "✅ Installed and stable (GPU acceleration verified)."
        else:
            status_msg = "⚠️  Installed, but running on CPU (no GPU offloading detected)."
    except Exception as e:
        status_msg = f"❌ Not installed or unstable ({e})."
        
    print(f"Status: {status_msg}")
    print("\nSelect llama-cpp-python Option:")
    print(" [1] Keep existing installation (Do nothing - Recommended if working)")
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

    if not skip_reinstall:
        # Uninstall llama-cpp-python only to prevent caching conflicts
        print("[*] Uninstalling llama-cpp-python (dependencies will not be touched)...")
        subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "llama-cpp-python"])

        # 3. Rebuild for i7-12700KF
        print("\nSelect Build Method:")
        print(" [1] Fast Build (Recommended)")
        print(" [2] Source Build (If Offload Toggle is broken)")
        build_choice = input("Choice [1/2] (default: 1): ").strip()
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

    # 4. Finalizing Web & Audio Dependencies
    print("\n[STEP 4]: Finalizing Environment...")
    if os.path.exists(REQUIREMENTS_FILE):
        print("  > Updating pip dependencies...")
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



