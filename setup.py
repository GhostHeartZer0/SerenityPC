import sys
import subprocess
import os
import platform

# Localize TEMP, TMP, and compiler/cache directories to bypass Windows security restrictions
_base_dir = os.path.dirname(os.path.abspath(__file__))
_local_temp = os.path.join(_base_dir, ".temp")
_local_cuda = os.path.join(_base_dir, ".cuda_cache")
_local_triton = os.path.join(_base_dir, ".triton_cache")
_local_torch = os.path.join(_base_dir, ".torch_extensions")

for _d in [_local_temp, _local_cuda, _local_triton, _local_torch]:
    os.makedirs(_d, exist_ok=True)

os.environ["TEMP"] = _local_temp
os.environ["TMP"] = _local_temp
os.environ["CUDA_CACHE_PATH"] = _local_cuda
os.environ["TRITON_CACHE_DIR"] = _local_triton
os.environ["TORCH_EXTENSIONS_DIR"] = _local_torch

import re

# --- Configuration ---
REQUIREMENTS_FILE = "requirements.txt"

def get_cuda_version():
    """Detects CUDA version for hardware matching."""
    try:
        output = subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT).decode()
        match = re.search(r"CUDA Version:\s+(\d+\.\d+)", output)
        if match:
            # We target CUDA 12.x for modern RTX cards (3050, etc)
            return "cu124" 
    except:
        return None

def run_with_stream(cmd_list):
    """Streams console output to prevent the 'buffer hang' during long installs."""
    process = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    return process.returncode == 0

def install_llm_backend():
    """Forces the high-performance GPU engine installation."""
    print("\n[ ! ] Initializing High-Performance GPU Engine...")
    system = platform.system()
    cuda_ver = get_cuda_version()
    
    # BLEEDING EDGE WHEELS (Supports Gemma-3 and RTX 30-series)
    wheel_url = "https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/"
    
    # The command that fixed it for you, now automated:
    gpu_cmd = [
        sys.executable, "-m", "pip", "install", 
        "llama-cpp-python>=0.3.16", 
        "--force-reinstall", 
        "--no-cache-dir", 
        "--prefer-binary", 
        "--extra-index-url", wheel_url
    ]

    if system == "Windows" and cuda_ver:
        print(f"  > NVIDIA GPU detected ({cuda_ver}). Mapping CUDA kernels...")
        if run_with_stream(gpu_cmd):
            # Also install the portable DLLs so users don't need the 3GB Toolkit
            print("  > Syncing portable CUDA runtime...")
            run_with_stream([sys.executable, "-m", "pip", "install", "nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12", "--quiet"])
            print("  > [OK] GPU Access Enabled.")
            return
    
    print("  > No NVIDIA GPU detected or install failed. Falling back to CPU...")
    run_with_stream([sys.executable, "-m", "pip", "install", "llama-cpp-python", "--force-reinstall"])

def main():
    print("--- Serenity AI: Universal Setup (Gemma-3 Optimized) ---")
    
    # 1. Install the specialized GPU engine first
    install_llm_backend()
    
    # 2. Install everything else in the requirements
    if os.path.exists(REQUIREMENTS_FILE):
        print("\n[ ! ] Finalizing dependencies...")
        run_with_stream([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_FILE])
    
    print("\n--- Setup Complete ---")
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()