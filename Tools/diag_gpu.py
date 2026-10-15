import os
import subprocess
import sys
import glob

def get_cuda_path():
    base_install = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if os.path.exists(base_install):
        versions = glob.glob(os.path.join(base_install, "v*")) 
        if versions:
            v12 = [v for v in versions if os.path.basename(v).startswith("v12")]
            if v12:
                return os.path.join(sorted(v12)[-1], "bin")
            latest = sorted(versions)[-1]
            return os.path.join(latest, "bin")
    return os.environ.get('CUDA_PATH') or os.environ.get('CUDA_HOME')

cuda_bin = get_cuda_path()
print(f"Detected CUDA Bin: {cuda_bin}")

if cuda_bin:
    os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")
    print(f"Added to PATH. Checking nvcc...")
    try:
        res = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        print(res.stdout)
    except Exception as e:
        print(f"Error running nvcc: {e}")

print("\nChecking llama-cpp-python installation status:")
try:
    import llama_cpp
    print(f"llama-cpp-python imported successfully.")
    print(f"GPU offload support: {llama_cpp.llama_supports_gpu_offload()}")
except Exception as e:
    print(f"Error importing llama-cpp-python: {e}")

print("\nChecking if requirements.txt contains llama-cpp-python:")
req_path = os.path.join(os.path.dirname(__file__), "..", "requirements.txt")
if os.path.exists(req_path):
    with open(req_path, "r") as f:
        content = f.read()
        if "llama-cpp-python" in content:
            print("Found llama-cpp-python in requirements.txt - This will clobber GPU build if Step 3 runs!")
