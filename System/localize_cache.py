import os
import sys

def localize():
    # Find workspace root: walk up until we find main.py or TODO.txt
    current = os.path.abspath(__file__)
    workspace_root = None
    while True:
        parent = os.path.dirname(current)
        if os.path.exists(os.path.join(parent, "main.py")) or os.path.exists(os.path.join(parent, "TODO.txt")):
            workspace_root = parent
            break
        if parent == current:
            # Fallback to current directory or script entry directory
            workspace_root = os.path.dirname(os.path.abspath(sys.argv[0]))
            break
        current = parent

    cache_root = os.path.join(workspace_root, "Cache")
    temp_dir = os.path.join(cache_root, "Temp")
    cuda_dir = os.path.join(cache_root, "Cuda")
    triton_dir = os.path.join(cache_root, "Triton")
    pip_dir = os.path.join(cache_root, "Pip")
    torch_dir = os.path.join(cache_root, "Torch")
    torch_ext_dir = os.path.join(torch_dir, "extensions")

    for path in [temp_dir, cuda_dir, triton_dir, pip_dir, torch_dir, torch_ext_dir]:
        os.makedirs(path, exist_ok=True)

    # Set environment variables for all subprocess backends
    os.environ["TEMP"] = temp_dir
    os.environ["TMP"] = temp_dir
    os.environ["TMPDIR"] = temp_dir
    os.environ["CUDA_CACHE_PATH"] = cuda_dir
    os.environ["TRITON_CACHE_DIR"] = triton_dir
    os.environ["PIP_CACHE_DIR"] = pip_dir
    os.environ["TORCH_HOME"] = torch_dir
    os.environ["TORCH_EXTENSIONS_DIR"] = torch_ext_dir

    # Force tempfile to use the new localized temp directory
    try:
        import tempfile
        tempfile.tempdir = temp_dir
    except:
        pass

# Execute on import to ensure early initialization
localize()
