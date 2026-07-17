import torch
import sys
import os

def run_diag():
    print(f"Python Version: {sys.version}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    try:
        import bitsandbytes
        print(f"BitsAndBytes Version: {getattr(bitsandbytes, '__version__', 'unknown')}")
        from bitsandbytes.cuda_setup.main import get_instance
        print(f"BNB CUDA Setup success: {get_instance().is_cuda_available()}")
        print(f"BNB CUDA Path: {get_instance().cuda_path}")
    except Exception as e:
        print(f"BitsAndBytes error: {e}")

    try:
        from transformers import BitsAndBytesConfig
        print("Transformers BitsAndBytesConfig available")
    except ImportError:
        print("Transformers BitsAndBytesConfig NOT available")

if __name__ == "__main__":
    run_diag()
