import os
import sys
import time
import logging
import platform
import subprocess
import json

# Determine paths
ENGINE_ROOT = os.path.dirname(os.path.abspath(__file__))
LIVE_ROOT = os.path.dirname(ENGINE_ROOT)
BASE_DIR = os.path.dirname(LIVE_ROOT)
LOGS_DIR = os.path.join(LIVE_ROOT, "Logs")
os.makedirs(LOGS_DIR, exist_ok=True)

ERROR_LOG = os.path.join(LOGS_DIR, "error_log.txt")
SYS_LOG = os.path.join(LOGS_DIR, "SysLog.txt")

def log_to_ui(message, is_error=False):
    """Writes to the logs visible in the Serenity UI."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"{timestamp} - [DIAGNOSTIC] - {message}\n"
    
    # Always log to SysLog
    try:
        with open(SYS_LOG, "a", encoding="utf-8") as f:
            f.write(formatted)
    except: pass
    
    # If it's an error, log to Error Log
    if is_error:
        try:
            with open(ERROR_LOG, "a", encoding="utf-8") as f:
                f.write(formatted)
        except: pass
    
    print(formatted.strip())

def check_environment():
    log_to_ui(f"System: {platform.system()} {platform.release()} ({platform.machine()})")
    log_to_ui(f"Python: {sys.version.split()[0]} at {sys.executable}")
    log_to_ui(f"Working Dir: {os.getcwd()}")

def check_torch():
    try:
        import torch
        log_to_ui(f"Torch: {torch.__version__} (Path: {os.path.dirname(torch.__file__)})")
        cuda_avail = torch.cuda.is_available()
        log_to_ui(f"CUDA Available: {cuda_avail}")
        if cuda_avail:
            log_to_ui(f"CUDA Version: {torch.version.cuda}")
            log_to_ui(f"GPU Device(s): {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                log_to_ui(f"  [{i}] {torch.cuda.get_device_name(i)}")
        else:
            log_to_ui("CRITICAL: CUDA NOT DETECTED. Torch will run on CPU.", is_error=True)
    except ImportError:
        log_to_ui("CRITICAL: torch module not found!", is_error=True)
    except Exception as e:
        log_to_ui(f"ERROR checking Torch: {e}", is_error=True)

def check_dependencies():
    deps = ["transformers", "bitsandbytes", "pynvml", "psutil", "fastapi", "uvicorn", "PIL", "pydantic"]
    for dep in deps:
        try:
            mod = __import__(dep)
            version = getattr(mod, "__version__", "unknown")
            log_to_ui(f"Module '{dep}': {version}")
        except ImportError:
            log_to_ui(f"CRITICAL: Module '{dep}' is MISSING!", is_error=True)
        except Exception as e:
            log_to_ui(f"ERROR checking module '{dep}': {e}", is_error=True)

def check_vram():
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_mb = info.free / 1024**2
        total_mb = info.total / 1024**2
        log_to_ui(f"VRAM Status: {free_mb:.0f}MB Free / {total_mb:.0f}MB Total")
        if free_mb < 500:
            log_to_ui("WARNING: Extremely low VRAM detected (<500MB). Engine stability may be compromised.", is_error=True)
    except Exception as e:
        log_to_ui(f"Note: VRAM detailed check skipped ({e})")

def run_full_diagnostic():
    log_to_ui("="*40)
    log_to_ui("STARTING SERENITY DIAGNOSTIC SWEEP")
    log_to_ui("="*40)
    check_environment()
    check_torch()
    check_dependencies()
    check_vram()
    log_to_ui("="*40)
    log_to_ui("DIAGNOSTIC SWEEP COMPLETE")
    log_to_ui("="*40)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Serenity Diagnostic Utility")
    parser.add_argument("--loop", type=int, help="Run in a loop with specified interval in seconds")
    args = parser.parse_args()

    if args.loop:
        log_to_ui(f"Starting diagnostic loop every {args.loop}s...")
        try:
            while True:
                run_full_diagnostic()
                time.sleep(args.loop)
        except KeyboardInterrupt:
            log_to_ui("Diagnostic loop stopped by user.")
    else:
        run_full_diagnostic()
