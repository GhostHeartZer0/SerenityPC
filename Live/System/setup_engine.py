import os
import sys

# --- Cache Localization ---
try:
    system_dir = os.path.dirname(os.path.abspath(__file__))
    live_dir = os.path.dirname(system_dir)
    parent_dir = os.path.dirname(live_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from System import localize_cache
except Exception as e:
    print(f"Cache localization failed to load: {e}")

import subprocess
import json

# --- PYTHON VERSION GUARD ---
# PyTorch and BitsAndBytes currently lack official CUDA wheels for Python 3.14+
if sys.version_info.major == 3 and sys.version_info.minor >= 14:
    print("\n" + "!"*60)
    print("WARNING: EXPERIMENTAL PYTHON VERSION DETECTED (3.14+)")
    print("Official CUDA support (GPU Acceleration) is currently UNSTABLE or")
    print("MISSING for this version of Python on Windows.")
    print("\nTo enable your GPU (RTX 3050), it is HIGHLY RECOMMENDED to use")
    print("Python 3.12 for the Serenity Engine.")
    print("!"*60 + "\n")

def create_file(path, content, overwrite=False):
    """Helper to write files to disk cleanly. Respects solid infrastructure if overwrite is False."""
    if not overwrite and os.path.exists(path):
        print(f"[!] {path} already exists. Skipping to preserve solid infrastructure.")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content.strip() + "\n")
    print(f"[+] Created file: {path}")

def create_shortcut(target_bat, icon_path, shortcut_name):
    """Uses a temporary VBScript to create a proper Windows Shortcut (.lnk) on the Desktop."""
    vbs_path = os.path.join(os.environ["TEMP"], "create_lnk.vbs")
    vbs_content = f"""
Set oWS = WScript.CreateObject("WScript.Shell")
sDesktop = oWS.SpecialFolders("Desktop")
sLinkFile = sDesktop & "\\{shortcut_name}.lnk"
Set oLink = oWS.CreateShortcut(sLinkFile)
oLink.TargetPath = "{target_bat}"
oLink.IconLocation = "{icon_path}"
oLink.WorkingDirectory = "{os.path.dirname(target_bat)}"
oLink.Save
"""
    with open(vbs_path, "w", encoding='utf-8') as f:
        f.write(vbs_content)
    
    subprocess.call(["cscript.exe", "/nologo", vbs_path])
    if os.path.exists(vbs_path):
        os.remove(vbs_path)
    print(f"[+] Shortcut created: {shortcut_name}")

def setup():
    print("=== Initializing Serenity Live Framework ===")
    
    # Path Resolution
    # setup_engine.py is in SerenityPC/Live/System/
    system_dir = os.path.dirname(os.path.abspath(__file__)) # Live/System
    base_dir = os.path.dirname(system_dir)                 # Live
    
    current_python = sys.executable 
    
    # Architecture Paths
    engine_dir = os.path.join(base_dir, "Engine")   
    media_dir = os.path.join(system_dir, "Media") 

    for folder in [engine_dir, system_dir, media_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # ==========================================
    # 1. THE ENGINE SCRIPT (t5_server.py)
    # ==========================================
    server_code = r"""
import torch, os, sys, subprocess, threading, winsound, pyttsx3, signal, psutil, logging, uvicorn
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from contextlib import asynccontextmanager
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [SERENITY ENGINE] - %(message)s")

def system_announce(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 170)
        voices = engine.getProperty('voices')
        if voices: engine.setProperty('voice', voices[0].id) 
        engine.say(text)
        engine.runAndWait()
    except: pass

def signal_ping():
    threading.Thread(target=lambda: winsound.Beep(1200, 150), daemon=True).start()

# Dynamic Base Paths
ENGINE_ROOT = os.path.dirname(os.path.abspath(__file__))
LIVE_ROOT = os.path.dirname(ENGINE_ROOT)

# MODEL TIERS — Light (Quicore), Med (Typicore), Heavy (Intellicore)
CORES = {
    "light":  os.path.join(LIVE_ROOT, "t5gemma-2-270m-270m"),   # Quicore
    "med":    os.path.join(LIVE_ROOT, "t5gemma-2-1b-1b"),       # Typicore
    "heavy":  os.path.join(LIVE_ROOT, "t5gemma-2-4b-4b")        # Intellicore
}

CORE_NAMES = {"light": "Quicore", "med": "Typicore", "heavy": "Intellicore"}

ACTIVE_CORE = os.environ.get("SERENITY_CORE", "med")
LOCAL_WEIGHTS_PATH = CORES.get(ACTIVE_CORE, "")

# REPO Fallbacks
REPO_MAP = {
    "light":  "google/t5gemma-2-270m-270m",   # Quicore
    "med":    "google/t5gemma-2-1b-1b",       # Typicore
    "heavy":  "google/t5gemma-2-4b-4b"        # Intellicore
}
target_repo = REPO_MAP.get(ACTIVE_CORE, "google/t5gemma-2-1b-1b")

# --- GUARDED UI SPAWN ---
# Only spawn the UI if the engine was NOT started BY the UI (prevents double loading)
if os.environ.get("SERENITY_SPAWNED_BY_UI") != "1" and os.environ.get("SERENITY_NO_UI") != "1":
    logging.info("Direct boot detected. Initiating UI spawn...")
    live_script = os.path.join(LIVE_ROOT, "serenity_live.py")
    # Robustly find pythonw.exe
    py_exe = sys.executable
    pyw_exe = py_exe.lower().replace("python.exe", "pythonw.exe") if "python.exe" in py_exe.lower() else py_exe
    # Use CREATE_NO_WINDOW (0x08000000) for a cleaner background launch
    subprocess.Popen([pyw_exe, live_script], cwd=LIVE_ROOT, creationflags=0x08000000)


system_announce(f"Initializing {CORE_NAMES.get(ACTIVE_CORE, ACTIVE_CORE)} engine.")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"
)

# --- MODEL INITIALIZATION & CACHING ---
try:
    logging.info(f"Attempting local offline initialization from {LOCAL_WEIGHTS_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_WEIGHTS_PATH, trust_remote_code=True, local_files_only=True)
    config = AutoConfig.from_pretrained(LOCAL_WEIGHTS_PATH, trust_remote_code=True, local_files_only=True)

    logging.info("Local configuration found. Loading model into VRAM...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        LOCAL_WEIGHTS_PATH, config=config, quantization_config=bnb_config,
        device_map={"": 0}, torch_dtype=torch.bfloat16, trust_remote_code=True, local_files_only=True
    )
except Exception as e:
    logging.warning(f"Local boot failed ({e}). Reverting to HF Hub...")
    
    tokenizer = AutoTokenizer.from_pretrained(target_repo, trust_remote_code=True, token=HF_TOKEN)
    config = AutoConfig.from_pretrained(target_repo, trust_remote_code=True, token=HF_TOKEN)
    
    logging.info(f"Saving architecture files to {LOCAL_WEIGHTS_PATH}...")
    tokenizer.save_pretrained(LOCAL_WEIGHTS_PATH)
    config.save_pretrained(LOCAL_WEIGHTS_PATH)
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        target_repo, config=config, quantization_config=bnb_config,
        device_map={"": 0}, torch_dtype=torch.bfloat16, trust_remote_code=True, token=HF_TOKEN
    )

logging.info("Model loaded successfully.")
system_announce(f"{CORE_NAMES.get(ACTIVE_CORE, ACTIVE_CORE)} is online.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info(f"Server bound to port 8001. {CORE_NAMES.get(ACTIVE_CORE, ACTIVE_CORE)} engine active.")
    yield

app = FastAPI(title="Serenity Engine", lifespan=lifespan)

async def verify_local_key(x_api_key: str = Header(None)):
    if x_api_key != LOCAL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid IPC Key")

class LogRequest(BaseModel):
    text: str
    max_tokens: int = 150
    temperature: float = 0.3

@app.post("/analyze", dependencies=[Depends(verify_local_key)])
async def analyze_log(request: LogRequest):
    signal_ping()
    device = next(model.parameters()).device
    
    # Inject Agent OS Thought Loop Prompt
    if ACTIVE_CORE == "light":
        system_prompt = (
            "You are Serenity, an AI assistant. Direct and friendly. "
            "Respond in JSON format with speech only. "
            "Example: {\"speech\": \"Hello!\"}\n\n"
            f"User: {request.text}\nResponse:"
        )
    else:
        system_prompt = (
            "You are Serenity, an advanced AI. You must respond with a strict JSON dictionary: "
            "'thought' (internal reasoning), 'action' (command like 'none', 'change_persona(X)', 'monitor_system'), "
            "and 'speech' (what you say). "
            "Example: {\"thought\": \"Greeting user.\", \"action\": \"none\", \"speech\": \"Hello!\"}\n\n"
            f"User: {request.text}\nSerenity JSON Response:"
        )
    
    inputs = tokenizer(text=system_prompt, return_tensors="pt").to(device)
    # Tier-specific inference params
    if ACTIVE_CORE == "light":
        gen_kwargs = {"temperature": 0.2, "top_k": 40, "repetition_penalty": 1.2, "no_repeat_ngram_size": 2}
    elif ACTIVE_CORE == "heavy":
        gen_kwargs = {"temperature": 0.5, "top_p": 0.95, "repetition_penalty": 1.5}
    else: # med
        gen_kwargs = {"temperature": 0.3, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.4, "no_repeat_ngram_size": 3}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            do_sample=True,
            **gen_kwargs
        )
    raw = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    # Strip the input prompt from the output if the model echoed it
    input_text = request.text.strip()
    if raw.lower().startswith(input_text.lower()):
        raw = raw[len(input_text):].strip()

    import re as _re
    raw = _re.sub(r'<unused\d+>', '', raw).strip()
    
    # Post-process: truncate at the first repeated sentence
    sentences = [s.strip() for s in _re.split(r'(?<=[.!?])\s+', raw) if s.strip()]
    seen = set()
    cleaned = []
    for s in sentences:
        s_lower = s.lower()
        if s_lower in seen and len(s_lower.split()) > 3: # Only penalize long repeated phrases, not short words like "Yes."
            break  
        seen.add(s_lower)
        cleaned.append(s)
        
    result = " ".join(cleaned).strip()
    # Extract Thought / Action / Speech structure
    result_dict = {
        "thought": "No internal thought generated.",
        "action": "none",
        "speech": result if result else "(No response from engine)"
    }
    
    # Try to parse structure if the model followed instructions
    try:
        import ast
        # Attempt to find dictionary-like structure
        start_idx = result.find('{')
        end_idx = result.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            dict_str = result[start_idx:end_idx+1]
            parsed = ast.literal_eval(dict_str)
            if isinstance(parsed, dict) and 'speech' in parsed:
                result_dict = {
                    "thought": parsed.get("thought", ""),
                    "action": parsed.get("action", "none"),
                    "speech": parsed.get("speech", "")
                }
    except Exception:
        pass # Fallback to raw output as speech if parsing fails

    return result_dict

@app.get("/diagnose", dependencies=[Depends(verify_local_key)])
async def diagnose():
    stats = {"CPU": f"{psutil.cpu_percent()}%", "RAM": f"{psutil.virtual_memory().percent}%"}
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, 0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        stats["GPU Util"] = f"{util.gpu}%"
        stats["GPU Temp"] = f"{temp}°C"
        stats["VRAM"] = f"{mem.used/1024**2:.0f} / {mem.total/1024**2:.0f} MB"
    except: pass
    return stats

@app.post("/shutdown", dependencies=[Depends(verify_local_key)])
async def shutdown_engine():
    system_announce("Backend shutting down.")
    threading.Thread(target=lambda: os.kill(os.getpid(), signal.SIGTERM), daemon=True).start()
    return {"status": "Offline"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="warning")
"""
    create_file(os.path.join(engine_dir, "t5_server.py"), server_code, overwrite=False)

    # ==========================================
    # (Removed dynamic file generation for Live Agent and Tray to prevent overwriting updates)
    # ==========================================

    # ==========================================
    # 4. BATCH & SHORTCUT
    # ==========================================
    bat_path = os.path.join(system_dir, "Launch Serenity.bat")
    python_exe = sys.executable
    pythonw_exe = python_exe.lower().replace("python.exe", "pythonw.exe") if "python.exe" in python_exe.lower() else python_exe
    # Updated template to use relative paths and exit cleaner
    bat_content = f'@echo off\ncd /d "%~dp0.."\nstart "" "{pythonw_exe}" Serenity_Tray.py\nexit'
    create_file(bat_path, bat_content, overwrite=True)
    
    icon_hq = os.path.join(system_dir, "transcendent_serenity_ws_hq.ico")
    if not os.path.exists(icon_hq):
        icon_hq = os.path.join(system_dir, "serenity.ico")

    print("\n--- Initializing Serenity Live Desktop Link ---")
    create_shortcut(bat_path, icon_hq, "Serenity Live")

    # ==========================================
    # 5. DEPENDENCIES
    # ==========================================
    print("\n=== Updating Dependencies ===")
    
    # --- PHASE: CUDA PRE-FLIGHT ---
    print("\n[!] PHASE: CUDA Check & PyTorch Hardware Acceleration...")
    
    # 1. Check for existing functional Torch-CUDA link
    torch_cuda_ok = False
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  > [V] PyTorch {torch.__version__} with CUDA detected. Skipping heavy download.")
            torch_cuda_ok = True
    except: pass

    if not torch_cuda_ok:
        # Python 3.14+ specifically needs the cu130 index for CUDA support
        if sys.version_info.major == 3 and sys.version_info.minor >= 14:
            print("  > Python 3.14 detected. Fetching official CUDA 13.0 binaries...")
            torch_url = "https://download.pytorch.org/whl/cu130/torch-2.10.0%2Bcu130-cp314-cp314-win_amd64.whl"
            subprocess.call([sys.executable, "-m", "pip", "install", torch_url, "--user", "--upgrade", "--force-reinstall"])
        else:
            # Standard fallback for older versions
            subprocess.call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu124", "--upgrade", "--user"])
    else:
        print("  > [V] GPU Acceleration verified for PyTorch.")
    
    # sounddevice is added as a more compatible alternative for modern Python versions
    reqs = ["requests", "psutil", "speechrecognition", "pyttsx3", "pystray", "Pillow", "PyAudio", "accelerate", "bitsandbytes", "fastapi", "uvicorn", "pydantic", "transformers", "sounddevice", "beautifulsoup4", "playwright", "chromadb", "sentence-transformers"]
    
    # We install dependencies individually or in a way that allows us to catch the PyAudio failure on Python 3.14
    core_reqs = [r for r in reqs if r != "PyAudio"]
    subprocess.call([sys.executable, "-m", "pip", "install", "--user", "--upgrade"] + core_reqs)

    print("\n[!] PHASE: Web Automation Driver Setup (Playwright)...")
    subprocess.call([sys.executable, "-m", "playwright", "install", "chromium"])

    print("\n[!] PHASE: Audio Driver Check (PyAudio)...")
    pyaudio_res = subprocess.call([sys.executable, "-m", "pip", "install", "pyaudio"])
    if pyaudio_res != 0:
        print("\n[WARNING] PyAudio failed to install. This is expected on early Python 3.14 builds.")
        print("          'sounddevice' has been installed as a fallback.")
        print("          If audio recording fails, consider using Python 3.12 for maximum compatibility.")

    print("\n=== Setup Engine Complete ===")

if __name__ == "__main__":
    setup()
