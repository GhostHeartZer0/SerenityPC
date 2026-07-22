import os
import tempfile

# --- Localize Temp and Cache Paths ---
_curr = os.path.abspath(__file__)
while True:
    _parent = os.path.dirname(_curr)
    if _parent == _curr:
        _workspace = os.path.dirname(os.path.abspath(__file__))
        break
    if os.path.exists(os.path.join(_parent, "serenity_resources.py")) or os.path.exists(os.path.join(_parent, ".git")):
        _workspace = _parent
        break
    _curr = _parent

_cache_dir = os.path.join(_workspace, ".serenity_cache")
_temp_dir = os.path.join(_cache_dir, "temp")
_cuda_dir = os.path.join(_cache_dir, "cuda")
_triton_dir = os.path.join(_cache_dir, "triton")
_torch_ext_dir = os.path.join(_cache_dir, "torch_extensions")
_pip_dir = os.path.join(_cache_dir, "pip")

for _d in [_temp_dir, _cuda_dir, _triton_dir, _torch_ext_dir, _pip_dir]:
    os.makedirs(_d, exist_ok=True)

os.environ["TEMP"] = _temp_dir
os.environ["TMP"] = _temp_dir
os.environ["TMPDIR"] = _temp_dir
os.environ["CUDA_CACHE_PATH"] = _cuda_dir
os.environ["TRITON_CACHE_DIR"] = _triton_dir
os.environ["TORCH_EXTENSIONS_DIR"] = _torch_ext_dir
os.environ["PIP_CACHE_DIR"] = _pip_dir
tempfile.tempdir = _temp_dir

import torch, sys, json, subprocess, threading, signal, psutil, logging, uvicorn, time
try:
    import winsound
except ImportError:
    winsound = None
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from transformers import AutoProcessor, AutoModelForSeq2SeqLM, AutoConfig, TextIteratorStreamer, AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Any, List, Dict, Union
from turboquant import TurboQuantCache
from PIL import Image
import base64
import io

ENGINE_ROOT = os.path.dirname(os.path.abspath(__file__))
LIVE_ROOT = os.path.dirname(ENGINE_ROOT)
BASE_DIR = os.path.dirname(LIVE_ROOT)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from System.kv_manager import KVManager
from serenity_resources import TRI_ATTENTION_ENABLED, TRI_ATTENTION_BUDGET

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [SERENITY ENGINE] - %(message)s")
# Add FileHandler to SysLog.txt
try:
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "SysLog.txt")
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setFormatter(logging.Formatter("%(asctime)s - [SERENITY ENGINE] - %(message)s"))
    logging.getLogger().addHandler(fh)
except Exception: pass

# --- PRE-FLIGHT DIAGNOSTICS ---
def run_preflight():
    try:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Logs")
        error_log = os.path.join(log_dir, "error_log.txt")
        
        info = [
            f"Python: {sys.version.split()[0]}",
            f"Torch: {torch.__version__}",
            f"CUDA: {torch.cuda.is_available()} (Version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'})",
        ]
        
        # Check critical modules
        for mod_name in ["transformers", "bitsandbytes"]:
            try:
                mod = __import__(mod_name)
                info.append(f"{mod_name}: {getattr(mod, '__version__', 'unknown')}")
            except ImportError:
                msg = f"CRITICAL MISSING MODULE: {mod_name}"
                logging.error(msg)
                with open(error_log, "a") as f: f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - [FATAL] - {msg}\n")
        
        logging.info(f"PRE-FLIGHT: {' | '.join(info)}")
    except Exception as e:
        logging.warning(f"Pre-flight check failed: {e}")

run_preflight()

# Capture tqdm progress bars from stderr and write them to SysLog.txt
class TqdmToSysLog:
    def __init__(self, original_stderr, log_path):
        self.original_stderr = original_stderr
        self.log_path = log_path
    
    def write(self, buf):
        self.original_stderr.write(buf)
        if "Loading weights:" in buf or "%|" in buf:
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    clean_buf = buf.replace('\r', '\n')
                    f.write(clean_buf)
            except: pass
            
    def flush(self):
        self.original_stderr.flush()

try:
    _log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Logs")
    _sys_log = os.path.join(_log_dir, "SysLog.txt")
    sys.stderr = TqdmToSysLog(sys.stderr, _sys_log)
except Exception: pass

torch.set_num_threads(6) # Optimized to prevent context-switching bottleneck

class GamingModeMonitor:
    """Background monitor for high-load games and VRAM pressure."""
    def __init__(self):
        self.gaming_mode_active = False
        self.monitored_processes = ["TheDivision2.exe", "Division2.exe"]
        self.lock = threading.Lock()
        self.vram_threshold = 0.90 # 90%
        
    def check(self):
        while True:
            active = False
            # Check Processes
            for proc in psutil.process_iter(['name']):
                if proc.info['name'] in self.monitored_processes:
                    active = True
                    break
            
            # Check VRAM
            if torch.cuda.is_available():
                try:
                    import pynvml as nvidia_ml_py
                    nvidia_ml_py.nvmlInit()
                    handle = nvidia_ml_py.nvmlDeviceGetHandleByIndex(0)
                    info = nvidia_ml_py.nvmlDeviceGetMemoryInfo(handle)
                    if (info.used / info.total) > self.vram_threshold:
                        active = True
                except: pass
            
            with self.lock:
                if active != self.gaming_mode_active:
                    self.gaming_mode_active = active
                    logging.info(f"[GAMING MODE] State changed to: {active}")
            
            time.sleep(10)

GAMING_MONITOR = GamingModeMonitor()
threading.Thread(target=GAMING_MONITOR.check, daemon=True).start()

def system_announce(text):
    if not pyttsx3: return
    try:
        # Cross-platform driver selection
        driver = 'sapi5' if os.name == 'nt' else None
        engine = pyttsx3.init(driverName=driver)
        engine.setProperty('rate', 170)
        voices = engine.getProperty('voices')
        if voices: engine.setProperty('voice', voices[0].id) 
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        logging.debug(f"TTS failed: {e}")

LAST_ACTIVITY_TIME = time.time()

def signal_ping():
    global LAST_ACTIVITY_TIME
    LAST_ACTIVITY_TIME = time.time()
    if winsound and os.name == 'nt':
        try:
             if hasattr(winsound, 'PlaySound'):
                  threading.Thread(target=lambda: winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS), daemon=True).start()
        except: pass
    else:
        # Linux/MacOS fallback: Terminal bell
        sys.stdout.write('\a')
        sys.stdout.flush()

# Dynamic Base Paths (defined at top of file)

# MODEL TIERS — Light (Quick-core), Med (Cormal), Heavy (Intelli-Core)
CORES = {
    "light":   os.path.join(LIVE_ROOT, "t5gemma-2-270m-270m"),
    "med":     os.path.join(LIVE_ROOT, "t5gemma-2-1b-1b"),
    "heavy":    os.path.join(LIVE_ROOT, "t5gemma-2-4b-4b"),
    "qwen27b": r"S:\LLM\Qwen3.6-27B",
    "qwen35b": r"S:\LLM\Qwen3.6-35B"
}

CORE_NAMES = {"light": "Quick-core", "med": "Cormal", "heavy": "Intelli-Core"}

ACTIVE_CORE = os.environ.get("SERENITY_CORE", "med")
LOCAL_WEIGHTS_PATH = CORES.get(ACTIVE_CORE, "")

# Live Context Sizes
LIVE_CONTEXT_SIZES = {
    "light": 8192,
    "med": 16384,
    "heavy": 32768
}
current_ctx = LIVE_CONTEXT_SIZES.get(ACTIVE_CORE, 16384)

kv_manager = None
if TRI_ATTENTION_ENABLED:
    kv_manager = KVManager(max_context_tokens=current_ctx, prune_ratio=TRI_ATTENTION_BUDGET)
    logging.info(f"TriAttention KV Manager Initialized (Budget: {current_ctx} tokens)")

# --- GUARDED UI SPAWN ---
# Only spawn the UI if the engine was NOT started BY the UI (prevents double loading)
if os.environ.get("SERENITY_SPAWNED_BY_UI") != "1" and os.environ.get("SERENITY_NO_UI") != "1":
    logging.info("Direct boot detected. Initiating UI spawn...")
    live_script = os.path.join(LIVE_ROOT, "serenity_live.py")
    # Robustly find pythonw.exe
    py_exe = sys.executable
    pyw_exe = py_exe.lower().replace("python.exe", "pythonw.exe") if "python.exe" in py_exe.lower() else py_exe
    # UI should NOT use CREATE_NO_WINDOW if we want to see it
    creation_flags = 0x08000000 if os.name == 'nt' else 0
    subprocess.Popen([pyw_exe, live_script], cwd=LIVE_ROOT, creationflags=creation_flags)




system_announce(f"Initializing {CORE_NAMES.get(ACTIVE_CORE, ACTIVE_CORE)} engine.")

# BitsAndBytesConfig removed for Hardware Isolation (CPU Inference)

# --- HELPER FUNCTIONS FOR MODEL INIT ---
from transformers import BitsAndBytesConfig

def apply_stability_config(config):
    """Defensive stability lock — config.json already sets these, but this guarantees it."""
    config.dropout_rate = 0.0
    config.attention_dropout = 0.0
    config.classifier_dropout_rate = 0.0
    config.image_token_index = 256001
    config.tie_word_embeddings = True
    if hasattr(config, "encoder"):
        config.encoder.image_token_index = 256001
        if hasattr(config.encoder, "text_config"):
            config.encoder.text_config.dropout_rate = 0.0
            config.encoder.text_config.attention_dropout = 0.0
        if hasattr(config.encoder, "vision_config"):
            config.encoder.vision_config.attention_dropout = 0.0
    if hasattr(config, "decoder"):
        config.decoder.dropout_rate = 0.0
        config.decoder.attention_dropout = 0.0
    return config

def get_bnb_config(core):
    """Build BitsAndBytesConfig for the active core tier."""

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["q_proj", "k_proj", "v_proj", "o_proj", "out_proj", "lm_head"]
    )

# --- MODEL INITIALIZATION & CACHING ---
n_gpu_layers = 0
try:
    params_file = os.path.join(LIVE_ROOT, "System", "params.json")
    if os.path.exists(params_file):
        with open(params_file, "r") as f:
            params = json.load(f)
            # Tier-specific GPU layer overrides
            if ACTIVE_CORE == "heavy":
                n_gpu_layers = params.get("heavy_layers", params.get("n_gpu_layers", 24))
            elif ACTIVE_CORE == "med":
                n_gpu_layers = params.get("med_layers", params.get("n_gpu_layers", 0))
            elif ACTIVE_CORE == "light":
                n_gpu_layers = params.get("light_layers", 0)

            else:
                n_gpu_layers = params.get("n_gpu_layers", 0)
except Exception as e:
    logging.warning(f"Could not read params.json: {e}")

# REPO Fallbacks (Moved to allow trouble_model override)
REPO_MAP = {
    "light":  "google/t5gemma-2-270m-270m",
    "med":    "google/t5gemma-2-1b-1b",
    "heavy":  "google/t5gemma-2-4b-4b",

    "qwen27b": "Qwen/Qwen3.6-27B-Chat",
    "qwen35b": "Qwen/Qwen3.6-35B-Chat"
}
target_repo = REPO_MAP.get(ACTIVE_CORE, "google/t5gemma-2-1b-1b")



bnb_config = get_bnb_config(ACTIVE_CORE)
model_dtype = torch.float16 if ACTIVE_CORE in ["qwen27b", "qwen35b"] else torch.bfloat16

model = None
processor = None
_tokenizer = None
MODEL_LOADED = False
MODEL_LOADING_ERROR = None

def load_model_background():
    global model, processor, _tokenizer, MODEL_LOADED, MODEL_LOADING_ERROR
    try:
        is_causal = ACTIVE_CORE in ["cg2b", "cg7b", "qwen27b", "qwen35b"] or any(x in target_repo.lower() for x in ["codegemma", "qwen"])
        
        logging.info(f"Attempting local offline initialization from {LOCAL_WEIGHTS_PATH}...")
        if is_causal:
            processor = AutoTokenizer.from_pretrained(LOCAL_WEIGHTS_PATH, trust_remote_code=True, local_files_only=True)
        else:
            processor = AutoProcessor.from_pretrained(LOCAL_WEIGHTS_PATH, trust_remote_code=True, local_files_only=True)
            
        config = AutoConfig.from_pretrained(LOCAL_WEIGHTS_PATH, trust_remote_code=True, local_files_only=True)
        apply_stability_config(config)

        device_map = {"": 0} if torch.cuda.is_available() else "cpu"
        logging.info(f"Local config found. Loading model with 4-bit quantization (device_map={device_map})...")
        if is_causal:
            model = AutoModelForCausalLM.from_pretrained(
                LOCAL_WEIGHTS_PATH, 
                quantization_config=bnb_config if torch.cuda.is_available() else None,
                device_map=device_map,
                dtype=model_dtype, 
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                LOCAL_WEIGHTS_PATH, 
                quantization_config=bnb_config if torch.cuda.is_available() else None,
                device_map=device_map,
                dtype=model_dtype, 
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        model.config.use_cache = True

    except Exception as e:
        # If it's a device mapping error, don't try to redownload, it's a code issue.
        if "meta device" in str(e) or "device_map" in str(e):
            logging.error(f"CRITICAL HARDWARE ROUTING ERROR: {e}")
            logging.error("The model keys do not match the expected architecture. Verify device_map.")
            MODEL_LOADING_ERROR = f"CRITICAL HARDWARE ROUTING ERROR: {e}"
            return
            
        logging.warning(f"Local weights boot failed ({e}). Attempting HF Hub fallback...")
        try:
            if is_causal:
                processor = AutoTokenizer.from_pretrained(target_repo, trust_remote_code=True, token=HF_TOKEN)
            else:
                processor = AutoProcessor.from_pretrained(target_repo, trust_remote_code=True, token=HF_TOKEN)
                
            config = AutoConfig.from_pretrained(target_repo, trust_remote_code=True, token=HF_TOKEN)
            apply_stability_config(config)

            logging.info(f"Saving architecture files to {LOCAL_WEIGHTS_PATH}...")
            processor.save_pretrained(LOCAL_WEIGHTS_PATH)
            config.save_pretrained(LOCAL_WEIGHTS_PATH)
            
            device_map = {"": 0} if torch.cuda.is_available() else "cpu"
            logging.info(f"Loading model from Hub with 4-bit quantization on {device_map}...")
            if is_causal:
                model = AutoModelForCausalLM.from_pretrained(
                    target_repo,
                    quantization_config=bnb_config if torch.cuda.is_available() else None,
                    device_map=device_map,
                    dtype=model_dtype,
                    trust_remote_code=True,
                    token=HF_TOKEN
                )
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    target_repo,
                    quantization_config=bnb_config if torch.cuda.is_available() else None,
                    device_map=device_map,
                    dtype=model_dtype,
                    trust_remote_code=True,
                    token=HF_TOKEN
                )
            model.config.use_cache = True
        except Exception as e2:
            logging.error(f"HF Hub fallback failed: {e2}")
            MODEL_LOADING_ERROR = f"HF Hub fallback failed: {e2}"
            return

    # Global tokenizer alias — works for both AutoProcessor (has .tokenizer) and AutoTokenizer (IS the tokenizer)
    _tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor

    logging.info("Model loaded successfully.")
    system_announce(f"{CORE_NAMES.get(ACTIVE_CORE, ACTIVE_CORE)} is online.")
    MODEL_LOADED = True

    if torch.cuda.is_available():
        try:
            def run_unified_benchmark(use_tq=False):
                torch.cuda.empty_cache()
                
                # Initializing the specific cache type
                cache = TurboQuantCache(bits=3) if use_tq else None
                
                device = next(model.parameters()).device
                prompt = "Hello, how are you today?"
                if is_causal:
                    inputs = _tokenizer(prompt, return_tensors="pt").to(device)
                else:
                    inputs = processor(text=prompt, return_tensors="pt").to(device)
                
                start_time = time.time()
                with torch.no_grad():
                    # Running the model to generate output tokens
                    outputs = model.generate(**inputs, max_new_tokens=100, past_key_values=cache)
                
                duration = time.time() - start_time
                
                # Isolating the Cache Memory
                # Instead of measuring the whole 2GB model, we measure the generated Cache size
                # For a 1.1B model: [Layers: 22, Heads: 32, Head_Dim: 64]
                num_tokens = outputs.shape[1]
                elements = 22 * 32 * 64 * num_tokens * 2 # Key + Value
                
                if use_tq:
                    mem_mb = (elements * 3) / (8 * 1024 * 1024) # 3-bit calculation
                else:
                    mem_mb = (elements * 16) / (8 * 1024 * 1024) # 16-bit calculation
                    
                return duration, mem_mb

            logging.info("[BENCHMARK] Running Unified TurboQuant vs FP16 Cache Benchmark...")
            base_time, base_mem = run_unified_benchmark(use_tq=False)
            tq_time, tq_mem = run_unified_benchmark(use_tq=True)

            print(f"--- THE VERDICT ---")
            print(f"Baseline (FP16) Cache: {base_mem:.2f} MB")
            print(f"TurboQuant (3-bit) Cache: {tq_mem:.2f} MB")
            print(f"Speedup: {base_time / tq_time:.2f}x")
            print(f"Memory Saved: {base_mem - tq_mem:.2f} MB")
            
            logging.info(f"[BENCHMARK] Verdict - FP16 Cache: {base_mem:.2f} MB | TurboQuant Cache: {tq_mem:.2f} MB | Speedup: {base_time / tq_time:.2f}x | Memory Saved: {base_mem - tq_mem:.2f} MB")
        except Exception as be:
            logging.warning(f"[BENCHMARK] Unified Benchmark failed to run: {be}")


# Start background loading thread
threading.Thread(target=load_model_background, daemon=True).start()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info(f"Server bound to port 8001. {CORE_NAMES.get(ACTIVE_CORE, ACTIVE_CORE)} engine active.")
    
    def autoswap_watcher():
        global LAST_ACTIVITY_TIME
        LAST_ACTIVITY_TIME = time.time()
        while True:
            time.sleep(5)
            # If we are in Supervisor mode (heavy/Live), fallback to FIM (cg2b) after 60s
            # Note: t5_server is usually used for workers, but if it runs a heavy core it can also supervise.
            is_supervisor = ACTIVE_CORE in ["heavy", "Live", "qwen35b"]
            if is_supervisor and time.time() - LAST_ACTIVITY_TIME > 60:
                logging.info("[SYSTEM] 60s inactivity on Supervisor. Autoswapping to CodeGemma FIM fallback...")
                params_path = os.path.join(LIVE_ROOT, "System", "params.json")
                if os.path.exists(params_path):
                    try:
                        with open(params_path, 'r') as f: p = json.load(f)
                        p["active_core"] = "cg2b"
                        p["persona_level"] = "7C"
                        with open(params_path, 'w') as f: json.dump(p, f, indent=4)
                        
                        # Trigger process restart to load FIM
                        os.kill(os.getpid(), signal.SIGTERM)
                    except Exception as e:
                        logging.error(f"Autoswap failed: {e}")

    # Start the watcher as a daemon thread
    threading.Thread(target=autoswap_watcher, daemon=True).start()
    
    yield

app = FastAPI(title="Serenity Engine", lifespan=lifespan)

from fastapi.responses import JSONResponse
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logging.error(f"[FATAL UNCAUGHT ERROR] on {request.url}: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": str(exc)})

async def verify_local_key(x_api_key: str = Header(None), authorization: Optional[str] = Header(None)):
    key = x_api_key
    if not key and authorization and authorization.startswith("Bearer "):
        key = authorization.split(" ")[1]
    
    if key != LOCAL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid IPC Key")

class LogRequest(BaseModel):
    text: str
    max_tokens: int = 150
    temperature: float = 0.3
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.15
    image_b64: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None

def resolve_kv_cache():
    try:
        params_file = os.path.join(LIVE_ROOT, "System", "params.json")
        global_kv = "Auto"
        if os.path.exists(params_file):
            with open(params_file, "r") as f:
                params = json.load(f)
                global_kv = params.get("global_kv_cache", "Auto")
        
        logging.info(f"[KV RESOLVER] Selected global_kv_cache: {global_kv}")
        
        gl = global_kv.lower()
        if "tq1" in gl or "turboquant 1" in gl:
            logging.info("[KV RESOLVER] Engaging TurboQuant 1-bit cache.")
            return TurboQuantCache(bits=1)
        elif "tq2" in gl or "turboquant 2" in gl:
            logging.info("[KV RESOLVER] Engaging TurboQuant 2-bit cache.")
            return TurboQuantCache(bits=2)
        elif "tq3" in gl or "turboquant 3" in gl:
            logging.info("[KV RESOLVER] Engaging TurboQuant 3-bit cache.")
            return TurboQuantCache(bits=3)
        elif "tq4" in gl or "turboquant 4" in gl:
            logging.info("[KV RESOLVER] Engaging TurboQuant 4-bit cache.")
            return TurboQuantCache(bits=4)
        
        # If not one of the standard keep tiers, map it to TurboQuant 4-bit as requested
        if gl not in ["f32", "f16", "q4_0", "q8_0", "auto"]:
            logging.info(f"[KV RESOLVER] Mapping custom option '{global_kv}' to TurboQuant 4-bit.")
            return TurboQuantCache(bits=4)
            
    except Exception as e:
        logging.error(f"[KV RESOLVER] Failed to resolve KV cache: {e}")
    
    return None

@app.post("/stream", dependencies=[Depends(verify_local_key)])
async def stream_log(request: LogRequest):
    signal_ping()
    device = next(model.parameters()).device
    
    # Process history
    history_str = ""
    if request.history:
        processed_history = request.history
        if kv_manager:
            try:
                processed_history = kv_manager.enforce_kv_budget(request.history)
            except Exception as e:
                logging.warning(f"KV Manager failed: {e}")
                
        for msg in processed_history:
            if msg.get("role") == "user":
                content = msg.get("content", "").replace("User: ", "").strip()
                history_str += f"User: {content}\n"
            elif msg.get("role") == "assistant":
                content = msg.get("content", "").strip()
                # Encapsulate older formatted history inside the new XML structure
                if "<speech>" not in content:
                    content = f"<speech>{content}</speech>"
                history_str += f"{content}\n\n"

    # Fully Transitioned to rigid XML Prompt Engineering structure
    if ACTIVE_CORE == "light":
        system_prompt = (
            "System: You are Serenity. You MUST use XML tags for your output.\n"
            "<thought>your internal reasoning</thought>\n<action>none</action>\n<speech>what you say</speech>\n"
            f"{history_str}"
            f"User: {request.text}\n"
            "<thought>"
        )
    else: 
        system_prompt = (
            "System: You are Serenity, an AI companion. You MUST structure your response EXACTLY with these XML tags:\n"
            "<thought>your reasoning here</thought>\n"
            "<action>none | chrome_search | play_media | vision_search</action>\n"
            "<speech>your response here</speech>\n\n"
            f"{history_str}"
            f"User: {request.text}\n"
            "<thought>"
        )
    
    if request.image_b64:
        try:
            b64_str: str = str(request.image_b64)
            image_bytes = base64.b64decode(b64_str)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            # T5-Gemma-2 uses <start_of_image> boi_token per README
            multimodal_prompt = f"<start_of_image> {system_prompt}"
            inputs = processor(images=image, text=multimodal_prompt, return_tensors="pt").to(device)
            logging.info("[VISION] Processing multimodal payload.")
        except Exception as ve:
            logging.error(f"[VISION ERROR] Could not process image: {ve}")
            raise HTTPException(status_code=400, detail=f"Vision processing failed: {ve}")
    else:
        inputs = processor(text=system_prompt, return_tensors="pt").to(device)

    # Intent-Based Parameter Refinement (Improving TTFT/Speed)
    is_analytical = False
    analytical_keywords = ["analyze", "summarize", "research", "compare", "plan", "explain", "why", "how", "describe", "break down"]
    vision_keywords = ["look", "see", "screen", "check", "view"]
    
    query_text = request.text.lower()
    if any(kw in query_text for kw in analytical_keywords) or request.image_b64 or any(kw in query_text for kw in vision_keywords):
        is_analytical = True
    
    # Conversational (short) check
    is_conversational = len(query_text.split()) < 10 and not is_analytical

    gen_kwargs = {}
    if ACTIVE_CORE == "light":
        # Prioritize speed for light core
        if is_conversational:
            gen_kwargs = {"temperature": 0.15, "top_p": 0.9, "top_k": 32, "repetition_penalty": 1.1}
        else:
            gen_kwargs = {"temperature": 0.25, "top_p": 0.95, "top_k": 64, "repetition_penalty": 1.15}
    elif ACTIVE_CORE == "heavy":
        # Heavy core is usually for analytical work
        if is_analytical:
            gen_kwargs = {"temperature": 0.45, "top_p": 0.95, "top_k": 96, "repetition_penalty": 1.3}
        else:
            gen_kwargs = {"temperature": 0.35, "top_p": 0.9, "top_k": 64, "repetition_penalty": 1.25}
    elif ACTIVE_CORE == "trouble":
        gen_kwargs = {"temperature": 0.3, "top_p": 0.95, "top_k": 64, "repetition_penalty": 1.20}
    else: # med
        if is_conversational:
            gen_kwargs = {"temperature": 0.2, "top_p": 0.9, "top_k": 40, "repetition_penalty": 1.1}
        else:
            gen_kwargs = {"temperature": 0.35, "top_p": 0.95, "top_k": 64, "repetition_penalty": 1.15}   
    
    request_dict = request.model_dump(exclude={"text", "max_tokens", "image_b64", "history"}) if hasattr(request, "model_dump") else request.dict(exclude={"text", "max_tokens", "image_b64", "history"})
    actual_gen_kwargs = {
        "do_sample": True, 
        "bos_token_id": 2, 
        "eos_token_id": 1, 
        "pad_token_id": 0, 
        "use_cache": True,
        **gen_kwargs
    }
    for k, v in request_dict.items():
        if v is not None:
            actual_gen_kwargs[k] = v

    # Resolve and engage TurboQuant Cache if configured
    kv_cache = resolve_kv_cache()
    if kv_cache is not None:
        actual_gen_kwargs["past_key_values"] = kv_cache
        actual_gen_kwargs["use_cache"] = True

    # Gaming Mode: Aggressive KV Compression (H6/Q4)

    if GAMING_MONITOR.gaming_mode_active:
        logging.info("[SYSTEM] Gaming Mode: Limiting context to preserve VRAM.")
        # Instead of quantized cache which might hang, aggressively clamp max_new_tokens
        request.max_tokens = min(request.max_tokens, 100)
        # Ensure torch doesn't keep fragmented memory
        torch.cuda.empty_cache()

    def handle_directive(raw_text):
        """Processes 'directive' key for autonomous reboots."""
        try:
            import json as _json, re as _re, sys as _sys
            match = _re.search(r'\{.*\}', raw_text, _re.DOTALL)
            if match:
                data = _json.loads(match.group(0))
                directive = data.get("directive")
                if directive and isinstance(directive, dict) and directive.get("action") == "reboot":
                    logging.info(f"[DIRECTIVE] Executing autonomous reboot: {directive}")
                    # Update params.json
                    params_path = os.path.join(LIVE_ROOT, "System", "params.json")
                    if os.path.exists(params_path):
                        with open(params_path, 'r') as f: params = _json.load(f)
                        if directive.get("core"): params["active_core"] = directive.get("core")
                        if directive.get("temperature"): 
                            pfx = "heavy_" if params["active_core"] == "heavy" else "med_"
                            params[f"{pfx}temp"] = str(directive.get("temperature"))
                        with open(params_path, 'w') as f: _json.dump(params, f, indent=4)
                    
                    # OS Execv Reboot
                    def _delayed_reboot():
                        time.sleep(1)
                        logging.info("[SYSTEM] Engine restarting via os.execv...")
                        os.execv(_sys.executable, [_sys.executable] + _sys.argv)
                    threading.Thread(target=_delayed_reboot, daemon=True).start()
        except: pass

    streamer = TextIteratorStreamer(_tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=request.max_tokens,
        stop_strings=["User:", "\nUser:", " User:", "<eos>", "<|endoftext|>"],
        tokenizer=_tokenizer,
        **actual_gen_kwargs,
        streamer=streamer
    )

    def generate_and_stream():
        try:
            with torch.no_grad():
                model.generate(**generation_kwargs)
        except Exception as e:
            logging.error(f"[FATAL GENERATION ERROR] {e}", exc_info=True)
            # Ensure the streamer doesn't block indefinitely
            streamer.text_queue.put(f"[ENGINE ERROR]: {str(e)}")
            streamer.text_queue.put(streamer.stop_signal)

    threading.Thread(target=generate_and_stream).start()

    def generate():
        full_text = []
        for new_text in streamer:
            if new_text:
                full_text.append(new_text)
                yield new_text + "\n"
        # Check for directive post-generation
        handle_directive("".join(full_text))
    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/analyze", dependencies=[Depends(verify_local_key)])
async def analyze_log(request: LogRequest):
    signal_ping()
    
    if not MODEL_LOADED:
        if MODEL_LOADING_ERROR:
            raise HTTPException(status_code=500, detail=f"Model loading failed: {MODEL_LOADING_ERROR}")
        raise HTTPException(
            status_code=503,
            detail="Model is currently loading into VRAM. Please retry in a few seconds."
        )
        
    device = next(model.parameters()).device
    
    # Process history
    history_str = ""
    if request.history:
        processed_history = request.history
        if kv_manager:
            try:
                processed_history = kv_manager.enforce_kv_budget(request.history)
            except Exception as e:
                logging.warning(f"KV Manager failed: {e}")
                
        for msg in processed_history:
            if msg.get("role") == "user":
                content = msg.get("content", "").replace("User: ", "").strip()
                history_str += f"User: {content}\n"
            elif msg.get("role") == "assistant":
                content = msg.get("content", "").strip()
                # Encapsulate older formatted history inside the new XML structure
                if "<speech>" not in content:
                    content = f"<speech>{content}</speech>"
                history_str += f"{content}\n\n"

    # t5gemma-2 Optimized /analyze Prompt
    if ACTIVE_CORE == "light":
        system_prompt = (
            "System: You are Serenity. Respond with plaintext tags.\n"
            "<thought>...</thought>\n<action>none</action>\n<speech>...</speech>\n"
            f"{history_str}"
            f"User: {request.text}\n"
            "<thought>"
        )
    else:
        system_prompt = (
            "System: Serenity is a proactive AI. Respond ONLY using this format: \n"
            "<thought>reasoning</thought>\n<action>none | chrome_search | play_media | vision_search</action>\n<speech>response</speech>\n\n"
            f"{history_str}"
            f"User: {request.text}\n"
            "<thought>"
        )
    
    if request.image_b64:
        try:
            b64_str: str = str(request.image_b64)
            image_bytes = base64.b64decode(b64_str)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            # T5-Gemma-2 uses <start_of_image> boi_token per README
            multimodal_prompt = f"<start_of_image> {system_prompt}"
            inputs = processor(images=image, text=multimodal_prompt, return_tensors="pt").to(device)
            logging.info("[VISION] Processing multimodal payload.")
        except Exception as ve:
            logging.error(f"[VISION ERROR] Could not process image: {ve}")
            raise HTTPException(status_code=400, detail=f"Vision processing failed: {ve}")
    else:
        inputs = processor(text=system_prompt, return_tensors="pt").to(device)

    # Intent-Based Parameter Refinement (Improving TTFT/Speed)
    is_analytical = False
    analytical_keywords = ["analyze", "summarize", "research", "compare", "plan", "explain", "why", "how", "describe", "break down"]
    vision_keywords = ["look", "see", "screen", "check", "view"]
    
    query_text = request.text.lower()
    if any(kw in query_text for kw in analytical_keywords) or request.image_b64 or any(kw in query_text for kw in vision_keywords):
        is_analytical = True
    
    # Conversational (short) check
    is_conversational = len(query_text.split()) < 10 and not is_analytical

    gen_kwargs = {}
    if ACTIVE_CORE == "light":
        if is_conversational:
            gen_kwargs = {"temperature": 0.15, "top_k": 32, "repetition_penalty": 1.1}
        else:
            gen_kwargs = {"temperature": 0.25, "top_k": 64, "repetition_penalty": 1.15}
    elif ACTIVE_CORE == "heavy":
        if is_analytical:
            gen_kwargs = {"temperature": 0.45, "top_p": 0.95, "top_k": 96, "repetition_penalty": 1.3}
        else:
            gen_kwargs = {"temperature": 0.35, "top_p": 0.9, "top_k": 64, "repetition_penalty": 1.25}
    else: # med
        if is_conversational:
            gen_kwargs = {"temperature": 0.2, "top_p": 0.9, "top_k": 40, "repetition_penalty": 1.1}
        else:
            gen_kwargs = {"temperature": 0.35, "top_p": 0.95, "top_k": 64, "repetition_penalty": 1.15}

    request_dict = request.model_dump(exclude={"text", "max_tokens", "image_b64"}) if hasattr(request, "model_dump") else request.dict(exclude={"text", "max_tokens", "image_b64"})
    actual_gen_kwargs = {
        "do_sample": True,
        "bos_token_id": model.config.decoder.bos_token_id if hasattr(model.config, "decoder") else 2,
        "eos_token_id": model.config.decoder.eos_token_id if hasattr(model.config, "decoder") else 1,
        "pad_token_id": model.config.decoder.pad_token_id if hasattr(model.config, "decoder") else 0,
        **gen_kwargs
    }
    for k, v in request_dict.items():
        if v is not None:
            actual_gen_kwargs[k] = v

    # Resolve and engage TurboQuant Cache if configured
    kv_cache = resolve_kv_cache()
    if kv_cache is not None:
        actual_gen_kwargs["past_key_values"] = kv_cache
        actual_gen_kwargs["use_cache"] = True
        actual_gen_kwargs.pop("cache_config", None)

    # Gaming Mode: Aggressive KV Compression (H6/Q4)

    if GAMING_MONITOR.gaming_mode_active:
        logging.info("[SYSTEM] Gaming Mode Active: Engaging KV-Cache Compression.")
        try:
            from transformers import QuantizedCacheConfig
            actual_gen_kwargs["cache_config"] = QuantizedCacheConfig(backend="bitsandbytes", nbits=4)
        except:
             request.max_tokens = min(request.max_tokens, 256)

    # Construct sequence_bias for /analyze — use plaintext tag format (not quoted JSON)
    sequence_bias = {}
    try:
        bias_words = {"thought:": 2.5, "Serenity:": 2.5, "action:": 1.5, "directive:": 1.5}
        for word, bias in bias_words.items():
            tokens = _tokenizer.encode(word, add_special_tokens=False)
            if tokens: sequence_bias[tuple(tokens)] = bias
        actual_gen_kwargs["sequence_bias"] = sequence_bias
    except Exception: pass

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            stop_strings=["User:", "\nUser:", " User:", "\n\n"],
            tokenizer=_tokenizer,
            **actual_gen_kwargs
        )
    
    raw = _tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    # If we forced a start bracket, ignore it
    if raw.startswith("{"): raw = raw[1:]
    
    import re as _re
    result = "thought: " + raw.strip()
    
    # Simple regex parsing for the /analyze endpoint so it still returns the JSON format to the UI
    # Fuzzier regex parsing to handle Gemma 2 / T5 artifacts and whitespace variations
    t_match = _re.search(r'thought:?\s*(.*?)(?=\n?\s*action:|\n?\s*directive:|\n?\s*Serenity:|$)', result, _re.DOTALL | _re.IGNORECASE)
    a_match = _re.search(r'action:?\s*(.*?)(?=\n?\s*directive:|\n?\s*Serenity:|$)', result, _re.DOTALL | _re.IGNORECASE)
    d_match = _re.search(r'directive:?\s*(.*?)(?=\n?\s*Serenity:|$)', result, _re.DOTALL | _re.IGNORECASE)
    s_match = _re.search(r'Serenity:?\s*(.*)', result, _re.DOTALL | _re.IGNORECASE)
    
    thought = t_match.group(1).strip() if t_match else "None"
    action = a_match.group(1).strip() if a_match else "none"
    directive = None
    if d_match:
        d_str = d_match.group(1).strip()
        if d_str and d_str.lower() not in ["null", "none"]:
            try: directive = json.loads(d_str)
            except: directive = d_str
    speech = s_match.group(1).strip() if s_match else result

    return {"thought": thought, "action": action, "directive": directive, "speech": speech}

@app.get("/diagnose", dependencies=[Depends(verify_local_key)])
async def diagnose():
    stats = {
        "CPU": f"{psutil.cpu_percent()}%", 
        "RAM": f"{psutil.virtual_memory().percent}%",
        "Torch": torch.__version__,
        "CUDA": torch.cuda.is_available()
    }
    try:
        import transformers, bitsandbytes
        stats["Transformers"] = transformers.__version__
        stats["BnB"] = bitsandbytes.__version__
    except: pass
    
    try:
        import pynvml as nvidia_ml_py
        nvidia_ml_py.nvmlInit()
        handle = nvidia_ml_py.nvmlDeviceGetHandleByIndex(0)
        util = nvidia_ml_py.nvmlDeviceGetUtilizationRates(handle)
        temp = nvidia_ml_py.nvmlDeviceGetTemperature(handle, 0)
        mem = nvidia_ml_py.nvmlDeviceGetMemoryInfo(handle)
        stats["GPU Util"] = f"{util.gpu}%"
        stats["GPU Temp"] = f"{temp}°C"
        stats["VRAM"] = f"{mem.used/1024**2:.0f} / {mem.total/1024**2:.0f} MB"
    except: pass

    # Add Config Version Locking
    try:
        params_path = os.path.join(LIVE_ROOT, "System", "params.json")
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                p = json.load(f)
                stats["config_version"] = p.get("config_version", 0)
    except:
        stats["config_version"] = -1

    return stats

@app.post("/shutdown", dependencies=[Depends(verify_local_key)])
async def shutdown_engine():
    system_announce("Backend shutting down.")
    def trigger_shutdown():
        time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=trigger_shutdown, daemon=True).start()
    return {"status": "Offline"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="warning")
