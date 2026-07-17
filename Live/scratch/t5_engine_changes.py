import torch, os, sys, json, subprocess, threading, signal, psutil, logging, uvicorn, time
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

try:
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "SysLog.txt")
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setFormatter(logging.Formatter("%(asctime)s - [SERENITY ENGINE] - %(message)s"))
    logging.getLogger().addHandler(fh)
except Exception: pass

def run_preflight():
    try:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Logs")
        error_log = os.path.join(log_dir, "error_log.txt")
        
        info = [
            f"Python: {sys.version.split()[0]}",
            f"Torch: {torch.__version__}",
            f"CUDA: {torch.cuda.is_available()} (Version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'})",
        ]
        
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

class TqdmToSysLog:
    def __init__(self, original_stderr, log_path):
        self.original_stderr = original_stderr
        self.log_path = log_path
    
    def write(self, buf):
        self.original_stderr.write(buf)
        if isinstance(buf, str) and ("Loading weights:" in buf or "%|" in buf):
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

torch.set_num_threads(6)

class GamingModeMonitor:
    def __init__(self):
        self.gaming_mode_active = False
        self.monitored_processes = ["TheDivision2.exe", "Division2.exe"]
        self.lock = threading.Lock()
        self.vram_threshold = 0.90
        
    def check(self):
        while True:
            active = False
            for proc in psutil.process_iter(['name']):
                if proc.info['name'] in self.monitored_processes:
                    active = True
                    break
            
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

***REMOVED***
***REMOVED***

def system_announce(text):
    if not pyttsx3: return
    try:
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
             if hasattr(winsound, 'Beep'):
                  threading.Thread(target=lambda: winsound.Beep(1200, 150), daemon=True).start()
        except: pass
    else:
        sys.stdout.write('\a')
        sys.stdout.flush()

CORES = {
    "light":   os.path.join(LIVE_ROOT, "t5gemma-2-270m-270m"),
    "med":     os.path.join(LIVE_ROOT, "t5gemma-2-1b-1b"),
    "heavy":    os.path.join(LIVE_ROOT, "t5gemma-2-4b-4b"),
    "trouble":  os.path.join(os.path.dirname(LIVE_ROOT), "Troubleshooter"),
    "cg2b":    os.path.join(os.path.dirname(LIVE_ROOT), "Troubleshooter", "codegemma-2b"),
    "cg7b":    os.path.join(os.path.dirname(LIVE_ROOT), "Troubleshooter", "codegemma-7b-it"),
    "qwen27b": r"S:\LLM\Qwen3.6-27B",
    "qwen35b": r"S:\LLM\Qwen3.6-35B"
}

CORE_NAMES = {"light": "Quick-core", "med": "Cormal", "heavy": "Intelli-Core", "trouble": "Troubleshooter"}

ACTIVE_CORE = os.environ.get("SERENITY_CORE", "med")
LOCAL_WEIGHTS_PATH = CORES.get(ACTIVE_CORE, "")

LIVE_CONTEXT_SIZES = {
    "light": 8192,
    "med": 16384,
    "heavy": 32768,
    "trouble": 16384
}
current_ctx = LIVE_CONTEXT_SIZES.get(ACTIVE_CORE, 16384)

kv_manager = None
if TRI_ATTENTION_ENABLED:
    kv_manager = KVManager(max_context_tokens=current_ctx, prune_ratio=TRI_ATTENTION_BUDGET)
    logging.info(f"TriAttention KV Manager Initialized (Budget: {current_ctx} tokens)")

if os.environ.get("SERENITY_SPAWNED_BY_UI") != "1" and os.environ.get("SERENITY_NO_UI") != "1":
    logging.info("Direct boot detected. Initiating UI spawn...")
    live_script = os.path.join(LIVE_ROOT, "serenity_live.py")
    py_exe = sys.executable
    pyw_exe = py_exe.lower().replace("python.exe", "pythonw.exe") if "python.exe" in py_exe.lower() else py_exe
    creation_flags = 0x08000000 if os.name == 'nt' else 0
    subprocess.Popen([pyw_exe, live_script], cwd=LIVE_ROOT, creationflags=creation_flags)

system_announce(f"Initializing {CORE_NAMES.get(ACTIVE_CORE, ACTIVE_CORE)} engine.")

from transformers import BitsAndBytesConfig

def apply_stability_config(config):
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
    if core == "trouble":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["gate_proj", "up_proj", "lm_head"]
        )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["q_proj", "k_proj", "v_proj", "o_proj", "out_proj", "lm_head"]
    )

n_gpu_layers = 0
try:
    params_file = os.path.join(LIVE_ROOT, "System", "params.json")
    if os.path.exists(params_file):
        with open(params_file, "r") as f:
            params = json.load(f)
            if ACTIVE_CORE == "heavy":
                n_gpu_layers = params.get("heavy_layers", params.get("n_gpu_layers", 24))
            elif ACTIVE_CORE == "med":
                n_gpu_layers = params.get("med_layers", params.get("n_gpu_layers", 0))
            elif ACTIVE_CORE == "light":
                n_gpu_layers = params.get("light_layers", 0)
            elif ACTIVE_CORE == "trouble":
                n_gpu_layers = params.get("trouble_layers", 0)
            else:
                n_gpu_layers = params.get("n_gpu_layers", 0)
except Exception as e:
    logging.warning(f"Could not read params.json: {e}")

REPO_MAP = {
    "light":  "google/t5gemma-2-270m-270m",
    "med":    "google/t5gemma-2-1b-1b",
    "heavy":  "google/t5gemma-2-4b-4b",
    "trouble": params.get("trouble_model", "google/t5gemma-2-1b-1b") if 'params' in locals() else "google/t5gemma-2-1b-1b",
    "cg2b":    "google/codegemma-2b",
    "cg7b":    "google/codegemma-7b-it",
    "qwen27b": "Qwen/Qwen3.6-27B-Chat",
    "qwen35b": "Qwen/Qwen3.6-35B-Chat"
}
target_repo = REPO_MAP.get(ACTIVE_CORE, "google/t5gemma-2-1b-1b")

if ACTIVE_CORE == "trouble":
    if "codegemma-2b" in target_repo:
        LOCAL_WEIGHTS_PATH = os.path.join(os.path.dirname(LIVE_ROOT), "Troubleshooter", "codegemma-2b")
    elif "codegemma-7b-it" in target_repo:
        LOCAL_WEIGHTS_PATH = os.path.join(os.path.dirname(LIVE_ROOT), "Troubleshooter", "codegemma-7b-it")

bnb_config = get_bnb_config(ACTIVE_CORE)
model_dtype = torch.float16 if ACTIVE_CORE in ["trouble", "qwen27b", "qwen35b"] else torch.bfloat16

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
            attn_implementation="sdpa" if torch.cuda.is_available() else "eager",
            dtype=model_dtype, 
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            LOCAL_WEIGHTS_PATH, 
            quantization_config=bnb_config if torch.cuda.is_available() else None,
            device_map=device_map,
            attn_implementation="sdpa" if torch.cuda.is_available() else "eager",
            dtype=model_dtype, 
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    model.config.use_cache = True

except Exception as e:
    if "meta device" in str(e) or "device_map" in str(e):
        logging.error(f"CRITICAL HARDWARE ROUTING ERROR: {e}")
        logging.error("The model keys do not match the expected architecture. Verify device_map.")
        raise e
        
    logging.warning(f"Local weights boot failed ({e}). Attempting HF Hub fallback...")
    
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
            attn_implementation="sdpa" if torch.cuda.is_available() else "eager",
            dtype=model_dtype,
            trust_remote_code=True,
            token=HF_TOKEN
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            target_repo,
            quantization_config=bnb_config if torch.cuda.is_available() else None,
            device_map=device_map,
            attn_implementation="sdpa" if torch.cuda.is_available() else "eager",
            dtype=model_dtype,
            trust_remote_code=True,
            token=HF_TOKEN
        )
    model.config.use_cache = True

_tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor

logging.info("Model loaded successfully.")
system_announce(f"{CORE_NAMES.get(ACTIVE_CORE, ACTIVE_CORE)} is online.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info(f"Server bound to port 8001. {CORE_NAMES.get(ACTIVE_CORE, ACTIVE_CORE)} engine active.")
    
    def autoswap_watcher():
        global LAST_ACTIVITY_TIME
        LAST_ACTIVITY_TIME = time.time()
        while True:
            time.sleep(5)
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
                        
                        os.kill(os.getpid(), signal.SIGTERM)
                    except Exception as e:
                        logging.error(f"Autoswap failed: {e}")

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


@app.post("/stream", dependencies=[Depends(verify_local_key)])
async def stream_log(request: LogRequest):
    signal_ping()
    device = next(model.parameters()).device
    
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
            multimodal_prompt = f"<image> {system_prompt}"
            inputs = processor(images=image, text=multimodal_prompt, return_tensors="pt").to(device)
            logging.info("[VISION] Processing multimodal payload.")
        except Exception as ve:
            logging.error(f"[VISION ERROR] Could not process image: {ve}")
            raise HTTPException(status_code=400, detail=f"Vision processing failed: {ve}")
    else:
        inputs = processor(text=system_prompt, return_tensors="pt").to(device)

    gen_kwargs = {}
    if ACTIVE_CORE == "light":
        gen_kwargs = {"temperature": 0.2, "top_p": 0.95, "top_k": 64, "repetition_penalty": 1.15}
    elif ACTIVE_CORE == "heavy":
        gen_kwargs = {"temperature": 0.4, "top_p": 0.95, "top_k": 64, "repetition_penalty": 1.25}
    elif ACTIVE_CORE == "trouble":
        gen_kwargs = {"temperature": 0.3, "top_p": 0.95, "top_k": 64, "repetition_penalty": 1.20}
    else: 
        gen_kwargs = {"temperature": 0.3, "top_p": 0.95, "top_k": 64, "repetition_penalty": 1.15}   
    
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

    if GAMING_MONITOR.gaming_mode_active:
        logging.info("[SYSTEM] Gaming Mode: Limiting context to preserve VRAM.")
        request.max_tokens = min(request.max_tokens, 100)
        torch.cuda.empty_cache()

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
            streamer.text_queue.put(f"[ENGINE ERROR]: {str(e)}")
            streamer.text_queue.put(streamer.stop_signal)

    threading.Thread(target=generate_and_stream).start()

    def generate():
        for new_text in streamer:
            if new_text:
                yield new_text + "\n"
    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/analyze", dependencies=[Depends(verify_local_key)])
async def analyze_log(request: LogRequest):
    signal_ping()
    device = next(model.parameters()).device
    
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
                if "<speech>" not in content:
                    content = f"<speech>{content}</speech>"
                history_str += f"{content}\n\n"

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
            multimodal_prompt = f"<image> {system_prompt}"
            inputs = processor(images=image, text=multimodal_prompt, return_tensors="pt").to(device)
            logging.info("[VISION] Processing multimodal payload.")
        except Exception as ve:
            logging.error(f"[VISION ERROR] Could not process image: {ve}")
            raise HTTPException(status_code=400, detail=f"Vision processing failed: {ve}")
    else:
        inputs = processor(text=system_prompt, return_tensors="pt").to(device)

    gen_kwargs = {}
    if ACTIVE_CORE == "light":
        gen_kwargs = {"temperature": 0.2, "top_k": 40, "repetition_penalty": 1.1}
    elif ACTIVE_CORE == "heavy":
        gen_kwargs = {"temperature": 0.4, "top_p": 0.9, "repetition_penalty": 1.2}
    else: 
        gen_kwargs = {"temperature": 0.35, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.15}

    request_dict = request.model_dump(exclude={"text", "max_tokens", "image_b64"}) if hasattr(request, "model_dump") else request.dict(exclude={"text", "max_tokens", "image_b64"})
    actual_gen_kwargs = {
        "do_sample": True,
        "bos_token_id": model.config.decoder.bos_token_id if hasattr(model.config, "decoder") else 2,
        "eos_token_id": model.config.decoder.eos_token_id if hasattr(model.config, "decoder") else 1,
        "pad_token_id": model.config.decoder.pad_token_id if hasattr(model.config, "decoder") else 0,
        "use_cache": True,
        **gen_kwargs
    }
    for k, v in request_dict.items():
        if v is not None:
            actual_gen_kwargs[k] = v

    if GAMING_MONITOR.gaming_mode_active:
        logging.info("[SYSTEM] Gaming Mode Active: Engaging VRAM Guard.")
        try:
            request.max_tokens = min(request.max_tokens, 256)
        except: pass

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            stop_strings=["User:", "\nUser:", " User:", "\n\n"],
            tokenizer=_tokenizer,
            **actual_gen_kwargs
        )
    
    raw = _tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    result = "<thought>" + raw.strip()
    
    import re as _re
    t_m = _re.search(r'<thought>(.*?)</thought>', result, _re.DOTALL | _re.IGNORECASE)
    a_m = _re.search(r'<action>(.*?)</action>', result, _re.DOTALL | _re.IGNORECASE)
    d_m = _re.search(r'<directive>(.*?)</directive>', result, _re.DOTALL | _re.IGNORECASE)
    s_m = _re.search(r'<speech>(.*?)</speech>', result, _re.DOTALL | _re.IGNORECASE)

    thought = t_m.group(1).strip() if t_m else "None"
    action = a_m.group(1).strip() if a_m else "none"
    directive = None
    if d_m:
        d_str = d_m.group(1).strip()
        if d_str and d_str.lower() not in ["null", "none"]:
            try: directive = json.loads(d_str)
            except: directive = d_str
    speech = s_m.group(1).strip() if s_m else result.replace("<thought>", "")

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
    return stats

class OpenAICompletionRequest(BaseModel):
    model: Optional[str] = "serenity-troubleshooter"
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.3
    top_p: Optional[float] = 0.95
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None

class OpenAIChatMessage(BaseModel):
    role: str
    content: str

class OpenAIChatRequest(BaseModel):
    model: Optional[str] = "serenity-troubleshooter"
    messages: List[OpenAIChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.3
    top_p: Optional[float] = 0.95
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None

@app.post("/v1/completions")
async def openai_completions(request: OpenAICompletionRequest, x_vs_debugger: Optional[str] = Header(None)):
    signal_ping()
    
    requested_model = request.model.lower() if request.model else ""
    
    if "gemma-4" in requested_model or "architect" in requested_model:
        if ACTIVE_CORE.lower() != "live":
            logging.info(f"[BRIDGE] VS Request for {requested_model} detected. Swapping to Architect (Live Core)...")
            params_path = os.path.join(LIVE_ROOT, "System", "params.json")
            if os.path.exists(params_path):
                 with open(params_path, 'r') as f: p = json.load(f)
                 p["active_core"] = "Live"
                 p["persona_level"] = "7"
                 with open(params_path, 'w') as f: json.dump(p, f, indent=4)
            threading.Thread(target=lambda: os.kill(os.getpid(), signal.SIGTERM), daemon=True).start()
            raise HTTPException(status_code=503, detail="Optimizing for Architect mode. Please retry.")

    if "qwen" in requested_model:
        if ACTIVE_CORE != "qwen35b":
            logging.info(f"[BRIDGE] VS Request for Qwen detected. Swapping to qwen35b...")
            params_path = os.path.join(LIVE_ROOT, "System", "params.json")
            if os.path.exists(params_path):
                 with open(params_path, 'r') as f: p = json.load(f)
                 p["active_core"] = "qwen35b"
                 p["persona_level"] = "7A"
                 with open(params_path, 'w') as f: json.dump(p, f, indent=4)
            threading.Thread(target=lambda: os.kill(os.getpid(), signal.SIGTERM), daemon=True).start()
            raise HTTPException(status_code=503, detail="Optimizing for Qwen35b mode. Please retry.")

    if "codegemma-7b-it" in requested_model or "implementer" in requested_model or "autocomplete" in requested_model:
        target_cg_core = "cg2b" if "2b" in requested_model or "autocomplete" in requested_model else "cg7b"
        persona_lvl = "7C" if target_cg_core == "cg2b" else "7B"
        if ACTIVE_CORE != target_cg_core:
             logging.info(f"[BRIDGE] VS Request for CodeGemma-7b-it detected. Swapping to {target_cg_core} for Implementer mode...")
             params_path = os.path.join(LIVE_ROOT, "System", "params.json")
             if os.path.exists(params_path):
                 with open(params_path, 'r') as f: p = json.load(f)
                 p["active_core"] = target_cg_core
                 p["persona_level"] = persona_lvl
                 with open(params_path, 'w') as f: json.dump(p, f, indent=4)
             threading.Thread(target=lambda: os.kill(os.getpid(), signal.SIGTERM), daemon=True).start()
             raise HTTPException(status_code=503, detail=f"Optimizing for {target_cg_core}. Please retry.")

    device = next(model.parameters()).device
    prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
    
    _tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
    is_causal_model = ACTIVE_CORE in ["cg2b", "cg7b", "qwen27b", "qwen35b"] or any(x in target_repo.lower() for x in ["codegemma", "qwen"])
    
    if is_causal_model:
        inputs = _tokenizer(prompt, return_tensors="pt").to(device)
    else:
        inputs = processor(text=prompt, return_tensors="pt").to(device)
    
    gen_kwargs = {
        "do_sample": True,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_new_tokens": request.max_tokens,
    }
    
    if is_causal_model:
        gen_kwargs["pad_token_id"] = _tokenizer.pad_token_id or _tokenizer.eos_token_id
        gen_kwargs["eos_token_id"] = _tokenizer.eos_token_id
    else:
        gen_kwargs["bos_token_id"] = model.config.decoder.bos_token_id if hasattr(model.config, "decoder") else 2
        gen_kwargs["eos_token_id"] = model.config.decoder.eos_token_id if hasattr(model.config, "decoder") else 1
        gen_kwargs["pad_token_id"] = model.config.decoder.pad_token_id if hasattr(model.config, "decoder") else 0
    
    if request.stop:
        gen_kwargs["stop_strings"] = request.stop if isinstance(request.stop, list) else [request.stop]

    if ACTIVE_CORE in ["cg2b", "cg7b"]:
        try:
            fim_tokens = _tokenizer.convert_tokens_to_ids(['<|fim_prefix|>', '<|fim_middle|>', '<|fim_suffix|>', '<|file_separator|>'])
            terminators = [t for t in fim_tokens if t is not None and t != 0]
            terminators.append(_tokenizer.eos_token_id)
            gen_kwargs["eos_token_id"] = list(set(terminators))
        except: pass

    if request.stream:
        streamer = TextIteratorStreamer(_tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(**inputs, tokenizer=_tokenizer, streamer=streamer, **gen_kwargs)
        threading.Thread(target=model.generate, kwargs=generation_kwargs).start()
        
        def stream_gen():
            for text in streamer:
                yield f"data: {json.dumps({'choices': [{'text': text, 'index': 0, 'finish_reason': None}]})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream_gen(), media_type="text/event-stream")
    else:
        with torch.no_grad():
            outputs = model.generate(**inputs, tokenizer=_tokenizer, **gen_kwargs)
        
        if is_causal_model:
            text = _tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        else:
            text = _tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{"text": text, "index": 0, "finish_reason": "stop"}]
        }

@app.post("/v1/chat/completions")
async def openai_chat_completions(request: OpenAIChatRequest):
    signal_ping()
    prompt = ""
    for msg in request.messages:
        prompt += f"{msg.role.capitalize()}: {msg.content}\n"
    prompt += "Assistant: "
    
    completion_req = OpenAICompletionRequest(
        model=request.model,
        prompt=prompt, max_tokens=request.max_tokens, 
        temperature=request.temperature, top_p=request.top_p, 
        stream=request.stream, stop=request.stop
    )
    return await openai_completions(completion_req)

@app.get("/v1/models")
async def openai_models():
    model_names = ["gemma-4", "gemma-4-26b-a4b", "codegemma", "codegemma-2b", "codegemma-7b-it", "architect", "implementer", "serenity-troubleshooter", "supervisor"]
    
    try:
        f_path = os.path.join(LIVE_ROOT, "System", "foundry_config.json")
        if os.path.exists(f_path):
            with open(f_path, 'r') as f:
                cfg = json.load(f)
                cores = cfg.get("cores", {})
                for k, v in cores.items():
                    if k not in model_names: model_names.append(k)
                    m_id = v.get("model_id")
                    if m_id and m_id not in model_names: model_names.append(m_id)
    except: pass
    
    try:
        foundry_path = os.path.join(os.path.dirname(LIVE_ROOT), "System", "foundry_config.json")
        if os.path.exists(foundry_path):
            with open(foundry_path, 'r') as f:
                cfg = json.load(f)
                cores = cfg.get("cores", {})
                for core_key, core_data in cores.items():
                    if core_key not in model_names:
                        model_names.append(core_key)
                    m_id = core_data.get("model_id")
                    if m_id and m_id not in model_names:
                        model_names.append(m_id)
    except Exception as e:
        logging.error(f"[MODELS] Config read failed: {e}")
    
    models = []
    for m_id in model_names:
        models.append({
            "id": m_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "serenity"
        })
    return {"object": "list", "data": models}

@app.post("/shutdown", dependencies=[Depends(verify_local_key)])
async def shutdown_engine():
    system_announce("Backend shutting down.")
    threading.Thread(target=lambda: os.kill(os.getpid(), signal.SIGTERM), daemon=True).start()
    return {"status": "Offline"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="warning")