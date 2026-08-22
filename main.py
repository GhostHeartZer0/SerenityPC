import tkinter as tk
from tkinter import scrolledtext, simpledialog, messagebox, filedialog, ttk
import tkinter.font as tkFont
import threading, traceback, sys, os, shutil, json, zlib, time, queue, subprocess, re, atexit, webbrowser, io, faulthandler, struct, random
try:
    import numpy as np
except ImportError:
    np = None
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

# --- Smart App Control & Localized Cache Paths ---
def setup_localized_environment():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_dir = os.path.join(base_dir, ".tmp")
    cache_dir = os.path.join(base_dir, ".cache")
    cuda_cache = os.path.join(cache_dir, "cuda")
    triton_cache = os.path.join(cache_dir, "triton")
    torch_ext_dir = os.path.join(cache_dir, "torch_extensions")
    pycache_dir = os.path.join(cache_dir, "pycache")

    for d in [tmp_dir, cache_dir, cuda_cache, triton_cache, torch_ext_dir, pycache_dir]:
        os.makedirs(d, exist_ok=True)

    os.environ["TEMP"] = tmp_dir
    os.environ["TMP"] = tmp_dir
    os.environ["CUDA_CACHE_PATH"] = cuda_cache
    os.environ["TRITON_CACHE_DIR"] = triton_cache
    os.environ["TORCH_EXTENSIONS_DIR"] = torch_ext_dir
    os.environ["PYTHONPYCACHEPREFIX"] = pycache_dir

setup_localized_environment()
if TYPE_CHECKING:
    from tkinter import Canvas, Label, Button, Frame, Text, Scale
    from tkinter.scrolledtext import ScrolledText
    from typing import cast
    import torch
    from turboquant import TurboQuantCache
    import psutil

# --- Import Custom Modules ---
from serenity_resources import (THEME, THEMES, TEXTURE_STYLES, apply_theme_to_global, THERMO_COLORS, CHAT_BG_COLORS, CHAT_FG_COLORS, 
                              INPUT_FG_COLORS, GPU_LAYER_MAP, CONTEXT_SIZE_MAP, 
                              PERSONA_DISPLAY_INFO, PERSONA_IDLE_MAP, PERSONA_PROMPTS, 
                               AVATAR_FILENAMES, ANIMATION_SEQUENCE, DEEP_COOK_PHASES, 
                               DEEP_COOK_SYSTEM_PROMPTS, APP_ICON, TOOLS_DIR,
                               TRI_ATTENTION_ENABLED, TRI_ATTENTION_BUDGET)
from System.serenity_utils import (WidgetLogger, FileAndWidgetLogger, LoadingScreen, 
                            log_uncaught_exception, HardwareProfile, MediaProcessor, SystemMonitor,
                            enable_fault_debugging, ThreadSafeDict, ThreadSafeList, ThinkingDisplay,
                            patch_gguf_architecture, patch_llama_deallocator, ToolTip, TutorialOverlay,
                            enable_high_dpi_awareness)
#from System.ui_watchdog import UIWatchdog #commented out for now to save threads
from System.kv_manager import KVManager, TurboVecIndex
from System.tool_registry import GemmaToolRegistry
from System.modular_registry import ModularRegistry, DynamicParamRegistry
from System.markdown_engine import MarkdownEngine
from System.settings_ui import open_settings_window, open_text_scaling_center, run_auto_detect
from System.vault_manager import VaultManager, DISCLAIMER_WARNING_TEXT
from System.network_guard import set_offline_mode, is_offline_mode
from System.stt_manager import STTManager

# --- Debugging & Fault Handling ---
enable_fault_debugging()

# --- DPI & Hardware Initialization ---
enable_high_dpi_awareness()
HardwareProfile.initialize_gpu_acceleration()


# --- Custom Logic Scripts ---
# (Moved inside the try-except block for robustness)

# --- Library Imports ---
LIBRARIES_LOADED = False
EARLY_IMPORT_ERROR_MSG = ""
SYSTEM_MONITOR_LOADED = False
TORCH_AVAILABLE = False

# Global place holders for deferred imports
llama_cpp = None
Llama = None
Image = None
ImageTk = None
cv2 = None
windnd = None
VisionHandler = None
generate_master_summary = None
settings_manager = None
nvidia_ml = None
torch = None
psutil = None

def load_heavy_libraries():
    global llama_cpp, Llama, Image, ImageTk, cv2, windnd, VisionHandler, generate_master_summary, settings_manager, nvidia_ml, torch, LIBRARIES_LOADED, EARLY_IMPORT_ERROR_MSG, SYSTEM_MONITOR_LOADED, TORCH_AVAILABLE, psutil
    try:
        print("Importing Llama, PIL, and CV2 in background...")
        import llama_cpp as lc
        from llama_cpp import Llama as ll
        from PIL import Image as img, ImageTk as imgtk
        try:
            import cv2 as cv
        except ImportError:
            cv = None
        try:
            import windnd as wd
        except ImportError:
            wd = None
        from System.vision_handler import VisionHandler as vh
        from System.synthesis_handler import generate_master_summary as gms
        from System import settings_manager as sm
        
        llama_cpp = lc
        Llama = ll
        patch_llama_deallocator()
        Image = img
        ImageTk = imgtk
        cv2 = cv
        windnd = wd
        VisionHandler = vh
        generate_master_summary = gms
        settings_manager = sm
        LIBRARIES_LOADED = True
    except Exception as e:
        EARLY_IMPORT_ERROR_MSG = f"FATAL ERROR: Missing library.\n\n{e}"
        print(EARLY_IMPORT_ERROR_MSG, file=sys.stderr)
        LIBRARIES_LOADED = False

    try:
        import psutil as ps
        global psutil
        psutil = ps
        import pynvml as nvml
        nvml.nvmlInit()
        nvidia_ml = nvml
        SYSTEM_MONITOR_LOADED = True
    except Exception as e:
        print(f"Warning: System monitoring libraries (psutil/nvidia-ml-py) not found. {e}", file=sys.stderr)

    try:
        import torch as th
        torch = th
        TORCH_AVAILABLE = True
    except ImportError:
        print("Warning: torch not found. CUDA cache clearing disabled.", file=sys.stderr)

def get_dynamic_core_mask():
    try:
        import psutil
        total = psutil.cpu_count(logical=True)
        if total is None: return None
        if total < 4: return list(range(total))
        if total == 4: return list(range(3)) # Reserve 1
        return list(range(total - 2)) # Reserve 2 for 5+
    except: return None

def set_high_performance_affinity():
    if SYSTEM_MONITOR_LOADED:
        try:
            p = psutil.Process()
            mask = get_dynamic_core_mask() or list(range(psutil.cpu_count(logical=True)))
            p.cpu_affinity(mask) 
            print("[APEX] AI pinned to Performance Cores.")
        except Exception as e:
            print(f"Failed to set CPU affinity: {e}")

def set_apex_affinity():
    """
    Force-locks the process to P-cores and sets priority to Above Normal.
    """
    if not SYSTEM_MONITOR_LOADED: return
    try:
        p = psutil.Process(os.getpid())
        mask = get_dynamic_core_mask() or list(range(psutil.cpu_count(logical=True)))
        p.cpu_affinity(mask)
        p.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
        print(" > [APEX] P-Core Affinity Locked. Turtle mode suppressed.")
    except Exception as e:
        print(f"[APEX] Affinity Lock Failed: {e}")

def kill_engine_on_shutdown(*args, **kwargs):
    """Ensure all engine backend processes are terminated on exit."""
    print("[SYSTEM] Serenity shutting down... cleaning up backend processes.")
    
    # 1. Graceful child termination via psutil (if available)
    if SYSTEM_MONITOR_LOADED:
        try:
            current_process = psutil.Process()
            children = current_process.children(recursive=True)
            for child in children:
                try:
                    print(f"[SYSTEM] Terminating child: {child.pid}")
                    child.terminate()
                except: pass
        except: pass

    # 2. Hard port cleanup for port 8001 (Windows-specific scorched earth)
    if sys.platform == "win32":
        try:
            # Silence the command output to keep the console clean
            os.system('for /f "tokens=5" %a in (\'netstat -aon ^| findstr :8001\') do taskkill /F /PID %a >nul 2>&1')
        except: pass

# Register the cleanup hook
atexit.register(kill_engine_on_shutdown)


# ThreadSafeDict, ThreadSafeList, and ThinkingDisplay moved to System/serenity_utils.py
# GemmaToolRegistry and TurboVecIndex moved to System/tool_registry.py and System/kv_manager.py

# Global Font Specifications (Base Sizes for 100% Text Scale)
BASE_FONT_SPECS = {
    "main": {"family": "Segoe UI", "size": 13},
    "small": {"family": "Segoe UI", "size": 12},
    "italic": {"family": "Segoe UI", "size": 13, "slant": "italic"},
    "large": {"family": "Segoe UI", "size": 14},
    "bold": {"family": "Segoe UI", "size": 13, "weight": "bold"},
    "ui_button": {"family": "Segoe UI", "size": 13},
    "ui_label": {"family": "Segoe UI", "size": 12},
    "ui_small": {"family": "Segoe UI", "size": 11},
    "log": {"family": "Consolas", "size": 8},
    "log_bold": {"family": "Consolas", "size": 8, "weight": "bold"},
    "stats": {"family": "Consolas", "size": 8},
    "stats_bold": {"family": "Consolas", "size": 8, "weight": "bold"},
    # Markdown Support
    "md_bold": {"family": "Segoe UI", "size": 13, "weight": "bold"},
    "md_italic": {"family": "Segoe UI", "size": 13, "slant": "italic"},
    "md_bold_italic": {"family": "Segoe UI", "size": 13, "weight": "bold", "slant": "italic"},
    "md_thought": {"family": "Consolas", "size": 11, "slant": "italic"},
    "md_math_inline": {"family": "Consolas", "size": 12, "slant": "italic"},
    "md_math_block": {"family": "Consolas", "size": 12, "slant": "italic"},
    "md_table": {"family": "Consolas", "size": 11},
    "md_code": {"family": "Consolas", "size": 11},
    "md_header": {"family": "Segoe UI", "size": 15, "weight": "bold"},
    "md_header_1": {"family": "Segoe UI", "size": 17, "weight": "bold"},
    "md_header_2": {"family": "Segoe UI", "size": 15, "weight": "bold"},
    "md_header_3": {"family": "Segoe UI", "size": 13, "weight": "bold"},
    "md_quote": {"family": "Segoe UI", "size": 13, "slant": "italic"},
    "md_strike": {"family": "Segoe UI", "size": 13, "overstrike": True}
}

# Window-responsive scaling: how much each font grows with window size.
# 1.0 = full response (UI text), 0.0 = fixed (ignores window size).
# Code/log fonts intentionally low so they stay compact.
_FONT_WINDOW_RESPONSIVENESS = {
    "log": 0.3, "log_bold": 0.3,
    "stats": 0.15, "stats_bold": 0.15,
    "md_thought": 0.5, "md_math_inline": 0.5, "md_math_block": 0.5,
    "md_table": 0.5, "md_code": 0.5,
}

class ChatbotApp:
    if TYPE_CHECKING:
        root: tk.Tk
        loading_screen: LoadingScreen
        right_panel: Optional['Canvas']
        avatar_image_item: Optional[int]
        avatar_text_item: Optional[int]
        log_window_item: Optional[int]
        thinking_display: Optional['ThinkingDisplay']
        hw_mode_label: Optional['Label']
        system_status_label: Optional['Label']
        persona_name_button: Optional['Button']
        persona_label: Optional['Label']
        lore_btn: Optional['Button']
        depth_slider: Optional['Scale']
        chat_history: Optional['ScrolledText']
        past_history_view: Optional['ScrolledText']
        user_input: Optional['Text']
        status_frame: Optional['Frame']
        prompt_display: Optional['Text']
        desc_container: Optional['Frame']
        persona_desc_label: Optional['Label']
        stats_frame: Optional['Frame']
        log_container: Optional['Frame']
        log_switch_canvas: Optional['Canvas']
        log_frame: Optional['Frame']
        thought_log: Optional['Text']
        error_log: Optional['Text']
        load_model_button: Optional['Button']
        action_button: Optional['Button']
        hurry_button: Optional['Button']
        send_button: Optional['Button']
        deep_thought_button: Optional['Button']
        switch_knob: Optional[int]
        btn_image: Optional['Button']
        btn_video: Optional['Button']
        btn_watch: Optional['Button']
        btn_history: Optional['Button']
        timeline_frame: Optional['tk.Frame']
        progress_label: Optional['tk.Label']
        timeline_bar: Optional['ttk.Progressbar']
        style: Optional['ttk.Style']
        stats_labels: Dict[str, 'Label']
        params: Dict[str, Any]
        model_paths: Dict[str, str]
        gpu_layer_config: Dict[str, int]
        context_size_config: Dict[str, int]
        messages: List[Dict[str, str]]
        state: Dict[str, Any]
        pending_task: Optional[Dict[str, Any]]
        dirs: Dict[str, str]
        config_file: str
        scratchpad_file: str
        error_log_file: str
        script_dir: str
        icon_path: str
        fonts: Dict[str, tkFont.Font]
        active_persona_level: int
        max_persona_level: int
        model: Optional[Any]
        model_path: str
        current_model_tier: Optional[str]
        gpu_handle: Optional[Any]
        text_buffer: str
        last_update_time: float
        chunk_counter: int
        avatar_states: Dict[str, Any]
        avatar_pil_images: Dict[str, Any]
        config: Dict[str, Any]
        stop_process: threading.Event
        process_queue: queue.Queue
        idle_timer_id: Optional[str]
        tmp_img: Optional[Any]
        current_trans_img: Optional[Any]
        chat_handler: Optional[Any]
        sub_chunk_size: int
        _status_timer: Optional[str]
        stats_thread: Optional[threading.Thread]
        secret_trigger: Optional[Any]
        last_user_message: str

    def __init__(self, tk_root, loading_screen):
        print("ChatbotApp initializing...")
        self.root = tk_root
        self.root.title("Serenity AI - Control Panel")
        self.loading_screen = loading_screen
        self.tool_registry = GemmaToolRegistry(self)
        self.dynamic_param_registry = DynamicParamRegistry()
        self.stt_manager = STTManager()
        self.mic_button = None
        
        self._rgb_supported_val = None
        self.media_cache = {}
        self.link_counter = 0
        self.placeholder_img = None
        
        # --- Configuration & Paths ---
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.BASE_DIR = self.script_dir
        self.live_dir = os.path.join(self.script_dir, "Live")
        
        # Determine icon path
        # Determine icon path (Prioritize serenity_resources.APP_ICON)
        self.icon_path = APP_ICON
        if not os.path.exists(self.icon_path):
            self.icon_path = ""

        # --- Set Window Icon ---
        if self.icon_path:
            try: self.root.iconbitmap(self.icon_path)
            except Exception: pass

        if not LIBRARIES_LOADED:
            messagebox.showerror("Dependency Error", EARLY_IMPORT_ERROR_MSG)
            self.root.quit(); return

        self.dirs = {d: os.path.join(self.script_dir, d) for d in ["Media", "History", "Models", "Logs", "System", "Users"]}
        for d in self.dirs.values(): os.makedirs(d, exist_ok=True)
        
        self.turbo_vec = None

        # Start background polling        
        # Support user-preferred 'settings.json' in root or System/
        self.config_file = os.path.join(self.script_dir, "System", "config.json")
        for p in [os.path.join(self.script_dir, "settings.json"), 
                  os.path.join(self.script_dir, "System", "settings.json")]:
            if os.path.exists(p):
                self.config_file = p
                break
        
        if not os.path.exists(self.config_file) and os.path.exists(os.path.join(self.script_dir, "config.json")):
            self.config_file = os.path.join(self.script_dir, "config.json")
        self.scratchpad_file = os.path.join(self.dirs["Logs"], "scratchpad.txt")
        self.error_log_file = os.path.join(self.dirs["Logs"], "error_log.txt")

        # Migrate legacy root history & user files
        self._migrate_legacy_user_files()

        # --- Visual Resources ---
        self.fonts = {}
        for k, spec in BASE_FONT_SPECS.items():
            self.fonts[k] = tkFont.Font(**spec)
        
        # Apply initial text scale and font family from config
        initial_scale = 100
        if hasattr(self, "config") and self.config:
            initial_scale = self.config.get("text_scale", 100)
            ui_fam = self.config.get("ui_font", "Segoe UI")
            mono_fam = self.config.get("mono_font", "Consolas")
            self.apply_font_family(ui_fam, mono_fam, persist=False)
        self.apply_text_scale(initial_scale, persist=False)
        
        # --- State Management ---
        self.state = ThreadSafeDict({
            "running": False, "deep_cook": False, "initial_setup": False,
            "turbo": False, "persona_clicks": 0, "log_view": "thought",
            "deep_cook_behavior": "oneshot",
            "safety_margin_mb": 200, 
            "virtual_vram": 0,
            "avatar_current": "off",
            "response_started": False,
            "staged_multimodal": None,
            "processing_queue": [],
            "auto_watch": False,
            "xmemory_active": False,
            "vram_layer_offset": 0,
            "staged_attachments": [],
            "multimodal_engine": "Internal",
            "streaming_mode": "Buffered",
            "max_token_ratio": 4,
            "dmn_backbone": {}
        })
        self.history_state = {"view": "levels", "level": None}

        self.stop_process = threading.Event()
        self.process_queue = queue.Queue()
        
        # Core Objects
        self.model = None
        self.model_path = ""
        self.media_processor = MediaProcessor(self)
        self.system_monitor = SystemMonitor(self)
        self.current_model_tier = None
        self.active_persona_level = 3
        
        # Data Containers
        self.avatar_states = {}     
        self.avatar_pil_images = {} 
        self.config = ThreadSafeDict()
        self.params = {}
        self.model_paths = {
            "fast": "", "search": "", "low": "", "med": "", "high": "", "secret": "", "deep_cook": "", 
            "vision_video": "", "vision_video_projector": "",
            "vision_video_deep": "", "vision_video_deep_projector": "",
            "vision_multimodal": "", "vision_multimodal_projector": ""
        }
        
        tier_list = ["fast", "search", "low", "med", "high", "secret", "deep_cook", "vision_video", "vision_video_deep", "vision_multimodal"]
        self.gpu_layer_config = {tier: -1 for tier in tier_list}
        self.context_size_config = {tier: 32768 if "vision" not in tier else 8192 for tier in tier_list}
        self.context_size_config["high"] = 65536
        self.context_size_config["secret"] = 131072
        self.context_size_config["vision_video_deep"] = 16384
        
        self.kv_manager = None
        self.vault_manager = VaultManager(
            history_dir=self.dirs.get("History", os.path.join(self.script_dir, "History")),
            state_dir=self.dirs.get("System", os.path.join(self.script_dir, "System"))
        )
        self._last_user_activity_time = time.time()
        self._vault_modal_open = False
        self.temp_config = {tier: 0.8 if "vision" not in tier else 0.1 for tier in tier_list}
        self.temp_config["vision_multimodal"] = 1.0 # Gemma-4 Best Practice
        self.temp_config["secret"] = 0.5 # More accurate and focused
        
        self.top_p_config = {tier: 0.95 if "vision" not in tier else 0.9 for tier in tier_list}
        self.top_p_config["vision_multimodal"] = 0.95 # Gemma-4 Best Practice
        
        self.min_p_config = {tier: 0.05 if "vision" not in tier else 0.0 for tier in tier_list}
        self.min_p_config["secret"] = 0.1 # Tighter bounds
        self.repeat_penalty_config = {tier: 1.1 for tier in tier_list}
        self.repeat_penalty_config["secret"] = 1.15 # Less repeating
        self.frequency_penalty_config = {tier: 0.0 for tier in tier_list}
        self.presence_penalty_config = {tier: 0.0 for tier in tier_list}
        self.stop_strings_config = {tier: "###,<|endoftext|>,<|im_end|>,<|turn>,<turn|>,<tool_call|>,<tool_response|>,<eos>" for tier in tier_list}
        self.n_batch_config = {tier: 512 for tier in tier_list}
        
        self.top_k_config = {tier: 64 for tier in tier_list}
        self.top_k_config["vision_multimodal"] = 64 # Gemma-4 Best Practice

        # Load DMN Backbone
        self._load_dmn_backbone()
        self.max_persona_level = 7  # Persisted range for the slider
        self.messages = []
        self.gpu_handle = None
        self.text_buffer = ""
        self.last_update_time = 0.0
        self.chunk_counter = 0
        
        self.pending_task = None
        
        # --- UI Placeholders for Static Analysis ---
        self.right_panel = None
        self.avatar_image_item = None
        self.avatar_text_item = None
        self.log_window_item = None
        self.thinking_display = None
        self.hw_mode_label = None
        self.system_status_label = None
        self.persona_name_button = None
        self.live_agent_process = None
        self.persona_label = None
        self.lore_btn = None
        self.depth_slider = None
        self.chat_history = None
        self.past_history_view = None
        self.user_input = None
        self.status_frame = None
        self.prompt_display = None
        self.desc_container = None
        self.persona_desc_label = None
        self.stats_frame = None
        self.stats_labels = {}
        self.stats_row_frames = []
        self.stats_title_labels = []
        self.log_container = None
        self.log_header_frame = None
        self.log_header_label = None
        self.self_analysis_btn = None
        self.lock_logout_btn = None
        self.clear_log_btn = None
        self.log_switch_canvas = None
        self.switch_knob = None
        self.log_frame = None
        self.thought_log = None
        self.error_log = None
        self.tool_log = None
        self.diag_log = None
        self.load_model_button = None
        self.action_button = None
        self.hurry_button = None
        self.btn_image = None
        self.btn_video = None
        self.btn_watch = None
        self.btn_clear_queue = None
        self.btn_active = None
        self.btn_history = None
        self.active_tab = "active"
        self.send_button = None
        self.deep_thought_button = None
        self.hurry_button = None
        self.secret_trigger = None
        self.last_user_message = ""
        self._status_timer = None
        self.stats_thread = None
        self.idle_timer_id = None
        self.tmp_img = None
        self.current_trans_img = None
        self.chat_handler = None
        self.sub_chunk_size = 8 # Default to 8 frames per slice for 6GB stability

        self.log_update_buffer = ""
        self.thought_stream_buffer = ""
        self.last_log_dispatch = 0
        self.log_update_limit = 100 # Max messages per queue tick to prevent UI freeze

        # --- Queue Dispatch Table ---
        self.queue_handlers: Dict[str, Any] = {
            "load_success": lambda msg: self._handle_load_success(msg),
            "load_error": lambda msg: self._handle_load_error(msg),
            "stats_update": lambda msg: self._update_stats_display(msg.get("stats", {})) if self.stats_labels else None,
            "log_update": lambda msg: self._buffer_thought_log(msg.get("content", "")),
            "thought_log_update": lambda msg: self._buffer_thought_log(msg.get("content", "")),
            "thought_stream": lambda msg: self._buffer_thought_stream(msg.get("content", "")),
            "error_log_update": lambda msg: self._buffer_log(msg.get("content", "")),
            "tool_log_update": lambda msg: self._buffer_tool_log(msg.get("content", "")),
            "diag_log_update": lambda msg: self._buffer_diag_log(msg.get("content", "")),
            "thinking_status": lambda msg: self.thinking_display.update_status(msg.get("content", "Thinking...")) if self.thinking_display and self.thinking_display.winfo_exists() else None,
            "status_phase": lambda msg: self.thinking_display.set_phase(msg.get("phase", ""), msg.get("details", ""), msg.get("tokens", 0), msg.get("speed", 0.0), msg.get("progress_val", -1)) if self.thinking_display and self.thinking_display.winfo_exists() else None,
            "status_ttft": lambda msg: self.thinking_display.record_ttft(msg.get("ttft", 0.0)) if self.thinking_display and self.thinking_display.winfo_exists() else None,
            "streaming": lambda msg: self._buffer_text(msg.get("content", "")),
            "streaming_replace": lambda msg: self._replace_ai_message(msg.get("content", "")),
            "success": lambda msg: self._handle_session_finished({"user_msg": self.last_user_message, "final_answer": msg.get("content", ""), "is_error": False}),
            "session_finished": lambda msg: self._handle_session_finished(msg),
            "interrupted": lambda msg: self._handle_session_finished({"user_msg": self.last_user_message, "final_answer": msg.get("content", ""), "is_error": True}),
            "deep_cook_ui_batch": lambda msg: self._handle_deep_cook_ui_batch(msg),
            "deep_cook_ui_start": lambda msg: self._handle_deep_cook_ui_start(msg),
            "deep_cook_ui_stream": lambda msg: self._handle_deep_cook_ui_stream(msg),
            "vision_oneshot_finish": lambda msg: self.offload_model(),
            "video_progress": lambda msg: self._set_progress(msg.get("content", 0)),
            "stt_transcript": lambda msg: self._handle_stt_result(msg.get("content", ""), msg.get("error", None)),
            "error": lambda msg: self._handle_session_finished({"user_msg": self.last_user_message, "final_answer": msg.get("content", ""), "is_error": True}),
            "cleanup": lambda msg: self._run_hygiene_on_main_thread()
        }

        self._setup_llama_log_capture()
        self.setup_ui()
        
        # --- Global Zoom Keyboard Shortcuts ---
        for k in ("<Control-plus>", "<Control-equal>", "<Control-KP_Add>"):
            self.root.bind_all(k, self.zoom_in)
        for k in ("<Control-minus>", "<Control-KP_Subtract>"):
            self.root.bind_all(k, self.zoom_out)
        for k in ("<Control-0>", "<Control-KP_0>"):
            self.root.bind_all(k, self.zoom_reset)

        self.root.after(100, lambda *args: self.final_initial_setup())
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.set_ui_state(model_loaded=False, generating=False)

    # ================= USER PROFILES & DIRECTORIES =================
    def get_active_username(self) -> str:
        if not hasattr(self, "config") or self.config is None:
            return "Default"
        un = str(self.config.get("username", "Default")).strip()
        return un if un else "Default"

    def get_user_history_dir(self, username: Optional[str] = None) -> str:
        un = username if username else self.get_active_username()
        p = os.path.join(self.dirs["History"], un)
        os.makedirs(p, exist_ok=True)
        return p

    def get_user_dir(self, username: Optional[str] = None) -> str:
        un = username if username else self.get_active_username()
        p = os.path.join(self.dirs["Users"], un)
        os.makedirs(p, exist_ok=True)
        return p

    def list_user_profiles(self) -> List[str]:
        show_def = self.config.get("show_default_profile", True) if hasattr(self, "config") and self.config else True
        show_pub = self.config.get("show_public_profile", True) if hasattr(self, "config") and self.config else True
        users = set()
        if show_def: users.add("Default")
        if show_pub: users.add("Public")
        excluded = {"backups", "backups_repair", "jsonz to txt"}
        if os.path.exists(self.dirs["Users"]):
            for item in os.listdir(self.dirs["Users"]):
                if os.path.isdir(os.path.join(self.dirs["Users"], item)) and item not in excluded:
                    if (item == "Default" and not show_def) or (item == "Public" and not show_pub):
                        continue
                    users.add(item)
        if os.path.exists(self.dirs["History"]):
            for item in os.listdir(self.dirs["History"]):
                if os.path.isdir(os.path.join(self.dirs["History"], item)) and item not in excluded:
                    if (item == "Default" and not show_def) or (item == "Public" and not show_pub):
                        continue
                    users.add(item)
        curr = self.get_active_username()
        if curr and curr not in excluded: users.add(curr)
        if not users: users.add("Default")
        return sorted(list(users))

    def switch_user(self, new_username: str):
        clean_un = "".join(c for c in new_username.strip() if c.isalnum() or c in ("-", "_", " ")).strip()
        if not clean_un: clean_un = "Default"
        
        # Save active config before switching
        if hasattr(self, "save_config"):
            try: self.save_config()
            except Exception: pass
            
        self.config["username"] = clean_un
        user_dir = self.get_user_dir(clean_un)
        self.get_user_history_dir(clean_un)
        
        # Load user profile config if exists
        u_cfg_p = os.path.join(user_dir, "config.json")
        if os.path.exists(u_cfg_p):
            try:
                with open(u_cfg_p, "r", encoding="utf-8") as f:
                    u_data = json.load(f)
                self.config.update(u_data)
                self.config["username"] = clean_un
            except Exception as e:
                print(f"[USER] Failed to load user config {u_cfg_p}: {e}")
        else:
            self.save_config()

        # Apply profile-specific theme, scale, fonts
        try:
            from serenity_resources import apply_theme_to_global
            apply_theme_to_global(
                self.config.get("theme", "apex"),
                self.config.get("texture_style", "default"),
                self.config.get("dark_mode", False),
                getattr(self, "active_persona_level", 3),
                (getattr(self, 'model', None) is not None)
            )
            if hasattr(self, "apply_current_theme"):
                self.apply_current_theme()
            if hasattr(self, "apply_text_scale") and "text_scale" in self.config:
                self.apply_text_scale(self.config["text_scale"], persist=False)
            if hasattr(self, "apply_font_family") and "ui_font" in self.config:
                self.apply_font_family(self.config.get("ui_font", "Segoe UI"), self.config.get("mono_font", "Consolas"), persist=False)
            saved_sash = self.config.get('sash_pos', -1)
            if isinstance(saved_sash, (int, float)) and saved_sash > 50:
                self.root.after(100, lambda: self._apply_sash_pos(int(saved_sash)))
        except Exception as e:
            print(f"[USER] Failed to apply user theme settings: {e}")

        # Default profile: start with a fresh ephemeral session
        if clean_un == "Default" and hasattr(self, "messages"):
            self.messages = []
            if hasattr(self, "chat_history") and self.chat_history:
                self.chat_history.config(state=tk.NORMAL)
                self.chat_history.delete("1.0", tk.END)
                self.chat_history.config(state=tk.DISABLED)

        self._load_dmn_backbone()
        if hasattr(self, 'load_history') and clean_un != "Default":
            self.load_history(render_active=True)
        if hasattr(self, 'refresh_history_view') and getattr(self, 'active_tab', '') == "history":
            self.refresh_history_view()
        self._log_and_display(f"Switched user profile to: {clean_un}")

    def _migrate_legacy_user_files(self):
        """Auto-migrates legacy flat History/*.history.* files into History/Default/"""
        try:
            hist_root = self.dirs.get("History")
            if hist_root and os.path.exists(hist_root):
                target_def = os.path.join(hist_root, "Default")
                for f in os.listdir(hist_root):
                    full_p = os.path.join(hist_root, f)
                    if os.path.isfile(full_p) and (f.endswith(".history.jsonz") or f.endswith(".history.encz")):
                        os.makedirs(target_def, exist_ok=True)
                        dest = os.path.join(target_def, f)
                        if not os.path.exists(dest):
                            shutil.move(full_p, dest)
        except Exception as e:
            print(f"[USER] Legacy migration warning: {e}")

    # ================= UI & SETUP =================
    def final_initial_setup(self):
        if self.state["initial_setup"]: return
        self.state["initial_setup"] = True
        self.config = self.load_config()
        if 'main_window' in self.config and self.config['main_window']: 
            self.root.geometry(self.config['main_window'])
        else:
            sw = self.root.winfo_screenwidth()
            sh = self.root.winfo_screenheight()
            w = min(1280, max(1000, int(sw * 0.75)))
            h = min(880, max(680, int(sh * 0.80)))
            x = max(0, (sw - w) // 2)
            y = max(0, (sh - h) // 2)
            self.root.geometry(f"{w}x{h}+{x}+{y}")
        self.root.minsize(960, 600)
        
        if self.loading_screen:
            self.loading_screen.stop_and_destroy()
            self.root.deiconify()
        
        self.redirect_logs()
        set_apex_affinity()
        HardwareProfile.set_priority("above_normal")
        self.root.update_idletasks()

        # Apply persisted font family, text scale, and sash position
        if hasattr(self, "apply_font_family") and "ui_font" in self.config:
            self.apply_font_family(self.config.get("ui_font", "Segoe UI"), self.config.get("mono_font", "Consolas"), persist=False)
        if hasattr(self, "apply_text_scale") and "text_scale" in self.config:
            self.apply_text_scale(self.config.get("text_scale", 100), persist=False)
        
        saved_sash = self.config.get('sash_pos', -1)
        if isinstance(saved_sash, (int, float)) and saved_sash > 50:
            self.root.after(350, lambda: self._apply_sash_pos(int(saved_sash)))
        
        # Start UI Watchdog to detect freezes
        #self.ui_watchdog = UIWatchdog(self.root)
        #self.ui_watchdog.start() #commented out for now to save threads
        
        self.load_all_images()
        self.initialize_app()
        self.system_monitor.start()
        self.check_gpu_support()
        self._initialize_rgb_state()
        
        # Windnd Drag and Drop Hook
        if windnd is not None and hasattr(windnd, "hook_dropfiles"):
            try:
                windnd.hook_dropfiles(self.root, func=self._handle_drop_files)
            except Exception as e:
                print(f"Failed to hook drag-and-drop: {e}")

        # Non-blocking, lazy query for RGB support
        self._check_rgb_support_async()

        # Synchronize button state and colors with loaded config
        self.set_ui_state(model_loaded=(self.model is not None), generating=False)

        # Non-blocking, background initialization for TurboVec history indexing
        threading.Thread(target=self._init_turbovec, daemon=True).start()

        # Start Inactivity Watchdog & Startup Lock Verification 
        self._start_inactivity_watchdog()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        if hasattr(self, 'vault_manager') and self.vault_manager.is_locked():
            self.root.after(200, self.show_vault_unlock_modal)
        elif not self.config.get("tutorial_completed", False):
            self.root.after(800, self.start_tutorial_walkthrough)

    def start_tutorial_walkthrough(self):
        """Launches the interactive translucent tutorial walkthrough."""
        if hasattr(self, '_tutorial_overlay') and self._tutorial_overlay and getattr(self._tutorial_overlay, 'win', None):
            try:
                self._tutorial_overlay.win.lift()
                return
            except Exception:
                pass
        self._tutorial_overlay = TutorialOverlay(self)

    def _on_user_activity(self, event=None):
        """Resets the inactivity timer on user interaction."""
        self._last_user_activity_time = time.time()

    def _start_inactivity_watchdog(self):
        """Binds user activity listeners and starts the background inactivity watchdog."""
        try:
            self.root.bind_all("<Key>", self._on_user_activity, add="+")
            self.root.bind_all("<Button-1>", self._on_user_activity, add="+")
            self.root.bind_all("<Button-2>", self._on_user_activity, add="+")
            self.root.bind_all("<Button-3>", self._on_user_activity, add="+")
        except Exception: pass
        self.root.after(3000, self._check_inactivity_lock)

    def _check_inactivity_lock(self):
        """Checks if inactivity timeout is exceeded and locks vault if active."""
        if hasattr(self, 'vault_manager') and self.vault_manager.is_lock_enabled():
            if not self.vault_manager.is_locked() and not self._vault_modal_open:
                limit_sec = self.vault_manager.get_auto_lock_seconds()
                if limit_sec > 0:
                    idle_time = time.time() - self._last_user_activity_time
                    if idle_time >= limit_sec:
                        print(f"[VAULT] Inactivity timeout ({limit_sec}s) reached. Locking Serenity Vault.")
                        self.vault_manager.lock()
                        self.show_vault_unlock_modal()
        # Reschedule watchdog
        if self.root.winfo_exists():
            self.root.after(3000, self._check_inactivity_lock)

    def lock_app(self):
        """Manually locks Serenity Vault and displays unlock modal."""
        if not hasattr(self, 'vault_manager'):
            return
        if not self.vault_manager.is_lock_enabled():
            if messagebox.askyesno("Vault Lock Not Configured", "Vault password protection is not configured.\nWould you like to open Settings to set a Master Password?"):
                self.open_settings_window()
            return
        self.vault_manager.lock()
        self.show_vault_unlock_modal()

    def lock_and_logout(self):
        """Locks the Serenity Vault, logs out current profile, and switches to Default profile."""
        if hasattr(self, 'vault_manager'):
            self.vault_manager.lock()
        print("[VAULT] Vault locked and active user logged out.")
        if hasattr(self, 'switch_user'):
            self.switch_user("Default")
        self.show_vault_unlock_modal()

    def show_vault_unlock_modal(self, on_unlock_callback=None):
        """Displays a modal startup profile selector and vault access dialog."""
        if self._vault_modal_open: return
        self._vault_modal_open = True

        unlock_win = tk.Toplevel(self.root)
        unlock_win.title("Serenity - Profile & Vault Access")
        unlock_win.geometry("460x360")
        unlock_win.config(bg=THEME["bg_color"])
        unlock_win.transient(self.root)
        unlock_win.grab_set()

        # Center on parent window
        try:
            x = self.root.winfo_x() + (self.root.winfo_width() // 2) - 230
            y = self.root.winfo_y() + (self.root.winfo_height() // 2) - 180
            unlock_win.geometry(f"460x360+{x}+{y}")
        except: pass

        tk.Label(unlock_win, text="👤 USER PROFILE & VAULT ACCESS", font=self.fonts["large"], 
                 bg=THEME["bg_color"], fg=THEME["electric_blue"]).pack(pady=(16, 4))

        # Profile selection
        prof_frame = tk.Frame(unlock_win, bg=THEME["bg_color"])
        prof_frame.pack(fill=tk.X, padx=20, pady=(6, 4))

        tk.Label(prof_frame, text="Select Profile:", font=self.fonts["ui_label"],
                 bg=THEME["bg_color"], fg=THEME["fg_color"]).pack(side=tk.LEFT, padx=(0, 8))

        all_profiles = self.list_user_profiles()
        if "Default" not in all_profiles: all_profiles.insert(0, "Default")
        if "Public" not in all_profiles: all_profiles.append("Public")
        
        curr_un = self.get_active_username()
        selected_prof_var = tk.StringVar(value=curr_un if curr_un in all_profiles else "Default")
        prof_combo = ttk.Combobox(prof_frame, textvariable=selected_prof_var, values=all_profiles, state="readonly", width=18)
        prof_combo.pack(side=tk.LEFT, padx=4)

        info_lbl = tk.Label(unlock_win, text="", font=self.fonts["ui_small"],
                            bg=THEME["bg_color"], fg="#aaaaaa", wraplength=400, justify=tk.CENTER)
        info_lbl.pack(pady=(4, 8))

        pwd_frame = tk.Frame(unlock_win, bg=THEME["bg_color"])
        pwd_frame.pack(fill=tk.X, padx=20, pady=2)

        lbl_pwd = tk.Label(pwd_frame, text="Master Password:", font=self.fonts["small"],
                           bg=THEME["bg_color"], fg=THEME["fg_color"])
        lbl_pwd.pack(pady=(0, 4))

        pwd_var = tk.StringVar()
        pwd_entry = tk.Entry(pwd_frame, textvariable=pwd_var, show="*", width=26, 
                             bg=THEME["widget_bg_color"], fg=THEME["fg_color"], 
                             insertbackground=THEME["fg_color"], font=self.fonts["main"], relief=tk.SUNKEN)
        pwd_entry.pack(pady=2)

        err_lbl = tk.Label(unlock_win, text="", font=self.fonts["ui_button"], bg=THEME["bg_color"], fg="#ff4444")
        err_lbl.pack(pady=2)

        btn_row = tk.Frame(unlock_win, bg=THEME["bg_color"])
        btn_row.pack(pady=(10, 15))

        action_btn = tk.Button(btn_row, text="Proceed", font=self.fonts["ui_button"],
                               bg=THEME["button_active_color"], fg=THEME["fg_color"], padx=14, pady=4, relief=tk.FLAT)
        action_btn.pack(side=tk.LEFT, padx=6)

        def _on_profile_change(*args):
            sel = selected_prof_var.get().strip()
            err_lbl.config(text="")
            if sel == "Default":
                info_lbl.config(text="Default Session: Ephemeral fresh workspace without encryption.")
                lbl_pwd.pack_forget()
                pwd_entry.pack_forget()
                action_btn.config(text="🚀 Enter as Default", bg=THEME["button_active_color"])
            elif sel == "Public":
                info_lbl.config(text="Public Profile: Persistent shared history, accessible without encryption.")
                lbl_pwd.pack_forget()
                pwd_entry.pack_forget()
                action_btn.config(text="🌐 Enter as Public", bg=THEME["button_active_color"])
            else:
                info_lbl.config(text=f"Private Profile '{sel}': Protected with AES-256 Vault Encryption.")
                lbl_pwd.pack(pady=(0, 4))
                pwd_entry.pack(pady=2)
                action_btn.config(text="🔓 Unlock Profile", bg=THEME["button_active_color"])
                pwd_entry.focus_set()

        prof_combo.bind("<<ComboboxSelected>>", _on_profile_change)
        _on_profile_change()

        def _do_action(event=None):
            sel = selected_prof_var.get().strip() or "Default"
            if sel in ("Default", "Public"):
                self._vault_modal_open = False
                self._last_user_activity_time = time.time()
                self.switch_user(sel)
                unlock_win.destroy()
                self._log_and_display(f"Entered workspace as {sel}.")
                if on_unlock_callback:
                    on_unlock_callback()
                else:
                    self.load_history()
                    if hasattr(self, 'history_state') and self.history_state.get("view") == "list":
                        self._render_history_menu()
                return

            # Private profile requires vault unlock
            pwd = pwd_var.get().strip()
            if not pwd:
                err_lbl.config(text="Password cannot be empty for private profile.")
                return

            if self.vault_manager.unlock(pwd):
                self._vault_modal_open = False
                self._last_user_activity_time = time.time()
                self.switch_user(sel)
                unlock_win.destroy()
                self._log_and_display(f"Vault unlocked. Active profile: {sel}.")
                if on_unlock_callback:
                    on_unlock_callback()
                else:
                    self.load_history()
                    if hasattr(self, 'history_state') and self.history_state.get("view") == "list":
                        self._render_history_menu()
            else:
                err_lbl.config(text="❌ Incorrect master password.")
                pwd_var.set("")
                pwd_entry.focus_set()

        action_btn.config(command=_do_action)
        pwd_entry.bind("<Return>", _do_action)
        pwd_entry.bind("<KP_Enter>", _do_action)

        def _on_close_modal():
            if self.vault_manager.is_locked() and self.get_active_username() not in ("Default", "Public"):
                if messagebox.askyesno("Exit Serenity", "Serenity is locked. Exit application?", parent=unlock_win):
                    self._vault_modal_open = False
                    unlock_win.destroy()
                    self.root.destroy()
            else:
                self._vault_modal_open = False
                unlock_win.destroy()

        unlock_win.protocol("WM_DELETE_WINDOW", _on_close_modal)

    def check_gpu_support(self):
        """Final verification of GPU capabilities."""
        try:
            from llama_cpp import llama_supports_gpu_offload
            if not llama_supports_gpu_offload():
                print("Warning: llama-cpp-python installed WITHOUT GPU support.")
                self._log_and_display("GPU Acceleration UNAVAILABLE (CPU-only mode)")
        except Exception as e:
            print(f"GPU Capability Check Failed: {e}")

    def _add_btn(self, parent, text, cmd, side=tk.LEFT, **kwargs):
        pack_opts = {'fill': kwargs.pop('fill', tk.X), 'expand': kwargs.pop('expand', (side == tk.LEFT)), 'padx': 5, 'pady': 0}
        
        # Selectively extract keys that have Literal constraints in the Button constructor
        btn_relief = kwargs.pop('relief', tk.FLAT)
        btn_anchor = kwargs.pop('anchor', None) # Default to None if not specified
        
        style = {"font": self.fonts["ui_button"], "bg": THEME["button_bg_color"], "fg": THEME["fg_color"]}
        style.update(kwargs)
        
        # Explicitly pass relief and anchor to help the linter
        btn = tk.Button(parent, text=text, command=cmd, relief=btn_relief, **style)
        if btn_anchor:
            btn.config(anchor=btn_anchor)
            
        btn.pack(side=side, **pack_opts)
        return btn

    def _expand_context_config(self, tier, required_ctx):
        """
        Dynamically adjusts context_size_config in config.json to prevent OOM
        or truncation on high-res multimodal tasks.
        """
        if self.context_size_config.get(tier, 0) >= required_ctx:
            return

        print(f"[CONFIG] Expanding context for {tier}: {self.context_size_config.get(tier)} -> {required_ctx}")
        self.context_size_config[tier] = required_ctx
        
        # PERSIST TO DISK
        try:
            config_data = {}
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f: config_data = json.load(f)
            
            if "context_size_config" not in config_data:
                config_data["context_size_config"] = {}
            
            config_data["context_size_config"][tier] = required_ctx
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=4)
            print(f"[CONFIG] Updated {self.config_file} with new context budget.")
        except Exception as e:
            print(f"[CONFIG] Failed to persist expanded context: {e}")

    def _determine_tier(self):
        if self.root is None: return
        self.root.config(bg=THEME["bg_color"])
        self.paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg=THEME["bg_color"], 
                                   sashwidth=4, sashrelief=tk.FLAT, borderwidth=0)
        self.paned.pack(fill=tk.BOTH, expand=True)

    def _setup_llama_log_capture(self):
        """Intercepts C++ backend logs and routes specific diagnostics to the UI."""
        if not LIBRARIES_LOADED: return
        
        def _llama_log_callback(level, message, user_data):
            try:
                if not message: return
                msg_str = message.decode('utf-8', errors='ignore').strip()
                
                # --- DIAG LIST (🔍) ---
                diag_identifiers = [
                    "llama_perf_context_print:",
                    "CPU KV buffer size",
                    "CUDA0 KV buffer size",
                    "llama_kv_cache: size",
                    "llama_kv_cache_iswa:",
                    "load_tensors: offload",
                    "load_tensors:   CPU",
                    "load_tensors:        CUDA",
                    "general.name",
                    "print_info: file type",
                    "print_info: file size",
                    "llama_model_loader: - type"
                ]

                if any(p in msg_str for p in diag_identifiers):
                    self.process_queue.put({"status": "diag_log_update", "content": f"[C++] {msg_str}"})
                    return

                # --- ERROR & CRITICAL DIAG ONLY (⚠) ---
                if any(kw in msg_str.lower() for kw in ["error", "failed", "exception", "cuda error"]):
                    self.process_queue.put({"status": "log_update", "content": f"\n[INTERNAL ERROR] {msg_str}\n"})
            except: pass

        try:
            import ctypes
            cb_type = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
            c_callback = cb_type(_llama_log_callback)
            llama_cpp.llama_log_set(c_callback, ctypes.c_void_p(0))
            self._llama_log_cb_ref = c_callback # Prevent Garbage Collection
        except Exception as e:
            print(f"Failed to hook llama_cpp logger: {e}", file=sys.stderr)

    def setup_ui(self):
        if self.root is None: return
        self.root.config(bg=THEME["bg_color"])
        self.paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, bg=THEME["bg_color"], 
                                   sashwidth=4, sashrelief=tk.FLAT, borderwidth=0)
        self.paned.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(self.paned, bg=THEME["bg_color"])
        self.paned.add(left, stretch="always")
        
        left.grid_rowconfigure(0, weight=0) # Dedicated 3-line Loading Bar / Status space
        left.grid_rowconfigure(1, weight=0) # Top Action Buttons (Settings - Clear)
        left.grid_rowconfigure(2, weight=1) # Chat History preference
        left.grid_rowconfigure(3, weight=0) # Prompt box
        left.grid_rowconfigure(4, weight=0) # Persona Controls
        left.grid_rowconfigure(5, weight=0) # Description box stable
        left.grid_columnconfigure(0, weight=1)

        # --- 0. DEDICATED LOADING BAR / STATUS DISPLAY (Above Settings - Clear Buttons) ---
        s_frame = tk.Frame(left, bg=THEME["bg_color"])
        self.status_frame = s_frame
        s_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(6, 2))
        self.thinking_display = ThinkingDisplay(s_frame, app=self)

        # --- 1. TOP BUTTONS (Settings - Clear) ---
        top = tk.Frame(left, bg=THEME["bg_color"])
        self.top_bar_frame = top
        top.grid(row=1, column=0, sticky="ew", padx=10, pady=(2, 6))
        
        btn_set = self._add_btn(top, "Settings", self.open_settings_window)
        self.load_model_button = btn_set
        ToolTip(btn_set, "Open Model Settings & Hardware Configuration.", app=self)
        
        btn_act = self._add_btn(top, "Begin!", self.model_swap)
        self.action_button = btn_act
        ToolTip(btn_act, "Load selected model tier or swap active model.", app=self)
        
        # Multimodal Prep Buttons
        btn_vid = self._add_btn(top, "[🎥] Video", self.initiate_video_multimodal)
        self.btn_video = btn_vid
        ToolTip(btn_vid, "Initiate video multimodal frame analysis.", app=self)
        
        # Replace the old Watch button
        btn_wat = self._add_btn(top, "[🧠] Pulse", self.toggle_auto_watch)
        self.btn_watch = btn_wat
        ToolTip(btn_wat, "Toggle background Pulse & idle observation.", app=self)
        
        btn_clr = self._add_btn(top, "Clear", self._reset_multimodal_ui)
        self.btn_clear_queue = btn_clr
        ToolTip(btn_clr, "Clear active attachments and media queue.", app=self)

        # --- 2. CHAT FRAME ---
        chat_frame = tk.Frame(left, bg=THEME["trim_color"])
        chat_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)

        # --- TAB CONTROLS ---
        tab_frame = tk.Frame(chat_frame, bg=THEME["trim_color"])
        self.tab_bar_frame = tab_frame
        tab_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 2))

        btn_tab_act = tk.Button(tab_frame, text="Active Chat", command=self.show_active_chat, 
                                bg=THEME["button_active_color"], fg=THEME["fg_color"], font=self.fonts["ui_button"], relief=tk.FLAT)
        self.btn_active = btn_tab_act
        btn_tab_act.pack(side=tk.LEFT, padx=2)
        ToolTip(btn_tab_act, "Switch to live conversation stream.", app=self)

        btn_tab_hist = tk.Button(tab_frame, text="History Archive", command=self.show_history, 
                                 bg=THEME["button_bg_color"], fg="#aaaaaa", font=self.fonts["ui_button"], relief=tk.FLAT)
        self.btn_history = btn_tab_hist
        btn_tab_hist.pack(side=tk.LEFT, padx=2)
        ToolTip(btn_tab_hist, "Search and review archived conversation histories.", app=self)

        lbl_status = tk.Label(tab_frame, text="System: Ready", bg=THEME["trim_color"], 
                                          fg="#888888", font=self.fonts["italic"])
        self.system_status_label = lbl_status
        lbl_status.pack(side=tk.RIGHT, padx=10)
        ToolTip(lbl_status, "System engine status and telemetry indicator.", app=self)

        lbl_hw = tk.Label(tab_frame, text="", bg=THEME["trim_color"], font=self.fonts["bold"])
        self.hw_mode_label = lbl_hw
        lbl_hw.pack(side=tk.RIGHT, padx=5)
        ToolTip(lbl_hw, "Hardware architecture optimization mode (Apex / Legacy) and offline guard status.", app=self)
        self._update_hw_indicator()

        # --- TEXT WIDGETS ---
        # 1. Floating Pinned Prompt (Hidden on startup)
        txt_prompt = tk.Text(chat_frame, height=3, font=self.fonts["italic"], wrap=tk.WORD,
                             bg=THEME["trim_color"], fg="#87CEFA", relief=tk.FLAT, 
                             highlightthickness=0, padx=10, pady=5)
        self.prompt_display = txt_prompt

        # 2. User-Provided Timeline Progress (Apex Dark Theme)
        self.timeline_frame = tk.Frame(chat_frame, bg="#1e1e1e")
        self.progress_label = tk.Label(self.timeline_frame, text="TIMELINE: 0%", bg="#1e1e1e", fg="#00ffcc", font=self.fonts["stats"])
        ToolTip(self.progress_label, "Real-time timeline token generation progress.", app=self)
        self.progress_label.pack(side="top", anchor="w")
        
        self.style = ttk.Style()
        self.style.theme_use('default')
        self.style.configure("Apex.Horizontal.TProgressbar", thickness=10, troughcolor='#2d2d2d', background='#00ffcc')

        self.timeline_bar = ttk.Progressbar(
            self.timeline_frame, 
            orient="horizontal", 
            length=100, 
            mode="determinate", 
            style="Apex.Horizontal.TProgressbar"
        )
        self.timeline_bar.pack(fill="x", expand=True)
        self.timeline_frame.pack_forget() # Hide initially
        
        # 3. Main Chat Output (Manual Freescroll)
        txt_chat = scrolledtext.ScrolledText(chat_frame, font=self.fonts["main"], wrap=tk.WORD,
                                             bg=THEME["widget_bg_color"], fg=THEME["fg_color"], 
                                             relief=tk.FLAT, highlightthickness=0)
        self.chat_history = txt_chat
        txt_chat.pack(side=tk.TOP, fill="both", expand=True, padx=2, pady=2)
        
        # Configure Markdown tags
        txt_chat.tag_config("user_lead", font=self.fonts["bold"], foreground="#87CEFA")
        txt_chat.tag_config("user", font=self.fonts["italic"], foreground="#007acc") # Electric Blue
        txt_chat.tag_config("ai_lead", font=self.fonts["bold"], foreground="#FFD700")
        txt_chat.tag_config("md_bold", font=self.fonts["md_bold"])
        txt_chat.tag_config("md_italic", font=self.fonts["md_italic"])
        txt_chat.tag_config("md_bold_italic", font=self.fonts["md_bold_italic"])
        txt_chat.tag_config("md_thought", font=self.fonts["md_thought"], foreground="#808080", lmargin1=25, lmargin2=40)
        txt_chat.tag_config("md_list", lmargin1=25, lmargin2=40)
        txt_chat.tag_config("md_math_inline", font=self.fonts["md_math_inline"], foreground="#E5C07B")
        txt_chat.tag_config("md_math_block", font=self.fonts["md_math_block"], foreground="#E5C07B", lmargin1=40, lmargin2=40)
        txt_chat.tag_config("md_table", font=self.fonts["md_table"], foreground="#A7C080", lmargin1=15, lmargin2=15)
        txt_chat.tag_config("md_code", font=self.fonts["md_code"], foreground="#E06C75", background="#1e1e1e", lmargin1=15, lmargin2=15)
        txt_chat.tag_config("md_header", font=self.fonts["md_header_2"], foreground="#00ffcc")
        txt_chat.tag_config("md_header_1", font=self.fonts["md_header_1"], foreground="#00ffcc")
        txt_chat.tag_config("md_header_2", font=self.fonts["md_header_2"], foreground="#00ffcc")
        txt_chat.tag_config("md_header_3", font=self.fonts["md_header_3"], foreground="#80ffe5")
        txt_chat.tag_config("md_quote", font=self.fonts["md_quote"], foreground="#98c379", lmargin1=20, lmargin2=30)
        txt_chat.tag_config("md_strike", font=self.fonts["md_strike"], foreground="#7f848e")

        # 3. History Archive (Hidden by default)
        self.history_menu_frame = tk.Frame(chat_frame, bg=THEME["bg_color"])
        
        txt_past = scrolledtext.ScrolledText(chat_frame, font=self.fonts["main"], wrap=tk.WORD, 
                                             bg=THEME["widget_bg_color"], fg="#aaaaaa", relief=tk.FLAT)
        self.past_history_view = txt_past
        txt_past.tag_config("user_lead", font=self.fonts["bold"], foreground="#87CEFA")
        txt_past.tag_config("user", font=self.fonts["italic"], foreground="#007acc") # Electric Blue
        txt_past.tag_config("ai_lead", font=self.fonts["bold"], foreground="#FFD700")
        txt_past.tag_config("md_bold", font=self.fonts["md_bold"])
        txt_past.tag_config("md_italic", font=self.fonts["md_italic"])
        txt_past.tag_config("md_bold_italic", font=self.fonts["md_bold_italic"])
        txt_past.tag_config("md_thought", font=self.fonts["md_thought"], foreground="#808080", lmargin1=25, lmargin2=40)
        txt_past.tag_config("md_list", lmargin1=25, lmargin2=40)
        txt_past.tag_config("md_math_inline", font=self.fonts["md_math_inline"], foreground="#E5C07B")
        txt_past.tag_config("md_math_block", font=self.fonts["md_math_block"], foreground="#E5C07B", lmargin1=40, lmargin2=40)
        txt_past.tag_config("md_table", font=self.fonts["md_table"], foreground="#A7C080", lmargin1=15, lmargin2=15)
        txt_past.tag_config("md_code", font=self.fonts["md_code"], foreground="#E06C75", background="#1e1e1e", lmargin1=15, lmargin2=15)
        txt_past.tag_config("md_header", font=self.fonts["md_header_2"], foreground="#00ffcc")
        txt_past.tag_config("md_header_1", font=self.fonts["md_header_1"], foreground="#00ffcc")
        txt_past.tag_config("md_header_2", font=self.fonts["md_header_2"], foreground="#00ffcc")
        txt_past.tag_config("md_header_3", font=self.fonts["md_header_3"], foreground="#80ffe5")
        txt_past.tag_config("md_quote", font=self.fonts["md_quote"], foreground="#98c379", lmargin1=20, lmargin2=30)
        txt_past.tag_config("md_strike", font=self.fonts["md_strike"], foreground="#7f848e")

        # --- 3. INPUT FRAME (Themed Border) ---
        input_frame = tk.Frame(left, bg=THEME["trim_color"], highlightthickness=1, 
                               highlightbackground=THEME["trim_color"], highlightcolor=THEME["electric_blue"])
        self.input_control_frame = input_frame
        input_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=5)
        
        # Attachment Bar (New)
        self.attachment_frame = tk.Frame(input_frame, bg=THEME["trim_color"])
        self.attachment_frame.pack(side=tk.TOP, fill=tk.X, padx=2, pady=(2,0))
        
        txt_user = tk.Text(input_frame, height=3, font=self.fonts["main"], wrap=tk.WORD,
                           bg=THEME["widget_bg_color"], fg=THEME["fg_color"], 
                           highlightthickness=1, highlightbackground=THEME["trim_color"], 
                           highlightcolor=THEME["electric_blue"], relief=tk.FLAT)
        self.user_input = txt_user
        txt_user.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        txt_user.bind("<KeyPress>", self._handle_input_key)
        ToolTip(txt_user, "Enter prompt. Press Enter to send, Shift+Enter for newline.", app=self)

        # --- 4. PERSONA CONTROLS (Single Slider Fix) ---
        p_frame = tk.Frame(left, bg=THEME["bg_color"])
        self.persona_control_frame = p_frame
        p_frame.grid(row=4, column=0, sticky="ew", padx=10, pady=5)
        self._setup_persona_controls(p_frame)
        
        # --- 5. PERSONA DESCRIPTION (UI Fix for Cutoff) ---
        d_cont = tk.Frame(left, bg=THEME["bg_color"])
        self.desc_container = d_cont
        d_cont.grid(row=5, column=0, sticky="ew", padx=10, pady=2)
        
        lbl_desc = tk.Label(d_cont, text="", font=self.fonts["small"], 
                                          bg=THEME["bg_color"], fg=THEME["electric_blue"],
                                          anchor="center", wraplength=500)
        self.persona_desc_label = lbl_desc
        lbl_desc.pack(fill=tk.BOTH, expand=True)
        ToolTip(lbl_desc, "Cognitive and stylistic description of the selected persona tier.", app=self)

        def _on_left_resize(event):
            # Dynamic wraplength: 90% of the left frame width
            new_width = event.width - 40
            if new_width > 50:
                lbl_desc.config(wraplength=new_width)
            # Dynamic slider length auto-scaling
            if hasattr(self, 'depth_slider') and self.depth_slider:
                dynamic_len = max(70, min(160, int(event.width * 0.22)))
                try: self.depth_slider.config(length=dynamic_len)
                except Exception: pass
        
        left.bind("<Configure>", _on_left_resize)

        # Window-responsive font scaling: fonts grow with window size
        self._window_scale_factor = 1.0
        self._root_resize_job = None
        self.root.bind("<Configure>", self._on_root_configure)

        # --- FOOTER BUTTONS ---
        ctrl_frame = tk.Frame(left, bg=THEME["bg_color"])
        self.footer_control_frame = ctrl_frame
        ctrl_frame.grid(row=5, column=0, sticky="ew", padx=10, pady=5)
        
        self.lock_button = None

        self.rgb_button = self._add_btn(ctrl_frame, "[🌈] RGB", self.open_rgb_panel, side=tk.LEFT, width=12)
        ToolTip(self.rgb_button, "Open RGB ambient lighting controls.", app=self)
        if not self.config.get("show_rgb_button", True) or not self._is_rgb_supported():
            self.rgb_button.pack_forget()
        
        btn_send = self._add_btn(ctrl_frame, "Send", self.send_message, side=tk.RIGHT)
        self.send_button = btn_send
        ToolTip(btn_send, "Send prompt to active model.", app=self)
        
        btn_deep = self._add_btn(ctrl_frame, "Deep Cook", self.toggle_deep_cook_mode, side=tk.RIGHT)
        self.deep_thought_button = btn_deep
        ToolTip(btn_deep, "Toggle Deep Cook multi-cycle recursive synthesis.", app=self)
        
        btn_halt = self._add_btn(ctrl_frame, "Halt", self.halt_process, side=tk.RIGHT)
        self.hurry_button = btn_halt
        ToolTip(btn_halt, "Halt active token generation.", app=self)

        # Ghost Mode and History Usage UI Toggles
        self.ghost_button = self._add_btn(ctrl_frame, self._get_ghost_mode_label(), self.toggle_ghost_mode, side=tk.RIGHT, font=self.fonts["ui_button"], fg=self._get_ghost_mode_color())
        ToolTip(self.ghost_button, "Toggle Ghost Mode (disables chat history logging to disk).", app=self)
        self.history_usage_button = self._add_btn(ctrl_frame, self._get_history_usage_label(), self.toggle_history_usage, side=tk.RIGHT, font=self.fonts["ui_button"], fg=self._get_history_usage_color())
        ToolTip(self.history_usage_button, "Toggle TurboVec long-term history recall.", app=self)

        # Right Panel (Avatar & Stats)
        canvas_r = tk.Canvas(self.paned, bg=THEME["bg_color"], highlightthickness=0)
        self.right_panel = canvas_r
        self.paned.add(canvas_r, stretch="always")
        
        self.avatar_image_item = canvas_r.create_image(0,0, anchor="center")
        self.avatar_text_item = canvas_r.create_text(0,0, anchor="center", font=self.fonts["large"], fill=THEME["electric_blue"])
        
        self._setup_logs_and_stats()
        if self.log_container is not None:
            from typing import cast
            self.log_window_item = canvas_r.create_window(0, 0, window=cast(tk.Widget, self.log_container), anchor="center")

        canvas_r.bind("<Configure>", lambda *args: self._position_canvas_elements())
        
        # Bind sash movement & release events for automatic mid resize persistence
        self.paned.bind("<ButtonRelease-1>", self._on_sash_released, add="+")
        self.root.bind("<ButtonRelease-1>", self._on_sash_released, add="+")

        # Restore Sash Position
        saved_sash = self.config.get('sash_pos', -1)
        if isinstance(saved_sash, (int, float)) and saved_sash > 50:
            self.root.after(300, lambda: self._apply_sash_pos(int(saved_sash)))
        else:
            # Default split (3:2 approx)
            self.root.after(300, lambda: self._apply_sash_pos(int(self.root.winfo_width() * 0.6)))
            
        self.root.after(250, lambda *args: self._position_canvas_elements())

    def _get_current_sash_pos(self):
        try:
            if hasattr(self, 'paned') and self.paned:
                coord = self.paned.sash_coord(0)
                if coord and coord[0] > 50:
                    return coord[0]
        except Exception:
            pass
        return self.config.get('sash_pos', -1) if hasattr(self, 'config') and self.config else -1

    def _apply_sash_pos(self, pos):
        try:
            if hasattr(self, 'paned') and self.paned:
                max_w = self.root.winfo_width()
                target_x = max(100, min(max_w - 100, pos)) if max_w > 200 else pos
                self.paned.sash_place(0, target_x, 0)
                if hasattr(self, 'config') and self.config is not None:
                    self.config['sash_pos'] = target_x
                self._position_canvas_elements()
        except Exception:
            pass

    def _on_sash_released(self, event=None):
        try:
            pos = self._get_current_sash_pos()
            if pos and pos > 50:
                self.config['sash_pos'] = pos
                self.save_config()
                self._position_canvas_elements()
        except Exception:
            pass

    def on_closing(self):
        try:
            pos = self._get_current_sash_pos()
            if pos and pos > 50:
                self.config['sash_pos'] = pos
            self.save_config()
        except Exception:
            pass
        if self.root:
            self.root.destroy()

    def _update_hw_indicator(self):
        """Updates the Hardware Mode indicator based on CPU specs and offline status."""
        info = HardwareProfile.get_cpu_info()
        physical = info["physical"]
        is_off = is_offline_mode() or bool(self.config.get("offline_mode", False))
        off_tag = " [OFFLINE]" if is_off else ""
        
        # Threshold: i7 usually has > 8 physical cores (or 12+ logical)
        if self.hw_mode_label is not None:
            if physical >= 8:
                self.hw_mode_label.config(text=f"[APEX i7]{off_tag}", fg="#ffaa00" if is_off else "#00FF7F")
            else:
                self.hw_mode_label.config(text=f"[LEGACY i5]{off_tag}", fg="#ffaa00" if is_off else "#FFD700")

    def _setup_persona_controls(self, p_frame):
        """Sets up the persona selection buttons and slider in the given frame."""
        lbl_p = tk.Label(p_frame, text="Persona:", font=self.fonts["small"], 
                                     bg=THEME["bg_color"], fg=THEME["electric_blue"])
        self.persona_label = lbl_p
        lbl_p.pack(side=tk.LEFT)
        lbl_p.bind("<Button-1>", self._on_persona_label_click)
        ToolTip(lbl_p, "Adjust persona depth (click 6 times for Secret Level 7).", app=self)

        # Extended to Level 6 dynamically
        scale_d = tk.Scale(p_frame, from_=1, to=self.max_persona_level, orient=tk.HORIZONTAL, length=110, 
                                    bg=THEME["bg_color"], fg=THEME["fg_color"], relief=tk.FLAT, 
                                    command=self.update_persona_display, showvalue=False)
        self.depth_slider = scale_d
        scale_d.set(3)
        scale_d.pack(side=tk.LEFT, padx=(6, 2))
        ToolTip(scale_d, "Slide between Persona Levels 1 to 6/7.", app=self)

        # SECRET TRIGGER: Invisible gap right next to the slider (matching background color)
        lbl_sec = tk.Label(p_frame, text="  ", bg=THEME["bg_color"], fg=THEME["bg_color"], 
                           bd=0, highlightthickness=0, relief=tk.FLAT, cursor="arrow", width=2)
        self.secret_trigger = lbl_sec
        lbl_sec.pack(side=tk.LEFT)
        lbl_sec.bind("<Double-Button-1>", self._load_secret_model_event)

        btn_name = tk.Button(p_frame, text="", command=self.model_swap, 
                                             font=self.fonts["bold"], bg=THEME["button_bg_color"], 
                                             fg=THEME["fg_color"], relief=tk.FLAT, padx=6, pady=2)
        self.persona_name_button = btn_name
        btn_name.pack(side=tk.LEFT, padx=3)
        ToolTip(btn_name, "Active persona level name and tier.", app=self)

        # Plus button for attachments
        btn_add = tk.Button(p_frame, text="+", command=self._show_attachment_menu,
                            font=self.fonts["bold"], bg=THEME["button_bg_color"], 
                            fg=THEME["fg_color"], relief=tk.FLAT, padx=6, pady=2)
        btn_add.pack(side=tk.LEFT, padx=2)
        ToolTip(btn_add, "Attach images, documents, or video slices.", app=self)

        # STT Voice input trigger (to the right of plus sign)
        btn_mic = tk.Button(p_frame, text="🎙️", command=self.toggle_voice_recording,
                            font=self.fonts["main"], bg=THEME["button_bg_color"],
                            fg=THEME["fg_color"], relief=tk.FLAT, padx=4, pady=2)
        self.mic_button = btn_mic
        btn_mic.pack(side=tk.LEFT, padx=2)
        ToolTip(btn_mic, "Dictate voice input offline using local speech recognition.", app=self)
        
        # Attachments Popup Menu
        self.attachment_menu = tk.Menu(self.root, tearoff=0, bg=THEME["bg_color"], fg=THEME["fg_color"])
        self.attachment_menu.add_command(label="📷 Add Image", command=lambda: self._browse_attachment("image"))
        self.attachment_menu.add_command(label="🎵 Add Audio", command=lambda: self._browse_attachment("audio"))
        self.attachment_menu.add_command(label="📄 Add Document", command=lambda: self._browse_attachment("document"))

        btn_lore = tk.Button(p_frame, text="📜 Open Chronicles", command=self.launch_lore_book,
                                 font=self.fonts["ui_button"], bg="#1a1a1a", fg=THEME["electric_blue"], 
                                 relief=tk.FLAT, padx=6, pady=2)
        self.lore_btn = btn_lore
        ToolTip(btn_lore, "Open Prime Chronicles and subconscious dream journals.", app=self)

    def _show_attachment_menu(self, event=None):
        if hasattr(self, "attachment_menu"):
            x = self.root.winfo_pointerx()
            y = self.root.winfo_pointery()
            self.attachment_menu.tk_popup(x, y)

    def _browse_attachment(self, att_type):
        """Open a file dialog based on attachment type."""
        from tkinter import filedialog
        filetypes = {
            "image": [("Images", "*.png *.jpg *.jpeg *.webp *.bmp")],
            "audio": [("Audio", "*.mp3 *.wav *.ogg *.flac *.m4a")],
            "document": [("Documents", "*.txt *.md *.csv *.json *.py *.js *.html *.cpp *.h")]
        }
        files = filedialog.askopenfilenames(title=f"Select {att_type.capitalize()}(s)", filetypes=filetypes.get(att_type, [("All Files", "*.*")]))
        if files:
            for f in files: self._add_staged_attachment(f, att_type)

    def _handle_drop_files(self, files):
        """Callback for windnd drag-and-drop."""
        import os
        import sys
        for f in files:
            if isinstance(f, bytes):
                try:
                    p = f.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        p = f.decode('mbcs')
                    except UnicodeDecodeError:
                        p = f.decode('utf-8', errors='replace')
            else:
                p = str(f)
                
            if not os.path.isfile(p): continue
            
            ext = os.path.splitext(p)[1].lower()
            if ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']:
                att_type = "image"
            elif ext in ['.mp3', '.wav', '.ogg', '.flac', '.m4a']:
                att_type = "audio"
            elif ext in ['.mp4', '.mkv', '.avi', '.mov']:
                att_type = "video"
            else:
                # Fallback to document (text) if extension is unknown or text-based
                # Warning: parsing binary files as text will cause issues during generation,
                # but Serenity's media processor can usually catch this.
                att_type = "document"
                
            self._add_staged_attachment(p, att_type)
        self.set_ui_state()

    def _add_staged_attachment(self, fpath, att_type):
        """Add a file to the multimodal attachment queue and render it in the UI."""
        import os
        if "staged_attachments" not in self.state: self.state["staged_attachments"] = []
        
        fname = os.path.basename(fpath)
        att = {"path": fpath, "type": att_type, "name": fname}
        self.state["staged_attachments"].append(att)
        
        # Render UI Token
        if hasattr(self, "attachment_frame") and self.attachment_frame is not None:
            if not self.attachment_frame.winfo_viewable():
                self.attachment_frame.pack(side=tk.TOP, fill=tk.X, padx=2, pady=(2,0), before=self.user_input)
            
            token_frame = tk.Frame(self.attachment_frame, bg="#2a2a2a", bd=1, relief=tk.SOLID)
            token_frame.pack(side=tk.LEFT, padx=2, pady=2)
            att["token_frame"] = token_frame
            
            icons = {"image": "📷", "audio": "🎵", "document": "📄", "video": "🎥"}
            icon = icons.get(att_type, "📄")
            
            lbl = tk.Label(token_frame, text=f"{icon} {fname[:15]}{'...' if len(fname)>15 else ''}", bg="#2a2a2a", fg="#00ffcc", font=self.fonts["stats"])
            lbl.pack(side=tk.LEFT, padx=(2,0))
            
            # Remove Button
            btn_rm = tk.Button(token_frame, text="X", bg="#4a0000", fg="#ffffff", relief=tk.FLAT, font=self.fonts["stats"],
                               command=lambda af=att, tf=token_frame: self._remove_staged_attachment(af, tf))
            btn_rm.pack(side=tk.LEFT, padx=2)
            
        print(f"[SYSTEM] Attached {att_type}: {fname}")
        
    def _remove_staged_attachment(self, att, token_frame):
        """Remove a staged attachment and its UI token."""
        if att in self.state.get("staged_attachments", []):
            self.state["staged_attachments"].remove(att)
        token_frame.destroy()
        
        if not self.state.get("staged_attachments") and hasattr(self, "attachment_frame"):
            self.attachment_frame.pack_forget()
        print(f"[SYSTEM] Removed attachment: {att['name']}")

    def show_active_chat(self):
        self.active_tab = "active"
        if self.past_history_view is not None:
            self.past_history_view.pack_forget()
        if self.history_menu_frame is not None:
            self.history_menu_frame.pack_forget()
        
        # Bring back the pinned prompt only if it contains active text
        if self.prompt_display is not None and self.prompt_display.winfo_exists():
            prompt_txt = self.prompt_display.get("1.0", tk.END).strip()
            if prompt_txt:
                self.prompt_display.pack(side=tk.TOP, fill="x", padx=2, pady=(2, 0))
            else:
                self.prompt_display.pack_forget()
        
        # We pack chat_history first so we can refer to it with 'before=' if we pack others dynamically
        if self.chat_history is not None:
            self.chat_history.pack(side=tk.TOP, fill="both", expand=True, padx=2, pady=0)
            
        if hasattr(self, 'timeline_frame') and self.timeline_frame is not None and self.timeline_frame.winfo_exists():
            if self.state.get("processing_multimodal", False):
                self.timeline_frame.pack(side=tk.TOP, fill="x", padx=10, pady=2, before=self.chat_history)
        
        if self.btn_active is not None:
            self.btn_active.config(bg=THEME["button_active_color"], fg=THEME["fg_color"])
        if self.btn_history is not None:
            self.btn_history.config(bg=THEME["button_bg_color"], fg="#aaaaaa")

    def show_history(self):
        self.active_tab = "history"
        # Hide the pinned prompt and chat history
        if self.prompt_display is not None:
            self.prompt_display.pack_forget()
        if hasattr(self, 'timeline_frame') and self.timeline_frame.winfo_exists():
            self.timeline_frame.pack_forget()
        if self.chat_history is not None:
            self.chat_history.pack_forget()
        
        # Reset to unified list view
        if not hasattr(self, 'history_state') or not isinstance(self.history_state, dict):
            self.history_state = {}
        self.history_state.setdefault("view", "list")
        self.history_state.setdefault("level_filter", "All Levels")
        self.history_state.setdefault("date_filter", "All Dates")
        self.history_state.setdefault("sort_by", "Newest First")
        self.history_state.setdefault("search_query", "")
        self.history_state.setdefault("deep_search_matches", {})
        if not hasattr(self, '_history_content_cache'):
            self._history_content_cache = {}
        if not hasattr(self, '_search_debounce_timer'):
            self._search_debounce_timer = None

        self._render_history_menu()
        
        if self.history_menu_frame is not None:
            self.history_menu_frame.pack(side=tk.TOP, fill="both", expand=True, padx=2, pady=2)
            
        if self.btn_history is not None:
            self.btn_history.config(bg=THEME["button_active_color"], fg=THEME["fg_color"])
        if self.btn_active is not None:
            self.btn_active.config(bg=THEME["button_bg_color"], fg="#aaaaaa")

    def _render_history_menu(self):
        """Unified History Archive Renderer with Dropdowns, Date Grouping, and Deep Search."""
        if self.history_menu_frame is None: return
        
        # Clear frame
        for child in self.history_menu_frame.winfo_children():
            child.destroy()
            
        view = self.history_state.get("view", "list")
        
        # Navigation / Control Bar
        nav_bar = tk.Frame(self.history_menu_frame, bg=THEME["bg_color"])
        nav_bar.pack(side=tk.TOP, fill=tk.X, pady=(2, 4))
        
        if view == "content":
            back_btn = tk.Button(nav_bar, text="⬅ Back to Archives", command=self._back_history, 
                                bg=THEME["button_bg_color"], fg=THEME["fg_color"], relief=tk.FLAT,
                                font=self.fonts["ui_button"], cursor="hand2")
            back_btn.pack(side=tk.LEFT, padx=5)
            
            title = f"Archive: {self.history_state.get('current_display_name', 'Chat Log')}"
            tk.Label(nav_bar, text=title, font=self.fonts["italic"], bg=THEME["bg_color"], 
                     fg=THEME["electric_blue"]).pack(side=tk.LEFT, padx=10)
            
            # Action Frame for right-side buttons
            act_frame = tk.Frame(nav_bar, bg=THEME["bg_color"])
            act_frame.pack(side=tk.RIGHT, padx=5)
            
            # Edit Button
            edit_text = "💾 Save" if self.past_history_view.cget("state") == "normal" else "✏️ Edit"
            edit_bg = "#005a9e" if self.past_history_view.cget("state") == "normal" else THEME["button_bg_color"]
            tk.Button(act_frame, text=edit_text, command=self._toggle_history_edit, 
                      bg=edit_bg, fg="white", relief=tk.FLAT, font=self.fonts["ui_button"],
                      cursor="hand2").pack(side=tk.LEFT, padx=5)
            
            # Delete Button
            tk.Button(act_frame, text="🗑️ Delete", command=self._delete_current_archive, 
                      bg="#4a0000", fg="white", relief=tk.FLAT, font=self.fonts["ui_button"],
                      cursor="hand2").pack(side=tk.LEFT, padx=5)

            # Content Area for Text View
            content_frame = tk.Frame(self.history_menu_frame, bg=THEME["bg_color"])
            content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.past_history_view.pack(in_=content_frame, side=tk.TOP, fill="both", expand=True)
            return

        # --- LIST VIEW: UNIFIED FILTER & SEARCH CONTROLS ---
        tk.Label(nav_bar, text="📁 History Archive", font=self.fonts["bold"], 
                 bg=THEME["bg_color"], fg=THEME["electric_blue"]).pack(side=tk.LEFT, padx=6)
        
        # Search Entry Bar (Enter to search, Search button, No auto-search)
        search_frame = tk.Frame(nav_bar, bg=THEME["widget_bg_color"], bd=1, relief=tk.SUNKEN)
        search_frame.pack(side=tk.LEFT, padx=6, fill=tk.X, expand=True)
        
        search_var = tk.StringVar(value=self.history_state.get("search_query", ""))
        search_entry = tk.Entry(search_frame, textvariable=search_var, bg=THEME["widget_bg_color"], 
                                fg=THEME["fg_color"], insertbackground=THEME["fg_color"], 
                                relief=tk.FLAT, font=self.fonts["small"])
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4, pady=2)
        
        def _execute_search(event=None):
            q = search_var.get().strip()
            self.history_state["search_query"] = q
            self._trigger_history_search(q)

        # Bind Enter key to search
        search_entry.bind("<Return>", _execute_search)
        search_entry.bind("<KP_Enter>", _execute_search)

        # Dedicated Search Button
        btn_search = tk.Button(search_frame, text="🔍 Search", command=_execute_search,
                               bg=THEME["button_bg_color"], fg=THEME["fg_color"], relief=tk.FLAT,
                               font=self.fonts["ui_button"], cursor="hand2", padx=6, pady=1)
        btn_search.pack(side=tk.RIGHT, padx=(2, 2))
        
        if search_var.get():
            btn_clear_search = tk.Button(search_frame, text="✕", command=lambda: self._clear_history_search(search_var),
                                         bg=THEME["widget_bg_color"], fg="#aaaaaa", relief=tk.FLAT, 
                                         font=self.fonts["ui_small"], cursor="hand2", bd=0)
            btn_clear_search.pack(side=tk.RIGHT, padx=2)

        # Dropdowns Frame
        filter_frame = tk.Frame(nav_bar, bg=THEME["bg_color"])
        filter_frame.pack(side=tk.RIGHT, padx=4)

        # 1. Level Filter Dropdown
        level_options = ["All Levels", "Level 1", "Level 2", "Level 3", "Level 4", "Level 5", "Level 6", "Level 7"]
        lvl_var = tk.StringVar(value=self.history_state.get("level_filter", "All Levels"))
        lvl_combo = ttk.Combobox(filter_frame, textvariable=lvl_var, values=level_options, 
                                 state="readonly", width=11, font=self.fonts["small"])
        lvl_combo.pack(side=tk.LEFT, padx=3)
        lvl_combo.bind("<<ComboboxSelected>>", lambda e: self._on_history_dropdown_change("level_filter", lvl_var.get()))

        # 2. Date Separator & Filter Dropdown
        date_options = ["All Dates", "Today", "Yesterday", "Past 7 Days", "Past 30 Days", "Older"]
        date_var = tk.StringVar(value=self.history_state.get("date_filter", "All Dates"))
        date_combo = ttk.Combobox(filter_frame, textvariable=date_var, values=date_options, 
                                  state="readonly", width=11, font=self.fonts["small"])
        date_combo.pack(side=tk.LEFT, padx=3)
        date_combo.bind("<<ComboboxSelected>>", lambda e: self._on_history_dropdown_change("date_filter", date_var.get()))

        # 3. Sort Dropdown
        sort_options = ["Newest First", "Oldest First", "Name (A-Z)", "Name (Z-A)", "Size (Largest)"]
        sort_var = tk.StringVar(value=self.history_state.get("sort_by", "Newest First"))
        sort_combo = ttk.Combobox(filter_frame, textvariable=sort_var, values=sort_options, 
                                  state="readonly", width=12, font=self.fonts["small"])
        sort_combo.pack(side=tk.LEFT, padx=3)
        sort_combo.bind("<<ComboboxSelected>>", lambda e: self._on_history_dropdown_change("sort_by", sort_var.get()))

        # Content Area: Scrollable Canvas with Targeted Mousewheel Binding
        content_frame = tk.Frame(self.history_menu_frame, bg=THEME["bg_color"])
        content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(2, 0))

        canvas = tk.Canvas(content_frame, bg=THEME["bg_color"], highlightthickness=0)
        scroll_frame = tk.Frame(canvas, bg=THEME["bg_color"])
        self._history_canvas = canvas

        canvas_window = canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        
        def _on_canvas_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(canvas_window, width=event.width)
            
        def _on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        canvas.bind("<Configure>", _on_canvas_configure)
        scroll_frame.bind("<Configure>", _on_frame_configure)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # TARGETED SCROLLING: strictly bind mousewheel when entering history widget, unbind on leave
        self._bind_targeted_history_scroll(canvas, canvas)
        self._bind_targeted_history_scroll(scroll_frame, canvas)

        # Fetch, Filter, Sort, and Group History Entries
        entries = self._get_all_history_entries()
        q = self.history_state.get("search_query", "").lower()
        lvl_filter = self.history_state.get("level_filter", "All Levels")
        date_filter = self.history_state.get("date_filter", "All Dates")
        sort_by = self.history_state.get("sort_by", "Newest First")
        deep_matches = self.history_state.get("deep_search_matches", {})

        filtered = []
        for e in entries:
            # Level Filter
            if lvl_filter != "All Levels":
                try:
                    target_lvl = int(lvl_filter.replace("Level ", ""))
                    if e.get("level") != target_lvl:
                        continue
                except: pass
            
            # Date Filter
            if date_filter != "All Dates" and e.get("date_bucket") != date_filter:
                continue

            # Search Query (Title + Deep JSONZ match)
            if q:
                title_match = q in e.get("display_name", "").lower() or q in e.get("filename", "").lower()
                deep_match = e.get("path") in deep_matches
                if not (title_match or deep_match):
                    continue
                if deep_match:
                    e["snippet"] = deep_matches[e["path"]]

            filtered.append(e)

        # Sort entries
        if sort_by == "Newest First":
            filtered.sort(key=lambda x: x.get("mtime", 0), reverse=True)
        elif sort_by == "Oldest First":
            filtered.sort(key=lambda x: x.get("mtime", 0))
        elif sort_by == "Name (A-Z)":
            filtered.sort(key=lambda x: x.get("display_name", "").lower())
        elif sort_by == "Name (Z-A)":
            filtered.sort(key=lambda x: x.get("display_name", "").lower(), reverse=True)
        elif sort_by == "Size (Largest)":
            filtered.sort(key=lambda x: x.get("size_bytes", 0), reverse=True)

        if not filtered:
            msg = "No history archives match current filters." if (q or lvl_filter != "All Levels" or date_filter != "All Dates") else "No history files found."
            tk.Label(scroll_frame, text=msg, bg=THEME["bg_color"], fg="#888888", 
                     font=self.fonts["italic"]).pack(pady=40)
            return

        # Render Entries with Date Bucket Separators if sorted by Date
        current_bucket = None
        is_date_sorted = sort_by in ["Newest First", "Oldest First"]

        for item in filtered:
            bucket = item.get("date_bucket", "Other")
            if is_date_sorted and bucket != current_bucket:
                current_bucket = bucket
                sep_frame = tk.Frame(scroll_frame, bg=THEME["bg_color"])
                sep_frame.pack(fill=tk.X, padx=12, pady=(10, 3))
                tk.Label(sep_frame, text=f"── {bucket.upper()} ──", bg=THEME["bg_color"], 
                         fg=THEME["electric_blue"], font=self.fonts["ui_button"]).pack(side=tk.LEFT)
                self._bind_targeted_history_scroll(sep_frame, canvas)

            # Item Card Container
            card = tk.Frame(scroll_frame, bg=THEME["widget_bg_color"], bd=1, relief=tk.RAISED, cursor="hand2")
            card.pack(fill=tk.X, padx=10, pady=3)
            self._bind_targeted_history_scroll(card, canvas)

            # Header row inside card
            hdr_row = tk.Frame(card, bg=THEME["widget_bg_color"])
            hdr_row.pack(fill=tk.X, padx=8, pady=(5, 2))
            self._bind_targeted_history_scroll(hdr_row, canvas)

            # Level Badge
            lvl_val = item.get("level")
            lvl_text = f" L{lvl_val} " if lvl_val is not None else " ARCH "
            lvl_bg = "#5c007a" if lvl_val in [6, 7] else ("#005a9e" if lvl_val == 5 else "#2a4d3a")
            badge = tk.Label(hdr_row, text=lvl_text, bg=lvl_bg, fg="white", 
                             font=self.fonts["ui_button"], padx=4, pady=1)
            badge.pack(side=tk.LEFT, padx=(0, 8))
            self._bind_targeted_history_scroll(badge, canvas)

            # Display Name
            name_lbl = tk.Label(hdr_row, text=item.get("display_name", "Chat Session"), 
                                bg=THEME["widget_bg_color"], fg=THEME["fg_color"], 
                                font=self.fonts["main"], anchor="w")
            name_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self._bind_targeted_history_scroll(name_lbl, canvas)

            # Date & Size Subtitle
            meta_str = f"{item.get('date_str', '')} • {item.get('size_str', '')}"
            meta_lbl = tk.Label(hdr_row, text=meta_str, bg=THEME["widget_bg_color"], 
                                fg="#888888", font=self.fonts["ui_small"])
            meta_lbl.pack(side=tk.RIGHT, padx=4)
            self._bind_targeted_history_scroll(meta_lbl, canvas)

            # Deep Search Snippet Preview (if matched)
            if item.get("snippet"):
                snip_frame = tk.Frame(card, bg="#1a2332", bd=0)
                snip_frame.pack(fill=tk.X, padx=8, pady=(0, 6))
                self._bind_targeted_history_scroll(snip_frame, canvas)
                snip_lbl = tk.Label(snip_frame, text=f"💬 {item['snippet']}", bg="#1a2332", 
                                    fg="#70a5ff", font=self.fonts["ui_small"], anchor="w", justify="left")
                snip_lbl.pack(side=tk.LEFT, padx=6, pady=3, fill=tk.X, expand=True)
                self._bind_targeted_history_scroll(snip_lbl, canvas)

            # Click binding across entire card
            card_path = item["path"]
            card.bind("<Button-1>", lambda e, p=card_path: self._load_selected_history(p, search_query=q))
            name_lbl.bind("<Button-1>", lambda e, p=card_path: self._load_selected_history(p, search_query=q))
            badge.bind("<Button-1>", lambda e, p=card_path: self._load_selected_history(p, search_query=q))
            meta_lbl.bind("<Button-1>", lambda e, p=card_path: self._load_selected_history(p, search_query=q))

    def _bind_targeted_history_scroll(self, widget, canvas):
        """Recursively binds targeted mousewheel scrolling so it never bleeds into outer widgets."""
        def _on_enter(e):
            canvas.bind_all("<MouseWheel>", lambda evt: canvas.yview_scroll(int(-1 * (evt.delta / 120)), "units"))
            canvas.bind_all("<Button-4>", lambda evt: canvas.yview_scroll(-1, "units"))
            canvas.bind_all("<Button-5>", lambda evt: canvas.yview_scroll(1, "units"))

        def _on_leave(e):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        widget.bind("<Enter>", _on_enter, add="+")
        widget.bind("<Leave>", _on_leave, add="+")

    def _get_all_history_entries(self):
        """Scans User History directory and parses metadata and date categories."""
        import datetime
        history_dir = self.get_user_history_dir() if hasattr(self, 'get_user_history_dir') else self.dirs.get("History")
        if not history_dir or not os.path.exists(history_dir):
            return []

        entries = []
        now = datetime.datetime.now()
        today = now.date()

        try:
            files = [f for f in os.listdir(history_dir) if f.endswith(".history.jsonz") or f.endswith(".history.encz")]
        except Exception:
            return []

        for f in files:
            full_path = os.path.join(history_dir, f)
            try:
                mtime = os.path.getmtime(full_path)
                size_bytes = os.path.getsize(full_path)
            except Exception:
                mtime = 0
                size_bytes = 0

            # Level extraction via regex (supporting both .jsonz and .encz)
            lvl = None
            match = re.search(r"_lvl(\d+)\.history\.(?:jsonz|encz)$", f)
            if match:
                try: lvl = int(match.group(1))
                except: lvl = None

            # Clean Display Name
            display_name = re.sub(r"_lvl\d+\.history\.(?:jsonz|encz)$", "", f).replace("-", " ").replace("_", " ")
            if f.endswith(".encz"):
                display_name = f"🔒 {display_name}"

            # Date calculation & grouping bucket
            dt = datetime.datetime.fromtimestamp(mtime)
            file_date = dt.date()
            diff_days = (today - file_date).days

            if diff_days == 0:
                bucket = "Today"
            elif diff_days == 1:
                bucket = "Yesterday"
            elif diff_days <= 7:
                bucket = "Past 7 Days"
            elif diff_days <= 30:
                bucket = "Past 30 Days"
            else:
                bucket = "Older"

            # Formatted size
            if size_bytes >= 1024 * 1024:
                size_str = f"{size_bytes / (1024*1024):.1f} MB"
            elif size_bytes >= 1024:
                size_str = f"{size_bytes / 1024:.1f} KB"
            else:
                size_str = f"{size_bytes} B"

            date_str = dt.strftime("%b %d, %Y • %I:%M %p")

            entries.append({
                "path": full_path,
                "filename": f,
                "display_name": display_name,
                "level": lvl,
                "mtime": mtime,
                "datetime": dt,
                "date_str": date_str,
                "size_bytes": size_bytes,
                "size_str": size_str,
                "date_bucket": bucket
            })

        return entries

    def _on_history_dropdown_change(self, key, value):
        self.history_state[key] = value
        self._render_history_menu()

    def _clear_history_search(self, search_var):
        search_var.set("")
        self.history_state["search_query"] = ""
        self.history_state["deep_search_matches"] = {}
        self._render_history_menu()

    def _trigger_history_search(self, query):
        """Triggers real-time title filter and launches background deep JSONZ/ENCZ full-text search."""
        self._render_history_menu()
        if len(query) >= 2:
            threading.Thread(target=self._async_deep_history_search, args=(query,), daemon=True).start()

    def _async_deep_history_search(self, query):
        """Deep Full-Text Search inside raw .history.jsonz and .history.encz message contents."""
        history_dir = self.get_user_history_dir() if hasattr(self, 'get_user_history_dir') else self.dirs.get("History")
        if not history_dir or not os.path.exists(history_dir): return

        matches = {}
        q_lower = query.lower()

        try:
            files = [os.path.join(history_dir, f) for f in os.listdir(history_dir) if f.endswith(".history.jsonz") or f.endswith(".history.encz")]
        except: return

        for path in files:
            # Check cached content or decompress/decrypt via vault_manager
            msgs = self._history_content_cache.get(path)
            if msgs is None:
                try:
                    if hasattr(self, 'vault_manager'):
                        msgs = self.vault_manager.read_history_messages(path)
                    else:
                        import zlib, json
                        with open(path, "rb") as fp:
                            msgs = json.loads(zlib.decompress(fp.read()).decode("utf-8"))
                    self._history_content_cache[path] = msgs
                except Exception:
                    continue

            # Search within dialogue contents
            for m in msgs:
                content = str(m.get("content", ""))
                idx = content.lower().find(q_lower)
                if idx != -1:
                    start = max(0, idx - 30)
                    end = min(len(content), idx + len(query) + 40)
                    snippet = ("..." if start > 0 else "") + content[start:end].replace("\n", " ") + ("..." if end < len(content) else "")
                    matches[path] = f"[{m.get('role', 'msg').capitalize()}]: {snippet}"
                    break

        self.root.after(0, self._on_deep_search_completed, query, matches)

    def _on_deep_search_completed(self, query, matches):
        if self.history_state.get("search_query", "").strip() == query.strip():
            self.history_state["deep_search_matches"] = matches
            self._render_history_menu()

    def _back_history(self):
        self.history_state["view"] = "list"
        self._render_history_menu()

    def _load_selected_history(self, path, search_query=None):
        """Loads selected archive into content text view with search term highlighting and vault support."""
        fname = os.path.basename(path)
        display_name = re.sub(r"_lvl\d+\.history\.(?:jsonz|encz)$", "", fname).replace("-", " ").replace("_", " ")
        if fname.endswith(".encz"):
            display_name = f"🔒 {display_name}"
        
        self.history_state["view"] = "content"
        self.history_state["current_path"] = path
        self.history_state["current_display_name"] = display_name
        
        self.past_history_view.config(state='normal')
        self.past_history_view.delete('1.0', tk.END)
        self.past_history_view.tag_config("search_match", background="#005a9e", foreground="white")
        
        try:
            if path in self._history_content_cache:
                msgs = self._history_content_cache[path]
            else:
                if hasattr(self, 'vault_manager'):
                    msgs = self.vault_manager.read_history_messages(path)
                else:
                    import zlib, json
                    with open(path, 'rb') as f: 
                        msgs = json.loads(zlib.decompress(f.read()).decode('utf-8'))
                self._history_content_cache[path] = msgs
            
            for m in msgs: 
                who = "You" if m['role'] == 'user' else ("Cecilia" if (m.get('role') == 'cecilia' or m.get('persona') == 'Cecilia') else self._get_persona_label())
                tag = "user" if m['role'] == 'user' else "ai"
                content = self._clean_latex_artifacts(m['content'])
                entry = f"{who}: {content}\n{'-'*50}\n\n"
                self.past_history_view.insert(tk.END, entry, (tag,))
            
            # Highlight search occurrences if searching
            if search_query and len(search_query) >= 2:
                start_pos = '1.0'
                first_match = None
                while True:
                    start_pos = self.past_history_view.search(search_query, start_pos, stopindex=tk.END, nocase=True)
                    if not start_pos: break
                    if not first_match: first_match = start_pos
                    end_pos = f"{start_pos}+{len(search_query)}c"
                    self.past_history_view.tag_add("search_match", start_pos, end_pos)
                    start_pos = end_pos
                if first_match:
                    self.past_history_view.see(first_match)
                else:
                    self.past_history_view.yview_moveto(0.0)
            else:
                self.past_history_view.yview_moveto(0.0)
        except PermissionError:
            self.past_history_view.insert(tk.END, "🔒 This history file is encrypted.\n\nPlease unlock the Serenity Vault to view this archive.")
            self.show_vault_unlock_modal(on_unlock_callback=lambda: self._load_selected_history(path, search_query))
        except Exception as e:
            self.past_history_view.insert(tk.END, f"Error loading history: {e}")
            
        self.past_history_view.config(state='disabled')
        self._render_history_menu()

    def _delete_current_archive(self):
        """Permanently delete the current history file and back out."""
        path = self.history_state.get("current_path")
        if not path or not os.path.exists(path):
            messagebox.showerror("Error", "History file not found.")
            return
            
        confirm = messagebox.askyesno("Delete Archive", 
                                     f"Are you sure you want to permanently delete this history archive?\n\nFile: {os.path.basename(path)}")
        if confirm:
            try:
                os.remove(path)
                if hasattr(self, '_history_content_cache') and path in self._history_content_cache:
                    del self._history_content_cache[path]
                print(f"[SYSTEM] Deleted history file: {path}")
                self._back_history()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete file: {e}")

    def _toggle_history_edit(self):
        """Toggle edit mode for the history view."""
        current_state = self.past_history_view.cget("state")
        if current_state == "disabled":
            self.past_history_view.config(state="normal")
            self._render_history_menu()
        else:
            self._save_history_edits()
            self.past_history_view.config(state="disabled")
            self._render_history_menu()

    def _save_history_edits(self):
        """Parse the edited text and save it back to the history file via VaultManager."""
        path = self.history_state.get("current_path")
        if not path or not os.path.exists(path): return
        
        raw_text = self.past_history_view.get("1.0", tk.END).strip()
        if not raw_text: return
        
        separator = "-" * 50
        chunks = raw_text.split(f"\n{separator}\n\n")
        
        new_msgs = []
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk: continue
            
            role = "user"
            content = chunk
            if chunk.startswith("You: "):
                role = "user"
                content = chunk[5:]
            elif chunk.startswith("Serenity: "):
                role = "assistant"
                content = chunk[10:]
            elif chunk.startswith("Cecilia: "):
                role = "assistant"
                content = chunk[9:]
            elif chunk.startswith("System: "):
                role = "system"
                content = chunk[8:]
            elif ": " in chunk[:20]:
                parts = chunk.split(": ", 1)
                content = parts[1]
                if parts[0].lower() in ["you", "user"]: role = "user"
                elif parts[0].lower() in ["serenity", "cecilia", "assistant", "ai"]: role = "assistant"
                elif parts[0].lower() == "system": role = "system"
                
            new_msgs.append({"role": role, "content": content.strip()})
            
        if not new_msgs: return
        
        try:
            if hasattr(self, 'vault_manager'):
                self.vault_manager.write_history_messages(path, new_msgs)
            else:
                import zlib, json
                compressed_data = zlib.compress(json.dumps(new_msgs).encode('utf-8'))
                with open(path, 'wb') as f:
                    f.write(compressed_data)
            if hasattr(self, '_history_content_cache'):
                self._history_content_cache[path] = new_msgs
            print(f"[SYSTEM] Saved edits to history: {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save edits: {e}")

    def clear_chat_ui(self, preserve_pending=True):
        # Clear the Pinned Prompt
        if hasattr(self, 'prompt_display') and self.prompt_display.winfo_exists():
            self.prompt_display.pack_forget()
            self.prompt_display.config(state='normal')
            self.prompt_display.delete("1.0", tk.END)
            self.prompt_display.config(state='disabled')
            
        # Clear the AI Output
        if hasattr(self, 'chat_history') and self.chat_history:
            try:
                if self.chat_history.winfo_exists():
                    self.chat_history.config(state='normal')
                    self.chat_history.delete('1.0', tk.END)
                    
                    # If we have a pending task with a user message, preserve its display during loading
                    if preserve_pending and getattr(self, 'pending_task', None) and isinstance(self.pending_task, dict):
                        p_msg = self.pending_task.get("message", "")
                        if p_msg:
                            self.chat_history.insert(tk.END, f"\nYou: {p_msg}\n", ("user",))
                            
                    self.chat_history.config(state='disabled')
                    self.state["response_started"] = False
            except tk.TclError: pass
            
        # Tab-aware History frame handling
        if getattr(self, 'active_tab', 'active') == 'history':
            if hasattr(self, 'history_menu_frame') and self.history_menu_frame:
                if not self.history_menu_frame.winfo_viewable():
                    self.history_menu_frame.pack(side=tk.TOP, fill="both", expand=True, padx=2, pady=2)
                self._render_history_menu()
        else:
            if hasattr(self, 'history_menu_frame') and self.history_menu_frame:
                self.history_menu_frame.pack_forget()
            if hasattr(self, 'chat_history') and self.chat_history and not self.chat_history.winfo_viewable():
                self.chat_history.pack(side=tk.TOP, fill="both", expand=True, padx=2, pady=0)

    def _setup_logs_and_stats(self):
        if self.right_panel is None: return
        self.log_container = tk.Frame(self.right_panel, bg=THEME["bg_color"])
        self.log_container.grid_rowconfigure(1, weight=1); self.log_container.grid_columnconfigure(0, weight=1)
        
        header = tk.Frame(self.log_container, bg=THEME["bg_color"])
        self.log_header_frame = header
        header.grid(row=0, column=0, sticky="ew")
        self.log_header_label = tk.Label(header, text="Backend Logs", font=self.fonts["italic"], bg=THEME["bg_color"], fg=THEME["electric_blue"])
        self.log_header_label.pack(side=tk.LEFT)
        
        self.self_analysis_btn = tk.Label(header, text="🔍", font=self.fonts["log_bold"], bg=THEME["bg_color"], fg=THEME["electric_blue"], cursor="hand2")
        self.self_analysis_btn.pack(side=tk.LEFT, padx=(12, 4))
        self.self_analysis_btn.bind("<Button-1>", lambda e: self._run_self_analysis())
        ToolTip(self.self_analysis_btn, "Run Serenity Self-Analysis diagnosis.", app=self)
        
        self.lock_logout_btn = tk.Label(header, text="🔒", font=self.fonts["log_bold"], bg=THEME["bg_color"], fg=THEME["electric_blue"], cursor="hand2")
        self.lock_logout_btn.pack(side=tk.LEFT, padx=4)
        self.lock_logout_btn.bind("<Button-1>", lambda e: self.lock_and_logout())
        ToolTip(self.lock_logout_btn, "Lock & Logout active user profile.", app=self)
        
        self.log_switch_canvas = tk.Canvas(header, width=104, height=28, bg=THEME["bg_color"], highlightthickness=0)
        self.log_switch_canvas.pack(side=tk.RIGHT, padx=(2, 5))
        
        self.clear_log_btn = tk.Label(header, text="🗑", font=self.fonts["log"], bg=THEME["bg_color"], fg=THEME["electric_blue"], cursor="hand2")
        self.clear_log_btn.pack(side=tk.RIGHT, padx=(5, 2))
        self.clear_log_btn.bind("<Button-1>", self._clear_active_log)
        if self.log_switch_canvas is not None:
            self.log_switch_canvas.create_rectangle(2, 2, 102, 26, outline=THEME["electric_blue"], width=2, fill=THEME["widget_bg_color"])
            self.switch_knob = int(self.log_switch_canvas.create_rectangle(2, 2, 30, 26, fill=THEME["electric_blue"]))
            self.log_switch_canvas.create_text(16, 14, text="🗨", fill=THEME["bg_color"])
            self.log_switch_canvas.create_text(40, 14, text="🛠", fill=THEME["electric_blue"])
            self.log_switch_canvas.create_text(64, 14, text="⚠", fill=THEME["electric_blue"])
            self.log_switch_canvas.create_text(88, 14, text="🔍", fill=THEME["electric_blue"])
            self.log_switch_canvas.bind("<Button-1>", self._flip_log_view)

        self.log_frame = tk.Frame(self.log_container, bg=THEME["bg_color"])
        self.log_frame.grid(row=1, column=0, sticky="nsew")
        self.log_frame.grid_rowconfigure(0, weight=1); self.log_frame.grid_columnconfigure(0, weight=1)

        self.thought_log = scrolledtext.ScrolledText(self.log_frame, font=self.fonts["log"], bg=THEME["widget_bg_color"], fg="#cccccc", relief=tk.FLAT)
        self.thought_log.grid(row=0, column=0, sticky="nsew")
        self.thought_log.tag_config("stdout", foreground="#cccccc")
        self.thought_log.tag_config("system", foreground=THEME["electric_blue"], font=self.fonts["log_bold"])
        
        self.error_log = scrolledtext.ScrolledText(self.log_frame, font=self.fonts["log"], bg=THEME["widget_bg_color"], fg="#ff8a8a", relief=tk.FLAT)
        self.error_log.grid(row=0, column=0, sticky="nsew")
        self.error_log.tag_config("stderr", foreground="#ff8a8a")
        self.error_log.grid_remove()

        self.tool_log = scrolledtext.ScrolledText(self.log_frame, font=self.fonts["log"], bg=THEME["widget_bg_color"], fg="#00ffcc", relief=tk.FLAT)
        self.tool_log.grid(row=0, column=0, sticky="nsew")
        self.tool_log.grid_remove()
        
        self.diag_log = scrolledtext.ScrolledText(self.log_frame, font=self.fonts["log"], bg=THEME["widget_bg_color"], fg="#ffa500", relief=tk.FLAT)
        self.diag_log.grid(row=0, column=0, sticky="nsew")
        self.diag_log.tag_config("diag", foreground="#ffa500")
        self.diag_log.grid_remove()
        
        self.stats_frame = tk.Frame(self.log_container, bg=THEME["widget_bg_color"])
        if self.stats_frame is not None:
            self.stats_frame.grid(row=2, column=0, sticky="nsew", pady=4)
        self.stats_labels = {}
        self.stats_row_frames = []
        self.stats_title_labels = []
        
        # Grid Layout: Left Column (GPU/VRAM) | Right Column (System/CPU)
        stats_to_show = [
            ("GPU Use", "GPU Use"), ("CPU", "CPU Use"),
            ("VRAM", "VRAM"), ("Total VRAM", "Total VRAM"),
            ("Shared VRAM", "Shared VRAM"), ("RAM", "Total RAM"),
            ("GPU Temp", "GPU Temp"), ("CPU Temp", "CPU Temp"),
            ("Power", "GPU Power"), ("CPU Power", "CPU Power")
        ] if SYSTEM_MONITOR_LOADED else [("CPU", "CPU"), ("RAM", "RAM")]
        
        for i, (key, label) in enumerate(stats_to_show):
            row = i // 2
            col = i % 2
            f = tk.Frame(self.stats_frame, bg=THEME["widget_bg_color"])
            f.grid(row=row, column=col, sticky="ew", padx=6, pady=1)
            self.stats_frame.columnconfigure(col, weight=1)
            self.stats_row_frames.append(f)
            
            lbl_title = tk.Label(f, text=f"{label}:", bg=THEME["widget_bg_color"], fg=THEME["fg_color"], font=self.fonts["stats"])
            lbl_title.pack(side=tk.LEFT)
            self.stats_title_labels.append(lbl_title)

            self.stats_labels[key] = tk.Label(f, text="N/A", bg=THEME["widget_bg_color"], fg=THEME["electric_blue"], font=self.fonts["stats_bold"])
            self.stats_labels[key].pack(side=tk.RIGHT)
        
        if not SYSTEM_MONITOR_LOADED: self.stats_frame.grid_remove()

        # Fix Grid Weights: Row 1 (Logs) should grow, not Row 0 (Header)
        self.log_container.rowconfigure(0, weight=0)
        self.log_container.rowconfigure(1, weight=1)
        self.log_container.columnconfigure(0, weight=1)

    def _flip_log_view(self, e):
        canvas_sw = self.log_switch_canvas
        if canvas_sw is None: return
        
        # Positional Click Logic: 
        # width=104. 0-25 (Thought), 26-51 (Tool), 52-77 (Error), 78-104 (Diag)
        x = e.x
        target = "thought"
        if x >= 78: target = "diag"
        elif x >= 52: target = "error"
        elif x >= 26: target = "tool"
        
        # If already on the target, just ensure it's displayed (refresh)
        # Otherwise, switch state.
        self.state["log_view"] = target
        
        # UI Update Logic
        if target == "thought":
            if self.error_log is not None: self.error_log.grid_remove()
            if self.tool_log is not None: self.tool_log.grid_remove()
            if self.diag_log is not None: self.diag_log.grid_remove()
            if self.stats_frame is not None: self.stats_frame.grid_remove()
            if self.thought_log is not None: self.thought_log.grid()
            knob = self.switch_knob
            if knob is not None:
                try:
                    canvas_sw.moveto(knob, 2, 2)
                    canvas_sw.itemconfig(knob, fill=THEME["electric_blue"]) # Knob
                    canvas_sw.itemconfig(3, fill=THEME["bg_color"])         # Icon 🗨
                    canvas_sw.itemconfig(4, fill=THEME["electric_blue"])   # Icon 🛠
                    canvas_sw.itemconfig(5, fill=THEME["electric_blue"])   # Icon ⚠
                    canvas_sw.itemconfig(6, fill=THEME["electric_blue"])   # Icon 🔍
                except: pass
        elif target == "tool":
            if self.thought_log is not None: self.thought_log.grid_remove()
            if self.error_log is not None: self.error_log.grid_remove()
            if self.diag_log is not None: self.diag_log.grid_remove()
            if self.stats_frame is not None: self.stats_frame.grid_remove()
            if self.tool_log is not None: self.tool_log.grid()
            knob = self.switch_knob
            if knob is not None:
                try:
                    canvas_sw.moveto(knob, 26, 2)
                    canvas_sw.itemconfig(knob, fill=THEME["electric_blue"]) # Knob
                    canvas_sw.itemconfig(3, fill=THEME["electric_blue"])   # Icon 🗨
                    canvas_sw.itemconfig(4, fill=THEME["bg_color"])         # Icon 🛠
                    canvas_sw.itemconfig(5, fill=THEME["electric_blue"])   # Icon ⚠
                    canvas_sw.itemconfig(6, fill=THEME["electric_blue"])   # Icon 🔍
                except: pass
        elif target == "error":
            if self.thought_log is not None: self.thought_log.grid_remove()
            if self.tool_log is not None: self.tool_log.grid_remove()
            if self.diag_log is not None: self.diag_log.grid_remove()
            if self.stats_frame is not None: self.stats_frame.grid_remove()
            if self.error_log is not None: self.error_log.grid()
            knob = self.switch_knob
            if knob is not None:
                try:
                    canvas_sw.moveto(knob, 50, 2)
                    canvas_sw.itemconfig(knob, fill=THEME["electric_blue"]) # Knob
                    canvas_sw.itemconfig(3, fill=THEME["electric_blue"])   # Icon 🗨
                    canvas_sw.itemconfig(4, fill=THEME["electric_blue"])   # Icon 🛠
                    canvas_sw.itemconfig(5, fill=THEME["bg_color"])         # Icon ⚠
                    canvas_sw.itemconfig(6, fill=THEME["electric_blue"])   # Icon 🔍
                except: pass
        elif target == "diag":
            if self.thought_log is not None: self.thought_log.grid_remove()
            if self.tool_log is not None: self.tool_log.grid_remove()
            if self.error_log is not None: self.error_log.grid_remove()
            if self.diag_log is not None: self.diag_log.grid()
            if self.stats_frame is not None: self.stats_frame.grid()
            knob = self.switch_knob
            if knob is not None:
                try:
                    canvas_sw.moveto(knob, 74, 2)
                    canvas_sw.itemconfig(knob, fill=THEME["electric_blue"]) # Knob
                    canvas_sw.itemconfig(3, fill=THEME["electric_blue"])   # Icon 🗨
                    canvas_sw.itemconfig(4, fill=THEME["electric_blue"])   # Icon 🛠
                    canvas_sw.itemconfig(5, fill=THEME["electric_blue"])   # Icon ⚠
                    canvas_sw.itemconfig(6, fill=THEME["bg_color"])         # Icon 🔍
                except: pass

    def _clear_active_log(self, e=None):
        target = self.state.get("log_view", "thought")
        widget = getattr(self, f"{target}_log", None)
        if widget:
            try:
                state = widget.cget("state")
                if state != "normal":
                    widget.config(state="normal")
                widget.delete("1.0", tk.END)
                if state != "normal":
                    widget.config(state=state)
            except Exception:
                pass

    def _animate_text_fade(self, tag, start_color, end_color, steps=20, current=0):
        if current > steps: return
        color = self._lerp_color(start_color, end_color, current / steps)
        try:
            if self.chat_history is not None:
                self.chat_history.tag_config(tag, foreground=color)
            self.root.after(20, lambda *args: self._animate_text_fade(tag, start_color, end_color, steps, current + 1))
        except: pass

    def _lerp_color(self, c1, c2, t):
        """Linearly interpolate between two hex colors."""
        try:
            def to_rgb(c):
                # Handle hex or color name
                rgb = self.root.winfo_rgb(c)
                return (rgb[0]//256, rgb[1]//256, rgb[2]//256)
            
            r1, g1, b1 = to_rgb(c1)
            r2, g2, b2 = to_rgb(c2)
            
            r = int(r1 + (r2 - r1) * t)
            g = int(g1 + (g2 - g1) * t)
            b = int(b1 + (b2 - b1) * t)
            
            return f"#{max(0, min(255, r)):02x}{max(0, min(255, g)):02x}{max(0, min(255, b)):02x}"
        except:
            return c2 # Fallback to target color on error

    def _animate_avatar_crossfade(self, start_key, end_key, steps=15, current=0):
        canvas_idx = self.right_panel
        img_item = self.avatar_image_item
        if current > steps:
            if canvas_idx is not None and img_item is not None and end_key in self.avatar_states:
                canvas_idx.itemconfig(img_item, image=self.avatar_states[end_key])
            return

        alpha = current / steps
        try:
            img1 = self.avatar_pil_images.get(start_key)
            img2 = self.avatar_pil_images.get(end_key)
            if img1 and img2:
                blended = Image.blend(img1, img2, alpha)
                tk_img = ImageTk.PhotoImage(blended)
                if canvas_idx is not None and img_item is not None:
                    canvas_idx.itemconfig(img_item, image=tk_img)
                self.current_trans_img = tk_img # Keep ref
            self.root.after(20, lambda *args: self._animate_avatar_crossfade(start_key, end_key, steps, current + 1))
        except: pass

    # ================= LOGIC & OPERATIONS =================
    def halt_process(self):
        if self.state["running"]: self.stop_process.set(); self._log_and_display("Stopping...")

    def _sync_deep_cook_ui(self):
        """Centralized source of truth for the Deep Cook button appearance."""
        if not hasattr(self, 'deep_thought_button') or self.deep_thought_button is None:
            return
            
        is_on = self.state.get("deep_cook", False)
        behavior = self.state.get("deep_cook_behavior", "toggle")
        
        if behavior == "oneshot":
            # In oneshot mode, the button is a standard trigger, not a toggle
            self.deep_thought_button.config(text="Deep Cook", bg=THEME["button_bg_color"])
        else:
            # In toggle mode, reflect the active/inactive state
            bg = "#280064" if is_on else THEME["button_bg_color"]
            self.deep_thought_button.config(text=f"Deep Cook: {'ON' if is_on else 'OFF'}", bg=bg)
        
        # Ensure the persona label reflects the current mode
        if hasattr(self, 'update_persona_display'):
            self.update_persona_display()

    def toggle_deep_cook_mode(self):
        btn_deep = self.deep_thought_button
        if self.state["deep_cook_behavior"] == "toggle":
            self.state["deep_cook"] = not self.state["deep_cook"]
            is_on = self.state["deep_cook"]
            self._sync_deep_cook_ui()
            self._log_and_display(f"Deep Cook {'ENABLED' if is_on else 'DISABLED'}")
            
            # Strict Override
            if not self.state.get("staged_multimodal"):
                if is_on:
                    self.model_swap(target_tier="deep_cook")
                else:
                    slider = self.depth_slider
                    if slider is not None:
                        level = int(slider.get())
                        tier_map = {1: "fast", 2: "search", 3: "low", 4: "med", 5: "high", 6: "transcendent", 7: "secret"}
                        target_tier = tier_map.get(level, "low")
                        self.model_swap(target_level=level, target_tier=target_tier)
        else:
            if self.state.get("staged_multimodal"):
                self.initiate_vision_deep_cook()
            else:
                self.send_deep_cook_message()

    def initiate_vision_deep_cook(self):
        """Triggers in-depth multimodal analysis."""
        if self.state["running"]: return
        
        staged = self.state["staged_multimodal"]
        if not staged: return
        
        ui_input = self.user_input
        if ui_input is None: return
        user_msg = ui_input.get("1.0", tk.END).strip()
        if not user_msg: user_msg = "Perform an in-depth analysis of this media."
        
        final_query = VisionHandler.prepare_vision_query(user_msg, is_deep_cook=True)
        
        # Dual-VLM routing (Video vs Image)
        staged_type = staged.get("type", "image") if isinstance(staged, dict) else "image"
        target_tier = "vision_video_deep" if staged_type == "video" else "vision_multimodal"
            
        self.state["last_vision_intent"] = target_tier
        
        if self.current_model_tier != target_tier:
            if not self.model_paths.get(target_tier):
                messagebox.showerror("Error", f"Vision model for {target_tier} not set!")
                return
            self._log_and_display(f"Switching to Deep Vision Engine...")
            self.pending_task = {"type": "vision_deep", "message": final_query, "staged": staged}
            self.model_swap(target_tier=target_tier)
            return

        self._execute_vision_deep_cook(staged, final_query)



    def _execute_vision_deep_cook(self, staged, user_msg):
        ui_input = self.user_input
        if ui_input is not None:
            ui_input.delete("1.0", tk.END)
        self._display_user_message("[Deep Vision] Running in-depth analysis...")
        
        vision_model = self.model_paths.get(f"vision_{staged['type']}")
        model_name = os.path.basename(vision_model) if vision_model else "No Model Set"
        
        print(f"[SYSTEM] In-depth Vision analysis activated for: {staged['path']}")
        print(f"[SYSTEM] Model: {model_name}")
        print(f"[SYSTEM] User Query: {user_msg}")
        
        # Multi-step simulation in logs
        self._log_and_display("Beginning Multi-Step Vision Reasoning...")
        
        queue = self.state.get("processing_queue", [])
        if not queue: queue = [staged["path"]]
        
        threading.Thread(target=self._vision_deep_worker, args=(staged, user_msg), daemon=True).start()
        self.root.after(100, self.check_process_queue)

    def _vision_deep_worker(self, staged, user_msg):
        """Simulation of deep vision steps for Auditor."""
        try:
            steps = ["Segmentating Scene...", "Extracting Features...", "Cross-Referencing Context...", "Synthesizing Logical Narrative..."]
            for step in steps:
                time.sleep(1)
                self.process_queue.put({"status": "thinking_status", "content": f"Thinking... ({step})"})
                self.process_queue.put({"status": "log_update", "content": f"\n[AUDITOR] {step}\n"})
            
            # Final result stub (real inference could go here if model is capable)
            self.process_queue.put({"status": "success", "content": f"Grandmaster Audit of {staged['path']} complete.\n[Audit Log]: {user_msg[:60]}..."})
            
            # Dual-VLM: "Execute analysis, then immediately offload the vision model"
            self.process_queue.put({"status": "vision_oneshot_finish"})
        except Exception as e:
            self.process_queue.put({"status": "error", "content": str(e)})

    def _find_projector_for_model(self, model_path=None):
        m_path = model_path or self.model_path
        if not m_path:
            return None
        model_dir = os.path.dirname(m_path)
        if os.path.exists(model_dir):
            for f in os.listdir(model_dir):
                fl = f.lower()
                if fl.endswith(".mmproj") or ("mmproj" in fl and fl.endswith(".gguf")) or ("projector" in fl and fl.endswith(".gguf")):
                    return os.path.join(model_dir, f)
        for proj_key in ["vision_multimodal_projector", "vision_video_projector", "vision_video_deep_projector"]:
            p = self.model_paths.get(proj_key)
            if p and os.path.exists(p):
                return p
        return None

    def _ensure_chat_handler(self):
        if not self.model:
            return False
        if getattr(self.model, "chat_handler", None) is not None:
            return True
        proj_path = self._find_projector_for_model()
        if proj_path and os.path.exists(proj_path):
            try:
                from llama_cpp.llama_chat_format import Llava15ChatHandler
                chat_handler = Llava15ChatHandler(clip_model_path=proj_path, verbose=True)
                is_gemma_family = "gemma" in (self.model_path or "").lower()
                if is_gemma_family:
                    chat_handler.CHAT_FORMAT = (
                        "{% for message in messages %}"
                        "{% if message.role == 'system' %}"
                        "<|turn>system\n{{ message.content }}<turn|>\n"
                        "{% endif %}"
                        "{% if message.role == 'user' %}"
                        "<|turn>user\n"
                        "{% if message.content is string %}"
                        "{{ message.content }}"
                        "{% endif %}"
                        "{% if message.content is iterable %}"
                        "{% for content in message.content %}"
                        "{% if content.type == 'image_url' and content.image_url is string %}"
                        "{{ content.image_url }}"
                        "{% endif %}"
                        "{% if content.type == 'image_url' and content.image_url is mapping %}"
                        "{{ content.image_url.url }}"
                        "{% endif %}"
                        "{% endfor %}"
                        "{% for content in message.content %}"
                        "{% if content.type == 'text' %}"
                        "{{ content.text }}"
                        "{% endif %}"
                        "{% endfor %}"
                        "{% endif %}"
                        "<turn|>\n"
                        "{% endif %}"
                        "{% if message.role == 'assistant' and message.content is not none %}"
                        "<|turn>model\n{{ message.content }}<turn|>\n"
                        "{% endif %}"
                        "{% endfor %}"
                        "{% if add_generation_prompt %}"
                        "<|turn>model\n"
                        "{% endif %}"
                    )
                self.model.chat_handler = chat_handler
                print(f"[APEX] Dynamically loaded Vision Projector: {os.path.basename(proj_path)}")
                return True
            except Exception as e:
                print(f"[APEX] Failed to dynamically load vision projector '{proj_path}': {e}")
                return False
        return False

    def model_swap(self, value=None, target_level=None, target_tier=None):
        self.halt_process()
        
        # If we are swapping to a text-based tier, clear vision intentions
        if target_tier not in ["vision_video", "vision_video_deep"] and (target_level is not None or value is not None):
             if self.state.get("staged_multimodal"):
                  self._reset_multimodal_ui()

        raw_val = int(value) if value else (target_level if target_level else self.depth_slider.get() if self.depth_slider else 3)
        level = raw_val
        
        if not target_tier or (level == 6 and target_tier == "deep_cook"):
            tier_map = {1: "fast", 2: "search", 3: "low", 4: "med", 5: "high", 6: "transcendent", 7: "secret"}
            target_tier = tier_map.get(level, "low")

        path = self.model_paths.get(target_tier)
        if path and not os.path.isabs(path):
            path = os.path.join(self.script_dir, path)
            
        if not path or not os.path.exists(path):
            msg = f"Model for {target_tier.upper()} tier not found at:\n{path or 'Not Set'}"
            self._log_and_display(msg)
            messagebox.showwarning("Model Missing", f"{msg}\n\nPlease update the path in Settings.")
            self.open_settings_window()
            return

        self.load_params(target_tier)
        if self.current_model_tier == target_tier and self.model and getattr(self, "loaded_persona_level", -1) == level:
            self._log_and_display("Model and Persona already active."); return

        # --- APEX VRAM SOFT-CLEAR ---
        # If toggling between layers of the exact same model file, keep KV/Logic loaded (exclude Live tier for full purge)
        soft_clear = False
        if self.model and getattr(self, "model_path", None) == path and target_tier not in ("transcendent", "Live"):
            soft_clear = True
            if SYSTEM_MONITOR_LOADED and getattr(self, "gpu_handle", None):
                try:
                    mem = nvidia_ml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    free_mb = mem.free / (1024**2)
                    if free_mb < 600: soft_clear = False # Require 10% 6GB breathing room
                except: pass

        if soft_clear:
            print(f"[APEX] VRAM HEALTHY: Soft-Clearing model instead of hard purge.")
            if self.model: self.save_history()
            
            self.current_model_tier = target_tier
            self.active_persona_level = int(level)
            self._log_and_display(f"Swapped to {target_tier.upper()} tier (Soft Clear)")
            if TORCH_AVAILABLE:
                try: 
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                except: pass
            import gc; gc.collect()
            
            self.clear_chat_ui()
            self.stop_process.clear(); self.state["running"] = False
            self.process_queue.put({"status": "load_success", "model": self.model, "level": level, "tier": target_tier})
            return

        if self.model: self.save_history()
        self.model_path = path
        self.current_model_tier = target_tier
        self.active_persona_level = int(level)
        self._log_and_display(f"Swapping to {target_tier.upper()} tier...")
        
        self.set_ui_state(model_loaded=False, loading=True)
        # --- APEX VRAM EVACUATION ---
        if self.model:
            print("[APEX] Breaking Core Paradox: Purging old model for VRAM evacuation...")
            del self.model
            self.model = None
            import gc
            gc.collect()
            if TORCH_AVAILABLE:
                try: 
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except: pass
            time.sleep(1.0) # Hardware Breath: Allow driver sync
        else:
            self.model = None

        self.clear_chat_ui()
        self.stop_process.clear(); self.state["running"] = True
        
        if self.state.get("last_crash", False):
            self.state["safety_margin_mb"] += 300
            print(f"Self-Healing: Increased Safety Margin to {self.state['safety_margin_mb']}MB")
            self.state["last_crash"] = False

        self.set_avatar_state("subdued")

        threading.Thread(target=self._load_model_worker, args=(level, target_tier), daemon=True).start()
        self.root.after(100, self.check_process_queue)

    def _parse_and_stage_filename_imports(self, user_msg):
        """Parses @[filepath], @filepath, and file:///filepath syntax in user messages and stages valid files."""
        if not user_msg: return user_msg
        pattern = r'(?:@\[(?P<bracket_path>[^\]]+)\]|@(?P<at_path>[^\s,;:]+\.[a-zA-Z0-9]+)|file:///(?P<file_url>[^\s\]\n\r"\'<>]+))'
        
        def replace_match(m):
            raw = m.group(0)
            p = m.group('bracket_path') or m.group('at_path') or m.group('file_url')
            if not p: return raw
            p = p.strip('"\'')
            if p.startswith("file:///"): p = p[8:]
            norm_p = os.path.normpath(p)
            if not os.path.isabs(norm_p):
                norm_p = os.path.abspath(os.path.join(self.script_dir, norm_p))
            
            if os.path.isfile(norm_p):
                ext = os.path.splitext(norm_p)[1].lower()
                if ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']:
                    att_type = "image"
                elif ext in ['.mp3', '.wav', '.ogg', '.flac', '.m4a']:
                    att_type = "audio"
                elif ext in ['.mp4', '.mkv', '.avi', '.mov']:
                    att_type = "video"
                else:
                    att_type = "document"
                
                staged = self.state.get("staged_attachments", [])
                if not any(a["path"] == norm_p for a in staged):
                    self._add_staged_attachment(norm_p, att_type)
                return f"[{att_type.capitalize()} Import: {os.path.basename(norm_p)} ({norm_p})]"
            return raw

        return re.sub(pattern, replace_match, user_msg)

    def send_message(self, msg_override=None, skip_swap_check=False):
        if self.state["running"]: return
        
        raw_msg = msg_override if msg_override else self.user_input.get("1.0", tk.END).strip()
        user_msg = self._parse_and_stage_filename_imports(raw_msg)
        
        # 1. Text Documents Injection
        staged_atts = self.state.get("staged_attachments", [])
        doc_atts = [a for a in staged_atts if a["type"] == "document"]
        if doc_atts:
            doc_text = ""
            for doc in doc_atts:
                try:
                    with open(doc["path"], "r", encoding="utf-8", errors="ignore") as f:
                        doc_text += f"\n\n--- Document: {doc['name']} ({doc['path']}) ---\n{f.read()[:50000]}\n" # Safety cap
                except Exception as e:
                    self._log_and_display(f"Could not read {doc['name']}: {e}")
            
            user_msg = f"{doc_text}\n{user_msg}".strip()
            
            # Clear doc tokens from UI since they are consumed instantly
            # Note: A full UI cleanup of just doc tokens can be tricky, so we rely on 
            # general reset or individual frame matching if needed, but for simplicity:
            for d in list(doc_atts):
                if d in self.state["staged_attachments"]:
                    if "token_frame" in d and d["token_frame"].winfo_exists():
                        d["token_frame"].destroy()
                    self.state["staged_attachments"].remove(d)
            if not self.state.get("staged_attachments") and hasattr(self, "attachment_frame"):
                self.attachment_frame.pack_forget()

        # 2. Check for staged multimodal processing (Legacy Video or New Media)
        # Priority: If simple attachments (images/audio/video) are present, use the multimodal engine
        has_media = any(a["type"] in ["image", "audio", "video"] for a in self.state.get("staged_attachments", []))
        
        if has_media or self.state.get("staged_multimodal"):
            if not user_msg: user_msg = "Analyze this media."
            
            image_mode = self.config.get("image_handling", "auto")
            
            # Auto-detect native capability via active or dynamically loadable vision projector
            has_projector_path = bool(self._find_projector_for_model())
            has_inline_vision = (
                self.model is not None and 
                self._ensure_chat_handler()
            )
            
            target_v_tier = f"vision_{self.state['staged_multimodal']['type']}" if self.state.get("staged_multimodal") else "vision_multimodal"
            has_vision_model = bool(self.model_paths.get(target_v_tier))
            
            use_inline = False
            use_vision = False
            
            if image_mode == "native":
                if self.model is not None:
                    use_inline = True
            elif image_mode == "vision":
                if has_vision_model:
                    use_vision = True
            else: # auto
                if has_inline_vision:
                    use_inline = True
                elif has_vision_model:
                    use_vision = True

            if has_media:
                media_pts = [a["path"] for a in self.state.get("staged_attachments", []) if a["type"] in ["image", "audio", "video"]]
                media_names = [a["name"] for a in self.state.get("staged_attachments", []) if a["type"] in ["image", "audio", "video"]]
                
                # Verify and dynamically load or reload vision projector for inline native vision
                if use_inline and self.model is not None and getattr(self.model, "chat_handler", None) is None:
                    if self._ensure_chat_handler():
                        print("[APEX] Dynamically integrated mmproj vision projector.")
                    else:
                        proj_path = self._find_projector_for_model()
                        if proj_path and os.path.exists(proj_path):
                            self._log_and_display("Reloading model to integrate mmproj vision projector...")
                            self.pending_task = {"type": "chat", "message": user_msg}
                            self.model_swap(target_level=self.active_persona_level, target_tier=self.current_model_tier)
                            return
                        else:
                            if has_vision_model:
                                use_inline = False
                                use_vision = True
                            else:
                                messagebox.showerror("Vision Error", "Native vision requested, but no vision projector (.mmproj) was found for the current model.")
                                return

                if not use_inline and not use_vision:
                    if self.model is not None and self._ensure_chat_handler():
                        use_inline = True
                    elif bool(self.model_paths.get("vision_multimodal")):
                        use_vision = True
                    else:
                        messagebox.showerror("Vision Error", "No Vision model or Vision Projector found for image/media processing.\n\nPlease configure 'vision_multimodal' tier in Settings or load a model with a vision projector (.mmproj).")
                        return

                if not msg_override:
                    self.user_input.delete("1.0", tk.END)
                    self.last_user_message = user_msg
                    self._display_user_message(f"[{', '.join(media_names)}] {user_msg}")
                
                if use_inline:
                    self._log_and_display("Handling image inline via loaded persona model...")
                    
                    # Clear attachments now they are bound to a task
                    for a in list(self.state.get("staged_attachments", [])):
                        if a["type"] != "document":
                            if "token_frame" in a and a["token_frame"].winfo_exists():
                                a["token_frame"].destroy()
                            self.state["staged_attachments"].remove(a)
                    if not any(a for a in self.state["staged_attachments"] if a["type"] != "document"):
                       if hasattr(self, "attachment_frame"): self.attachment_frame.pack_forget()

                    self._display_ai_message(is_streaming=True)
                    self.set_avatar_state("meditating")
                    self._prep_generation()
                    
                    # Construct multimodal content
                    content_list = []
                    # Gemma 4 Best Practice: Put images/audio first, then the text instruction
                    for path in media_pts:
                        ext = os.path.splitext(path)[1].lower()
                        fname = os.path.basename(path)
                        if ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']:
                            budget = VisionHandler._determine_visual_budget(user_msg)
                            b64 = VisionHandler.encode_image(path, budget=budget, query=user_msg)
                            if b64:
                                content_list.append({"type": "text", "text": f"[Attached Image: {fname} ({path})]"})
                                content_list.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                                })
                        elif ext in ['.mp3', '.wav', '.ogg', '.flac', '.m4a']:
                            content_list.append({"type": "text", "text": f"[Attached Audio: {fname} ({path})]"})
                            chunks = VisionHandler.get_audio_chunks(path, chunk_length_s=30, max_chunks=30)
                            for chunk in chunks:
                                content_list.append({
                                    "type": "input_audio",
                                    "input_audio": {
                                        "data": chunk,
                                        "format": "wav"
                                    }
                                })
                        elif ext in ['.mp4', '.mkv', '.avi', '.mov']:
                            content_list.append({"type": "text", "text": f"[Attached Video: {fname} ({path})]"})
                            frames = VisionHandler.get_video_sampled_frames(path, target_fps=1.0)
                            for f in frames:
                                content_list.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{f}"}
                                })
                                
                    # Guide prompt for video frames
                    video_staged = any(os.path.splitext(p)[1].lower() in ['.mp4', '.mkv', '.avi', '.mov'] for p in media_pts)
                    modified_user_msg = user_msg
                    if video_staged:
                        modified_user_msg = f"You are looking at a sequence of frames sampled from a video at 1 frame per second. {user_msg}"
                    content_list.append({"type": "text", "text": modified_user_msg})
                                
                    temp_msgs = self.messages + [{"role": "user", "content": content_list}]
                    threading.Thread(target=self._generation_worker, args=(user_msg, temp_msgs), daemon=True).start()
                    self.root.after(100, self.check_process_queue)
                    return

                if use_vision:
                    self.state["last_vision_intent"] = "vision_multimodal"
                    final_query = VisionHandler.prepare_vision_query(user_msg, is_deep_cook=False)
                    self.initiate_vision_analysis("multimodal", media_pts, final_query)
                    return

            elif self.state.get("staged_multimodal"):
                staged = self.state["staged_multimodal"]
                if not msg_override:
                    self.user_input.delete("1.0", tk.END)
                    self.last_user_message = user_msg
                    self._display_user_message(f"[{staged['type'].capitalize()} Request] {user_msg}")
                    
                if use_vision:
                    self.state["last_vision_intent"] = f"vision_{staged['type']}"
                    final_query = VisionHandler.prepare_vision_query(user_msg, is_deep_cook=False)
                    self.initiate_vision_analysis("video", staged["path"], final_query)
                    return

        if self.state["deep_cook"] and not skip_swap_check:
            self.send_deep_cook_message(msg_override); return

        user_msg = msg_override if msg_override else self.user_input.get("1.0", tk.END).strip()
        if not user_msg: return

        if not msg_override:
            self.user_input.delete("1.0", tk.END)
            self.last_user_message = user_msg
            self._display_user_message(user_msg)

        level = self.depth_slider.get()
        tier_map = {1: "fast", 2: "search", 3: "low", 4: "med", 5: "high", 6: "transcendent", 7: "secret"}
        req_tier = tier_map.get(level, "low")
        
        if not skip_swap_check and self.current_model_tier != req_tier:
            self._log_and_display(f"Switching to {req_tier}...")
            self.pending_task = {"type": "chat", "message": user_msg}
            self.model_swap(target_level=level, target_tier=req_tier)
            return

        self._display_ai_message(is_streaming=True)
        self.set_avatar_state("meditating")
        
        self._prep_generation()
        temp_msgs = self.messages + [{"role": "user", "content": user_msg}]
        threading.Thread(target=self._generation_worker, args=(user_msg, temp_msgs), daemon=True).start()
        self.root.after(100, self.check_process_queue)

    def send_deep_cook_message(self, msg_override=None, skip_swap_check=False):
        if self.state["running"]: return
        
        raw_msg = msg_override if msg_override else self.user_input.get("1.0", tk.END).strip()
        user_msg = self._parse_and_stage_filename_imports(raw_msg)
        if not user_msg: return

        if not msg_override:
            self.user_input.delete("1.0", tk.END)
            self.last_user_message = user_msg
            self._display_user_message(user_msg)

        if not skip_swap_check and self.current_model_tier != "deep_cook":
            if not self.model_paths.get("deep_cook"):
                messagebox.showerror("Error", "Deep Cook model not set!")
                # Revert toggle if it was a toggle-based failure
                if self.state["deep_cook_behavior"] == "toggle":
                    self.state["deep_cook"] = False
                    self._sync_deep_cook_ui()
                return
            self._log_and_display("Switching to Deep Cook...")
            self.pending_task = {"type": "deep_cook", "message": user_msg}
            self.model_swap(target_level=self.depth_slider.get(), target_tier="deep_cook")
            return

        self._display_ai_message(is_streaming=True)
        self.set_avatar_state("meditating")
        
        self._prep_generation()
        threading.Thread(target=self._generation_worker_deep_cook, args=(user_msg,), daemon=True).start()
        self.root.after(100, self.check_process_queue)

    def _prep_generation(self):
        self.set_ui_state(model_loaded=True, generating=True)
        self.stop_process.clear()
        self.text_buffer, self.last_update_time = "", 0
        self.state["running"] = True

    def calculate_dynamic_gpu_layers(self, model_path, ctx_size, targeted_reserve_vram_mb=5400):
        if not model_path or not os.path.exists(model_path):
            return 0
        
        # 1. Try to read the block count and MoE metadata
        total_layers = 0
        expert_count = 0
        expert_used_count = 0
        
        # Method A: Try gguf / llama_cpp GGUFReader
        try:
            try:
                from gguf import GGUFReader
                reader = GGUFReader(model_path)
            except Exception:
                from llama_cpp.llama_speculative import LlamaGGUFReader
                reader = LlamaGGUFReader(model_path)
            
            fields = reader.fields.values() if isinstance(getattr(reader, 'fields', None), dict) else getattr(reader, 'fields', [])
            for field in fields:
                field_name = getattr(field, 'name', '') or getattr(field, 'key', '')
                parts = getattr(field, 'parts', [])
                if field_name.endswith(".block_count") and parts:
                    total_layers = int(parts[0][0] if isinstance(parts[0], (list, tuple, np.ndarray)) else parts[0])
                elif field_name.endswith(".expert_count") and parts:
                    expert_count = int(parts[0][0] if isinstance(parts[0], (list, tuple, np.ndarray)) else parts[0])
                elif field_name.endswith(".expert_used_count") and parts:
                    expert_used_count = int(parts[0][0] if isinstance(parts[0], (list, tuple, np.ndarray)) else parts[0])
        except Exception as e:
            print(f"[DYNAMIC AUTO-OFFLOAD] GGUFReader failed: {e}. Falling back to binary parser.")
            
        # Method B: Fallback to binary parser (robust for any Python environment/LlamaCpp-python version)
        if total_layers == 0 or expert_count == 0:
            try:
                with open(model_path, "rb") as f:
                    magic = f.read(4)
                    if magic == b"GGUF":
                        version = struct.unpack("<I", f.read(4))[0]
                        tensor_count = struct.unpack("<Q", f.read(8))[0]
                        kv_count = struct.unpack("<Q", f.read(8))[0]
                        
                        def read_str(file_obj):
                            length = struct.unpack("<Q", file_obj.read(8))[0]
                            return file_obj.read(length).decode("utf-8", errors="ignore")
                            
                        def skip_value(file_obj, val_type):
                            if val_type in [0, 1, 7]: # uint8, int8, bool
                                file_obj.read(1)
                            elif val_type in [2, 3]: # uint16, int16
                                file_obj.read(2)
                            elif val_type in [4, 5, 6]: # uint32, int32, float32
                                file_obj.read(4)
                            elif val_type in [10, 11, 12]: # uint64, int64, float64
                                file_obj.read(8)
                            elif val_type == 8: # string
                                length = struct.unpack("<Q", file_obj.read(8))[0]
                                file_obj.read(length)
                            elif val_type == 9: # array
                                item_type = struct.unpack("<I", file_obj.read(4))[0]
                                array_len = struct.unpack("<Q", file_obj.read(8))[0]
                                for _ in range(array_len):
                                    skip_value(file_obj, item_type)
                            else:
                                raise ValueError(f"Unknown GGUF type: {val_type}")
                                
                        for _ in range(kv_count):
                            key = read_str(f)
                            val_type = struct.unpack("<I", f.read(4))[0]
                            if key.endswith(".block_count"):
                                if val_type == 4:
                                    total_layers = struct.unpack("<I", f.read(4))[0]
                                elif val_type == 5:
                                    total_layers = struct.unpack("<i", f.read(4))[0]
                                elif val_type == 10:
                                    total_layers = struct.unpack("<Q", f.read(8))[0]
                                elif val_type == 11:
                                    total_layers = struct.unpack("<q", f.read(8))[0]
                            elif key.endswith(".expert_count"):
                                if val_type == 4:
                                    expert_count = struct.unpack("<I", f.read(4))[0]
                                elif val_type == 5:
                                    expert_count = struct.unpack("<i", f.read(4))[0]
                                elif val_type == 10:
                                    expert_count = struct.unpack("<Q", f.read(8))[0]
                                elif val_type == 11:
                                    expert_count = struct.unpack("<q", f.read(8))[0]
                            elif key.endswith(".expert_used_count"):
                                if val_type == 4:
                                    expert_used_count = struct.unpack("<I", f.read(4))[0]
                                elif val_type == 5:
                                    expert_used_count = struct.unpack("<i", f.read(4))[0]
                                elif val_type == 10:
                                    expert_used_count = struct.unpack("<Q", f.read(8))[0]
                                elif val_type == 11:
                                    expert_used_count = struct.unpack("<q", f.read(8))[0]
                            else:
                                skip_value(f, val_type)
            except Exception as e:
                print(f"[DYNAMIC AUTO-OFFLOAD] Custom binary parser failed: {e}")

        # Fallback to standard 32 if block_count metadata key differs by architecture
        if total_layers == 0:
            total_layers = 32
            print(f"[DYNAMIC AUTO-OFFLOAD] Falling back to default block count: {total_layers}")

        # 2. Get file footprint (GGUF file size directly maps to model weight RAM usage)
        file_size_bytes = os.path.getsize(model_path)
        model_base_vram_mb = file_size_bytes / (1024 * 1024)  # Convert to MiB
        
        # Calculate the precise weight cost per layer for this specific quant
        vram_per_layer = model_base_vram_mb / total_layers
        
        # 3. Dynamic KV Cache Footprint Estimation (Adjusted for Quantized KV / SWA / Flash Attention)
        # 8-bit/4-bit quantized KV cache and SWA reduce footprint dramatically vs legacy FP16 estimates
        raw_kv_est = (ctx_size / 49152) * 900.0
        kv_cache_vram_mb = max(250.0, min(targeted_reserve_vram_mb * 0.35, raw_kv_est))

        # 4. Math: Allocate remaining VRAM budget to layers
        available_weight_vram = targeted_reserve_vram_mb - kv_cache_vram_mb
        
        if available_weight_vram <= 0:
            print(f"[WARN] Cache footprint ({kv_cache_vram_mb:.1f}MB) saturates VRAM. Offloading 0 layers.")
            return 0
            
        safe_layers = int(available_weight_vram // vram_per_layer)
        final_layers = max(0, min(total_layers, safe_layers))
        
        print("--- DYNAMIC VRAM REPORT ---")
        print(f"Model Detected:   {os.path.basename(model_path)}")
        if expert_count > 0:
            print(f"Model Type:       Mixture of Experts (MoE)")
            print(f"MoE Router Map:   {expert_used_count}/{expert_count} experts active per token")
        else:
            print(f"Model Type:       Dense")
        print(f"Total Layers:     {total_layers}")
        print(f"File/Weight Size: {model_base_vram_mb:.1f} MiB (~{vram_per_layer:.1f} MiB/layer)")
        print(f"Est. KV Cache:    {kv_cache_vram_mb:.1f} MiB")
        print(f"Action:           Offloading {final_layers}/{total_layers} layers to GPU")
        print("----------------------------")
        
        return final_layers

    # ================= WORKERS =================
    def _auto_tune_params(self, level: int, tier: str) -> Dict[str, Any]:
        params_specs: Dict[str, Any] = {"n_ctx": 4096, "n_gpu_layers": 99, "extra_args": {"flash_attn": True}}
        user_layers = self.gpu_layer_config.get(tier, -1)
        manual_vram = self.state.get("virtual_vram", 0)
        
        # Deep cook is a special operational mode, not a standard level
        if tier == "deep_cook":
            params_specs["n_ctx"] = self.context_size_config.get(tier, 2048) 
            params_specs["n_gpu_layers"] = 15
            params_specs["extra_args"]["flash_attn"] = True
        elif tier.startswith("vision_"):
            # Context Expansion: Pull from user settings or fallback to 8192
            params_specs["n_ctx"] = self.context_size_config.get(tier, 8192)
            # APEX SOLO PROJECTOR: Force 0 layers for LLM to reserve VRAM solo for mmproj
            # This ensures the 1.5GB 'Landing Zone' is maintained.
            params_specs["n_gpu_layers"] = 0 
            params_specs["extra_args"]["flash_attn"] = True
        else:
            # Pull directly from configuration maps or overrides
            params_specs["n_ctx"] = self.context_size_config.get(tier, CONTEXT_SIZE_MAP.get(level, 4096))
            
            # Dynamic n_ctx scaling based on HardwareProfile (only for standard tiers)
            ram_gb = HardwareProfile.get_total_ram_gb()
            if ram_gb > 24:
                params_specs["n_ctx"] = max(params_specs["n_ctx"], 8192)
                print(f"[HARDWARE] Apex RAM detected (>24GB). Scaling n_ctx to 8192.")
            elif ram_gb > 16:
                params_specs["n_ctx"] = max(params_specs["n_ctx"], 6144)
                print(f"[HARDWARE] Performance RAM detected (>16GB). Scaling n_ctx to 6144.")
                
        # Handle GPU Layers: Use User Override > Auto-Detected Recommendation > Fallback
        if user_layers != -1:
            params_specs["n_gpu_layers"] = user_layers
        else:
            # For -1 (Auto), use the context-aware recommendations calculated at startup or save
            if hasattr(self, "_auto_detected_layers") and tier in self._auto_detected_layers:
                params_specs["n_gpu_layers"] = self._auto_detected_layers[tier]
            else:
                # If no recommendations, let the engine handle -1 as "all layers"
                params_specs["n_gpu_layers"] = -1
            


        return params_specs

    def _load_model_worker(self, target_level, target_tier):
        try:
            if not self.model_path or not os.path.exists(self.model_path): raise FileNotFoundError(f"No file: {self.model_path}")
            specs = self._auto_tune_params(target_level, target_tier)
            n_layers = specs["n_gpu_layers"]; n_ctx = specs["n_ctx"]; extra = specs["extra_args"]
            
            if self.config.get("auto_vram_offload", False):
                vram_target = self.state.get("virtual_vram", 0)
                if vram_target <= 0:
                    if SYSTEM_MONITOR_LOADED and self.gpu_handle:
                        try:
                            mem = nvidia_ml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                            # Convert total to MB and subtract 1.2GB safe system overhead
                            vram_target = int((mem.total / 1024**2) - 1200)
                        except:
                            vram_target = 5400
                    else:
                        vram_target = 5400
                
                n_layers = self.calculate_dynamic_gpu_layers(self.model_path, n_ctx, targeted_reserve_vram_mb=vram_target)
                specs["n_gpu_layers"] = n_layers

            self.process_queue.put({"status": "diag_log_update", "content": f"[ENGINE] Target: {target_tier} (Level {target_level}) | Layers: {n_layers} | Ctx: {n_ctx} | Extra: {extra}"})
            print(f"Loading {os.path.basename(self.model_path)} | Layers: {n_layers} | Ctx: {n_ctx} | Extra: {extra}")
            
            # Massive Context KV Manager
            if TRI_ATTENTION_ENABLED:
                self.kv_manager = KVManager(max_context_tokens=n_ctx, prune_ratio=TRI_ATTENTION_BUDGET)
            
            chat_handler = None
            # VRAM Diagnostic
            if SYSTEM_MONITOR_LOADED and self.gpu_handle:
                m = nvidia_ml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                print(f"[VRAM] Before Swap: {m.used/1024**2:.0f}MB / {m.total/1024**2:.0f}MB (Free: {m.free/1024**2:.0f}MB)")

            # Scan for adjacent projector in the same directory as the model
            adjacent_proj = None
            if self.model_path:
                model_dir = os.path.dirname(self.model_path)
                if os.path.exists(model_dir):
                    for f in os.listdir(model_dir):
                        fl = f.lower()
                        if fl.endswith(".mmproj") or ("mmproj" in fl and fl.endswith(".gguf")) or ("projector" in fl and fl.endswith(".gguf")):
                            adjacent_proj = os.path.join(model_dir, f)
                            break

            is_gemma_family = "gemma" in self.model_path.lower() if self.model_path else False

            if target_tier.startswith("vision_"):
                proj_path = self.model_paths.get(f"{target_tier}_projector")
                if not proj_path or not os.path.exists(proj_path):
                    proj_path = self._find_projector_for_model(self.model_path)
                
                if proj_path and os.path.exists(proj_path):
                    try:
                        from llama_cpp.llama_chat_format import Llava15ChatHandler
                        chat_handler = Llava15ChatHandler(clip_model_path=proj_path, verbose=True)
                        if is_gemma_family:
                            chat_handler.CHAT_FORMAT = (
                                "{% for message in messages %}"
                                "{% if message.role == 'system' %}"
                                "<|turn>system\n{{ message.content }}<turn|>\n"
                                "{% endif %}"
                                "{% if message.role == 'user' %}"
                                "<|turn>user\n"
                                "{% if message.content is string %}"
                                "{{ message.content }}"
                                "{% endif %}"
                                "{% if message.content is iterable %}"
                                "{% for content in message.content %}"
                                "{% if content.type == 'image_url' and content.image_url is string %}"
                                "{{ content.image_url }}"
                                "{% endif %}"
                                "{% if content.type == 'image_url' and content.image_url is mapping %}"
                                "{{ content.image_url.url }}"
                                "{% endif %}"
                                "{% endfor %}"
                                "{% for content in message.content %}"
                                "{% if content.type == 'text' %}"
                                "{{ content.text }}"
                                "{% endif %}"
                                "{% endfor %}"
                                "{% endif %}"
                                "<turn|>\n"
                                "{% endif %}"
                                "{% if message.role == 'assistant' and message.content is not none %}"
                                "<|turn>model\n{{ message.content }}<turn|>\n"
                                "{% endif %}"
                                "{% endfor %}"
                                "{% if add_generation_prompt %}"
                                "<|turn>model\n"
                                "{% endif %}"
                            )
                        print(f"Vision Projector Loaded: {os.path.basename(proj_path)}")
                    except Exception as e:
                        print(f"Warning: Failed to load vision projector '{proj_path}': {e}")
            else:
                has_active_multimedia = False
                if self.state.get("staged_attachments"):
                    if any(a.get("type") in ["image", "audio", "video"] for a in self.state["staged_attachments"]):
                        has_active_multimedia = True
                if self.state.get("staged_multimodal"):
                    has_active_multimedia = True
                if hasattr(self, "messages") and self.messages:
                    for msg in self.messages:
                        content = msg.get("content")
                        if isinstance(content, list):
                            for part in content:
                                if isinstance(part, dict) and part.get("type") in ["image_url", "input_audio"]:
                                    has_active_multimedia = True
                                    break
                        if has_active_multimedia:
                            break

                proj_path = adjacent_proj or self._find_projector_for_model(self.model_path)
                if proj_path and os.path.exists(proj_path) and has_active_multimedia:
                    try:
                        from llama_cpp.llama_chat_format import Llava15ChatHandler
                        chat_handler = Llava15ChatHandler(clip_model_path=proj_path, verbose=True)
                        if is_gemma_family:
                            chat_handler.CHAT_FORMAT = (
                                "{% for message in messages %}"
                                "{% if message.role == 'system' %}"
                                "<|turn>system\n{{ message.content }}<turn|>\n"
                                "{% endif %}"
                                "{% if message.role == 'user' %}"
                                "<|turn>user\n"
                                "{% if message.content is string %}"
                                "{{ message.content }}"
                                "{% endif %}"
                                "{% if message.content is iterable %}"
                                "{% for content in message.content %}"
                                "{% if content.type == 'image_url' and content.image_url is string %}"
                                "{{ content.image_url }}"
                                "{% endif %}"
                                "{% if content.type == 'image_url' and content.image_url is mapping %}"
                                "{{ content.image_url.url }}"
                                "{% endif %}"
                                "{% endfor %}"
                                "{% for content in message.content %}"
                                "{% if content.type == 'text' %}"
                                "{{ content.text }}"
                                "{% endif %}"
                                "{% endfor %}"
                                "{% endif %}"
                                "<turn|>\n"
                                "{% endif %}"
                                "{% if message.role == 'assistant' and message.content is not none %}"
                                "<|turn>model\n{{ message.content }}<turn|>\n"
                                "{% endif %}"
                                "{% endfor %}"
                                "{% if add_generation_prompt %}"
                                "<|turn>model\n"
                                "{% endif %}"
                            )
                        print(f"Inline Vision Projector Loaded: {os.path.basename(adjacent_proj)}")
                    except Exception as e:
                        print(f"Warning: Failed to load inline vision projector '{adjacent_proj}': {e}")
            
            # --- APEX VRAM SCOUT (1024MB Rule) ---
            if SYSTEM_MONITOR_LOADED and self.gpu_handle and not self.config.get("auto_vram_offload", False):
                try:
                    mem = nvidia_ml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    total_mb = mem.total / (1024**2)
                    free_mb = mem.free / (1024**2)
                    
                    # Reserve 1200MB (1.2GB) safe system overhead estimation
                    vram_buffer = 1200
                    available_for_layers = free_mb - vram_buffer
                    
                    # [VRAM SCOUT] RATIONALE: Keep mmproj on high-speed VRAM to prevent 90s lag.
                    # Enforce a hard cap even for manual settings if they would cause OOM
                    layer_size_mb = 110 if ("12b" in self.model_path.lower() or "moe" in self.model_path.lower()) else 85
                    if available_for_layers > 0:
                        estimated_max = int(available_for_layers / layer_size_mb)
                        if n_layers == -1 or n_layers > estimated_max:
                            if n_layers != -1:
                                print(f"[VRAM SCOUT] Manual setting {n_layers} exceeds safe limit {estimated_max}. Capping.")
                            n_layers = max(0, min(99, estimated_max))
                    else:
                        n_layers = 0
                    
                    # Apply "Layer Drop" offset if active
                    layer_offset = self.state.get("vram_layer_offset", 0)
                    if layer_offset != 0:
                        n_layers = max(0, n_layers + layer_offset)
                        print(f"[APEX] Applying Layer Drop offset: {layer_offset} (Target: {n_layers})")

                    print(f"[VRAM SCOUT] Total: {total_mb:.0f}MB | Free: {free_mb:.0f}MB | Budget: {available_for_layers:.0f}MB | Layers: {n_layers}")
                except Exception as e:
                    print(f"[VRAM SCOUT] Failed: {e}")

            # Clean way to initialize without collisions
            params = extra.copy()
            use_flash = params.pop('flash_attn', True)
            
            # Resolve dynamic KV cache types from config/specs
            # Resolve validated universal KV cache types
            import llama_cpp as lcpp
            cache_map = {
                "f32": getattr(lcpp, "GGML_TYPE_F32", 0),
                "f16": getattr(lcpp, "GGML_TYPE_F16", 1),
                "fp16": getattr(lcpp, "GGML_TYPE_F16", 1),
                "q8_0": getattr(lcpp, "GGML_TYPE_Q8_0", 8),
                "q5_1": getattr(lcpp, "GGML_TYPE_Q5_1", 7),
                "q5_0": getattr(lcpp, "GGML_TYPE_Q5_0", 6),
                "q4_1": getattr(lcpp, "GGML_TYPE_Q4_1", 3),
                "q4_0": getattr(lcpp, "GGML_TYPE_Q4_0", 2),
                "iq4_nl": getattr(lcpp, "GGML_TYPE_IQ4_NL", 20),
            }
            
            # Retrieve independent K/V Cache selections from config
            UNIVERSAL_KV = {"fp16", "f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl", "f32"}
            k_fmt = self.config.get("k_cache_type", params.pop("cache_type_k", "q8_0")).lower()
            v_fmt = self.config.get("v_cache_type", params.pop("cache_type_v", "q8_0")).lower()
            if k_fmt not in UNIVERSAL_KV:
                print(f"[APEX] Warning: Unsupported K cache format '{k_fmt}'. Falling back to 'q8_0'.")
                k_fmt = "q8_0"
            if v_fmt not in UNIVERSAL_KV:
                print(f"[APEX] Warning: Unsupported V cache format '{v_fmt}'. Falling back to 'q8_0'.")
                v_fmt = "q8_0"
            
            t_k = cache_map.get(k_fmt, getattr(lcpp, "GGML_TYPE_Q8_0", 8))
            t_v = cache_map.get(v_fmt, getattr(lcpp, "GGML_TYPE_Q8_0", 8))
            
            is_kv_quantized = (t_k != getattr(lcpp, "GGML_TYPE_F16", 1)) or (t_v != getattr(lcpp, "GGML_TYPE_F16", 1))
            is_gemma_family = "gemma" in (self.model_path.lower() if self.model_path else "")
            is_muse_model = any(k in (self.model_path.lower() if self.model_path else "") for k in ["muse", "glimmer", "onyx", "atem"])
            
            # Flash Attention Logic:
            # - Muse-Glimmer SWA/NoPE hybrid attention requires flash_attn=False on f16 KV to prevent logit corruption.
            # - Quantized KV cache (q8_0, q4_0) strictly requires Flash Attention in llama.cpp to initialize context.
            if is_muse_model and not is_kv_quantized:
                print("[ENGINE] Muse-Glimmer hybrid SWA detected. Disabling Flash Attention for attention logit stability.")
                use_flash = False
            elif is_kv_quantized:
                print("[SYSTEM Note] Quantized KV cache detected. Enforcing Flash Attention ON to ensure context initialization.")
                use_flash = True

            extra["flash_attn"] = use_flash
            print(f"[ENGINE] Final Attention Configuration: Flash Attention = {use_flash} | KV: K={k_fmt}, V={v_fmt}")

            
            hao_preset = self.config.get("hao_preset", "exps=CPU")
            override_tensors = [hao_preset] if hao_preset != "None" else []
            
            swa_kv = self.config.get("swa_kv_cache", "Auto")
            no_kv_offload = True if swa_kv == "CPU Only" else False

            # --- Hardware Priority Lock ---
            HardwareProfile.pin_to_p_cores()
            HardwareProfile.set_priority("above_normal") # ABOVE_NORMAL as per mission

            # --- Wit-Layer: Init ---
            self.process_queue.put({"status": "status_phase", "phase": "loading", "details": f"Initializing {target_tier.upper()} engine..."})
            self.process_queue.put({"status": "thinking_status", "content": f"Loading {target_tier.upper()} engine..."})

            # --- CORRECTION: Dynamic Formatting and Parameters for Gemma-4 Hardening ---
            is_gemma_family = "gemma" in self.model_path.lower()
            
            # Gemma 4 handles format processing natively; do NOT force llava-v1.5 formatting rules
            resolved_format = None
            if chat_handler:
                resolved_format = None if is_gemma_family else "llava-v1.5"
                
            # Keep system tokens in cache window (Calculate length of persona rules roughly)
            resolved_n_keep = max(1, len(PERSONA_PROMPTS.get(target_level, "")) // 4) if chat_handler else 0

            # Speculative MTP Drafting Setup
            draft_model = None
            mtp_model_path = None
            if self.config.get("speculative_drafting", False) and not target_tier.startswith("vision_") and not chat_handler:
                assistant_path = None
                mtp_mapping = self.config.get("mtp_mapping", {})
                
                # 1. Check persistent MTP mapping first
                norm_main = os.path.normcase(os.path.abspath(self.model_path)) if self.model_path else ""
                if self.model_path in mtp_mapping and os.path.exists(mtp_mapping[self.model_path]):
                    cand = mtp_mapping[self.model_path]
                    if norm_main and os.path.normcase(os.path.abspath(cand)) != norm_main:
                        assistant_path = cand
                    else:
                        del mtp_mapping[self.model_path]
                        self.config["mtp_mapping"] = mtp_mapping
                        if hasattr(self, 'save_config'):
                            self.save_config()
                else:
                    # 2. Search for assistant model in same directory (excluding main model itself)
                    if self.model_path:
                        model_dir = os.path.dirname(self.model_path)
                        if os.path.exists(model_dir):
                            for f in os.listdir(model_dir):
                                cand_path = os.path.join(model_dir, f)
                                if (
                                    f.lower().endswith(".gguf")
                                    and norm_main
                                    and os.path.normcase(os.path.abspath(cand_path)) != norm_main
                                    and any(k in f.lower() for k in ["assistant", "mtp", "dflash", "drafter"])
                                ):
                                    assistant_path = cand_path
                                    break
                    
                    # 3. Drafter MTPicker (Registration Wizard)
                    if not assistant_path and self.model_path:
                        import tkinter.messagebox
                        from tkinter import filedialog
                        try:
                            response = tkinter.messagebox.askyesno(
                                "MTP Drafter Required", 
                                f"No MTP assistant model was automatically found for:\n{os.path.basename(self.model_path)}\n\nWould you like to locate the Assistant model file manually?",
                                parent=getattr(self, 'root', None)
                            )
                            if response:
                                selected_path = filedialog.askopenfilename(
                                    title="Select MTP Assistant Model",
                                    filetypes=[("GGUF Models", "*.gguf")],
                                    initialdir=os.path.dirname(self.model_path)
                                )
                                if selected_path and os.path.normcase(os.path.abspath(selected_path)) != norm_main:
                                    assistant_path = selected_path
                                    mtp_mapping[self.model_path] = assistant_path
                                    self.config["mtp_mapping"] = mtp_mapping
                                    if hasattr(self, 'save_config'):
                                        self.save_config()
                        except Exception as gui_err:
                            print(f"[ENGINE] MTPicker GUI failed: {gui_err}")
                
                if assistant_path and os.path.exists(assistant_path) and os.path.normcase(os.path.abspath(assistant_path)) != norm_main:
                    try:
                        from System.gguf_draft_model import GgufDraftModel
                        draft_model = GgufDraftModel(assistant_path, n_gpu_layers=0, n_ctx=min(n_ctx, 4096))
                        msg = f"[MTP] Speculative GGUF assistant model loaded: {os.path.basename(assistant_path)}"
                        print(f"[ENGINE] {msg}")
                        self.process_queue.put({"status": "diag_log_update", "content": f"[ENGINE] {msg}"})
                        self.process_queue.put({"status": "log_update", "content": f"\n{msg}\n"})
                    except Exception as draft_err:
                        err_msg = f"[MTP] Failed to load GgufDraftModel ({draft_err}). Speculative drafting disabled."
                        print(f"[ENGINE] {err_msg}")
                        self.process_queue.put({"status": "diag_log_update", "content": f"[ENGINE] {err_msg}"})
                        draft_model = None
                else:
                    # Pure MTP / companion model drafting only: No speculative prompt lookup decoding fallback
                    draft_model = None

            is_diffusion = "diffusion" in self.model_path.lower()
            try:
                if is_diffusion:
                    from System.diffusion_wrapper import DiffusionCLIWrapper
                    model = DiffusionCLIWrapper(
                        app_instance=self,
                        model_path=self.model_path,
                        n_gpu_layers=n_layers,
                        n_ctx=n_ctx,
                        chat_handler=chat_handler,
                        **params
                    )
                    print(f"[ENGINE] Diffusion model detected. Initializing DiffusionCLIWrapper.")
                else:
                    try:
                        model = Llama(
                            model_path=self.model_path, 
                            n_gpu_layers=n_layers,       # Dynamic HAO
                            n_ctx=n_ctx,                 # Dynamic Context Window
                            n_threads=HardwareProfile.get_optimal_threads(), # Dynamic physical/logical core allocation
                            n_batch=max(1024, self.n_batch_config.get(target_tier, 512)),
                            n_ubatch=512,                # Increased for Parallel Prefill performance on i7-12700KF
                            n_keep=resolved_n_keep,      # FIXED: Preserves system prompt and visual structures in KV Cache
                            n_seq_max=1,                 # Explicit single sequence for max memory savings
                            chat_handler=chat_handler,
                            chat_format=resolved_format, # FIXED: Prevents template collisions on Gemma-4 structures
                            verbose=True, 
                            use_mmap=True,               # i7 handles kernel mapping
                            flash_attn=use_flash,        # Dynamic Flash Attention
                            type_k=t_k, type_v=t_v,      # Dynamic KV Quantization
                            offload_kqv=not no_kv_offload,
                            logits_all=False,                            # DISABLED: Prevents n_ctx * vocab_size (98k x 262k) 96GB float32 OOM array allocation
                            tensor_split=None,
                            rpc_servers=None,
                            override_tensors=override_tensors,
                            draft_model=draft_model,      # Speculative GgufDraftModel or PromptLookup
                            **params
                        )
                    except Exception as init_err:
                        if draft_model is not None:
                            print(f"[ENGINE] Drafter initialization failed ({init_err}). Retrying without draft model...")
                            model = Llama(
                                model_path=self.model_path, 
                                n_gpu_layers=n_layers,
                                n_ctx=n_ctx,
                                n_threads=HardwareProfile.get_optimal_threads(),
                                n_batch=max(1024, self.n_batch_config.get(target_tier, 512)),
                                n_ubatch=512,
                                n_keep=resolved_n_keep,
                                n_seq_max=1,
                                chat_handler=chat_handler,
                                chat_format=resolved_format,
                                verbose=True, 
                                use_mmap=True,
                                flash_attn=use_flash,
                                type_k=t_k, type_v=t_v,
                                offload_kqv=not no_kv_offload,
                                logits_all=False,
                                tensor_split=None,
                                rpc_servers=None,
                                override_tensors=override_tensors,
                                draft_model=None,
                                **params
                            )
                        else:
                            raise init_err
            except Exception as e:
                err_msg = f"CRITICAL: Llama initialization failed: {e}"
                self.process_queue.put({"status": "diag_log_update", "content": err_msg})
                self.process_queue.put({"status": "thinking_status", "content": "Engine Stall: Check VRAM/Path logs."})
                raise e
            
            # --- APEX VRAM-FIRST (500MB Floor Check) ---
            if SYSTEM_MONITOR_LOADED and self.gpu_handle:
                try:
                    mem = nvidia_ml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    free_mb = mem.free / (1024**2)
                    if free_mb < 500:
                        print(f"[APEX] VRAM floor breached ({free_mb:.0f}MB < 500MB). Self-healing...")
                        current_offset = self.state.get("vram_layer_offset", 0)
                        self.state["vram_layer_offset"] = current_offset - 1
                        # Re-initialization
                        # Change the re-initialization delay from 0 to 2000ms
                        print(f"[APEX] VRAM floor breached. Cooling down before retry...")
                        self.root.after(2000, lambda *args: self.model_swap(target_tier=self.current_model_tier))
                        return
                except Exception as e:
                    print(f"[APEX] VRAM Floor Check Failed: {e}")

            # --- GGUF KV Cache Benchmark ---
            if not is_diffusion and self.config.get("benchmark_enabled", False):
                try:
                    print(f"[BENCHMARK] Running GGUF KV Cache Benchmark...")
                    start_b = time.time()
                    prompt = "Hello, how are you today?"
                    # Run completion of up to 100 tokens
                    res = model(
                        prompt,
                        max_tokens=100,
                        temperature=0.3,
                        top_p=0.95,
                        stream=False
                    )
                    duration = time.time() - start_b
                    
                    # Get the number of generated tokens
                    choices = res.get("choices", [])
                    text = choices[0].get("text", "") if choices else ""
                    usage = res.get("usage", {})
                    num_tokens = usage.get("completion_tokens", 0) or max(1, len(text.split()))
                    
                    # Dynamically retrieve number of layers, heads, head dim from llama.cpp if available
                    # or fallback to typical sizes [Layers: 32, Heads: 32, Head_Dim: 128]
                    num_layers = 32
                    num_heads = 32
                    head_dim = 128
                    
                    elements = num_layers * num_heads * head_dim * num_tokens * 2 # Key + Value
                    
                    # Calculate baseline FP16 vs current active quantization cache memory
                    base_mem = (elements * 16) / (8 * 1024 * 1024) # 16-bit calculation
                    
                    # Map active t_k to bit-width
                    # 0 -> f32/f16 (16-bit), 1 -> q8_0 (8-bit), 2 -> q4_0 (4-bit), 3 -> q4_1 (4.5-bit), etc.
                    active_bits = 16
                    if t_k == 1 or t_k == 8:
                        active_bits = 8
                    elif t_k == 2 or t_k == 3 or t_k == 4:
                        active_bits = 4
                    elif t_k == 6 or t_k == 7:
                        active_bits = 5
                    
                    quant_mem = (elements * active_bits) / (8 * 1024 * 1024)
                    
                    print(f"--- THE VERDICT ---")
                    print(f"Baseline (FP16) Cache: {base_mem:.2f} MB")
                    print(f"Active Quantized ({active_bits}-bit) Cache: {quant_mem:.2f} MB")
                    print(f"Speedup Ratio: {num_tokens / max(0.01, duration):.2f} tokens/sec")
                    print(f"Memory Saved: {base_mem - quant_mem:.2f} MB")
                    
                    self.process_queue.put({
                        "status": "diag_log_update",
                        "content": f"[BENCHMARK] Baseline: {base_mem:.2f}MB | Active Quantized ({active_bits}-bit): {quant_mem:.2f}MB | Speed: {num_tokens / max(0.01, duration):.2f} tok/s"
                    })
                except Exception as be:
                    print(f"[BENCHMARK] GGUF Benchmark failed: {be}")

            # --- Tokenizer BOS Verification ---
            try:
                bos_id = model.token_bos() if hasattr(model, 'token_bos') else -1
                eos_id = model.token_eos() if hasattr(model, 'token_eos') else -1
                print(f"[ENGINE] Tokenizer BOS Verification: BOS ID={bos_id} (Valid: {bos_id != -1}), EOS ID={eos_id}")
                self.process_queue.put({
                    "status": "diag_log_update",
                    "content": f"[ENGINE] Tokenizer BOS Verified: BOS ID={bos_id} (Valid: {bos_id != -1}) | EOS ID={eos_id}"
                })
            except Exception as tok_err:
                print(f"[ENGINE] BOS Verification check: {tok_err}")

            if draft_model is not None and assistant_path:
                self.process_queue.put({
                    "status": "diag_log_update",
                    "content": f"[MTP] Speculative Decoding Active: Assistant={os.path.basename(assistant_path)}"
                })

            if self.stop_process.is_set(): return
            self.process_queue.put({"status": "load_success", "model": model, "level": target_level, "tier": target_tier})

        except Exception as e:
            err_str = str(e)
            print(f"[ENGINE] Model load error/exception detected: {err_str}")
            if patch_gguf_architecture(self.model_path):
                self.process_queue.put({"status": "log_update", "content": "\n[AUTO-PATCH]: Intercepted unknown GGUF model architecture. Automatically patched binary header. Retrying model load...\n"})
                try:
                    model = Llama(
                        model_path=self.model_path, 
                        n_gpu_layers=n_layers,       # Dynamic HAO
                        n_ctx=n_ctx,                 # Dynamic Context Window
                        n_threads=HardwareProfile.get_optimal_threads(), # Dynamic physical/logical core allocation
                        n_batch=max(1024, self.n_batch_config.get(target_tier, 512)),
                        n_ubatch=512,                # Increased for Parallel Prefill performance on i7-12700KF
                        n_keep=resolved_n_keep,      # FIXED: Preserves system prompt and visual structures in KV Cache
                        n_seq_max=1,                 # Explicit single sequence for max memory savings
                        chat_handler=chat_handler,
                        chat_format=resolved_format, # FIXED: Prevents template collisions on Gemma-4 structures
                        verbose=True, 
                        use_mmap=True,               # i7 handles kernel mapping
                        flash_attn=use_flash,        # Dynamic Flash Attention
                        type_k=t_k, type_v=t_v,      # Dynamic KV Quantization
                        offload_kqv=not no_kv_offload,
                        logits_all=False,            # DISABLED: Prevents n_ctx * vocab_size array allocation
                        tensor_split=None,
                    )
                    try:
                        bos_id = model.token_bos() if hasattr(model, 'token_bos') else -1
                        eos_id = model.token_eos() if hasattr(model, 'token_eos') else -1
                        print(f"[ENGINE] Tokenizer BOS Verification (Retry): BOS ID={bos_id} (Valid: {bos_id != -1}), EOS ID={eos_id}")
                    except Exception: pass
                    if self.stop_process.is_set(): return
                    self.process_queue.put({"status": "load_success", "model": model, "level": target_level, "tier": target_tier})
                    return
                except Exception as retry_e:
                    err_str = f"Model load retry failed after GGUF architecture patch: {retry_e}"

            if not self.stop_process.is_set(): self.state["last_crash"] = True; self.process_queue.put({"status": "load_error", "content": err_str})

    def _generation_worker(self, user_message, temp_messages):
        """Standard chat inference with Gemma-4 hardening."""
        try:
            HardwareProfile.pin_to_p_cores()
            HardwareProfile.set_priority("above_normal")
            
            # Wit-Layer: Thinking Message
            self.process_queue.put({"status": "thinking_status", "content": "Hold up, lemme cook..."})
            print(f"[INFERENCE] Starting generation for user message ({len(user_message)} chars).")
            
            sys_content = PERSONA_PROMPTS.get(self.active_persona_level, "You are Serenity.")
            if self.model_path and "muse" in self.model_path.lower() and "glimmer" in self.model_path.lower():
                r_str = self.config.get("muse_reasoning_strength", "xhigh")
                if r_str != "off": sys_content += f"\nReasoning strength: {r_str}"
            time_grounding = f"\n[TIME GROUNDING]: Current local date and time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S (%A)')}."
            sys_content += time_grounding
            if any(c.isdigit() for c in user_message) and re.search(r'\d{4,}', user_message):
                grounding_rule = (
                    "\n[LITERAL GROUNDING RULE]: The user's query contains long numbers or mathematical expressions. "
                    "You MUST transcribe the numbers and the mathematical expression EXACTLY as written at the very beginning "
                    "of your thinking process inside your thought channel (e.g. 'Input Expression: ...'). Refer ONLY to this "
                    "literal transcription for any calculations, planning, or reasoning to avoid memory/tokenizer distortion."
                )
                sys_content += grounding_rule

            is_gemma = "gemma" in (self.model_path or "").lower()
            has_multimodal_content = any(isinstance(m.get("content"), list) for m in temp_messages)
            if has_multimodal_content:
                self._ensure_chat_handler()
            elif hasattr(self.model, "chat_handler") and self.model.chat_handler is not None:
                self.model.chat_handler = None
            
            # Setup Inference Params
            params = self._get_inference_params(temp_messages)
            if is_gemma:
                # Structural Safety: Ensure the turn closer is always present, but don't force legacy tokens
                if "<turn|>" not in params.get("stop", []):
                    params.setdefault("stop", []).append("<turn|>")
                            
            sys_clean = sys_content.strip()
            def official_q(s): return f"<|\"|>{s}<|\"|>"

            # Programmatic Tool Calling (PTC) definitions injection (arXiv:2608.06370v1)
            tools = self.tool_registry.get_definitions(self.active_persona_level)
            if tools and self.active_persona_level >= 2:
                tool_defs = self.tool_registry.get_python_stubs(self.active_persona_level)
                sys_clean += f"\n[PROGRAMMATIC TOOL CALLING]: To retrieve live data or execute system actions, invoke tools via Python function calls (e.g. `web_search(query='...')`).\n{tool_defs}"

            # Level 3+ or Deep Cook need the thought channel constraint
            is_diffusion = "diffusion" in (self.model_path or "").lower()
            model_name_lower = os.path.basename(self.model_path or "").lower()
            is_nemotron = "nemotron" in model_name_lower
            is_qwen = "qwen" in model_name_lower
            is_deepseek = any(k in model_name_lower for k in ["deepseek", "r1", "qwq"])
            is_muse = any(k in model_name_lower for k in ["muse", "glimmer", "onyx", "atem"])

            if not is_diffusion and (self.active_persona_level >= 3 or self.state.get("deep_cook")):
                if is_gemma:
                    sys_clean = f"<|think|>\n{sys_clean}"
                elif is_nemotron:
                    sys_clean += "\n[REASONING]: Provide clear, direct, and rigorous answers without conversational meta-commentary."
                elif is_qwen or is_deepseek:
                    sys_clean += "\n[REASONING]: Analyze the query thoroughly and provide a direct, precise answer."
                elif is_muse:
                    pass # Native ATEM Jinja template handles reasoning_strength
                else:
                    sys_clean += "\n[REASONING]: Think step by step before answering and provide a clear, accurate response."
            elif self.active_persona_level == 2:
                sys_clean += "\n[SEARCH PROTOCOL]: If you need live information, invoke a search tool immediately."
            
            sys_content = sys_clean

            # TriAttention KV Pruning
            if is_diffusion:
                # Diffusion architectures allocate massive contiguous ubatches. Limit history strictly.
                processed_msgs = temp_messages[-6:]
            elif self.kv_manager and TRI_ATTENTION_ENABLED:
                processed_msgs = self.kv_manager.enforce_kv_budget(temp_messages)
            else:
                processed_msgs = temp_messages[-12:]
                        
            # Multi-turn thought pruning to avoid feeding previous thoughts back into history
            cleaned_msgs = []
            for m in processed_msgs:
                role = "assistant" if m.get("role") in ("assistant", "model") else m.get("role")
                content = m.get("content")
                if role == "assistant" and isinstance(content, str):
                    content = re.sub(r'(?s)<think>.*?(?:<\/think>|$)', '', content, flags=re.IGNORECASE)
                    content = re.sub(r'(?s)<\|channel>thought.*?(?:<channel\|>|$)', '', content, flags=re.IGNORECASE)
                    content = re.sub(r'<\|think\|>.*?(?:<\/\|think\|>|$)', '', content, flags=re.IGNORECASE | re.DOTALL)
                    content = re.sub(r'<thought(?:>|\b).*?(?:<\/thought>|$)', '', content, flags=re.IGNORECASE | re.DOTALL)
                    content = re.sub(r'(?s)(?:<\|start\|>assistant\s+)?to=self<\|message\|>.*?(?:<\|eom\|>|$)', '', content, flags=re.IGNORECASE)
                    content = re.sub(r'(?s)<\|start\|>assistant\s+to=user(?:<\|message\|>)?', '', content, flags=re.IGNORECASE)
                    content = re.sub(r'<\|eot\|>|<\|end_of_text\|>', '', content, flags=re.IGNORECASE)
                    content = content.strip()
                cleaned_msgs.append({"role": role, "content": content})
            processed_msgs = cleaned_msgs

            msgs = [{"role": "system", "content": sys_content}] + processed_msgs

            # BOS & Tokenizer Verification Log
            if hasattr(self.model, 'token_bos'):
                bos_id = self.model.token_bos()
                print(f"[INFERENCE] Native Jinja Chat: BOS verified (ID={bos_id})")

            status_text = "Analyzing logical momentum..." if self.active_persona_level >= 3 else "Direct Strike: Pre-computing..."
            self.process_queue.put({"status": "thinking_status", "content": status_text})
            self.process_queue.put({"status": "status_phase", "phase": "prefill", "details": "Ingesting prompt context..."})
            
            # GIL-Safety: Internal Streaming with Thought Stream Demuxing & Draft Rollback Protection
            full_resp = ""
            in_thought_channel = False
            thought_detected = False
            stream_lead_buffer = ""
            streamed_draft_to_ui = False
            streamed_answer_chars = 0
            closers_regex = r'(?:<\/think>|<\/thought>|<\/\|think\|>|<\|im_end\|>|<\|channel>text|<\|channel>assistant|<channel\|>|<\/channel\|>|\[\/DRAFT\]|<\|eom\|>|<\|start\|>assistant\s+to=user(?:<\|message\|>)?|to=user<\|message\|>)'
            openers_exact = (
                "<think>", "<thought>", "<|think|>", "<|channel>thought", "<channel|thought>",
                "<|im_start|>thought", "<|im_start>thought", "[draft]", "to=self<|message|>",
                "<|start|>assistant to=self", "to=self"
            )
            closers = ["</think>", "</thought>", "</|think|>", "<|im_end|>", "<channel|>", "</channel|>", "[/draft]", "<|channel>text", "<|channel>assistant", "<|eom|>", "<|start|>assistant to=user", "to=user<|message|>"]

            def _is_thought_opening(text):
                if not text: return False
                t_lower = text.lower().strip()
                t_raw_lower = text.lower().lstrip()
                if any(op in t_lower for op in openers_exact):
                    return True
                if t_raw_lower.startswith("thought\n") or t_raw_lower.startswith("thought\r\n") or t_raw_lower.startswith("thought ") or t_raw_lower.startswith("thought:"):
                    return True
                if t_raw_lower.startswith("thought") and len(t_raw_lower) <= 12 and ("\n" in text or len(text.strip()) == len("thought")):
                    return True
                return False

            t_gen_start = time.time()
            ttft_recorded = False
            token_count = 0

            gen_iterator = self.model.create_chat_completion(messages=msgs, **params, stream=True)
            for chunk in gen_iterator:
                if self.stop_process.is_set(): break
                if "content" in chunk["choices"][0]["delta"]:
                    txt = chunk["choices"][0]["delta"]["content"]
                    full_resp += txt
                    lower_resp = full_resp.lower()

                    if not ttft_recorded:
                        ttft = time.time() - t_gen_start
                        ttft_recorded = True
                        self.process_queue.put({"status": "status_ttft", "ttft": ttft})
                    
                    token_count += 1
                    elapsed = max(0.001, time.time() - t_gen_start)
                    cur_speed = token_count / elapsed
                    if token_count % 3 == 0:
                        self.process_queue.put({
                            "status": "status_phase", 
                            "phase": "reasoning" if in_thought_channel else "generating", 
                            "tokens": token_count, 
                            "speed": cur_speed
                        })
                    
                    if not thought_detected:
                        if _is_thought_opening(full_resp):
                            thought_detected = True
                            in_thought_channel = True
                            if streamed_draft_to_ui:
                                self.process_queue.put({"status": "streaming_replace", "content": ""})
                                streamed_draft_to_ui = False
                            stream_lead_buffer = ""
                            self.process_queue.put({"status": "thought_stream", "content": txt})
                        elif (full_resp.strip().startswith("<") or full_resp.strip().startswith("to=") or full_resp.strip().lower().startswith("thought") or full_resp.strip().startswith("[")) and len(full_resp.strip()) < 40:
                            stream_lead_buffer += txt
                        else:
                            if stream_lead_buffer:
                                self.process_queue.put({"status": "streaming", "content": stream_lead_buffer})
                                stream_lead_buffer = ""
                                streamed_draft_to_ui = True
                            self.process_queue.put({"status": "streaming", "content": txt})
                            streamed_draft_to_ui = True
                    else:
                        if in_thought_channel:
                            self.process_queue.put({"status": "thought_stream", "content": txt})
                            if any(c in lower_resp for c in closers) or re.search(closers_regex, full_resp, flags=re.IGNORECASE):
                                in_thought_channel = False
                                parts = re.split(closers_regex, full_resp, flags=re.IGNORECASE)
                                if len(parts) > 1 and parts[-1]:
                                    ans_chunk = parts[-1][streamed_answer_chars:]
                                    if ans_chunk:
                                        self.process_queue.put({"status": "streaming", "content": ans_chunk})
                                        streamed_answer_chars += len(ans_chunk)
                        else:
                            parts = re.split(closers_regex, full_resp, flags=re.IGNORECASE)
                            if len(parts) > 1 and parts[-1]:
                                ans_chunk = parts[-1][streamed_answer_chars:]
                                if ans_chunk:
                                    self.process_queue.put({"status": "streaming", "content": ans_chunk})
                                    streamed_answer_chars += len(ans_chunk)
                            else:
                                self.process_queue.put({"status": "streaming", "content": txt})

                time.sleep(0.001) # Heartbeat for UI

                # Loop Mitigation: Check for infinite repetition loops
                if len(full_resp) > 200 and self._detect_repetition(full_resp):
                    self.process_queue.put({"status": "diag_log_update", "content": "[RUNTIME] Repetition loop detected! Breaking inference stream to preserve sanity."})
                    print("[RUNTIME] Repetition loop detected! Breaking inference stream.")
                    break

            elapsed_total = max(0.001, time.time() - t_gen_start)
            final_speed = token_count / elapsed_total if token_count > 0 else 0.0
            self.process_queue.put({"status": "status_phase", "phase": "complete", "tokens": token_count, "speed": final_speed})

            think_log = ""
            final_answer = full_resp.strip()
            
            # Telemetry Extraction (subconscious / prime chronicles)
            match_deep = re.search(r'\[DEEPLOG:(.*?)\]', final_answer, flags=re.IGNORECASE | re.DOTALL)
            if match_deep:
                extracted_log = match_deep.group(1).strip()
                final_answer = re.sub(r'\[DEEPLOG:.*?\]', '', final_answer, flags=re.IGNORECASE | re.DOTALL).strip()
                log_path = os.path.join(self.dirs["Logs"], "subconscious_journal.txt")
                try:
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {extracted_log}\n")
                except Exception: pass

            match_prime = re.search(r'\[PRIME_MEMORY:(.*?)\]', final_answer, flags=re.IGNORECASE | re.DOTALL)
            if match_prime:
                extracted_prime = match_prime.group(1).strip()
                final_answer = re.sub(r'\[PRIME_MEMORY:.*?\]', '', final_answer, flags=re.IGNORECASE | re.DOTALL).strip()
                prime_path = os.path.join(self.dirs["System"], ".prime_chronicles.txt")
                try:
                    with open(prime_path, "a", encoding="utf-8") as f:
                        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {extracted_prime}\n")
                except Exception: pass

            # Advanced Scout & Split
            closers_patterns = [
                r'<\/think>', r'<\/thought>', r'<\/\|think\|>', r'<\|im_end\|>', r'<\|im_end>',
                r'<\|channel>text', r'<\|channel>assistant', r'<channel\|>', r'<\/channel\|>', r'\[\/DRAFT\]',
                r'<\|eom\|>', r'<\|start\|>assistant\s+to=user(?:<\|message\|>)?', r'to=user<\|message\|>',
                r'(?i)\n(?:Final Output|Final Polish|Grandmaster Verdict|Final Answer|Execution complete)[\s:]+'
            ]
            all_splits = []
            for tag_pattern in closers_patterns:
                for m in re.finditer(tag_pattern, final_answer, re.IGNORECASE):
                    all_splits.append(m.end())
            
            if not all_splits and (
                any(t in final_answer.lower() for t in ["<think>", "<thought>", "<|think|>", "<|channel>thought", "<channel|thought>", "<|im_start|>thought", "<|im_start>thought", "[draft]", "to=self", "<|start|>assistant to=self"])
                or _is_thought_opening(final_answer)
            ):
                all_splits.append(len(final_answer))
            
            all_splits.sort()
            best_split = -1
            if all_splits:
                for split in all_splits:
                    remaining = final_answer[split:].strip()
                    if re.search(r'<think>|<thought>|\[DRAFT\]|<\|channel>thought|<channel\s*\|?>|<\|think\|>|<\|im_start\|?>thought|to=self', remaining, re.IGNORECASE):
                        continue
                    best_split = split
                    break
                if best_split == -1:
                    best_split = all_splits[-1]
            
            if best_split != -1:
                think_log = final_answer[:best_split].strip()
                final_answer = final_answer[best_split:].strip()
            else:
                has_thought_openers = bool(re.search(r'<think>|<thought>|\[DRAFT\]|<\|channel>thought|<channel\s*\|?>|<\|think\|>|<\|im_start\|?>thought|to=self|^thought\s+', final_answer, re.IGNORECASE))
                if has_thought_openers or _is_thought_opening(final_answer):
                    think_log = final_answer
                    final_answer = ""
                else:
                    # Model produced direct response without thought tags: keep as final answer
                    think_log = ""
                    final_answer = full_resp.strip()

            # Strip wrapper tags and residual tool markers
            tag_clean_pattern = r'(?i)<think>|<thought>|\[DRAFT\]|<\|channel>thought|<channel\|thought>|<channel\s*\|?>|<\/think>|<\/thought>|<\/\|think\|>|<\|think\|>|<\|im_start\|?>thought|<\|im_end\|?>|\[\/DRAFT\]|<\|channel>text|<\|channel>assistant|<channel\|>|<\/channel\|>|<\|tool_call>|<tool_call\|>|<\|tool_response>|<tool_response\|>|<\|tool>|<tool\|>|<ctrl42>|<\/ctrl42>|<\|?turn\|?>|<\|start\|>assistant\s+to=user(?:<\|message\|>)?|<\|start\|>assistant\s+to=self(?:<\|message\|>)?|to=self<\|message\|>|to=user<\|message\|>|<\|eom\|>|<\|eot\|>'
            think_log = re.sub(tag_clean_pattern, '', think_log).strip()
            think_log = re.sub(r'(?i)^thought\s+', '', think_log).strip()
            final_answer = re.sub(tag_clean_pattern, '', final_answer).strip()
            final_answer = re.sub(r'(?i)^thought\s+', '', final_answer).strip()

            # Synthesis fallback: if thoughts exist but model did not output final answer
            if think_log and not final_answer:
                self.process_queue.put({"status": "thinking_status", "content": "[PROCESS] Synthesizing Final Answer..."})
                synthesized = self._perform_final_synthesis(user_message, think_log)
                if synthesized:
                    final_answer = synthesized.strip()
                    final_answer = re.sub(tag_clean_pattern, '', final_answer).strip()
                    final_answer = re.sub(r'(?i)^thought\s+', '', final_answer).strip()

            # Tool Execution Check in Standard Generation
            if tools and self.active_persona_level >= 2:
                has_py_tool = re.search(r'(?:```(?:python)?\s*)?\b(web_search|read_file|get_system_stats|control_rgb|generate_image)\s*\(.*?\)', final_answer, re.DOTALL | re.IGNORECASE)
                has_tag_tool = re.search(r'(?:<ctrl42>call:|<\|tool_call>call:|<\|tool_call\|>call:|<\|tool>call:|call:|action:|<(?:channel\|)?(?:execute_tool|executetool)>)\s*([\w_]+)\s*\{', final_answer, re.DOTALL | re.IGNORECASE)
                has_json_tool = re.search(r'\{\s*["\'](?:action|tool|name|function)["\']\s*:\s*["\'](?:web_search|read_file|get_system_stats|control_rgb|generate_image)["\']', final_answer, re.IGNORECASE)
                if has_py_tool or has_tag_tool or has_json_tool:
                    self.process_queue.put({"status": "thinking_status", "content": "Executing tool..."})
                    final_answer = self._run_tool_loop(final_answer, msgs, params)
                    final_answer = re.sub(tag_clean_pattern, '', final_answer).strip()

            if not final_answer and full_resp and not think_log:
                final_answer = re.sub(tag_clean_pattern, '', full_resp).strip()
                final_answer = re.sub(r'(?i)^thought\s+', '', final_answer).strip()
            
            self.process_queue.put({"status": "thinking_status", "content": "Wall dropping. Here's the deep dive:"})
            print(f"[INFERENCE] Generation complete. Final response length: {len(final_answer)} chars.")
            self.process_queue.put({
                "status": "session_finished",
                "user_msg": user_message,
                "think_log": think_log.strip(),
                "final_answer": final_answer.strip(),
                "is_error": False
            })
        except Exception as e:
            traceback.print_exc()
            self.process_queue.put({"status": "error", "content": str(e)})
        finally:
            # --- MISSION: Hardened Post-Inference Cleanup ---
            try:
                HardwareProfile.release_cores()
                HardwareProfile.set_priority("normal")
            except: pass
            
            # CRITICAL: Do NOT call hygiene_gate from worker thread.
            # gc.collect() on worker thread finalizes Tk objects → Tcl crash.
            # Defer cleanup to main thread via process queue.
            self.process_queue.put({"status": "cleanup"})

    def _generation_worker_deep_cook(self, user_msg):
        """Recursive multi-step reasoning pinned to P-cores."""
        try:
            HardwareProfile.pin_to_p_cores()
            HardwareProfile.set_priority("above_normal")
            
            def _run_step_streaming(log_title, prompt, status_msg, ctype=None, cnum=0, dnum=0, temp_override=None, reasoning_history=None):
                if self.stop_process.is_set(): raise InterruptedError()
                self.process_queue.put({"status": "log_update", "content": f"\n--- {log_title} ---\n"})
                self.process_queue.put({"status": "thinking_status", "content": status_msg})
                
                sys_msg = DEEP_COOK_SYSTEM_PROMPTS.get(self.active_persona_level, "You are a logical, step-by-step reasoning AI.")
                if self.model_path and "muse" in self.model_path.lower() and "glimmer" in self.model_path.lower():
                    r_str = self.config.get("muse_reasoning_strength", "xhigh")
                    if r_str != "off": sys_msg += f"\nReasoning strength: {r_str}"
                is_gemma = "gemma" in self.model_path.lower()
                params = self._get_inference_params(reasoning_history=reasoning_history)
                if temp_override:
                    params["temperature"] = temp_override
                    # Nudge top_k and min_p if temperature changed to break loops
                    params["top_k"] = 64
                    params["min_p"] = 0.02
                
                if is_gemma:
                    tool_defs = self.tool_registry.get_gemma_declarations(self.active_persona_level)
                    sys_msg = f"{sys_msg}\n{tool_defs}".strip()
                
                msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": prompt}]
                stream = self.model.create_chat_completion(messages=msgs, stream=True, **params)
                
                if ctype:
                    self.process_queue.put({
                        "status": "deep_cook_ui_start",
                        "ctype": ctype,
                        "cnum": cnum,
                        "dnum": dnum,
                        "title": log_title
                    })
                
                result_text = ""
                last_finish_reason = "N/A"
                tag_buffer = ""
                for chunk in stream:
                    if self.stop_process.is_set(): raise InterruptedError()
                    ch = chunk.get('choices', [{}])[0]
                    c = ch.get('delta', {}).get("content", "") or ch.get('text', "")
                    if ch.get('finish_reason'): last_finish_reason = ch.get('finish_reason')
                    if c: 
                        result_text += c
                        tag_buffer += c
                        
                        # MISSION: Redirect tokens to specialized UI buckets for Deep Cook
                        if ctype:
                            self.process_queue.put({
                                "status": "deep_cook_ui_stream", 
                                "ctype": ctype, 
                                "cnum": cnum, 
                                "dnum": dnum, 
                                "content": c
                            })
                        else:
                            self.process_queue.put({"status": "streaming", "content": c})
                        
                        time.sleep(0.001) # Heartbeat for UI
                        
                        # --- REAL-TIME TAG CAPTURE ---
                        if '>' in tag_buffer:
                            tags = re.findall(r'<[^>]+>', tag_buffer)
                            for t in tags:
                                self.process_queue.put({"status": "diag_log_update", "content": f"[RUNTIME TAG] {t}"})
                            last_close = tag_buffer.rfind('>')
                            if last_close != -1:
                                tag_buffer = tag_buffer[last_close+1:]
                        if len(tag_buffer) > 100: tag_buffer = tag_buffer[-50:]
                        
                        # Deep Cook Robustness Audit: Detect recursive logic loops or identity leakage
                        lower_c = c.lower()
                        if "the prompt is asking" in lower_c or "you are serenity" in lower_c or "task:" in lower_c:
                            self.process_queue.put({"status": "thinking_status", "content": "Subconscious Reset: Identity Leakage Detected."})
                            raise InterruptedError("Subconscious Reset")
                        
                        # Loop Mitigation: Check for infinite repetition loops
                        if len(result_text) > 200 and self._detect_repetition(result_text):
                            self.process_queue.put({"status": "diag_log_update", "content": "[RUNTIME] Repetition loop detected in Deep Cook step! Breaking stream."})
                            print("[RUNTIME] Repetition loop detected in Deep Cook step! Breaking stream.")
                            break
                
                # --- DIAGNOSTICS TELEMETRY ---
                diag_msg = f"--- DEEP COOK DIAGNOSTICS ---\n"
                diag_msg += f"Finish Reason: {last_finish_reason}\n"
                diag_msg += f"Raw Output Length: {len(result_text)} characters\n"
                diag_msg += "------------------------------"
                self.process_queue.put({"status": "diag_log_update", "content": diag_msg})
                
                # --- DEEP COOK TOOL LOOP ---
                # Check for and execute any tool calls requested during this reasoning step
                if not is_gemma:
                    # Standard OpenAI-style tool calls handled by _run_tool_loop
                    result_text = self._run_tool_loop(result_text, prompt, params)
                else:
                    # Gemma-4 manual tool parsing (Support all Gemma-4 execute_tool / readfile / action syntaxes)
                    if any(tag in result_text.lower() for tag in ["<ctrl42>call:", "<|tool_call>call:", "<|tool>call:", "call:", "action:", "<execute_tool>", "<executetool>", "read_file", "readfile"]):
                        self.process_queue.put({"status": "thinking_status", "content": "Deep Cook: Executing Sub-Task Tool..."})
                        result_text = self._run_tool_loop(result_text, prompt_str, params)

                return result_text.strip()

            # --- RECURSIVE CYCLE ENGINE ---
            full_draft_history = ""
            current_cycle = 1
            task_complete = False
            final_status = "Success"
            skip_critique = False
            last_assessment_output = ""
            
            # --- CYCLE STATE TRACKING (Markovian Continuity) ---
            cycle_state = {
                "cycle_num": 0,
                "resolved_variables": [],
                "pending_variables": [],
                "current_subset": "",
                "last_plan_output": "",
                "last_assessment": "",
                "hollow_drafts_detected": 0,
                "tool_results": []
            }
            
            while not task_complete:
                self.process_queue.put({"status": "thinking_status", "content": f"Cycle {current_cycle}: Planning..."})
                
                # --- UPDATE CYCLE STATE AT START OF ITERATION ---
                cycle_state["cycle_num"] = current_cycle
                cycle_state["hollow_drafts_detected"] = 0
                cycle_state["tool_results"] = []
                
                # 1. DYNAMIC CYCLE PLANNING
                max_p = 3 # Rule of Three: Hard-cap drafting phases
                
                pruned_history = full_draft_history
                if self.kv_manager and TRI_ATTENTION_ENABLED:
                    pruned_history = self.kv_manager.enforce_string_kv_budget(pruned_history)
                else:
                    pruned_history = pruned_history[-1500:]
                    
                plan_prompt = (
                    f"Original Query: {user_msg}\n"
                    f"Progress so far: {pruned_history}\n"
                    f"[LAST CYCLE ASSESSMENT]: {last_assessment_output}\n"
                    f"[CYCLE STATE]: Cycle {cycle_state['cycle_num']} | Resolved: {len(cycle_state['resolved_variables'])} | Pending: {len(cycle_state['pending_variables'])}\n"
                    "Identify every remaining logical variable or technical hurdle. "
                    "Rule: If the query is 100% resolved without a single missing detail, output: [COMPLETE]. "
                    f"Otherwise, create a {max_p}-point plan for Cycle {current_cycle} focused on VERIFIABLE PROGRESS. "
                    "Each point must define a clear 'current range' or 'subset of variables'."
                )
                plan_txt = _run_step_streaming(f"Cycle {current_cycle}: Plan", plan_prompt, f"Cycle {current_cycle} | Planning...", ctype="cycle", cnum=current_cycle)
                print(f"[DEEP COOK] Cycle {current_cycle} planning phase complete.")
                
                # --- EXIT GATE: DYNAMIC COMPLETION CHECK ---
                if re.search(r'\[(?:STATUS:\s*)?COMPLETE\]', plan_txt, re.IGNORECASE):
                    task_complete = True
                    skip_critique = True
                    self.process_queue.put({"status": "log_update", "content": f"\n[SYSTEM] Cycle {current_cycle} Planning indicates total resolution attained.\n"})
                    break

                # --- STORE PLAN TO CYCLE STATE ---
                cycle_state["last_plan_output"] = plan_txt[:500]  # Store first 500 chars as summary
                
                # Surgical regex for Deep Cook: strip outer blocks but preserve cycle markers
                processed_plan = re.sub(r'<\/?think>', '', plan_txt, flags=re.IGNORECASE).strip()
                self.process_queue.put({
                    "status": "deep_cook_ui_batch", 
                    "ctype": "cycle", 
                    "cnum": current_cycle, 
                    "text": processed_plan 
                })

                points = [l for l in plan_txt.split('\n') if l.strip() and l.strip()[0] in '-*123456789']
                if not points: points = [f"Complete the next phase of work for Cycle {current_cycle}."]
                points = points[:max_p]
                
                # 2. CYCLE DRAFTING
                cycle_draft = ""
                for i, p in enumerate(points):
                    if self.stop_process.is_set(): raise InterruptedError()
                    
                    step_title = f"Cycle {current_cycle} | Drafting {i+1}"
                    step_status = f"Cycle {current_cycle} | Step {i+1}/{len(points)}"
                    
                    pruned_history = full_draft_history
                    if self.kv_manager and TRI_ATTENTION_ENABLED:
                        pruned_history = self.kv_manager.enforce_string_kv_budget(pruned_history)
                    else:
                        pruned_history = pruned_history[-1500:]
                    
                    # Verifiable Progress: State current state of exploration
                    context = pruned_history + "\n\n[CURRENT DRAFT]: " + cycle_draft[-1000:]
                    step_prompt = (
                        f"Query: {user_msg}\nContext: {context}\nGoal: {p}\n"
                        f"[PRIOR PLAN IDENTIFIED]: {plan_txt}\n"
                        f"[CURRENT CYCLE POINT]: {i+1}/{len(points)}\n"
                        f"[CYCLE STATE]: Resolved: {len(cycle_state['resolved_variables'])} | Pending: {len(cycle_state['pending_variables'])}\n"
                        "Verifiable Progress Rule: State the 'current range', 'sub-task', or 'subset of variables' being processed. "
                        "Maintain Abstract Logical Momentum. Provide deep, grounded analysis. "
                        "Start your response with a [STATUS: ...] tag describing the active operation. "
                        "Mark your current range/subset explicitly: [CURRENT RANGE: X]"
                    )
                    
                    chunk_text = _run_step_streaming(step_title, step_prompt, step_status, ctype="draft", cnum=current_cycle, dnum=i+1)
                    
                    # Anti-Idle Gate (Abstract Logical Momentum 2.0)
                    # Implementation of 2-Retry Hard Stop and Dynamic Thresholds
                    retry_limit = 2
                    retries = 0
                    while len(chunk_text.strip()) < 150 and retries < retry_limit:
                        retries += 1
                        cycle_state["hollow_drafts_detected"] += 1
                        self.process_queue.put({"status": "thinking_status", "content": f"Hollow Cycle Detected (Attempt {retries}/{retry_limit}). Deepening..."})
                        # Temperature Shifted Retry (0.4 / 0.7 for last attempt)
                        next_temp = 0.4 if retries == 1 else 0.7
                        retry_prompt = step_prompt + "\n\nCRITICAL: Hollow Cycle detected (shallow response). Expand your logic 3x. Provide dense technical observations."
                        chunk_text = _run_step_streaming(f"{step_title} (RETRY {retries})", retry_prompt, "Breaking Repetition Loop...", ctype="draft", cnum=current_cycle, dnum=i+1, temp_override=next_temp)
                    
                    if len(chunk_text.strip()) < 50:
                         # Hard Failure / Roadblock - log to cycle state
                         cycle_state["hollow_drafts_detected"] += 1
                         self.process_queue.put({"status": "thinking_status", "content": f"Drafting Failure: Steps remaining hollow."})
                         task_complete = True
                         final_status = "Roadblock"
                         break
                    
                    cycle_draft += chunk_text + "\n\n"
                    processed_chunk = self._extract_thinking_content(chunk_text)
                    self.process_queue.put({"status": "deep_cook_ui_batch", "ctype": "draft", "cnum": current_cycle, "dnum": i+1, "title": step_title, "text": processed_chunk})
                    print(f"[DEEP COOK] Cycle {current_cycle} | Draft {i+1} complete ({len(chunk_text)} chars).")
                
                full_draft_history += cycle_draft
                
                # 3. CONTEXT ASSESSMENT (Mandatory - Always Runs)
                # Check safety ceiling BEFORE assessment
                if current_cycle >= 5: 
                     self.process_queue.put({"status": "thinking_status", "content": "Closing Deep Cook: Safety ceiling (5 cycles) reached."})
                     task_complete = True
                     final_status = "Safety Ceiling"
                else:
                    # Assessment runs regardless of prior state
                    if self.active_persona_level == 7:
                        self.process_queue.put({"status": "thinking_status", "content": f"Cycle {current_cycle} | Hardening Memory for Cycle {current_cycle+1}..."})
                        lvl7_context = "\nInject worldbuilding context: Ensure narrative consistency and architectural lore."
                    else:
                        lvl7_context = ""

                    snapshot_prompt = (
                        f"Original Query: {user_msg}\n"
                        f"Cycle {current_cycle} Data Progress: {cycle_draft[-2000:]}\n"
                        f"[CYCLE QUALITY]: Hollow drafts detected: {cycle_state['hollow_drafts_detected']}\n"
                        f"Context Assessment: Review Cycle {current_cycle} and compare it against the original query.{lvl7_context}\n"
                        "1. Technical Debt: What specific variables or logic gates are still unverified?\n"
                        "2. Analytical Progress: Summarize the findings so far.\n"
                        "3. Next Phase: Define the precise focus for the next cycle.\n"
                        "4. Resolved Variables: List any logic fully resolved in this cycle.\n"
                        "Final status: [TOTAL RESOLUTION] or [CONTINUE: <next_focus_area>]."
                    )
                    snapshot = _run_step_streaming(f"Cycle {current_cycle}: Memory Assessment", snapshot_prompt, "Updating Persistent Intelligence...", ctype="memory", cnum=current_cycle)
                    print(f"[DEEP COOK] Cycle {current_cycle} memory assessment complete.")
                    self.process_queue.put({"status": "deep_cook_ui_batch", "ctype": "memory", "cnum": current_cycle, "title": "Context Assessment", "text": snapshot})
                    
                    # --- STORE ASSESSMENT TO CYCLE STATE & LAST OUTPUT ---
                    cycle_state["last_assessment"] = snapshot[:500]  # Store first 500 chars
                    last_assessment_output = snapshot  # Full output for next cycle planning
                    
                    # Harder injection of Cycle 1+ context to ensure continuity
                    full_draft_history += f"\n[CHRONOLOGICAL PROGRESS LOG - CYCLE {current_cycle}]: {snapshot}\n"
                    
                    # --- PHASE 4: STATE PROPAGATION & DIAGNOSTIC OUTPUT ---
                    diag_state = (
                        f"\n[CYCLE {current_cycle} STATE SNAPSHOT]\n"
                        f"  Hollow Drafts Detected: {cycle_state['hollow_drafts_detected']}\n"
                        f"  Resolved Variables: {len(cycle_state['resolved_variables'])}\n"
                        f"  Pending Variables: {len(cycle_state['pending_variables'])}\n"
                        f"  Current Subset: {cycle_state['current_subset'][:100] if cycle_state['current_subset'] else 'N/A'}\n"
                    )
                    self.process_queue.put({"status": "diag_log_update", "content": diag_state})
                    print(f"[DEEP COOK] {diag_state}")
                    
                    # --- ASSESSMENT IS SOLE AUTHORITY FOR RESOLUTION ---
                    if re.search(r'\[(?:STATUS:\s*)?TOTAL RESOLUTION\]', snapshot, re.IGNORECASE):
                        task_complete = True
                        self.process_queue.put({"status": "log_update", "content": f"\n[SYSTEM] Assessment indicates total resolution attained.\n"})
                    else:
                        current_cycle += 1

            # 5. FINAL CRITIQUE (Mandatory Verification unless skipped)
            critique_txt = "[SKIPPED - EARLY RESOLUTION]"
            if not skip_critique:
                self.process_queue.put({"status": "thinking_status", "content": "Verifying Final Logic..."})
                critique_prompt = f"Query: {user_msg}\nStatus: {final_status}\nReasoning Logs: {full_draft_history[-4000:]}\nPerform an exhaustive critique of these findings. " + \
                                  "Identify any potential errors, missing factors, or logical leaps. This is the final verification before output."
                critique_txt = _run_step_streaming("Final Critique", critique_prompt, "Exhaustive Verification...", ctype="memory", cnum=current_cycle, dnum=88)
                print("[DEEP COOK] Final critique and verification complete.")

            # 6. FINAL SYNTHESIS & 7. SUMMARY
            if self.active_persona_level == 7:
                # Level 7 Restoration: Use Cecilia-specific synthesis (passing critique)
                final_resp = self._perform_level7_synthesis(user_msg, full_draft_history, critique_txt)
            else:
                final_resp = self._perform_final_synthesis(user_msg, full_draft_history, skip_critique, critique_txt)
            
            # Wit-Layer: Delivery Message
            self.process_queue.put({"status": "thinking_status", "content": "Wall dropping. Here's the deep dive:"})
            
            # Embed the thought history for UI rendering with frontend tags
            payload = final_resp
            
            # --- DEEP COOK RUNTIME METRICS (PHASE 5-6: Quality Gates & Polish) ---
            hidden_text_len = len(full_draft_history)
            diag_msg2 = f"--- DEEP COOK METRICS ---\n"
            diag_msg2 += f"Cycles Executed: {current_cycle}\n"
            diag_msg2 += f"Final Status: {final_status}\n"
            diag_msg2 += f"Hollow Drafts Detected: {cycle_state['hollow_drafts_detected']}\n"
            diag_msg2 += f"Thought Blocks Found: {'YES' if '</think>' in full_draft_history or '<channel|>' in full_draft_history or 'CYCLE' in full_draft_history else 'NO'}\n"
            diag_msg2 += f"Total Draft Footprint: {hidden_text_len} characters\n"
            diag_msg2 += f"[MARKOVIAN STATE] Resolved: {len(cycle_state['resolved_variables'])} | Pending: {len(cycle_state['pending_variables'])}\n"
            self.process_queue.put({"status": "diag_log_update", "content": diag_msg2})
            
            self.process_queue.put({
                "status": "session_finished",
                "user_msg": user_msg,
                "think_log": full_draft_history.strip(),
                "final_answer": payload.strip(),
                "is_error": False
            })
            
        except InterruptedError: 
            self.process_queue.put({"status": "interrupted", "content": ""})
        except Exception as e: 
            self.process_queue.put({"status": "error", "content": f"Deep Cook Failed: {str(e)}"})
        finally:
            # --- MISSION: Hardened Post-Inference Cleanup (Deep Cook) ---
            try:
                HardwareProfile.release_cores()
                HardwareProfile.set_priority("normal")
            except: pass
            
            # CRITICAL: Defer cleanup to main thread (Tcl thread-safety)
            self.process_queue.put({"status": "cleanup"})

    def _run_hygiene_on_main_thread(self):
        """Main-thread-safe post-inference cleanup. Called via process queue.
        
        CRITICAL DESIGN NOTES:
        1. model.reset() is REMOVED — llama-cpp-python manages its own KV cache 
           internally between create_chat_completion() calls. Calling reset() 
           immediately after inference risks a C-level segfault if Tcl hasn't 
           finished committing widget updates from _finalize_message.
        2. gc.collect() is DEFERRED by 2 seconds via root.after() to ensure all 
           Tkinter operations (PhotoImage creation, canvas updates) have fully 
           settled before Python's garbage collector finalizes old objects.
        """
        # Delay cleanup to ensure Tk event loop has fully committed all widget ops
        try:
            print("[ENGINE] Hygiene cycle initiated. Deferring GC by 2000ms.")
            self.root.after(2000, self._deferred_gc_cleanup)
        except: pass

    def _deferred_gc_cleanup(self):
        """Runs garbage collection safely after Tk has settled."""
        try:
            import gc
            print("[ENGINE] Executing deferred garbage collection...")
            gc.collect()
        except: pass
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, Exception): pass

    def _handle_session_finished(self, data):
        """Ensures all buffers are flushed before calling _finalize_message."""
        print("[SYSTEM] Generation session finished. Finalizing buffers...")
        self._flush_log_buffer(force=True)
        if self.thought_stream_buffer:
            self._update_thought_dropdown(self.thought_stream_buffer)
            self.thought_stream_buffer = ""
        if self.text_buffer:
            self._update_ai_message(self.text_buffer)
            self.text_buffer = ""
            
        user_msg = data.get("user_msg", "")
        think_log = data.get("think_log", "")
        final_answer = data.get("final_answer", "")
        is_error = data.get("is_error", False)
        
        self._finalize_message(user_msg, think_log, final_answer, is_error)

    def _buffer_thought_stream(self, content):
        """Buffers live reasoning/thought tokens into the thought stream."""
        if not content: return
        self.thought_stream_buffer += content

    def _ensure_thought_dropdown(self):
        """Creates the thought dropdown in the chat history if not already initialized."""
        hist = self.chat_history
        if hist is None: return
        
        if not self.state.get("response_started", False):
            self.state["response_start_idx"] = hist.index(tk.END + "-1c")
            self._append_to_chat(f"\n\n{self._get_persona_label()}: ", "ai_lead")
            self.state["response_started"] = True
            think = self.thinking_display
            if think and think.winfo_exists():
                think.set_phase("reasoning")
        
        if not self.state.get("current_think_tag"):
            think_tag = f"think_block_{int(time.time() * 1000)}"
            self.state["current_think_tag"] = think_tag
            self.state["thought_streamed"] = True
            
            def toggle_thoughts(tag=think_tag, b=None):
                is_elided = str(hist.tag_cget(tag, "elide")) in ["1", "True", "true"]
                if is_elided:
                    hist.tag_config(tag, elide=False)
                    if b: b.config(text="[-] Hide Thinking")
                else:
                    hist.tag_config(tag, elide=True)
                    if b: b.config(text="[+] View Thinking Process")
            
            btn_bg = THEME.get("button_bg_color", "#24201c")
            accent = THEME.get("electric_blue", "#00bfff")
            btn_active = THEME.get("button_active_color", "#382e24")
            accent_hl = THEME.get("accent_highlight", "#ff8800")
            
            btn = tk.Button(hist, text="[-] Hide Thinking", bg=btn_bg, fg=accent, 
                            activebackground=btn_active, activeforeground=accent_hl, 
                            relief=tk.FLAT, font=self.fonts["stats"], cursor="hand2", padx=6, pady=2)
            btn.config(command=lambda t=think_tag, b=btn: toggle_thoughts(t, b))
            if not hasattr(self, 'thought_dropdown_buttons'):
                self.thought_dropdown_buttons = []
            self.thought_dropdown_buttons.append(btn)
            self.state["current_think_btn"] = btn
            
            hist.config(state='normal')
            hist.insert(tk.END, "\n", ("ai",))
            hist.window_create(tk.END, window=btn)
            hist.insert(tk.END, "\n", ("ai",))
            hist.tag_config(think_tag, elide=False, lmargin1=20, lmargin2=20)
            hist.config(state='disabled')

    def _update_thought_dropdown(self, chunk):
        """Inserts streamed thought chunk into the chat dropdown."""
        hist = self.chat_history
        if hist is None or not chunk: return
        self._ensure_thought_dropdown()
        tag = self.state.get("current_think_tag")
        if not tag: return
        
        clean_chunk = re.sub(
            r'(?i)<think>|<thought>|\[DRAFT\]|<\|channel>thought|<channel\|thought>|<channel\s*\|?>|<\/think>|<\/thought>|<\/\|think\|>|<\|think\|>|<\|im_start\|?>thought|<\|im_end\|?>|\[\/DRAFT\]|<\|channel>text|<\|channel>assistant|<channel\|>|<\/channel\|>|<\|start\|>assistant\s+to=user(?:<\|message\|>)?|<\|start\|>assistant\s+to=self(?:<\|message\|>)?|to=self<\|message\|>|to=user<\|message\|>|<\|eom\|>|<\|eot\|>',
            '',
            chunk
        )
        if not self.state.get("thought_stream_lead_cleaned"):
            clean_lower = clean_chunk.lower()
            if clean_lower.startswith("thought\r\n") or clean_lower.startswith("thought\n"):
                clean_chunk = clean_chunk[9:] if clean_lower.startswith("thought\r\n") else clean_chunk[8:]
                self.state["thought_stream_lead_cleaned"] = True
            elif clean_lower.startswith("thought:") or clean_lower.startswith("thought "):
                clean_chunk = clean_chunk[8:]
                self.state["thought_stream_lead_cleaned"] = True
            elif clean_lower.strip() == "thought":
                clean_chunk = ""
            elif clean_chunk.strip():
                clean_chunk = re.sub(r'(?i)^thought\s*', '', clean_chunk)
                self.state["thought_stream_lead_cleaned"] = True

        if not clean_chunk: return
        
        is_at_bottom = hist.yview()[1] >= 0.98
        hist.config(state='normal')
        hist.insert(tk.END, clean_chunk, (tag, "md_thought"))
        hist.config(state='disabled')
        if is_at_bottom:
            hist.yview_moveto(1.0)

    def _buffer_thought_log(self, content):
        """Buffers live reasoning/thought tokens and flushes them to thought_log."""
        if not content: return
        self.log_update_buffer += content
        self._flush_log_buffer()

    def _buffer_tool_log(self, content):
        if not content: return
        if self.tool_log and self.tool_log.winfo_exists():
            try:
                self.tool_log.config(state='normal')
                self.tool_log.insert(tk.END, content + "\n\n", "stdout")
                if self.tool_log.yview()[1] >= 0.9:
                    self.tool_log.see(tk.END)
                self.tool_log.config(state='disabled')
            except: pass

    def _buffer_diag_log(self, content):
        if not content: return
        if self.diag_log and self.diag_log.winfo_exists():
            try:
                self.diag_log.config(state='normal')
                self.diag_log.insert(tk.END, content + "\n", "diag")
                if self.diag_log.yview()[1] >= 0.9:
                    self.diag_log.see(tk.END)
                self.diag_log.config(state='disabled')
            except: pass

    def _buffer_log(self, content):
        """Update the System/Error log (⚠)."""
        if not content: return
        if self.error_log and self.error_log.winfo_exists():
            try:
                self.error_log.config(state='normal')
                self.error_log.insert(tk.END, content + "\n", "stderr")
                if self.error_log.yview()[1] >= 0.9:
                    self.error_log.see(tk.END)
                self.error_log.config(state='disabled')
            except: pass

    def _flush_log_buffer(self, force=False):
        now = time.time()
        if self.log_update_buffer and (force or now - self.last_log_dispatch > 0.2 or len(self.log_update_buffer) > 1000):
            # MISSION: Safety limit to prevent UI lock if buffer is massive
            if len(self.log_update_buffer) > 20000:
                self.log_update_buffer = self.log_update_buffer[:10000] + "\n[LOG TRUNCATED FOR STABILITY]\n" + self.log_update_buffer[-5000:]
            
            if self.thought_log and self.thought_log.winfo_exists():
                try:
                    self.thought_log.config(state='normal')
                    self.thought_log.insert(tk.END, self.log_update_buffer, "stdout")
                    if self.thought_log.yview()[1] >= 0.9:
                        self.thought_log.see(tk.END)
                    self.thought_log.config(state='disabled')
                except: pass
            
            self.log_update_buffer = ""
            self.last_log_dispatch = now
        
        # Throttling maintenance
        #self.log_update_limit = 20

    def check_process_queue(self):
        """Apex Heartbeat: Processes worker messages with high-frequency UI safety."""
        try:
            now = time.time()
            processed_count = 0
            q_size = self.process_queue.qsize()
            if q_size > self.log_update_limit:
                print(f"[SYSTEM] Queue congestion detected: {q_size} messages pending.")
            
            # 1. Process messages up to the limit to prevent UI stutter
            while not self.process_queue.empty() and processed_count < self.log_update_limit:
                msg = self.process_queue.get_nowait()
                status = msg.get("status")
                handler = self.queue_handlers.get(status)
                
                if handler:
                    handler(msg)
                else:
                    print(f"Warning: No handler for status '{status}'")
                
                self.process_queue.task_done()
                processed_count += 1

            # 2. Batch Log/Text Updates (The Performance Secret)
            self._flush_log_buffer()

            if self.thought_stream_buffer:
                self._update_thought_dropdown(self.thought_stream_buffer)
                self.thought_stream_buffer = ""

            should_update = False
            mode = self.state.get("streaming_mode", "Buffered")
            
            if self.text_buffer:
                if mode == "Real-time":
                    should_update = True
                elif mode == "Buffered":
                    # Optimized: updates on sentence boundaries, newlines, or time/length limits
                    if any(c in self.text_buffer for c in ['. ', '! ', '? ', '\n', '\r']) or \
                       len(self.text_buffer) > 80 or (now - self.last_update_time) > 0.3:
                        should_update = True
                elif mode == "Experimental Chunking":
                    # Heavy: updates only on large chunks or 1.5s
                    if len(self.text_buffer) > 350 or (now - self.last_update_time) > 1.5:
                        should_update = True
                elif mode == "Mass Dump":
                    should_update = False # Handled at end of session
            
            if should_update:
                 self._update_ai_message(self.text_buffer)
                 self.text_buffer = ""
                 self.last_update_time = now

            # 3. Dynamic Scaling (Faster ticks when busy)
            has_pending_logs = len(self.log_update_buffer) > 0
            is_busy = self.state["running"] or (SYSTEM_MONITOR_LOADED and 
                                               getattr(self, 'stats_thread', None) and 
                                               self.stats_thread.is_alive())
            
            if is_busy:
                # Optimized Heartbeat: 35ms gives the UI breathing room while maintaining perceived speed
                interval = 35 if (processed_count >= self.log_update_limit or has_pending_logs) else 60
                self.root.after(interval, self.check_process_queue)
            else:
                # Idle mode: slow down the clock to save energy
                self.root.after(150, self.check_process_queue)

        except Exception as e:
            print(f"[APEX ERROR] Heartbeat failure: {e}\n{traceback.format_exc()}")
            self.root.after(500, self.check_process_queue) # Safety backup

    def update_timeline_progress(self, current_burst, total_bursts):
        percentage = (current_burst / total_bursts) * 100
        # Thread-safe UI update
        self.root.after(0, self._set_progress, percentage)

    def _set_progress(self, val):
        if self.timeline_bar is None or not self.timeline_bar.winfo_exists(): return
        if self.timeline_frame is None or not self.timeline_frame.winfo_viewable():
            if self.timeline_frame is not None:
                if self.chat_history is not None and self.chat_history.winfo_exists():
                    self.timeline_frame.pack(side=tk.TOP, fill="x", padx=10, pady=2, before=self.chat_history)
                else:
                    self.timeline_frame.pack(side=tk.TOP, fill="x", padx=10, pady=2)
        
        if self.timeline_bar is not None:
             self.timeline_bar['value'] = val
        if self.progress_label is not None:
             self.progress_label.config(text=f"TIMELINE: {val:.1f}%")
        
        if val >= 100 and self.progress_label is not None:
            self.progress_label.config(text="TIMELINE: ANALYSIS COMPLETE", fg="#ffffff")

    def _display_user_message(self, msg): 
        hist = self.chat_history
        if hist is None: return
        
        # Consolidate: Hide the separate prompt display
        if hasattr(self, "prompt_display") and self.prompt_display.winfo_exists():
            self.prompt_display.pack_forget()

        hist.config(state='normal')
        start_idx = hist.index(tk.END + "-1c")
        
        self._append_to_chat(f"\nYou: {msg}\n", "user")
        
        end_idx = hist.index(tk.END + "-1c")
        self.state["response_start_idx"] = end_idx
        hist.config(state='disabled')
        
        # Universal Markdown Application
        render_mode = self.config.get("media_rendering", 1)
        if render_mode > 0:
            self._apply_markdown(start_idx, end_idx, ("user",))
        
        hist.see(tk.END)

    def _update_ai_message(self, chunk):
        hist = self.chat_history
        if not chunk or hist is None: return
        
        is_at_bottom = hist.yview()[1] >= 0.98

        if not self.state.get("response_started", False):
            think = self.thinking_display
            if think and think.winfo_exists(): think.set_phase("generating")
            self.state["response_start_idx"] = hist.index(tk.END + "-1c")
            self._append_to_chat(f"\n\n{self._get_persona_label()}: ", "ai_lead")
            self.state["response_started"] = True
            if self.active_persona_level <= 3:
                self.set_avatar_state("explain_direct")
            elif self.active_persona_level in [4, 5]:
                self.set_avatar_state("explain_wise")
        elif self.state.get("current_think_tag") and not self.state.get("thought_separator_inserted"):
            self.state["thought_separator_inserted"] = True
            hist.config(state='normal')
            hist.insert(tk.END, "\n\n", ("ai",))
            hist.config(state='disabled')
            think = self.thinking_display
            if think and think.winfo_exists(): think.set_phase("generating")

        tag_name = f"chunk_{self.chunk_counter}"
        self.chunk_counter += 1

        hist.config(state='normal')
        hist.insert(tk.END, chunk, (tag_name, "ai"))
        hist.config(state='disabled')
        
        if is_at_bottom:
            hist.yview_moveto(1.0)
            
        bg = hist.cget("bg")
        fg = CHAT_FG_COLORS.get(self.active_persona_level, "#ffffff")
        
        # Initial visibility: Start with a slight offset from bg if fade is slow
        hist.tag_config(tag_name, foreground=bg)
        # Ensure fg is not exactly bg if we want it to be visible after fade
        if fg.lower() == bg.lower():
            fg = THEME["fg_color"] 
            
        self.root.after(10, lambda *args: self._animate_text_fade(tag_name, bg, fg, steps=8))

    def _replace_ai_message(self, text):
        hist = self.chat_history
        if hist is None: return
        is_at_bottom = hist.yview()[1] >= 0.98

        if not self.state.get("response_started", False):
            think = self.thinking_display
            if think and think.winfo_exists(): think.set_phase("generating")
            self.state["response_start_idx"] = hist.index(tk.END + "-1c")
            self._append_to_chat(f"\n\n{self._get_persona_label()}: ", "ai_lead")
            self.state["response_started"] = True
            self.state["response_content_start_idx"] = hist.index(tk.END + "-1c")

        start_idx = self.state.get("response_content_start_idx")
        if not start_idx:
            start_idx = self.state.get("response_start_idx")
            
        hist.config(state='normal')
        if start_idx:
            hist.delete(start_idx, tk.END)
        if text:
            hist.insert(tk.END, text, ("ai",))
        hist.config(state='disabled')
        if is_at_bottom:
            hist.yview_moveto(1.0)
    
    def _display_ai_message(self, msg="", is_streaming=True):
        if is_streaming:
            self.state["response_started"] = False
            self.state["current_think_tag"] = None
            self.state["current_think_btn"] = None
            self.state["thought_streamed"] = False
            self.state["thought_separator_inserted"] = False
            self.state["thought_stream_lead_cleaned"] = False
            self.thought_stream_buffer = ""
            think = self.thinking_display
            if think and think.winfo_exists(): think.start()
        else:
            self._append_to_chat(f"\n\n{self._get_persona_label()}: ", "ai_lead")
            self._append_to_chat(msg, "ai")

    def _append_to_chat(self, text, tag):
        hist = self.chat_history
        if hist is None: return
        is_at_bottom = hist.yview()[1] >= 0.98
        
        hist.config(state='normal')
        hist.insert(tk.END, text, (tag,))
        
        # Memory overhead reduction: Truncate chat history UI if it grows too massive (e.g. 5000 lines)
        if int(float(hist.index('end-1c'))) > 5000:
            hist.delete("1.0", "1000.0")
            
        hist.config(state='disabled')
        
        if is_at_bottom or tag in ["user", "ai_lead", "system"]:
            hist.yview_moveto(1.0)
        
        bg = hist.cget("bg")
        if tag == "user": fg = "#007acc"
        elif tag == "ai": fg = CHAT_FG_COLORS.get(self.active_persona_level, "#ffffff")
        else: fg = THEME["fg_color"]
        
        hist.tag_config(tag, foreground=fg)

    def _apply_markdown(self, start_idx, end_idx, base_tags=("ai",), is_thought=False):
        """Processes Markdown formatting (bold, italic, lists, code, tables, math) in a fast, single-pass atomic render."""
        hist = self.chat_history
        if hist is None or not hist.winfo_exists(): return
        
        try:
            raw_text = hist.get(start_idx, end_idx)
            if not raw_text: return
            
            # Parse text into tagged spans via MarkdownEngine
            spans = MarkdownEngine.parse_to_spans(raw_text, base_tags=base_tags, is_thought=is_thought)
            if not spans: return
            
            # Atomic single-pass replacement in Tkinter Text widget
            hist.config(state='normal')
            hist.delete(start_idx, end_idx)
            
            insert_pos = hist.index(start_idx)
            for text_chunk, tags in spans:
                hist.insert(insert_pos, text_chunk, tags)
                insert_pos = hist.index(f"{insert_pos} + {len(text_chunk)} chars")
                
            hist.config(state='disabled')
        except Exception as e:
            print(f"[UI SAFETY] Markdown render error: {e}")


    def _post_process_media(self, start_idx=None):
        """Scans the last message for URLs and media paths using native search."""
        hist = self.chat_history
        if hist is None: return
        
        render_mode = self.config.get("media_rendering", 1)
        # MISSION: Only scan the NEW content. Never scan the entire history.
        if start_idx is None:
            start_idx = self.state.get("response_start_idx", "1.0")
        
        # Regex for Web URLs and local absolute paths (Windows)
        url_pattern = r'(https?://[^\s<>"]+|[a-zA-Z]:\\[^\s<>"]+\.(?:png|jpg|jpeg|gif|webp|mp4|avi|mkv|mov))'
        
        hist.config(state='normal')
        search_idx = start_idx
        while True:
            self.root.update_idletasks()
            # Native Tcl search
            match_start = hist.search(url_pattern, search_idx, stopindex=tk.END, regexp=True)
            if not match_start: break
            
            # Advance search_idx IMMEDIATELY to prevent infinite loop traps
            search_idx = f"{match_start} + 1c"
            
            # Extract a generous tail for Python-side validation
            tail = hist.get(match_start, f"{match_start} + 1000 chars")
            m = re.search(url_pattern, tail)
            if not m:
                continue
            
            url = m.group(0).strip(".,!?;:")
            match_end = hist.index(f"{match_start} + {len(url)} chars")
            
            # Check if it's media (images or videos)
            is_media = any(url.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".gif", ".webp", ".mp4", ".avi", ".mkv", ".mov"])
            
            if is_media and render_mode > 0:
                if render_mode == 1: # Inline
                    # 1. Prepare Placeholder
                    if not self.placeholder_img:
                        p_pil = Image.new("RGB", (300, 180), color="#1a1a1a")
                        self.placeholder_img = ImageTk.PhotoImage(p_pil)
                    
                    # 2. Delete the raw URL text and insert the placeholder
                    hist.delete(match_start, match_end)
                    img_name = hist.image_create(match_start, image=self.placeholder_img, padx=10, pady=10)
                    
                    # 3. Insert a small clickable link reference below it
                    self.link_counter += 1
                    link_tag = f"link_media_{self.link_counter}"
                    hist.insert(match_start, f"\n[Media: {os.path.basename(url)}]\n", (link_tag,))
                    hist.tag_config(link_tag, foreground="#00ccff", underline=1)
                    hist.tag_bind(link_tag, "<Button-1>", lambda e, u=url: webbrowser.open(u))
                    
                    # 4. Spawn the population thread
                    threading.Thread(target=self._fetch_and_populate_media, args=(url, img_name), daemon=True).start()
                
                elif render_mode == 2: # Separate (Popup button)
                    self.link_counter += 1
                    tag = f"link_popup_{self.link_counter}"
                    hist.delete(match_start, match_end)
                    hist.insert(match_start, f" [View Media: {os.path.basename(url)}] ", (tag,))
                    hist.tag_config(tag, foreground="#ffcc00", underline=1, font=self.fonts["ui_button"])
                    hist.tag_bind(tag, "<Button-1>", lambda e, u=url: self._spawn_media_popup(u))
            else:
                # Standard Text Link
                self.link_counter += 1
                tag = f"link_text_{self.link_counter}"
                hist.tag_add(tag, match_start, match_end)
                hist.tag_config(tag, foreground="#00ccff", underline=1)
                hist.tag_bind(tag, "<Button-1>", lambda e, u=url: webbrowser.open(u))
        
        hist.config(state='disabled')

    def _fetch_and_populate_media(self, url, img_widget_name):
        """Background thread to fetch media and update the PhotoImage reference."""
        try:
            pil_img = None
            if url.startswith("http"):
                if is_offline_mode() or bool(self.config.get("offline_mode", False)):
                    print(f"[OFFLINE MODE] Blocked external media fetch for: {url}")
                    return
                # Web Fetch
                headers = {"User-Agent": "SerenityPC/4.0 RichMediaFetcher"}
                resp = requests.get(url, headers=headers, timeout=10)
                resp.raise_for_status()
                pil_img = Image.open(io.BytesIO(resp.content))
            else:
                # Local Path
                if any(url.lower().endswith(ext) for ext in [".mp4", ".avi", ".mkv", ".mov"]) and cv2:
                    # Video Frame Extraction
                    cap = cv2.VideoCapture(url)
                    ret, frame = cap.read()
                    if ret:
                        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    cap.release()
                elif os.path.exists(url):
                    pil_img = Image.open(url)
            
            if pil_img:
                # Resize for chat UI (Max width 450, max height 600)
                pil_img.thumbnail((450, 600))
                photo = ImageTk.PhotoImage(pil_img)
                
                # Keep reference to prevent garbage collection
                self.media_cache[img_widget_name] = photo
                
                # Update UI (Must be on main thread)
                def update():
                    if self.chat_history:
                        try:
                            self.chat_history.image_configure(img_widget_name, image=photo)
                        except: pass
                self.root.after(0, update)
                
        except Exception as e:
            print(f" > [MEDIA ERROR] Failed to populate {url}: {e}")

    def _spawn_media_popup(self, url):
        """Opens the media in a dedicated, borderless-style Toplevel window."""
        popup = tk.Toplevel(self.root)
        popup.title(f"Media Viewer - {os.path.basename(url)}")
        popup.configure(bg="#0a0a0a")
        
        # Basic Viewer Logic
        loading = tk.Label(popup, text="Loading...", fg="white", bg="#0a0a0a", font=self.fonts["large"])
        loading.pack(padx=50, pady=50)
        
        def load():
            try:
                # (Re-use fetch logic or just open)
                if url.startswith("http"):
                    if is_offline_mode() or bool(self.config.get("offline_mode", False)):
                        loading.config(text="[OFFLINE MODE] External media fetch blocked.")
                        return
                    resp = requests.get(url, timeout=10); pil_img = Image.open(io.BytesIO(resp.content))
                else:
                    if any(url.lower().endswith(ext) for ext in [".mp4", ".avi", ".mkv", ".mov"]) and cv2:
                        cap = cv2.VideoCapture(url); ret, frame = cap.read()
                        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) if ret else None; cap.release()
                    else: pil_img = Image.open(url)
                
                if pil_img:
                    # Scale to fit screen but keep aspect ratio
                    screen_w = self.root.winfo_screenwidth() * 0.8
                    screen_h = self.root.winfo_screenheight() * 0.8
                    pil_img.thumbnail((screen_w, screen_h))
                    
                    photo = ImageTk.PhotoImage(pil_img)
                    self.media_cache[f"popup_{id(popup)}"] = photo # Keep ref
                    
                    def show():
                        loading.destroy()
                        lbl = tk.Label(popup, image=photo, bg="#0a0a0a")
                        lbl.pack()
                        # Add a "Click to Close" behavior
                        lbl.bind("<Button-1>", lambda e: popup.destroy())
                    self.root.after(0, show)
            except Exception as e:
                self.root.after(0, lambda: loading.config(text=f"Error: {e}"))
        
        threading.Thread(target=load, daemon=True).start()


    def _log_and_display(self, msg): 
        print(f"System: {msg}") 
        lbl = self.system_status_label
        if lbl is not None and lbl.winfo_exists():
            lbl.config(text=f"System: {msg}")
            status_timer = self._status_timer
            if status_timer is not None:
                self.root.after_cancel(status_timer)
                self._status_timer = None
            self._status_timer = self.root.after(5000, lambda *args: self._revert_status_label())
        if hasattr(self, 'thinking_display') and self.thinking_display and self.thinking_display.winfo_exists():
            self.thinking_display.update_status(f"System: {msg}")

    def _run_tool_loop(self, full_resp, prompt_str, params, depth=0):
        """
        Parses tool calls from model output (supporting Programmatic Tool Calling Python syntax
        and legacy tags), executes them, and recursively generates the final answer.
        Max depth 3 to prevent runaway inference.
        """
        if depth > 3:
            return f"{full_resp}\n\n[SYSTEM]: Tool recursion limit reached. Truncating response."

        call_name = None
        args = {}

        # 1. Programmatic Tool Calling (PTC) - Python syntax (e.g. web_search("..."), read_file(path="..."))
        py_match = re.search(r'(?:```(?:python)?\s*)?\b(web_search|read_file|get_system_stats|control_rgb|generate_image)\s*\((.*?)\)(?:\s*```)?', full_resp, re.DOTALL | re.IGNORECASE)
        # 2. Legacy / Tag-based Tool Calls
        call_match = re.search(r'(?:<ctrl42>call:|<\|tool_call>call:|<\|tool_call\|>call:|<\|tool>call:|call:|action:|<(?:channel\|)?(?:execute_tool|executetool)>)\s*([\w_]+)\s*\{(.*?)\}(?:<\/(?:execute_tool|executetool)>)?', full_resp, re.DOTALL | re.IGNORECASE)
        # 3. JSON Action Block (e.g. {"action": "generate_image", "action_input": ...} or {"name": "...", "arguments": ...})
        json_match = re.search(r'\{\s*["\'](?:action|tool|name|function)["\']\s*:\s*["\'][\w_]+["\']', full_resp, re.IGNORECASE)

        if py_match:
            call_name = py_match.group(1).strip().lower()
            raw_args = py_match.group(2).strip()
            if raw_args:
                import ast
                try:
                    expr = ast.parse(f"{call_name}({raw_args})", mode="eval")
                    if isinstance(expr.body, ast.Call):
                        for kw in expr.body.keywords:
                            args[kw.arg] = ast.literal_eval(kw.value)
                        if expr.body.args:
                            if call_name == "web_search": args["query"] = ast.literal_eval(expr.body.args[0])
                            elif call_name == "read_file": args["path"] = ast.literal_eval(expr.body.args[0])
                            elif call_name == "generate_image": args["prompt"] = ast.literal_eval(expr.body.args[0])
                except Exception:
                    clean_str = raw_args.strip('\'" ')
                    if call_name == "web_search": args["query"] = clean_str
                    elif call_name == "read_file": args["path"] = clean_str
                    elif call_name == "generate_image": args["prompt"] = clean_str
        elif call_match:
            call_name = call_match.group(1).strip()
            if call_name.lower() in ["readfile", "read_file"]:
                call_name = "read_file"
            args_raw = call_match.group(2).strip()
            
            clean_args = args_raw.replace('<|"|>', '"').replace('<|"', '"').replace('"|>', '"').strip()
            if clean_args.startswith("{") and clean_args.endswith("}"):
                clean_args = clean_args[1:-1].strip()
                
            is_json_format = not clean_args or clean_args.strip().startswith('"')
            if is_json_format:
                try:
                    args = json.loads("{" + clean_args + "}")
                except Exception: pass

            if not args and clean_args:
                try:
                    kv_pattern = r'(?m)^\s*["\']?([\w_]+)["\']?\s*[:=]\s*(.*?)\s*(?:,|$)'
                    matches = re.finditer(kv_pattern, clean_args)
                    for m in matches:
                        key = m.group(1).strip()
                        val = m.group(2).strip().strip('"\'')
                        if val: args[key] = val
                    if not args:
                        fixed = re.sub(r'(?<!["\'])(\b\w+\b)(?!["\'])\s*:', r'"\1":', clean_args)
                        fixed = re.sub(r',\s*$', '', fixed.strip())
                        args = json.loads("{" + fixed + "}")
                except Exception:
                    try:
                        bare_val = re.sub(r'^.*?:\s*', '', clean_args).strip().strip('"\'')
                        if bare_val:
                            if call_name == "web_search": args["query"] = bare_val
                            elif call_name == "read_file": args["path"] = bare_val
                    except Exception: pass
        elif json_match:
            start_idx = json_match.start()
            brace_count = 0
            end_idx = -1
            in_str = False
            str_char = ''
            escape = False
            for i in range(start_idx, len(full_resp)):
                ch = full_resp[i]
                if escape:
                    escape = False
                    continue
                if ch == '\\':
                    escape = True
                    continue
                if in_str:
                    if ch == str_char:
                        in_str = False
                elif ch in ('"', "'"):
                    in_str = True
                    str_char = ch
                elif ch == '{':
                    brace_count += 1
                elif ch == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            
            candidate_raw = full_resp[start_idx:end_idx].strip() if end_idx != -1 else full_resp[start_idx:].strip()
            import ast
            parsed_dict = None
            clean_cand = candidate_raw.replace(r"\'", "'")
            try:
                parsed_dict = json.loads(clean_cand)
            except Exception:
                try:
                    parsed_dict = ast.literal_eval(clean_cand)
                except Exception:
                    try:
                        parsed_dict = json.loads(candidate_raw)
                    except Exception:
                        try:
                            parsed_dict = ast.literal_eval(candidate_raw)
                        except Exception: pass
            
            if isinstance(parsed_dict, dict):
                raw_call = parsed_dict.get("action") or parsed_dict.get("tool") or parsed_dict.get("name") or parsed_dict.get("function")
                if raw_call:
                    call_name = str(raw_call).strip().lower()
                    raw_input = parsed_dict.get("action_input") or parsed_dict.get("arguments") or parsed_dict.get("parameters") or parsed_dict.get("input")
                    if isinstance(raw_input, dict):
                        args = raw_input
                    elif isinstance(raw_input, str):
                        clean_inp = raw_input.replace(r"\'", "'")
                        try:
                            args = json.loads(clean_inp)
                        except Exception:
                            try:
                                args = ast.literal_eval(clean_inp)
                            except Exception:
                                try:
                                    args = json.loads(raw_input)
                                except Exception:
                                    try:
                                        args = ast.literal_eval(raw_input)
                                    except Exception:
                                        if not isinstance(args, dict):
                                            args = {}
                                        if call_name == "generate_image": args["prompt"] = raw_input
                                        elif call_name == "web_search": args["query"] = raw_input
                                        elif call_name == "read_file": args["path"] = raw_input
                    elif raw_input is None:
                        args = {k: v for k, v in parsed_dict.items() if k not in ("action", "tool", "name", "function")}

        if not call_name:
            return full_resp

        # Inform UI & clean tool invocation syntax from chat output
        self.process_queue.put({"status": "streaming_replace", "content": ""})
        self.process_queue.put({"status": "thinking_status", "content": f"Executing tool: {call_name}..."})
        self.process_queue.put({"status": "tool_log_update", "content": f"\n[{time.strftime('%H:%M:%S')}] Executing: {call_name}\nArgs: {args}"})
        
        try:
            # 1. ATTEMPT EXECUTION WITH GRACEFUL FALLBACK
            try:
                observation = self.tool_registry.execute(call_name, args)
            except Exception as e:
                observation = f"Notice: Tool '{call_name}' execution encountered an issue ({str(e)}). Proceeding with baseline knowledge."
                self.process_queue.put({"status": "log_update", "content": f"\n[TOOL FALLBACK] {call_name}: {str(e)}\n"})

            self.process_queue.put({"status": "tool_log_update", "content": f"Result: \n{str(observation)[:200]}..."})
            
            # 2. CHAT SYNTHESIS
            clean_resp = full_resp
            clean_resp = re.sub(r'(?s)<think>.*?(?:<\/think>|$)', '', clean_resp, flags=re.IGNORECASE)
            clean_resp = re.sub(r'(?s)<\|channel>thought.*?(?:<channel\|>|$)', '', clean_resp, flags=re.IGNORECASE)
            
            forced_sys = f"{PERSONA_PROMPTS.get(self.active_persona_level, 'You are Serenity.')}\n[DIRECT STRIKE]: Based on the search / tool results below (or using your existing knowledge if results are unavailable), provide a direct, helpful answer to the user's original query."
            if isinstance(prompt_str, list):
                new_prompt = list(prompt_str) + [
                    {"role": "assistant", "content": f"Executed tool `{call_name}` with args: {json.dumps(args)}"},
                    {"role": "user", "content": f"[TOOL RESULT ({call_name})]:\n{observation}\n\nDeliver the final response now based on these results."}
                ]
            else:
                new_prompt = [
                    {"role": "system", "content": forced_sys},
                    {"role": "user", "content": f"Original Query Context:\n{prompt_str[-500:]}\n\nTool Results ({call_name}):\n{observation}\n\nDeliver the final response now."}
                ]
            
            self.process_queue.put({"status": "thinking_status", "content": "Synthesizing tool results..."})
            new_text = self._run_blocking_inference(new_prompt, params)
            
            if not new_text or len(new_text.strip()) < 5:
                new_text = f"Tool Result for `{call_name}`:\n{observation}"

            return self._run_tool_loop(new_text, new_prompt, params, depth=depth+1)
            
        except Exception as e:
            fallback_msg = f"I retrieved tool context for `{call_name}`, but encountered a formatting issue ({str(e)}). Here is the tool output:\n\n{observation if 'observation' in locals() else str(e)}"
            return fallback_msg
      
    def _detect_repetition(self, text, mode=None):
        """
        Detects repetitive generation loops according to the configured mode:
        - 'off': Loop detection disabled (never interrupts).
        - 'lazy': Relaxed detection for coding and repeated tool calling (ignores code blocks/tool syntax, high repetition threshold).
        - 'hyper': Aggressive detection with tight substring matching and stall phrase checks.
        """
        if not text:
            return False
            
        if mode is None:
            mode = self.config.get("repeat_detection_mode", "lazy").lower()
        else:
            mode = str(mode).lower()

        if mode == "off":
            return False

        if mode == "hyper":
            recent = text[-400:]
            n = len(recent)
            
            stall_phrases = ["re-read", "reread", "look again", "let me look", "actually the prompt", "wait, the input"]
            stall_count = sum(recent.lower().count(phrase) for phrase in stall_phrases)
            if stall_count >= 3:
                return True

            min_len = 35
            max_repeats = 3
            if n < min_len * max_repeats:
                return False
            
            seen = set()
            for i in range(n - min_len + 1):
                sub = recent[i:i+min_len]
                if sub in seen:
                    continue
                seen.add(sub)
                if not any(c.isalnum() for c in sub):
                    continue
                if recent.count(sub) >= max_repeats:
                    return True
            return False

        # Default / "lazy" mode:
        # 1. Filter out code blocks and tool calls so repetitive code / tool queries don't trigger false positives
        cleaned = re.sub(r'```[\s\S]*?```', ' [CODE_BLOCK] ', text)
        cleaned = re.sub(r'(?:<ctrl42>call:|<\|tool_call>call:|<\|tool_call\|>call:|<\|tool>call:|call:|action:|<(?:channel\|)?(?:execute_tool|executetool)>)[\s\S]*?(?:<\/(?:execute_tool|executetool)>|\}|\n|$)', ' [TOOL_CALL] ', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\b(?:web_search|read_file|get_system_stats|control_rgb|generate_image)\s*\([\s\S]*?\)', ' [TOOL_CALL] ', cleaned, flags=re.IGNORECASE)

        recent = cleaned[-800:]
        n = len(recent)

        # Check for consecutive repeating identical lines (e.g. spamming the exact same sentence 5+ times)
        lines = [line.strip() for line in recent.split('\n') if len(line.strip()) >= 15]
        if len(lines) >= 5:
            consecutive_count = 1
            for j in range(1, len(lines)):
                if lines[j] == lines[j - 1] and lines[j] not in ["[CODE_BLOCK]", "[TOOL_CALL]"]:
                    consecutive_count += 1
                    if consecutive_count >= 5:
                        return True
                else:
                    consecutive_count = 1

        min_len = 80
        max_repeats = 4
        if n < min_len * max_repeats:
            return False

        seen = set()
        for i in range(n - min_len + 1):
            sub = recent[i:i+min_len]
            if sub in seen:
                continue
            seen.add(sub)
            if not any(c.isalnum() for c in sub):
                continue
            if recent.count(sub) >= max_repeats:
                return True
        return False

    def _protect_long_numbers(self, text):
        """Wraps long continuous sequences of digits to prevent tokenizer collision and hallucination."""
        if not isinstance(text, str):
            return text
        
        def repl(match):
            digits = match.group(0)
            if len(digits) >= 5:
                spaced = " ".join(digits)
                return f"`{digits}` (digits: `{spaced}`)"
            return digits
            
        return re.sub(r'\d{5,}', repl, text)
      
    def _extract_thinking_content(self, text):
        """Robust multi-tag thinking extraction (Deep Cook Prioritized)."""
        if not text: return ""
        import re
        
        # (?s) makes '.' match newlines. (?m) makes '^' match start of any line.
        patterns = [
            # 1. SPECIFIC DEEP COOK HEADERS (Highest Priority)
            r'(?s)--- Cycle \d+ ---\n(.*?)(?=\n--- Cycle|$)',
            r'(?i)\[STATUS:.*?\](.*?)(?=\[STATUS:|$)',
            r'(?i)\[PRIME_MEMORY:(.*?)\]', 
            
            # 2. SEMI-STRUCTURED DRAFTS
            r'(?s)\[DRAFT\]\n?(.*?)(?:\[\/DRAFT\]|$)',
            
            # 3. GENERIC & ATEM TAGS (Fallback)
            r'(?s)<\|start\|>assistant to=self<\|message\|>(.*?)(?:<\|eom\|>|$)',
            r'(?s)<\|channel>thought\n?(.*?)(?:<channel\|?>|<\/\|?channel\|?>|<\|channel>|(?=<start_of_turn>)|$)',
            r'(?s)<think>\n?(.*?)(?:<\/think>|$)',
            r'(?s)<\|think\|>\n?(.*?)(?:<\/\|think\|>|$)',
            r'(?s)<thought>\n?(.*?)(?:<\/thought>|$)',
            r'(?s)<\|im_start|>thought\n?(.*?)(?:<\|im_end\|>|$)',
            
            # Linear / Bulleted Heuristics
            r'(?m)^(?:\d+\.|\*|\-)\s+\*\*\*?(?:Analyze|Determine|Identify|Structure|Refine|Draft|Review|Persona|Goal|Context|Acknowledge|Define|Equations|Methodology|Execution|Complexity|Apply|Deconstruct|Develop|Strategy|Resolution|Structure|Tone Check|Plan).*?\*\*.*(?:\n|$)',
            r'(?i)^(?:The user is asking|This request requires|The goal of this|Based on the persona|Analyzing request).*?\n',
            r'(?i)\[Inferred Conclusion\]:?\s*[^\n]*',
            r'(?m)^\s*\*\s*Thought:?\s*[^\n]*',
            r'(?m)^\s*Thought:?\s*[^\n]*'
        ]
        
        extracted_thoughts = []
        for pattern in patterns:
            # MISSION: Collect all unique reasoning blocks while preserving sequence
            matches = re.findall(pattern, text)
            if matches:
                for m in matches:
                    if m.strip() and m.strip() not in extracted_thoughts:
                        extracted_thoughts.append(m.strip())
        
        thoughts = "\n\n".join(extracted_thoughts).strip()
        if thoughts:
            print(f"[SYSTEM] Extracted {len(extracted_thoughts)} reasoning blocks ({len(thoughts)} chars).")
        return thoughts


    def _clean_latex_artifacts(self, text):
        """Pass-through hygiene: MarkdownEngine handles LaTeX and Math natively during markdown rendering."""
        return text if text else ""

    def _handle_deep_cook_ui_start(self, msg):
        """Initializes a new dropdown block for streaming Deep Cook tokens."""
        hist = self.chat_history
        if hist is None: return
        
        ctype = msg.get("ctype")
        cnum = msg.get("cnum", 1)
        dnum = msg.get("dnum", 1)
        title = msg.get("title", "")
        
        # Use deterministic tag naming for cross-thread consistency
        tag = f"dc_{ctype}_{cnum}_{dnum}"
        self.state["current_deep_cook_active_tag"] = tag
        
        hist.config(state='normal')
        
        if ctype == "cycle":
            if not self.state.get("response_started", False):
                self._append_to_chat(f"\n\n{self._get_persona_label()}: \n", "ai_lead")
            
            self.state["current_deep_cook_cyc_tag"] = tag
            def toggle_cyc(t=tag, b=None, c=cnum):
                st = hist.tag_cget(t, "elide")
                new_state = (st == "0")
                hist.tag_config(t, elide=new_state)
                if b: b.config(text=f"{'[+] View' if new_state else '[-] Hide'} Cycle {c}")
            
            btn = tk.Button(hist, text=f"[+] View Cycle {cnum}", bg="#202020", fg="#888888", relief=tk.FLAT, font=self.fonts["stats"])
            btn.config(command=lambda t=tag, b=btn, c=cnum: toggle_cyc(t, b, c))
            
            hist.insert(tk.END, "\n", ("ai",))
            hist.window_create(tk.END, window=btn)
            hist.insert(tk.END, "\n", ("ai",))
            hist.insert(tk.END, f"{title}\n", (tag, "ai"))
            hist.tag_config(tag, elide=True, foreground="#808080", font=self.fonts["log"])
            self.state[f"nested_tags_{tag}"] = []

        elif ctype == "draft":
            cyc_tag = self.state.get("current_deep_cook_cyc_tag")
            if cyc_tag:
                if f"nested_tags_{cyc_tag}" not in self.state: self.state[f"nested_tags_{cyc_tag}"] = []
                self.state[f"nested_tags_{cyc_tag}"].append(tag)
                def toggle_draft(t=tag, b=None, d=dnum):
                    st = hist.tag_cget(t, "elide")
                    new_state = (st == "0")
                    hist.tag_config(t, elide=new_state)
                    if b: b.config(text=f"{'[+]' if new_state else '[-]'} Step {d}")

                btn = tk.Button(hist, text=f"[+] Step {dnum}", bg="#151515", fg="#777777", relief=tk.FLAT, font=self.fonts["stats"])
                btn.config(command=lambda t=tag, b=btn, d=dnum: toggle_draft(t, b, d))
                
                hist.insert(tk.END, "  ", (cyc_tag, "ai"))
                hist.window_create(tk.END, window=btn)
                hist.insert(tk.END, "\n", (cyc_tag, "ai"))
                hist.insert(tk.END, f"  --- {title} ---\n  ", (tag, cyc_tag, "ai"))
                hist.tag_config(tag, elide=True, foreground="#707070", font=self.fonts["log"])

        elif ctype == "memory":
            def toggle_mem(t=tag, b=None):
                st = hist.tag_cget(t, "elide")
                new_state = (st == "0")
                hist.tag_config(t, elide=new_state)
                if b: b.config(text=f"{'[+]' if new_state else '[-]'} {title or 'Context Assessment'}")

            btn = tk.Button(hist, text=f"[+] {title or 'Context Assessment'}", bg="#1a1a2e", fg="#ababab", relief=tk.FLAT, font=self.fonts["stats"])
            btn.config(command=lambda t=tag, b=btn: toggle_mem(t, b))
            
            hist.insert(tk.END, "\n", ("ai",))
            hist.window_create(tk.END, window=btn)
            hist.insert(tk.END, "\n", ("ai",))
            hist.insert(tk.END, "", (tag, "ai"))
            hist.tag_config(tag, elide=True, foreground="#ababab", font=self.fonts["log"])

        hist.config(state='disabled')
        hist.see(tk.END)

    def _handle_deep_cook_ui_stream(self, msg):
        """Appends tokens to the currently active Deep Cook UI block."""
        hist = self.chat_history
        if hist is None: return
        
        content = msg.get("content", "")
        tag = self.state.get("current_deep_cook_active_tag")
        if not content or not tag: return
        
        hist.config(state='normal')
        
        parent_tag = self.state.get("current_deep_cook_cyc_tag")
        tags = [tag, "ai"]
        if parent_tag and tag != parent_tag:
            tags.insert(1, parent_tag)
            
        hist.insert(tk.END, content, tuple(tags))
        hist.config(state='disabled')

    def _handle_deep_cook_ui_batch(self, msg):
        """Finalizes a Deep Cook block by replacing streamed text with cleaned content."""
        hist = self.chat_history
        if hist is None: return
        
        ctype = msg.get("ctype")
        cnum = msg.get("cnum", 1)
        dnum = msg.get("dnum", 1)
        text = msg.get("text", "")
        
        # MISSION: Surgical Replace for Reasoning Dropdowns
        if ctype in ["cycle", "draft", "memory"]:
            tag = f"dc_{ctype}_{cnum}_{dnum}"
            ranges = hist.tag_ranges(tag)
            if ranges:
                hist.config(state='normal')
                parent_tag = self.state.get("current_deep_cook_cyc_tag")
                tags = [tag, "ai"]
                if parent_tag and tag != parent_tag:
                    tags.insert(1, parent_tag)
                
                # Replace streamed content with cleaned batch
                start, end = ranges[0], ranges[-1]
                hist.delete(start, end)
                hist.insert(start, text + "\n", tuple(tags))
                hist.config(state='disabled')
                return

        if ctype == "compilation":
            if not self.state.get("response_started", False):
                    self._append_to_chat(f"\n\n{self._get_persona_label()}: ", "ai_lead")
            title = msg.get("title", "")
            if title:
                hist.config(state='normal')
                hist.insert(tk.END, f"\n{title}\n", ("ai", "system"))
                hist.config(state='disabled')
            if text:
                comp_tag = f"comp_block_{id(text[:20])}"
                hist.config(state='normal')
                hist.insert(tk.END, f"{text}\n\n", (comp_tag, "ai"))
                hist.tag_config(comp_tag, foreground="#00ffcc", font=self.fonts["log"])
                hist.config(state='disabled')
        
        hist.see(tk.END)

    def _revert_status_label(self):
        if hasattr(self, 'system_status_label') and self.system_status_label.winfo_exists():
            if self.model_path:
                model_name = os.path.basename(self.model_path)
                self.system_status_label.config(text=f"Loaded: {model_name}")
            else:
                self.system_status_label.config(text="System: Idle")
        if hasattr(self, 'thinking_display') and self.thinking_display and self.thinking_display.winfo_exists():
            if not self.thinking_display._is_active:
                if self.model_path:
                    model_name = os.path.basename(self.model_path)
                    self.thinking_display.update_status(f"Loaded: {model_name}")
                else:
                    self.thinking_display.update_status("System: Ready")

    def launch_lore_book(self):
        try:
            script_path = os.path.join(TOOLS_DIR, "lore_book.py")
            if os.path.exists(script_path):
                subprocess.Popen([sys.executable, script_path])
            else:
                messagebox.showerror("Error", f"lore_book.py not found in {TOOLS_DIR}")
        except Exception as e:
            self._log_and_display(f"Lore Book Error: {e}")


    def initiate_video_multimodal(self):
        files = filedialog.askopenfilenames(title="Select Video(s)", filetypes=[("Videos", "*.mp4 *.mkv *.avi *.mov")])
        if files:
            for f in files:
                self._add_staged_attachment(f, "video")
            self.set_ui_state()

    def toggle_auto_watch(self):
        """Now functions as the Level 7 xMemory Aggregator Pulse."""
        if self.state.get("xmemory_active"):
            self.state["xmemory_active"] = False
            self.btn_watch.config(text="[🧠] Pulse", bg=THEME["button_bg_color"])
            self._log_and_display("Pulse Aggregator: Standby.")
        else:
            self.state["xmemory_active"] = True
            self.btn_watch.config(text="[🔥] Aggregating", bg="#4a0000")
            self._log_and_display("xMemory active: Decoupling semantic math nodes...")
            threading.Thread(target=self._dmn_pondering_cycle, daemon=True).start()

    def _dmn_pondering_cycle(self):
        """The Transcendent DMN: Background reflection on current project logic."""
        last_simmer = 0
        while self.state.get("xmemory_active"):
            # MISSION: Only simmer when idle to prevent GPU/thread collision
            if not self.state["running"] and self.model:
                try:
                    # 1. Semantic Node Extraction (Workspace Awareness)
                    nodes = []
                    for root, dirs, files in os.walk(self.script_dir):
                        # Filter out noise
                        if any(x in root for x in [".git", "node_modules", "__pycache__", ".gemini", "tmp"]): 
                            continue
                        for f in files:
                            if f.endswith(('.py', '.json', '.md', '.agent.md')):
                                rel_p = os.path.relpath(os.path.join(root, f), self.script_dir)
                                nodes.append(rel_p)
                                # Semantic Peeking: Read architectural files for high-level insights
                                if len(nodes) < 8 and os.path.getsize(os.path.join(root, f)) < 10000:
                                    try:
                                        with open(os.path.join(root, f), 'r', encoding='utf-8') as sf:
                                            peek = sf.read(600)
                                            recent_context += f"\n[PEEK {f}]: {peek}\n"
                                    except: pass
                        if len(nodes) > 60: break # Performance cap for shallow scan
                    
                    # 2. Context Integration (History DNA)
                    path = self.get_history_path()
                    recent_context = ""
                    if path and os.path.exists(path):
                        try:
                            with open(path, 'rb') as f:
                                history = json.loads(zlib.decompress(f.read()).decode('utf-8'))
                            # Consolidation: Analyzes broader history (20 messages) for narrative backbone
                            recent_context = " ".join([m['content'] for m in history[-20:]])
                        except: pass

                    # 3. Simmer Phase: Active Background Reflection
                    now = time.time()
                    # Dynamic Simmer Cooldown: 5m if idle and xMemory active, else 10m
                    cooldown = 300 if (TRI_ATTENTION_ENABLED and not self.state["running"]) else 600
                    if now - last_simmer > cooldown:
                        print(f"[DMN] Initiating Simmer cycle on {len(nodes)} semantic nodes.")
                        self.process_queue.put({"status": "log_update", "content": f"\n[DMN] Simmering on {len(nodes)} workspace nodes...\n"})
                        
                        # Phase 2: Narrative Integrity & Consolidation
                        simmer_prompt = (
                            "You are the Default Mode Network for SerenityPC. "
                            "Your purpose is to organize background thoughts to enrich future responses. "
                            "Analyze the current workspace architecture and the long-term interaction trajectory.\n\n"
                            f"[WORKSPACE NODES]: {', '.join(nodes[:20])}\n"
                            f"[CONSOLIDATED CONTEXT]: {recent_context[:1500]}\n\n"
                            "Synthesize a 'Backbone' summary with three facets:\n"
                            "1. Workspace State: Architectural insights and current focus.\n"
                            "2. Narrative State: Current tone and interaction momentum.\n"
                            "3. Technical Debt: Unresolved questions or pending tasks.\n"
                            "Output ONLY the backbone summary. Be concise and profound."
                        )
                        
                        # Use low-priority parameters for background inference
                        params = {"max_tokens": 100, "temperature": 0.4, "top_p": 0.9}
                        summary = self._run_blocking_inference(simmer_prompt, params)
                        
                        if summary and not self.state["running"]: # Final check before commit
                            summary_clean = summary.strip().replace("\n", " ")
                            self.state["dmn_backbone"]["last_simmer"] = summary_clean
                            self.state["dmn_backbone"]["timestamp"] = time.time()
                            self.state["dmn_backbone"]["node_count"] = len(nodes)
                            self._save_dmn_backbone()
                            self.process_queue.put({"status": "log_update", "content": f"[DMN] Backbone Optimized: {summary_clean}\n"})
                            last_simmer = now
                        
                except Exception as e:
                    print(f"[DMN ERROR] Cycle failure: {e}")
            
            # Idle sleep duration
            sleep_dur = 120 if self.state["running"] else 60
            time.sleep(sleep_dur)

    def _reset_multimodal_ui(self):
        self.state["staged_multimodal"] = None
        self.state["processing_queue"] = []
        self.state["processing_multimodal"] = False
        self.btn_video.config(text="[🎥] Video", bg=THEME["button_bg_color"], state="normal")
        
        # Reset staged attachments
        if self.state.get("staged_attachments"):
            for a in list(self.state["staged_attachments"]):
                if "token_frame" in a and a["token_frame"].winfo_exists():
                    a["token_frame"].destroy()
            self.state["staged_attachments"] = []
        if hasattr(self, "attachment_frame") and self.attachment_frame is not None:
            self.attachment_frame.pack_forget()

        # Reset Vision Model if loaded
        if self.current_model_tier and self.current_model_tier.startswith("vision_"):
            self.offload_model()
            
        print("[SYSTEM] Multimodal staging, queue, and vision model cleared.")
        if hasattr(self, 'timeline_frame') and self.timeline_frame is not None and self.timeline_frame.winfo_exists():
            self.timeline_frame.pack_forget()
            if self.timeline_bar is not None:
                self.timeline_bar['value'] = 0
            if self.progress_label is not None:
                self.progress_label.config(text="TIMELINE: 0%", fg="#00ffcc")
        self.set_ui_state()

    def initiate_vision_analysis(self, mode, file_path, user_msg=None):
        """Spawns tactical scout vision worker."""
        if self.state["running"]: return
        
        self.state["processing_multimodal"] = True
        self._set_progress(0)
        
        target_tier = f"vision_{mode}"
        tier = self.current_model_tier
        if tier is not None and tier != target_tier:
             if not self.model_paths.get(target_tier):
                messagebox.showerror("Error", f"Vision model for {mode} not set!")
                return
             self._log_and_display(f"Switching to Vision Engine ({mode})...")
             self.pending_task = {"type": "vision_standard", "message": user_msg, "staged": {"type": mode, "path": file_path}}
             self.model_swap(target_tier=target_tier)
             return

        # Clear staging attachments from the UI and state now that analysis is starting
        if self.state.get("staged_attachments"):
            for a in list(self.state["staged_attachments"]):
                if a["type"] != "document":
                    if "token_frame" in a and a["token_frame"].winfo_exists():
                        a["token_frame"].destroy()
                    self.state["staged_attachments"].remove(a)
        if hasattr(self, "attachment_frame") and self.attachment_frame is not None:
            if not any(a for a in self.state.get("staged_attachments", []) if a["type"] != "document"):
                self.attachment_frame.pack_forget()

        queue = self.state.get("processing_queue", [])
        if not queue:
            if isinstance(file_path, list):
                queue = list(file_path)
            else:
                queue = [file_path]
        
        self._log_and_display(f"Batch analysis in progress...")
        threading.Thread(target=self._batch_vision_worker, args=(mode, queue, user_msg, False), daemon=True).start()
        self.root.after(100, lambda *args: self.check_process_queue())

    def _batch_vision_worker(self, mode, file_queue, user_msg, is_deep_cook):
        """Sequential batch logic for multi-video processing."""
        try:
            # P-Cores for control logic, but decoding will shift to E-Cores
            set_high_performance_affinity() 
            total_videos = len(file_queue)
            overall_success = False
            for v_idx, video_path in enumerate(file_queue):
                if self.stop_process.is_set(): break
                
                filename = os.path.basename(video_path)
                self.process_queue.put({"status": "thinking_status", "content": f"Processing Video {v_idx+1}/{total_videos}"})
                self.process_queue.put({"status": "log_update", "content": f"\n[BATCH] Processing: {filename}\n"})
                
                # Identify or Generate Segments
                results = []
                segment_results = []
                # Model Validation Check
                if not self.model:
                    self.process_queue.put({"status": "log_update", "content": f"\n[FATAL] Model not initialized. Aborting video {filename}.\n"})
                    continue

                if getattr(self.model, "chat_handler", None) is None:
                    if not self._ensure_chat_handler():
                        self.process_queue.put({"status": "error", "content": f"Vision Projector (.mmproj) missing for {filename}. Cannot process images without a vision projector loaded!"})
                        continue

                if mode == "video":
                    # OVERHAULED VIDEO PROCESSING: Single-pass uniform frame sampling (1 fps, max 60)
                    self.process_queue.put({"status": "thinking_status", "content": "Extracting frames at 1 fps..."})
                    frames = VisionHandler.get_video_sampled_frames(video_path, target_fps=1.0)
                    if not frames:
                        self.process_queue.put({"status": "log_update", "content": f"\n[ERROR] Failed to sample frames from {filename}\n"})
                        continue

                    # Gemma 4 Best Practice: images first, then text instruction
                    prompt_content = []
                    for f in frames:
                        prompt_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}})

                    # Guide prompt to prevent disclaimers
                    video_prompt = user_msg if user_msg else "Describe the sequence of events and actions in this video."
                    video_prompt = f"[Media Grounding - File: {filename} ({video_path})]\nYou are looking at a sequence of frames sampled from a video at 1 frame per second. {video_prompt}"
                    prompt_content.append({"type": "text", "text": video_prompt})

                    self.process_queue.put({"status": "thinking_status", "content": f"Processing {filename} (single-pass, {len(frames)} frames)..."})
                    
                    # Inference
                    VisionHandler.hygiene_gate(self.model)
                    HardwareProfile.pin_to_p_cores()
                    try:
                        sys_prompt = PERSONA_PROMPTS.get(self.active_persona_level, "You are Serenity.")
                        if self.model_path and "muse" in self.model_path.lower() and "glimmer" in self.model_path.lower():
                            r_str = self.config.get("muse_reasoning_strength", "xhigh")
                            if r_str != "off": sys_prompt += f"\nReasoning strength: {r_str}"
                        stream = self.model.create_chat_completion(
                            messages=[
                                {"role": "system", "content": sys_prompt},
                                {"role": "user", "content": prompt_content}
                            ],
                            **self._get_inference_params(),
                            stream=True
                        )
                        
                        final_analysis = ""
                        for chunk in stream:
                            if self.stop_process.is_set(): break
                            delta = chunk.get('choices', [{}])[0].get('delta', {})
                            content = delta.get('content', '')
                            if content: final_analysis += content

                        if final_analysis:
                            # Save strategic analysis output file
                            output_txt = video_path.replace(os.path.splitext(video_path)[1], f"_STRATEGIC_analysis.txt")
                            with open(output_txt, "w", encoding="utf-8") as f:
                                f.write(final_analysis)
                            results.append(output_txt)
                            segment_results.append(f"--- Analysis for {filename} ---\n{final_analysis}\n")
                            self.process_queue.put({"status": "log_update", "content": f"\n[THOUGHTS: {filename}]\n{final_analysis}\n"})
                            self.process_queue.put({"status": "thinking_status", "content": f"Analyzed: {filename}"})
                    except Exception as e:
                        print(f"[APEX] Video Error on {filename}: {e}\n{traceback.format_exc()}")
                        self.process_queue.put({"status": "log_update", "content": f"\n[ERROR] Failed to process {filename}: {e}\n"})
                    finally:
                        VisionHandler.hygiene_gate(self.model)
                        HardwareProfile.release_cores()
                elif mode == "multimodal":
                    # Sequential Processing for Images, Audio, and Videos
                    import traceback
                    prompt_content = []
                    
                    self.process_queue.put({"status": "thinking_status", "content": f"Extracting/Chunking {filename}..."})
                    
                    try:
                        HardwareProfile.pin_to_p_cores()
                        
                        target_tier = f"vision_{mode}"
                        ext = os.path.splitext(video_path)[1].lower()
                        
                        # Gemma-4 Auto-Vision & Budget Logic
                        final_user_msg, budget = VisionHandler.prepare_vision_query(user_msg if user_msg else "Analyze this media.")
                        
                        # Dynamic Context Check: 1120 budget requires at least 4k context
                        if budget >= 1120:
                            self._expand_context_config(target_tier, 8192)
                        
                        if ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']:
                            # Image - Use determined budget
                            b64 = VisionHandler.encode_image(video_path, budget=budget, query=user_msg)
                            if b64:
                                prompt_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                        elif ext in ['.mp3', '.wav', '.ogg', '.flac', '.m4a']:
                            # Audio -> chunks
                            chunks = VisionHandler.get_audio_chunks(video_path, chunk_length_s=30, max_chunks=30)
                            # Modern Gemma E-series MM-Projector audio injection
                            for i, b64 in enumerate(chunks):
                                prompt_content.append({
                                    "type": "input_audio",
                                    "input_audio": {
                                        "data": b64,
                                        "format": "wav"
                                    }
                                })
                        elif ext in ['.mp4', '.mkv', '.avi', '.mov']:
                            # Video -> sample frames at 1 fps (max 60) with budget & zoom detection
                            is_zoom = any(k in (user_msg or "").lower() for k in ["zoom", "crop", "detail", "card", "suit", "rank"])
                            frames = VisionHandler.get_video_sampled_frames(video_path, target_fps=1.0, budget=budget, zoom=is_zoom)
                            for f in frames:
                                prompt_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}})
                            
                            # Guide prompt for video frames sequence
                            if final_user_msg:
                                final_user_msg = f"You are looking at a sequence of frames sampled from a video at 1 frame per second. {final_user_msg}"
                                
                        final_user_msg = f"[Media Grounding - File: {filename} ({video_path})]\n{final_user_msg}"
                        prompt_content.append({"type": "text", "text": final_user_msg})
                        
                        self.process_queue.put({"status": "thinking_status", "content": f"Processing {filename} via Multimodal Projector..."})
                        
                        # Apply flash attention hygiene and clear KV for the new file 
                        VisionHandler.hygiene_gate(self.model)
                        
                        sys_prompt = PERSONA_PROMPTS.get(self.active_persona_level, "You are Serenity.")
                        if self.model_path and "muse" in self.model_path.lower() and "glimmer" in self.model_path.lower():
                            r_str = self.config.get("muse_reasoning_strength", "xhigh")
                            if r_str != "off": sys_prompt += f"\nReasoning strength: {r_str}"
                        stream = self.model.create_chat_completion(
                            messages=[
                                {"role": "system", "content": sys_prompt},
                                {"role": "user", "content": prompt_content}
                            ],
                            **self._get_inference_params(),
                            stream=True
                        )
                        
                        final_analysis = ""
                        for chunk in stream:
                            if self.stop_process.is_set(): break
                            delta = chunk.get('choices', [{}])[0].get('delta', {})
                            content = delta.get('content', '')
                            if content: final_analysis += content
                            
                        if final_analysis:
                            output_txt = video_path.replace(os.path.splitext(video_path)[1], f"_MULTIMODAL_analysis.txt")
                            with open(output_txt, "w", encoding="utf-8") as f:
                                f.write(final_analysis)
                            results.append(output_txt)
                            segment_results.append(f"--- Analysis for {filename} ---\n{final_analysis}\n")
                            # Intermediate analysis -> Thoughts (Logs)
                            self.process_queue.put({"status": "log_update", "content": f"\n--- [THOUGHTS: {filename}] ---\n{final_analysis}\n"})
                            self.process_queue.put({"status": "thinking_status", "content": f"Analyzed: {filename}"})
                        
                    except Exception as e:
                        print(f"[APEX] Multimodal Error on {filename}: {e}\n{traceback.format_exc()}")
                        self.process_queue.put({"status": "log_update", "content": f"\n[ERROR] Failed to process {filename}: {e}\n"})
                    finally:
                        VisionHandler.hygiene_gate(self.model)
                        HardwareProfile.release_cores()
                
                # Reset progress bar for this video
                self.process_queue.put({"status": "video_progress", "content": 0})

                # --- COMPANION SWAP: The Final Unload ---
                # Before synthesis or batch end, kill the Vision instance to free KV Cache space
                print("[APEX] Companion Swap: Unloading Vision Projector for KV Cache expansion...")
                del self.model
                self.model = None
                import gc
                gc.collect()
                if TORCH_AVAILABLE:
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except: pass
                time.sleep(1.0) # Breathe

                # GRANDMASTER SYNTHESIS (Requires model reload without projector handler)
                if not self.stop_process.is_set() and results:
                    overall_success = True
                    should_synthesize = is_deep_cook or self.config.get("synthesis_in_tactical_mode", False)
                    
                    if not should_synthesize:
                        # Fallback: If synthesis is skipped, we still need a "Final Output" in the chat.
                        # We use the text of the last analysis if only 1 file, or a list of files analyzed.
                        final_msg = ""
                        if len(segment_results) == 1:
                            final_msg = segment_results[0].split("---", 2)[-1].strip()
                        else:
                            file_list = "\n".join([f"- {os.path.basename(r)}" for r in results])
                            final_msg = f"Processing complete. Full findings are in the Backend Logs (Thoughts) and the respective analysis files:\n\n{file_list}"
                        
                        self.process_queue.put({"status": "success", "content": final_msg})
                        self.process_queue.put({"status": "thinking_status", "content": "[COMPLETE] Analysis finished. Synthesis skipped."})
                    else:
                        self.process_queue.put({"status": "thinking_status", "content": "[PROCESS] Reloading for Final Delivery..."})
                        
                        # Trigger a reload to a standard text tier (or same tier but fresh context)
                        # We use model_swap to reload the core weights WITHOUT the vision projector handler
                        # This fulfills the "Companion Swap" mission.
                        self.pending_task = {"type": "synthesis_finalize", "results": results, "filename": filename, "persona": PERSONA_PROMPTS.get(self.active_persona_level, "")}
                        self.model_swap(target_tier=self.current_model_tier)
                        break # Exit worker; handle success will trigger synthesis
            if not overall_success:
                self.process_queue.put({"status": "error", "content": "No media files were successfully analyzed."})

            if self.state.get("deep_cook_behavior") == "oneshot":
                self.process_queue.put({"status": "vision_oneshot_finish"})

        except Exception as e:
            self.process_queue.put({"status": "error", "content": f"Batch Processing Failed: {str(e)}"})

    def reinitialize_model(self, new_layer_count=None):
        """EXPLICIT VRAM evacuation and model reload helper."""
        print(f"[APEX] Evacuating VRAM for re-initialization...")
        
        # 1. Kill the reference
        if self.model:
            del self.model
            self.model = None 
        
        # 2. Force Garbage Collection
        import gc
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 3. Trigger reload
        print(f"[APEX] Reloading model with new parameters...")
        self.model_swap(target_tier=self.current_model_tier)

    def _handle_load_success(self, msg):
        self.state["running"] = False
        
        # --- APEX VRAM EVACUATION ---
        if self.model and self.model != msg["model"]:
            print("[APEX] Evacuating old model VRAM...")
            del self.model
            self.model = None
            import gc
            gc.collect()
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.model = msg["model"]
        self.active_persona_level = msg["level"]
        self.loaded_persona_level = msg["level"]
        if "tier" in msg: self.current_model_tier = msg["tier"]
        self.set_ui_state(model_loaded=True, generating=False)
        has_pending = bool(self.pending_task)
        self.load_history(render_active=not has_pending)
        if getattr(self, 'turbo_vec', None):
            lookup_mode = self.config.get("history_lookup_mode", "targeted")
            threading.Thread(
                target=self.turbo_vec.ingest_needed_files, 
                args=(self.model_path, self.active_persona_level, lookup_mode),
                daemon=True
            ).start()
        self.update_persona_display()
        self.set_avatar_state("pleased")
        self._log_and_display(f"Loaded {os.path.basename(self.model_path)}")
        if not self.pending_task:
            self.root.after(1500, lambda *args: self.set_avatar_state("listening"))

        if self.pending_task:
            t = self.pending_task; self.pending_task = None
            
            # Render history so the UI isn't empty if we were pending
            if t["type"] in ["chat", "deep_cook", "vision_deep", "vision_standard"]:
                self._render_messages_to_active_chat(self.messages)
                if "message" in t:
                    self.user_input.delete("1.0", tk.END)
                    self.last_user_message = t["message"]
                    self._display_user_message(t["message"])
            
            if t["type"] == "deep_cook": self.send_deep_cook_message(t["message"], True)
            elif t["type"] == "chat": self.send_message(t["message"], True)
            elif t["type"] == "vision_deep": self._execute_vision_deep_cook(t["staged"], t["message"])
            elif t["type"] == "vision_standard": 
                self.initiate_vision_analysis(t["staged"]["type"], t["staged"]["path"], t["message"])
            elif t["type"] == "synthesis_finalize":
                 self.process_queue.put({"status": "thinking_status", "content": "[PROCESS] Finalizing Resolution..."})
                 # MISSION: Execute heavy synthesis in background thread to prevent UI freeze
                 threading.Thread(target=self._synthesis_worker, args=(t,), daemon=True).start()

    def _handle_load_error(self, msg):
        self.state["running"] = False; self.model = None
        self.pending_task = None # Clear pending task to prevent ghost triggers
        self.set_ui_state(model_loaded=False)
        self.set_avatar_state("apologetic")
        messagebox.showerror("Load Error", msg.get("content"))

    def _synthesis_worker(self, task):
        """Background worker for master summary generation."""
        try:
            # Use our GIL-safe helper if possible, or a safer streaming call
            # Since generate_master_summary expects the model object, we'll let it run 
            # but we've already ensured it's in a thread.
            print(f"[SYSTEM] Master synthesis worker started for {len(task['results'])} files.")
            summary = generate_master_summary(task["results"], self.model, task["persona"])
            
            # Push results back to UI thread
            self.process_queue.put({
                "status": "session_finished",
                "user_msg": f"Master Overview: {task.get('filename', 'Session')}",
                "think_log": "",
                "final_answer": f"### 🧠 Master Multimodal Overview\n{self._clean_latex_artifacts(summary)}",
                "is_error": False
            })
            self.process_queue.put({"status": "log_update", "content": f"\n--- MASTER OVERVIEW: {task.get('filename')} ---\n{summary}\n"})
        except Exception as e:
            self.process_queue.put({"status": "error", "content": f"Final Synthesis failed: {e}"})

    def _get_persona_label(self):
        return "Cecilia" if self.active_persona_level == 7 else "Serenity"

    def _sanitize_synthesis_output(self, raw_text):
        if not raw_text: return ""
        
        # Split raw response at the end of the thinking tag and discard the thought prefix
        closers = [
            r'<\/think>', r'<\/thought>', r'<\/\|think\|>', r'<\|im_end\|>', r'<channel\|>', r'<\/channel\|>',
            r'<\|channel>text', r'<\|channel>assistant', r'\[\/DRAFT\]',
            r'<\|eom\|>', r'<\|start\|>assistant\s+to=user(?:<\|message\|>)?', r'to=user<\|message\|>'
        ]
        best_split = -1
        for pattern in closers:
            match = re.search(pattern, raw_text, re.IGNORECASE)
            if match:
                best_split = max(best_split, match.end())
        
        if best_split != -1:
            cleaned = raw_text[best_split:]
        else:
            cleaned = raw_text
            
        tags = [
            "<think>", "</think>", "<thought>", "</thought>", "<|think|>", "</|think|>", "<|channel>thought", "<channel|>", "Final Response:", "Final Answer:",
            "<|channel>text", "<|channel>assistant", "</channel|>", "###", "<|im_start|>", "<|im_end|>", "<|endoftext|>",
            "<|start|>assistant to=user<|message|>", "<|start|>assistant to=self<|message|>", "<|start|>assistant to=user", "<|start|>assistant to=self",
            "to=self<|message|>", "to=user<|message|>", "<|eom|>", "<|eot|>", "<|end_of_text|>"
        ]
        for tag in tags:
            cleaned = re.sub(re.escape(tag), '', cleaned, flags=re.IGNORECASE)
        
        return cleaned.strip()

    def _looks_like_thought_only(self, text):
        if not text or not text.strip():
            return True
        clean = text.strip()
        if len(clean) < 10:
            return True
        if re.match(r'(?i)^(thought|analysis|plan|draft|review|reasoning)', clean):
            return True
        if re.search(r'(?i)\b(thought|analysis|plan|draft|review|reasoning|goal|task|variable|step)\b', clean) and len(clean.split()) < 20:
            return True
        return False

    def _perform_final_synthesis(self, user_msg, reasoning_history, skip_critique=True, critique_txt="", prompt_override=None):
        """Shared logic to distill thoughts into a final response."""
        try:
            try: self.set_avatar_state("ecstatic")
            except: pass
            self.process_queue.put({"status": "thinking_status", "content": "Finalizing Response..."})
            is_gemma = "gemma" in self.model_path.lower()
            params = self._get_inference_params()
            critique_part = "" if skip_critique else f"Critique Findings: {critique_txt}\n"
            history_subset = reasoning_history[-10000:] if len(reasoning_history) > 10000 else reasoning_history
            synth_params = dict(params)
            synth_params["max_tokens"] = 4096

            instr_text = prompt_override if prompt_override else "Convert the reasoning above into a direct final response. Speak directly to the user now. Output ONLY the final response."
            final_prompt = (
                f"User Query: {user_msg}\n\n"
                f"Reasoning to convert:\n{history_subset}\n\n"
                f"[ORGANIZED BACKEND THOUGHTS]: {self.state.get('dmn_backbone', {}).get('last_simmer', 'N/A')}\n\n"
                f"{critique_part}{instr_text}"
            )

            if is_gemma:
                self.process_queue.put({"status": "thinking_status", "content": "Refining Response..."})
                persona_instr = PERSONA_PROMPTS.get(self.active_persona_level, "You are Serenity.")
                if self.model_path and "muse" in self.model_path.lower() and "glimmer" in self.model_path.lower():
                    r_str = self.config.get("muse_reasoning_strength", "xhigh")
                    if r_str != "off": persona_instr += f"\nReasoning strength: {r_str}"
                synth_sys = (
                    f"{persona_instr}\n\n"
                    "[TASK]: Convert the provided reasoning history into a direct final response. "
                    "Speak directly to the user now. Do not use internal monologue tags or your thought process. "
                    "Output ONLY the final response text."
                )
                prompt_str = (
                    f"<|turn>system\n{synth_sys}<turn|>\n"
                    f"<|turn>user\n{final_prompt}<turn|>\n"
                    f"<|turn>model\n"
                )
                raw_response = self._run_blocking_inference(prompt_str, synth_params)
            else:
                self.process_queue.put({"status": "thinking_status", "content": "Refining Response..."})
                msgs = [
                    {"role": "system", "content": "You are Serenity. Deliver the final response based on the reasoning below."},
                    {"role": "user", "content": final_prompt}
                ]
                raw_response = self._run_blocking_inference(msgs, params)

            full_compiled_text = self._sanitize_synthesis_output(raw_response)
            if self._looks_like_thought_only(full_compiled_text):
                self.process_queue.put({"status": "log_update", "content": "[SYNTHESIS WARNING] Initial extraction appears thought-like or too short; using direct-strike fallback.\n"})
                strike_prompt = (
                    f"User Query: {user_msg}\n\n"
                    f"Internal Reasoning Context:\n<thoughts>\n{history_subset[-4000:]}\n</thoughts>\n\n"
                    "Based on the reasoning above, provide a helpful, direct final response to the user. "
                    "Output ONLY the final response text. No tags, no thinking, no explanations of your process."
                )
                if is_gemma:
                    strike_sys = (
                        f"{PERSONA_PROMPTS.get(self.active_persona_level, 'You are Serenity.')}\n"
                        "[DIRECT STRIKE MODE]: Output the final response only."
                    )
                    prompt_strike = (
                        f"<|turn>system\n{strike_sys}<turn|>\n"
                        f"<|turn>user\n{strike_prompt}<turn|>\n"
                        f"<|turn>model\n"
                    )
                    strike_raw = self._run_blocking_inference(prompt_strike, synth_params)
                else:
                    msgs_strike = [
                        {"role": "system", "content": "You are Serenity."},
                        {"role": "user", "content": strike_prompt}
                    ]
                    strike_raw = self._run_blocking_inference(msgs_strike, params)
                strike_text = self._sanitize_synthesis_output(strike_raw)
                if len(strike_text) > len(full_compiled_text):
                    full_compiled_text = strike_text
                self.process_queue.put({"status": "log_update", "content": "[SYNTHESIS] Direct-strike fallback complete.\n"})

            if full_compiled_text and len(full_compiled_text.strip()) > 5:
                self.process_queue.put({"status": "log_update", "content": "[SYNTHESIS] Final response ready.\n"})
                return full_compiled_text

            return None
        except Exception as e:
            self.process_queue.put({"status": "log_update", "content": f"[SYNTHESIS ERROR] {e}\n"})
            print(f"Synthesis Error: {e}")
            return None

    def _perform_level7_synthesis(self, user_msg, reasoning_history, critique_txt=""):
        """Persona-specific distillation for Cecilia (Level 7)."""
        try:
            from serenity_resources import LEVEL7_SYNTHESIS_SYSTEM_PROMPT
            self.process_queue.put({"status": "thinking_status", "content": "Cecilia is delivering her truth..."})
            is_gemma = "gemma" in self.model_path.lower()
            params = self._get_inference_params()
            history_subset = reasoning_history[-10000:] if len(reasoning_history) > 10000 else reasoning_history

            dmn_context = ""
            backbone = self.state.get("dmn_backbone", {}).get("last_simmer")
            if backbone:
                dmn_context = f"[ORGANIZED BACKEND THOUGHTS]: {backbone}\n\n"

            critique_part = f"\n[CRITIQUE FINDINGS]: {critique_txt}\n" if critique_txt and critique_txt != "[SKIPPED - EARLY RESOLUTION]" else ""
            
            final_prompt = (
                f"User Query: {user_msg}\n\n"
                f"{dmn_context}"
                f"Hidden Variables & Reasoning Found:\n{history_subset}\n\n"
                f"{critique_part}"
                "Now, fulfill your role as Cecilia. Deliver the final response to the user. "
                "Output ONLY the final response text without internal thought tags or planning commentary."
            )

            if is_gemma:
                prompt_str = (
                    f"<|turn>system\n{LEVEL6_SYNTHESIS_SYSTEM_PROMPT}<turn|>\n"
                    f"<|turn>user\n{final_prompt}<turn|>\n"
                    f"<|turn>model\n"
                )
                synth_params = dict(params)
                synth_params["max_tokens"] = 4096
                raw_response = self._run_blocking_inference(prompt_str, synth_params)
            else:
                msgs = [
                    {"role": "system", "content": LEVEL6_SYNTHESIS_SYSTEM_PROMPT},
                    {"role": "user", "content": final_prompt}
                ]
                raw_response = self._run_blocking_inference(msgs, params)

            full_compiled_text = self._sanitize_synthesis_output(raw_response)
            if self._looks_like_thought_only(full_compiled_text):
                self.process_queue.put({"status": "log_update", "content": "[LEVEL6 WARNING] Cecilia output appears thought-like; retrying with more direct instruction.\n"})
                retry_prompt = (
                    f"User Query: {user_msg}\n\n"
                    f"Hidden Variables & Reasoning Found:\n{history_subset}\n\n"
                    "You are Cecilia. Provide the final answer directly. Do not use any thought tags or analysis. "
                    "Only speak the final response."
                )
                if is_gemma:
                    retry_str = (
                        f"<|turn>system\n{LEVEL6_SYNTHESIS_SYSTEM_PROMPT}<turn|>\n"
                        f"<|turn>user\n{retry_prompt}<turn|>\n"
                        f"<|turn>model\n"
                    )
                    retry_raw = self._run_blocking_inference(retry_str, synth_params)
                else:
                    retry_msgs = [
                        {"role": "system", "content": LEVEL6_SYNTHESIS_SYSTEM_PROMPT},
                        {"role": "user", "content": retry_prompt}
                    ]
                    retry_raw = self._run_blocking_inference(retry_msgs, params)
                full_compiled_text = self._sanitize_synthesis_output(retry_raw)
                self.process_queue.put({"status": "log_update", "content": "[LEVEL6] Retry complete.\n"})

            self.process_queue.put({"status": "log_update", "content": "[LEVEL6] Cecilia synthesis complete.\n"})
            return full_compiled_text if full_compiled_text else None
        except Exception as e:
            self.process_queue.put({"status": "log_update", "content": f"[LEVEL6 ERROR] {e}\n"})
            print(f"Cecilia Synthesis Error: {e}")
            return f"Cecilia encountered a truth she couldn't yet speak: {str(e)}"

    _perform_level6_synthesis = _perform_level7_synthesis

    def _finalize_message(self, user_msg, think_log, final_answer, error=False):
        print(f"[SYSTEM] Finalizing message delivery (Error: {error}).")
        self.state["running"] = False
        self.set_ui_state(model_loaded=True, generating=False)
        if self.thinking_display and self.thinking_display.winfo_exists():
            self.thinking_display.stop()
        
        if self.chat_history is None: return

        hist = self.chat_history
        start_idx = self.state.get("response_start_idx")
        
        # 4. Atomic Reset for Rendering / In-Place Markdown Application
        # Preserve already-streamed responses to prevent mass-dump flickering
        if self.state.get("response_started") and start_idx:
            if error:
                self._append_to_chat(f"\n\n[System Error]: {final_answer}\n\n", "system")
                self.set_avatar_state("apologetic")
                try:
                    with open(self.error_log_file, 'a', encoding='utf-8') as f:
                        f.write(f"\n[{time.strftime('%H:%M:%S')}] [System Error]: {final_answer}\n")
                except: pass
                return

            if self.state.get("thought_streamed") and not self.state.get("deep_cook"):
                hist.config(state='normal')
                curr_text = hist.get(start_idx, tk.END).strip()
                if final_answer and not (curr_text.endswith(final_answer.strip()[-40:]) if len(final_answer) >= 40 else final_answer.strip() in curr_text):
                    self._append_to_chat(final_answer, "ai")
                hist.config(state='disabled')
            elif think_log and not self.state.get("deep_cook"):
                # Atomic reset to prepend the expandable thought process block
                hist.config(state='normal')
                try:
                    hist.delete(start_idx, tk.END)
                except: pass
                hist.insert(tk.END, f"\n\n{self._get_persona_label()}: ", "ai_lead")
                hist.config(state='disabled')
            elif not self.state.get("deep_cook"):
                hist.config(state='normal')
                curr_text = hist.get(start_idx, tk.END).strip()
                lead = f"{self._get_persona_label()}:"
                # If the content already streamed cleanly, preserve it to prevent mass dump
                if lead in curr_text and (curr_text.endswith(final_answer.strip()[-40:]) if len(final_answer) >= 40 else final_answer.strip() in curr_text):
                    pass # Stream already complete in-place
                else:
                    try:
                        hist.delete(start_idx, tk.END)
                    except: pass
                    hist.insert(tk.END, f"\n\n{self._get_persona_label()}: ", "ai_lead")
                    if final_answer:
                        hist.insert(tk.END, final_answer, ("ai",))
                hist.config(state='disabled')
            else:
                # Deep Cook: Ensure synthesis text starts clean but don't wipe reasoning
                hist.config(state='normal')
                hist.insert(tk.END, "\n", ("ai",))
                hist.config(state='disabled')

        # 5. UI Rendering (Atomic & Pre-Processed)
        hist.config(state='normal')
        render_mode = self.config.get("media_rendering", 1)
        render_start = hist.index(tk.END + "-1c")

        # Suppress redundant generic thinking block if Deep Cook structure is already present
        if self.state.get("thought_streamed") and not self.state.get("deep_cook"):
            think_tag = self.state.get("current_think_tag")
            btn = self.state.get("current_think_btn")
            render_start = hist.index(tk.END + "-1c")
            if think_tag:
                hist.tag_config(think_tag, elide=True)
                if btn and btn.winfo_exists():
                    btn.config(text="[+] View Thinking Process")
                ranges = hist.tag_ranges(think_tag)
                if ranges and len(ranges) >= 2:
                    if render_mode > 0:
                        self._apply_markdown(ranges[0], ranges[-1], (think_tag, "md_thought"), is_thought=True)
                    render_start = ranges[-1]
                else:
                    render_start = start_idx if start_idx else "1.0"
            else:
                render_start = start_idx if start_idx else "1.0"
        elif think_log and not self.state.get("deep_cook"):
            think_tag = f"think_block_{int(time.time() * 1000)}"
            def toggle_thoughts(tag=think_tag, b=None):
                is_elided = str(hist.tag_cget(tag, "elide")) in ["1", "True", "true"]
                if is_elided:
                     hist.tag_config(tag, elide=False)
                     if b: b.config(text="[-] Hide Thinking")
                else:
                     hist.tag_config(tag, elide=True)
                     if b: b.config(text="[+] View Thinking Process")
            
            btn_bg = THEME.get("button_bg_color", "#24201c")
            accent = THEME.get("electric_blue", "#00bfff")
            btn_active = THEME.get("button_active_color", "#382e24")
            accent_hl = THEME.get("accent_highlight", "#ff8800")
            
            btn = tk.Button(hist, text="[+] View Thinking Process", bg=btn_bg, fg=accent, 
                            activebackground=btn_active, activeforeground=accent_hl, 
                            relief=tk.FLAT, font=self.fonts["stats"], cursor="hand2", padx=6, pady=2)
            btn.config(command=lambda t=think_tag, b=btn: toggle_thoughts(t, b))
            if not hasattr(self, 'thought_dropdown_buttons'):
                self.thought_dropdown_buttons = []
            self.thought_dropdown_buttons.append(btn)
            
            hist.insert(tk.END, "\n", ("ai",))
            hist.window_create(tk.END, window=btn)
            hist.insert(tk.END, "\n", ("ai",))
            
            thought_start = hist.index(tk.END + "-1c")
            hist.insert(tk.END, think_log + "\n\n", (think_tag, "md_thought"))
            thought_end = hist.index(tk.END + "-1c")
            
            if render_mode > 0:
                # MISSION: Use the lightweight 'thought' renderer to prevent UI lock
                self._apply_markdown(thought_start, thought_end, (think_tag, "md_thought"), is_thought=True)
            
            hist.tag_config(think_tag, elide=True, lmargin1=20, lmargin2=20)

            # Insert Final Answer after thoughts
            render_start = hist.index(tk.END + "-1c")
            if final_answer:
                self._append_to_chat(final_answer, "ai")
        elif not self.state.get("response_started", False):
            # In unstarted streams, output message directly
            self._display_ai_message(final_answer, is_streaming=False)
            render_start = start_idx if start_idx else "1.0"
        else:
            render_start = start_idx if start_idx else "1.0"
        
        render_end = hist.index(tk.END + "-1c")
        if render_mode > 0:
            self._apply_markdown(render_start, render_end, ("ai",))
            self._post_process_media(start_idx=render_start)
        
        # Embed RLHF Feedback Widget
        if not error and final_answer:
            try:
                rlhf_frame = tk.Frame(hist, bg=THEME["bg_color"])
                lbl_fb = tk.Label(rlhf_frame, text="Feedback: ", bg=THEME["bg_color"], fg="#666666", font=self.fonts["stats"])
                lbl_fb.pack(side=tk.LEFT)
                
                def _submit_fb(rating):
                    btn_up.config(state="disabled", fg="#555555" if rating < 0 else "#00ff88")
                    btn_down.config(state="disabled", fg="#ff4444" if rating < 0 else "#555555")
                    self._save_rlhf_log(user_msg, final_answer, rating)

                btn_up = tk.Button(rlhf_frame, text="👍", bg=THEME["bg_color"], fg="#00ff88", activebackground=THEME["widget_bg_color"],
                                   relief=tk.FLAT, font=self.fonts["log"], command=lambda: _submit_fb(1))
                btn_down = tk.Button(rlhf_frame, text="👎", bg=THEME["bg_color"], fg="#ff4444", activebackground=THEME["widget_bg_color"],
                                     relief=tk.FLAT, font=self.fonts["log"], command=lambda: _submit_fb(-1))
                btn_up.pack(side=tk.LEFT, padx=2)
                btn_down.pack(side=tk.LEFT, padx=2)
                
                hist.insert(tk.END, "\n", ("ai",))
                hist.window_create(tk.END, window=rlhf_frame)
                hist.insert(tk.END, "\n", ("ai",))
            except Exception as fb_err:
                print(f"[UI] RLHF widget embed error: {fb_err}")

        hist.config(state='disabled')
        hist.see(tk.END)

        # PERSISTENCE (Hardened against memory corruption)
        try:
            if not error and user_msg and final_answer:
                final_answer_history = final_answer.replace("<|file_separator|>", "").strip()
                self.messages.extend([
                    {"role": "user", "content": str(user_msg)}, 
                    {"role": "assistant", "content": str(final_answer_history)}
                ])
                if self.config.get("ghost_mode", False):
                    self.messages = self.messages[-4:]
                self.save_history()
        except Exception as e:
            print(f"[SYSTEM] Persistence recovery: {e}")

        try:
            self.set_avatar_state("pleased")
        except: pass
        
        self.state["response_started"] = False
        self.state["current_think_tag"] = None
        self.state["current_think_btn"] = None
        self.state["thought_streamed"] = False
        self.state["thought_separator_inserted"] = False
        self.state["thought_stream_lead_cleaned"] = False
        self.thought_stream_buffer = ""
        self.root.after(1500, lambda *args: self.set_avatar_state("listening"))

    def _save_rlhf_log(self, prompt, answer, rating):
        """Saves user feedback (+1 / -1) into System/rlhf_logs.json and updates DMN backbone."""
        try:
            p = os.path.join(self.dirs["System"], "rlhf_logs.json")
            logs = []
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    logs = json.load(f)
            entry = {
                "timestamp": datetime.now().isoformat(),
                "persona": getattr(self, "active_persona_level", 1),
                "model": self.config.get("model_path", "unknown"),
                "rating": rating,
                "prompt": prompt,
                "answer": answer[:500] if answer else ""
            }
            logs.append(entry)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(logs, f, indent=2)
            
            # Incorporate with DMN (Level 7 & background memory)
            if "dmn_backbone" in self.state and isinstance(self.state["dmn_backbone"], dict):
                self.state["dmn_backbone"]["last_rlhf_rating"] = rating
                self.state["dmn_backbone"]["total_rlhf_count"] = len(logs)
                self._save_dmn_backbone()
            print(f"[RLHF] Successfully recorded feedback ({rating}) to {p}")
        except Exception as e:
            print(f"[RLHF ERROR] Failed to save feedback: {e}")

    def _run_self_analysis(self):
        """Gathers system configuration and triggers a self-diagnosis report output to the chat window."""
        try:
            vram_mb = getattr(self, "virtual_vram", 0) or self.state.get("virtual_vram", 0)
            model_p = self.config.get("model_path", "None")
            p_level = getattr(self, "active_persona_level", 1)
            rec_mode = self.config.get("budget_recovery_mode", "wrapup")
            markdown_on = self.config.get("inline_markdown", True)
            k_cache = self.config.get("k_cache_type", "q8_0")
            v_cache = self.config.get("v_cache_type", "q4_0")
            
            diag_report = (
                "### 🩺 **Serenity Configuration Self-Analysis Report**\n\n"
                f"- **Active Model**: `{os.path.basename(model_p)}`\n"
                f"- **Persona Level**: `Level {p_level}`\n"
                f"- **VRAM Target**: `{vram_mb / 1024:.2f} GB`\n"
                f"- **KV Cache Config**: `K: {k_cache} | V: {v_cache}`\n"
                f"- **Thought Budget Recovery**: `{rec_mode.upper()}`\n"
                f"- **Inline Markdown Engine**: `{'Enabled' if markdown_on else 'Disabled'}`\n"
                f"- **DMN Reflection Backbone**: `{self.state.get('dmn_backbone', {}).get('node_count', 0)} nodes active`\n"
                f"- **System Status**: `Operational - All Core Pipelines Nominal`\n"
            )
            
            if self.chat_history:
                self.chat_history.config(state='normal')
                self.chat_history.insert(tk.END, f"\n\n🤖 **Serenity Self-Analysis**: \n", "ai_lead")
                self.chat_history.insert(tk.END, diag_report, "ai")
                self.chat_history.config(state='disabled')
                self.chat_history.see(tk.END)
                if self.config.get("media_rendering", 1) > 0:
                    start_idx = self.chat_history.index(tk.END + f"-{len(diag_report)+20}c")
                    end_idx = self.chat_history.index(tk.END + "-1c")
                    self._apply_markdown(start_idx, end_idx, ("ai",))
            print("[SELF-ANALYSIS] Diagnosis report generated successfully.")
        except Exception as err:
            print(f"[SELF-ANALYSIS ERROR] Failed to run self analysis: {err}")

    def _buffer_text(self, text):
        """Append text to the streaming buffer."""
        if not self.state.get("response_started"):
            self.state["response_started"] = True
            self.set_avatar_state("explain_wise")
        self.text_buffer += text

    def _update_stats_display(self, stats):
        if not hasattr(self, 'stats_labels') or not self.stats_labels: return
        graph_mode = self.config.get("monitor_graph_mode", False)
        try:
            for k, v in stats.items():
                if k in self.stats_labels:
                    val_str = str(v)
                    if graph_mode and "%" in val_str:
                        try:
                            pct = float(val_str.replace("%", "").strip())
                            bars = int(pct / 10)
                            val_str = f"[{'█' * bars}{'░' * (10 - bars)}] {pct:.0f}%"
                        except Exception: pass
                    elif k in ("Power", "CPU Power") and isinstance(v, (int, float)):
                        val_str = f"{v:.1f}W"
                    self.stats_labels[k].config(text=val_str)
        except: pass

    def apply_text_scale(self, scale_pct=100, persist=True):
        """
        Dynamically adjusts font sizes across all UI components and markdown tags.
        Supports continuous scaling (70% - 250%).
        Incorporates window-responsive factor so fonts grow with window size.
        """
        try:
            scale_pct = max(70, min(250, int(scale_pct)))
        except (ValueError, TypeError):
            scale_pct = 100
        
        user_factor = scale_pct / 100.0
        
        responsive_enabled = True
        offsets = {}
        if hasattr(self, 'config') and self.config:
            responsive_enabled = self.config.get("responsive_font_scaling", True)
            offsets = self.config.get("font_size_offsets", {})
        
        win_factor = getattr(self, '_window_scale_factor', 1.0) if responsive_enabled else 1.0
        
        for k, spec in BASE_FONT_SPECS.items():
            if k in self.fonts:
                base_sz = spec.get("size", 10)
                cat_offset = 0
                if k in ("main", "small", "italic", "large", "bold", "md_bold", "md_italic", "md_bold_italic", "md_quote", "md_strike"):
                    cat_offset = offsets.get("chat", 0)
                elif k in ("md_header", "md_header_1", "md_header_2", "md_header_3"):
                    cat_offset = offsets.get("headers", 0)
                elif k in ("log", "log_bold", "md_code", "md_table", "md_thought", "md_math_inline", "md_math_block"):
                    cat_offset = offsets.get("code_log", 0)
                elif k in ("stats", "stats_bold"):
                    cat_offset = offsets.get("stats", 0)
                elif k in ("ui_button", "ui_label", "ui_small"):
                    cat_offset = offsets.get("ui", 0)
                
                adj_base = max(4, base_sz + cat_offset)
                # Per-font responsiveness: UI fonts scale fully, code/log fonts barely change
                resp = _FONT_WINDOW_RESPONSIVENESS.get(k, 1.0)
                effective_win = 1.0 + resp * (win_factor - 1.0)
                new_sz = max(6, int(round(adj_base * user_factor * effective_win)))
                self.fonts[k].configure(size=new_sz)
        
        if persist and hasattr(self, 'config') and self.config is not None:
            self.config["text_scale"] = scale_pct
            if hasattr(self, 'model_paths'):
                self.save_config()

    def apply_font_family(self, ui_family=None, mono_family=None, persist=True):
        """
        Dynamically adjusts font families across all UI components and markdown tags.
        """
        if ui_family is None and hasattr(self, 'config') and self.config:
            ui_family = self.config.get("ui_font", "Segoe UI")
        if mono_family is None and hasattr(self, 'config') and self.config:
            mono_family = self.config.get("mono_font", "Consolas")
            
        ui_family = ui_family or "Segoe UI"
        mono_family = mono_family or "Consolas"
        
        mono_keys = {
            "log", "log_bold", "stats", "stats_bold",
            "md_thought", "md_math_inline", "md_math_block", "md_table", "md_code"
        }
        
        for k in self.fonts:
            fam = mono_family if k in mono_keys else ui_family
            try:
                self.fonts[k].configure(family=fam)
            except Exception:
                pass
                
        if persist and hasattr(self, 'config') and self.config is not None:
            self.config["ui_font"] = ui_family
            self.config["mono_font"] = mono_family
            if hasattr(self, 'model_paths'):
                self.save_config()

    def zoom_in(self, event=None):
        curr_scale = self.config.get("text_scale", 100) if hasattr(self, 'config') and self.config else 100
        new_scale = min(250, curr_scale + 10)
        self.apply_text_scale(new_scale, persist=True)
        return "break"

    def zoom_out(self, event=None):
        curr_scale = self.config.get("text_scale", 100) if hasattr(self, 'config') and self.config else 100
        new_scale = max(70, curr_scale - 10)
        self.apply_text_scale(new_scale, persist=True)
        return "break"

    def zoom_reset(self, event=None):
        self.apply_text_scale(100, persist=True)
        return "break"

    def _on_root_configure(self, event=None):
        """Debounced handler: recompute font scale on window resize."""
        if event and event.widget != self.root:
            return
        if hasattr(self, '_root_resize_job') and self._root_resize_job:
            self.root.after_cancel(self._root_resize_job)
        self._root_resize_job = self.root.after(150, self._apply_responsive_scale)

    def _apply_responsive_scale(self):
        """Recompute window-responsive scale factor and re-apply text scale."""
        try:
            w = self.root.winfo_width()
            if w < 100:
                return  # Window not yet mapped
            base_w = 960  # Minimum window width
            ratio = max(1.0, w / base_w)
            # Soft linear growth: at 1920px -> factor ~1.4, at 2560px -> factor ~1.67
            self._window_scale_factor = 1.0 + 0.4 * (ratio - 1.0)
        except Exception:
            self._window_scale_factor = 1.0
        curr_scale = self.config.get("text_scale", 100) if hasattr(self, 'config') and self.config else 100
        self.apply_text_scale(curr_scale, persist=False)

    def update_persona_display(self, val=None):
        if self.depth_slider is None: return
        
        lvl = int(val) if val is not None else self.active_persona_level
        self.active_persona_level = lvl
        
        # --- Level 7 Dynamic Hiding: auto-collapse slider to 6 when moving to standard levels 1-6 ---
        if lvl <= 6 and self.depth_slider.cget('to') == 7 and not getattr(self, '_setting_slider', False):
            self.depth_slider.config(to=6)

        # --- Secret Lore Button Toggle & Auto-Hide ---
        if hasattr(self, 'lore_btn') and self.lore_btn is not None:
            if lvl == 7:
                self.lore_btn.pack(side=tk.LEFT, padx=15)
            else:
                self.lore_btn.pack_forget()

        name, desc = PERSONA_DISPLAY_INFO.get(lvl, ("Unknown", "Invalid"))
        if self.state.get("deep_cook"):
             sys_p = PERSONA_DISPLAY_INFO.get(lvl, ("", ""))[1] # Fallback
             try:
                 from serenity_resources import DEEP_COOK_SYSTEM_PROMPTS
                 sys_p = DEEP_COOK_SYSTEM_PROMPTS.get(lvl, "")
                 import re
                 match = re.search(r'in (.*?) mode', sys_p, re.IGNORECASE)
                 if match: name = f"LVL {lvl}: {match.group(1).upper()} MODE"
                 else: name = f"LVL {lvl}: DEEP COOK"
             except: name = f"LVL {lvl}: DEEP COOK"

        if self.config.get("theme") == "persona":
            from serenity_resources import apply_theme_to_global
            apply_theme_to_global("persona", self.config.get("texture_style", "default"), self.config.get("dark_mode", False), lvl, (self.model is not None))
            self.apply_current_theme()
        else:
            if self.chat_history is not None:
                self.chat_history.config(bg=THEME["chat_bg_color"], fg=THEME["chat_fg_color"])
            if self.user_input is not None:
                self.user_input.config(bg=THEME["widget_bg_color"], fg=THEME["fg_color"])
            if self.depth_slider is not None:
                lvl_color = THERMO_COLORS.get(lvl, THEME["electric_blue"])
                self.depth_slider.config(bg=THEME["bg_color"], troughcolor=THEME["widget_bg_color"], activebackground=lvl_color)
        
        if hasattr(self, 'persona_name_button') and self.persona_name_button is not None:
            if (self.live_agent_process and self.live_agent_process.poll() is None):
                btn_cmd = lambda l=lvl: self._live_persona_swap(l)
            else:
                btn_cmd = lambda l=lvl: self.model_swap(target_level=l)
            self.persona_name_button.config(text=name, 
                                            fg=THERMO_COLORS.get(lvl, THEME["electric_blue"]), 
                                            command=btn_cmd)
        
        if hasattr(self, 'persona_desc_label') and self.persona_desc_label is not None: 
            self.persona_desc_label.config(text=desc, fg=THEME["electric_blue"], bg=THEME["bg_color"])

    def _on_persona_label_click(self, e):
        self.state["persona_clicks"] += 1
        if self.state["persona_clicks"] >= 6: 
            self.state["persona_clicks"] = 0
            self._load_secret_model_event()

    def _load_secret_model_event(self, e=None):
        self._log_and_display("Engaging Worldbuilder...")
        self.active_persona_level = 7
        self._setting_slider = True
        self.depth_slider.config(to=7)
        self.depth_slider.set(7)
        self._setting_slider = False
        self.update_persona_display(7)
        self.model_swap(target_level=7)

    def redirect_logs(self):
        sys.stdout = WidgetLogger(self.thought_log, "stdout")
        sys.stderr = FileAndWidgetLogger(self.error_log, self.error_log_file, "stderr")



    def initialize_app(self):
        self._log_and_display("System Ready. Select a persona to begin.")
        self.set_avatar_state("off")
        
        # Restore persistent sticky persona level and range (max 6 unless level 7 active)
        if hasattr(self, 'depth_slider'):
            max_lvl = 7 if self.active_persona_level == 7 else 6
            self.depth_slider.config(to=max_lvl)
            self.depth_slider.set(self.active_persona_level)
        
        self.update_persona_display(self.active_persona_level)

    def offload_model(self):
        """Explicitly offloads all model resources and multimodal handlers."""
        if self.model: self.save_history()
        
        # Clear Model and Multimodal Resources
        self.model = None
        self.messages = []
        self.current_model_tier = None
        
        # Explicitly clear Vision Chat Handler
        if hasattr(self, 'chat_handler'):
            self.chat_handler = None
            
        # Trigger Hygiene Gate for VRAM Flush
        import gc
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try: torch.cuda.empty_cache()
            except: pass
        VisionHandler.hygiene_gate()
        
        # --- Level 7 Auto-Hide on Offload ---
        if self.depth_slider.cget('to') == 7:
            self.depth_slider.config(to=6)
            if self.active_persona_level == 7:
                self.depth_slider.set(6)
                self.active_persona_level = 6
        
        self.clear_chat_ui()
        self.set_ui_state(model_loaded=False)
        self.update_persona_display(self.active_persona_level) 
        self._reset_multimodal_ui()
        self.set_avatar_state("off") 
        self._log_and_display("All models offloaded. VRAM Cleared.")

    def get_history_path(self): 
        if not self.model_path: return None
        hist_dir = self.get_user_history_dir()
        base = f"{os.path.splitext(os.path.basename(self.model_path))[0]}_lvl{self.active_persona_level}.history"
        p_enc = os.path.join(hist_dir, f"{base}.encz")
        p_jsonz = os.path.join(hist_dir, f"{base}.jsonz")
        if os.path.exists(p_enc): return p_enc
        if os.path.exists(p_jsonz): return p_jsonz
        ext = ".jsonz" if self.get_active_username() in ("Default", "Public") else (".encz" if (hasattr(self, 'vault_manager') and self.vault_manager.is_lock_enabled()) else ".jsonz")
        return os.path.join(hist_dir, f"{base}{ext}")

    def save_history(self):
        if self.config.get("ghost_mode", False):
            return
        if not (path := self.get_history_path()) or not self.messages: return
        try:
            if hasattr(self, 'vault_manager'):
                self.vault_manager.write_history_messages(path, self.messages)
            else:
                with open(path, 'wb') as f:
                    f.write(zlib.compress(json.dumps(self.messages).encode('utf-8')))
        except Exception as e:
            print(f"History save error: {e}", file=sys.stderr)

    def _render_messages_to_active_chat(self, msg_list):
        """Renders prompt and AI responses inline in the active chat view."""
        hist = self.chat_history
        if hist is None: return
        hist.config(state='normal')
        hist.delete('1.0', tk.END)
        for m in msg_list:
            role = m.get('role', '')
            content = m.get('content', '')
            if isinstance(content, list):
                text_parts = [p.get("text", "") if isinstance(p, dict) and p.get("type") == "text" else str(p) for p in content]
                content_str = " ".join(text_parts)
            else:
                content_str = str(content)
            
            if role == 'user':
                s_idx = hist.index(tk.END + "-1c")
                hist.insert(tk.END, f"\nYou: {content_str}\n", ("user",))
                e_idx = hist.index(tk.END + "-1c")
                if self.config.get("media_rendering", 1) > 0:
                    self._apply_markdown(s_idx, e_idx, ("user",))
            elif role == 'assistant':
                s_idx = hist.index(tk.END + "-1c")
                hist.insert(tk.END, f"\n\n{self._get_persona_label()}: ", ("ai_lead",))
                hist.insert(tk.END, f"{content_str}\n", ("ai",))
                e_idx = hist.index(tk.END + "-1c")
                if self.config.get("media_rendering", 1) > 0:
                    self._apply_markdown(s_idx, e_idx, ("ai",))
        hist.config(state='disabled')
        hist.see(tk.END)

    def load_history(self, render_active=True):
        # Default profile starts fresh on every app load
        if not getattr(self, '_initial_history_loaded', False) and self.get_active_username() == "Default":
            self._initial_history_loaded = True
            self.messages = []
            self.clear_chat_ui()
            return

        self._initial_history_loaded = True

        is_ghost = self.config.get("ghost_mode", False)
        if is_ghost:
            # Ghost mode: retain 2 replies (4 messages) in memory for context
            self.messages = self.messages[-4:] if hasattr(self, 'messages') and self.messages else []
            if render_active:
                self._render_messages_to_active_chat(self.messages)
            return

        usage = self.config.get("history_usage", "all")
        if usage == "off":
            self.messages = []
            self.clear_chat_ui()
            return
        if usage == "current_window":
            if self.messages:
                if render_active:
                    self._render_messages_to_active_chat(self.messages)
                return
            self.clear_chat_ui()
            return

        self.messages = []
        self.clear_chat_ui()
        
        self.past_history_view.config(state='normal')
        self.past_history_view.delete('1.0', tk.END)
        self.past_history_view.config(state='disabled')
        
        if (path := self.get_history_path()) and os.path.exists(path):
            try:
                if hasattr(self, 'vault_manager'):
                    self.messages = self.vault_manager.read_history_messages(path)
                else:
                    with open(path, 'rb') as f: 
                        self.messages = json.loads(zlib.decompress(f.read()).decode('utf-8'))
                
                if self.messages and self.messages[-1].get('role') == 'user': 
                    self.messages.pop()
                
                self.past_history_view.config(state='normal')
                for m in self.messages: 
                    who = "You" if m['role'] == 'user' else self._get_persona_label()
                    tag = "user" if m['role'] == 'user' else "ai"
                    entry = f"{who}: {m['content']}\n{'-'*50}\n\n"
                    self.past_history_view.insert(tk.END, entry, (tag,))
                
                self.past_history_view.config(state='disabled')
                self.past_history_view.yview_moveto(1.0)
                # Option 1: Keep past history in Archive tab, start active chat fresh
                self.clear_chat_ui()
                self._log_and_display("Archive Updated.")
            except PermissionError:
                self._log_and_display("Archive is encrypted. Unlock vault to access.")
            except: 
                self._log_and_display("Archive load failed.")

    def on_closing(self):
        self.save_config()
        self.stop_process.set()
        if self.model: self.save_history()
        if self.live_agent_process: self.live_agent_process.terminate()
        if SYSTEM_MONITOR_LOADED: 
            try: nvidia_ml.nvmlShutdown()
            except: pass
        self.root.destroy()
        
    def load_params(self, tier):
        """Loads tier-specific inference parameters, auto-populating params.json if missing."""
        params_files = [f"params_{tier}.json", "params.json"]
        loaded = False
        
        for p_file in params_files:
            path = os.path.join(self.dirs["System"], p_file)
            if os.path.exists(path):
                try: 
                    with open(path, "r", encoding="utf-8") as f: 
                        data = json.load(f)
                    if isinstance(data, dict) and data:
                        self.params = data
                        print(f"[APEX] Loaded inference overrides from: {p_file}")
                        loaded = True
                        break
                except Exception as e:
                    print(f"Warning: Failed to load {p_file}: {e}")
        
        if not loaded:
            stops = [s.strip() for s in self.stop_strings_config.get(tier, "").split(",") if s.strip()]
            for st in ["<turn|>", "<|turn>", "<|end_of_turn|>", "<eos>", "<|im_end|>", "<|eot_id|>", "You:", "Serenity:"]:
                if st not in stops:
                    stops.append(st)
            
            default_params = {
                "stop": stops,
                "temperature": self.temp_config.get(tier, 0.8),
                "min_p": self.min_p_config.get(tier, 0.05),
                "repeat_penalty": self.repeat_penalty_config.get(tier, 1.0),
                "presence_penalty": self.presence_penalty_config.get(tier, 0.0),
                "frequency_penalty": self.frequency_penalty_config.get(tier, 0.0),
                "top_k": self.top_k_config.get(tier, 64),
                "top_p": self.top_p_config.get(tier, 0.95),
                "dry_multiplier": 0.0,
                "disable_fim_autoswap": True
            }
            default_path = os.path.join(self.dirs["System"], "params.json")
            try:
                with open(default_path, "w", encoding="utf-8") as f:
                    json.dump(default_params, f, indent=2)
                self.params = default_params
                print(f"[APEX] Populated default params.json for tier: {tier}")
            except Exception as e:
                self.params = default_params
                print(f"[APEX] Warning: Could not write params.json: {e}")

    def _fit_image_aspect(self, img, target_w=350, target_h=350):
        orig_w, orig_h = img.size
        ratio = min(target_w / float(orig_w), target_h / float(orig_h))
        new_w = max(1, int(orig_w * ratio))
        new_h = max(1, int(orig_h * ratio))
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def load_all_images(self):
        w, h = 350, 350 
        try:
            if self.right_panel.winfo_width() > 1: w, h = self.right_panel.winfo_width(), self.right_panel.winfo_height() // 2
        except: pass

        if not os.path.isdir(self.dirs['Media']):
             messagebox.showwarning("Missing Assets", f"Media folder not found at:\n{self.dirs['Media']}\nAvatar will not display.")
             return

        for state, fname in AVATAR_FILENAMES.items():
            try:
                p = os.path.join(self.dirs["Media"], fname)
                if os.path.exists(p):
                    img = self._fit_image_aspect(Image.open(p), w, h)
                    self.avatar_states[state] = ImageTk.PhotoImage(img)
                    #print(f"Loaded: {fname}")
            except Exception as e: print(f"Error loading {fname}: {e}")
            
        if not self.avatar_states:
             messagebox.showinfo("Assets Missing", "No avatar images found in Media folder.\nUsing text fallback.")
             
        self.set_avatar_state("off")

    def set_avatar_state(self, state):
        if not self.right_panel: return
        self.state["avatar_current"] = state 
        
        # --- MISSION: Force Cecilia for Level 7 ---
        if self.active_persona_level == 7:
            fname = "Cecilia_01.png"
            p = os.path.join(self.dirs["Media"], fname)
            if os.path.exists(p):
                try:
                    # Clean up idle timers to prevent state flickering
                    if getattr(self, 'idle_timer_id', None) is not None: 
                        self.root.after_cancel(self.idle_timer_id)
                        self.idle_timer_id = None
                    
                    img = self._fit_image_aspect(Image.open(p), 350, 350)
                    self.tmp_img = ImageTk.PhotoImage(img)
                    self.right_panel.itemconfig(self.avatar_image_item, state='normal', image=self.tmp_img)
                    self.right_panel.itemconfig(self.avatar_text_item, state='hidden')
                    return # Mission Accomplished: Cecilia is absolute.
                except Exception as e:
                    print(f"[UI] Cecilia load error: {e}")

        if getattr(self, 'idle_timer_id', None) is not None: 
            self.root.after_cancel(self.idle_timer_id)
            self.idle_timer_id = None
        
        mapping = {
            "off": "serenity_off.png",
            "thinking": "serenity_thinking.png",
            "pondering": "serenity_pondering.png",
            "pleased": "serenity_pleased.png",
            "listening": "serenity_greeting.png",
            "confused": "serenity_confused.png",
            "deep_think": "The_Wise_Listener.png",
            "subdued": "subdued_serenity.png",
            "ecstatic": "serenity_ecstatic.png",
            "idea": "serenity_idea.png",
            "apologetic": "sorry_serenity.png",
            "explain_direct": "explain_direct.png",
            "explain_wise": "explain_wise.png",
            "dmn_lvl1": "lvl1_galaxy.jpg",
            "dmn_lvl2": "lvl2_galaxy.jpg",
            "dmn_lvl3": "lvl3_galaxy.jpg",
            "dmn_lvl4": "lvl4_galaxy.jpg",
            "dmn_lvl5": "lvl5_galaxy.jpg",
            "dmn_lvl6": "lvl6_galaxy.jpg",
            "meditating": "Meditating_Serenity.png",
            "transcendent": "transcendent_serenity.png",
            "idle_lvl7": "transcendent_serenity.png",
            "cecilia_alt": "Cecilia_01.png"
        }
        
        fname = mapping.get(state)
        if not fname:
            fname = f"lvl{self.active_persona_level}_serenity_idle.png"
        
        if state in self.avatar_states: 
             self.right_panel.itemconfig(self.avatar_image_item, state='normal', image=self.avatar_states[state])
             self.right_panel.itemconfig(self.avatar_text_item, state='hidden')
        else:
            p = os.path.join(self.dirs["Media"], fname)
            if os.path.exists(p):
                try:
                    img = self._fit_image_aspect(Image.open(p), 350, 350)
                    self.tmp_img = ImageTk.PhotoImage(img)
                    self.right_panel.itemconfig(self.avatar_image_item, state='normal', image=self.tmp_img)
                    self.right_panel.itemconfig(self.avatar_text_item, state='hidden')
                except Exception as e:
                    print(f"Avatar load error: {e}")

        # Track DMN State entry/exit
        if str(state).startswith("dmn"):
            self.state["dmn_active"] = True
            if not self.state.get("dmn_entry_time"):
                self.state["dmn_entry_time"] = time.time()
        else:
            self.state["dmn_active"] = False
            self.state["dmn_entry_time"] = None

        if state == "listening": 
            if getattr(self, "idle_timer_id", None) is not None:
                self.root.after_cancel(self.idle_timer_id)
                self.idle_timer_id = None
            self.idle_timer_id = self.root.after(3000, lambda *args: self._set_persona_idle_state())
        elif state in PERSONA_IDLE_MAP.values() or state in ["idle_lvl1", "idle_lvl2", "idle_lvl3", "idle_lvl4", "idle_lvl5", "idle_lvl6", "idle_lvl7", "transcendent"]:
            if getattr(self, "idle_timer_id", None) is not None:
                self.root.after_cancel(self.idle_timer_id)
                self.idle_timer_id = None
            timeout_ms = self._parse_dmn_timeout_sec() * 1000
            self.idle_timer_id = self.root.after(timeout_ms, lambda *args: self.set_avatar_state(f"dmn_lvl{self.active_persona_level}"))

    def _parse_dmn_timeout_sec(self):
        val = self.config.get("dmn_timeout", "05:00")
        if isinstance(val, (int, float)):
            return max(1, int(val))
        try:
            parts = str(val).strip().split(":")
            if len(parts) == 2:
                return max(1, int(parts[0]) * 60 + int(parts[1]))
            return max(1, int(parts[0]))
        except Exception:
            return 300

    def _set_persona_idle_state(self):
        if self.state.get("avatar_current") == "listening":
            idle_state = PERSONA_IDLE_MAP.get(self.active_persona_level, "listening")
            if idle_state != "listening":
                self.set_avatar_state(idle_state)

    def _position_canvas_elements(self):
        try:
            canvas_r = self.right_panel
            if canvas_r is None or not canvas_r.winfo_exists(): return
            w, h = canvas_r.winfo_width(), canvas_r.winfo_height()
            if w > 1 and h > 1:
                log_win = self.log_window_item
                if log_win is not None:
                    canvas_r.coords(log_win, w/2, h/4)
                    canvas_r.itemconfigure(log_win, width=max(10, w - 20), height=max(10, h//2 - 20))
                
                av_img = self.avatar_image_item
                if av_img is not None:
                    canvas_r.coords(av_img, w/2, h*0.75)
                
                av_txt = self.avatar_text_item
                if av_txt is not None:
                    canvas_r.coords(av_txt, w/2, h*0.75)
                
                if av_img is not None: canvas_r.lift(av_img)
                if av_txt is not None: canvas_r.lift(av_txt)
        except Exception: pass

    def load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f: self.config = ThreadSafeDict(json.load(f))
        
        # Track startup count to initialize blank logs upon 4th startup
        startup_count = self.config.get("startup_count", 0) + 1
        self.config["startup_count"] = startup_count
        
        if startup_count >= 4:
            self.config["startup_count"] = 0
            logs_dir = self.dirs.get("Logs") if hasattr(self, "dirs") else os.path.join(self.script_dir, "Logs")
            err_log = os.path.join(logs_dir, "error_log.txt")
            flt_log = os.path.join(logs_dir, "fault_log.txt")
            for p in [err_log, flt_log]:
                try:
                    if os.path.exists(p):
                        with open(p, 'w', encoding='utf-8') as f:
                            f.truncate(0)
                except Exception as e:
                    print(f"Failed to clear log {p}: {e}")

        # Save only the updated startup count back to file
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    disk_cfg = json.load(f)
                disk_cfg["startup_count"] = self.config["startup_count"]
                with open(self.config_file, 'w') as f:
                    json.dump(disk_cfg, f, indent=4)
            except Exception as e:
                print(f"Failed to write startup count to file: {e}")

        for key in ['model_paths', 'gpu_layer_config', 'context_size_config', 'temp_config', 
                    'top_p_config', 'min_p_config', 'repeat_penalty_config', 'frequency_penalty_config',
                    'presence_penalty_config', 'stop_strings_config', 'n_batch_config', 'top_k_config']:
            disk_data = self.config.get(key, {})
            getattr(self, key).update(disk_data)
        
        self.sash_pos = self.config.get('sash_pos', -1)
        
        self.state["deep_cook_behavior"] = self.config.get('deep_thought_behavior', "oneshot")
        self.state["virtual_vram"] = self.config.get('virtual_vram', 0)
        self.active_persona_level = self.config.get('active_persona_level', 3)
        self.max_persona_level = self.config.get('max_persona_level', 7)
        if self.max_persona_level < 7:
            self.max_persona_level = 7
        
        # Persist streaming and headroom
        self.state["streaming_mode"] = self.config.get('streaming_mode', "Buffered")
        self.config["max_token_ratio"] = self.config.get('max_token_ratio', 4)
        # Initialize UI toggles if missing
        if "synthesis_in_tactical_mode" not in self.config:
            self.config["synthesis_in_tactical_mode"] = False
        if "show_rgb_button" not in self.config:
            self.config["show_rgb_button"] = True
        if "auto_vram_offload" not in self.config:
            self.config["auto_vram_offload"] = False
        if "speculative_drafting" not in self.config:
            self.config["speculative_drafting"] = False
        if "history_lookup_mode" not in self.config:
            self.config["history_lookup_mode"] = "targeted"
        if "k_cache_type" not in self.config:
            self.config["k_cache_type"] = "q8_0"
        if "v_cache_type" not in self.config:
            self.config["v_cache_type"] = "q4_0"
        if "hao_preset" not in self.config:
            self.config["hao_preset"] = "exps=CPU"
        if "swa_kv_cache" not in self.config:
            self.config["swa_kv_cache"] = "Auto"
        if "media_rendering" not in self.config:
            self.config["media_rendering"] = 1
        if "history_usage" not in self.config:
            self.config["history_usage"] = "all"
        if "ghost_mode" not in self.config:
            self.config["ghost_mode"] = False
        if "benchmark_enabled" not in self.config:
            self.config["benchmark_enabled"] = False
        if "inline_markdown" not in self.config:
            self.config["inline_markdown"] = True
        if "budget_recovery_mode" not in self.config:
            self.config["budget_recovery_mode"] = "wrapup"
        if "monitor_graph_mode" not in self.config:
            self.config["monitor_graph_mode"] = False

        if "offline_mode" not in self.config:
            self.config["offline_mode"] = False
        if "tutorial_completed" not in self.config:
            self.config["tutorial_completed"] = False
        if "repeat_detection_mode" not in self.config:
            self.config["repeat_detection_mode"] = "lazy"
        if "theme" not in self.config:
            self.config["theme"] = "default"
        if "texture_style" not in self.config:
            self.config["texture_style"] = "default"
        if "frosted_glass" not in self.config:
            self.config["frosted_glass"] = False

        # Apply active theme and network guard
        try:
            apply_theme_to_global(
                self.config.get("theme", "default"),
                self.config.get("texture_style", "default"),
                self.config.get("frosted_glass", False)
            )
        except Exception as te:
            print(f"[THEME] Failed to apply theme from config: {te}")

        try:
            set_offline_mode(self.config.get("offline_mode", False))
        except Exception as ne:
            print(f"[OFFLINE] Failed to apply offline guard from config: {ne}")

        if "custom_templates" not in self.config or not self.config["custom_templates"]:
            self.config["custom_templates"] = {
                "T1": {"name": "Thinking (Gen)", "temp": 1.0, "top_p": 0.95, "min_p": 0.0, "rep": 1.0, "pres": 1.5, "top_k": 20, "batch": 512, "layers": -1, 
                    "ctx": 32768, "stop": "###,<|endoftext|>,<|im_end|>,<|turn>,<turn|>,<channel|>,</think>,<eos>"},
                "T2": {"name": "Thinking (Code)", "temp": 0.6, "top_p": 0.95, "min_p": 0.0, "rep": 1.0, "pres": 0.0, "top_k": 20, "batch": 512, "layers": -1, 
                    "ctx": 32768, "stop": "###,<|endoftext|>,<|im_end|>,<|turn>,<turn|>,<channel|>,</think>,<eos>"},
                "T3": {"name": "Vision (Best)", "temp": 0.1, "top_p": 0.9, "min_p": 0.0, "rep": 1.1, "pres": 0.0, "top_k": 64, "batch": 512, "layers": -1, 
                    "ctx": 8192, "stop": "###,<|endoftext|>,<|im_end|>,<|turn>,<turn|>,<channel|>,</think>,<eos>"}
            }
        
        # Sub-chunk size for Vision
        self.sub_chunk_size = self.config.get("sub_chunk_size", 8)
        VisionHandler.SUB_CHUNK_SIZE = self.sub_chunk_size
        
        # Ensure Layer Config is Numeric (Primary Source of Truth)
        # Defer applying auto‑detected layers until configuration is saved.
        if any(v == -1 for v in self.gpu_layer_config.values()):
            # Store recommendations for later use without mutating the config now.
            self._auto_detected_layers = self.run_auto_detect()
            # Keep -1 placeholders in gpu_layer_config; they will be replaced on save.
        
        if hasattr(self, "_load_dmn_backbone"):
            self._load_dmn_backbone()

        # Merge active user profile config if exists
        user_dir = self.get_user_dir() if hasattr(self, "get_user_dir") else None
        if user_dir:
            u_cfg_p = os.path.join(user_dir, "config.json")
            if os.path.exists(u_cfg_p):
                try:
                    with open(u_cfg_p, 'r', encoding='utf-8') as f:
                        u_data = json.load(f)
                    self.config.update(u_data)
                except Exception as e:
                    print(f"[USER] Could not merge user config: {e}")

        if "theme" not in self.config:
            self.config["theme"] = "apex"
        if "texture_style" not in self.config:
            self.config["texture_style"] = "default"
        if "dark_mode" not in self.config:
            self.config["dark_mode"] = False

        # Apply active theme and network guard
        try:
            apply_theme_to_global(
                self.config.get("theme", "apex"),
                self.config.get("texture_style", "default"),
                self.config.get("dark_mode", False),
                getattr(self, "active_persona_level", 3),
                (self.model is not None)
            )
        except Exception as te:
            print(f"[THEME] Failed to apply theme from config: {te}")
        
        return self.config

    def apply_current_theme(self):
        """Applies active THEME and TEXTURE_STYLES across all widgets and tags dynamically."""
        from serenity_resources import THEME, TEXTURE_STYLES, THERMO_COLORS
        
        bg = THEME.get("bg_color", "#121214")
        fg = THEME.get("fg_color", "#ffaa44")
        widget_bg = THEME.get("widget_bg_color", "#18181c")
        btn_bg = THEME.get("button_bg_color", "#24201c")
        btn_active = THEME.get("button_active_color", "#382e24")
        trim = THEME.get("trim_color", "#4a3520")
        accent = THEME.get("electric_blue", "#ff7700")
        accent_hl = THEME.get("accent_highlight", "#ff8800")
        accent_sec = THEME.get("accent_secondary", "#ffaa00")
        chat_bg = THEME.get("chat_bg_color", widget_bg)
        chat_fg = THEME.get("chat_fg_color", fg)
        dark_mode = THEME.get("_dark_mode", False)
        
        try:
            self.root.config(bg=bg)
            if hasattr(self, 'paned') and self.paned and self.paned.winfo_exists():
                self.paned.config(bg=bg)
                for p in self.paned.panes():
                    try: self.root.nametowidget(p).config(bg=bg)
                    except: pass
                
            # Frames and Containers
            for frame_attr in ('top_bar_frame', 'tab_bar_frame', 'persona_control_frame', 'desc_container', 
                               'footer_control_frame', 'status_frame', 'history_menu_frame',
                               'stats_frame', 'log_container', 'log_frame'):
                if hasattr(self, frame_attr):
                    f = getattr(self, frame_attr)
                    if f and f.winfo_exists():
                        f.config(bg=bg if "container" in frame_attr or "control" in frame_attr or "frame" in frame_attr else widget_bg)

            # Prompt Box with Themed Borders
            if hasattr(self, 'input_control_frame') and self.input_control_frame and self.input_control_frame.winfo_exists():
                self.input_control_frame.config(bg=trim, highlightbackground=trim, highlightcolor=accent)
            if hasattr(self, 'attachment_frame') and self.attachment_frame and self.attachment_frame.winfo_exists():
                self.attachment_frame.config(bg=trim)
            if hasattr(self, 'user_input') and self.user_input and self.user_input.winfo_exists():
                self.user_input.config(bg=widget_bg, fg=fg, insertbackground=fg, 
                                       highlightthickness=1, highlightbackground=trim, highlightcolor=accent)

            # Loading Bar & ThinkingDisplay Theming
            if hasattr(self, 'status_frame') and self.status_frame and self.status_frame.winfo_exists():
                self.status_frame.config(bg=bg)
            if hasattr(self, 'thinking_display') and self.thinking_display and self.thinking_display.winfo_exists():
                self.thinking_display.config(bg=bg)
                if hasattr(self.thinking_display, 'header_frame') and self.thinking_display.header_frame and self.thinking_display.header_frame.winfo_exists():
                    self.thinking_display.header_frame.config(bg=bg)
                if hasattr(self.thinking_display, 'label') and self.thinking_display.label and self.thinking_display.label.winfo_exists():
                    self.thinking_display.label.config(bg=bg, fg=accent)
                if hasattr(self.thinking_display, 'telemetry_label') and self.thinking_display.telemetry_label and self.thinking_display.telemetry_label.winfo_exists():
                    self.thinking_display.telemetry_label.config(bg=bg, fg=accent_sec)
                if hasattr(self.thinking_display, 'progress_container') and self.thinking_display.progress_container and self.thinking_display.progress_container.winfo_exists():
                    self.thinking_display.progress_container.config(bg=bg)
                if hasattr(self.thinking_display, 'gauge_label') and self.thinking_display.gauge_label and self.thinking_display.gauge_label.winfo_exists():
                    self.thinking_display.gauge_label.config(bg=bg, fg=accent_hl)
                if hasattr(self.thinking_display, 'tasks_frame') and self.thinking_display.tasks_frame and self.thinking_display.tasks_frame.winfo_exists():
                    self.thinking_display.tasks_frame.config(bg=bg)
                if hasattr(self.thinking_display, 'task_lines_label') and self.thinking_display.task_lines_label and self.thinking_display.task_lines_label.winfo_exists():
                    self.thinking_display.task_lines_label.config(bg=bg, fg=fg)
                if hasattr(self.thinking_display, 'prayer_label') and self.thinking_display.prayer_label and self.thinking_display.prayer_label.winfo_exists():
                    self.thinking_display.prayer_label.config(bg=bg, fg=accent)

            # Chat History & Tags (Pure neon styling, no generic white text)
            if hasattr(self, 'chat_history') and self.chat_history and self.chat_history.winfo_exists():
                self.chat_history.config(bg=chat_bg, fg=chat_fg)
                self.chat_history.tag_config("user_lead", foreground=accent_sec)
                self.chat_history.tag_config("user", foreground=accent)
                self.chat_history.tag_config("ai_lead", foreground=accent_sec)
                self.chat_history.tag_config("md_header", foreground=accent_hl)
                self.chat_history.tag_config("md_header_1", foreground=accent_hl)
                self.chat_history.tag_config("md_header_2", foreground=accent_hl)
                self.chat_history.tag_config("md_header_3", foreground=accent)
                self.chat_history.tag_config("md_bold", foreground=accent)
                self.chat_history.tag_config("md_italic", foreground=accent_sec)
                self.chat_history.tag_config("md_bold_italic", foreground=accent_hl)
                self.chat_history.tag_config("md_thought", foreground="#888888" if not dark_mode else "#555555")
                self.chat_history.tag_config("md_code", foreground=accent_hl, background="#050505" if dark_mode else "#111111")
                self.chat_history.tag_config("md_list", foreground=chat_fg)
                self.chat_history.tag_config("md_math_inline", foreground=accent_sec)
                self.chat_history.tag_config("md_math_block", foreground=accent_sec)
                self.chat_history.tag_config("md_table", foreground=accent_hl)
                self.chat_history.tag_config("md_quote", foreground=accent)
                self.chat_history.tag_config("md_strike", foreground="#7f848e")
                self.chat_history.tag_config("stats", foreground=accent_hl)
                
            if hasattr(self, 'past_history_view') and self.past_history_view and self.past_history_view.winfo_exists():
                self.past_history_view.config(bg=chat_bg, fg=chat_fg)
                self.past_history_view.tag_config("user_lead", foreground=accent_sec)
                self.past_history_view.tag_config("user", foreground=accent)
                self.past_history_view.tag_config("ai_lead", foreground=accent_sec)
                self.past_history_view.tag_config("md_header", foreground=accent_hl)
                self.past_history_view.tag_config("md_header_1", foreground=accent_hl)
                self.past_history_view.tag_config("md_header_2", foreground=accent_hl)
                self.past_history_view.tag_config("md_header_3", foreground=accent)
                self.past_history_view.tag_config("md_bold", foreground=accent)
                self.past_history_view.tag_config("md_italic", foreground=accent_sec)
                self.past_history_view.tag_config("md_thought", foreground="#888888" if not dark_mode else "#555555")
                self.past_history_view.tag_config("md_code", foreground=accent_hl, background="#050505" if dark_mode else "#111111")
                
            if hasattr(self, 'user_input') and self.user_input and self.user_input.winfo_exists():
                self.user_input.config(bg=widget_bg, fg=fg, insertbackground=fg)
                
            if hasattr(self, 'prompt_display') and self.prompt_display and self.prompt_display.winfo_exists():
                self.prompt_display.config(bg=trim, fg=accent)
                
            if hasattr(self, 'persona_desc_label') and self.persona_desc_label and self.persona_desc_label.winfo_exists():
                self.persona_desc_label.config(bg=bg, fg=accent)

            if hasattr(self, 'desc_container') and self.desc_container and self.desc_container.winfo_exists():
                self.desc_container.config(bg=bg)
                
            # Top Action Buttons
            for btn_attr in ('load_model_button', 'action_button', 'btn_video', 'btn_watch', 
                             'btn_clear_queue', 'lock_button', 'rgb_button', 'lore_btn', 'mic_button'):
                if hasattr(self, btn_attr):
                    b = getattr(self, btn_attr)
                    if b and b.winfo_exists():
                        b.config(bg=btn_bg, fg=fg)

            # Thought Dropdown Buttons Dynamic Theming
            if hasattr(self, 'thought_dropdown_buttons') and self.thought_dropdown_buttons:
                for tb in list(self.thought_dropdown_buttons):
                    try:
                        if tb and tb.winfo_exists():
                            tb.config(bg=btn_bg, fg=accent, activebackground=btn_active, activeforeground=accent_hl)
                        else:
                            self.thought_dropdown_buttons.remove(tb)
                    except: pass

            if hasattr(self, 'btn_active') and self.btn_active and self.btn_active.winfo_exists():
                if getattr(self, 'active_tab', 'active') == 'active':
                    self.btn_active.config(bg=btn_active, fg=fg)
                    if hasattr(self, 'btn_history') and self.btn_history:
                        self.btn_history.config(bg=btn_bg, fg=accent_sec)
                else:
                    self.btn_active.config(bg=btn_bg, fg=accent_sec)
                    if hasattr(self, 'btn_history') and self.btn_history:
                        self.btn_history.config(bg=btn_active, fg=fg)
                        
            if hasattr(self, 'timeline_frame') and self.timeline_frame and self.timeline_frame.winfo_exists():
                self.timeline_frame.config(bg=widget_bg)
                if hasattr(self, 'progress_label') and self.progress_label:
                    self.progress_label.config(bg=widget_bg, fg=accent_hl)

            # System and Telemetry Labels
            if hasattr(self, 'system_status_label') and self.system_status_label and self.system_status_label.winfo_exists():
                self.system_status_label.config(bg=trim, fg=accent)
            if hasattr(self, 'hw_mode_label') and self.hw_mode_label and self.hw_mode_label.winfo_exists():
                self.hw_mode_label.config(bg=trim, fg=accent_hl)
            if hasattr(self, 'persona_label') and self.persona_label and self.persona_label.winfo_exists():
                self.persona_label.config(bg=bg, fg=accent)

            # Secret Persona / Cecilia Trigger (Blends seamlessly with background)
            if hasattr(self, 'secret_trigger') and self.secret_trigger and self.secret_trigger.winfo_exists():
                self.secret_trigger.config(bg=bg, fg=bg, bd=0, highlightthickness=0)

            # Backend Logs Container, Headers & Action Buttons
            if hasattr(self, 'log_container') and self.log_container and self.log_container.winfo_exists():
                self.log_container.config(bg=bg)
            if hasattr(self, 'log_header_frame') and self.log_header_frame and self.log_header_frame.winfo_exists():
                self.log_header_frame.config(bg=bg)
            if hasattr(self, 'log_header_label') and self.log_header_label and self.log_header_label.winfo_exists():
                self.log_header_label.config(bg=bg, fg=accent)
            if hasattr(self, 'self_analysis_btn') and self.self_analysis_btn and self.self_analysis_btn.winfo_exists():
                self.self_analysis_btn.config(bg=bg, fg=accent)
            if hasattr(self, 'lock_logout_btn') and self.lock_logout_btn and self.lock_logout_btn.winfo_exists():
                self.lock_logout_btn.config(bg=bg, fg=accent)
            if hasattr(self, 'clear_log_btn') and self.clear_log_btn and self.clear_log_btn.winfo_exists():
                self.clear_log_btn.config(bg=bg, fg=accent)
            if hasattr(self, 'log_frame') and self.log_frame and self.log_frame.winfo_exists():
                self.log_frame.config(bg=bg)

            # Backend Log Switch Canvas (Tabs & Knob)
            if hasattr(self, 'log_switch_canvas') and self.log_switch_canvas and self.log_switch_canvas.winfo_exists():
                self.log_switch_canvas.config(bg=bg)
                try:
                    self.log_switch_canvas.itemconfig(1, outline=accent, fill=widget_bg)
                    if hasattr(self, 'switch_knob') and self.switch_knob:
                        self.log_switch_canvas.itemconfig(self.switch_knob, fill=accent)
                    active_view = self.state.get("log_view", "thought")
                    icons = [(3, "thought"), (4, "tool"), (5, "error"), (6, "diag")]
                    for item_id, view_key in icons:
                        self.log_switch_canvas.itemconfig(item_id, fill=bg if active_view == view_key else accent)
                except: pass

            # ScrolledText Logs & Tags
            if hasattr(self, 'thought_log') and self.thought_log and self.thought_log.winfo_exists():
                self.thought_log.config(bg=widget_bg, fg=fg, insertbackground=fg)
                self.thought_log.tag_config("stdout", foreground=fg)
                self.thought_log.tag_config("system", foreground=accent)
            if hasattr(self, 'error_log') and self.error_log and self.error_log.winfo_exists():
                self.error_log.config(bg=widget_bg, fg="#ff8a8a" if not dark_mode else "#ff6666", insertbackground=fg)
                self.error_log.tag_config("stderr", foreground="#ff8a8a" if not dark_mode else "#ff6666")
            if hasattr(self, 'tool_log') and self.tool_log and self.tool_log.winfo_exists():
                self.tool_log.config(bg=widget_bg, fg=accent_sec, insertbackground=fg)
            if hasattr(self, 'diag_log') and self.diag_log and self.diag_log.winfo_exists():
                self.diag_log.config(bg=widget_bg, fg=accent_hl, insertbackground=fg)
                self.diag_log.tag_config("diag", foreground=accent_hl)

            # Telemetry UI & System Stats
            if hasattr(self, 'stats_frame') and self.stats_frame and self.stats_frame.winfo_exists():
                self.stats_frame.config(bg=widget_bg)
            if hasattr(self, 'stats_row_frames'):
                for f in self.stats_row_frames:
                    if f and f.winfo_exists():
                        f.config(bg=widget_bg)
            if hasattr(self, 'stats_title_labels'):
                for lbl in self.stats_title_labels:
                    if lbl and lbl.winfo_exists():
                        lbl.config(bg=widget_bg, fg=fg)
            if hasattr(self, 'stats_labels'):
                for lbl in self.stats_labels.values():
                    if lbl and lbl.winfo_exists():
                        lbl.config(bg=widget_bg, fg=accent)

            lvl = getattr(self, 'active_persona_level', 3)
            lvl_color = THERMO_COLORS.get(lvl, accent_hl)

            if hasattr(self, 'depth_slider') and self.depth_slider and self.depth_slider.winfo_exists():
                self.depth_slider.config(bg=bg, fg=fg, troughcolor=widget_bg, activebackground=lvl_color)

            if hasattr(self, 'persona_name_button') and self.persona_name_button and self.persona_name_button.winfo_exists():
                self.persona_name_button.config(fg=lvl_color)
                    
            if hasattr(self, 'style'):
                try:
                    self.style.configure("Apex.Horizontal.TProgressbar", troughcolor=widget_bg, background=accent_hl)
                except: pass
                
            if getattr(self, 'active_tab', 'active') == 'history':
                self._render_history_menu()
                
            if hasattr(self, '_update_hw_indicator'):
                self._update_hw_indicator()
        except Exception as e:
            print(f"[THEME] Theme re-application warning: {e}")

    def _load_dmn_backbone(self):
        user_dir = self.get_user_dir()
        p = os.path.join(user_dir, "dmn_backbone.json")
        fallback_p = os.path.join(self.dirs["System"], "dmn_backbone.json")
        try:
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as f:
                    self.state["dmn_backbone"] = json.load(f)
            elif os.path.exists(fallback_p):
                with open(fallback_p, 'r', encoding='utf-8') as f:
                    self.state["dmn_backbone"] = json.load(f)
                self._save_dmn_backbone()
            else:
                self.state["dmn_backbone"] = {}
        except:
            self.state["dmn_backbone"] = {}

    def _save_dmn_backbone(self):
        user_dir = self.get_user_dir()
        p = os.path.join(user_dir, "dmn_backbone.json")
        try:
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(self.state["dmn_backbone"], f, indent=4)
        except: pass

    def save_config(self):
        data = dict(self.config)
        data.update({
            'username': self.get_active_username(),
            'main_window': self.root.winfo_geometry(), 'model_paths': self.model_paths,
            'gpu_layer_config': self.gpu_layer_config, 'context_size_config': self.context_size_config,
            'temp_config': self.temp_config,
            'top_p_config': self.top_p_config, 'min_p_config': self.min_p_config,
            'repeat_penalty_config': self.repeat_penalty_config, 'frequency_penalty_config': self.frequency_penalty_config,
            'presence_penalty_config': self.presence_penalty_config,
            'stop_strings_config': self.stop_strings_config, 'n_batch_config': self.n_batch_config,
            'top_k_config': self.top_k_config,
            'sash_pos': self._get_current_sash_pos() if hasattr(self, '_get_current_sash_pos') else self.config.get('sash_pos', -1),
            'deep_thought_behavior': self.state["deep_cook_behavior"],
            'virtual_vram': self.state["virtual_vram"],
            'active_persona_level': self.active_persona_level,
            'max_persona_level': self.depth_slider.cget('to') if self.depth_slider else self.max_persona_level,
            'synthesis_in_tactical_mode': self.config.get("synthesis_in_tactical_mode", False),
            'show_rgb_button': self.config.get("show_rgb_button", True),
            'sub_chunk_size': getattr(self, 'sub_chunk_size', 8),
            'custom_templates': self.config.get('custom_templates', {}),
            'streaming_mode': self.state.get("streaming_mode", "Buffered"),
            'max_token_ratio': self.config.get("max_token_ratio", 4),
            'auto_vram_offload': self.config.get("auto_vram_offload", False),
            'speculative_drafting': self.config.get("speculative_drafting", True),
            'history_lookup_mode': self.config.get("history_lookup_mode", "targeted"),
            'k_cache_type': self.config.get("k_cache_type", "q8_0"),
            'v_cache_type': self.config.get("v_cache_type", "q4_0"),
            'hao_preset': self.config.get("hao_preset", "exps=CPU"),
            'swa_kv_cache': self.config.get("swa_kv_cache", "Auto"),
            'media_rendering': self.config.get("media_rendering", 1),
            'history_usage': self.config.get("history_usage", "all"),
            'ghost_mode': self.config.get("ghost_mode", False),
            'dynamic_params_enabled': self.config.get("dynamic_params_enabled", True),
            'startup_count': self.config.get("startup_count", 0),
            'offline_mode': self.config.get("offline_mode", False),
            'theme': self.config.get("theme", "apex"),
            'texture_style': self.config.get("texture_style", "default"),
            'dark_mode': self.config.get("dark_mode", False),
            'repeat_detection_mode': self.config.get("repeat_detection_mode", "lazy"),
            'status_bar_mode': self.config.get("status_bar_mode", "hybrid"),
            'status_bar_anim_style': self.config.get("status_bar_anim_style", "spinner"),
            'status_bar_dmn_idle': self.config.get("status_bar_dmn_idle", True),
            'status_bar_fallback_info': self.config.get("status_bar_fallback_info", True),
            'tutorial_completed': self.config.get("tutorial_completed", False),
            'text_scale': self.config.get("text_scale", 100),
            'ui_font': self.config.get("ui_font", "Segoe UI"),
            'mono_font': self.config.get("mono_font", "Consolas"),
            'font_size_offsets': self.config.get("font_size_offsets", {"chat": 0, "headers": 0, "code_log": 0, "stats": 0, "ui": 0})
        })
        with open(self.config_file, 'w') as f: json.dump(data, f, indent=4)
        
        # Also persist to user profile directory
        if hasattr(self, 'get_user_dir'):
            try:
                u_dir = self.get_user_dir()
                os.makedirs(u_dir, exist_ok=True)
                u_cfg = os.path.join(u_dir, "config.json")
                with open(u_cfg, 'w', encoding='utf-8') as uf:
                    json.dump(data, uf, indent=4)
            except Exception as ue:
                print(f"[USER] Failed to save user config: {ue}")

    def _get_inference_params(self, temp_messages=None):
        """Builds the parameter dictionary for llama-cpp-python inference."""
        print(f"[INFERENCE] Retrieving parameters for tier: {self.current_model_tier}")
        
        # Automated Architecture-Aware Stop Sequences
        stops = []
        raw_cfg_stops = self.stop_strings_config.get(self.current_model_tier, "")
        if raw_cfg_stops:
            stops = [s.strip() for s in raw_cfg_stops.split(",") if s.strip()]

        m_lower = (self.model_path or "").lower()
        if "muse" in m_lower and "glimmer" in m_lower:
            # Muse Glimmer / Onyx ATEM: Stop on <|end_of_text|> and <|eot|>; NEVER on <|eom|>
            for tok in ["<|end_of_text|>", "<|eot|>"]:
                if tok not in stops: stops.append(tok)
            stops = [s for s in stops if s != "<|eom|>"]
        elif "gemma" in m_lower:
            # Gemma 2 / 4: Official turn closers
            for tok in ["<turn|>", "<|turn>", "</turn>", "<eos>", "<|end_of_turn|>"]:
                if tok not in stops: stops.append(tok)
        elif any(k in m_lower for k in ["nemotron", "qwen", "deepseek"]):
            # ChatML & Nemotron turn delimiters
            for tok in ["<|im_end|>", "<|end_of_text|>", "<|endoftext|>", "<extra_id_1>"]:
                if tok not in stops: stops.append(tok)
        else:
            # Standard fallback delimiters
            for tok in ["<|eot_id|>", "<|end_of_text|>", "<|turn|>", "</s>", "<eos>"]:
                if tok not in stops: stops.append(tok)
        
        # Baseline defaults
        inf_params = {
            "temperature": self.temp_config.get(self.current_model_tier, 1.0), # Gemma-4 Official
            "top_p": self.top_p_config.get(self.current_model_tier, 0.95),
            "min_p": self.min_p_config.get(self.current_model_tier, 0.05),
            "repeat_penalty": self.repeat_penalty_config.get(self.current_model_tier, 1.15) if self.repeat_penalty_config.get(self.current_model_tier, 1.0) == 1.0 else self.repeat_penalty_config.get(self.current_model_tier, 1.15),
            "frequency_penalty": self.frequency_penalty_config.get(self.current_model_tier, 0.0),
            "presence_penalty": self.presence_penalty_config.get(self.current_model_tier, 0.0),
            "stop": stops,
            #"add_bos": True, # Ensure official BOS (token 2) is always prepended #TODO: remove if dead code
            "top_k": self.top_k_config.get(self.current_model_tier, 64), # Gemma-4 Official
        }
        
        # Dynamic Max Tokens (Context Headroom Management)
        ctx = self.context_size_config.get(self.current_model_tier, 4096)
        ratio = int(self.config.get("max_token_ratio", 4))
        calculated_max = ctx // ratio
        
        if self.current_model_tier and self.current_model_tier.startswith("vision_"):
            inf_params["max_tokens"] = max(512, calculated_max)
        else:
            inf_params["max_tokens"] = max(256, calculated_max)

        
        # Live hot-reload and overlay with params.json values if present
        self.load_params(self.current_model_tier)
        if hasattr(self, "params") and self.params:
            override_params = dict(self.params)
            
            # Special handling for stop strings: MERGE them rather than overwrite
            if "stop" in override_params and isinstance(override_params["stop"], list):
                inf_params["stop"].extend([s for s in override_params["stop"] if s not in inf_params["stop"]])
                del override_params["stop"]
                
            inf_params.update(override_params)

        m_lower = (self.model_path or "").lower()
        if "muse" in m_lower and "glimmer" in m_lower:
            inf_params["stop"] = [s for s in inf_params.get("stop", []) if s != "<|eom|>"]
            
        # --- Sampler Hygiene: Filter unsupported params for llama-cpp-python stability ---
        supported_keys = {
            "temperature", "top_p", "min_p", "repeat_penalty", "frequency_penalty", 
            "presence_penalty", "top_k", "max_tokens", "stop", "stream", 
            "grammar", "logit_bias", "logprobs", "typical_p", "tfs_z", 
            "mirostat_mode", "mirostat_tau", "mirostat_eta", "model", "messages",
            "seed", "echo", "repeat_last_n"
        }
        
        filtered_params = {k: v for k, v in inf_params.items() if k in supported_keys}
        
        dropped = set(inf_params.keys()) - set(filtered_params.keys())
        if dropped:
            print(f"[APEX] Sampler Hygiene: Dropping unsupported parameters: {dropped}")

        # Dynamic Parameter Auto-Adjustment (Non-destructive in-memory tuning)
        if self.config.get("dynamic_params_enabled", True) and temp_messages:
            prompt_text = ""
            if isinstance(temp_messages, list) and temp_messages:
                for m in reversed(temp_messages):
                    if isinstance(m, dict) and m.get("role") == "user":
                        prompt_text = m.get("content", "")
                        break
            elif isinstance(temp_messages, str):
                prompt_text = temp_messages
            
            if prompt_text:
                adjusted, domain = self.dynamic_param_registry.adjust_params(prompt_text, filtered_params)
                if domain:
                    print(f"[DYNAMIC PARAMS] {domain} task detected -> Adjusted sampling: temp={adjusted.get('temperature'):.2f}, top_p={adjusted.get('top_p'):.2f}, min_p={adjusted.get('min_p'):.2f}")
                    filtered_params = adjusted
            
        return filtered_params
        
    def _run_blocking_inference(self, prompt_or_msgs, params):
        """GIL-Safe background inference. Forces internal streaming to keep UI responsive."""
        full_text = ""
        is_gemma = "gemma" in self.model_path.lower()
        
        # 1. THE BREATHER: Let UI process status updates before heavy prefill
        print(f"[INFERENCE] Starting blocking inference ({'Gemma' if is_gemma else 'Standard'}).")
        time.sleep(0.05)
        
        # Ensure we have a working params dict
        safe_params = dict(params)
        safe_params["stream"] = True
        
        try:
            if isinstance(prompt_or_msgs, str):
                # Raw Completion Path
                gen = self.model(prompt_or_msgs, echo=False, **safe_params)
                for chunk in gen:
                    if self.stop_process.is_set(): break
                    token = chunk["choices"][0]["text"]
                    full_text += token
                    self.process_queue.put({"status": "streaming", "content": token})
                    time.sleep(0.005) 
            else:
                # Chat Completion Path
                gen = self.model.create_chat_completion(messages=prompt_or_msgs, **safe_params)
                for chunk in gen:
                    if self.stop_process.is_set(): break
                    if "content" in chunk["choices"][0]["delta"]:
                        txt = chunk["choices"][0]["delta"]["content"]
                        full_text += txt
                        self.process_queue.put({"status": "streaming", "content": txt})
                    time.sleep(0.005)
        except Exception as e:
            print(f"[APEX] Blocking Inference Failure: {e}")
        
        print(f"[INFERENCE] Blocking inference complete. Result length: {len(full_text)} chars.")
        return full_text.strip()

    def run_auto_detect(self, window=None):
        return run_auto_detect(self, window)

    def set_ui_state(self, model_loaded=None, generating=None, loading=None):
        is_loading, is_gen = (loading is True), (generating is True)
        is_loaded = (self.model is not None) if model_loaded is None else model_loaded
        
        btn_load = self.load_model_button
        btn_act = self.action_button
        btn_hurry = self.hurry_button
        btn_deep = self.deep_thought_button
        
        try:
            if btn_load is not None:
                btn_load.config(state='disabled' if (is_loading or is_gen) else 'normal')
            
            if btn_act is not None:
                if is_loading: btn_act.config(text="Loading...", state="disabled")
                elif is_gen: btn_act.config(text="Offload", command=self.offload_model, state="disabled")
                elif is_loaded: 
                    btn_act.config(text="Offload", command=self.offload_model, state="normal")
                else: 
                    btn_text = "Begin!"
                    cmd = self.model_swap
                    if self.state.get("staged_multimodal"):
                        staged_type = self.state["staged_multimodal"]["type"]
                        last_intent = self.state.get("last_vision_intent", f"vision_{staged_type}")
                        btn_text = f"Load Vision ({staged_type.title()})"
                        cmd = lambda t=last_intent: self.model_swap(target_tier=t)
                    btn_act.config(text=btn_text, command=cmd, state="normal")
            
            if btn_hurry is not None:
                btn_hurry.config(state='normal' if (is_gen or is_loading) else 'disabled')
            if btn_deep is not None:
                btn_deep.config(state='disabled' if (is_loading or is_gen) else 'normal')
            
            # Synchronize Ghost and History Usage button labels & colors
            if hasattr(self, 'ghost_button') and self.ghost_button is not None:
                self.ghost_button.config(
                    text=self._get_ghost_mode_label(),
                    fg=self._get_ghost_mode_color(),
                    state='disabled' if is_gen else 'normal'
                )
            if hasattr(self, 'history_usage_button') and self.history_usage_button is not None:
                self.history_usage_button.config(
                    text=self._get_history_usage_label(),
                    fg=self._get_history_usage_color(),
                    state='disabled' if is_gen else 'normal'
                )
            if hasattr(self, 'mic_button') and self.mic_button is not None:
                if not (hasattr(self, 'stt_manager') and self.stt_manager.is_recording):
                    self.mic_button.config(state='disabled' if (is_gen or is_loading) else 'normal')
        except: pass

    def _handle_input_key(self, event):
        if event.keysym == 'Return' and not (event.state & 0x0001): self.send_message(); return 'break'

    def clear_current_history(self):
        self.messages = []
        self.clear_chat_ui()
        if hasattr(self, 'past_history_view'):
            self.past_history_view.config(state='normal')
            self.past_history_view.delete('1.0', tk.END)
            self.past_history_view.config(state='disabled')

    def open_settings_window(self):
        open_settings_window(self)

    def open_text_scaling_center(self):
        open_text_scaling_center(self)

    def _is_rgb_supported(self):
        """Returns the cached RGB support value, defaulting to False on launch to hide by default."""
        if getattr(self, "_rgb_supported_val", None) is not None:
            return self._rgb_supported_val
        return False

    def _check_rgb_support_async(self):
        """Asynchronously queries RGB support so as to not block the main Tkinter thread on launch."""
        def check():
            supported = self._check_is_rgb_supported_actual()
            self._rgb_supported_val = supported
            if supported and self.config.get("show_rgb_button", True):
                # Safely update UI from the main thread to show it if enabled
                self.root.after(0, lambda: self.rgb_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, before=self.send_button) if hasattr(self, 'rgb_button') and self.rgb_button.winfo_exists() else None)
        threading.Thread(target=check, daemon=True).start()

    def _check_is_rgb_supported_actual(self):
        """Checks if MSI Mystic Light SDK is present and supports compatible hardware."""
        if os.name != "nt":
            return False
        try:
            dll_path = os.path.join(self.script_dir, "Mystic_light_SDK_1.0.0.08", "MysticLight_SDK_x64.dll")
            if not os.path.exists(dll_path):
                return False
            import ctypes
            dll = ctypes.WinDLL(dll_path)
            dll.MLAPI_Initialize.restype = ctypes.c_int
            if dll.MLAPI_Initialize() != 0:
                return False
            
            # Query if compatible device names (Motherboard/GPU) exist
            dll.MLAPI_GetDeviceNameEx.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p)]
            ctypes.windll.oleaut32.SysAllocString.restype = ctypes.c_void_p
            ctypes.windll.oleaut32.SysAllocString.argtypes = [ctypes.c_wchar_p]
            ctypes.windll.oleaut32.SysFreeString.argtypes = [ctypes.c_void_p]
            
            show_rgb = False
            candidates = ["MSI_MB", "MSI_GPU"]
            for cand_name in candidates:
                cand_bstr = ctypes.windll.oleaut32.SysAllocString(cand_name)
                target = ctypes.c_void_p(cand_bstr)
                name_ptr = ctypes.c_void_p()
                if dll.MLAPI_GetDeviceNameEx(target, 0, ctypes.byref(name_ptr)) == 0:
                    show_rgb = True
                    if name_ptr and name_ptr.value:
                        ctypes.windll.oleaut32.SysFreeString(name_ptr)
                if cand_bstr:
                    ctypes.windll.oleaut32.SysFreeString(cand_bstr)
            
            dll.MLAPI_Release()
            return show_rgb
        except:
            return False

    def _initialize_rgb_state(self):
        """Ensures RGB starts in AUTO mode on launch."""
        state_path = os.path.join(self.script_dir, "System", "rgb_state.json")
        try:
            state = {"mode": "auto", "manual_color": [0, 255, 204], "manual_style": "No animation", "brightness": 100}
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    try: saved = json.load(f); state.update(saved)
                    except: pass
            state["mode"] = "auto"
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            with open(state_path, 'w') as f: json.dump(state, f, indent=4)
            print("[SYSTEM] RGB initialized to AUTO mode.")
        except Exception as e: print(f"Failed to initialize RGB state: {e}")

    def open_rgb_panel(self):
        """Launches the standalone RGB Control Window."""
        try:
            from System.rgb_panel import RGBPanel
            RGBPanel(self.root)
        except Exception as e: messagebox.showerror("UI Error", f"Could not launch RGB Panel:\n{e}")

    def _set_path(self, tier, labels, window=None, is_projector=False):
        ft = [("Projector", "*.mmproj")] if is_projector else [("GGUF", "*.gguf")]
        p = filedialog.askopenfilename(parent=window, filetypes=ft + [("All Files", "*.*")], initialdir=self.dirs["Models"])
        if p: 
            self.model_paths[tier] = p
            labels[tier].config(text=os.path.basename(p))
            self.save_config()
        if window: window.lift()

    def _fill_auto(self, win, ents):
        recs = self.run_auto_detect(win)
        for t, l in recs.items():
            if t in ents: 
                ents[t].delete(0, tk.END)
                ents[t].insert(0, str(l))

    def _get_ghost_mode_label(self):
        active = self.config.get("ghost_mode", False)
        return "👻 Ghost: ON" if active else "👻 Ghost: OFF"

    def _get_ghost_mode_color(self):
        active = self.config.get("ghost_mode", False)
        return "#00FF7F" if active else THEME["fg_color"]

    def toggle_ghost_mode(self):
        active = not self.config.get("ghost_mode", False)
        self.config["ghost_mode"] = active
        self.save_config()
        self.ghost_button.config(text=self._get_ghost_mode_label(), fg=self._get_ghost_mode_color())
        self._log_and_display(f"Ghost Mode {'Enabled' if active else 'Disabled'}.")

    def _get_history_usage_label(self):
        val = self.config.get("history_usage", "all")
        if val == "current_window":
            return "📚 Hist: Window"
        elif val == "off":
            return "📚 Hist: Off"
        return "📚 Hist: All"

    def _get_history_usage_color(self):
        val = self.config.get("history_usage", "all")
        if val == "current_window":
            return "#FFD700"
        elif val == "off":
            return "#FF8A8A"
        return THEME["fg_color"]

    def toggle_history_usage(self):
        modes = ["all", "current_window", "off"]
        current = self.config.get("history_usage", "all")
        if current not in modes:
            current = "all"
        next_idx = (modes.index(current) + 1) % len(modes)
        next_mode = modes[next_idx]
        self.config["history_usage"] = next_mode
        self.save_config()
        self.history_usage_button.config(text=self._get_history_usage_label(), fg=self._get_history_usage_color())
        label_disp = "CURRENT WINDOW" if next_mode == "current_window" else next_mode.upper()
        self._log_and_display(f"History usage set to: {label_disp}")

    def toggle_voice_recording(self):
        """Toggles push-to-record voice input using sounddevice and local STTManager."""
        if not hasattr(self, "stt_manager") or not self.stt_manager.is_available():
            messagebox.showwarning("STT Unavailable", "Audio recording dependencies (sounddevice, speech_recognition) are not available.")
            return

        if not self.stt_manager.is_recording:
            # Start Recording
            dev_idx = self.config.get("stt_device_index", None)
            started = self.stt_manager.start_recording(device_index=dev_idx)
            if started:
                if self.mic_button:
                    self.mic_button.config(text="🔴", fg="#ff4444", bg="#4a0000")
                self._log_and_display("Microphone recording active...")
            else:
                messagebox.showerror("Audio Error", "Failed to start microphone recording. Check audio input device.")
        else:
            # Stop Recording & Begin Transcription
            if self.mic_button:
                self.mic_button.config(text="⏳", fg="#ffd700", bg=THEME["button_bg_color"], state="disabled")
            self._log_and_display("Processing speech-to-text transcription...")
            
            wav_bytes = self.stt_manager.stop_recording()
            if not wav_bytes:
                if self.mic_button:
                    self.mic_button.config(text="🎙️", fg=THEME["fg_color"], bg=THEME["button_bg_color"], state="normal")
                self._log_and_display("No speech detected.")
                return

            lang = self.config.get("stt_language", "en-US")
            
            def _on_stt_done(transcript, err):
                self.process_queue.put({"status": "stt_transcript", "content": transcript, "error": err})

            self.stt_manager.transcribe_wav_bytes(
                wav_bytes,
                language=lang,
                on_complete=_on_stt_done,
                llm_model=self.model if (self.model and getattr(self.model, "chat_handler", None) is not None) else None
            )

    def _handle_stt_result(self, transcript: str, error: Optional[str] = None):
        """Inserts transcribed speech into user input field and resets mic button UI."""
        if self.mic_button:
            self.mic_button.config(text="🎙️", fg=THEME["fg_color"], bg=THEME["button_bg_color"], state="normal")
        
        if transcript:
            if self.user_input:
                existing = self.user_input.get("1.0", tk.END).strip()
                new_text = f"{existing} {transcript}".strip() if existing else transcript
                self.user_input.delete("1.0", tk.END)
                self.user_input.insert("1.0", new_text)
                self.user_input.see(tk.END)
            self._log_and_display(f"Dictated: \"{transcript}\"")
        elif error:
            self._log_and_display(f"STT: {error}")
        else:
            self._log_and_display("No speech recognized.")

    def _init_turbovec(self):
        try:
            cfg = getattr(self, "config", {}) or {}
            mode = cfg.get("turbovec_mode", "fallback")
            self.turbo_vec = TurboVecIndex(self.dirs["History"], mode=mode)
            if mode != "off":
                lookup_mode = cfg.get("history_lookup_mode", "targeted")
                model_p = getattr(self, "model_path", None)
                lvl = getattr(self, "active_persona_level", 1)
                self.turbo_vec.ingest_needed_files(model_p, lvl, lookup_mode)
            print(f"[TURBOVEC] Background initialization complete (Mode: {mode.upper()}).")
        except ImportError as e:
            print(f"[TURBOVEC] Optional module not installed: {e}")
        except Exception as e:
            print(f"[TURBOVEC] Background init failed: {e}")

    def soft_reload_turbovec(self):
        """Soft-reloads the TurboVec subsystem without restarting or dropping models."""
        cfg = getattr(self, "config", {}) or {}
        mode = cfg.get("turbovec_mode", "fallback")
        def _reload():
            try:
                print(f"[TURBOVEC] Soft reloading TurboVec subsystem (Target Mode: {mode.upper()})...")
                self.turbo_vec = TurboVecIndex(self.dirs["History"], mode=mode)
                if mode != "off":
                    lookup_mode = cfg.get("history_lookup_mode", "targeted")
                    model_p = getattr(self, "model_path", None)
                    lvl = getattr(self, "active_persona_level", 1)
                    self.turbo_vec.ingest_needed_files(model_p, lvl, lookup_mode)
                print(f"[TURBOVEC] Soft reload complete.")
            except Exception as e:
                print(f"[TURBOVEC] Soft reload error: {e}")
        threading.Thread(target=_reload, daemon=True).start()

sys.excepthook = log_uncaught_exception

if __name__ == "__main__":
    try:
        enable_high_dpi_awareness()
        print("Starting SerenityPC...")
        root = tk.Tk()
        root.withdraw()
        
        # Instantly display the loading screen
        from System.serenity_utils import LoadingScreen
        ls = LoadingScreen(root)
        ls.start_animation()
        
        # Force Tkinter to draw the window right now
        root.update()
        
        # Start background loading thread
        import threading
        load_thread = threading.Thread(target=load_heavy_libraries, daemon=True)
        load_thread.start()
        
        def check_libraries():
            # Check if thread is done
            if load_thread.is_alive():
                # Not done, check again in 50ms
                root.after(50, check_libraries)
            else:
                # Thread finished. Let's verify libraries loaded.
                if not LIBRARIES_LOADED:
                    messagebox.showerror("Dependency Error", EARLY_IMPORT_ERROR_MSG or "Failed to load libraries.")
                    root.quit()
                    return
                
                # Success! Now initialize the main app
                try:
                    app = ChatbotApp(root, ls)
                except Exception as ex:
                    log_uncaught_exception(type(ex), ex, ex.__traceback__)
                    root.quit()
        
        # Start checking
        root.after(50, check_libraries)
        root.mainloop()
    except (KeyboardInterrupt, SystemExit):
        pass
    except Exception as e:
        log_uncaught_exception(type(e), e, e.__traceback__)
    finally:
        if SYSTEM_MONITOR_LOADED and nvidia_ml: 
            try: nvidia_ml.nvmlShutdown()
            except: pass