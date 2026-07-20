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

import tkinter as tk
from tkinter import scrolledtext, simpledialog, messagebox, filedialog, ttk
import tkinter.font as tkFont
import threading, traceback, sys, json, zlib, time, queue, subprocess, re, atexit, webbrowser, requests, io, faulthandler, struct
from typing import TYPE_CHECKING, Any, Dict, List, Optional
if TYPE_CHECKING:
    from tkinter import Canvas, Label, Button, Frame, Text, Scale
    from tkinter.scrolledtext import ScrolledText
    from typing import cast
    import torch
    from turboquant import TurboQuantCache
    import psutil

# --- Import Custom Modules ---
from serenity_resources import (THEME, THERMO_COLORS, CHAT_BG_COLORS, CHAT_FG_COLORS, 
                              INPUT_FG_COLORS, GPU_LAYER_MAP, CONTEXT_SIZE_MAP, 
                              PERSONA_DISPLAY_INFO, PERSONA_IDLE_MAP, PERSONA_PROMPTS, 
                               AVATAR_FILENAMES, ANIMATION_SEQUENCE, DEEP_COOK_PHASES, 
                               DEEP_COOK_SYSTEM_PROMPTS, APP_ICON, TOOLS_DIR,
                               TRI_ATTENTION_ENABLED, TRI_ATTENTION_BUDGET)
from System.serenity_utils import (WidgetLogger, FileAndWidgetLogger, LoadingScreen, 
                            log_uncaught_exception, HardwareProfile, MediaProcessor, SystemMonitor,
                            enable_fault_debugging)
#from System.ui_watchdog import UIWatchdog #commented out for now to save threads
from System.kv_manager import KVManager

# --- Debugging & Fault Handling ---
enable_fault_debugging()

# --- Hardware Initialization ---
HardwareProfile.initialize_gpu_acceleration()


# --- Custom Logic Scripts ---
# (Moved inside the try-except block for robustness)

# --- Library Imports ---
LIBRARIES_LOADED = False
try:
    print("Importing Llama, PIL, and CV2...")
    import llama_cpp
    from llama_cpp import Llama
    from PIL import Image, ImageTk
    import cv2
    import windnd
    from System.vision_handler import VisionHandler
    from System.synthesis_handler import generate_master_summary
    from System import settings_manager
    LIBRARIES_LOADED = True
except Exception as e:
    EARLY_IMPORT_ERROR_MSG = f"FATAL ERROR: Missing library.\n\n{e}"
    print(EARLY_IMPORT_ERROR_MSG, file=sys.stderr)

SYSTEM_MONITOR_LOADED = False
TORCH_AVAILABLE = False
try:
    import psutil
    import pynvml as nvidia_ml
    nvidia_ml.nvmlInit()
    SYSTEM_MONITOR_LOADED = True
except ImportError as e:
    print(f"Warning: System monitoring libraries (psutil/pynvml) not found. {e}", file=sys.stderr)

try:
    import torch
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


# --- GIL-free Thread-Safety Utilities (Python 3.13/3.14+) ---
class ThreadSafeDict(dict):
    """
    A thread-safe dictionary subclass wrapper designed for GIL-free Python (3.13/3.14+).
    Uses a reentrant lock to synchronize all read, write, and deletion operations.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = threading.RLock()

    def __getitem__(self, key):
        with self._lock:
            return super().__getitem__(key)

    def __setitem__(self, key, value):
        with self._lock:
            super().__setitem__(key, value)

    def __delitem__(self, key):
        with self._lock:
            super().__delitem__(key)

    def __contains__(self, key):
        with self._lock:
            return super().__contains__(key)

    def get(self, key, default=None):
        with self._lock:
            return super().get(key, default)

    def setdefault(self, key, default=None):
        with self._lock:
            return super().setdefault(key, default)

    def pop(self, key, default=None):
        with self._lock:
            return super().pop(key, default)

    def popitem(self):
        with self._lock:
            return super().popitem()

    def clear(self):
        with self._lock:
            super().clear()

    def update(self, *args, **kwargs):
        with self._lock:
            super().update(*args, **kwargs)

    def keys(self):
        with self._lock:
            return list(super().keys())

    def values(self):
        with self._lock:
            return list(super().values())

    def items(self):
        with self._lock:
            return list(super().items())

    def copy(self):
        with self._lock:
            return ThreadSafeDict(super().copy())

    def __len__(self):
        with self._lock:
            return super().__len__()

    def __repr__(self):
        with self._lock:
            return super().__repr__()


class ThreadSafeList(list):
    """
    A thread-safe list subclass wrapper designed for GIL-free Python (3.13/3.14+).
    Uses a reentrant lock to synchronize all read, write, and iteration operations.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = threading.RLock()

    def append(self, item):
        with self._lock:
            super().append(item)

    def extend(self, iterable):
        with self._lock:
            super().extend(iterable)

    def insert(self, index, item):
        with self._lock:
            super().insert(index, item)

    def remove(self, item):
        with self._lock:
            super().remove(item)

    def pop(self, index=-1):
        with self._lock:
            return super().pop(index)

    def clear(self):
        with self._lock:
            super().clear()

    def __getitem__(self, index):
        with self._lock:
            if isinstance(index, slice):
                return ThreadSafeList(super().__getitem__(index))
            return super().__getitem__(index)

    def __setitem__(self, index, value):
        with self._lock:
            super().__setitem__(index, value)

    def __delitem__(self, index):
        with self._lock:
            super().__delitem__(index)

    def __len__(self):
        with self._lock:
            return super().__len__()

    def __iter__(self):
        with self._lock:
            return iter(list(super().__iter__()))

    def __repr__(self):
        with self._lock:
            return super().__repr__()

    def copy(self):
        with self._lock:
            return ThreadSafeList(super().copy())


class ThinkingDisplay(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, bg=THEME["bg_color"], *args, **kwargs)
        self.label = tk.Label(self, text="Thinking...", font=("Open Sans", 10, "italic"), 
                            fg=THEME["electric_blue"], bg=THEME["bg_color"])
        self.label.pack(side=tk.LEFT, padx=5)
        self.progress = ttk.Progressbar(self, mode='indeterminate', length=150)
        self.progress.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    
    def start(self):
        if not self.winfo_exists(): return
        self.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        self.progress.start(15)
        
    def stop(self):
        if not self.winfo_exists(): return
        self.progress.stop()
        self.pack_forget()
        self.label.config(text="Thinking...")

    def update_status(self, text):
        if not self.winfo_exists(): return
        self.label.config(text=text)


class GemmaToolRegistry:
    """Handles tool definitions and execution for Gemma-4 models."""
    def __init__(self, chatbot_app):
        self.app = chatbot_app
        self.tools = [
            {
                "function": {
                    "name": "get_system_stats",
                    "description": "Returns current CPU, RAM, and GPU utilization for hardware health monitoring.",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "function": {
                    "name": "read_file",
                    "description": "Reads the first 5000 characters of a local text file for analysis.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Absolute path to the file."}
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "function": {
                    "name": "web_search",
                    "description": "Searches the live web for real-time data, current events, weather, news, and specialized technical info not present in your training data.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query."}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "function": {
                    "name": "control_rgb",
                    "description": "Adjusts the system RGB lighting color or style.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "color": {"type": "array", "items": {"type": "integer"}, "description": "[R, G, B] values (0-255)."},
                            "style": {"type": "string", "description": "Hardware style: 'Steady', 'Breathing', 'Rainbow', 'Flash', etc."}
                        },
                        "required": []
                    }
                }
            },

            {
                "function": {
                    "name": "generate_image",
                    "description": "Generates an image or diagram. Use markdown formatting or Mermaid logic if drawing a technical diagram.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string", "description": "Description of image or raw Mermaid/SVG code."},
                            "type": {"type": "string", "description": "Type: 'image' or 'diagram'"}
                        },
                        "required": ["prompt", "type"]
                    }
                }
            }
        ]

    def execute(self, call_name, args):
        """Executes a tool call and returns the result as a string."""
        print(f"[TOOL] Executing: {call_name} with args: {args}")
        try:
            if call_name == "web_search":
                query = args.get("query", "").strip()
                if not query: return "Error: Search query cannot be empty."
                
                import requests
                import urllib.parse
                from bs4 import BeautifulSoup
                
                # Chrome-based User-Agent
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'DNT': '1'
                }
                
                # Tiered Resilience Strategy: Brave (Primary) -> Bing -> Playwright Fallback
                
                # 1. Brave Search (Fast & Resilient)
                try:
                    brave_url = "https://search.brave.com/search?q=" + urllib.parse.quote(query)
                    resp = requests.get(brave_url, headers=headers, timeout=8)
                    if resp.status_code == 200:
                        soup = BeautifulSoup(resp.text, 'html.parser')
                        results = []
                        # Selector-Agnostic Harvesting: Target any div that looks like a result
                        for res in soup.select('div.snippet, div.result, .search-result, .snippet'):
                            title = res.select_one('h2, .title, .search-snippet-title')
                            snippet = res.select_one('p, .content, .snippet-description, .snippet-content')
                            if title and snippet:
                                results.append(f"[{title.get_text(strip=True)}]\n{snippet.get_text(strip=True)}")
                        
                        if len(results) >= 1:
                            proof_msg = f"[SEARCH PROOF] Provider: Brave | Status: {resp.status_code} | Found: {len(results)}"
                            self.app.process_queue.put({"status": "tool_log_update", "content": f"\n{proof_msg}"})
                            return f"Brave Search Results for '{query}':\n\n" + "\n\n".join(results[:5])
                except Exception as e:
                    print(f"[SEARCH DEBUG] Brave failed: {e}")

                # 2. Bing Scraper (High Recall)
                try:
                    bing_url = f"https://www.bing.com/search?q={urllib.parse.quote(query)}"
                    resp = requests.get(bing_url, headers=headers, timeout=5)
                    if resp.status_code == 200:
                        soup = BeautifulSoup(resp.text, 'html.parser')
                        results = []
                        for res in soup.select('li.b_algo'):
                            title = res.select_one('h2')
                            snippet = res.select_one('.b_caption p, .b_snippet')
                            if title and snippet:
                                results.append(f"[{title.get_text(strip=True)}]\n{snippet.get_text(strip=True)}")
                        
                        if len(results) >= 2:
                            proof_msg = f"[SEARCH PROOF] Provider: Bing | Status: {resp.status_code} | Found: {len(results)}"
                            self.app.process_queue.put({"status": "tool_log_update", "content": f"\n{proof_msg}"})
                            return f"Bing Search Context for '{query}':\n\n" + "\n\n".join(results[:5])
                except: pass

                # 3. DuckDuckGo (Scraper Friendly Fallback)
                try:
                    ddg_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
                    resp = requests.get(ddg_url, headers=headers, timeout=5)
                    if resp.status_code == 200:
                        soup = BeautifulSoup(resp.text, 'html.parser')
                        results = []
                        for res in soup.select('.result__body'):
                            title = res.select_one('.result__title')
                            snippet = res.select_one('.result__snippet')
                            if title and snippet:
                                results.append(f"[{title.get_text(strip=True)}]\n{snippet.get_text(strip=True)}")
                        if len(results) >= 2:
                            proof_msg = f"[SEARCH PROOF] Provider: DuckDuckGo | Status: {resp.status_code} | Found: {len(results)}"
                            self.app.process_queue.put({"status": "tool_log_update", "content": f"\n{proof_msg}"})
                            return f"DuckDuckGo Context for '{query}':\n\n" + "\n\n".join(results[:5])
                except: pass

                # 4. Deep Browse (Playwright Headless) - The 'Nuclear Option'
                try:
                    from playwright.sync_api import sync_playwright
                    self.app.process_queue.put({"status": "thinking_status", "content": "USR: Initiating Stealth Browser Instance..."})
                    
                    with sync_playwright() as p:
                        browser = p.chromium.launch(headless=True)
                        page = browser.new_page(user_agent=headers['User-Agent'])
                        page.goto(f"https://www.bing.com/search?q={urllib.parse.quote(query)}", wait_until="load", timeout=15000)
                        
                        # Wait a bit for JS results to render
                        page.wait_for_timeout(3000)
                        
                        # Broader harvesting: Target Bing results OR any substantial text blocks
                        content = page.evaluate("""() => {
                            const results = [];
                            const bingResults = document.querySelectorAll('li.b_algo, .b_caption, .b_snippet');
                            if (bingResults.length > 0) {
                                bingResults.forEach(el => results.push(el.innerText));
                            } else {
                                // Fallback to any substantial text block (selector-agnostic)
                                document.querySelectorAll('p, span, div, h2').forEach(el => {
                                    const txt = el.innerText.trim();
                                    if (txt.length > 80 && !txt.includes('{')) {
                                        results.push(txt);
                                    }
                                });
                            }
                            return results.slice(0, 10);
                        }""")
                        browser.close()
                        
                        if content:
                            proof_msg = f"[SEARCH PROOF] Provider: Playwright (Bing) | Content Fragments: {len(content)}"
                            self.app.process_queue.put({"status": "tool_log_update", "content": f"\n{proof_msg}"})
                            return f"Deep Web Extract for '{query}':\n\n" + "\n\n".join(content)
                except Exception as e:
                    print(f"[SEARCH DEBUG] Playwright failed: {e}")
                
                return "Error: All search providers were unreachable or blocked."
            
            
            if call_name == "generate_image":
                prompt = args.get("prompt", "")
                req_type = args.get("type", "image")
                
                # Standalone borderless viewer script call
                import threading, subprocess
                def spawn_viewer():
                    # Use project-relative paths instead of hardcoded S: drive
                    scratch_dir = os.path.join(self.app.script_dir, "scratch")
                    os.makedirs(scratch_dir, exist_ok=True)
                    temp_script = os.path.join(scratch_dir, "temp_viewer.py")
                    
                    script_content = f"""import tkinter as tk
root = tk.Tk()
root.overrideredirect(True)
root.attributes('-topmost', True)
root.attributes('-alpha', 0.9)
root.geometry("400x300+100+100")
root.config(bg='black')
tk.Label(root, text='[Serenity Image / Diagram Viewer]', fg='#00ffcc', bg='black', font=('Consolas', 10)).pack(pady=10)
tk.Label(root, text={repr(prompt)[:500]}, fg='white', bg='black', wraplength=380).pack(pady=10)
tk.Button(root, text='[X] Close', command=root.destroy, bg='#222', fg='white').pack(side=tk.BOTTOM, pady=10)
root.mainloop()"""
                    with open(temp_script, "w", encoding="utf-8") as f:
                        f.write(script_content)
                    subprocess.Popen(["python", temp_script])
                
                threading.Thread(target=spawn_viewer, daemon=True).start()
                return f"Successfully generated and displayed {req_type} via borderless HUD overlay."
                
            if call_name == "get_system_stats":
                # Leverage existing SystemMonitor logic if possible, or just raw psutil
                stats = {
                    "cpu": f"{psutil.cpu_percent()}%",
                    "ram": f"{psutil.virtual_memory().percent}%",
                }
                if nvidia_ml:
                    try:
                        handle = nvidia_ml.nvmlDeviceGetHandleByIndex(0)
                        mem = nvidia_ml.nvmlDeviceGetMemoryInfo(handle)
                        stats["vram"] = f"{mem.used/1024**2:.0f} / {mem.total/1024**2:.0f} MB"
                    except: pass
                return json.dumps(stats)
            
            elif call_name == "read_file":
                path = args.get("path")
                if not path or not os.path.exists(path): return "Error: File not found."
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read(5000)
            
            
            elif call_name == "control_rgb":
                state_path = os.path.join(self.app.script_dir, "System", "rgb_state.json")
                try:
                    with open(state_path, 'r') as f:
                        state = json.load(f)
                    
                    if "color" in args: state["manual_color"] = args["color"]
                    if "style" in args: state["manual_style"] = args["style"]
                    state["mode"] = "manual"
                    
                    with open(state_path, 'w') as f:
                        json.dump(state, f, indent=4)
                    return f"RGB adjusted: Mode=Manual, Color={args.get('color')}, Style={args.get('style')}"
                except Exception as e:
                    return f"Error controlling RGB: {str(e)}"

            return f"Error: Tool {call_name} not implemented."
        except Exception as e:
            return f"Error executing tool: {str(e)}"

    def get_definitions(self, level=1):
        """Returns tool definitions permitted for the current persona level."""
        if level < 2: return [] # Lvl 1 has no tools for maximum speed
        if level < 5: return [self.tools[0], self.tools[2], self.tools[3]] # Lvl 2-4 get Stats, Search, and RGB
        return self.tools # Lvl 5+ get everything

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
        tool_log: Optional['Text']
        diag_log: Optional['Text']
        load_model_button: Optional['Button']
        action_button: Optional['Button']
        live_agent_button: Optional['Button']
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
        live_agent_process: Optional[Any]
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

        self.dirs = {d: os.path.join(self.script_dir, d) for d in ["Media", "History", "Models", "Logs", "System"]}
        for d in self.dirs.values(): os.makedirs(d, exist_ok=True)
        
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

        # --- Visual Resources ---
        self.fonts = {
            "main": tkFont.Font(family="Open Sans", size=12),
            "small": tkFont.Font(family="Open Sans", size=14),
            "italic": tkFont.Font(family="Open Sans", size=14, slant="italic"),
            "large": tkFont.Font(family="Open Sans", size=18),
            "bold": tkFont.Font(family="Open Sans", size=18, weight="bold"),
            # Markdown Support
            "md_bold": tkFont.Font(family="Open Sans", size=12, weight="bold"),
            "md_italic": tkFont.Font(family="Open Sans", size=12, slant="italic"),
            "md_bold_italic": tkFont.Font(family="Open Sans", size=12, weight="bold", slant="italic"),
            "md_thought": tkFont.Font(family="Consolas", size=9, slant="italic")
        }
        
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
        
        # Load DMN Backbone
        self._load_dmn_backbone()
        self.max_persona_level = 5  # Persisted range for the slider
        self.messages = []
        self.live_agent_process = None
        self.gpu_handle = None
        self.text_buffer = ""
        self.last_update_time = 0.0
        self.chunk_counter = 0 
        
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
        self.log_container = None
        self.log_switch_canvas = None
        self.switch_knob = None
        self.log_frame = None
        self.thought_log = None
        self.error_log = None
        self.tool_log = None
        self.diag_log = None
        self.load_model_button = None
        self.action_button = None
        self.live_agent_button = None
        self.btn_image = None
        self.btn_video = None
        self.btn_watch = None
        self.btn_clear_queue = None
        self.btn_active = None
        self.btn_history = None
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
        self.last_log_dispatch = 0
        self.log_update_limit = 100 # Max messages per queue tick to prevent UI freeze

        # --- Queue Dispatch Table ---
        self.queue_handlers: Dict[str, Any] = {
            "load_success": lambda msg: self._handle_load_success(msg),
            "load_error": lambda msg: self._handle_load_error(msg),
            "stats_update": lambda msg: self._update_stats_display(msg.get("stats", {})) if self.stats_labels else None,
            "log_update": lambda msg: self._buffer_log(msg.get("content", "")),
            "tool_log_update": lambda msg: self._buffer_tool_log(msg.get("content", "")),
            "diag_log_update": lambda msg: self._buffer_diag_log(msg.get("content", "")),
            "thinking_status": lambda msg: self.thinking_display.update_status(msg.get("content", "Thinking...")) if self.thinking_display and self.thinking_display.winfo_exists() else None,
            "streaming": lambda msg: self._buffer_text(msg.get("content", "")),
            "success": lambda msg: self._handle_session_finished({"user_msg": self.last_user_message, "final_answer": msg.get("content", ""), "is_error": False}),
            "session_finished": lambda msg: self._handle_session_finished(msg),
            "interrupted": lambda msg: self._handle_session_finished({"user_msg": self.last_user_message, "final_answer": msg.get("content", ""), "is_error": True}),
            "deep_cook_ui_batch": lambda msg: self._handle_deep_cook_ui_batch(msg),
            "deep_cook_ui_start": lambda msg: self._handle_deep_cook_ui_start(msg),
            "deep_cook_ui_stream": lambda msg: self._handle_deep_cook_ui_stream(msg),
            "vision_oneshot_finish": lambda msg: self.offload_model(),
            "video_progress": lambda msg: self._set_progress(msg.get("content", 0)),
            "error": lambda msg: self._handle_session_finished({"user_msg": self.last_user_message, "final_answer": msg.get("content", ""), "is_error": True}),
            "cleanup": lambda msg: self._run_hygiene_on_main_thread()
        }

        self._setup_llama_log_capture()
        self.setup_ui()
        self.root.after(100, lambda *args: self.final_initial_setup())
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.set_ui_state(model_loaded=False, generating=False)

    # ================= UI & SETUP =================
    def final_initial_setup(self):
        if self.state["initial_setup"]: return
        self.state["initial_setup"] = True
        self.config = self.load_config()
        if 'main_window' in self.config: self.root.geometry(self.config['main_window'])
        
        if self.loading_screen:
            self.loading_screen.stop_and_destroy()
            self.root.deiconify()
        
        self.redirect_logs()
        set_apex_affinity()
        HardwareProfile.set_priority("above_normal")
        self.root.update_idletasks()
        
        # Start UI Watchdog to detect freezes
        #self.ui_watchdog = UIWatchdog(self.root)
        #self.ui_watchdog.start() #commented out for now to save threads
        
        self.load_all_images()
        self.initialize_app()
        self.system_monitor.start()
        self.check_gpu_support()
        self._initialize_rgb_state()
        
        # Windnd Drag and Drop Hook
        try:
            windnd.hook_dropfiles(self.root, func=self._handle_drop_files)
        except Exception as e:
            print(f"Failed to hook drag-and-drop: {e}")

        # Non-blocking, lazy query for RGB support
        self._check_rgb_support_async()

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
        
        style = {"font": self.fonts["large"], "bg": THEME["button_bg_color"], "fg": THEME["fg_color"]}
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
        
        left.grid_rowconfigure(1, weight=1) # Chat History preference
        left.grid_rowconfigure(4, weight=0) # Description box stable
        left.grid_columnconfigure(0, weight=1)

        # --- TOP BUTTONS ---
        top = tk.Frame(left, bg=THEME["bg_color"])
        top.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        
        btn_set = self._add_btn(top, "Settings", self.open_settings_window)
        self.load_model_button = btn_set
        
        btn_act = self._add_btn(top, "Begin!", self.model_swap)
        self.action_button = btn_act
        
        btn_live = self._add_btn(top, "Live", self.toggle_live_agent)
        self.live_agent_button = btn_live
        
        # Multimodal Prep Buttons
        btn_vid = self._add_btn(top, "[🎥] Video", self.initiate_video_multimodal)
        self.btn_video = btn_vid
        
        # Replace the old Watch button
        btn_wat = self._add_btn(top, "[🧠] Pulse", self.toggle_auto_watch)
        self.btn_watch = btn_wat
        
        btn_clr = self._add_btn(top, "Clear", self._reset_multimodal_ui)
        self.btn_clear_queue = btn_clr

        chat_frame = tk.Frame(left, bg=THEME["trim_color"])
        chat_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        # --- TAB CONTROLS ---
        tab_frame = tk.Frame(chat_frame, bg=THEME["trim_color"])
        tab_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 2))

        btn_tab_act = tk.Button(tab_frame, text="Active Chat", command=self.show_active_chat, 
                                bg=THEME["button_active_color"], fg=THEME["fg_color"], relief=tk.FLAT)
        self.btn_active = btn_tab_act
        btn_tab_act.pack(side=tk.LEFT, padx=2)

        btn_tab_hist = tk.Button(tab_frame, text="History Archive", command=self.show_history, 
                                 bg=THEME["button_bg_color"], fg="#aaaaaa", relief=tk.FLAT)
        self.btn_history = btn_tab_hist
        btn_tab_hist.pack(side=tk.LEFT, padx=2)

        lbl_status = tk.Label(tab_frame, text="System: Ready", bg=THEME["trim_color"], 
                                          fg="#888888", font=("Open Sans", 10, "italic"))
        self.system_status_label = lbl_status
        lbl_status.pack(side=tk.RIGHT, padx=10)

        lbl_hw = tk.Label(tab_frame, text="", bg=THEME["trim_color"], font=("Open Sans", 10, "bold"))
        self.hw_mode_label = lbl_hw
        lbl_hw.pack(side=tk.RIGHT, padx=5)
        self._update_hw_indicator()

        # Thinking Display
        s_frame = tk.Frame(chat_frame, bg=THEME["trim_color"])
        self.status_frame = s_frame
        s_frame.pack(side=tk.TOP, fill=tk.X)
        self.thinking_display = ThinkingDisplay(s_frame)

        # --- TEXT WIDGETS ---
        # 1. Floating Pinned Prompt (Hidden on startup)
        txt_prompt = tk.Text(chat_frame, height=3, font=self.fonts["italic"], wrap=tk.WORD,
                             bg=THEME["trim_color"], fg="#87CEFA", relief=tk.FLAT, 
                             highlightthickness=0, padx=10, pady=5)
        self.prompt_display = txt_prompt

        # 2. User-Provided Timeline Progress (Apex Dark Theme)
        self.timeline_frame = tk.Frame(chat_frame, bg="#1e1e1e")
        self.progress_label = tk.Label(self.timeline_frame, text="TIMELINE: 0%", bg="#1e1e1e", fg="#00ffcc", font=("Consolas", 9))
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

        input_frame = tk.Frame(left, bg=THEME["trim_color"])
        input_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        
        # Attachment Bar (New)
        self.attachment_frame = tk.Frame(input_frame, bg=THEME["trim_color"])
        self.attachment_frame.pack(side=tk.TOP, fill=tk.X, padx=2, pady=(2,0))
        
        txt_user = tk.Text(input_frame, height=3, font=self.fonts["main"], wrap=tk.WORD,
                           bg=THEME["widget_bg_color"], fg=THEME["fg_color"], relief=tk.FLAT)
        self.user_input = txt_user
        txt_user.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        txt_user.bind("<KeyPress>", self._handle_input_key)

        # --- PERSONA CONTROLS (Single Slider Fix) ---
        p_frame = tk.Frame(left, bg=THEME["bg_color"])
        p_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=5)
        self._setup_persona_controls(p_frame)
        
        # --- PERSONA DESCRIPTION (UI Fix for Cutoff) ---
        d_cont = tk.Frame(left, bg=THEME["bg_color"])
        self.desc_container = d_cont
        d_cont.grid(row=4, column=0, sticky="ew", padx=10, pady=2)
        
        lbl_desc = tk.Label(d_cont, text="", font=self.fonts["small"], 
                                          bg=THEME["bg_color"], fg=THEME["electric_blue"],
                                          anchor="center", wraplength=500)
        self.persona_desc_label = lbl_desc
        lbl_desc.pack(fill=tk.BOTH, expand=True)

        def _on_left_resize(event):
            # Dynamic wraplength: 90% of the left frame width
            new_width = event.width - 40
            if new_width > 50:
                lbl_desc.config(wraplength=new_width)
        
        left.bind("<Configure>", _on_left_resize)

        # --- FOOTER BUTTONS ---
        ctrl_frame = tk.Frame(left, bg=THEME["bg_color"])
        ctrl_frame.grid(row=5, column=0, sticky="ew", padx=10, pady=5)
        
        self.rgb_button = self._add_btn(ctrl_frame, "[🌈] RGB", self.open_rgb_panel, side=tk.LEFT, width=12)
        if not self.config.get("show_rgb_button", True) or not self._is_rgb_supported():
            self.rgb_button.pack_forget()
        
        btn_send = self._add_btn(ctrl_frame, "Send", self.send_message, side=tk.RIGHT)
        self.send_button = btn_send
        
        btn_deep = self._add_btn(ctrl_frame, "Deep Cook", self.toggle_deep_cook_mode, side=tk.RIGHT)
        self.deep_thought_button = btn_deep
        
        btn_halt = self._add_btn(ctrl_frame, "Halt", self.halt_process, side=tk.RIGHT)
        self.hurry_button = btn_halt

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
        
        # Restore Sash Position
        if 'sash_pos' in self.config:
            self.root.after(300, lambda: self.paned.sash_place(0, self.config['sash_pos'], 0))
        else:
            # Default split (3:2 approx)
            self.root.after(300, lambda: self.paned.sash_place(0, int(self.root.winfo_width() * 0.6), 0))
            
        self.root.after(250, lambda *args: self._position_canvas_elements())

    def _update_hw_indicator(self):
        """Updates the Hardware Mode indicator based on CPU specs."""
        info = HardwareProfile.get_cpu_info()
        physical = info["physical"]
        # Threshold: i7 usually has > 8 physical cores (or 12+ logical)
        if self.hw_mode_label is not None:
            if physical >= 8:
                self.hw_mode_label.config(text="[APEX i7]", fg="#00FF7F") # Spring Green
            else:
                self.hw_mode_label.config(text="[LEGACY i5]", fg="#FFD700") # Gold

    def _setup_persona_controls(self, p_frame):
        """Sets up the persona selection buttons and slider in the given frame."""
        lbl_p = tk.Label(p_frame, text="Persona:", font=self.fonts["small"], 
                                     bg=THEME["bg_color"], fg=THEME["electric_blue"])
        self.persona_label = lbl_p
        lbl_p.pack(side=tk.LEFT)
        lbl_p.bind("<Button-1>", self._on_persona_label_click)

        # Extended to Level 6 dynamically
        scale_d = tk.Scale(p_frame, from_=1, to=self.max_persona_level, orient=tk.HORIZONTAL, length=200, 
                                    bg=THEME["bg_color"], fg=THEME["fg_color"], relief=tk.FLAT, 
                                    command=self.update_persona_display, showvalue=False)
        self.depth_slider = scale_d
        scale_d.set(3)
        scale_d.pack(side=tk.LEFT, padx=10)

        # SECRET TRIGGER: Invisible gap right next to the slider
        lbl_sec = tk.Label(p_frame, text="      ", bg=THEME["bg_color"], cursor="arrow", width=4)
        self.secret_trigger = lbl_sec
        lbl_sec.pack(side=tk.LEFT)
        lbl_sec.bind("<Double-Button-1>", self._load_secret_model_event)

        btn_name = tk.Button(p_frame, text="", command=self.model_swap, 
                                             font=self.fonts["bold"], bg=THEME["button_bg_color"], 
                                             fg=THEME["fg_color"], relief=tk.FLAT)
        self.persona_name_button = btn_name
        btn_name.pack(side=tk.LEFT, padx=5)

        # Plus button for attachments
        btn_add = tk.Button(p_frame, text="+", command=self._show_attachment_menu,
                            font=self.fonts["bold"], bg=THEME["button_bg_color"], 
                            fg=THEME["fg_color"], relief=tk.FLAT, padx=5)
        btn_add.pack(side=tk.LEFT)
        
        # Attachments Popup Menu
        self.attachment_menu = tk.Menu(self.root, tearoff=0, bg=THEME["bg_color"], fg=THEME["fg_color"])
        self.attachment_menu.add_command(label="📷 Add Image", command=lambda: self._browse_attachment("image"))
        self.attachment_menu.add_command(label="🎵 Add Audio", command=lambda: self._browse_attachment("audio"))
        self.attachment_menu.add_command(label="📄 Add Document", command=lambda: self._browse_attachment("document"))

        btn_lore = tk.Button(p_frame, text="📜 Open Chronicles", command=self.launch_lore_book,
                                 font=("Open Sans", 10, "bold"), bg="#1a1a1a", fg=THEME["electric_blue"], 
                                 relief=tk.FLAT, padx=10)
        self.lore_btn = btn_lore
        btn_lore.pack(side=tk.LEFT, padx=15)

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
        for f in files:
            p = f.decode('utf-8') if isinstance(f, bytes) else str(f)
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
                
            if att_type == "video":
                # Mutual Exclusivity: Clear other simple attachments if a video is drog
                if self.state.get("staged_attachments"):
                    for a in list(self.state["staged_attachments"]):
                        if a["type"] != "document":
                            if "token_frame" in a and a["token_frame"].winfo_exists():
                                a["token_frame"].destroy()
                            self.state["staged_attachments"].remove(a)
                    if not any(a for a in self.state["staged_attachments"] if a["type"] != "document"):
                        if hasattr(self, "attachment_frame"): self.attachment_frame.pack_forget()

                # Fallback into the existing staging queue for video
                self.state["processing_queue"].append(p)
                if not self.state.get("staged_multimodal"):
                    self.state["staged_multimodal"] = {"type": "video", "path": p}
                if hasattr(self, 'btn_video'): self.btn_video.config(text=f"[❌] Clear ({len(self.state['processing_queue'])})", bg="#4a0000")
                self._log_and_display(f"Dropped video: {os.path.basename(p)}")
            else:
                # Mutual Exclusivity: Clear video staging if image/audio is drog
                if self.state.get("staged_multimodal") or self.state.get("processing_queue"):
                    self._reset_multimodal_ui()
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
            
            icons = {"image": "📷", "audio": "🎵", "document": "📄"}
            icon = icons.get(att_type, "📄")
            
            lbl = tk.Label(token_frame, text=f"{icon} {fname[:15]}{'...' if len(fname)>15 else ''}", bg="#2a2a2a", fg="#00ffcc", font=("Consolas", 8))
            lbl.pack(side=tk.LEFT, padx=(2,0))
            
            # Remove Button
            btn_rm = tk.Button(token_frame, text="X", bg="#4a0000", fg="#ffffff", relief=tk.FLAT, font=("Consolas", 8),
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
        if self.past_history_view is not None:
            self.past_history_view.pack_forget()
        if self.history_menu_frame is not None:
            self.history_menu_frame.pack_forget()
        
        # Bring back the pinned prompt and chat history
        if self.prompt_display is not None:
            self.prompt_display.pack(side=tk.TOP, fill="x", padx=2, pady=(2, 0))
        
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
        # Hide the pinned prompt and chat history
        if self.prompt_display is not None:
            self.prompt_display.pack_forget()
        if hasattr(self, 'timeline_frame') and self.timeline_frame.winfo_exists():
            self.timeline_frame.pack_forget()
        if self.chat_history is not None:
            self.chat_history.pack_forget()
        
        # Reset to level selection for a fresh entry
        self.history_state = {"view": "levels", "level": None}
        self._render_history_menu()
        
        if self.history_menu_frame is not None:
            self.history_menu_frame.pack(side=tk.TOP, fill="both", expand=True, padx=2, pady=2)
            
        if self.btn_history is not None:
            self.btn_history.config(bg=THEME["button_active_color"], fg=THEME["fg_color"])
        if self.btn_active is not None:
            self.btn_active.config(bg=THEME["button_bg_color"], fg="#aaaaaa")

    def _render_history_menu(self):
        """Hierarchical history menu renderer (Refactor 2026)."""
        if self.history_menu_frame is None: return
        
        # Clear frame
        for child in self.history_menu_frame.winfo_children():
            child.destroy()
            
        view = self.history_state["view"]
        
        # Navigation Bar
        nav_bar = tk.Frame(self.history_menu_frame, bg=THEME["bg_color"])
        nav_bar.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        if view != "levels":
            back_btn = tk.Button(nav_bar, text="⬅ Back", command=self._back_history, 
                                bg=THEME["button_bg_color"], fg=THEME["fg_color"], relief=tk.FLAT,
                                font=("Segoe UI", 10, "bold"), cursor="hand2")
            back_btn.pack(side=tk.LEFT, padx=5)
            
        title = "History Archive"
        if view == "models": title = f"Archive: Level {self.history_state['level']}"
        elif view == "content": 
            title = f"Archive: {self.history_state.get('current_display_name', 'Chat Log')}"
            
            # Action Frame for right-side buttons
            act_frame = tk.Frame(nav_bar, bg=THEME["bg_color"])
            act_frame.pack(side=tk.RIGHT, padx=5)
            
            # Edit Button
            edit_text = "💾 Save" if self.past_history_view.cget("state") == "normal" else "✏️ Edit"
            edit_bg = "#005a9e" if self.past_history_view.cget("state") == "normal" else THEME["button_bg_color"]
            tk.Button(act_frame, text=edit_text, command=self._toggle_history_edit, 
                      bg=edit_bg, fg="white", relief=tk.FLAT, font=("Segoe UI", 9, "bold"),
                      cursor="hand2").pack(side=tk.LEFT, padx=5)
            
            # Delete Button
            tk.Button(act_frame, text="🗑️ Delete", command=self._delete_current_archive, 
                      bg="#4a0000", fg="white", relief=tk.FLAT, font=("Segoe UI", 9, "bold"),
                      cursor="hand2").pack(side=tk.LEFT, padx=5)
        
        tk.Label(nav_bar, text=title, font=self.fonts["italic"], bg=THEME["bg_color"], fg=THEME["electric_blue"]).pack(side=tk.LEFT, padx=10)

        # Content Area
        content_frame = tk.Frame(self.history_menu_frame, bg=THEME["bg_color"])
        content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        if view == "levels":
            # Show Unlocked Levels
            levels = [1, 2, 3, 4, 5]
            if self.max_persona_level >= 6: levels.append(6)
            if self.live_agent_process and self.live_agent_process.poll() is None:
                levels.append(7)
                
            for lvl in levels:
                btn = tk.Button(content_frame, text=f"Level {lvl}", height=2,
                               command=lambda l=lvl: self._select_history_level(l),
                               bg=THEME["widget_bg_color"], fg=THEME["fg_color"], relief=tk.RAISED,
                               font=self.fonts["main"], cursor="hand2")
                btn.pack(fill=tk.X, padx=40, pady=8)
                
        elif view == "models":
            lvl = self.history_state["level"]
            history_dir = self.dirs["History"]
            try:
                files = [f for f in os.listdir(history_dir) if f.endswith(f"_lvl{lvl}.history.jsonz")]
            except: files = []
            
            if not files:
                tk.Label(content_frame, text="No history files found for this level.", 
                         bg=THEME["bg_color"], fg="#888888", font=self.fonts["italic"]).pack(pady=40)
            else:
                # Scrollable list for models if there are many
                canvas = tk.Canvas(content_frame, bg=THEME["bg_color"], highlightthickness=0)
                scrollbar = tk.Scrollbar(content_frame, orient="vertical", command=canvas.yview)
                scroll_frame = tk.Frame(canvas, bg=THEME["bg_color"])
                
                scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
                chat_frame_width = self.history_menu_frame.winfo_width()-20
                canvas.create_window((0, 0), window=scroll_frame, anchor="nw", width=chat_frame_width)
                canvas.configure(yscrollcommand=scrollbar.set)
                
                canvas.pack(side="left", fill="both", expand=True)
                scrollbar.pack(side="right", fill="y")

                for f in files:
                    display_name = f.replace(f"_lvl{lvl}.history.jsonz", "").replace("-", " ").replace("_", " ")
                    btn = tk.Button(scroll_frame, text=display_name, height=2, anchor="w", padx=15,
                                   command=lambda path=os.path.join(history_dir, f): self._load_selected_history(path),
                                   bg=THEME["widget_bg_color"], fg=THEME["fg_color"], relief=tk.RAISED,
                                   font=self.fonts["small"], cursor="hand2")
                    btn.pack(fill=tk.X, padx=10, pady=4)
                    
        elif view == "content":
            # Show the text view
            self.past_history_view.pack(in_=content_frame, side=tk.TOP, fill="both", expand=True)

    def _select_history_level(self, lvl):
        self.history_state["view"] = "models"
        self.history_state["level"] = lvl
        self._render_history_menu()

    def _back_history(self):
        if self.history_state["view"] == "content":
            self.past_history_view.pack_forget()
            self.history_state["view"] = "models"
        elif self.history_state["view"] == "models":
            self.history_state["view"] = "levels"
            self.history_state["level"] = None
        self._render_history_menu()

    def _load_selected_history(self, path):
        # Extract display name from path
        fname = os.path.basename(path)
        lvl_suffix = f"_lvl{self.history_state.get('level', '')}.history.jsonz"
        display_name = fname.replace(lvl_suffix, "").replace("-", " ").replace("_", " ")
        
        self.history_state["view"] = "content"
        self.history_state["current_path"] = path
        self.history_state["current_display_name"] = display_name
        self._render_history_menu()
        
        self.past_history_view.config(state='normal')
        self.past_history_view.delete('1.0', tk.END)
        
        try:
            import zlib, json
            with open(path, 'rb') as f: 
                msgs = json.loads(zlib.decompress(f.read()).decode('utf-8'))
            
            for m in msgs: 
                who = "You" if m['role'] == 'user' else "Serenity"
                tag = "user" if m['role'] == 'user' else "ai"
                content = self._clean_latex_artifacts(m['content'])
                entry = f"{who}: {content}\n{'-'*50}\n\n"
                self.past_history_view.insert(tk.END, entry, (tag,))
            
            self.past_history_view.yview_moveto(0.0) # Start from top for long history
        except Exception as e:
            self.past_history_view.insert(tk.END, f"Error loading history: {e}")
            
        self.past_history_view.config(state='disabled')

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
                print(f"[SYSTEM] Deleted history file: {path}")
                self._back_history()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete file: {e}")

    def _toggle_history_edit(self):
        """Toggle edit mode for the history view."""
        current_state = self.past_history_view.cget("state")
        if current_state == "disabled":
            # Switch to EDIT mode
            self.past_history_view.config(state="normal")
            self._render_history_menu() # Refresh buttons
        else:
            # Switch to VIEW mode and SAVE
            self._save_history_edits()
            self.past_history_view.config(state="disabled")
            self._render_history_menu() # Refresh buttons

    def _save_history_edits(self):
        """Parse the edited text and save it back to the compressed history file."""
        path = self.history_state.get("current_path")
        if not path or not os.path.exists(path): return
        
        raw_text = self.past_history_view.get("1.0", tk.END).strip()
        if not raw_text: return
        
        # Split by the separator used in _load_selected_history
        separator = "-" * 50
        chunks = raw_text.split(f"\n{separator}\n\n")
        
        new_msgs = []
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk: continue
            
            # Determine role
            role = "user"
            content = chunk
            if chunk.startswith("You: "):
                role = "user"
                content = chunk[5:]
            elif chunk.startswith("Serenity: "):
                role = "assistant"
                content = chunk[10:]
            elif chunk.startswith("System: "):
                role = "system"
                content = chunk[8:]
            elif ": " in chunk[:20]: # Try to guess if user changed the name
                parts = chunk.split(": ", 1)
                content = parts[1]
                # Default role mapping
                if parts[0].lower() in ["you", "user"]: role = "user"
                elif parts[0].lower() in ["serenity", "assistant", "ai"]: role = "assistant"
                elif parts[0].lower() == "system": role = "system"
                
            new_msgs.append({"role": role, "content": content.strip()})
            
        if not new_msgs: return
        
        try:
            import zlib, json
            compressed_data = zlib.compress(json.dumps(new_msgs).encode('utf-8'))
            with open(path, 'wb') as f:
                f.write(compressed_data)
            print(f"[SYSTEM] Saved edits to history: {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save edits: {e}")

    def clear_chat_ui(self):
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
                    self.chat_history.config(state='disabled')
                    self.state["response_started"] = False
            except tk.TclError: pass
            
        if hasattr(self, 'history_menu_frame') and self.history_menu_frame:
            self.history_menu_frame.pack_forget()

    def _setup_logs_and_stats(self):
        if self.right_panel is None: return
        self.log_container = tk.Frame(self.right_panel, bg=THEME["bg_color"])
        self.log_container.grid_rowconfigure(1, weight=1); self.log_container.grid_columnconfigure(0, weight=1)
        
        header = tk.Frame(self.log_container, bg=THEME["bg_color"])
        header.grid(row=0, column=0, sticky="ew")
        tk.Label(header, text="Backend Logs", font=self.fonts["italic"], bg=THEME["bg_color"], fg=THEME["electric_blue"]).pack(side=tk.LEFT)
        
        self.log_switch_canvas = tk.Canvas(header, width=104, height=28, bg=THEME["bg_color"], highlightthickness=0)
        self.log_switch_canvas.pack(side=tk.RIGHT, padx=5)
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

        self.thought_log = scrolledtext.ScrolledText(self.log_frame, font=("Consolas", 10), bg=THEME["widget_bg_color"], fg="#cccccc", relief=tk.FLAT)
        self.thought_log.grid(row=0, column=0, sticky="nsew")
        self.thought_log.tag_config("stdout", foreground="#cccccc")
        self.thought_log.tag_config("system", foreground=THEME["electric_blue"], font=("Consolas", 10, "bold"))
        
        self.error_log = scrolledtext.ScrolledText(self.log_frame, font=("Consolas", 10), bg=THEME["widget_bg_color"], fg="#ff8a8a", relief=tk.FLAT)
        self.error_log.grid(row=0, column=0, sticky="nsew")
        self.error_log.tag_config("stderr", foreground="#ff8a8a")
        self.error_log.grid_remove()

        self.tool_log = scrolledtext.ScrolledText(self.log_frame, font=("Consolas", 10), bg=THEME["widget_bg_color"], fg="#00ffcc", relief=tk.FLAT)
        self.tool_log.grid(row=0, column=0, sticky="nsew")
        self.tool_log.grid_remove()
        
        self.diag_log = scrolledtext.ScrolledText(self.log_frame, font=("Consolas", 10), bg=THEME["widget_bg_color"], fg="#ffa500", relief=tk.FLAT)
        self.diag_log.grid(row=0, column=0, sticky="nsew")
        self.diag_log.tag_config("diag", foreground="#ffa500")
        self.diag_log.grid_remove()
        
        self.stats_frame = tk.Frame(self.log_container, bg=THEME["widget_bg_color"])
        if self.stats_frame is not None:
            self.stats_frame.grid(row=2, column=0, sticky="nsew", pady=5)
        self.stats_labels = {}
        
        # MISSION: Reorganized stats into 2 columns (Left: GPU-centric, Right: System-centric)
        # Grid Layout: Left Column (GPU/VRAM) | Right Column (System/CPU)
        # Order ensures CPU Use is stacked directly above Total RAM in the system column.
        stats_to_show = [
            ("GPU Use", "GPU Use"), ("CPU", "CPU Use"),
            ("VRAM", "VRAM"), ("RAM", "Total RAM"),
            ("Shared VRAM", "Shared VRAM"), ("Disk", "Disk Use"),
            ("GPU Temp", "GPU Temp"), ("Power", "Power")
        ] if SYSTEM_MONITOR_LOADED else [("CPU", "CPU"), ("RAM", "RAM")]
        
        for i, (key, label) in enumerate(stats_to_show):
            row = i // 2
            col = i % 2
            f = tk.Frame(self.stats_frame, bg=THEME["widget_bg_color"])
            f.grid(row=row, column=col, sticky="ew", padx=10, pady=2)
            self.stats_frame.columnconfigure(col, weight=1)
            
            tk.Label(f, text=f"{label}:", bg=THEME["widget_bg_color"], fg=THEME["fg_color"], font=("Consolas", 9)).pack(side=tk.LEFT)
            self.stats_labels[key] = tk.Label(f, text="N/A", bg=THEME["widget_bg_color"], fg=THEME["electric_blue"], font=("Consolas", 9, "bold"))
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
                        tier_map = {1: "fast", 2: "search", 3: "low", 4: "med", 5: "high", 6: "secret", 7: "Live"}
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
        
        # Dual-VLM routing (Video only)
        target_tier = "vision_video_deep"
            
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

    def _live_multimodal_worker(self, paths, user_msg):
        """Handoff worker for T5-Gemma multimodal engine (Serenity Live)."""
        import base64, requests, io
        try:
            filename = os.path.basename(paths[0])
            self.process_queue.put({"status": "thinking_status", "content": f"Live Handoff: Preparing {filename}..."})
            
            # Start Live Agent if not running
            if not self.live_agent_process or self.live_agent_process.poll() is not None:
                 self.process_queue.put({"status": "log_update", "content": "[SYSTEM] Waking Serenity Live for handoff...\n"})
                 self.toggle_live_agent()
                 time.sleep(3.0) # Wait for startup
            
            # Encode first media file (T5-Gemma is usually single-image per turn)
            # Future: handle multiple interleaving if API supports it
            img_b64 = None
            ext = os.path.splitext(paths[0])[1].lower()
            if ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']:
                with open(paths[0], "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode('utf-8')
            
            # Endpoint: T5 Engine (Default: 8001)
            url = "http://127.0.0.1:8001/analyze"
            payload = {
                "text": user_msg,
                "image_b64": img_b64,
                "max_tokens": 1024,
                "temperature": 0.4
            }
            headers = {"x-api-key": "serenity-alpha-core-77X"}
            
            self.process_queue.put({"status": "thinking_status", "content": "Live Handoff: Waiting for T5 response..."})
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            
            if response.status_code == 200:
                data = response.json()
                # T5-Gemma models usually respond with a JSON-wrapped thought/speech structure in this project
                # But sometimes it's raw text depending on the server branch.
                if isinstance(data, dict):
                    full_resp = data.get("speech", data.get("response", str(data)))
                else:
                    full_resp = str(data)
                
                self.process_queue.put({"status": "success", "content": f"**[Live Analysis: {filename}]**\n{full_resp}"})
            else:
                self.process_queue.put({"status": "error", "content": f"Live Engine Error: {response.status_code} - {response.text}"})
        except Exception as e:
            print(f"[APEX] Live Handoff Error: {e}\n{traceback.format_exc()}")
            self.process_queue.put({"status": "error", "content": f"Failed to handoff to Live Engine: {e}"})
        finally:
            self.state["running"] = False
            self.set_ui_state(model_loaded=True, generating=False)

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
        
        threading.Thread(target=self._batch_vision_worker, args=("video", queue, user_msg, True), daemon=True).start()
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

    def model_swap(self, value=None, target_level=None, target_tier=None):
        self.halt_process()
        
        # If we are swapping to a text-based tier, clear vision intentions
        if target_tier not in ["vision_video", "vision_video_deep"] and (target_level is not None or value is not None):
             if self.state.get("staged_multimodal"):
                  self._reset_multimodal_ui()

        raw_val = int(value) if value else (target_level if target_level else self.depth_slider.get() if self.depth_slider else 3)
        
        # --- Level Mapping Logic ---
        is_secret = (self.max_persona_level >= 6)
        is_live = (self.live_agent_process and self.live_agent_process.poll() is None)
        if is_live and not is_secret and raw_val == 6:
            level = 7
        else:
            level = raw_val

        if not target_tier or (level == 6 and target_tier == "deep_cook"):
            tier_map = {1: "fast", 2: "search", 3: "low", 4: "med", 5: "high", 6: "secret", 7: "Live"}
            target_tier = tier_map.get(level, "low")
            
        # --- APEX T5 ENGINE CO-EXISTENCE GUARD ---
        # If loading any level 1-7 GGUF model, stop the T5 engine subprocess and port to prevent VRAM dual load!
        if getattr(self, "live_agent_process", None) is not None:
            try:
                self.live_agent_process.terminate()
                self.live_agent_process = None
            except: pass
        
        # Hard kill anything on Port 8001 to make sure VRAM is fully evacuated
        try:
            import psutil
            for conn in psutil.net_connections():
                laddr = getattr(conn, 'laddr', None)
                if laddr and hasattr(laddr, 'port') and laddr.port == 8001:
                    pid = getattr(conn, 'pid', None)
                    if pid and pid != os.getpid():
                        print(f"[APEX] Halting competing T5 process (PID {pid})...")
                        p = psutil.Process(pid)
                        p.terminate()
                        p.wait(timeout=1.0)
        except: pass

        
        # --- Architect Tier (GGUF) Dynamic Resolution ---
        if target_tier == "Live":
            params_file = os.path.join(getattr(self, 'live_dir', os.path.join(self.script_dir, "Live")), "System", "params.json")
            if os.path.exists(params_file):
                try:
                    with open(params_file, "r") as f:
                        params = json.load(f)
                        arch_model = params.get("architect_model", "Gemma-4 26B (FP4)")
                        mapping = {
                            "Gemma-4 26B (FP4)": r"Models\gemma-4-26B MoE\gemma-4-26B-A4B-it-MXFP4_MOE.gguf",
                            "Gemma-4 26B (Q5)": r"Models\gemma-4-26B MoE\gemma-4-26B-A4B-it-UD-Q5_K_XL.gguf",
                            "Qwen 3.6 (27B)": r"S:\LLM\Qwen3.6-27B\Qwen3.6-27B-UD-Q4_K_XL.gguf",
                            "Qwen 3.6 (35B)": r"S:\LLM\Qwen3.6-35B\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
                        }
                        rel_path = mapping.get(arch_model)
                        if rel_path:
                            # Use absolute if it has a drive letter, otherwise join with BASE_DIR
                            if ":" in rel_path: self.model_paths["Live"] = rel_path
                            else: self.model_paths["Live"] = os.path.join(self.BASE_DIR, rel_path)
                except: pass
            

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
        # If toggling between layers of the exact same model file, keep KV/Logic loaded
        soft_clear = False
        if self.model and getattr(self, "model_path", None) == path:
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

    def send_message(self, msg_override=None, skip_swap_check=False):
        if self.state["running"]: return
        
        user_msg = msg_override if msg_override else self.user_input.get("1.0", tk.END).strip()
        
        # 1. Text Documents Injection
        staged_atts = self.state.get("staged_attachments", [])
        doc_atts = [a for a in staged_atts if a["type"] == "document"]
        if doc_atts:
            doc_text = ""
            for doc in doc_atts:
                try:
                    with open(doc["path"], "r", encoding="utf-8", errors="ignore") as f:
                        doc_text += f"\n\n--- Document: {doc['name']} ---\n{f.read()[:50000]}\n" # Safety cap
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
        # Priority: If simple attachments (images/audio) are present, use the multimodal engine
        has_media = any(a["type"] in ["image", "audio"] for a in self.state.get("staged_attachments", []))
        
        if has_media or self.state.get("staged_multimodal"):
            if not user_msg: user_msg = "Analyze this media."
            
            # ROUTING: Internal (llama.cpp) vs Live (T5 Agent)
            use_live = self.state.get("multimodal_engine") == "Live"
            
            if has_media:
                media_pts = [a["path"] for a in self.state.get("staged_attachments", []) if a["type"] in ["image", "audio"]]
                media_names = [a["name"] for a in self.state.get("staged_attachments", []) if a["type"] in ["image", "audio"]]
                
                # Check for inline vision capability
                has_inline_vision = (
                    not use_live and 
                    self.model is not None and 
                    getattr(self.model, "chat_handler", None) is not None
                )
                
                if not msg_override:
                    self.user_input.delete("1.0", tk.END)
                    self.last_user_message = user_msg
                    self._display_user_message(f"[{', '.join(media_names)}] {user_msg}")
                
                # Clear attachments now they are bound to a task
                for a in list(self.state.get("staged_attachments", [])):
                    if a["type"] != "document":
                        if "token_frame" in a and a["token_frame"].winfo_exists():
                            a["token_frame"].destroy()
                        self.state["staged_attachments"].remove(a)
                if not any(a for a in self.state["staged_attachments"] if a["type"] != "document"):
                   if hasattr(self, "attachment_frame"): self.attachment_frame.pack_forget()

                if has_inline_vision:
                    self._log_and_display("Handling image inline via loaded persona model...")
                    self._display_ai_message(is_streaming=True)
                    self.set_avatar_state("pondering" if self.active_persona_level >= 4 else "thinking")
                    self._prep_generation()
                    
                    # Construct multimodal content
                    content_list = [{"type": "text", "text": user_msg}]
                    for path in media_pts:
                        ext = os.path.splitext(path)[1].lower()
                        if ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']:
                            budget = VisionHandler._determine_visual_budget(user_msg)
                            b64 = VisionHandler.encode_image(path, budget=budget)
                            if b64:
                                content_list.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                                })
                        elif ext in ['.mp3', '.wav', '.ogg', '.flac', '.m4a']:
                            chunks = VisionHandler.get_audio_chunks(path, chunk_length_s=30, max_chunks=30)
                            for chunk in chunks:
                                content_list.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:audio/wav;base64,{chunk}"}
                                })
                                
                    temp_msgs = self.messages + [{"role": "user", "content": content_list}]
                    threading.Thread(target=self._generation_worker, args=(user_msg, temp_msgs), daemon=True).start()
                    self.root.after(100, self.check_process_queue)
                    return

                if use_live:
                    self._log_and_display("Routing to Serenity Live Engine (T5)...")
                    threading.Thread(target=self._live_multimodal_worker, args=(media_pts, user_msg), daemon=True).start()
                else:
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
                    
                if use_live and staged["type"] != "video": # Live handles images/audio; video usually stays Internal for sub-chunking
                    self._log_and_display("Routing to Serenity Live Engine (T5)...")
                    threading.Thread(target=self._live_multimodal_worker, args=([staged["path"]], user_msg), daemon=True).start()
                else:
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
        tier_map = {1: "fast", 2: "search", 3: "low", 4: "med", 5: "high", 6: "secret", 7: "Live"}
        req_tier = tier_map.get(level, "low")
        
        if not skip_swap_check and self.current_model_tier != req_tier:
            self._log_and_display(f"Switching to {req_tier}...")
            self.pending_task = {"type": "chat", "message": user_msg}
            self.model_swap(target_level=level, target_tier=req_tier)
            return

        self._display_ai_message(is_streaming=True)
        self.set_avatar_state("pondering" if self.active_persona_level >= 4 else "thinking")
        
        self._prep_generation()
        temp_msgs = self.messages + [{"role": "user", "content": user_msg}]
        threading.Thread(target=self._generation_worker, args=(user_msg, temp_msgs), daemon=True).start()
        self.root.after(100, self.check_process_queue)

    def send_deep_cook_message(self, msg_override=None, skip_swap_check=False):
        if self.state["running"]: return
        
        user_msg = msg_override if msg_override else self.user_input.get("1.0", tk.END).strip()
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
        self.set_avatar_state("deep_think")
        
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
        
        # 1. Try to read the block count
        total_layers = 0
        
        # Method A: Try llama_cpp LlamaGGUFReader
        try:
            from llama_cpp.llama_speculative import LlamaGGUFReader
            reader = LlamaGGUFReader(model_path)
            for field in reader.fields:
                if field.name.endswith(".block_count"):
                    total_layers = int(field.parts[0][0])
                    break
        except Exception as e:
            print(f"[DYNAMIC AUTO-OFFLOAD] LlamaGGUFReader failed: {e}. Falling back to binary parser.")
            
        # Method B: Fallback to binary parser (robust for any Python environment/LlamaCpp-python version)
        if total_layers == 0:
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
                                break
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
        
        # 3. Dynamic KV Cache Footprint Estimation (SWA / Turbo3 Adjusted)
        # Scales linearly based on context window target
        if ctx_size <= 49152:
            kv_cache_vram_mb = 3150.0  # Safe ceiling based on explicit ISWA baseline
        else:
            kv_cache_vram_mb = (ctx_size / 49152) * 3150.0

        # 4. Math: Allocate remaining VRAM budget to layers
        available_weight_vram = targeted_reserve_vram_mb - kv_cache_vram_mb
        
        if available_weight_vram <= 0:
            print(f"[WARN] Cache footprint ({kv_cache_vram_mb:.1f}MB) saturates VRAM. Offloading 0 layers.")
            return 0
            
        safe_layers = int(available_weight_vram // vram_per_layer)
        final_layers = max(0, min(total_layers, safe_layers))
        
        print("--- DYNAMIC VRAM REPORT ---")
        print(f"Model Detected:   {os.path.basename(model_path)}")
        print(f"Total Layers:     {total_layers}")
        print(f"File/Weight Size: {model_base_vram_mb:.1f} MiB (~{vram_per_layer:.1f} MiB/layer)")
        print(f"Est. KV Cache:    {kv_cache_vram_mb:.1f} MiB")
        print(f"Action:           Offloading {final_layers}/{total_layers} layers to GPU")
        print("----------------------------")
        
        return final_layers

    # ================= WORKERS =================
    def _auto_tune_params(self, level: int, tier: str) -> Dict[str, Any]:
        params_specs: Dict[str, Any] = {"n_ctx": 4096, "n_gpu_layers": 99, "extra_args": {}}
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

            # Map specific extra args based on tier/level
            if level in [1, 2] or tier in ["fast", "search"]:
                # You can experiment with TurboQuant here if your llama.cpp supports it 
                # (e.g., changing "q4_0" to "tq1_0" or "tq2_0" for extreme 2-3 bit KV compression)
                params_specs["extra_args"]["cache_type_k"] = "q8_0"
                params_specs["extra_args"]["cache_type_v"] = "q4_0"
            elif tier == "med" or level == 4:
                params_specs["extra_args"]["cache_type_k"] = "q8_0"
                params_specs["extra_args"]["cache_type_v"] = "q4_0"
            elif tier == "high" or level == 5:
                params_specs["extra_args"]["flash_attn"] = True
            elif tier == "secret" or level == 6:
                params_specs["extra_args"]["flash_attn"] = True
            elif level == 7:
                # Strictly decouple Live Agent settings
                params_specs["extra_args"]["flash_attn"] = True
                
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
                    vram_target = 5400  # Default fallback 5.4 GB
                
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
                        if f.lower().endswith(".mmproj"):
                            adjacent_proj = os.path.join(model_dir, f)
                            break

            if target_tier.startswith("vision_"):
                proj_path = self.model_paths.get(f"{target_tier}_projector")
                if not proj_path or not os.path.exists(proj_path):
                    proj_path = adjacent_proj
                
                if proj_path and os.path.exists(proj_path):
                    try:
                        from llama_cpp.llama_chat_format import Llava15ChatHandler
                        chat_handler = Llava15ChatHandler(clip_model_path=proj_path, verbose=True)
                        print(f"Vision Projector Loaded: {os.path.basename(proj_path)}")
                    except Exception as e:
                        print(f"Warning: Failed to load vision projector '{proj_path}': {e}")
            else:
                if adjacent_proj and os.path.exists(adjacent_proj):
                    try:
                        from llama_cpp.llama_chat_format import Llava15ChatHandler
                        chat_handler = Llava15ChatHandler(clip_model_path=adjacent_proj, verbose=True)
                        print(f"Inline Vision Projector Loaded: {os.path.basename(adjacent_proj)}")
                    except Exception as e:
                        print(f"Warning: Failed to load inline vision projector '{adjacent_proj}': {e}")
            
            # --- APEX VRAM SCOUT (1024MB Rule) ---
            if SYSTEM_MONITOR_LOADED and self.gpu_handle and not self.config.get("auto_vram_offload", False):
                try:
                    mem = nvidia_ml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    total_mb = mem.total / (1024**2)
                    free_mb = mem.free / (1024**2)
                    
                    # Reserve exactly 1536MB (1.5GB) for mmproj/NVDEC
                    vram_buffer = 1536 
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
            
            # Resolve dynamic KV cache types from specs
            cache_map = {
                "f32": 0, "f16": 0, "q8_0": 1, "q4_0": 2, "q4_1": 3, "q5_0": 6,
                "tq2": 42, "turbo2": 42,
                "tq3": 43, "turbo3": 43,
                "tq4": 44, "turbo4": 44
            }
            global_kv = self.config.get("global_kv_cache", "Auto")
            
            if global_kv != "Auto":
                gl = global_kv.lower()
                if gl in cache_map:
                    t_k = t_v = cache_map[gl]
                elif "tq" in gl or "turbo" in gl:
                    if "2" in gl:
                        t_k = t_v = 42 # GGML_TYPE_TURBO2_0
                    elif "3" in gl:
                        t_k = t_v = 43 # GGML_TYPE_TURBO3_0
                    elif "4" in gl:
                        t_k = t_v = 44 # GGML_TYPE_TURBO4_0
                    else:
                        t_k = t_v = 43 # Default to Turbo3
                else:
                    t_k = t_v = cache_map.get(gl, 2)
                params.pop("cache_type_k", None)
                params.pop("cache_type_v", None)
            else:
                t_k = cache_map.get(params.pop("cache_type_k", "q4_0").lower(), 2)
                t_v = cache_map.get(params.pop("cache_type_v", "q4_0").lower(), 2)

            
            hao_preset = self.config.get("hao_preset", "exps=CPU")
            override_tensors = [hao_preset] if hao_preset != "None" else []
            
            swa_kv = self.config.get("swa_kv_cache", "Auto")
            no_kv_offload = True if swa_kv == "CPU Only" else False

            # --- Hardware Priority Lock ---
            HardwareProfile.pin_to_p_cores()
            HardwareProfile.set_priority("above_normal") # ABOVE_NORMAL as per mission

            # --- Wit-Layer: Init ---
            self.process_queue.put({"status": "thinking_status", "content": "Waking up the experts... (SATA speeds, hang tight)"})

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
            if self.config.get("speculative_drafting", True) and not target_tier.startswith("vision_") and not chat_handler:
                # 1. Search for assistant model in same directory
                assistant_path = None
                if self.model_path:
                    model_dir = os.path.dirname(self.model_path)
                    if os.path.exists(model_dir):
                        for f in os.listdir(model_dir):
                            if f.lower().endswith(".gguf") and "assistant" in f.lower():
                                assistant_path = os.path.join(model_dir, f)
                                break
                
                if assistant_path and os.path.exists(assistant_path):
                    try:
                        from System.gguf_draft_model import GgufDraftModel
                        draft_ngl = max(0, n_layers // 3)
                        draft_model = GgufDraftModel(assistant_path, n_gpu_layers=draft_ngl, n_ctx=n_ctx)
                        print(f"[ENGINE] Speculative GGUF assistant model detected and loaded: {os.path.basename(assistant_path)} on {draft_ngl} GPU layers")
                    except Exception as spec_err:
                        print(f"[ENGINE] Failed to initialize speculative GGUF draft model: {spec_err}")
                
                # 2. Fallback to prompt lookup decoding if no assistant model GGUF is found
                if draft_model is None:
                    try:
                        from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
                        pred_tokens = 8 if n_layers > 0 else 2
                        draft_model = LlamaPromptLookupDecoding(num_pred_tokens=pred_tokens)
                        print(f"[ENGINE] Speculative drafting enabled: LlamaPromptLookupDecoding(num_pred_tokens={pred_tokens})")
                    except Exception as spec_err:
                        print(f"[ENGINE] Failed to initialize prompt lookup speculative draft model: {spec_err}")

            try:
                model = Llama(
                    model_path=self.model_path, 
                    n_gpu_layers=n_layers,       # Dynamic HAO
                    n_ctx=n_ctx,                 # Dynamic Context Window
                    n_threads=8,                 # Strictly pin to 8 P-Cores
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
                    logits_all=True if chat_handler else False, # FIXED: Grants projector threads visibility for custom kernels
                    tensor_split=None,
                    rpc_servers=None,
                    override_tensors=override_tensors,
                    draft_model=draft_model,      # Speculative MTP Drafting
                    **params
                )
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
            try:
                print(f"[BENCHMARK] Running GGUF KV Cache Benchmark...")
                start_b = time.time()
                prompt = "Hello, how are you today?"
                # Run completion of up to 100 tokens
                res = model(
                    prompt,
                    max_tokens=100,
                    temperature=0.3,
                    top_p=0.95
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

            if self.stop_process.is_set(): return
            self.process_queue.put({"status": "load_success", "model": model, "level": target_level, "tier": target_tier})

        except Exception as e:
            if not self.stop_process.is_set(): self.state["last_crash"] = True; self.process_queue.put({"status": "load_error", "content": str(e)})

    def _generation_worker(self, user_message, temp_messages):
        """Standard chat inference with Gemma-4 hardening."""
        try:
            HardwareProfile.pin_to_p_cores()
            HardwareProfile.set_priority("above_normal")
            
            # Wit-Layer: Thinking Message
            self.process_queue.put({"status": "thinking_status", "content": "Hold up, lemme cook..."})
            print(f"[INFERENCE] Starting generation for user message ({len(user_message)} chars).")
            
            sys_content = PERSONA_PROMPTS.get(self.active_persona_level, "You are Serenity.")
            is_gemma = "gemma" in self.model_path.lower() and getattr(self.model, "chat_handler", None) is None
            
            # Setup Inference Params
            params = self._get_inference_params(temp_messages)
            if is_gemma:
                # Structural Safety: Ensure the turn closer is always present, but don't force legacy tokens
                if "<end_of_turn>" not in params.get("stop", []):
                    params.setdefault("stop", []).append("<end_of_turn>")
                if "<turn|>" not in params.get("stop", []):
                    params.setdefault("stop", []).append("<turn|>")
                            
                sys_clean = sys_content.strip()
                def official_q(s): return f"<|\"|>{s}<|\"|>"

                # Gemma-4: Tool definitions injection (Template-Aligned)
                tool_defs = ""
                tools = self.tool_registry.get_definitions(self.active_persona_level)
                for t in tools:
                    f = t["function"]
                    # Official template format: declaration:name{description:<|"|>...<|"|>,parameters:{properties:{...},required:[...]}}
                    # We'll use a semi-manual construction to ensure token alignment
                    t_params = f.get("parameters", {})
                    t_props = t_params.get("properties", {})
                    t_prop_str = ",".join([f"{k}:{{description:{official_q(v.get('description',''))},type:{official_q(v.get('type','STRING'))}}}" for k, v in t_props.items()])
                    t_req_str = ",".join([official_q(r) for r in t_params.get("required", [])])
                    
                    tool_defs += f"<|tool>declaration:{f['name']}{{description:{official_q(f['description'])},parameters:{{properties:{{{t_prop_str}}},required:[{t_req_str}]}}}}<tool|>\n"
                
                if tools:
                    tool_defs += "\n" # Gemma-4 alignment safety
                    sys_clean += (
                        f"\n[TOOL CALLING RULE]: If the user's request requires information you do not have (e.g. weather, news, file content), you MUST immediately output a tool call tag. "
                        f"DO NOT provide a final answer, summarize your plan, or explain why you are searching in the final response. EXECUTE it using this format: <|tool_call>call:tool_name{{arg1:{official_q('value1')}}}<tool_call|>"
                    )
                    sys_clean += "\n[PERMISSION]: You are an autonomous agent. You have PERMISSION to execute any tool call immediately when needed. Do NOT ask for permission or wait for further instructions before calling a tool."
                
                # Level 3+ or Deep Cook need the thought channel constraint
                if self.active_persona_level >= 3 or self.state.get("deep_cook"):
                    if is_gemma:
                        sys_clean += (
                            "\n[CRITICAL RESTRICTION]: You MUST begin your response by opening the thought channel. Complete ALL internal analysis, planning, and tone-checks inside that channel before closing it."
                        )
                    else:
                        sys_clean += (
                            "\n[CRITICAL RESTRICTION]: You MUST begin your response inside a '<think>' block. Complete ALL internal analysis, planning, and tone-checks inside that block. "
                            "If you need to call a tool, you MUST do so INSIDE the <think> block before closing it. "       
                            "When done thinking or if otherwise ready to respond, DO NOT HESITATE to output the </think> tag and begin final response. "
                        )                
                elif self.active_persona_level == 2:
                    sys_clean += "\n[SEARCH PROTOCOL]: If you need information, output a tool call IMMEDIATELY. Do not explain your reasoning unless the search fails."
                            
                is_gemma4 = "gemma-4" in self.model_path.lower() or "gemma_4" in self.model_path.lower() or "gemma 4" in self.model_path.lower()
                
                # Gemma-4: Enable logic mode via <|think|> at start of system prompt for high levels
                thinking_tag = "<|think|>\n" if (self.active_persona_level >= 3 or self.state.get("deep_cook")) else ""
                # Use official template structure: <|turn>system\n<|think|>\n...<turn|>\n for Gemma-4
                if is_gemma4:
                    prompt_str = f"<|turn>system\n{thinking_tag}{sys_clean}\n{tool_defs}<turn|>\n"
                else:
                    prompt_str = f"<start_of_turn>system\n{thinking_tag}{sys_clean}\n{tool_defs}<end_of_turn>\n"

                            
                # TriAttention KV Pruning
                if self.kv_manager and TRI_ATTENTION_ENABLED:
                     processed_msgs = self.kv_manager.enforce_kv_budget(temp_messages)
                else:
                     processed_msgs = temp_messages[-12:]
                            
                for m in processed_msgs:
                    role = "model" if m["role"] == "assistant" else m["role"]
                    content = m["content"]
                    if isinstance(content, str):
                        content = content.strip()
                    
                    # HF Template alignment: Strip previous thoughts from the context window (Gemma-4 Official Rule)
                    if role == "model" and isinstance(content, str):
                        content = re.sub(r'(?s)<think>.*?(?:<\/think>|$)', '', content, flags=re.IGNORECASE)
                        content = re.sub(r'(?s)<\|channel>thought.*?(?:<channel\|>|$)', '', content, flags=re.IGNORECASE)
                        content = re.sub(r'<\|think\|>.*?(?:<\/\|think\|>|$)', '', content, flags=re.IGNORECASE | re.DOTALL)
                        content = re.sub(r'<thought(?:>|\b).*?(?:<\/thought>|$)', '', content, flags=re.IGNORECASE | re.DOTALL)
                        content = content.strip()
                        
                    if is_gemma4:
                        prompt_str += f"<|turn>{role}\n{content}<turn|>\n"
                    else:
                        prompt_str += f"<start_of_turn>{role}\n{content}<end_of_turn>\n"
                            
                # Level 2 (Search) needs to be able to fire tools immediately. 
                # Gemma-4 natively starts the thought channel if <|think|> is in the system prompt.
                if is_gemma4:
                    prompt_str += "<|turn>model\n"
                else:
                    prompt_str += "<start_of_turn>model\n"
                if (self.active_persona_level >= 3 or self.state.get("deep_cook")) and not is_gemma:
                    prompt_str += "<think>\n"

                            
                status_text = "Analyzing logical momentum..." if self.active_persona_level >= 3 else "Direct Strike: Pre-computing..."
                
                # Wit-Layer: Batching Message
                self.process_queue.put({"status": "thinking_status", "content": status_text})
                
                # APEX GIL-SAFETY: We use internal streaming even for "non-streaming" responses.
                # This ensures the background thread yields to the UI thread between tokens.
                full_resp = ""
                print(f"[INFERENCE] Prefill phase starting. Prompt length: {len(prompt_str)} chars.")
                gen_iterator = self.model(prompt_str, stream=True, echo=False, **params)
                
                for chunk in gen_iterator:
                    if self.stop_process.is_set():
                        break
                    token_text = chunk["choices"][0]["text"]
                    full_resp += token_text
                    
                    # Heartbeat for UI & Real-time streaming delivery
                    self.process_queue.put({"status": "streaming", "content": token_text})
                    time.sleep(0.001) 
                    
                    # Loop Mitigation: Check for infinite repetition loops
                    if len(full_resp) > 200 and self._detect_repetition(full_resp):
                        self.process_queue.put({"status": "diag_log_update", "content": "[RUNTIME] Repetition loop detected! Breaking inference stream to preserve sanity."})
                        print("[RUNTIME] Repetition loop detected! Breaking inference stream.")
                        break 
                
                # Re-format as a response object for usage/diag extraction below
                response = {"choices": [{"text": full_resp, "finish_reason": "stop"}], "usage": {}}
                
                # --- DIAGNOSTICS TELEMETRY ---
                diag_msg = f"--- GENERATION DIAGNOSTICS ---\n"
                diag_msg += f"Finish Reason: {response['choices'][0].get('finish_reason', 'N/A')}\n"
                usage = response.get('usage', {})
                diag_msg += f"Tokens Used: {usage.get('completion_tokens', 'N/A')} (Total: {usage.get('total_tokens', 'N/A')})\n"
                diag_msg += f"Raw Output Length: {len(full_resp)} characters\n"
                
                # Extract and list all structural tags found
                tags = re.findall(r'<[^>]+>', full_resp)
                if tags:
                    diag_msg += f"Generated Tags: {', '.join(tags)}\n"
                    
                diag_msg += "------------------------------"
                self.process_queue.put({"status": "diag_log_update", "content": diag_msg})
                
                # --- GEMMA-4 HARMONIZATION PATCH (Background) ---
                # 1. Telemetry Extraction
                match_deep = re.search(r'\[DEEPLOG:(.*?)\]', full_resp, re.IGNORECASE | re.DOTALL)
                if match_deep:
                    extracted_log = match_deep.group(1).strip()
                    full_resp = re.sub(r'\[DEEPLOG:.*?\]', '', full_resp, flags=re.IGNORECASE | re.DOTALL).strip()
                    log_path = os.path.join(self.dirs["Logs"], "subconscious_journal.txt")
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {extracted_log}\n")

                match_prime = re.search(r'\[PRIME_MEMORY:(.*?)\]', full_resp, re.IGNORECASE | re.DOTALL)
                if match_prime:
                    extracted_prime = match_prime.group(1).strip()
                    full_resp = re.sub(r'\[PRIME_MEMORY:.*?\]', '', full_resp, flags=re.IGNORECASE | re.DOTALL).strip()
                    prime_path = os.path.join(self.dirs["System"], ".prime_chronicles.txt")
                    with open(prime_path, "a", encoding="utf-8") as f:
                        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {extracted_prime}\n")

                # 2. Advanced Scout & Split (Gemma-4 Optimized)
                closers = [
                    r'<\/think>', r'<\|channel>text', r'<\|channel>assistant', r'<channel\|>', r'<\/channel\|>', r'\[\/DRAFT\]',
                    r'(?i)\n(?:Final Output|Final Polish|Grandmaster Verdict|Final Answer|Execution complete)[\s:]+'
                ]

                all_splits = []
                for tag_pattern in closers:
                    # MISSION: Search the entire response to find ALL potential boundaries
                    # (Removed the 15k limit to handle ultra-long Deep Cook reasoning)
                    for m in re.finditer(tag_pattern, full_resp, re.IGNORECASE):
                        all_splits.append(m.end())
                
                # Structural Fallback: If no explicit closer, check if model started with thought but stopped at turn closer
                if not all_splits and ("<think>" in full_resp or "<|channel>thought" in full_resp):
                    # Check for implicit end-of-thought at the very end of the string
                    all_splits.append(len(full_resp))

                all_splits.sort()
                
                best_split = -1
                if all_splits:
                    # MISSION: Find the most logical split point (Smart Splitter)
                    # We want the split that leaves a non-empty final answer and is not followed by more thoughts.
                    for split in all_splits:
                        remaining = full_resp[split:].strip()
                        # If there's another thought block starting after this closer, this isn't the final boundary.
                        if re.search(r'<think>|<thought>|\[DRAFT\]|<\|channel>thought|<channel\s*\|?>', remaining, re.IGNORECASE):
                            continue
                        
                        # We found a split that leaves a response, and no more thoughts follow.
                        best_split = split
                        break
                    
                    # Fallback to the absolute last split if no "clean" split was identified
                    if best_split == -1:
                        best_split = all_splits[-1]
                
                if best_split != -1:
                    think_log = full_resp[:best_split].strip()
                    final_answer = full_resp[best_split:].strip()
                else:
                    # --- APEX FIX: EMERGENCY SPLIT ---
                    was_thinking_expected = (self.active_persona_level >= 3 or self.state.get("deep_cook"))
                    
                    if was_thinking_expected:
                        headers = r'(?i)\n(?:Final Output|Final Polish|Grandmaster Verdict|Final Response|Final Verdict|Final Answer)[\s:]+'
                        parts = re.split(headers, full_resp)
                        if len(parts) > 1:
                            think_log = parts[0].strip()
                            final_answer = parts[-1].strip()
                        else:
                            # Attempt heuristic extraction
                            think_log = self._extract_thinking_content(full_resp) 
                            if think_log and len(think_log) < len(full_resp) * 0.9:
                                # MISSION: Safely isolate the final answer even if multiple blocks were found
                                # Find the end of the very last identified thought block in the raw response
                                last_thought_end = 0
                                # Search for ALL common block patterns, including open tags (ending at $)
                                block_patterns = r'(?s)<think>.*?(?:<\/think>|$)|<thought>.*?(?:<\/thought>|$)|\[DRAFT\].*?(?:\[\/DRAFT\]|$)|<\|channel>thought.*?(?:<\/\|?channel\|?>|$)'
                                for m in re.finditer(block_patterns, full_resp, re.IGNORECASE):
                                    last_thought_end = max(last_thought_end, m.end())
                                
                                if last_thought_end > 0:
                                    think_log = full_resp[:last_thought_end].strip()
                                    final_answer = full_resp[last_thought_end:].strip()
                                else:
                                    # Fallback to standard replace
                                    final_answer = full_resp.replace(think_log, "", 1).strip()
                            else:
                                # If extraction is ambiguous, treat the whole thing as thinking 
                                # and trigger synthesis to get a clean answer.
                                think_log = full_resp
                                final_answer = ""
                    else:
                        think_log = ""
                        final_answer = full_resp

                # 3. Structural Cleaning (Enhanced for Qwen/Gemma-4)
                structural_tags = [
                    r'<\|?channel>(?:text|thought)?>?', r'<\/\|?channel\|?>?', r'<channel\s*\|?>?', r'<\/channel\s*\|?>?',
                    r'(?:<channel\s*\|?>|<\/channel\s*\|?>)+', r'<think>?', r'<\/think>?', r'<thought>?', r'<\/thought>?', 
                    r'<\|/>', r'<turn/>', r'<\|im_start|>(?:thought|assistant)?', r'<\|im_end|>', r'<\|endoftext|>'
                ]

                for tag in structural_tags:
                    final_answer = re.sub(tag, '', final_answer, flags=re.IGNORECASE | re.DOTALL)
                    think_log = re.sub(tag, '', think_log, flags=re.IGNORECASE | re.DOTALL)
                
                # 3b. SYNTHESIS FALLBACK (Hardened)
                clean_answer = final_answer.strip()
                was_thinking_expected = (self.active_persona_level >= 3 or self.state.get("deep_cook"))
                finish_reason = response["choices"][0].get("finish_reason", "")
                
                # Trigger synthesis if the model gave zero or extremely short output after thinking
                needs_synthesis = False
                if was_thinking_expected and len(clean_answer) < 15 and len(think_log) > 50:
                    needs_synthesis = True
                elif "length" in str(finish_reason) and was_thinking_expected and len(clean_answer) < 20:
                    needs_synthesis = True
                
                if needs_synthesis and self.active_persona_level > 1:
                    reasoning_source = think_log if think_log else full_resp
                    if self.active_persona_level == 6:
                        self.process_queue.put({"status": "thinking_status", "content": "Cecilia is gathering her thoughts..."})
                        synthesized = self._perform_level6_synthesis(user_message, reasoning_source)
                    else:
                        self.process_queue.put({"status": "thinking_status", "content": "Synthesizing final response..."})
                        synthesized = self._perform_final_synthesis(user_message, reasoning_source)
                    
                    if synthesized:
                        final_answer = synthesized.strip()
                        for tag in structural_tags:
                            final_answer = re.sub(tag, '', final_answer, flags=re.IGNORECASE | re.DOTALL)
                    else:
                        print("[SYNTHESIS] Pass failed or returned None; retaining original draft.")
                
                # 3c. TOOL LOOP INTEGRATION (Hardened for standard worker)
                final_answer = self._run_tool_loop(final_answer, prompt_str, params)
                
                # 4. LaTeX Artifact Removal
                final_answer = self._clean_latex_artifacts(final_answer.strip())
                if not final_answer and full_resp:
                    final_answer = full_resp.strip() # Fallback to raw if hygiene nuked it
                
                # 5. Delivery
                self.process_queue.put({"status": "thinking_status", "content": "Wall dropping. Here's the deep dive:"})
                print(f"[INFERENCE] Generation complete. Final response length: {len(final_answer)} chars.")
                self.process_queue.put({
                    "status": "session_finished",
                    "user_msg": user_message,
                    "think_log": think_log.strip(),
                    "final_answer": final_answer.strip(),
                    "is_error": False
                })
                return
            else:
                if self.kv_manager and TRI_ATTENTION_ENABLED:
                    processed_msgs = self.kv_manager.enforce_kv_budget(temp_messages)
                else:
                    processed_msgs = temp_messages[-12:]
                
                msgs = [{"role": "system", "content": sys_content}] + processed_msgs
                self.process_queue.put({"status": "thinking_status", "content": "Almost there... batching the response."})
                
                # GIL-Safety: Internal Streaming
                full_resp = ""
                gen_iterator = self.model.create_chat_completion(messages=msgs, **params, stream=True)
                for chunk in gen_iterator:
                    if self.stop_process.is_set(): break
                    if "content" in chunk["choices"][0]["delta"]:
                        txt = chunk["choices"][0]["delta"]["content"]
                        full_resp += txt
                        self.process_queue.put({"status": "streaming", "content": txt})
                    time.sleep(0.001) # Heartbeat for UI
                
                # Perform the same background cleaning for standard path
                final_answer = self._clean_latex_artifacts(full_resp.strip())
                self.process_queue.put({"status": "thinking_status", "content": "Wall dropping. Here's the deep dive:"})
                self.process_queue.put({
                    "status": "session_finished",
                    "user_msg": user_message,
                    "think_log": "",
                    "final_answer": final_answer,
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
            
            def _run_step_streaming(log_title, prompt, status_msg, ctype=None, cnum=0, dnum=0, temp_override=None):
                if self.stop_process.is_set(): raise InterruptedError()
                self.process_queue.put({"status": "log_update", "content": f"\n--- {log_title} ---\n"})
                self.process_queue.put({"status": "thinking_status", "content": status_msg})
                
                sys_msg = DEEP_COOK_SYSTEM_PROMPTS.get(self.active_persona_level, "You are a logical, step-by-step reasoning AI.")
                is_gemma = "gemma" in self.model_path.lower()
                params = self._get_inference_params(prompt)
                if temp_override:
                    params["temperature"] = temp_override
                    # Nudge top_k and min_p if temperature changed to break loops
                    params["top_k"] = 64
                    params["min_p"] = 0.02
                
                if is_gemma:
                    for req_stop in ["<end_of_turn>", "<start_of_turn>", "<turn|>", "<|turn>", "<|file_separator|>", "<eos>"]:
                        if req_stop not in params.get("stop", []):
                            params.setdefault("stop", []).append(req_stop)
                    
                    # Gemma-4: Tool definitions injection for Deep Cook (Template-Aligned)
                    def oq(s): return f"<|\"|>{s}<|\"|>"
                    tool_defs = ""
                    tools = self.tool_registry.get_definitions(self.active_persona_level)
                    for t in tools:
                        f = t["function"]
                        t_params = f.get("parameters", {})
                        t_props = t_params.get("properties", {})
                        t_prop_str = ",".join([f"{k}:{{description:{oq(v.get('description',''))},type:{oq(v.get('type','STRING'))}}}" for k, v in t_props.items()])
                        t_req_str = ",".join([oq(r) for r in t_params.get("required", [])])
                        
                        tool_defs += f"<|tool>declaration:{f['name']}{{description:{oq(f['description'])},parameters:{{properties:{{{t_prop_str}}},required:[{t_req_str}]}}}}<tool|>\n"
                    
                    sys_msg += (
                        "\n[CRITICAL RESTRICTION]: You MUST begin your response by opening the thought channel. Complete ALL planning, tone adjustments, and technical notes INSIDE that channel before closing it."
                    )
                    
                    # FORCED THINKING MODALITY (Gemma-4 Official)
                    # We let the model start its native thinking channel automatically.
                    prompt_str = f"<start_of_turn>system\n{sys_msg}\n{tool_defs}<end_of_turn>\n<start_of_turn>user\nTask: {prompt}<end_of_turn>\n<start_of_turn>model\n"
                    stream = self.model(prompt_str, stream=True, echo=False, **params)
                else:
                    # Standard logic for non-Gemma models
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
                    if is_gemma:
                        ch = chunk.get('choices', [{}])[0]
                        c = ch.get('text', "")
                        if ch.get('finish_reason'): last_finish_reason = ch.get('finish_reason')
                    else:
                        ch = chunk.get('choices', [{}])[0]
                        c = ch.get('delta', {}).get("content", "")
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
                    # Gemma-4 manual tool parsing
                    if any(tag in result_text for tag in ["<ctrl42>call:", "<|tool_call>call:", "<|tool>call:", "call:", "action:"]):
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
                    if self.active_persona_level == 6:
                        self.process_queue.put({"status": "thinking_status", "content": f"Cycle {current_cycle} | Hardening Memory for Cycle {current_cycle+1}..."})
                        lvl6_context = "\nInject worldbuilding context: Ensure narrative consistency and architectural lore."
                    else:
                        lvl6_context = ""

                    snapshot_prompt = (
                        f"Original Query: {user_msg}\n"
                        f"Cycle {current_cycle} Data Progress: {cycle_draft[-2000:]}\n"
                        f"[CYCLE QUALITY]: Hollow drafts detected: {cycle_state['hollow_drafts_detected']}\n"
                        f"Context Assessment: Review Cycle {current_cycle} and compare it against the original query.{lvl6_context}\n"
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
            if self.active_persona_level == 6:
                # Level 6 Restoration: Use Cecilia-specific synthesis (passing critique)
                final_resp = self._perform_level6_synthesis(user_msg, full_draft_history, critique_txt)
            else:
                final_resp = self._perform_final_synthesis(user_msg, full_draft_history, skip_critique, critique_txt)
            
            # Wit-Layer: Delivery Message
            self.process_queue.put({"status": "thinking_status", "content": "Wall dropping. Here's the deep dive:"})
            
            # Embed the thought history for UI rendering with frontend tags
            payload = final_resp if final_resp else "Synthesis was unable to complete cleanly due to hardware constraints. Please view the thoughts for detailed logic."
            
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
        if self.text_buffer:
            self._update_ai_message(self.text_buffer)
            self.text_buffer = ""
            
        user_msg = data.get("user_msg", "")
        think_log = data.get("think_log", "")
        final_answer = data.get("final_answer", "")
        is_error = data.get("is_error", False)
        
        self._finalize_message(user_msg, think_log, final_answer, is_error)

    def _buffer_tool_log(self, content):
        if self.tool_log and self.tool_log.winfo_exists():
            try:
                self.tool_log.config(state='normal')
                self.tool_log.insert(tk.END, content + "\n\n", "stdout")
                if self.tool_log.yview()[1] >= 0.9:
                    self.tool_log.see(tk.END)
                self.tool_log.config(state='disabled')
            except: pass

    def _buffer_diag_log(self, content):
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
        if self.error_log and self.error_log.winfo_exists():
            try:
                self.error_log.config(state='normal')
                self.error_log.insert(tk.END, content + "\n", "stderr")
                if self.error_log.yview()[1] >= 0.9:
                    self.error_log.see(tk.END)
                self.error_log.config(state='disabled')
            except: pass
        
        # Also mirror to terminal
        try:
            sys.__stdout__.write(content + "\n")
            sys.__stdout__.flush()
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
            if think and think.winfo_exists(): think.stop()
            self.state["response_start_idx"] = hist.index(tk.END + "-1c")
            self._append_to_chat(f"\n\n{self._get_persona_label()}: ", "ai_lead")
            self.state["response_started"] = True

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
    
    def _display_ai_message(self, msg="", is_streaming=True):
        if is_streaming:
            self.state["response_started"] = False
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
        """Processes Markdown formatting (bold, italic, lists) in a chunked, UI-safe manner."""
        print(f"[RENDER] Applying markdown to range {start_idx} -> {end_idx} (Thought: {is_thought}).")
        hist = self.chat_history
        if hist is None: return
        
        # MISSION: If this is a massive thought log, use the async scheduler anyway to prevent UI death
        if is_thought:
            steps = [
                (r'\*\*(.+?)\*\*', "md_bold", "regex"),
                (r'\*(.+?)\*', "md_italic", "regex")
            ]
        else:
            # For normal responses, we use a more comprehensive set of rules
            steps = [
                (r'^\s*### ', "md_bold", "header"),
                (r'^\s*[\*\-] ', "md_list", "list"),
                (r'\*\*\*(.+?)\*\*\*', "md_bold_italic", "regex"),
                (r'___(.+?)___', "md_bold_italic", "regex"),
                (r'\*\*(.+?)\*\*', "md_bold", "regex"),
                (r'__(.+?)__', "md_bold", "regex"),
                (r'\*(.+?)\*', "md_italic", "regex"),
                (r'_(.+?)_', "md_italic", "regex")
            ]

        
        def process_step(index=0):
            if index >= len(steps) or not hist.winfo_exists():
                return
            
            pattern, tag, mode = steps[index]
            hist.config(state='normal')
            max_iters = 500  # Safety limit to prevent infinite loops
            
            if mode == "header":
                search_idx = start_idx
                iters = 0
                while iters < max_iters:
                    iters += 1
                    m = hist.search(pattern, search_idx, stopindex=end_idx, regexp=True)
                    if not m or hist.compare(m, ">=", end_idx): break
                    line_end = hist.index(f"{m} lineend")
                    hist.tag_add(tag, m, line_end)
                    # CRITICAL: Advance to the START of the NEXT line, not just lineend.
                    # Tcl's ^ anchor can re-match the same line from lineend position.
                    next_line = hist.index(f"{m} + 1 lines linestart")
                    if hist.compare(next_line, "<=", search_idx):
                        break  # No advancement — bail to prevent infinite loop
                    search_idx = next_line
            elif mode == "list":
                search_idx = start_idx
                iters = 0
                while iters < max_iters:
                    iters += 1
                    m = hist.search(pattern, search_idx, stopindex=end_idx, regexp=True)
                    if not m or hist.compare(m, ">=", end_idx): break
                    line_text = hist.get(f"{m} linestart", f"{m} lineend")
                    marker = re.search(r'^\s*[\*\-] ', line_text)
                    if marker:
                        m_start = hist.index(f"{m} linestart + {marker.start()} chars")
                        m_end = hist.index(f"{m} linestart + {marker.end()} chars")
                        hist.delete(m_start, m_end)
                        hist.insert(m_start, " • ", base_tags)
                        hist.tag_add("md_list", f"{m_start} linestart", f"{m_start} lineend")
                    # CRITICAL: Advance to next line start to prevent re-matching
                    next_line = hist.index(f"{m} + 1 lines linestart")
                    if hist.compare(next_line, "<=", search_idx):
                        break  # No advancement — bail
                    search_idx = next_line
            else:
                self._regex_format(pattern, tag, start_idx, end_idx, base_tags)
            
            hist.config(state='disabled')
            # CRITICAL: Process next step synchronously to prevent state='normal'/state='disabled' 
            # race conditions with _finalize_message and check_process_queue callbacks
            process_step(index + 1)

        # Process all markdown steps synchronously (no async scheduling)
        process_step(0)

    def _regex_format(self, pattern, tag, start_idx, end_idx, base_tags):
        """High-performance native search/format for chat history with absolute index safety."""
        #print(f"[RENDER] Regex scan: {pattern}")
        hist = self.chat_history
        if not hist or not start_idx or not end_idx: return
        
        # MISSION: Ensure indices are canonical to avoid Tcl errors
        try:
            search_idx = hist.index(start_idx)
            limit_idx = hist.index(end_idx)
        except: return
        
        max_iterations = 400
        iterations = 0
        start_time = time.time()
        
        while iterations < max_iterations:
            # MISSION: Prevent long-running regex loops from freezing the UI
            # Yield if we've spent more than 15ms in this pattern
            if (time.time() - start_time) > 0.015:
                self.root.after(5, lambda: self._regex_format_continuation(pattern, tag, search_idx, limit_idx, base_tags, iterations, start_time))
                return

            iterations += 1
            match_count = tk.IntVar()
            
            # Perform search with local safety
            try:
                match_idx = hist.search(pattern, search_idx, stopindex=limit_idx, regexp=True, count=match_count)
            except Exception as e:
                print(f"[UI SAFETY] Search error: {e}")
                break
                
            if not match_idx: break
            
            # Canonicalize match_idx immediately
            match_idx = hist.index(match_idx)
            if hist.compare(match_idx, ">=", limit_idx): break
            
            count = match_count.get()
            if count <= 0:
                search_idx = f"{match_idx} + 1c"
                continue
                
            full_match_text = hist.get(match_idx, f"{match_idx} + {count} chars")
            m = re.search(pattern, full_match_text)
            if not m:
                search_idx = f"{match_idx} + {count} chars"
                continue
                
            inner_text = m.group(1)
            m_end_idx = hist.index(f"{match_idx} + {count} chars")
            
            hist.config(state='normal')
            try:
                hist.delete(match_idx, m_end_idx)
                hist.insert(match_idx, inner_text, base_tags + (tag,))
            finally:
                hist.config(state='disabled')
            
            # Strictly advance to avoid infinite loop traps
            search_idx = hist.index(f"{match_idx} + {len(inner_text)} chars")
            if hist.compare(search_idx, "<=", match_idx):
                search_idx = f"{match_idx} + 1c"
        
        if iterations >= max_iterations:
            print(f"[UI SAFETY] High recursion in _regex_format for pattern: {pattern}")

    def _regex_format_continuation(self, pattern, tag, search_idx, limit_idx, base_tags, iterations, start_time):
        """Allows _regex_format to resume after yielding to the UI loop."""
        hist = self.chat_history
        if not hist or not hist.winfo_exists(): return
        
        # Reset timer for the next chunk
        new_start_time = time.time()
        
        # Resume the loop
        self._regex_format_logic(pattern, tag, search_idx, limit_idx, base_tags, iterations, new_start_time)

    def _regex_format_logic(self, pattern, tag, search_idx, limit_idx, base_tags, iterations, start_time):
        """Internal logic for _regex_format to support yielding."""
        hist = self.chat_history
        max_iterations = 400
        
        while iterations < max_iterations:
            if (time.time() - start_time) > 0.015:
                self.root.after(5, lambda: self._regex_format_continuation(pattern, tag, search_idx, limit_idx, base_tags, iterations, start_time))
                return

            iterations += 1
            match_count = tk.IntVar()
            try:
                match_idx = hist.search(pattern, search_idx, stopindex=limit_idx, regexp=True, count=match_count)
            except: break
                
            if not match_idx: break
            match_idx = hist.index(match_idx)
            if hist.compare(match_idx, ">=", limit_idx): break
            
            count = match_count.get()
            if count <= 0:
                search_idx = f"{match_idx} + 1c"
                continue
                
            full_match_text = hist.get(match_idx, f"{match_idx} + {count} chars")
            m = re.search(pattern, full_match_text)
            if not m:
                search_idx = f"{match_idx} + {count} chars"
                continue
                
            inner_text = m.group(1)
            m_end_idx = hist.index(f"{match_idx} + {count} chars")
            
            hist.config(state='normal')
            try:
                hist.delete(match_idx, m_end_idx)
                hist.insert(match_idx, inner_text, base_tags + (tag,))
            finally:
                hist.config(state='disabled')
            
            search_idx = hist.index(f"{match_idx} + {len(inner_text)} chars")
            if hist.compare(search_idx, "<=", match_idx):
                search_idx = f"{match_idx} + 1c"
        
        if iterations >= max_iterations:
            print(f"[UI SAFETY] High recursion in _regex_format for pattern: {pattern}")


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
                    hist.tag_config(tag, foreground="#ffcc00", underline=1, font=("Open Sans", 9, "bold"))
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
                # Web Fetch
                headers = {"User-Agent": "SerenityPC/4.0 RichMediaFetcher"}
                resp = requests.get(url, headers=headers, timeout=10)
                resp.raise_for_status()
                pil_img = Image.open(io.BytesIO(resp.content))
            else:
                # Local Path
                if any(url.lower().endswith(ext) for ext in [".mp4", ".avi", ".mkv", ".mov"]):
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
        loading = tk.Label(popup, text="Loading...", fg="white", bg="#0a0a0a", font=("Open Sans", 12))
        loading.pack(padx=50, pady=50)
        
        def load():
            try:
                # (Re-use fetch logic or just open)
                if url.startswith("http"):
                    resp = requests.get(url, timeout=10); pil_img = Image.open(io.BytesIO(resp.content))
                else:
                    if any(url.lower().endswith(ext) for ext in [".mp4", ".avi", ".mkv", ".mov"]):
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

    def _run_tool_loop(self, full_resp, prompt_str, params, depth=0):
        """
        Parses tool calls from model output, executes them, and recursively 
        generates the final answer. Max depth 3 to prevent runaway inference.
        """
        if depth > 3:
            return f"{full_resp}\n\n[SYSTEM]: Tool recursion limit reached. Truncating response."

        # Added support for Gemma-4 'action' tags and missing 'call:' prefixes
        call_match = re.search(r'(?:<ctrl42>call:|<\|tool_call>call:|<\|tool_call\|>call:|<\|tool>call:|call:|action:)([\w_]+)\s*\{(.*?)\}', full_resp, re.DOTALL | re.IGNORECASE)
        if not call_match:
            return full_resp

        call_name = call_match.group(1).strip()
        args_raw = call_match.group(2).strip()
        
        # Robust Argument Parsing (Template-Aware)
        # MISSION: Handle Gemma-4 <|"|> quotes and unquoted JSON keys from reasoning models
        clean_args = args_raw.replace('<|"|>', '"').strip()
        if clean_args.startswith("{") and clean_args.endswith("}"):
            clean_args = clean_args[1:-1].strip()
            
        args = {}
        # Multi-stage JSON recovery
        # Prevent VS Code debugger from halting on expected JSONDecodeErrors
        is_json_format = not clean_args or clean_args.strip().startswith('"')
        
        if is_json_format:
            try:
                # Stage 1: Standard JSON (Fast path)
                args = json.loads("{" + clean_args + "}")
            except Exception:
                pass

        if not args and clean_args:
            try:
                # Stage 2: Key-Value Pair Extraction Fallback (Robust path)
                # MISSION: Handle unquoted keys and unquoted string values from logic models
                kv_pattern = r'(?m)^\s*["\']?([\w_]+)["\']?\s*[:=]\s*(.*?)\s*(?:,|$)'
                matches = re.finditer(kv_pattern, clean_args)
                for m in matches:
                    key = m.group(1).strip()
                    val = m.group(2).strip().strip('"\'')
                    if val: args[key] = val
                
                if not args:
                    # Stage 3: Quote unquoted keys (Traditional fallback)
                    fixed = re.sub(r'(?<!["\'])(\b\w+\b)(?!["\'])\s*:', r'"\1":', clean_args)
                    fixed = re.sub(r',\s*$', '', fixed.strip())
                    args = json.loads("{" + fixed + "}")
            except Exception:
                try:
                    # Stage 4: Bare value extraction (for single-arg tools)
                    bare_val = re.sub(r'^.*?:\s*', '', clean_args).strip().strip('"\'')
                    if bare_val:
                        if call_name == "web_search": args["query"] = bare_val
                        elif call_name == "read_file": args["path"] = bare_val
                except Exception:
                    pass # Absolute failure
        
        # Inform UI
        self.process_queue.put({"status": "thinking_status", "content": f"Executing tool: {call_name}"})
        self.process_queue.put({"status": "tool_log_update", "content": f"\n[{time.strftime('%H:%M:%S')}] Executing: {call_name}\nArgs: {args}"})
        
        try:
            # 1. ATTEMPT EXECUTION (TIGHT WRAP)
            try:
                observation = self.tool_registry.execute(call_name, args)
                if not observation or "Error:" in observation or "unreachable" in observation or "blocked" in observation.lower():
                    observation = str(observation or "") + "\n\n[SYSTEM]: The tool call returned no results or failed. Do not give up. Please proceed by using your internal knowledge, logical deduction, or alternative strategies to answer the user's request directly. Do not simply state that you cannot search."
            except Exception as e:
                observation = f"Error: Tool execution failed. System Exception: {str(e)}\n\n[SYSTEM]: Please proceed by using your internal knowledge to formulate the best possible response."
                self.process_queue.put({"status": "log_update", "content": f"\n[TOOL FAILURE] {call_name}: {str(e)}\n"})

            self.process_queue.put({"status": "tool_log_update", "content": f"Result: \n{str(observation)[:200]}..."})
            
            def oq(s): return f"<|\"|>{s}<|\"|>"
            
            # 2. CHAT TEMPLATE ALIGNMENT
            clean_resp = full_resp
            clean_resp = re.sub(r'(?s)<think>.*?(?:<\/think>|$)', '', clean_resp, flags=re.IGNORECASE)
            clean_resp = re.sub(r'(?s)<\|channel>thought.*?(?:<channel\|>|$)', '', clean_resp, flags=re.IGNORECASE)
            if "<think>" in clean_resp.lower() or "<|channel>thought" in clean_resp.lower():
                clean_resp = re.sub(r'.*?(?=<ctrl42>|<\|tool_call>)', '', clean_resp, flags=re.IGNORECASE | re.DOTALL)
            
            clean_resp = re.sub(r'^.*?<\|tool_call>', '<|tool_call>', clean_resp, flags=re.IGNORECASE | re.DOTALL)
            clean_resp = re.sub(r'^.*?<ctrl42>', '<ctrl42>', clean_resp, flags=re.IGNORECASE | re.DOTALL)
            
            if not clean_resp.strip().endswith("<turn|>"):
                clean_resp = clean_resp.strip() + "<turn|>\n"
            
            response_turn = f"<|turn>user\n<|tool_response>response:{call_name}{{value:{oq(observation)}}}<tool_response|><turn|>\n"
            new_prompt = prompt_str + clean_resp + response_turn + "<|turn>model\n"
            
            self.process_queue.put({"status": "thinking_status", "content": "Synthesizing tool results..."})
            new_text = self._run_blocking_inference(new_prompt, params)
            
            # 3b. SYNTHESIS SAFETY CHECK (Prevent "0 response" issue)
            if not new_text or len(new_text.strip()) < 5:
                # If synthesis failed or is empty, try a "forced" synthesis with a stricter prompt
                forced_sys = f"{PERSONA_PROMPTS.get(self.active_persona_level, 'You are Serenity.')}\n[DIRECT STRIKE]: Based on the search results below, provide a helpful answer."
                forced_prompt = f"Original Query: {prompt_str[-500:]}\n\nTool Results: {observation}\n\nDeliver the final response now."
                new_text = self._run_blocking_inference([{"role": "system", "content": forced_sys}, {"role": "user", "content": forced_prompt}], params)

            return self._run_tool_loop(new_text, new_prompt, params, depth=depth+1)
            
        except Exception as e:
            return f"{full_resp}\n\nI apologize, but I encountered a system-level error during the synthesis phase: {str(e)}. Please try rephrasing your request."
      
    def _detect_repetition(self, text, min_len=20, max_repeats=3):
        """Detects if any substring of at least min_len characters is repeated max_repeats or more times in the last 400 characters."""
        if not text:
            return False
        recent = text[-400:]
        n = len(recent)
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
            
            # 3. GENERIC TAGS (Fallback)
            r'(?s)<think>\n?(.*?)(?:<\/think>|$)',
            r'(?s)(?:<\|channel>)+thought\n?(.*?)(?:<channel\|?>|<\/\|?channel\|?>|<\|channel>|(?=<start_of_turn>)|$)',
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
        """Removes MathJax/LaTeX formatting artifacts that Tkinter cannot render."""
        if not text: return ""
        initial_len = len(text)
        
        # 1. Math Block Wrappers (Do these first to expose internal commands)
        text = re.sub(r'\$\$(.*?)\$\$', r'\1', text, flags=re.DOTALL)
        text = re.sub(r'\\\[(.*?)\\\]', r'\1', text, flags=re.DOTALL)
        
        # 2. Mathematical Structures
        text = re.sub(r'\\(?:d|t)?frac\{([^{}]*)\}\{([^{}]*)\}', r'\1/\2', text)
        
        # 3. Formatting Commands
        tags = ["mathbf", "mathrm", "mathit", "text", "textbf", "textit", "underline"]
        pattern = r'\\(?:' + '|'.join(tags) + r')\{([^}]*)\}'
        text = re.sub(pattern, r'\1', text)
        text = text.replace(r'\left(', '(').replace(r'\right)', ')')
        text = text.replace(r'\left[', '[').replace(r'\right]', ']')
        text = text.replace(r'\left\{', '{').replace(r'\right\}', '}')
        
        # 4. Special Symbols
        symbol_map = {
            r'\rightarrow': '->', r'\to': '->', r'\Rightarrow': '=>',
            r'\checkmark': '✓', r'\neg': 'NOT ',
            r'\times': 'x', r'\cdot': '*', r'\div': '/',
            r'\pm': '+/-', r'\approx': '~', r'\sim': '~',
            r'\Phi': 'Phi', r'\phi': 'phi', r'\infty': 'infinity',
            r'\le': '<=', r'\ge': '>=', r'\ne': '!=',
            r'\alpha': 'alpha', r'\beta': 'beta', r'\gamma': 'gamma',
            r'\theta': 'theta', r'\lambda': 'lambda', r'\mu': 'mu',
            r'\pi': 'pi', r'\sigma': 'sigma', r'\omega': 'omega',
            r'\Delta': 'Delta', r'\dots': '...', r'\ldots': '...'
        }
        for lat, plain in symbol_map.items():
            text = text.replace(lat, plain)
            
        # 5. Context-Aware "No-Backslash" Hygiene
        # MISSION: Only replace 'rightarrow', 'to', 'ge', 'ne' etc. if they are inside $...$ 
        if '$' in text:
            def safe_math_clean(match):
                inner = match.group(1)
                orig = f"${inner}$"
                # Apply replacements only as whole words inside the math block
                inner = inner.replace('rightarrow', '->').replace('Rightarrow', '=>').replace('to', '->')
                inner = inner.replace('ge', '>=').replace('ne', '!=')
                # Only strip the dollars if we actually cleaned something internal
                if f"${match.group(1)}$" != orig: return inner
                if any(x in inner for x in ['->', '=>', '>=', '!=', 'alpha', 'beta', 'gamma']): return inner
                return orig
            text = re.sub(r'(?<!\\)\$([^{}$\n]+?)(?<!\\)\$', safe_math_clean, text)
            
        # 6. Escaped Characters
        text = text.replace(r'\{', '{').replace(r'\}', '}').replace(r'\%', '%').replace(r'\$', '$').replace(r'\#', '#')
        
        if len(text) != initial_len:
            print(f"[SYSTEM] LaTeX hygiene complete. Reduced artifacts by {initial_len - len(text)} chars.")
            
        return text.strip()

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
            
            btn = tk.Button(hist, text=f"[+] View Cycle {cnum}", bg="#202020", fg="#888888", relief=tk.FLAT, font=("Consolas", 8))
            btn.config(command=lambda t=tag, b=btn, c=cnum: toggle_cyc(t, b, c))
            
            hist.insert(tk.END, "\n", ("ai",))
            hist.window_create(tk.END, window=btn)
            hist.insert(tk.END, "\n", ("ai",))
            hist.insert(tk.END, f"{title}\n", (tag, "ai"))
            hist.tag_config(tag, elide=True, foreground="#808080", font=("Consolas", 9))
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

                btn = tk.Button(hist, text=f"[+] Step {dnum}", bg="#151515", fg="#777777", relief=tk.FLAT, font=("Consolas", 7))
                btn.config(command=lambda t=tag, b=btn, d=dnum: toggle_draft(t, b, d))
                
                hist.insert(tk.END, "  ", (cyc_tag, "ai"))
                hist.window_create(tk.END, window=btn)
                hist.insert(tk.END, "\n", (cyc_tag, "ai"))
                hist.insert(tk.END, f"  --- {title} ---\n  ", (tag, cyc_tag, "ai"))
                hist.tag_config(tag, elide=True, foreground="#707070", font=("Consolas", 8))

        elif ctype == "memory":
            def toggle_mem(t=tag, b=None):
                st = hist.tag_cget(t, "elide")
                new_state = (st == "0")
                hist.tag_config(t, elide=new_state)
                if b: b.config(text=f"{'[+]' if new_state else '[-]'} {title or 'Context Assessment'}")

            btn = tk.Button(hist, text=f"[+] {title or 'Context Assessment'}", bg="#1a1a2e", fg="#ababab", relief=tk.FLAT, font=("Consolas", 8, "italic"))
            btn.config(command=lambda t=tag, b=btn: toggle_mem(t, b))
            
            hist.insert(tk.END, "\n", ("ai",))
            hist.window_create(tk.END, window=btn)
            hist.insert(tk.END, "\n", ("ai",))
            hist.insert(tk.END, "", (tag, "ai"))
            hist.tag_config(tag, elide=True, foreground="#ababab", font=("Consolas", 9))

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
                hist.tag_config(comp_tag, foreground="#00ffcc", font=("Consolas", 10))
                hist.config(state='disabled')
        
        hist.see(tk.END)

    def _revert_status_label(self):
        if hasattr(self, 'system_status_label') and self.system_status_label.winfo_exists():
            if self.model_path:
                model_name = os.path.basename(self.model_path)
                self.system_status_label.config(text=f"Loaded: {model_name}")
            else:
                self.system_status_label.config(text="System: Idle")

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
        if self.state.get("staged_multimodal") and self.state["staged_multimodal"]["type"] == "video":
            self._reset_multimodal_ui()
            return

        files = filedialog.askopenfilenames(title="Select Video(s)", filetypes=[("Videos", "*.mp4 *.mkv *.avi *.mov")])
        if files:
            self.state["processing_queue"] = list(files)
            total_size_mb = sum(os.path.getsize(f) for f in files) / (1024 * 1024)
            
            self.state["staged_multimodal"] = {"type": "video", "path": files[0]} # Keep path for backward compat if needed
            self.btn_video.config(text=f"[❌] Clear ({len(files)})", bg="#4a0000")
            
            queue_msg = f"Queue: {len(files)} videos | {total_size_mb:.1f} MB"
            self._log_and_display(queue_msg)
            print(f"[SYSTEM] Videos staged: {len(files)} files.")
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
        if self.model is None or tier != target_tier:
             if not self.model_paths.get(target_tier):
                messagebox.showerror("Error", f"Vision model for {mode} not set!")
                return
             self._log_and_display(f"Switching to Vision Engine ({mode})...")
             self.pending_task = {"type": "vision_standard", "message": user_msg, "staged": {"type": mode, "path": file_path}}
             self.model_swap(target_tier=target_tier)
             return

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


                if mode == "video": # APEX START
                    # Strategic Burst: Process batches of 10 from DRAM Staging
                    # Decoding was already done on P-Cores into System RAM
                    
                    # Get total chunks for progress tracking
                    _v = cv2.VideoCapture(video_path)
                    _total_f = int(_v.get(cv2.CAP_PROP_FRAME_COUNT))
                    _fps = _v.get(cv2.CAP_PROP_FPS)
                    _v.release()
                    _f_step = max(int(_fps / 5.0), 1)
                    _eff_f = (_total_f + _f_step - 1) // _f_step
                    _total_c = (_eff_f + 4) // 5 

                    # STEP 1: DRAM Staging Buffer (i7 "Crush")
                    self.process_queue.put({"status": "thinking_status", "content": "Phase 0: Staging all frames into DRAM (i7 Crush)..."})
                    dram_frame_buffer = []
                    for c_idx, frame_batch in enumerate(VisionHandler.get_chunked_frames(video_path, chunk_size=5)):
                        if self.stop_process.is_set(): break
                        dram_frame_buffer.append(frame_batch)
                        self.update_timeline_progress(c_idx + 1, _total_c)
                    
                    if self.stop_process.is_set(): continue

                    # STEP 2: The GPU "Bash" (Inference Loop)
                    dram_telemetry_logs = []
                    sliding_window = []
                    unicorn_frames = []
                    
                    for c_idx, frame_batch in enumerate(dram_frame_buffer):
                        if self.stop_process.is_set(): break
                        
                        # Update Progress for Inference Phase
                        self.update_timeline_progress(c_idx + 1, _total_c)
                        
                        # 5-frame Strategic Burst: Timeline injection
                        start_s = (c_idx * 5) / 5.0 
                        end_s = (c_idx * 5 + len(frame_batch)) / 5.0
                        t_tag = f"[[TIMELINE: {start_s:.1f}s - {end_s:.1f}s]]"
                        self.process_queue.put({
                            "status": "thinking_status", 
                            "content": f"Apex Burst: {t_tag}..."
                        })
                        
                        try:
                            # Apex Alignment: Shift to P-Cores for Inference
                            HardwareProfile.pin_to_p_cores()
                            start_time = time.time()
                            try:
                                # [STRATEGIC BURST] Processing Logic
                                # THE STRATEGIC BURST: Direct injection from DRAM to VRAM
                                # Verified Solo Swap: Projector is loaded solo (verified in hardware guards)
                                
                                # Phase 1: Silent Scout (Context Flush & Sliding Window)
                                prompt_content = []
                                if sliding_window:
                                    prompt_content.append({"type": "text", "text": f"[ROLLING CONTEXT - LAST 3 BURSTS]:\n" + "\n".join(sliding_window) + "\n\n"})
                                
                                prompt_content.append({"type": "text", "text": f"{t_tag} " + (user_msg if user_msg else VisionHandler.SILENT_SCOUT_UNIVERSAL_PROMPT)})
                                for f in frame_batch:
                                    prompt_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}})
                                
                                stream = self.model.create_chat_completion(
                                    messages=[{"role": "user", "content": prompt_content}],
                                    **self._get_inference_params(),
                                    stream=True
                                )
                                
                                sub_analysis = ""
                                ttft_measured = False
                                for chunk in stream:
                                    if not ttft_measured:
                                        ttft_measured = True
                                        inference_ms = (time.time() - start_time) * 1000
                                        print(f"[APEX] Swap Jig Inference Latency (TTFT): {inference_ms:.1f}ms (Card/Bus)")
                                    
                                    delta = chunk.get('choices', [{}])[0].get('delta', {})
                                    content = delta.get('content', '')
                                    if content: sub_analysis += content
                                
                                duration = time.time() - start_time
                                if not sub_analysis or len(sub_analysis) < 5:
                                    raise ValueError("Empty vision response.")
                                
                                # Unicorn Frame Selection via quality heuristic
                                try:
                                    import json
                                    clean_json = sub_analysis.strip()
                                    if clean_json.startswith('```json'): clean_json = clean_json[7:]
                                    if clean_json.startswith('```'): clean_json = clean_json[3:]
                                    if clean_json.endswith('```'): clean_json = clean_json[:-3]
                                    parsed = json.loads(clean_json.strip())
                                    q_val = parsed.get('q', '0')
                                    q_score = int(str(q_val).split('/')[0].strip()) if isinstance(q_val, (str, int)) else 0
                                    
                                    if q_score >= 8 and len(unicorn_frames) < 15:
                                        unicorn_frames.append(frame_batch[len(frame_batch)//2]) # Save middle frame of burst
                                except Exception: 
                                    pass

                                dram_telemetry_logs.append(f"{t_tag}: {sub_analysis}")
                                sliding_window.append(sub_analysis)
                                if len(sliding_window) > 3:
                                    sliding_window.pop(0)

                                segment_results.append(f"--- {t_tag} TELEMETRY ---\n{sub_analysis}\n")


                            except Exception as e:
                                print(f"[APEX] Burst Error: {e}")
                                segment_results.append(f"--- {t_tag} ERROR ---\n[Burst failed: {e}]\n")
                            
                            finally:
                                VisionHandler.hygiene_gate(self.model)
                        finally:
                            HardwareProfile.release_cores()

                    # Phase 2: Final Verdict Synthesis
                    self.process_queue.put({"status": "thinking_status", "content": "Phase 2: Compiling Final Verdict..."})
                    try:
                        HardwareProfile.pin_to_p_cores()
                        final_prompt = [{"type": "text", "text": VisionHandler.FINAL_VERDICT_PROMPT + "\n\n[DRAM LOGS]:\n" + "\n".join(dram_telemetry_logs)}]
                        for f in unicorn_frames:
                            final_prompt.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{f}"}})
                            
                        # Ensure VRAM is empty of old image tokens
                        VisionHandler.hygiene_gate(self.model)
                        
                        stream = self.model.create_chat_completion(
                            messages=[{"role": "user", "content": final_prompt}],
                            stream=True
                        )
                        final_analysis = ""
                        for chunk in stream:
                            delta = chunk.get('choices', [{}])[0].get('delta', {})
                            content = delta.get('content', '')
                            if content: final_analysis += content
                            
                        segment_results.append("\n=== APEX FINAL VERDICT ===\n" + final_analysis + "\n")
                    except Exception as e:
                        print(f"[APEX] Final Verdict Error: {e}")
                    finally:
                        HardwareProfile.release_cores()

                    # Export Aggregate Data before Synthesis Swap
                    if segment_results:
                        combined_analysis = "\n".join(segment_results)
                        output_txt = video_path.replace(os.path.splitext(video_path)[1], f"_STRATEGIC_analysis.txt")
                        with open(output_txt, "w", encoding="utf-8") as f:
                            f.write(combined_analysis)
                        results.append(output_txt)
                        self.process_queue.put({"status": "log_update", "content": f"Generated Aggregate: {os.path.basename(output_txt)} (Staged Burst)\n"})
                elif mode == "multimodal":
                    # Sequential Processing for Images and Audio
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
                            b64 = VisionHandler.encode_image(video_path, budget=budget)
                            if b64:
                                prompt_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
                        elif ext in ['.mp3', '.wav', '.ogg', '.flac', '.m4a']:
                            # Audio -> chunks
                            chunks = VisionHandler.get_audio_chunks(video_path, chunk_length_s=30, max_chunks=30)
                            # Modern Gemma E-series MM-Projector audio injection
                            for i, b64 in enumerate(chunks):
                                prompt_content.append({"type": "image_url", "image_url": {"url": f"data:audio/wav;base64,{b64}"}})
                                
                        prompt_content.append({"type": "text", "text": final_user_msg})
                        
                        self.process_queue.put({"status": "thinking_status", "content": f"Processing {filename} via Multimodal Projector..."})
                        
                        # Apply flash attention hygiene and clear KV for the new file 
                        VisionHandler.hygiene_gate(self.model)
                        
                        stream = self.model.create_chat_completion(
                            messages=[{"role": "user", "content": prompt_content}],
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
        self.load_history()
        self.update_persona_display()
        self.set_avatar_state("pleased")
        self._log_and_display(f"Loaded {os.path.basename(self.model_path)}")
        self.root.after(1500, lambda *args: self.set_avatar_state("listening"))

        if self.pending_task:
            t = self.pending_task; self.pending_task = None
            if t["type"] == "deep_cook": self.send_deep_cook_message(t["message"], True)
            elif t["type"] == "chat": self.send_message(t["message"], True)
            elif t["type"] == "vision_standard": 
                self.initiate_vision_analysis(t["staged"]["type"], t["staged"]["path"], t["message"])
            elif t["type"] == "synthesis_finalize":
                 self.process_queue.put({"status": "thinking_status", "content": "[PROCESS] Finalizing Resolution..."})
                 # MISSION: Execute heavy synthesis in background thread to prevent UI freeze
                 import threading
                 threading.Thread(target=self._synthesis_worker, args=(t,), daemon=True).start()

    def _handle_load_error(self, msg):
        self.state["running"] = False; self.model = None
        self.set_ui_state(model_loaded=False)
        self.set_avatar_state("confused")
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
        return "Cecilia" if self.active_persona_level == 6 else "Serenity"

    def _sanitize_synthesis_output(self, raw_text):
        if not raw_text: return ""
        
        # Split raw response at the end of the thinking tag and discard the thought prefix
        closers = [r'<\/think>', r'<channel\|>', r'<\/channel\|>', r'\[\/DRAFT\]']
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
            "<think>", "</think>", "<|channel>thought", "<channel|>", "Final Response:", "Final Answer:",
            "<|channel>text", "<|channel>assistant", "</channel|>", "###", "<|im_start|>", "<|im_end|>", "<|endoftext|>"
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

    def _perform_final_synthesis(self, user_msg, reasoning_history, skip_critique=True, critique_txt=""):
        """Shared logic to distill thoughts into a final response."""
        try:
            self.process_queue.put({"status": "thinking_status", "content": "Finalizing Response..."})
            is_gemma = "gemma" in self.model_path.lower()
            params = self._get_inference_params(reasoning_history)
            critique_part = "" if skip_critique else f"Critique Findings: {critique_txt}\n"
            history_subset = reasoning_history[-10000:] if len(reasoning_history) > 10000 else reasoning_history
            synth_params = dict(params)
            synth_params["max_tokens"] = 4096

            final_prompt = (
                f"User Query: {user_msg}\n\n"
                f"Reasoning to convert:\n{history_subset}\n\n"
                f"[ORGANIZED BACKEND THOUGHTS]: {self.state.get('dmn_backbone', {}).get('last_simmer', 'N/A')}\n\n"
                f"{critique_part}" f"Convert the reasoning above into a direct final response. Speak directly to the user now. Output ONLY the final response."
            )

            if is_gemma:
                self.process_queue.put({"status": "thinking_status", "content": "Refining Response..."})
                persona_instr = PERSONA_PROMPTS.get(self.active_persona_level, "You are Serenity.")
                synth_sys = (
                    f"{persona_instr}\n\n"
                    "[TASK]: Convert the provided reasoning history into a direct final response. "
                    "Speak directly to the user now. Do not use internal monologue tags or your thought process. "
                    "Output ONLY the final response text."
                )
                prompt_str = (
                    f"<start_of_turn>system\n{synth_sys}<end_of_turn>\n"
                    f"<start_of_turn>user\n{final_prompt}<end_of_turn>\n"
                    f"<start_of_turn>model\n"
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
                        f"<start_of_turn>system\n{strike_sys}<end_of_turn>\n"
                        f"<start_of_turn>user\n{strike_prompt}<end_of_turn>\n"
                        f"<start_of_turn>model\n"
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

    def _perform_level6_synthesis(self, user_msg, reasoning_history, critique_txt=""):
        """Persona-specific distillation for Cecilia (Level 6)."""
        try:
            from serenity_resources import LEVEL6_SYNTHESIS_SYSTEM_PROMPT
            self.process_queue.put({"status": "thinking_status", "content": "Cecilia is delivering her truth..."})
            is_gemma = "gemma" in self.model_path.lower()
            params = self._get_inference_params(reasoning_history)
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
                    f"<start_of_turn>system\n{LEVEL6_SYNTHESIS_SYSTEM_PROMPT}<end_of_turn>\n"
                    f"<start_of_turn>user\n{final_prompt}<end_of_turn>\n"
                    f"<start_of_turn>model\n"
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
                        f"<start_of_turn>system\n{LEVEL6_SYNTHESIS_SYSTEM_PROMPT}<end_of_turn>\n"
                        f"<start_of_turn>user\n{retry_prompt}<end_of_turn>\n"
                        f"<start_of_turn>model\n"
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

    def _finalize_message(self, user_msg, think_log, final_answer, error=False):
        print(f"[SYSTEM] Finalizing message delivery (Error: {error}).")
        self.state["running"] = False
        self.set_ui_state(model_loaded=True, generating=False)
        if self.thinking_display and self.thinking_display.winfo_exists():
            self.thinking_display.stop()
        
        if self.chat_history is None: return

        hist = self.chat_history
        start_idx = self.state.get("response_start_idx")
        
        # 4. Atomic Reset for Rendering
        # If we were streaming, we need to clear the raw buffer before inserting formatted markdown
        # MISSION: Preserve Deep Cook dropdowns by avoiding nuke if they exist.
        if self.state.get("response_started") and start_idx:
            if not self.state.get("deep_cook"):
                hist.config(state='normal')
                try:
                    hist.delete(start_idx, tk.END)
                except: pass
                
                # Re-insert the Lead (e.g. "Cecilia: ")
                hist.insert(tk.END, f"\n\n{self._get_persona_label()}: ", "ai_lead")
                hist.config(state='disabled')
            else:
                # Deep Cook: Ensure synthesis text starts clean but don't wipe reasoning
                hist.config(state='normal')
                hist.insert(tk.END, "\n", ("ai",))
                hist.config(state='disabled')
            
            if error:
                self._append_to_chat(f"\n\n[System Error]: {final_answer}\n\n", "system")
                self.set_avatar_state("confused")
                try:
                    with open(self.error_log_file, 'a', encoding='utf-8') as f:
                        f.write(f"\n[{time.strftime('%H:%M:%S')}] [System Error]: {final_answer}\n")
                except: pass
                return

        # 5. UI Rendering (Atomic & Pre-Processed)
        hist.config(state='normal')
        render_mode = self.config.get("media_rendering", 1)
        render_start = hist.index(tk.END + "-1c")

        # Suppress redundant generic thinking block if Deep Cook structure is already present
        if think_log and not self.state.get("deep_cook"):
            think_tag = f"think_block_{int(time.time())}"
            def toggle_thoughts(tag=think_tag, b=None):
                is_elided = str(hist.tag_cget(tag, "elide")) in ["1", "True", "true"]
                if is_elided:
                     hist.tag_config(tag, elide=False)
                     if b: b.config(text="[-] Hide Thinking")
                else:
                     hist.tag_config(tag, elide=True)
                     if b: b.config(text="[+] View Thinking Process")
            
            btn = tk.Button(hist, text="[+] View Thinking Process", bg="#1a1a1a", fg="#00bfff", 
                            activebackground="#333333", activeforeground="#00bfff", 
                            relief=tk.FLAT, font=("Consolas", 8))
            btn.config(command=lambda t=think_tag, b=btn: toggle_thoughts(t, b))
            
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

        # 6. Insert Final Answer
        print(f"[SYSTEM] Delivery: {len(final_answer)} chars (Started: {self.state.get('response_started')}).")
        if not self.state.get("response_started", False):
            # In Deep Cook or fast synthesis, the lead might not be in chat yet
            # MISSION: For Deep Cook, the lead was already added by ui_start, so we only add if missing
            self._display_ai_message(final_answer, is_streaming=False)
        else:
            if final_answer:
                self._append_to_chat(final_answer, "ai")
        
        render_end = hist.index(tk.END + "-1c")
        if render_mode > 0:
            self._apply_markdown(render_start, render_end, ("ai",))
            self._post_process_media(start_idx=render_start)
        
        hist.config(state='disabled')
        hist.see(tk.END)
        
        # PERSISTENCE (Hardened against memory corruption)
        try:
            # MISSION: Prune history to prevent memory death on ultra-long sessions
            #if len(self.messages) > 100:
                #self.messages = self.messages[-60:]
                
            final_answer_history = final_answer.replace("<|file_separator|>", "").strip()
            self.messages.extend([
                {"role": "user", "content": str(user_msg)}, 
                {"role": "assistant", "content": str(final_answer_history)}
            ])
        except Exception as e:
            print(f"[SYSTEM] Persistence recovery: {e}")

        try:
            self.set_avatar_state("pleased")
        except: pass
        
        self.state["response_started"] = False
        self.root.after(5000, lambda *args: self.set_avatar_state("listening"))

    def _buffer_text(self, text):
        """Append text to the streaming buffer."""
        self.text_buffer += text

    def _update_stats_display(self, stats):
        if not hasattr(self, 'stats_labels'): return
        try:
            for k, v in stats.items():
                if k in self.stats_labels:
                    display_text = str(v)
                    if k == "Power" and isinstance(v, (int, float)):
                        display_text = f"{v:.1f}W"
                    self.stats_labels[k].config(text=display_text)
        except: pass


        if not SYSTEM_MONITOR_LOADED and self.stats_labels:
             if self.stats_labels.get("CPU") and self.stats_labels["CPU"].cget("text") == "N/A":
                 self.stats_labels["CPU"].config(text="No Libs")
                 self.stats_labels["RAM"].config(text="pip install")
                 if "GPU Util" in self.stats_labels:
                     self.stats_labels["GPU Util"].config(text="psutil nvidia_ml")

    def update_persona_display(self, val=None):
        if self.depth_slider is None: return
        raw_val = int(val) if val else self.active_persona_level
        
        # --- Level Mapping Logic ---
        is_secret = (self.max_persona_level >= 6)
        is_live = (self.live_agent_process and self.live_agent_process.poll() is None)
        
        if is_live and not is_secret and raw_val == 6:
            lvl = 7
        else:
            lvl = raw_val
            
        self.active_persona_level = lvl
        
        # --- Level 6/7 Auto-Hide Logic ---
        current_to = self.depth_slider.cget('to')
        
        if lvl < 6 and not is_live and current_to >= 6:
            self.depth_slider.config(to=5)
        
        if not is_live and current_to == 7:
            new_max = 6 if self.max_persona_level >= 6 else int(self.max_persona_level)
            self.depth_slider.config(to=max(new_max, 5))
        
        # --- Secret Lore Button Toggle & Auto-Hide ---
        if hasattr(self, 'lore_btn') and self.lore_btn is not None:
            if lvl == 6:
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
             
        idx = lvl if self.model else 0
        
        if self.chat_history is not None:
            self.chat_history.config(bg=CHAT_BG_COLORS.get(idx, THEME["widget_bg_color"]), 
                                     fg=CHAT_FG_COLORS.get(idx, THEME["fg_color"]))
        if self.user_input is not None:
            self.user_input.config(bg=THERMO_COLORS.get(idx, THEME["widget_bg_color"]), 
                                   fg=INPUT_FG_COLORS.get(idx, THEME["fg_color"]))
        
        if hasattr(self, 'persona_name_button') and self.persona_name_button is not None:
            if (self.live_agent_process and self.live_agent_process.poll() is None):
                btn_cmd = lambda l=lvl: self._live_persona_swap(l)
            else:
                btn_cmd = lambda l=lvl: self.model_swap(target_level=l)
            self.persona_name_button.config(text=name, 
                                            fg=THERMO_COLORS.get(lvl, THEME["electric_blue"]), 
                                            command=btn_cmd)
        
        if hasattr(self, 'persona_desc_label') and self.persona_desc_label is not None: 
            self.persona_desc_label.config(text=desc)
            
        self.depth_slider.config(bg=THERMO_COLORS.get(idx, THEME["widget_bg_color"]), 
                                 troughcolor=THEME["midnight_blue"], 
                                 activebackground="#7D0000")

    def _on_persona_label_click(self, e):
        self.state["persona_clicks"] += 1
        if self.state["persona_clicks"] >= 6: 
            self.state["persona_clicks"] = 0
            self._load_secret_model_event()

    def _load_secret_model_event(self, e=None):
        self._log_and_display("Engaging Worldbuilder...")
        self.max_persona_level = 6
        is_live = (self.live_agent_process and self.live_agent_process.poll() is None)
        self.depth_slider.config(to=7 if is_live else 6)
        self.depth_slider.set(6)
        self.update_persona_display(6)
        self.model_swap(target_level=6)

    def redirect_logs(self):
        sys.stdout = WidgetLogger(self.thought_log, "stdout")
        sys.stderr = FileAndWidgetLogger(self.error_log, self.error_log_file, "stderr")

    def toggle_live_agent(self):
        print("[BRIDGE] Toggling Live Agent state...")
        if self.live_agent_process and self.live_agent_process.poll() is None: 
            # --- DEACTIVATION: Kill Live + Bridge/Engine, auto-hide Level 7 ---
            self.live_agent_process.terminate()
            self.live_agent_process = None
            self._kill_live_engine()
            self.live_agent_button.config(text="Live")
            # Auto-hide Level 7
            if self.depth_slider.cget('to') == 7:
                new_max = 6 if self.max_persona_level >= 6 else int(self.max_persona_level)
                self.depth_slider.config(to=max(new_max, 5))
                if self.active_persona_level == 7:
                    self.depth_slider.set(int(self.max_persona_level))
                    self.active_persona_level = int(self.max_persona_level)
            self.update_persona_display(self.active_persona_level)
            self._log_and_display("SerenityLive deactivated.")
        else:
            p = os.path.join(self.script_dir, 'Live', 'serenity_live.py')
            if not os.path.exists(p): messagebox.showerror("Error", "live_agent.py not found."); return
            self.live_agent_button.config(text="Starting...")
            # Extend slider range (Position 6 maps to Lvl 7 if Secret is locked)
            new_to = 7 if self.max_persona_level >= 6 else 6
            self.depth_slider.config(to=new_to)
            self.depth_slider.set(new_to)
            self._live_persona_swap(7)
            threading.Thread(target=self._start_agent_thread, args=(p,), daemon=True).start()

    def _start_agent_thread(self, path):
        try: 
            env = os.environ.copy()
            env["SERENITY_SPAWNED_BY_UI"] = "1"
            
            # Using pythonw to prevent a background console from hanging or popping up
            py_exe = sys.executable.lower().replace("python.exe", "pythonw.exe") if "python.exe" in sys.executable.lower() else sys.executable
            self.live_agent_process = subprocess.Popen(
                [py_exe, path], 
                cwd=os.path.dirname(path), 
                env=env, 
                creationflags=0x08000000
            )

            self.root.after(0, lambda: self.live_agent_button.config(text="Agent Active") if self.live_agent_button else None)
            self.root.after(0, lambda: self._log_and_display("SerenityLive Active."))
        except Exception as e: 
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))

    def _start_t5_engine(self):
        """Spawn t5_server.py for Level 7."""
        print("[ENGINE] Spawning T5 Engine (Level 7)...")
        engine_script = os.path.join(self.script_dir, 'Live', 'Engine', 't5_server.py')
        if not os.path.exists(engine_script):
            self._log_and_display("Error: t5_server.py not found.")
            return
        env = os.environ.copy()
        live_params = self._read_live_params()
        env["SERENITY_CORE"] = live_params.get("active_core", "med")
        env["SERENITY_SPAWNED_BY_UI"] = "1"
        py_exe = sys.executable
        if "pythonw" in py_exe.lower():
            py_exe = py_exe.lower().replace("pythonw", "python")
        if not py_exe.lower().endswith(".exe") and os.name == "nt":
            py_exe += ".exe"
        try:
            subprocess.Popen(
                [py_exe, engine_script],
                cwd=os.path.join(self.script_dir, 'Live'),
                env=env,
                creationflags=0x08000000
            )
            self._log_and_display(f"Live Engine: t5_server.py booting ({env['SERENITY_CORE'].upper()})...")
        except Exception as e:
            self._log_and_display(f"Engine launch failed: {e}")

    def _kill_live_engine(self):
        """Kill t5_server.py engine on port 8001 to free VRAM."""
        print("[ENGINE] Termination signal sent to T5 Engine.")
        try:
            import requests as _req
            _req.post("http://127.0.0.1:8001/shutdown",
                      headers={"x-api-key": "serenity-alpha-core-77X"}, timeout=2)
        except: pass
        # Hard kill via port scan
        if SYSTEM_MONITOR_LOADED:
            import time as _time
            _time.sleep(1.0)
            for conn in psutil.net_connections():
                laddr = getattr(conn, 'laddr', None)
                if laddr and hasattr(laddr, 'port') and laddr.port == 8001:
                    pid = getattr(conn, 'pid', None)
                    if pid and pid != os.getpid():
                        try:
                            p = psutil.Process(pid)
                            p.terminate()
                            _time.sleep(0.5)
                            if p.is_running(): p.kill()
                        except: pass
        
        # Absolute kill via OS tools for port 8001 (Windows)
        if os.name == 'nt':
            try:
                cmd = 'for /f "tokens=5" %a in (\'netstat -aon ^| find ":8001" ^| find "LISTENING"\') do taskkill /f /pid %a'
                subprocess.call(cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            except: pass

    def _write_live_persona(self, level):
        """Write persona level to Live/System/params.json for serenity_live.py."""
        self._write_live_params({"persona_level": str(level)})

    def _write_live_params(self, updates):
        """Update Live/System/params.json with a dict of changes."""
        live_params_file = os.path.join(self.script_dir, 'Live', 'System', 'params.json')
        try:
            with open(live_params_file, 'r') as f:
                live_params = json.load(f)
        except:
            live_params = {}
        
        live_params.update(updates)
        
        try:
            os.makedirs(os.path.dirname(live_params_file), exist_ok=True)
            with open(live_params_file, 'w') as f:
                json.dump(live_params, f, indent=4)
        except Exception as e:
            print(f"[LIVE ENGINE] Failed to write params: {e}")

    def _read_live_params(self):
        """Read Live/System/params.json."""
        live_params_file = os.path.join(self.script_dir, 'Live', 'System', 'params.json')
        try:
            with open(live_params_file, 'r') as f:
                return json.load(f)
        except:
            return {"active_core": "med"}

    def initialize_app(self):
        self._log_and_display("System Ready. Select a persona to begin.")
        self.set_avatar_state("off")
        
        # Restore persistent sticky persona level and range
        if hasattr(self, 'depth_slider'):
            self.depth_slider.config(to=self.max_persona_level)
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
            if not (self.live_agent_process and self.live_agent_process.poll() is None):
                new_max = 6 if self.max_persona_level >= 6 else int(self.max_persona_level)
                self.depth_slider.config(to=max(new_max, 5))
                if self.active_persona_level == 7:
                    self.depth_slider.set(int(self.max_persona_level))
                    self.active_persona_level = int(self.max_persona_level)
        
        self.clear_chat_ui()
        self.set_ui_state(model_loaded=False)
        self.update_persona_display(self.active_persona_level) 
        self._reset_multimodal_ui()
        self.set_avatar_state("off") 
        self._log_and_display("All models offloaded. VRAM Cleared.")

    def get_history_path(self): 
        return os.path.join(self.dirs["History"], f"{os.path.splitext(os.path.basename(self.model_path))[0]}_lvl{self.active_persona_level}.history.jsonz") if self.model_path else None

    def save_history(self):
        if not (path := self.get_history_path()) or not self.messages: return
        try:
            with open(path, 'wb') as f:
                f.write(zlib.compress(json.dumps(self.messages).encode('utf-8')))
        except Exception as e:
            print(f"History save error: {e}", file=sys.stderr)

    def load_history(self):
        self.messages = []
        self.clear_chat_ui()
        
        self.past_history_view.config(state='normal')
        self.past_history_view.delete('1.0', tk.END)
        self.past_history_view.config(state='disabled')
        
        if (path := self.get_history_path()) and os.path.exists(path):
            try:
                with open(path, 'rb') as f: 
                    self.messages = json.loads(zlib.decompress(f.read()).decode('utf-8'))
                
                if self.messages and self.messages[-1].get('role') == 'user': 
                    self.messages.pop()
                
                self.past_history_view.config(state='normal')
                for m in self.messages: 
                    who = "You" if m['role'] == 'user' else "Serenity"
                    tag = "user" if m['role'] == 'user' else "ai"
                    entry = f"{who}: {m['content']}\n{'-'*50}\n\n"
                    self.past_history_view.insert(tk.END, entry, (tag,))
                
                self.past_history_view.config(state='disabled')
                self.past_history_view.yview_moveto(1.0) 
                self._log_and_display("Archive Updated.")
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
        """Loads tier-specific inference parameters if available, otherwise falls back to system defaults."""
        # Try tier-specific params first, then global params
        params_files = [f"params_{tier}.json", "params.json"]
        loaded = False
        
        for p_file in params_files:
            path = os.path.join(self.dirs["System"], p_file)
            if os.path.exists(path):
                try: 
                    with open(path) as f: 
                        self.params = json.load(f)
                        print(f"[APEX] Loaded inference overrides from: {p_file}")
                        loaded = True
                        break
                except Exception as e:
                    print(f"Warning: Failed to load {p_file}: {e}")
        
        if not loaded:
            self.params = {}
            print(f"[APEX] No inference overrides found for {tier}. Using engine defaults.")

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
                    img = Image.open(p).resize((w, h), Image.Resampling.LANCZOS)
                    self.avatar_states[state] = ImageTk.PhotoImage(img)
                    #print(f"Loaded: {fname}")
            except Exception as e: print(f"Error loading {fname}: {e}")
            
        if not self.avatar_states:
             messagebox.showinfo("Assets Missing", "No avatar images found in Media folder.\nUsing text fallback.")
             
        self.set_avatar_state("off")

    def set_avatar_state(self, state):
        if not self.right_panel: return
        self.state["avatar_current"] = state 
        
        # --- MISSION: Force Cecilia for Level 6 ---
        if self.active_persona_level == 6:
            fname = "Cecilia_01.png"
            p = os.path.join(self.dirs["Media"], fname)
            if os.path.exists(p):
                try:
                    # Clean up idle timers to prevent state flickering
                    if getattr(self, 'idle_timer_id', None) is not None: 
                        self.root.after_cancel(self.idle_timer_id)
                        self.idle_timer_id = None
                    
                    img = Image.open(p).resize((350, 350), Image.Resampling.LANCZOS)
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
            "deep_think": "serenity_idea.png",
            "subdued": "subdued_serenity.png"
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
                    img = Image.open(p).resize((350, 350), Image.Resampling.LANCZOS)
                    self.tmp_img = ImageTk.PhotoImage(img)
                    self.right_panel.itemconfig(self.avatar_image_item, state='normal', image=self.tmp_img)
                    self.right_panel.itemconfig(self.avatar_text_item, state='hidden')
                except Exception as e:
                    print(f"Avatar load error: {e}")

        if state == "listening": 
            if getattr(self, "idle_timer_id", None) is not None:
                self.root.after_cancel(self.idle_timer_id)
                self.idle_timer_id = None
            self.idle_timer_id = self.root.after(5000, lambda *args: self._set_persona_idle_state())

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
                    canvas_r.itemconfigure(log_win, width=w, height=h//2)
                
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
        
        for key in ['model_paths', 'gpu_layer_config', 'context_size_config', 'temp_config', 
                    'top_p_config', 'min_p_config', 'repeat_penalty_config', 'frequency_penalty_config',
                    'presence_penalty_config', 'stop_strings_config', 'n_batch_config', 'top_k_config']:
            disk_data = self.config.get(key, {})
            getattr(self, key).update(disk_data)
        
        self.sash_pos = self.config.get('sash_pos', -1)
        
        self.state["deep_cook_behavior"] = self.config.get('deep_thought_behavior', "oneshot")
        self.state["virtual_vram"] = self.config.get('virtual_vram', 0)
        self.active_persona_level = self.config.get('active_persona_level', 3)
        self.max_persona_level = self.config.get('max_persona_level', 5)
        
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
            self.config["speculative_drafting"] = True

        if "custom_templates" not in self.config or not self.config["custom_templates"]:
            self.config["custom_templates"] = {
                "T1": {"name": "Thinking (Gen)", "temp": 1.0, "top_p": 0.95, "min_p": 0.0, "rep": 1.0, "pres": 1.5, "top_k": 20, "batch": 512, "layers": -1, 
                    "ctx": 32768, "stop": "###,<|endoftext|>,<|im_end|>,<|turn>,<turn|>,</think>,<eos>"},
                "T2": {"name": "Thinking (Code)", "temp": 0.6, "top_p": 0.95, "min_p": 0.0, "rep": 1.0, "pres": 0.0, "top_k": 20, "batch": 512, "layers": -1, 
                    "ctx": 32768, "stop": "###,<|endoftext|>,<|im_end|>,<|turn>,<turn|>,</think>,<eos>"},
                "T3": {"name": "Vision (Best)", "temp": 0.1, "top_p": 0.9, "min_p": 0.0, "rep": 1.1, "pres": 0.0, "top_k": 64, "batch": 512, "layers": -1, 
                    "ctx": 8192, "stop": "###,<|endoftext|>,<|im_end|>,<|turn>,<turn|>,</think>,<eos>"}
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
        
        return self.config

    def _load_dmn_backbone(self):
        p = os.path.join(self.dirs["System"], "dmn_backbone.json")
        try:
            if os.path.exists(p):
                with open(p, 'r') as f:
                    self.state["dmn_backbone"] = json.load(f)
            else:
                self.state["dmn_backbone"] = {}
        except:
            self.state["dmn_backbone"] = {}

    def _save_dmn_backbone(self):
        p = os.path.join(self.dirs["System"], "dmn_backbone.json")
        try:
            with open(p, 'w') as f:
                json.dump(self.state["dmn_backbone"], f, indent=4)
        except: pass

    def save_config(self):
        data = {
            'main_window': self.root.winfo_geometry(), 'model_paths': self.model_paths,
            'gpu_layer_config': self.gpu_layer_config, 'context_size_config': self.context_size_config,
            'temp_config': self.temp_config,
            'top_p_config': self.top_p_config, 'min_p_config': self.min_p_config,
            'repeat_penalty_config': self.repeat_penalty_config, 'frequency_penalty_config': self.frequency_penalty_config,
            'presence_penalty_config': self.presence_penalty_config,
            'stop_strings_config': self.stop_strings_config, 'n_batch_config': self.n_batch_config,
            'top_k_config': self.top_k_config,
            'sash_pos': self.paned.sash_coord(0)[0] if hasattr(self, 'paned') else -1,
            'deep_thought_behavior': self.state["deep_cook_behavior"],
            'virtual_vram': self.state["virtual_vram"],
            'active_persona_level': self.active_persona_level,
            'max_persona_level': self.depth_slider.cget('to'),
            'synthesis_in_tactical_mode': self.config.get("synthesis_in_tactical_mode", False),
            'show_rgb_button': self.config.get("show_rgb_button", True),
            'sub_chunk_size': getattr(self, 'sub_chunk_size', 8),
            'custom_templates': self.config.get('custom_templates', {}),
            'streaming_mode': self.state.get("streaming_mode", "Buffered"),
            'max_token_ratio': self.config.get("max_token_ratio", 4),
            'auto_vram_offload': self.config.get("auto_vram_offload", False),
            'speculative_drafting': self.config.get("speculative_drafting", True)
        }
        with open(self.config_file, 'w') as f: json.dump(data, f, indent=4)

    def _calculate_active_logit_bias(self, context_data):
        """Analyzes the recent message history or context prompt to identify key thematic keywords and builds a logit_bias dict to reinforce them."""
        if not hasattr(self, 'model') or not self.model:
            return {}
        
        # 1. Gather all content text
        text_corpus = ""
        if isinstance(context_data, list):
            for m in context_data:
                if isinstance(m, dict):
                    content = m.get("content", "")
                    if isinstance(content, str):
                        text_corpus += " " + content.lower()
        elif isinstance(context_data, str):
            text_corpus = context_data.lower()
            
        if not text_corpus:
            return {}
        
        # 2. Extract words and filter stop words/short words
        words = re.findall(r'\b[a-z]{4,15}\b', text_corpus)
        stopwords = {
            "that", "with", "this", "they", "from", "have", "would", "their", "what",
            "about", "there", "which", "when", "your", "them", "some", "other", "could",
            "these", "then", "into", "than", "only", "such", "more", "even", "most", "also",
            "just", "make", "over", "your", "here", "were", "been", "hello", "would", "could",
            "should", "will", "would", "about", "their", "there", "these", "those"
        }
        
        filtered_words = [w for w in words if w not in stopwords]
        
        # Count frequencies
        from collections import Counter
        counts = Counter(filtered_words)
        
        logit_bias = {}
        # Take up to the top 12 keywords that appear at least twice (or once for short context)
        min_occurrence = 2 if isinstance(context_data, list) else 1
        for word, count in counts.most_common(12):
            if count >= min_occurrence:
                try:
                    # Tokenize the word
                    tokens = self.model.tokenize(word.encode('utf-8'), add_bos=False)
                    bias_val = min(1.2, 0.3 + (count * 0.15))
                    for tok in tokens:
                        if tok > 100: # Skip control/special/very common tokens
                            logit_bias[tok] = bias_val
                            logit_bias[str(tok)] = bias_val
                except Exception as e:
                    pass
                    
        if logit_bias:
            print(f"[CONTEXT RECURSION] Active Logit Bias tracked keywords: {[w for w, c in counts.most_common(12) if c >= min_occurrence]}")
            
        return logit_bias

    def _get_inference_params(self, temp_messages=None):
        """Builds the parameter dictionary for llama-cpp-python inference."""
        print(f"[INFERENCE] Retrieving parameters for tier: {self.current_model_tier}")
        
        # Sampler Refresh: Include mandatory stop sequences to prevent run-on outputs
        stops = [s.strip() for s in self.stop_strings_config.get(self.current_model_tier, "").split(",") if s.strip()]
        if "<end_of_turn>" not in stops: stops.append("<end_of_turn>")
        if "<turn|>" not in stops: stops.append("<turn|>")
        if "<|end_of_turn|>" not in stops: stops.append("<|end_of_turn|>") # Fallback for old models
        if "<|/>" not in stops: stops.append("<|/>")
        if "<turn/>" not in stops: stops.append("<turn/>")
        
        # Baseline defaults
        inf_params = {
            "temperature": self.temp_config.get(self.current_model_tier, 1.0), # Gemma-4 Official
            "top_p": self.top_p_config.get(self.current_model_tier, 0.95),
            "min_p": self.min_p_config.get(self.current_model_tier, 0.05),
            "repeat_penalty": self.repeat_penalty_config.get(self.current_model_tier, 1.0), # Gemma-4 recommendation
            "frequency_penalty": self.frequency_penalty_config.get(self.current_model_tier, 0.0),
            "presence_penalty": self.presence_penalty_config.get(self.current_model_tier, 0.0),
            "stop": stops,
            #"add_bos": True, # Ensure official BOS (token 2) is always prepended #TODO: remove if dead code
            "top_k": self.top_k_config.get(self.current_model_tier, 64), # Gemma-4 Official
        }
        
        # Support CFG parameters if configured in self.config
        if self.config.get("cfg_scale"):
            inf_params["cfg_scale"] = float(self.config.get("cfg_scale"))
        if self.config.get("cfg_negative_prompt"):
            inf_params["cfg_negative_prompt"] = self.config.get("cfg_negative_prompt")
            
        # Support active logit bias tracking for context recursion
        if temp_messages:
            lb = self._calculate_active_logit_bias(temp_messages)
            if lb:
                inf_params["logit_bias"] = lb
        
        # Dynamic Max Tokens (Context Headroom Management)
        ctx = self.context_size_config.get(self.current_model_tier, 4096)
        ratio = int(self.config.get("max_token_ratio", 4))
        calculated_max = ctx // ratio
        
        if self.current_model_tier and self.current_model_tier.startswith("vision_"):
            inf_params["max_tokens"] = max(512, calculated_max)
        else:
            inf_params["max_tokens"] = max(256, calculated_max)

        
        # Overlay with params.json values if loaded
        if hasattr(self, "params") and self.params:
            override_params = dict(self.params)
            
            # Special handling for stop strings: MERGE them rather than overwrite
            if "stop" in override_params and isinstance(override_params["stop"], list):
                inf_params["stop"].extend([s for s in override_params["stop"] if s not in inf_params["stop"]])
                del override_params["stop"]
                
            inf_params.update(override_params)
            
        # --- Sampler Hygiene: Filter unsupported params for llama-cpp-python stability ---
        supported_keys = {
            "temperature", "top_p", "min_p", "repeat_penalty", "frequency_penalty", 
            "presence_penalty", "top_k", "max_tokens", "stop", "stream", 
            "grammar", "logit_bias", "logprobs", "typical_p", "tfs_z", 
            "mirostat_mode", "mirostat_tau", "mirostat_eta", "model", "messages",
            "seed", "echo", "repeat_last_n", "cfg_scale", "cfg_negative_prompt"
        }
        
        filtered_params = {k: v for k, v in inf_params.items() if k in supported_keys}
        
        dropped = set(inf_params.keys()) - set(filtered_params.keys())
        if dropped:
            print(f"[APEX] Sampler Hygiene: Dropping unsupported parameters: {dropped}")
            
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
        """
        Attempts to automatically calculate optimal GPU layers based on VRAM,
        model complexity (MoE vs Dense), and non-linear KV-cache requirements.
        [RLHF]: Incorporates stability feedback from historical session loads.
        """
        self._log_and_display("Analyzing hardware/model complexity (RLHF Alpha)...")
        
        # 1. Determine VRAM source
        manual_vram_mb = self.state.get("virtual_vram", 0)
        vram_gb = None
        if manual_vram_mb > 0:
            vram_gb = manual_vram_mb / 1024
            self._log_and_display(f"Using Manual VRAM Target: {vram_gb:.2f}GB")
        elif SYSTEM_MONITOR_LOADED and self.gpu_handle:
            try:
                mem = nvidia_ml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                vram_gb = mem.total / 1024**3
            except Exception: pass

        if vram_gb is None:
             self._log_and_display("Hardware detection offline. Defaulting to CPU.")
             vram_gb = 0 # Baseline for logic fallback

        # 2. RLHF Stability Feedback
        rlhf_penalty = 0
        rlhf_path = os.path.join(self.dirs["System"], "rlhf_stability.json")
        if os.path.exists(rlhf_path):
            try:
                with open(rlhf_path, 'r') as f:
                    rlhf_data = json.load(f)
                    rlhf_penalty = rlhf_data.get("vram_global_penalty", 0)
            except: pass

        # 3. Model-Aware Tier Scaling
        tiers = ["fast", "search", "low", "med", "high", "secret", "deep_cook", 
                 "vision_video", "vision_video_deep", "vision_multimodal"]
        
        level_map = {
            "fast": 1, "search": 2, "low": 3, "med": 4, 
            "high": 5, "secret": 6, "deep_cook": 5
        }
        
        recommendations = {}
        for tier in tiers:
            path = self.model_paths.get(tier, "").lower()
            
            # APEX GUARD: Force everything to -1 for E-series
            if any(x in path for x in ["e2b", "e4b", "tiny"]):
                recommendations[tier] = -1
                continue

            # MOE/Large Model GUARD: Cap at 14 layers for 6GB stability (Verified limit)
            if vram_gb < 7 and any(x in path for x in ["26b", "31b", "moe"]):
                    recommendations[tier] = 14
                    continue
            
            # 4. Standard Linear Calculation (Layers per GB)
            # 3050 LP (6GB) -> ~4.5 layers per GB for Q8 is a safe base
            base_ratio = 4.5
            calc = int((vram_gb - (rlhf_penalty / 1024)) * base_ratio)
            
            # 5. Cap to sensible limits
            final = max(0, min(64, calc))
            
            recommendations[tier] = final

        self._log_and_display(f"Auto-detection complete. Applied {rlhf_penalty} layers RLHF safety margin.")
        return recommendations

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
        try:
            win = tk.Toplevel(self.root)
            win.title("Model Settings")
            if self.icon_path:
                try: win.iconbitmap(self.icon_path)
                except: pass
            win.geometry(self.config.get("settings_window_geometry", "800x950"))
            win.config(bg=THEME["bg_color"])
            win.attributes("-topmost", False) # Removed per user request
            
            # --- Fixed Top Action Bar ---
            btn_frame = tk.Frame(win, bg=THEME["bg_color"], pady=5)
            btn_frame.pack(side=tk.TOP, fill=tk.X, padx=10)
            
            # --- Scrollable Container ---
            container = tk.Frame(win, bg=THEME["bg_color"])
            container.pack(fill=tk.BOTH, expand=True)
            
            canvas = tk.Canvas(container, bg=THEME["bg_color"], highlightthickness=0)
            v_scroll = tk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
            scrollable_frame = tk.Frame(canvas, bg=THEME["bg_color"])
            
            scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
            canvas_win = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            
            def _on_canvas_resize(event):
                canvas.itemconfig(canvas_win, width=event.width)
            canvas.bind("<Configure>", _on_canvas_resize)
            
            canvas.configure(yscrollcommand=v_scroll.set)
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            
            def _on_mousewheel(event):
                if canvas.winfo_exists():
                    canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            win.bind_all("<MouseWheel>", _on_mousewheel)
            
            def on_closing():
                try:
                    win.unbind_all("<MouseWheel>")
                    print("[UI] Settings listener detached.")
                except: pass
                win.destroy()
            win.protocol("WM_DELETE_WINDOW", on_closing)
    
            main = scrollable_frame 
            
            # --- TOP HEADER SETTINGS ---
            header_settings = tk.Frame(main, bg=THEME["bg_color"])
            header_settings.pack(fill=tk.X, padx=10, pady=5)
            
            center_header = tk.Frame(header_settings, bg=THEME["bg_color"])
            center_header.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
            
            left_header = tk.Frame(header_settings, bg=THEME["bg_color"])
            left_header.pack(side=tk.LEFT, fill=tk.Y)
            
            tk.Label(left_header, text="Deep Cook:", bg=THEME["bg_color"], fg=THEME["electric_blue"]).pack(anchor="w")
            v_behavior = tk.StringVar(value=self.state.get("deep_cook_behavior", "oneshot"))
            behavior_frame = tk.Frame(left_header, bg=THEME["bg_color"])
            behavior_frame.pack(anchor="w", padx=10)
            for val, txt in [("oneshot", "One-Shot"), ("toggle", "Toggle Mode")]:
                 tk.Radiobutton(behavior_frame, text=txt, variable=v_behavior, value=val, bg=THEME["bg_color"], fg=THEME["fg_color"], 
                                selectcolor=THEME["widget_bg_color"]).pack(side=tk.LEFT, padx=5)
            
            vram_frame = tk.Frame(left_header, bg=THEME["bg_color"])
            vram_frame.pack(anchor="w", pady=(5, 0))
            tk.Label(vram_frame, text="VRAM (GB):", bg=THEME["bg_color"], fg=THEME["electric_blue"]).pack(side=tk.LEFT)
            vram_ent = tk.Entry(vram_frame, bg=THEME["widget_bg_color"], fg=THEME["fg_color"], width=6)
            vram_ent.insert(0, f"{self.state.get('virtual_vram', 0)/1024:g}" if self.state.get('virtual_vram', 0) > 0 else "0")
            vram_ent.pack(side=tk.LEFT, padx=5)
            
            m_frame = tk.Frame(left_header, bg=THEME["bg_color"])
            m_frame.pack(anchor="w", pady=(10, 0))
            tk.Label(m_frame, text="Engine:", bg=THEME["bg_color"], fg=THEME["electric_blue"]).pack(side=tk.LEFT)
            mm_engine_var = tk.StringVar(value=self.state.get("multimodal_engine", "Internal"))
            for opt in ["Internal", "Live"]:
                tk.Radiobutton(m_frame, text=opt, variable=mm_engine_var, value=opt,
                               bg=THEME["bg_color"], fg=THEME["fg_color"], selectcolor=THEME["widget_bg_color"]).pack(side=tk.LEFT, padx=2)
    
            if self._is_rgb_supported():
                toggle_frame = tk.Frame(left_header, bg=THEME["bg_color"])
                toggle_frame.pack(anchor="w", pady=(10, 0))
        
                show_rgb_var = tk.BooleanVar(value=self.config.get("show_rgb_button", True))
                def _toggle_rgb():
                    self.config["show_rgb_button"] = show_rgb_var.get()
                    if show_rgb_var.get():
                        if hasattr(self, 'rgb_button') and self.rgb_button.winfo_exists():
                            self.rgb_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, before=self.send_button)
                    else:
                        if hasattr(self, 'rgb_button') and self.rgb_button.winfo_exists():
                            self.rgb_button.pack_forget()
                    self.save_config()
        
                tk.Checkbutton(toggle_frame, text="Show RGB Button", variable=show_rgb_var, command=_toggle_rgb,
                               bg=THEME["bg_color"], fg=THEME["electric_blue"], selectcolor=THEME["widget_bg_color"]).pack(anchor="w")
    
            tk.Label(left_header, text="Default v1 Model:", bg=THEME["bg_color"], fg=THEME["electric_blue"]).pack(anchor="w", pady=(5, 0))
            live_core_var = tk.StringVar(value=self.config.get("live_core_selection", "gemma-4-26b-a4b"))
            live_core_frame = tk.Frame(left_header, bg=THEME["bg_color"])
            live_core_frame.pack(anchor="w", padx=10)
            for opt in ["g4-26b", "cg7b", "q3.6", "cg2b"]:
                val_map = {"g4-26b": "gemma-4-26b-a4b", "cg7b": "codegemma-7b-it", "q3.6": "qwen3.6-35b", "cg2b": "codegemma-2b"}
                tk.Radiobutton(live_core_frame, text=opt, variable=live_core_var, value=val_map[opt],
                               bg=THEME["bg_color"], fg=THEME["fg_color"], selectcolor=THEME["widget_bg_color"]).pack(side=tk.LEFT, padx=1)
    
            tk.Label(center_header, text="Templating Engine:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=("Open Sans", 9, "bold")).pack(anchor="n")
            self.template_mode = tk.StringVar(value="modify")
            self.active_template = tk.StringVar(value="")
            
            t_action_frame = tk.Frame(center_header, bg=THEME["bg_color"])
            t_action_frame.pack(anchor="n", pady=2)
            for val, txt in [("save", "Save"), ("write", "Write"), ("modify", "Modify")]:
                tk.Radiobutton(t_action_frame, text=txt, variable=self.template_mode, value=val, indicatoron=0, 
                               bg=THEME["widget_bg_color"], fg=THEME["fg_color"], selectcolor=THEME["button_active_color"]).pack(side=tk.LEFT, padx=2)
    
            t_grid = tk.Frame(center_header, bg=THEME["bg_color"])
            t_grid.pack(anchor="n", pady=5)
            
            template_buttons = []
            for i in range(2):
                for j in range(4):
                    slot_id = f"T{(i*4)+j+1}"
                    t_name = self.config.get("custom_templates", {}).get(slot_id, {}).get("name", slot_id)
                    b = tk.Radiobutton(t_grid, text=t_name, variable=self.active_template, value=slot_id, indicatoron=0, width=12, 
                                       bg=THEME["widget_bg_color"], fg=THEME["electric_blue"], selectcolor=THEME["button_active_color"])
                    b.grid(row=i, column=j, padx=2, pady=2)
                    b.slot_id = slot_id
                    template_buttons.append(b)
    
            def _on_template_select(*args):
                mode = self.template_mode.get()
                t_id = self.active_template.get()
                if not t_id: return
                if mode == "modify":
                    t_win = tk.Toplevel(win)
                    t_win.title(f"Modify {t_id}")
                    t_win.geometry("300x480")
                    t_win.config(bg=THEME["bg_color"])
                    t_win.attributes("-topmost", False)
                    current = self.config.get("custom_templates", {}).get(t_id, {})
                    tk.Label(t_win, text="Name:", bg=THEME["bg_color"], fg=THEME["electric_blue"]).pack(anchor="w", padx=10, pady=(10,0))
                    name_ent = tk.Entry(t_win, bg=THEME["widget_bg_color"], fg=THEME["fg_color"])
                    name_ent.insert(0, current.get("name", t_id))
                    name_ent.pack(fill=tk.X, padx=10)
                    param_list = [("Temp:", "temp", 0.8), ("Top P:", "top_p", 0.9), ("Min P:", "min_p", 0.05), ("Rep Pen:", "rep", 1.1), ("Pres Pen:", "pres", 0.0),
                    ("Freq Pen:", "freq", 0.0), ("Top K:", "top_k", 40), ("Batch:", "batch", 512), ("Layers:", "layers", -1), ("Ctx Size:", "ctx", 8192)]
                    fields = {}
                    grid_f = tk.Frame(t_win, bg=THEME["bg_color"])
                    grid_f.pack(fill=tk.X, padx=10, pady=5)
                    for idx, (label, key, default) in enumerate(param_list):
                        r, c = divmod(idx, 2)
                        c *= 2
                        tk.Label(grid_f, text=label, bg=THEME["bg_color"], fg=THEME["electric_blue"], width=9, anchor="w").grid(row=r, column=c, padx=(0,2), pady=2)
                        e = tk.Entry(grid_f, bg=THEME["widget_bg_color"], fg=THEME["fg_color"], width=8)
                        e.insert(0, str(current.get(key, default)))
                        e.grid(row=r, column=c+1, padx=(0,10), pady=2)
                        fields[key] = e
                    tk.Label(t_win, text="Stop Tokens (comma sep):", bg=THEME["bg_color"], fg=THEME["electric_blue"]).pack(anchor="w", padx=10)
                    stop_ent = tk.Entry(t_win, bg=THEME["widget_bg_color"], fg=THEME["fg_color"])
                    stop_str = current.get("stop", "")
                    if isinstance(stop_str, list): stop_str = ", ".join(stop_str)
                    stop_ent.insert(0, stop_str)
                    stop_ent.pack(fill=tk.X, padx=10)
                    def _save_mod():
                        stops = [s.strip() for s in stop_ent.get().split(',') if s.strip()]
                        t_data = {"name": name_ent.get(), "stop": ", ".join(stops)}
                        for k, e in fields.items():
                            try: t_data[k] = float(e.get()) if '.' in e.get() else int(e.get())
                            except: t_data[k] = current.get(k, 0)
                        if "custom_templates" not in self.config: self.config["custom_templates"] = {}
                        self.config["custom_templates"][t_id] = t_data
                        for btn in template_buttons:
                            if btn.slot_id == t_id: btn.config(text=t_data["name"])
                        self.save_config()
                        t_win.destroy()
                    tk.Button(t_win, text="Save & Close", command=_save_mod, bg=THEME["button_active_color"], fg=THEME["fg_color"]).pack(pady=15)
            self.active_template.trace_add("write", _on_template_select)
    
            right_header = tk.Frame(header_settings, bg=THEME["bg_color"])
            right_header.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(20, 0))
            tk.Label(right_header, text="Video Processing Sub-Chunk Size:", bg=THEME["bg_color"], fg=THEME["electric_blue"]).pack(anchor="w")
            sc_frame = tk.Frame(right_header, bg=THEME["bg_color"])
            sc_frame.pack(fill=tk.X, padx=5)
            sc_val = tk.IntVar(value=getattr(self, 'sub_chunk_size', 8))
            tk.Scale(sc_frame, from_=1, to=128, orient=tk.HORIZONTAL, variable=sc_val, 
                     bg=THEME["bg_color"], fg=THEME["fg_color"], highlightthickness=0, resolution=1).pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            def _reset_sc(): sc_val.set(8)
            tk.Button(sc_frame, text="Reset", command=_reset_sc).pack(side=tk.RIGHT, padx=5)
    
            ctx_ref_frame = tk.Frame(right_header, bg=THEME["bg_color"])
            ctx_ref_frame.pack(fill=tk.X, pady=(10, 0))
            tk.Label(ctx_ref_frame, text="Reference Context Windows:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=("Open Sans", 9, "bold")).pack(anchor="w")
            ctx_str = " | ".join([f"Lvl {k}: {v//1024}k" for k, v in CONTEXT_SIZE_MAP.items()])
            tk.Label(ctx_ref_frame, text=ctx_str, bg=THEME["bg_color"], fg=THEME["fg_color"], font=("Consolas", 8)).pack(anchor="w")
            
            kv_map_frame = tk.Frame(right_header, bg=THEME["bg_color"])
            kv_map_frame.pack(fill=tk.X, pady=(5, 0))
            tk.Label(kv_map_frame, text="KV Cache Translator Map:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=("Open Sans", 9, "bold")).pack(anchor="w")
            tk.Label(kv_map_frame, text="f16=0 | q8_0=1 | q4_0=2 | q4_1=3 | q5_0=6", bg=THEME["bg_color"], fg=THEME["fg_color"], font=("Consolas", 8)).pack(anchor="w")
    
            labels, ents, ctx_ents, n_batch_ents, temp_ents, top_p_ents, min_p_ents, top_k_ents, rep_ents, freq_ents, pres_ents, stop_ents = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
            
            def _on_tier_box_click(tier_name):
                mode = self.template_mode.get()
                t_id = self.active_template.get()
                if not t_id: return
                if mode == "save":
                    t_data = self.config.get("custom_templates", {}).get(t_id, {})
                    t_data["name"] = t_data.get("name", t_id)
                    try: t_data["temp"] = float(temp_ents[tier_name].get())
                    except: pass
                    try: t_data["top_p"] = float(top_p_ents[tier_name].get())
                    except: pass
                    try: t_data["min_p"] = float(min_p_ents[tier_name].get())
                    except: pass
                    try: t_data["rep"] = float(rep_ents[tier_name].get())
                    except: pass
                    try: t_data["pres"] = float(pres_ents[tier_name].get())
                    except: pass
                    try: t_data["freq"] = float(freq_ents[tier_name].get())
                    except: pass
                    try: t_data["top_k"] = int(top_k_ents[tier_name].get())
                    except: pass
                    try: t_data["batch"] = int(n_batch_ents[tier_name].get())
                    except: pass
                    try: t_data["layers"] = int(ents[tier_name].get())
                    except: pass
                    try: t_data["ctx"] = int(ctx_ents[tier_name].get())
                    except: pass
                    try: t_data["stop"] = stop_ents[tier_name].get()
                    except: pass
                    if "custom_templates" not in self.config: self.config["custom_templates"] = {}
                    self.config["custom_templates"][t_id] = t_data
                    self.save_config()
                    messagebox.showinfo("Templating", f"Saved {tier_name.upper()} settings to {t_data['name']}!")
                elif mode == "write":
                    t_data = self.config.get("custom_templates", {}).get(t_id, {})
                    if not t_data: return
                    for k, d in [("temp", temp_ents), ("top_p", top_p_ents), ("min_p", min_p_ents), ("rep", rep_ents), ("pres", pres_ents), ("freq", freq_ents), ("top_k", top_k_ents), 
                    ("batch", n_batch_ents), ("layers", ents), ("ctx", ctx_ents), ("stop", stop_ents)]:
                        if k in t_data: 
                            d[tier_name].delete(0, tk.END)
                            d[tier_name].insert(0, str(t_data[k]))
                    messagebox.showinfo("Templating", f"Applied {t_data['name']} to {tier_name.upper()}!")
    
            def _create_tier_block(parent, tier_name, row=0, col=0, is_vision=False):
                key = f"vision_{tier_name}" if is_vision else tier_name
                lvl_map = {"fast": "1", "search": "2", "low": "3", "med": "4", "high": "5", "secret": "6"}
                title_suffix = f" (Lvl {lvl_map[tier_name]})" if tier_name in lvl_map else ""
                lf = tk.LabelFrame(parent, text=f"Engine: {tier_name.upper()}{title_suffix}", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=("Open Sans", 10, "bold"), pady=5)
                lf.grid(row=row, column=col, sticky="nsew", padx=10, pady=5)
                
                def _bind_click(w):
                    w.bind("<Button-1>", lambda e: _on_tier_box_click(key), add="+")
                    for c in w.winfo_children(): _bind_click(c)
                
                r1 = tk.Frame(lf, bg=THEME["bg_color"]); r1.pack(fill=tk.X, padx=5)
                tk.Button(r1, text="Set Path", command=lambda t=key: self._set_path(t, labels, win)).pack(side=tk.LEFT)
                labels[key] = tk.Label(r1, text=os.path.basename(self.model_paths.get(key, "") or "Not Set"), bg=THEME["bg_color"], fg=THEME["fg_color"], font=("Open Sans", 8))
                labels[key].pack(side=tk.LEFT, padx=5)
                
                r1b = tk.Frame(lf, bg=THEME["bg_color"]); r1b.pack(fill=tk.X, padx=5, pady=2)
                tk.Label(r1b, text="Layers:", bg=THEME["bg_color"], fg=THEME["fg_color"]).pack(side=tk.LEFT)
                ents[key] = tk.Entry(r1b, width=4); ents[key].insert(0, str(self.gpu_layer_config.get(key, -1))); ents[key].pack(side=tk.LEFT, padx=2)
                tk.Label(r1b, text="Ctx:", bg=THEME["bg_color"], fg=THEME["fg_color"]).pack(side=tk.LEFT, padx=(5, 0))
                ctx_ents[key] = tk.Entry(r1b, width=6); ctx_ents[key].insert(0, str(self.context_size_config.get(key, 4096))); ctx_ents[key].pack(side=tk.LEFT, padx=2)
                tk.Label(r1b, text="Batch:", bg=THEME["bg_color"], fg=THEME["fg_color"]).pack(side=tk.LEFT, padx=(5, 0))
                n_batch_ents[key] = tk.Entry(r1b, width=5); n_batch_ents[key].insert(0, str(self.n_batch_config.get(key, 512))); n_batch_ents[key].pack(side=tk.LEFT, padx=2)
    
                r2 = tk.Frame(lf, bg=THEME["bg_color"]); r2.pack(fill=tk.X, padx=5)
                for l, d, c, df in [("Temp", temp_ents, self.temp_config, 0.8), ("Top-P", top_p_ents, self.top_p_config, 0.95), ("Min-P", min_p_ents, self.min_p_config, 0.05), ("Top-K", top_k_ents, self.top_k_config, 40)]:
                    tk.Label(r2, text=f"{l}:", bg=THEME["bg_color"], fg=THEME["fg_color"], font=("Open Sans", 8)).pack(side=tk.LEFT, padx=(2, 0))
                    d[key] = tk.Entry(r2, width=5); d[key].insert(0, f"{c.get(key, df):g}"); d[key].pack(side=tk.LEFT, padx=2)
                    
                r2b = tk.Frame(lf, bg=THEME["bg_color"]); r2b.pack(fill=tk.X, padx=5)
                for l, d, c, df in [("Rep", rep_ents, self.repeat_penalty_config, 1.1), ("Freq", freq_ents, self.frequency_penalty_config, 0.0), ("Pres", pres_ents, self.presence_penalty_config, 0.0)]:
                    tk.Label(r2b, text=f"{l}:", bg=THEME["bg_color"], fg=THEME["fg_color"], font=("Open Sans", 8)).pack(side=tk.LEFT, padx=(2, 0))
                    d[key] = tk.Entry(r2b, width=5); d[key].insert(0, f"{c.get(key, df):g}"); d[key].pack(side=tk.LEFT, padx=2)
    
                r3 = tk.Frame(lf, bg=THEME["bg_color"]); r3.pack(fill=tk.X, padx=5)
                tk.Label(r3, text="Stop:", bg=THEME["bg_color"], fg=THEME["fg_color"], font=("Open Sans", 8)).pack(side=tk.LEFT)
                stop_ents[key] = tk.Entry(r3, font=("Open Sans", 8), bg=THEME["widget_bg_color"], fg=THEME["fg_color"], width=50)
                stop_ents[key].insert(0, self.stop_strings_config.get(key, "")); stop_ents[key].pack(side=tk.LEFT, padx=5)
                
                if is_vision:
                    pk = f"{key}_projector"
                    r4 = tk.Frame(lf, bg=THEME["bg_color"]); r4.pack(fill=tk.X, padx=5, pady=2)
                    tk.Button(r4, text="Projector", command=lambda k=pk: self._set_path(k, labels, win, True)).pack(side=tk.LEFT)
                    labels[pk] = tk.Label(r4, text=os.path.basename(self.model_paths.get(pk, "") or "Not Set"), bg=THEME["bg_color"], fg=THEME["fg_color"], font=("Open Sans", 8))
                    labels[pk].pack(side=tk.LEFT, padx=5)
                _bind_click(lf)
    
            media_frame = tk.Frame(main, bg=THEME["bg_color"])
            media_frame.pack(fill=tk.X, pady=10)
            tk.Label(media_frame, text="Rich Media Rendering:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=("Open Sans", 10, "bold")).pack(side=tk.LEFT, padx=10)
            media_var = tk.IntVar(value=self.config.get("media_rendering", 1))
            for v, t in [(0, "None"), (1, "Inline"), (2, "Popup")]:
                tk.Radiobutton(media_frame, text=t, variable=media_var, value=v, bg=THEME["bg_color"], fg=THEME["fg_color"], selectcolor=THEME["widget_bg_color"]).pack(side=tk.LEFT, padx=5)
    
            tier_grid = tk.Frame(main, bg=THEME["bg_color"])
            tier_grid.pack(fill=tk.X, pady=10)
            tier_grid.grid_columnconfigure(0, weight=1); tier_grid.grid_columnconfigure(1, weight=1)
            tiers = ["fast", "search", "low", "med", "high", "secret", "deep_cook"]
            for i, tier in enumerate(tiers):
                r, c = divmod(i, 2)
                _create_tier_block(tier_grid, tier, r, c)
    
            over_lf = tk.LabelFrame(tier_grid, text="Global Overrides", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=("Open Sans", 10, "bold"), pady=5)
            over_lf.grid(row=3, column=1, sticky="nsew", padx=10, pady=5)
            
            kv_var = tk.StringVar(value=self.config.get("global_kv_cache", "Auto"))
            tk.Label(over_lf, text="Global KV Cache:", bg=THEME["bg_color"], fg=THEME["electric_blue"]).pack(anchor="w", padx=5)
            kv_f = tk.Frame(over_lf, bg=THEME["bg_color"]); kv_f.pack(anchor="w", padx=10)
            for o in ["Auto", "f32", "f16", "q8_0", "q4_0", "TQ2", "TQ3", "TQ4"]:
                tk.Radiobutton(kv_f, text=o, variable=kv_var, value=o, 
                               bg=THEME["widget_bg_color"], fg=THEME["fg_color"],
                               selectcolor=THEME["electric_blue"], indicatoron=False,
                               activebackground=THEME["electric_blue"], width=7).pack(side=tk.LEFT, padx=1)

    
            hao_var = tk.StringVar(value=self.config.get("hao_preset", "exps=CPU"))
            tk.Label(over_lf, text="HAO Preset:", bg=THEME["bg_color"], fg=THEME["electric_blue"]).pack(anchor="w", padx=5, pady=(5,0))
            hao_f = tk.Frame(over_lf, bg=THEME["bg_color"]); hao_f.pack(anchor="w", padx=10)
            for o in ["None", "exps=CPU"]:
                tk.Radiobutton(hao_f, text=o, variable=hao_var, value=o, 
                               bg=THEME["widget_bg_color"], fg=THEME["fg_color"],
                               selectcolor=THEME["electric_blue"], indicatoron=False,
                               activebackground=THEME["electric_blue"], width=10).pack(side=tk.LEFT, padx=2)
    
            swa_var = tk.StringVar(value=self.config.get("swa_kv_cache", "Auto"))
            tk.Label(over_lf, text="SWA Offload:", bg=THEME["bg_color"], fg=THEME["electric_blue"]).pack(anchor="w", padx=5, pady=(5,0))
            swa_f = tk.Frame(over_lf, bg=THEME["bg_color"]); swa_f.pack(anchor="w", padx=10)
            for o in ["Auto", "CPU Only"]:
                tk.Radiobutton(swa_f, text=o, variable=swa_var, value=o, 
                               bg=THEME["widget_bg_color"], fg=THEME["fg_color"],
                               selectcolor=THEME["electric_blue"], indicatoron=False,
                               activebackground=THEME["electric_blue"], width=10).pack(side=tk.LEFT, padx=2)

            stream_var = tk.StringVar(value=self.state.get("streaming_mode", "Buffered"))
            tk.Label(over_lf, text="Streaming Behavior:", bg=THEME["bg_color"], fg=THEME["electric_blue"]).pack(anchor="w", padx=5, pady=(5,0))
            stream_f = tk.Frame(over_lf, bg=THEME["bg_color"]); stream_f.pack(anchor="w", padx=10)
            for o in ["Real-time", "Buffered", "Experimental Chunking", "Mass Dump"]:
                tk.Radiobutton(stream_f, text=o, variable=stream_var, value=o, 
                               bg=THEME["widget_bg_color"], fg=THEME["fg_color"],
                               selectcolor=THEME["electric_blue"], indicatoron=False,
                               activebackground=THEME["electric_blue"], width=0).pack(side=tk.LEFT, padx=5, pady=2)

            ratio_var = tk.IntVar(value=self.config.get("max_token_ratio", 4))
            tk.Label(over_lf, text="Response Headroom (ctx/N):", bg=THEME["bg_color"], fg=THEME["electric_blue"]).pack(anchor="w", padx=5, pady=(5,0))
            ratio_f = tk.Frame(over_lf, bg=THEME["bg_color"]); ratio_f.pack(anchor="w", padx=10)
            for val, lbl in [(16, "U-Fast (16)"), (8, "Fast (8)"), (4, "Balanced (4)"), (2, "Deep (2)")]:
                tk.Radiobutton(ratio_f, text=lbl, variable=ratio_var, value=val, 
                               bg=THEME["widget_bg_color"], fg=THEME["fg_color"],
                               selectcolor=THEME["electric_blue"], indicatoron=False,
                               activebackground=THEME["electric_blue"], width=0).pack(side=tk.LEFT, padx=5, pady=2)
            
            auto_vram_var = tk.BooleanVar(value=self.config.get("auto_vram_offload", False))
            spec_draft_var = tk.BooleanVar(value=self.config.get("speculative_drafting", True))
            auto_vram_f = tk.Frame(over_lf, bg=THEME["bg_color"]); auto_vram_f.pack(anchor="w", padx=10, pady=(5,0))
            tk.Checkbutton(auto_vram_f, text="Dynamic Auto-Offload", variable=auto_vram_var,
                           bg=THEME["bg_color"], fg=THEME["electric_blue"], selectcolor=THEME["widget_bg_color"]).pack(side=tk.LEFT)
            tk.Checkbutton(auto_vram_f, text="Speculative MTP Drafting", variable=spec_draft_var,
                           bg=THEME["bg_color"], fg=THEME["electric_blue"], selectcolor=THEME["widget_bg_color"]).pack(side=tk.LEFT, padx=(15, 0))
    
            tk.Label(main, text="Vision Engines:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=("Open Sans", 10, "bold")).pack(anchor="w", padx=10, pady=(15, 5))
            v_grid = tk.Frame(main, bg=THEME["bg_color"]); v_grid.pack(fill=tk.X, pady=5)
            v_grid.grid_columnconfigure(0, weight=1); v_grid.grid_columnconfigure(1, weight=1)
            for i, vt in enumerate(["video", "video_deep", "multimodal"]):
                r, c = divmod(i, 2)
                _create_tier_block(v_grid, vt, r, c, True)
    
            def _save():
                self.config["media_rendering"] = media_var.get()
                self.state["deep_cook_behavior"] = v_behavior.get()
                # MISSION: Robustly reset Deep Cook state if switching to oneshot mode
                if self.state["deep_cook_behavior"] == "oneshot":
                    self.state["deep_cook"] = False
                self._sync_deep_cook_ui()
                self.state["multimodal_engine"] = mm_engine_var.get()
                self.config["global_kv_cache"] = kv_var.get()
                try:
                    for path in [
                        os.path.join(self.script_dir, "Live", "System", "params.json"),
                        os.path.join("Live", "System", "params.json")
                    ]:
                        if os.path.exists(path):
                            with open(path, "r") as f:
                                p_data = json.load(f)
                            p_data["global_kv_cache"] = kv_var.get()
                            with open(path, "w") as f:
                                json.dump(p_data, f, indent=4)
                except Exception as pe:
                    print(f"[UI] Warning: Could not write global_kv_cache to Live params: {pe}")
                self.config["hao_preset"] = hao_var.get()

                self.config["swa_kv_cache"] = swa_var.get()
                self.config["auto_vram_offload"] = auto_vram_var.get()
                self.config["speculative_drafting"] = spec_draft_var.get()
                self.state["streaming_mode"] = stream_var.get()
                self.config["max_token_ratio"] = ratio_var.get()
                self.config["live_core_selection"] = live_core_var.get()
                try: self.state["virtual_vram"] = int(float(vram_ent.get()) * 1024)
                except: pass
                for t, e in ents.items():
                    try: self.gpu_layer_config[t] = int(e.get())
                    except: pass
                for t, e in ctx_ents.items():
                    try: self.context_size_config[t] = int(e.get())
                    except: pass
                for t, e in n_batch_ents.items():
                    try: self.n_batch_config[t] = int(e.get())
                    except: pass
                for t, e in temp_ents.items():
                    try: self.temp_config[t] = float(e.get())
                    except: pass
                for t, e in top_p_ents.items():
                    try: self.top_p_config[t] = float(e.get())
                    except: pass
                for t, e in min_p_ents.items():
                    try: self.min_p_config[t] = float(e.get())
                    except: pass
                for t, e in top_k_ents.items():
                    try: self.top_k_config[t] = int(float(e.get()))
                    except: pass
                for t, e in rep_ents.items():
                    try: self.repeat_penalty_config[t] = float(e.get())
                    except: pass
                for t, e in freq_ents.items():
                    try: self.frequency_penalty_config[t] = float(e.get())
                    except: pass
                for t, e in stop_ents.items():
                    self.stop_strings_config[t] = e.get()
                self.save_config()
                messagebox.showinfo("Success", "Settings saved!")
                win.destroy()
    
            tk.Button(btn_frame, text="Save & Apply", command=_save, bg=THEME["button_active_color"], fg=THEME["fg_color"]).pack(side=tk.RIGHT, padx=5)
            tk.Button(btn_frame, text="Clear History", command=self.clear_current_history, bg="#660000", fg="white").pack(side=tk.RIGHT, padx=5)
            
            def _reset_defaults():
                if messagebox.askyesno("Reset", "Restore system defaults for all layers and samplers?"):
                    recs = self.run_auto_detect(win)
                    for t in recs:
                        if t in ents: 
                            ents[t].delete(0, tk.END)
                            ents[t].insert(0, str(recs[t]))
                    messagebox.showinfo("Reset", "System recommendations applied to visible fields. Click 'Save' to persist.")
            
            tk.Button(btn_frame, text="Auto-Detect", command=_reset_defaults).pack(side=tk.RIGHT, padx=5)
            tk.Button(btn_frame, text="Cancel", command=win.destroy).pack(side=tk.RIGHT, padx=5)
    
        except Exception as e:
            import traceback
            err_msg = f"Settings Window Crash: {e}\n{traceback.format_exc()}"
            print(err_msg)
            with open("Logs/ui_crash.txt", "w") as f: f.write(err_msg)
            messagebox.showerror("UI Error", f"Settings window failed to open:\n{e}")

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

sys.excepthook = log_uncaught_exception

if __name__ == "__main__":
    try:
        print("Starting SerenityPC...")
        root = tk.Tk()
        root.withdraw()
        ls = LoadingScreen(root)
        ls.start_animation()
        app = ChatbotApp(root, ls)
        root.mainloop()
    except Exception as e: log_uncaught_exception(type(e), e, e.__traceback__)
    finally:
        if SYSTEM_MONITOR_LOADED: 
            try: nvidia_ml.nvmlShutdown()
            except: pass