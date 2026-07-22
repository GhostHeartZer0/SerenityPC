#Serenity Live Agent UI
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

import json
import threading
import time
import re
from datetime import datetime
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageGrab # type: ignore
import requests # type: ignore
import speech_recognition as sr # type: ignore
import psutil # type: ignore
import socket
import sys
import ctypes
import subprocess
from typing import Optional, Any
import winsound
import sounddevice as sd
import queue


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

class ScrollableFrame(tk.Frame):
    """A reusable scrollable frame for Tkinter."""
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self, bg=kwargs.get("bg", "#1a1a24"), highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg=kwargs.get("bg", "#1a1a24"))

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self._window_id = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        def _configure_canvas(event):
            if self.scrollable_frame.winfo_reqwidth() != self.canvas.winfo_width():
                self.canvas.itemconfigure(self._window_id, width=self.canvas.winfo_width())
        self.canvas.bind("<Configure>", _configure_canvas)

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Mousewheel support only when mouse is over this widget or its children
        def _on_mousewheel(event):
            widget = event.widget
            try:
                while widget:
                    if widget == self.canvas:
                        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
                        break
                    widget = widget.master
            except:
                pass
        self.scrollable_frame.bind_all("<MouseWheel>", _on_mousewheel)

class SoundDeviceMicrophone(sr.AudioSource):
    """A cross-platform AudioSource wrapper for sounddevice, replacing PyAudio."""
    def __init__(self, device_index=None, sample_rate=16000, chunk_size=1024):
        self.device_index = device_index
        self.format = 8  # Equivalent to PyAudio's paInt16
        self.SAMPLE_WIDTH = 2
        self.SAMPLE_RATE = sample_rate
        self.CHUNK = chunk_size
        self.stream = None
        self.q = queue.Queue()

    def __enter__(self):
        def callback(indata, frames, time, status):
            self.q.put(bytes(indata))

        class StreamWrapper:
            def __init__(self, q):
                self.q = q
            def read(self, size):
                if hasattr(self, 'buffer') and len(self.buffer) >= size * 2:
                    ret = self.buffer[:size * 2]
                    self.buffer = self.buffer[size * 2:]
                    return ret
                return self.q.get()

        self.raw_stream = sd.RawInputStream(
            samplerate=self.SAMPLE_RATE, blocksize=self.CHUNK,
            device=self.device_index, channels=1, dtype='int16',
            callback=callback
        )
        self.raw_stream.start()
        self.stream = StreamWrapper(self.q)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if hasattr(self, 'raw_stream') and self.raw_stream:
            self.raw_stream.stop()
            self.raw_stream.close()
        self.stream = None

# Single Instance Lock
inst_lock = None
def get_lock():
    global inst_lock
    try:
        inst_lock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        inst_lock.bind(("127.0.0.1", 47201)) # Arbitrary unused port
    except (socket.error, OSError):
        print("Visual Core already running.")
        sys.exit(0)

get_lock()

try:
    import pynvml as nvidia_ml_py
    nvidia_ml_py.nvmlInit()
    HAS_NVML = True
except (ImportError, Exception):
    HAS_NVML = False

SERENITY_ENGINE_URL = "http://127.0.0.1:8001/analyze"

class SerenityUI:
    def __init__(self, agent: Any):
        self.agent = agent
        self.root = tk.Tk()
        self.root.title("Serenity Live Core")
        
        # --- Set Window Icon ---
        icon_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "System", "transcendent_serenity_ws_hq.ico")
        if os.path.exists(icon_path) and os.name == 'nt':
            try: self.root.iconbitmap(icon_path)
            except Exception: pass
        elif not os.name == 'nt':
            # Linux/Mac icon fallback (PNG)
            png_icon = icon_path.replace(".ico", ".png")
            if os.path.exists(png_icon):
                try:
                    img = tk.PhotoImage(file=png_icon)
                    self.root.iconphoto(True, img)
                except Exception: pass
        self.root.geometry("650x650")
        self.root.configure(bg="#0d0d12")
        
        style = ttk.Style()
        style.theme_use('clam')
        # Load Style (Green)
        style.configure("Green.Horizontal.TProgressbar", foreground='#00ff96', background='#00ff96', thickness=10, bordercolor='#1a1a24', lightcolor='#00ff96', darkcolor='#00ff96')
        # Inference Style (Orange)
        style.configure("Orange.Horizontal.TProgressbar", foreground='#ff9900', background='#ff9900', thickness=10, bordercolor='#1a1a24', lightcolor='#ff9900', darkcolor='#ff9900')
        
        style.layout("Green.Horizontal.TProgressbar", [('Horizontal.Progressbar.trough', {'children': [('Horizontal.Progressbar.pbar', {'side': 'left', 'sticky': 'ns'})], 'sticky': 'nswe'})])
        style.layout("Orange.Horizontal.TProgressbar", [('Horizontal.Progressbar.trough', {'children': [('Horizontal.Progressbar.pbar', {'side': 'left', 'sticky': 'ns'})], 'sticky': 'nswe'})])
        
        # Determine directories robustly
        current_dir = os.path.abspath(os.path.dirname(__file__))
        parts = current_dir.replace('\\', '/').split('/')
        if "Live" in parts:
            idx = len(parts) - 1 - list(reversed(parts)).index("Live")
            self.live_dir = "/".join([parts[i] for i in range(idx + 1)])
        else:
            self.live_dir = current_dir
            
        self.media_dir = os.path.join(self.live_dir, "System", "Media")
        self.logs_dir = os.path.join(self.agent.live_dir, "Logs")
        self.history_dir = os.path.join(self.agent.live_dir, "History")
        os.makedirs(self.history_dir, exist_ok=True)
        self.system_dir = os.path.join(self.live_dir, "System")
        
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.history_dir, exist_ok=True)

        self.log_window: Optional[tk.Toplevel] = None
        self.log_container: Optional[tk.Frame] = None
        self.log_tabs: dict[str, tk.Text] = {}
        self.log_frames: dict[str, tk.Frame] = {}
        self.nav_buttons: dict[str, tk.Button] = {}
        self.avatar_photo: Optional[ImageTk.PhotoImage] = None
        self.current_avatar_name: Optional[str] = None
        self.active_log_tab: Optional[str] = None
        
        self.model_var = tk.StringVar(value="Coremal (1B)")
        
        # Light settings
        self.light_layers_var = tk.IntVar(value=self.agent.params.get("light_layers", 0))
        self.light_ctx_var = tk.IntVar(value=self.agent.params.get("light_ctx", 8192))
        self.light_temp_var = tk.DoubleVar(value=self.agent.params.get("light_temp", 0.2))
        self.light_top_p_var = tk.DoubleVar(value=self.agent.params.get("light_top_p", 0.9))
        self.light_top_k_var = tk.IntVar(value=self.agent.params.get("light_top_k", 40))
        self.light_repeat_var = tk.DoubleVar(value=self.agent.params.get("light_repeat", 1.1))
        self.light_tokens_var = tk.IntVar(value=self.agent.params.get("light_tokens", 256))

        # Med settings
        self.med_layers_var = tk.IntVar(value=self.agent.params.get("med_layers", self.agent.params.get("n_gpu_layers", 0)))
        self.med_ctx_var = tk.IntVar(value=self.agent.params.get("med_ctx", self.agent.params.get("n_ctx", 4096)))
        self.med_temp_var = tk.DoubleVar(value=self.agent.params.get("med_temp", self.agent.params.get("temperature", 0.35)))
        self.med_top_p_var = tk.DoubleVar(value=self.agent.params.get("med_top_p", self.agent.params.get("top_p", 0.9)))
        self.med_top_k_var = tk.IntVar(value=self.agent.params.get("med_top_k", self.agent.params.get("top_k", 50)))
        self.med_repeat_var = tk.DoubleVar(value=self.agent.params.get("med_repeat", self.agent.params.get("repeat_penalty", 1.15)))
        self.med_tokens_var = tk.IntVar(value=self.agent.params.get("med_tokens", self.agent.params.get("max_tokens", 512)))

        # Heavy settings
        self.heavy_layers_var = tk.IntVar(value=self.agent.params.get("heavy_layers", self.agent.params.get("n_gpu_layers", 24)))
        self.heavy_ctx_var = tk.IntVar(value=self.agent.params.get("heavy_ctx", self.agent.params.get("n_ctx", 4096)))
        self.heavy_temp_var = tk.DoubleVar(value=self.agent.params.get("heavy_temp", self.agent.params.get("temperature", 0.4)))
        self.heavy_top_p_var = tk.DoubleVar(value=self.agent.params.get("heavy_top_p", self.agent.params.get("top_p", 0.9)))
        self.heavy_top_k_var = tk.IntVar(value=self.agent.params.get("heavy_top_k", self.agent.params.get("top_k", 50)))
        self.heavy_repeat_var = tk.DoubleVar(value=self.agent.params.get("heavy_repeat", self.agent.params.get("repeat_penalty", 1.2)))
        self.heavy_tokens_var = tk.IntVar(value=self.agent.params.get("heavy_tokens", self.agent.params.get("max_tokens", 512)))


        self.vram_threshold_var = tk.IntVar(value=self.agent.params.get("vram_threshold_mb", 600))
        
        # Granular Offload controls
        self.encoder_gpu_var = tk.BooleanVar(value=self.agent.params.get("encoder_on_gpu", False))
        self.tied_gpu_var = tk.BooleanVar(value=self.agent.params.get("tied_on_gpu", False))
        self.cache_compress_var = tk.StringVar(value=self.agent.params.get("global_kv_cache", "Auto"))

        
        # --- Pre-initialize elements to prevent attribute errors ---
        self.mic_status_label: Optional[tk.Label] = None
        self.learning_var = tk.BooleanVar(value=True)
        self.history_listbox: Optional[tk.Listbox] = None
        self.history_viewer: Optional[tk.Text] = None
        self.engine_status_label: Optional[tk.Label] = None
        self.tab_progress: Optional[ttk.Progressbar] = None
        self._log_refresh_running = False
        self._inference_start_time: float = 0.0
        self._inference_eta: float = 0.0
        self._timer_id: Optional[str] = None
        
        self._file_mtimes = {} # Track mod times for logs
        
        # Create Layout
        self.main_frame = tk.Frame(self.root, bg="#0d0d12")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self.left_frame = tk.Frame(self.main_frame, bg="#0d0d12")
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        self.right_frame = tk.Frame(self.main_frame, bg="#1a1a24", width=200)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0), pady=0)
        
        self.right_inner = tk.Frame(self.right_frame, bg="#1a1a24")
        self.right_inner.pack(fill=tk.BOTH, expand=True, pady=10)

        # Base Left items
        self.image_label = tk.Label(self.left_frame, bg="#0d0d12")
        self.image_label.pack(pady=20)
        
        self.console = tk.Text(self.left_frame, height=12, width=45, bg="#1a1a24", fg="#00ff96", 
                               font=("Consolas", 10), borderwidth=0, padx=10, pady=10)
        self.console.pack(pady=10, fill=tk.BOTH, expand=True)

        # Base Right items
        tk.Label(self.right_inner, text="SYSTEM INTELLIGENCE", fg="#00ff96", bg="#1a1a24", font=("Consolas", 10, "bold")).pack(pady=(20, 10))
        self.sysinfo_label = tk.Label(self.right_inner, text="", fg="#00ff96", bg="#1a1a24", font=("Consolas", 9), justify=tk.LEFT)
        self.sysinfo_label.pack(pady=10, padx=10, anchor="w")

        # Global loading bar for main screen
        self.main_progress = ttk.Progressbar(self.left_frame, style="Green.Horizontal.TProgressbar", orient="horizontal", length=200, mode="determinate")
        # ETA Label
        self.eta_label = tk.Label(self.left_frame, text="", fg="#ff9900", bg="#0d0d12", font=("Consolas", 9, "bold"))
        
        # Only show when loading/thinking
        self.main_progress.pack(pady=(10, 0), padx=20, fill=tk.X)
        self.eta_label.pack(pady=(2, 10))
        self.main_progress.pack_forget()
        self.eta_label.pack_forget()


        tk.Label(self.right_inner, text="LOGS & MEMORY", fg="#00ff96", bg="#1a1a24", font=("Consolas", 10, "bold")).pack(pady=(30, 10))
        
        self.nav_buttons = {}
        for name in ["Settings", "SysLog", "Error Log", "Thoughts", "Histories", "Learning Log"]:
            btn = tk.Button(self.right_inner, text=name, 
                            command=lambda n=name: self.open_log_tab(n), # type: ignore
                            bg="#2a2a35", fg="#00ff96", font=("Consolas", 9), 
                            width=18, borderwidth=0, pady=5)
            btn.pack(pady=5, anchor="e")
            self.nav_buttons[name] = btn
            
        # VRAM Management Widget
        tk.Label(self.right_inner, text="VRAM ENGINE CONTROL", fg="#00ff96", bg="#1a1a24", font=("Consolas", 10, "bold")).pack(pady=(20, 5))
        
        self.vram_status_frame = tk.Frame(self.right_inner, bg="#1a1a24")
        self.vram_status_frame.pack(pady=5, anchor="e")
        
        self.llama_status_label = tk.Label(self.vram_status_frame, text="Llama Core: Loaded", fg="#00ff96", bg="#1a1a24", font=("Consolas", 9))
        self.llama_status_label.pack(anchor="e")
        
        self.t5_status_label = tk.Label(self.vram_status_frame, text="T5 Engine: Offline", fg="#7a7a8a", bg="#1a1a24", font=("Consolas", 9))
        self.t5_status_label.pack(anchor="e")
        
        # Start/Offload controls
        self.vram_control_btn = tk.Button(self.right_inner, text="Offload Llama Core", 
                                          command=self.toggle_vram_model,
                                          bg="#2a2a35", fg="#ff9900", font=("Consolas", 9, "bold"), 
                                          width=18, borderwidth=0, pady=5)
        self.vram_control_btn.pack(pady=5, anchor="e")
        
        self.update_avatar("transcendent_serenity.png")
        self.log("--- Serenity Visual Interface Online ---")
        
        self.update_sysinfo()
        threading.Thread(target=self.agent.listen_and_process, args=(self,), daemon=True).start()
        
        # Shutdown hook
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        if os.environ.get("SERENITY_NO_UI") != "1":
            # Using a lambda to ensure the callback signature is clean for tk.after
            self.root.after(2000, lambda: self.open_log_tab("Thoughts")) # type: ignore

    def update_sysinfo(self):
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory()
        ram_used = ram.used / (1024**3)
        ram_total = ram.total / (1024**3)
        
        info_text = f"CPU:  {cpu:.1f}%\nRAM:  {ram_used:.1f} / {ram_total:.1f} GB\n\n"
        
        if HAS_NVML:
            try:
                handle = nvidia_ml_py.nvmlDeviceGetHandleByIndex(0)
                meminfo = nvidia_ml_py.nvmlDeviceGetMemoryInfo(handle)
                util = nvidia_ml_py.nvmlDeviceGetUtilizationRates(handle)
                vram_used = meminfo.used / (1024**3)
                vram_total = meminfo.total / (1024**3)
                info_text += f"GPU:  {util.gpu}%\nVRAM: {vram_used:.1f} / {vram_total:.1f} GB"
            except Exception:
                info_text += "GPU:  N/A\nVRAM: N/A"
        else:
            info_text += "GPU:  N/A\nVRAM: N/A"
            
        # Update Model Status labels & control button dynamically
        try:
            llama_loaded = self.agent.model is not None
            
            # Check T5 Engine status
            t5_active = False
            if getattr(self.agent, "live_agent_process", None) is not None and self.agent.live_agent_process.poll() is None:
                t5_active = True
            else:
                # Fallback: check if port 8001 has a process bound to it
                try:
                    for conn in psutil.net_connections():
                        laddr = getattr(conn, 'laddr', None)
                        if laddr and hasattr(laddr, 'port') and laddr.port == 8001:
                            t5_active = True
                            break
                except: pass
            
            if llama_loaded:
                self.llama_status_label.configure(text="Llama Core: Loaded", fg="#00ff96")
            else:
                self.llama_status_label.configure(text="Llama Core: Offloaded", fg="#7a7a8a")
                
            if t5_active:
                self.t5_status_label.configure(text="T5 Engine: Active", fg="#00ff96")
            else:
                self.t5_status_label.configure(text="T5 Engine: Offline", fg="#7a7a8a")
                
            if llama_loaded and t5_active:
                self.vram_control_btn.configure(text="RESOLVE DUAL LOAD", fg="#ff4444")
            elif llama_loaded:
                self.vram_control_btn.configure(text="Offload Llama Core", fg="#ff9900")
            elif t5_active:
                self.vram_control_btn.configure(text="Shutdown T5 Engine", fg="#ff9900")
            else:
                self.vram_control_btn.configure(text="Load Model Core", fg="#00ff96")
        except Exception: pass
            
        try:
            self.sysinfo_label.configure(text=info_text)
            self.root.after(1000, lambda: self.update_sysinfo()) # type: ignore
        except (tk.TclError, RuntimeError): pass

    def update_mic_status(self):
        try:
            if not getattr(self, "mic_status_label", None) or not self.mic_status_label.winfo_exists(): return # type: ignore
            status = "Connected (Ready)" if hasattr(self.agent, "microphone") and self.agent.microphone is not None else "Not Detected (Retrying...)"
            color = "#00ff96" if "Connected" in status else "#ff4d4d"
            if self.mic_status_label is not None and self.mic_status_label.winfo_exists(): # type: ignore
                self.mic_status_label.configure(text=status, fg=color) # type: ignore
            self.root.after(3000, lambda: self.update_mic_status()) # type: ignore
        except (tk.TclError, RuntimeError, AttributeError): pass

    def open_log_tab(self, log_name: str):
        lw = self.log_window
        if lw is not None and lw.winfo_exists():
            if self.nav_buttons[log_name]["bg"] == "#1a1a24" or self.nav_buttons[log_name]["bg"] == "#ffffff":
                self.close_log_window()
                return

        lw2 = self.log_window
        if lw2 is None or not lw2.winfo_exists():
            self._file_mtimes = {} 
            
            new_window = tk.Toplevel(self.root)
            self.log_window = new_window
            new_window.title("Serenity Live Logs")
            
            new_window.resizable(True, True)
            new_window.transient(self.root)
            new_window.configure(bg="#1a1a24")
            
            def initial_sync():
                lw_sync = self.log_window
                if lw_sync is not None and lw_sync.winfo_exists():
                    x = self.root.winfo_rootx() + self.root.winfo_width()
                    y = self.root.winfo_rooty()
                    lw_sync.geometry(f"600x{self.root.winfo_height()}+{x}+{y}")
            
            self.root.after(100, initial_sync)
            
            new_container = tk.Frame(new_window, bg="#1a1a24")
            self.log_container = new_container
            new_container.pack(fill=tk.BOTH, expand=True)
            
            close_btn = tk.Button(new_container, text="  X  ", bg="#1a1a24", fg="#ff4d4d",
                                  font=("Consolas", 10, "bold"), borderwidth=0,
                                  command=self.close_log_window, activebackground="#ff4d4d", activeforeground="#1a1a24")
            close_btn.place(relx=1.0, rely=0.0, anchor="ne")

            self.log_tabs = {}
            self.log_frames = {}
            for name in ["Settings", "SysLog", "Error Log", "Thoughts", "Histories", "Learning Log"]:
                frame = tk.Frame(new_container, bg="#1a1a24")
                self.log_frames[name] = frame
                
                tk.Label(frame, text=name.upper(), fg="#00ff96", bg="#1a1a24", font=("Consolas", 12, "bold")).pack(pady=(20, 10))
                
                if name == "Settings":
                    sf = ScrollableFrame(frame, bg="#1a1a24")
                    sf.pack(fill=tk.BOTH, expand=True)
                    c = sf.scrollable_frame 

                    tk.Button(c, text="Reset Live Connection", bg="#2a2a35", fg="#00ff96", font=("Consolas", 11), borderwidth=0, 
                               command=self.reset_connection).pack(pady=(10, 20), padx=20, fill=tk.X)
                              
                    self.learning_var = tk.BooleanVar(value=self.agent.params.get("learning_enabled", True))
                    tk.Checkbutton(c, text="Enable Continuous Cadence Learning", variable=self.learning_var, bg="#1a1a24", fg="#00ff96",
                                   selectcolor="#2a2a35", activebackground="#1a1a24", activeforeground="#00ff96", font=("Consolas", 10),
                                   command=self.toggle_learning).pack(pady=10, anchor="w", padx=20)
                                   
                    tk.Label(c, text="Microphone Health:", fg="#ffffff", bg="#1a1a24", font=("Consolas", 10)).pack(pady=(20, 5), anchor="w", padx=20)
                    lbl = tk.Label(c, text="Checking...", fg="#ffffff", bg="#1a1a24", font=("Consolas", 10))
                    lbl.pack(pady=5, anchor="w", padx=20)
                    self.mic_status_label = lbl
                    self.update_mic_status()


                    tk.Label(c, text="Active Model Core:", fg="#ffffff", bg="#1a1a24", font=("Consolas", 10)).pack(pady=(20, 5), anchor="w", padx=20)
                    current_core = self.agent.params.get("active_core", "med")
                    names = {"light": "Quick-core (270M)", "med": "Coremal (1B)", "heavy": "Intelli-Core (4B)"}
                    self.model_var.set(names.get(current_core, "Coremal (1B)"))
                    
                    cb = ttk.Combobox(c, textvariable=self.model_var, values=list(names.values()), state="readonly", font=("Consolas", 10))
                    cb.pack(fill=tk.X, padx=20, pady=5)
                    
                    tk.Label(c, text="KV Cache Sizing / Compression (TurboQuant):", fg="#ffffff", bg="#1a1a24", font=("Consolas", 10)).pack(pady=(15, 5), anchor="w", padx=20)
                    compress_cb = ttk.Combobox(c, textvariable=self.cache_compress_var, values=["Auto", "f32", "f16", "q8_0", "q4_0", "TQ2", "TQ3", "TQ4"], state="readonly", font=("Consolas", 10))
                    compress_cb.pack(fill=tk.X, padx=20, pady=5)

                    
                    tk.Label(c, text="ENGINE PARAMETERS (Hardware & Logic):", fg="#ff9900", bg="#1a1a24", font=("Consolas", 10, "bold")).pack(pady=(25, 10), anchor="w", padx=20)
                    
                    def create_setting(parent, label, variable, from_=0, to=100, resolution=1, row=0):
                        tk.Label(parent, text=label, fg="#ffffff", bg="#1a1a24", font=("Consolas", 9)).grid(row=row, column=0, sticky="w", pady=2)
                        scale = tk.Scale(parent, variable=variable, from_=from_, to=to, resolution=resolution, orient=tk.HORIZONTAL, 
                                         bg="#1a1a24", fg="#00ff96", highlightthickness=0, troughcolor="#2a2a35", font=("Consolas", 8), length=110)
                        scale.grid(row=row, column=1, sticky="e", padx=(5, 0))
                        return row + 1

                    param_grid = tk.Frame(c, bg="#1a1a24")
                    param_grid.pack(fill=tk.BOTH, expand=True, padx=20)

                    tiers = [
                        ("QUICK-CORE (270M)", "#00ff96", [
                            ("GPU Offload (Layers):", self.light_layers_var, 0, 64, 1),
                            ("Context Window:", self.light_ctx_var, 512, 32768, 512),
                            ("Max Response Tokens:", self.light_tokens_var, 32, 2048, 16)
                        ]),
                        ("COREMAL (1B)", "#00ff96", [
                            ("GPU Offload (Layers):", self.med_layers_var, 0, 64, 1),
                            ("Context Window:", self.med_ctx_var, 512, 32768, 512),
                            ("Max Response Tokens:", self.med_tokens_var, 32, 2048, 16)
                        ]),
                        ("INTELLI-CORE (4B)", "#ff9900", [
                            ("GPU Offload (Layers):", self.heavy_layers_var, 0, 64, 1),
                            ("Context Window:", self.heavy_ctx_var, 512, 32768, 512),
                            ("Max Response Tokens:", self.heavy_tokens_var, 32, 2048, 16)
                        ]),

                    ]

                    for idx, (t_name, t_color, t_settings) in enumerate(tiers):
                        col = idx % 2
                        row = idx // 2
                        t_frame = tk.Frame(param_grid, bg="#1a1a24", highlightbackground=t_color, highlightthickness=1, padx=5, pady=5)
                        t_frame.grid(row=row, column=col, sticky="nsew", padx=5, pady=5)
                        
                        tk.Label(t_frame, text=t_name, fg=t_color, bg="#1a1a24", font=("Consolas", 9, "bold")).grid(row=0, column=0, columnspan=2, pady=(0, 5))
                        
                        curr_row = 1
                        for s_label, s_var, s_from, s_to, s_res in t_settings:
                            curr_row = create_setting(t_frame, s_label, s_var, s_from, s_to, s_res, curr_row)


                
                elif name == "Histories":
                    h_frame = tk.Frame(frame, bg="#1a1a24")
                    h_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
                    
                    self.history_listbox = tk.Listbox(h_frame, bg="#0d0d12", fg="#00ff96", font=("Consolas", 10), borderwidth=0)
                    self.history_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                    self.history_listbox.bind("<<ListboxSelect>>", lambda e: self.read_history_file())
                    
                    self.history_viewer = tk.Text(h_frame, bg="#0d0d12", fg="#ffffff", font=("Consolas", 10), borderwidth=0, width=40)
                    self.history_viewer.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
                
                else:
                    txt = tk.Text(frame, bg="#0d0d12", fg="#00ff96", font=("Consolas", 10), borderwidth=0, padx=10, pady=10)
                    txt.pack(fill=tk.BOTH, expand=True)
                    self.log_tabs[name] = txt
                    
                    btn_frame = tk.Frame(frame, bg="#1a1a24")
                    btn_frame.pack(fill=tk.X)
                    tk.Button(btn_frame, text=f"Clear {name}", command=lambda n=name: self.clear_log(n),
                              bg="#1a1a24", fg="#ff4d4d", font=("Consolas", 9)).pack(side=tk.RIGHT, padx=10, pady=5)

        if self.log_window is not None:
            self.log_window.lift() # type: ignore

        for name, btn in self.nav_buttons.items():
            if name == log_name:
                btn.configure(bg="#1a1a24", fg="#ffffff")
            else:
                btn.configure(bg="#2a2a35", fg="#00ff96")
        
        for frame in self.log_frames.values():
            frame.pack_forget()
        self.log_frames[log_name].pack(fill=tk.BOTH, expand=True)
        self.active_log_tab = log_name

        if log_name in self.log_tabs:
            self.refresh_log(log_name)
        elif log_name == "Histories":
            self.refresh_history_list()
        
        if not hasattr(self, "_log_refresh_running") or not self._log_refresh_running:
            self._log_refresh_running = True
            self.auto_refresh_logs()

    def refresh_history_list(self):
        if not hasattr(self, "history_listbox") or self.history_listbox is None: return
        self.history_listbox.delete(0, tk.END) # type: ignore
        if os.path.exists(self.history_dir):
            files = sorted([f for f in os.listdir(self.history_dir) if f.endswith(".txt")], reverse=True)
            for f in files:
                self.history_listbox.insert(tk.END, f) # type: ignore

    def read_history_file(self):
        selection = getattr(self, "history_listbox", tk.Listbox()).curselection() if getattr(self, "history_listbox", None) else None # type: ignore
        if not selection: return
        filename = getattr(self, "history_listbox", tk.Listbox()).get(selection[0]) if getattr(self, "history_listbox", None) else None # type: ignore
        filepath = os.path.join(self.history_dir, filename)
        
        if getattr(self, "history_viewer", None):
            self.history_viewer.delete(1.0, tk.END) # type: ignore
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    self.history_viewer.insert(tk.END, f.read()) # type: ignore
            except Exception as e:
                self.history_viewer.insert(tk.END, f"[Error reading file: {e}]") # type: ignore

    def close_log_window(self):
        lw_del = self.log_window
        if lw_del is not None:
            lw_del.destroy() 
            self.log_window = None
        for btn in self.nav_buttons.items():
            btn[1].configure(bg="#2a2a35", fg="#00ff96")

    def refresh_log(self, name):
        if name not in ["SysLog", "Error Log", "Thoughts", "Learning Log"]: return
        file_map = {
            "SysLog": "SysLog.txt",
            "Error Log": "error_log.txt",
            "Thoughts": "scratchpad.txt",
            "Learning Log": "subconscious_journal.txt"
        }
        filename = file_map.get(name)
        if filename:
            filepath = os.path.join(self.logs_dir, filename)
            widget = self.log_tabs[name]
            was_at_bottom = True
            
            if os.path.exists(filepath):
                try:
                    mtime = os.path.getmtime(filepath)
                    if self._file_mtimes.get(name) == mtime:
                        return 
                    self._file_mtimes[name] = mtime
                    
                    # Capture scroll position before modifying
                    was_at_bottom = widget.yview()[1] >= 0.999
                    
                    widget.delete(1.0, tk.END) # type: ignore
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        lines = f.readlines()
                        content = "".join(lines)
                        widget.insert(tk.END, content) # type: ignore
                        
                        if name == "SysLog" and getattr(self, "engine_status_label", None):
                            engine_lines = [l for l in lines if "[SERENITY ENGINE]" in l]
                            if engine_lines:
                                    try:
                                        last_line = engine_lines[-1]
                                        last_msg = last_line.split("] - ", 1)[-1].strip()
                                        msg_slice: str = last_msg[-75:] if len(last_msg) > 75 else last_msg # type: ignore
                                        display_text = f"» {msg_slice}"
                                        lbl = self.engine_status_label
                                        if lbl is not None and lbl.winfo_exists():
                                            lbl.configure(text=display_text)
                                    except (AttributeError, IndexError): pass
                            
                            progress_match = re.findall(r'(\d+)%', content)
                            if progress_match:
                                percent = int(progress_match[-1])
                                if self.tab_progress is not None:
                                    self.tab_progress["value"] = percent # type: ignore
                                if self.main_progress is not None:
                                    if self.main_progress["style"] == "Green.Horizontal.TProgressbar":
                                        self.main_progress["value"] = percent
                                        if percent < 100:
                                            self.main_progress.pack(pady=(10, 0), padx=20, fill=tk.X, before=self.console)
                                            if hasattr(self, "eta_label") and self.eta_label is not None:
                                                self.eta_label.configure(text=f"Loading Core Weights: {percent}%", fg="#00ff96")
                                                self.eta_label.pack(pady=(2, 10), before=self.console)
                                        else:
                                            self.main_progress.pack_forget()
                                            if hasattr(self, "eta_label") and self.eta_label is not None:
                                                self.eta_label.pack_forget()
                except Exception as e:
                    widget.delete(1.0, tk.END) # type: ignore
                    widget.insert(tk.END, f"[Error reading {filename}: {e}]") # type: ignore
            else:
                widget.delete(1.0, tk.END) # type: ignore
                widget.insert(tk.END, f"[Log file not yet generated: {filename}]") # type: ignore
            
            if hasattr(widget, "see") and was_at_bottom: widget.see(tk.END) # type: ignore

    def clear_log(self, name):
        if name not in ["SysLog", "Error Log", "Thoughts", "Learning Log"]: return
        file_map = {"SysLog": "SysLog.txt", "Error Log": "error_log.txt", "Thoughts": "scratchpad.txt", "Learning Log": "subconscious_journal.txt"}
        filename = file_map.get(name)
        if filename:
            filepath = os.path.join(self.logs_dir, filename)
            try:
                open(filepath, 'w').close()
                self.refresh_log(name)
            except Exception: pass

    def auto_refresh_logs(self):
        try:
            if getattr(self, "log_window", None) and self.log_window.winfo_exists(): # type: ignore
                if self.active_log_tab in ["SysLog", "Thoughts", "Error Log"]:
                    self.refresh_log(self.active_log_tab)
            self.root.after(2000, self.auto_refresh_logs) # type: ignore
        except (tk.TclError, RuntimeError):
            self._log_refresh_running = False

    def reset_connection(self):
        """Kill the current engine and reload the chosen active model core."""
        self.console.delete(1.0, tk.END)
        
        # Read the currently selected core from the dropdown and persist it
        names_rev = {"Quick-core (270M)": "light", "Coremal (1B)": "med", "Intelli-Core (4B)": "heavy", "Troubleshooter": "trouble"}
        new_core = names_rev.get(self.model_var.get(), "med")
        self.agent.params["active_core"] = new_core
        self.agent.save_params()
        
        self.log(f"[SYSTEM] Resetting connection — reloading {new_core.upper()} core...")
        self.swap_model()

    def toggle_learning(self):
        self.agent.params["learning_enabled"] = self.learning_var.get()
        self.agent.save_params()
        self.log(f"[SYSTEM] Continuous Cadence Learning {'enabled' if self.learning_var.get() else 'disabled'}.")

    def start_inference_eta(self, text):
        core = self.agent.params.get("active_core", "med")
        base_map = {"med": 3.5}
        char_factor = {"med": 40}
        
        eta = base_map.get(core, 3.5) + (len(text) / char_factor.get(core, 40))
        if eta < 1.0: eta = 1.0
        
        self.main_progress["style"] = "Orange.Horizontal.TProgressbar"
        self.main_progress["value"] = 0
        self.main_progress.pack(pady=(10, 0), padx=20, fill=tk.X, before=self.console)
        
        self.eta_label.configure(text=f"Est. Response: {eta:.1f}s", fg="#ff9900")
        self.eta_label.pack(pady=(2, 10), before=self.console)
        
        self._inference_start_time = float(time.time())
        self._inference_eta = float(eta)
        self._update_inference_timer()

    def _update_inference_timer(self):
        if not hasattr(self, "_inference_start_time"): return
        
        elapsed = time.time() - self._inference_start_time
        remaining = max(0.0, float(self._inference_eta) - elapsed)
        percent = min(99.0, (elapsed / float(self._inference_eta)) * 100)
        
        self.main_progress["value"] = percent
        self.eta_label.configure(text=f"Est. Response: {remaining:.1f}s")
        
        if elapsed < self._inference_eta + 5:
            self._timer_id = self.root.after(100, lambda: self._update_inference_timer()) # type: ignore

    def stop_inference_eta(self):
        if self._timer_id is not None:
            self.root.after_cancel(self._timer_id) # type: ignore
            self._timer_id = None
        if hasattr(self, "_inference_start_time"):
            self._inference_start_time = 0.0
            
        self.main_progress.pack_forget()
        self.eta_label.pack_forget()
        self.main_progress["style"] = "Green.Horizontal.TProgressbar" 

    def offload_engine(self):
        """Shut down the T5 engine and free VRAM. Updates the control button to allow re-loading."""
        self.log("[SYSTEM] Offloading backend engine and halting processes...")
        self.agent.awake = False
        self.update_avatar("serenity_off.png")
        
        # 1. Kill tracked subprocess handle
        if getattr(self.agent, "live_agent_process", None) is not None:
            try:
                self.agent.live_agent_process.terminate()
                self.agent.live_agent_process.wait(timeout=3)
            except: pass
            self.agent.live_agent_process = None
            
        # 2. Graceful HTTP shutdown request
        try:
            requests.post("http://127.0.0.1:8001/shutdown", headers={"x-api-key": "REVOKED"}, timeout=3)
            time.sleep(1.0)  # Give it a moment to self-terminate
        except Exception: pass

        # 3. Force-kill any surviving t5_server.py processes (works without elevated perms)
        killed_pids = []
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info.get('cmdline') or []
                    if any("t5_server.py" in s for s in cmdline) and proc.pid != os.getpid():
                        self.log(f"[SYSTEM] Killing engine process (PID {proc.pid})...")
                        proc.terminate()
                        try:
                            proc.wait(timeout=2)
                        except psutil.TimeoutExpired:
                            proc.kill()
                        killed_pids.append(proc.pid)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            self.agent.log_error(f"Engine halt error: {e}")
            self.log(f"[SYSTEM] Engine halt error: {e}")

        # 4. Update the VRAM control button to reflect offloaded state
        try:
            if hasattr(self, "vram_control_btn") and self.vram_control_btn.winfo_exists():
                self.vram_control_btn.configure(text="Load Model Core", fg="#00ff96")
        except (tk.TclError, RuntimeError): pass

        if killed_pids:
            self.log(f"[SYSTEM] Engine offloaded (killed PIDs: {killed_pids}). VRAM released.")
        else:
            self.log("[SYSTEM] Engine offloaded. VRAM released.")

    def save_and_reboot(self):
        self.agent.params["light_layers"] = self.light_layers_var.get()
        self.agent.params["light_ctx"] = self.light_ctx_var.get()
        self.agent.params["light_temp"] = self.light_temp_var.get()
        self.agent.params["light_top_p"] = self.light_top_p_var.get()
        self.agent.params["light_top_k"] = self.light_top_k_var.get()
        self.agent.params["light_repeat"] = self.light_repeat_var.get()
        self.agent.params["light_tokens"] = self.light_tokens_var.get()
        
        self.agent.params["med_layers"] = self.med_layers_var.get()
        self.agent.params["med_ctx"] = self.med_ctx_var.get()
        self.agent.params["med_temp"] = self.med_temp_var.get()
        self.agent.params["med_top_p"] = self.med_top_p_var.get()
        self.agent.params["med_top_k"] = self.med_top_k_var.get()
        self.agent.params["med_repeat"] = self.med_repeat_var.get()
        self.agent.params["med_tokens"] = self.med_tokens_var.get()

        self.agent.params["heavy_layers"] = self.heavy_layers_var.get()
        self.agent.params["heavy_ctx"] = self.heavy_ctx_var.get()
        self.agent.params["heavy_temp"] = self.heavy_temp_var.get()
        self.agent.params["heavy_top_p"] = self.heavy_top_p_var.get()
        self.agent.params["heavy_top_k"] = self.heavy_top_k_var.get()
        self.agent.params["heavy_repeat"] = self.heavy_repeat_var.get()
        self.agent.params["heavy_tokens"] = self.heavy_tokens_var.get()

        
        self.agent.params["vram_threshold_mb"] = self.vram_threshold_var.get()
        self.agent.params["encoder_on_gpu"] = self.encoder_gpu_var.get()
        self.agent.params["tied_on_gpu"] = self.tied_gpu_var.get()
        self.agent.params["global_kv_cache"] = self.cache_compress_var.get()

        
        names_rev = {"Quick-core (270M)": "light", "Coremal (1B)": "med", "Intelli-Core (4B)": "heavy"}
        new_core = names_rev.get(self.model_var.get(), "med")
        self.agent.params["active_core"] = new_core
        
        map_pfx = new_core if new_core in ["light", "med", "heavy"] else "med"
        self.agent.params["n_gpu_layers"] = self.agent.params.get(f"{map_pfx}_layers", 0)
        self.agent.params["n_ctx"] = self.agent.params.get(f"{map_pfx}_ctx", 4096)
        self.agent.params["temperature"] = self.agent.params.get(f"{map_pfx}_temp", 0.3)
        self.agent.params["top_p"] = self.agent.params.get(f"{map_pfx}_top_p", 0.9)
        self.agent.params["top_k"] = self.agent.params.get(f"{map_pfx}_top_k", 50)
        self.agent.params["repeat_penalty"] = self.agent.params.get(f"{map_pfx}_repeat", 1.15)
        self.agent.params["max_tokens"] = self.agent.params.get(f"{map_pfx}_tokens", 512)
        
        # Increment Config Version for Locking
        current_v = self.agent.params.get("config_version", 0)
        self.agent.params["config_version"] = current_v + 1

        self.agent.save_params()
        self.swap_model()

    def reset_defaults(self):
        self.light_layers_var.set(18)
        self.light_ctx_var.set(8192)
        self.light_temp_var.set(0.2)
        self.light_top_p_var.set(0.95)
        self.light_top_k_var.set(64)
        self.light_repeat_var.set(1.1)
        self.light_tokens_var.set(256)

        self.med_layers_var.set(26)
        self.med_ctx_var.set(16384)
        self.med_temp_var.set(0.35)
        self.med_top_p_var.set(0.95)
        self.med_top_k_var.set(64)
        self.med_repeat_var.set(1.15)
        self.med_tokens_var.set(512)

        self.heavy_layers_var.set(34)
        self.heavy_ctx_var.set(4096)
        self.heavy_temp_var.set(0.4)
        self.heavy_top_p_var.set(0.95)
        self.heavy_top_k_var.set(64)
        self.heavy_repeat_var.set(1.2)
        self.heavy_tokens_var.set(512)


        self.vram_threshold_var.set(600)
        self.cache_compress_var.set("Auto")
        self.log("[SYSTEM] Parameters reset to defaults. Click Apply to save.")


    def swap_model(self):
        lvl = int(self.agent.params.get("persona_level", 7))
        if lvl != 7:
            self.log(f"[SYSTEM] Core swap disabled — main.py bridge active for Level {lvl}.")
            return
        
        selection = self.model_var.get()
        mapping = {"Quick-core (270M)": "light", "Coremal (1B)": "med", "Intelli-Core (4B)": "heavy"}
        tier = mapping.get(selection, "med")
        
        self.log(f"[SYSTEM] Initiating reboot to: {tier.upper()}...")
        self.update_avatar("subdued_serenity.png")
        self.agent.awake = False
        self.agent.params["active_core"] = tier
        self.agent.save_params()
        
        # OFF-LOAD Llama Core from main.py before starting T5 engine to prevent VRAM dual load!
        self.agent.offload_model()
        self.offload_engine()
        time.sleep(2.0) 

        env = os.environ.copy()
        env["SERENITY_CORE"] = tier
        env["SERENITY_SPAWNED_BY_UI"] = "1"
        
        python_exe = sys.executable
        if "pythonw" in python_exe.lower():
            python_exe = python_exe.lower().replace("pythonw", "python")
        if not python_exe.lower().endswith(".exe") and os.name == "nt":
            python_exe += ".exe"
            
        engine_script = os.path.join(self.agent.live_dir, "Engine", "t5_server.py")
        
        try:
            self.log(f"[SYSTEM] Dispatching: {python_exe} {engine_script}")
            
            ts_start = datetime.now().strftime("%Y%m%d_%H%M%S")
            spawn_log_path = os.path.join(self.logs_dir, f"EngineSpawn_{ts_start}.log")
            
            f = None
            try:
                f = open(spawn_log_path, "a", encoding="utf-8")
                f.write(f"\n--- Spawn Attempt: {datetime.now()} ---\n")
                f.write(f"Cmd: {python_exe} {engine_script}\n")
                f.flush()
            except Exception as le:
                self.log(f"[SYSTEM] Warning: Could not open spawn log: {le}")

            creation_flags = 0x08000000 if os.name == 'nt' else 0
            p = subprocess.Popen(
                [python_exe, engine_script], 
                cwd=self.agent.live_dir, 
                env=env, 
                creationflags=creation_flags,
                stdout=f if f else None, 
                stderr=f if f else None
            )
            self.agent.live_agent_process = p
            self.log("[SYSTEM] Reboot signal sent. Waiting for engine wake-up...")
        except Exception as e:
            msg = f"Reboot Failure: {e}"
            self.agent.log_error(msg)
            self.log(f"[SYSTEM] ERROR: {msg}")
        
        threading.Thread(target=self.agent.wait_for_engine, args=(self,), daemon=True).start()

    def toggle_vram_model(self):


        llama_loaded = self.agent.model is not None
        t5_active = False
        if getattr(self.agent, "live_agent_process", None) is not None and self.agent.live_agent_process.poll() is None:
            t5_active = True
        else:
            try:
                for conn in psutil.net_connections():
                    laddr = getattr(conn, 'laddr', None)
                    if laddr and hasattr(laddr, 'port') and laddr.port == 8001:
                        t5_active = True
                        break
            except: pass
        
        if llama_loaded and t5_active:
            self.log("[SYSTEM] Dual-load collision resolved: Offloading all engines...")
            self.agent.offload_model()
            self.offload_engine()
        elif llama_loaded:
            self.log("[SYSTEM] Offloading Llama Core...")
            self.agent.offload_model()
        elif t5_active:
            self.log("[SYSTEM] Shutting down T5 Engine...")
            self.offload_engine()
        else:
            lvl = int(self.agent.params.get("persona_level", 5))
            if lvl == 7:
                self.log("[SYSTEM] Loading T5 Engine...")
                self.swap_model()
            else:
                self.log(f"[SYSTEM] Loading Llama Core (Level {lvl})...")
                def bg_load():
                    self.agent.model_swap(target_level=lvl)
                threading.Thread(target=bg_load, daemon=True).start()

    def update_avatar(self, filename):
        if getattr(self, "current_avatar_name", None) == filename:
            return
        path = os.path.join(self.media_dir, filename)
        if not os.path.exists(path):
            path = os.path.join(self.media_dir, "transcendent_serenity.png")
            if not os.path.exists(path):
                self.log(f"[UI] Avatar photo missing: {filename}")
                return

        try:
            img = Image.open(path)
            img.thumbnail((350, 350), Image.Resampling.LANCZOS)
            self.avatar_photo = ImageTk.PhotoImage(img) # type: ignore
            self.image_label.configure(image=self.avatar_photo) # type: ignore
            self.current_avatar_name = filename
        except Exception as e:
            self.log(f"[UI] Avatar load error: {e}")

    def update_console(self, text):
        """Append text to console in real-time for TTFT improvements."""
        was_at_bottom = self.console.yview()[1] >= 0.999
        self.console.insert(tk.END, text)
        if was_at_bottom:
            self.console.see(tk.END)
        self.root.update_idletasks()

    def log(self, message):
        try:
            was_at_bottom = self.console.yview()[1] >= 0.999
            self.console.insert(tk.END, message + "\n")
            if was_at_bottom:
                self.console.see(tk.END)
        except (tk.TclError, RuntimeError): pass

    def on_closing(self):
        try:
            self.log("[SYSTEM] Shimming shutdown sequence...")
            self.offload_engine()
            time.sleep(0.5)
        except: pass
        os._exit(0)



    def run(self):
        self.root.mainloop()

class SerenityAgent:
    def __init__(self):
        current_dir = os.path.abspath(os.path.dirname(__file__))
        parts = current_dir.replace('\\', '/').split('/')
        if "Live" in parts:
            idx = len(parts) - 1 - list(reversed(parts)).index("Live")
            self.live_dir = "/".join([parts[i] for i in range(idx + 1)])
        else:
            self.live_dir = current_dir
            
        self.system_dir = os.path.join(self.live_dir, "System")
        self.params_file = os.path.join(self.system_dir, "params.json")
        self.params = ThreadSafeDict()
        self.base_pause_threshold = 0.8
        self.load_params()
        self.awake = False
        self._monitor_active = False 
        self.microphone: Optional[Any] = None 
        
        self.recognizer = sr.Recognizer()
        
        # TTS Queue and Worker - Refactored to initialize Pyttsx3 ONLY ONCE
        import queue
        self.speech_queue = queue.Queue()
        self.stop_speaking = threading.Event()
        self._speaking = False
        
        self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.speech_thread.start()
        
        self.mic_lock = threading.Lock()
        try:
            self.microphone = SoundDeviceMicrophone() # type: ignore
        except OSError:
            print("[SERENITY LIVE] - WARNING: No Default Input Device Available.")
            self.microphone = None
            
        self._is_querying = False
        self.current_mood = "meditating" 
        
        self.persona_map = {
            "meditating": {
                "idle": "Meditating_Serenity.png",
                "listen": "The_Wise_Listener.png",
                "think": "serenity_thinking.png",
                "speak": "explain_wise.png",
                "loading": "serenity_pondering.png"
            },
            "transcendent": {
                "idle": "transcendent_serenity.png",
                "listen": "serenity_pondering.png",
                "think": "serenity_idea.png",
                "speak": "explain_direct.png",
                "loading": "serenity_thinking.png"
            }
        }
        
        self.logs_dir = os.path.join(self.live_dir, "Logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        
        self.get_available_models = lambda: []
        self.swap_model = lambda m: False
        self.handle_vs_request = lambda m, d=False: False

        self.session_history = ThreadSafeList()
        self.context_limit = 4096
        self.diary_file = os.path.join(self.logs_dir, "Serenity_Diary.md")
        
        try:
            sys.path.append(os.path.join(self.live_dir, "System"))
            from tools import MemoryEngine
            self.memory = MemoryEngine(db_path=os.path.join(self.logs_dir, "vector_memory"))
        except Exception as e:
            print(f"Memory Engine failed to load: {e}")
            self.memory = None

    def store_memory(self, text, metadata=None):
        if self.memory:
            self.memory.save(text, metadata)
            self.log_thought(f"[MEMORY STORED] {text[:100]}...")

    def recall_memories(self, query, top_k=4):
        if self.memory:
            res = self.memory.recall(query, n_results=int(top_k))
            if res:
                return f"Relevant memories: {res}"
        return ""

    def _speech_worker(self):
        """A persistent worker that reads from the queue. Avoids SAPI5 crashes."""
        import pyttsx3 # type: ignore
        try:
            engine = pyttsx3.init(driverName='sapi5' if os.name == 'nt' else None)
            voices = engine.getProperty('voices')
            if len(voices) > 1: engine.setProperty('voice', voices[1].id)
            engine.setProperty('rate', 165)
        except Exception as e:
            self.log_error(f"TTS Init Error: {e}")
            return
            
        while True:
            try:
                text = self.speech_queue.get()
                if text is None: break
                
                # Check interruption status
                if self.stop_speaking.is_set():
                    self.speech_queue.task_done()
                    continue

                self._speaking = True
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception as e:
                    self.log_error(f"TTS Worker execution error: {e}")
                finally:
                    self._speaking = False
                    self.speech_queue.task_done()
            except Exception:
                time.sleep(0.1)

    def speak(self, text, ui=None):
        """Enqueues text for the continuous background speech worker."""
        self.stop_speaking.clear()
        
        if not getattr(self, "_monitor_active", False):
            self._monitor_active = True
            threading.Thread(target=self._interrupt_monitor, args=(ui,), daemon=True).start()
            
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sentence in sentences:
            if sentence.strip():
                self.speech_queue.put(sentence.strip())

    def _interrupt_monitor(self, ui=None):
        import audioop # type: ignore
        CHUNK = 1024
        THRESHOLD = 2000 
        
        def callback(indata, frames, time, status):
            if not self._speaking:
                raise sd.CallbackStop()
            
            rms = audioop.rms(indata, 2)
            if rms > THRESHOLD:
                self.stop_speaking.set()
                # Dump the queue to immediately silence remaining buffered chunks
                with self.speech_queue.mutex:
                     self.speech_queue.queue.clear()
                raise sd.CallbackStop()

        try:
            with sd.RawInputStream(samplerate=16000, channels=1, dtype='int16', 
                               blocksize=CHUNK, callback=callback):
                while True: # Monitor lifecycle matches the session now
                    time.sleep(0.1)
        except Exception:
            pass

    def _is_complex_query(self, text):
        if self.params.get("active_core", "med") != "heavy":
            return False
            
        complex_keywords = ["analyze", "summarize", "research", "compare", "plan",
                            "complex", "detailed", "explain", "why", "how does",
                            "what if", "describe", "break down", "help me understand"]
        return any(kw in text.lower() for kw in complex_keywords)

    def log_thought(self, text):
        try:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            scratchpad_path = os.path.join(self.logs_dir, "scratchpad.txt")
            
            if os.path.exists(scratchpad_path) and os.path.getsize(scratchpad_path) > 50 * 1024:
                self.archive_thoughts(scratchpad_path)
                
            with open(scratchpad_path, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] {text}\n")
        except Exception:
            pass

    def archive_thoughts(self, current_path):
        try:
            history_dir = os.path.join(self.live_dir, "History")
            os.makedirs(history_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_path = os.path.join(history_dir, f"thought_history_{ts}.txt")
            
            import shutil
            shutil.move(current_path, archive_path)
            open(current_path, 'w').close()
            self.log_sys(f"Thoughts archived to {archive_path}")
        except Exception as e:
            self.log_error(f"Archival failed: {e}")

    def log_error(self, text):
        try:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(os.path.join(self.logs_dir, "error_log.txt"), "a", encoding="utf-8") as f:
                f.write(f"[{ts}] {text}\n")
        except Exception:
            pass

    def log_sys(self, text):
        try:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(os.path.join(self.logs_dir, "SysLog.txt"), "a", encoding="utf-8") as f:
                f.write(f"[{ts}] {text}\n")
        except Exception:
            pass
            
    def log_subconscious(self, text):
        try:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(os.path.join(self.logs_dir, "subconscious_journal.txt"), "a", encoding="utf-8") as f:
                f.write(f"[{ts}] {text}\n")
        except Exception:
            pass

    def load_params(self):
        self.params = ThreadSafeDict()
        if os.path.exists(self.params_file):
            try:
                with open(self.params_file, 'r') as f: self.params = ThreadSafeDict(json.load(f))
            except json.JSONDecodeError:
                pass
        self.base_pause_threshold = float(self.params.get("learned_pause_threshold", 0.8))

    def save_params(self):
        try:
            with open(self.params_file, 'w') as f: json.dump(self.params, f, indent=4)
        except Exception:
            pass

    def adapt_cadence(self, audio, transcript):
        if not self.params.get("learning_enabled", True): 
            return
            
        audio_duration = len(audio.frame_data) / (audio.sample_rate * audio.sample_width)
        word_count = len(transcript.split())
        if word_count > 2 and audio_duration > 0:
            words_per_second = word_count / audio_duration
            target_threshold = max(0.6, min(2.5, 3.0 / max(words_per_second, 0.5)))
            old_threshold = self.base_pause_threshold
            self.base_pause_threshold = float((0.9 * self.base_pause_threshold) + (0.1 * target_threshold))
            val = round(self.base_pause_threshold, 3) # type: ignore
            self.params["learned_pause_threshold"] = str(val)
            self.save_params()
            
            if abs(old_threshold - self.base_pause_threshold) > 0.05:
                speed = "faster" if target_threshold < old_threshold else "slower"
                self.log_subconscious(f"[CADENCE ADAPTATION] User speaking {speed} ({words_per_second:.1f} wps). Adjusted listener gate to {val}s.")

    def estimate_tokens(self, text):
        return int(len(text.split()) * 1.5)

    def fuzzy_json_decode(self, text, stream_speech=""):
        """Robust XML tag extraction. Far more reliable than loosely formatted JSON/plaintext strings."""
        import json as _json
        import re as _re
        
        thought = "No internal thought generated."
        action = "none"
        directive = None
        speech = stream_speech if stream_speech else text

        t_m = _re.search(r'<thought>(.*?)</thought>', text, _re.DOTALL | _re.IGNORECASE)
        a_m = _re.search(r'<action>(.*?)</action>', text, _re.DOTALL | _re.IGNORECASE)
        d_m = _re.search(r'<directive>(.*?)</directive>', text, _re.DOTALL | _re.IGNORECASE)
        s_m = _re.search(r'<speech>(.*?)</speech>', text, _re.DOTALL | _re.IGNORECASE)

        def _valid(val):
            if not val: return False
            v = val.strip().lower()
            return v not in ["", "none", "null", "undefined"]

        if t_m and _valid(t_m.group(1)): 
            thought = t_m.group(1).strip()
            
        if a_m and _valid(a_m.group(1)): 
            action = a_m.group(1).strip()
            
        if d_m and _valid(d_m.group(1)):
            d_str = d_m.group(1).strip()
            try: directive = _json.loads(d_str)
            except: directive = d_str # Fallback to raw string if JSON fail
            
        if s_m and _valid(s_m.group(1)):
            parsed_speech = s_m.group(1).strip()
            if len(parsed_speech) > 1:
                speech = parsed_speech
        
        # Comprehensive fallback for speech if XML tags failed completely
        if not _valid(speech):
            # If we have a stream_speech, use it. Otherwise, look for any text AFTER </action> or </thought>
            if _valid(stream_speech):
                speech = stream_speech
            else:
                fallback_speech = text
                for tag in ["</speech>", "</action>", "</thought>", "</directive>"]:
                    if tag in text.lower():
                        parts = text.lower().split(tag)
                        if len(parts) > 1 and _valid(parts[-1]):
                            fallback_speech = parts[-1].strip()
                            break
                speech = fallback_speech if _valid(fallback_speech) else "(No audible response generated)"
            
        # Clean up speech from artifacts
        speech = _re.sub(r'<unused\d+>', '', speech).strip()
        speech = speech.replace('"}', '').replace('}','').strip()

        return {"thought": thought, "action": action, "directive": directive, "speech": speech}

    def compact_session(self, ui):
        threading.Thread(target=self._compact_session_task, args=(ui,), daemon=True).start()

    def _compact_session_task(self, ui):
        ui.log("[SYSTEM] Context threshold reached (75%). Activating Nahida’s Garden...")
        
        session_text = "\n".join([f"{m['role']}: {m['content']}" for m in self.session_history])
        prompt = (
            "Summarize this session into exactly five Markdown buckets: "
            "### Directives: active constraints. "
            "### Lore: new updates to the Chronicles. "
            "### Hardware State: GPU/VRAM notes. "
            "### Decisions: validated technical choices. "
            "### Unresolved: pending tasks. "
            f"\nSession History:\n{session_text}"
        )
        
        headers = {"x-api-key": "REVOKED"}
        try:
            resp = requests.post(SERENITY_ENGINE_URL, json={"text": prompt, "max_tokens": 1024}, headers=headers, timeout=60)
            if resp.status_code == 200:
                summary = resp.json().get("speech", "Compaction failed.")
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                entry = f"\n\n# Session Diary - {ts}\n{summary}\n"
                
                with open(self.diary_file, "a", encoding="utf-8") as f:
                    f.write(entry)
                    
                import re
                try:
                    lore_match = re.search(r'### Lore:(.*?)(?=###|$)', summary, re.DOTALL)
                    dec_match = re.search(r'### Decisions:(.*?)(?=###|$)', summary, re.DOTALL)
                    
                    def _is_valid_entry(text_val):
                        t = text_val.strip()
                        return t and "none" not in t.lower() and "n/a" not in t.lower() and len(t) > 10
                        
                    extracted_memories = []
                    if lore_match and _is_valid_entry(lore_match.group(1)):
                        extracted_memories.append(f"Established Lore: {lore_match.group(1).strip()}")
                    if dec_match and _is_valid_entry(dec_match.group(1)):
                        extracted_memories.append(f"Key Decision: {dec_match.group(1).strip()}")
                        
                    for mem in extracted_memories:
                        self.store_memory(mem, metadata={"type": "compacted_logic", "timestamp": ts})
                        self.log_subconscious(f"[CONSOLIDATION] Stored Logic in Vector Space: {mem[:80]}...")
                except Exception as ex:
                    self.log_error(f"Failed to extract long-term memory elements: {ex}")
                
                ui.log("[SYSTEM] Session logic compacted to Serenity_Diary.md and Vector Memory.")
                history_list = list(self.session_history)
                self.session_history = history_list[-6:] if len(history_list) > 6 else history_list # type: ignore
        except Exception as e:
            self.log_error(f"Compaction error: {e}")

    def query_engine(self, text_input, image_b64=None, memory_context=None, ui=None):
        headers = {"x-api-key": "REVOKED"}
        
        if memory_context is None:
            try:
                 memory_context = self.recall_memories(text_input)
            except Exception: pass

        if memory_context:
            enriched_input = f"[Context: {memory_context}] {text_input}"
        else:
            enriched_input = text_input

        # Atomic Version Handshake before query
        current_v = self.params.get("config_version", 0)
        try:
            r = requests.get("http://127.0.0.1:8001/diagnose", headers=headers, timeout=2)
            if r.status_code == 200:
                engine_v = r.json().get("config_version", -1)
                if engine_v != current_v:
                    self.log_sys(f"Version mismatch in query_engine (UI: v{current_v}, Engine: v{engine_v}). Synchronizing...")
                    self.wait_for_engine(ui)
            else:
                self.wait_for_engine(ui)
        except:
            self.wait_for_engine(ui)
        
        self.log_thought(f"QUERY >> {text_input}")
        self.session_history.append({"role": "user", "content": enriched_input})
        
        total_est = sum(self.estimate_tokens(m["content"]) for m in self.session_history)
        if total_est > (self.context_limit * 0.75):
            self.compact_session(ui)

        last_error = ""
        prefill_start = time.time()
        for attempt in range(10):
            try:
                STREAM_URL = SERENITY_ENGINE_URL.replace("/analyze", "/stream")
                core = self.params.get("active_core", "med")
                pfx = f"{core}_" if core in ["light", "med", "heavy"] else "med_"
                
                payload = {
                    "text": enriched_input, 
                    "max_tokens": self.params.get(f"{pfx}tokens", self.params.get("max_tokens", 512)), 
                    "temperature": self.params.get(f"{pfx}temp", self.params.get("temperature", 0.35)),
                    "top_p": self.params.get(f"{pfx}top_p", self.params.get("top_p", 0.9)),
                    "top_k": self.params.get(f"{pfx}top_k", self.params.get("top_k", 50)),
                    "repetition_penalty": self.params.get(f"{pfx}repeat", self.params.get("repeat_penalty", 1.15))
                }
                if len(self.session_history) > 1:
                    payload["history"] = self.session_history[:-1]
                    
                if image_b64:
                    payload["image_b64"] = image_b64
                
                response = requests.post(STREAM_URL, json=payload, headers=headers, timeout=120, stream=True)
                
                full_response_str = ""
                speech_captured = ""
                in_speech_block = False
                speech_buffer = ""
                
                ui.root.after(0, ui.log, "\nSerenity: ") # Initial label
                
                for line in response.iter_lines():
                    if not line: continue
                    token = line.decode("utf-8")
                    if not token: continue
                    
                    full_response_str += token
                    
                    # Instantaneous visual TTFT parsing using XML tags
                    if not in_speech_block:
                        if "<speech>" in full_response_str:
                            in_speech_block = True
                            chunk = full_response_str.split("<speech>")[-1]
                            speech_buffer += chunk
                            if chunk: ui.root.after(0, ui.update_console, chunk)
                    else:
                        if "</speech>" in token:
                            chunk = token.replace("</speech>", "")
                            speech_buffer += chunk
                            if chunk: ui.root.after(0, ui.update_console, chunk)
                            break # Terminate stream parsing early
                        else:
                            speech_buffer += token
                            ui.root.after(0, ui.update_console, token)
                        
                        # TTS Chunking Evaluator
                        word_count = len(speech_buffer.split())
                        if any(p in speech_buffer for p in ['. ', '? ', '! ', '.\n']) or word_count > 15:
                            for p in ['. ', '? ', '! ', '.\n']:
                                if p in speech_buffer:
                                     parts = speech_buffer.rsplit(p, 1)
                                     sentence = parts[0] + p
                                     if len(sentence.strip()) > 3:
                                         self.speak(sentence.strip(), ui)
                                         speech_captured += sentence
                                     speech_buffer = parts[1]
                                     break
                
                if speech_buffer.strip():
                    self.speak(speech_buffer.strip(), ui)
                    speech_captured += speech_buffer
                
                response.raise_for_status() 

                # --- XML DECODE PHASE ---
                result_data = self.fuzzy_json_decode(full_response_str, stream_speech=speech_captured)
                thought = result_data["thought"]
                action = result_data["action"]
                directive = result_data["directive"]
                speech = result_data["speech"]
                
                # Format into history properly
                resp_text = ""
                if thought and thought not in ["None", "No internal thought generated."]:
                    resp_text += f"<thought>{thought}</thought>\n"
                if action and action != "none":
                    resp_text += f"<action>{action}</action>\n"
                if directive:
                    import json as _json
                    resp_text += f"<directive>{_json.dumps(directive)}</directive>\n"
                resp_text += f"<speech>{speech}</speech>"
                
                self.session_history.append({"role": "assistant", "content": resp_text})
                
                self.log_thought(f"THOUGHT << {thought}")
                self.log_thought(f"ACTION << {action}")
                if directive: self.log_thought(f"DIRECTIVE << {directive}")
                self.log_thought(f"SPEECH << {speech}")
                
                return result_data
                
            except requests.ConnectionError:
                last_error = f"Engine not reachable (attempt {attempt+1}/20)"
                self.log_error(last_error)
                time.sleep(1.5)
            except requests.Timeout:
                last_error = "Engine timed out after 120s."
                self.log_error(last_error)
                break
            except requests.HTTPError as e:
                last_error = f"Engine HTTP {e.response.status_code}: {e.response.text[:200]}"
                self.log_error(last_error)
                break
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                self.log_error(last_error)
                break
        
        self.log_error(f"QUERY FAILED: {last_error}")
        return {"speech": f"[Engine Error] {last_error}", "thought": "Engine failure.", "action": "none"}

    def _bg_query_worker(self, text, ui):
        self._is_querying = True
        image_b64 = None
        memory_context = None
        try:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                memory_future = executor.submit(self.recall_memories, text)
                
                vision_keywords = ["take a look", "check this out", "see the screen", "look at this", "what is this"]
                context_prefix = ""
                if any(kw in text.lower() for kw in vision_keywords):
                    ui.root.after(0, ui.log, "[VISION] Opening eyes to see the screen...")
                    vision_data = self._tool_capture_screen()
                    if isinstance(vision_data, dict):
                        desc = vision_data.get("description", "")
                        image_b64 = vision_data.get("image_b64")
                    else:
                        desc = vision_data
                    context_prefix = f"[VISUAL CONTEXT: {desc}] "
                    self.log_thought(f"[VISION] Captured: {desc}")
                    self.store_memory(f"Screen Capture Analysis: {desc}", metadata={"type": "vision", "timestamp": str(time.time())})
                
                try:
                    memory_context = memory_future.result(timeout=4.0)
                except Exception as e:
                    self.log_error(f"Async Memory Recall timeout/failure: {e}")
            
            ui.root.after(0, ui.start_inference_eta, text)
            
            if self._is_complex_query(text):
                self.speak("Hmm...", ui) 
                ui.root.after(0, ui.log, "Serenity: [Thinking Deeply]")

            query_text = context_prefix + text
            ai_response = self.query_engine(query_text, image_b64=image_b64, memory_context=memory_context, ui=ui)
            
            thought = ai_response.get("thought", "")
            if thought and len(thought) > 10:
                self.store_memory(f"Thought on '{text}': {thought}", metadata={"type": "thought", "query": text})
            action = ai_response.get("action", "none")
            directive = ai_response.get("directive", None)
            
            if thought and thought.lower() not in ["no internal thought generated.", "none", "", "null"]:
                ui.root.after(0, ui.log, f"Thought: {thought}")
                import random
                placeholders = ["reading the user", "asking the user", "user intent", "thinking about", "[", "]", "(", ")", "logic here", "analysis of", "no internal thought", "thought"]
                if not any(p in thought.lower() for p in placeholders):
                    if random.random() < 0.35: 
                        ui.root.after(0, ui.log, f"Serenity (Muttering): {thought}")
                        self.speak(f"Hmm... {thought}", ui)
                        time.sleep(1.5)
                else:
                    ui.root.after(0, ui.log, f"Serenity (Thinking...): {thought}")
            
            ui.root.after(0, ui.stop_inference_eta)
            ui.root.after(0, ui.update_avatar, self.persona_map[self.current_mood]["speak"])
            ui.root.after(0, ui.update_avatar, self.persona_map[self.current_mood]["idle"])
            
            if (action and str(action).lower() != "none") or directive:
                threading.Thread(target=self.execute_agent_action_async, args=(action, directive, text, ui), daemon=True).start()
                
        finally:
            self._is_querying = False
            ui.root.after(0, ui.stop_inference_eta)
            self.awake = True

    def handle_impatience(self, ui):
        phrases = [
            "Just a moment, I'm thinking...",
            "Working on it, please hold.",
            "I hear you, almost there.",
            "Processing your request now.",
            "I'm still here, just a second."
        ]
        import random
        resp = random.choice(phrases)
        ui.root.after(0, ui.log, f"Serenity (Thinking): {resp}")
        self.speak(resp, ui)
        
    def _inject_silent_response(self, text, ui, tag="SYSTEM"):
        import time
        time.sleep(1.2)
        formatted_text = f"[{tag}: {text}]"
        threading.Thread(target=self._bg_query_worker, args=(formatted_text, ui), daemon=True).start()

    def execute_agent_action_async(self, action_string, directive, original_prompt, ui):
        try:
            action = str(action_string).strip() if action_string else "none"
            
            if directive and isinstance(directive, dict):
                d_action = directive.get("action")
                if d_action == "reboot":
                    self.speak("Hmm...", ui) 
                    new_core = directive.get("core")
                    if new_core:
                        ui.root.after(0, ui.model_var.set, {
                            "light": "Quick-core (270M)", 
                            "med": "Coremal (1B)", 
                            "heavy": "Intelli-Core (4B)"
                        }.get(new_core, "Coremal (1B)"))
                    
                    if "temperature" in directive:
                        val = float(directive["temperature"])
                        if new_core == "heavy": ui.root.after(0, ui.heavy_temp_var.set, val)
                        elif new_core == "light": ui.root.after(0, ui.light_temp_var.set, val)
                        else: ui.root.after(0, ui.med_temp_var.set, val)
                        
                    if "top_p" in directive:
                        val = float(directive["top_p"])
                        if new_core == "heavy": ui.root.after(0, ui.heavy_top_p_var.set, val)
                        elif new_core == "light": ui.root.after(0, ui.light_top_p_var.set, val)
                        else: ui.root.after(0, ui.med_top_p_var.set, val)
                        
                    if "top_k" in directive:
                        val = int(directive["top_k"])
                        if new_core == "heavy": ui.root.after(0, ui.heavy_top_k_var.set, val)
                        elif new_core == "light": ui.root.after(0, ui.light_top_k_var.set, val)
                        else: ui.root.after(0, ui.med_top_k_var.set, val)
                        
                    if "repeat_penalty" in directive or "repetition_penalty" in directive:
                        val = float(directive.get("repeat_penalty", directive.get("repetition_penalty")))
                        if new_core == "heavy": ui.root.after(0, ui.heavy_repeat_var.set, val)
                        elif new_core == "light": ui.root.after(0, ui.light_repeat_var.set, val)
                        else: ui.root.after(0, ui.med_repeat_var.set, val)
                        
                    self.log_thought(f"[SYSTEM] Agent self-assigned reboot directive: {directive}")
                    ui.root.after(0, ui.save_and_reboot)
                    
                    def _re_queue():
                        import time
                        time.sleep(12)
                        self.wait_for_engine(ui)
                        msg = f"[SYSTEM: Parameter adjustment completed. Please answer the user's previous query now:] {original_prompt}"
                        self._bg_query_worker(msg, ui)
                    threading.Thread(target=_re_queue, daemon=True).start()
                    return

            if action.startswith("chrome_search") or action.startswith("web_search"):
                self.speak("One second, let me check that for you.", ui)
                try:
                    import sys
                    sys.path.append(os.path.join(self.live_dir, "System"))
                    from tools import BrowserTools
                    bt = BrowserTools()
                    import re
                    match = re.search(r'\((.*?)\)', action)
                    query = match.group(1).strip('"\'') if match else original_prompt.replace("search", "").strip()
                    res = bt.search_web(query)
                    self.log_thought(f"[SYSTEM] Chrome Search: {res[:100]}...")
                    self._inject_silent_response(res, ui, tag="WEB_RESULT")
                except Exception as e:
                    self.log_error(f"Search tool failed: {e}")

            elif action.startswith("play_media") or action.startswith("play("):
                self.speak("Picking out a song for you...", ui)
                try:
                    import sys
                    sys.path.append(os.path.join(self.live_dir, "System"))
                    from tools import BrowserTools
                    bt = BrowserTools()
                    import re
                    match = re.search(r'\((.*?)\)', action)
                    query = match.group(1).strip('"\'') if match else original_prompt.replace("play", "").strip()
                    res = bt.play_media(query)
                    self.log_thought(f"[SYSTEM] Media Action: {res}")
                    self._inject_silent_response(res, ui, tag="MEDIA_FINAL")
                except Exception as e:
                    self.log_error(f"Media tool failed: {e}")

            elif action.startswith("vision_search"):
                self.speak("Looking closer at your screen now...", ui)
                try:
                    import sys
                    sys.path.append(os.path.join(self.live_dir, "System"))
                    from tools import BrowserTools
                    bt = BrowserTools()
                    import re
                    match = re.search(r'\((.*?)\)', action)
                    query = match.group(1).strip('"\'') if match else original_prompt.replace("vision", "").strip()
                    res = bt.search_web(query)
                    self.log_thought(f"[SYSTEM] Vision Search: {res[:100]}...")
                    self._inject_silent_response(res, ui, tag="WEB_RESULT")
                except Exception as e:
                    self.log_error(f"Vision Search tool failed: {e}")

            elif action.startswith("save_memory"):
                try:
                    import re
                    match = re.search(r'\((.*?)\)', action)
                    content = match.group(1).strip('"\'') if match else original_prompt
                    self.store_memory(content)
                    self.log_thought(f"[SYSTEM] Memory Saved: {content[:50]}...")
                except Exception as e:
                    self.log_error(f"Memory save failed: {e}")

            elif action.startswith("change_persona("):
                import re
                match = re.search(r'\d+', action)
                if match:
                    lvl = int(match.group())
                    self.params["persona_level"] = str(lvl)
                    if int(lvl) >= 7: self.current_mood = "transcendent"
                    else: self.current_mood = "meditating"
                    self.save_params()
                    self.log_thought(f"[SYSTEM] Agent requested persona shift to Level {lvl} ({self.current_mood}).")
            elif action == "offload_engine":
                requests.post("http://127.0.0.1:8001/shutdown", headers={"x-api-key": "REVOKED"}, timeout=2)
                self.log_thought("[SYSTEM] Agent self-offloaded engine.")
            elif action == "disable_learning":
                self.params["learning_enabled"] = "False"
                self.save_params()
                self.log_thought("[SYSTEM] Agent self-disabled continuous learning.")
            elif action.startswith("store_memory("):
                import re
                match = re.search(r'\((.+?)\)', action)
                if match:
                    self.store_memory(match.group(1))
            elif action == "monitor_system":
                stats = self._tool_monitor_system()
                self.log_thought("[SYSTEM] Agent requested system monitor.")
                self._inject_silent_response(f"[System Monitoring Output: {stats}]", ui)
            elif action == "clean_storage":
                res = self._tool_clean_storage()
                self.log_thought(f"[SYSTEM] Agent requested storage clean: {res}")
            elif action.startswith("view_photo("):
                import re
                match = re.search(r'\((.*?)\)', action)
                if match:
                    path = match.group(1)
                    res = self._tool_view_photo(path)
                    self.log_thought(f"[SYSTEM] Agent viewed photo: {res}")
                    self._inject_silent_response(f"[Photo Output: {res}]", ui)
            elif action != "none":
                self.log_thought(f"[SYSTEM] Unknown action requested: {action}")
        except Exception as e:
            self.log_error(f"Action execution failed: {e}")

    def _tool_monitor_system(self):
        try:
            import sys
            sys.path.append(os.path.join(self.live_dir, "System"))
            from tools import SystemMonitor
            monitor = SystemMonitor()
            stats = monitor.get_stats()
            self.log_thought(f"[TOOL OUTPUT] {stats}")
            return stats
        except Exception as e:
            return f"Monitor failed: {e}"

    def _tool_clean_storage(self):
        import tempfile, shutil
        temp_dir = tempfile.gettempdir()
        freed = 0
        try:
            for item in os.listdir(temp_dir):
                item_path = os.path.join(temp_dir, item)
                try:
                    size = os.path.getsize(item_path) if os.path.isfile(item_path) else 0
                    if os.path.isfile(item_path): os.unlink(item_path)
                    elif os.path.isdir(item_path): shutil.rmtree(item_path)
                    freed += size
                except: pass
            msg = f"Storage cleaned. Freed {freed / (1024**2):.1f} MB."
        except Exception as e:
            msg = f"Storage clean failed: {e}"
        self.log_thought(f"[TOOL OUTPUT] {msg}")
        return msg

    def _tool_view_photo(self, path):
        import re
        path = re.sub(r'[\'"]', '', path).strip()
        if not os.path.exists(path):
            return f"Error: Cannot find photo at {path}"
        try:
            size_mb = os.path.getsize(path) / (1024**2)
            self.log_thought(f"[TOOL OUTPUT] Photo viewed at {path} ({size_mb:.2f} MB). Note: Full vision model API not connected yet, relaying metadata.")
            return f"Photo found at {path}. Size: {size_mb:.2f} MB. You can't see the exact pixels yet, but you know the file is there."
        except Exception as e:
            return f"Photo access error: {e}"

    def _tool_capture_screen(self):
        import base64
        import io
        try:
            screenshot = ImageGrab.grab()
            
            # Resize to 720p (Crucial for VRAM hygiene)
            screenshot.thumbnail((1280, 720), Image.Resampling.LANCZOS)
            
            save_path = os.path.join(self.logs_dir, "LastScreen.png")
            screenshot.save(save_path)
            
            buffered = io.BytesIO()
            screenshot.convert("RGB").save(buffered, format="JPEG", quality=80)
            img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            hwnd = ctypes.windll.user32.GetForegroundWindow() # type: ignore
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd) # type: ignore
            buff = ctypes.create_unicode_buffer(length + 1)
            ctypes.windll.user32.GetWindowTextW(hwnd, buff, length + 1) # type: ignore
            active_window = buff.value
            
            width, height = screenshot.size
            
            desc = f"User is viewing {active_window} at {width}x{height} resolution. Image saved to Logs/LastScreen.png."
            return {"description": desc, "image_b64": img_b64}
        except Exception as e:
            return f"Vision failed: {e}"

    def wait_for_engine(self, ui):
        lvl = int(self.params.get("persona_level", 7))
        mood = "transcendent" if lvl >= 7 else "meditating"
        ui.root.after(0, ui.update_avatar, self.persona_map[mood]["loading"])
        
        engine_up = False
        is_debug = os.environ.get("SERENITY_DEBUG") == "1"
        probe_timeout = 30 if is_debug else 2
        
        try:
            r = requests.get("http://127.0.0.1:8001/diagnose", headers={"x-api-key": "REVOKED"}, timeout=probe_timeout)
            if r.status_code == 200:
                engine_up = True
        except Exception:
            pass

        if not engine_up:
            engine_running = False
            for proc in psutil.process_iter(['name', 'cmdline']):
                try:
                    if proc.info['cmdline'] and any("t5_server.py" in s for s in proc.info['cmdline']):
                        engine_running = True
                        break
                except (psutil.NoSuchProcess, psutil.AccessDenied): continue
            
            if not engine_running:
                ui.root.after(0, ui.log, "[SYSTEM] Engine offline. Proactively booting backend...")
                try:
                    import subprocess as _subp
                    self.load_params() 
                    engine_script = os.path.join(self.live_dir, "Engine", "t5_server.py")
                    python_exe = sys.executable.replace("pythonw.exe", "python.exe")
                    env = os.environ.copy()
                    env["SERENITY_CORE"] = self.params.get("active_core", "med")
                    env["SERENITY_SPAWNED_BY_UI"] = "1"
                    _subp.Popen([python_exe, engine_script], cwd=self.live_dir, env=env, creationflags=0x08000000)
                except Exception as e:
                    self.log_error(f"Proactive engine launch failed: {e}")
            else:
                ui.root.after(0, ui.log, "[SYSTEM] Engine process detected. Synchronization in progress...")
        else:
            ui.root.after(0, ui.log, "[SYSTEM] Waiting for engine to finish loading...")

        attempts: int = 0
        while True:
            try:
                wait_timeout = 60 if is_debug else 5
                r = requests.get("http://127.0.0.1:8001/diagnose", headers={"x-api-key": "REVOKED"}, timeout=wait_timeout)
                if r.status_code == 200:
                    data = r.json()
                    engine_v = data.get("config_version", -1)
                    current_v = self.params.get("config_version", 0)
                    
                    if engine_v == current_v:
                        ui.root.after(0, ui.log, f"[SYSTEM] Engine synchronized (v{engine_v}) and responsive!")
                        self.log_sys(f"Engine connection established at v{engine_v}.")
                        return
                    else:
                        if attempts % 5 == 0:
                            ui.root.after(0, ui.log, f"[SYNC] Waiting for engine version match (UI: v{current_v}, Engine: v{engine_v})...")
                        time.sleep(1) # Extra cooldown for version drift recovery
            except Exception:
                attempts = int(attempts) + 1
                if attempts % 5 == 0:
                    ui.root.after(0, ui.log, f"[HEARTBEAT] Synchronizing... (Attempt {attempts}/90)")
                
                if attempts == 90 and not is_debug: 
                    ui.root.after(0, ui.log, f"[SYSTEM] Engine stall detected. Forcing rapid restart...")
                    self.log_sys(f"Rapid startup triggered (Attempt {attempts}).")
                    try:
                        import subprocess as _subp
                        self.load_params() 
                        engine_script = os.path.join(self.live_dir, "Engine", "t5_server.py")
                        python_exe = sys.executable.replace("pythonw.exe", "python.exe")
                        env = os.environ.copy()
                        env["SERENITY_CORE"] = self.params.get("active_core", "med")
                        env["SERENITY_SPAWNED_BY_UI"] = "1"
                        _subp.Popen([python_exe, engine_script], cwd=self.live_dir, env=env, creationflags=0x08000000)
                    except Exception as e:
                        self.log_error(f"Rapid engine launch failed: {e}")
                
                if attempts % 10 == 0:
                    ui.root.after(0, ui.log, f"[SYSTEM] VRAM initialization in progress... ({attempts * 2}s elapsed)")
            time.sleep(2)

    def listen_and_process(self, ui):
        self.wait_for_engine(ui)

        def log_mic_list():
            try:
                pass
            except Exception as e:
                self.log_error(f"Could not list microphones: {e}")

        log_mic_list()
        
        while self.microphone is None: # type: ignore
            ui.root.after(0, ui.log, "[SYSTEM] No microphone detected. Retrying...") # type: ignore
            try:
                self.microphone = SoundDeviceMicrophone() # type: ignore
                self.log_sys(f"Microphone initialized: {self.microphone}") # type: ignore
            except OSError as e:
                self.log_error(f"Mic init failed: {e}")
                time.sleep(3)
                continue

        try:
            with self.microphone as source: # type: ignore
                ui.root.after(0, ui.update_avatar, "serenity_pondering.png")
                ui.root.after(0, ui.log, "[SYSTEM] Calibrating microphone...")
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
                
                ui.root.after(0, ui.log, "[SYSTEM] Awaiting wake word.")
                _last_state = None 
                _awake_timeout_counter = 0 
                
                while True:
                    if self.microphone is None: # type: ignore
                        break 

                    try:
                        lvl = self.params.get("persona_level", 7)
                        if int(lvl) >= 7: self.current_mood = "transcendent"
                        else: self.current_mood = "meditating"
                        
                        if self.awake:
                            self.recognizer.pause_threshold = self.base_pause_threshold * 1.5
                            ui.root.after(0, ui.update_avatar, self.persona_map[self.current_mood]["listen"])
                            if _last_state != "awake":
                                ui.root.after(0, ui.log, "\n[SERENITY ACTIVE - Listening...]")
                                _last_state = "awake"
                        else:
                            self.recognizer.pause_threshold = self.base_pause_threshold
                            ui.root.after(0, ui.update_avatar, self.persona_map[self.current_mood]["idle"])
                            if _last_state != "idle":
                                ui.root.after(0, ui.log, f"\n[SERENITY STANDBY - Waiting for Wake Word...]")
                                _last_state = "idle"
                        
                        with self.mic_lock:    
                            # Reduced phrase_time_limit for snappier responses
                            audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=15)
                            
                        transcript = self.recognizer.recognize_google(audio).lower()
                        _awake_timeout_counter = 0 
                        
                        ui.root.after(0, ui.log, f"You: {transcript}")
                        self.adapt_cadence(audio, transcript)
                        
                        if self._is_querying:
                            impatience_keywords = ["hello", "you there", "well", "send it", "hurry", "wait"]
                            if any(kw in transcript for kw in impatience_keywords):
                                self.handle_impatience(ui)
                            continue

                        if "shut down" in transcript or "system exit" in transcript:
                            ui.root.after(0, ui.update_avatar, "serenity_off.png")
                            ui.root.after(0, ui.log, "Serenity: Going offline. Goodbye.")
                            self.speak("Going offline. Goodbye.")
                            os._exit(0)
                        
                        if "beep" in transcript:
                            self.log_thought("BEEP command detected.")
                            threading.Thread(target=lambda: winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS), daemon=True).start() # type: ignore
                            ui.root.after(0, ui.log, "Serenity: *Ding*")
                            continue
                            
                        if "go to sleep" in transcript or "nevermind" in transcript:
                            if self.awake:
                                self.awake = False
                                _last_state = None
                                ui.root.after(0, ui.update_avatar, "subdued_serenity.png")
                                ui.root.after(0, ui.log, "Serenity: Standing by.")
                                self.speak("Standing by.")
                            continue
                            
                        if self.awake:
                            if self._is_complex_query(transcript):
                                ui.root.after(0, ui.update_avatar, self.persona_map[self.current_mood]["loading"])
                                ui.root.after(0, ui.log, "Serenity: Hold on, let me think about that.")
                                self.speak("Hold on, let me think about that.")
                            
                            ui.root.after(0, ui.update_avatar, self.persona_map[self.current_mood]["think"])
                            threading.Thread(target=self._bg_query_worker, args=(transcript, ui), daemon=True).start()
                            continue
                            
                        if "serenity" in transcript:
                            command = transcript.split("serenity", 1)[1].strip()
                            if command:
                                if self._is_complex_query(command):
                                    ui.root.after(0, ui.update_avatar, self.persona_map[self.current_mood]["loading"])
                                    ui.root.after(0, ui.log, "Serenity: Hold on, let me think about that.")
                                    self.speak("Hold on, let me think about that.")
                                
                                ui.root.after(0, ui.update_avatar, self.persona_map[self.current_mood]["think"])
                                threading.Thread(target=self._bg_query_worker, args=(command, ui), daemon=True).start()
                            else:
                                self.awake = True
                                _last_state = None
                                ui.root.after(0, ui.update_avatar, self.persona_map[self.current_mood]["listen"])
                                # Quick ding confirmation
                                try: winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS | winsound.SND_ASYNC)
                                except: pass
                                ui.root.after(0, ui.log, "Serenity: [Acknowledged]")
                        
                    except sr.WaitTimeoutError:
                        if self.awake:
                            _awake_timeout_counter += 1
                            if _awake_timeout_counter >= 2:
                                self.awake = False
                                _last_state = None
                                _awake_timeout_counter = 0
                                ui.root.after(0, ui.update_avatar, "subdued_serenity.png")
                                ui.root.after(0, ui.log, "[SYSTEM] Conversation timed out. Returning to sleep.")
                        continue
                    except sr.UnknownValueError:
                        if self.awake:
                            _awake_timeout_counter += 1 
                        continue
                    except sr.RequestError as e:
                        ui.root.after(0, ui.log, f"[SYSTEM] Speech Recognition API error: {e}")
                        continue
                    except Exception as e:
                        self.log_error(f"Listen loop inner error: {type(e).__name__}: {e}")
                        time.sleep(1)
                        continue

        except Exception as e:
            self.log_error(f"Listen loop critical failure: {e}. Attempting recovery...")
            ui.root.after(0, ui.log, "[SYSTEM] Audio stream lost. Reconnecting...")
            self.microphone = None
            time.sleep(5)

if __name__ == "__main__":
    agent = SerenityAgent()
    ui = SerenityUI(agent)
    ui.root.mainloop()