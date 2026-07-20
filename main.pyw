import os
import sys

# Localize TEMP, TMP, and compiler/cache directories to bypass Windows security restrictions
_base_dir = os.path.dirname(os.path.abspath(__file__))
_local_temp = os.path.join(_base_dir, ".temp")
_local_cuda = os.path.join(_base_dir, ".cuda_cache")
_local_triton = os.path.join(_base_dir, ".triton_cache")
_local_torch = os.path.join(_base_dir, ".torch_extensions")

for _d in [_local_temp, _local_cuda, _local_triton, _local_torch]:
    os.makedirs(_d, exist_ok=True)

os.environ["TEMP"] = _local_temp
os.environ["TMP"] = _local_temp
os.environ["CUDA_CACHE_PATH"] = _local_cuda
os.environ["TRITON_CACHE_DIR"] = _local_triton
os.environ["TORCH_EXTENSIONS_DIR"] = _local_torch

import site
import json
import threading
import gc
import time
import subprocess
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk

# --- 1. DLL & DATA INITIALIZATION ---
def inject_bundled_cuda():
    """Links the DLLs located in the local /Runtime folder."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    runtime_path = os.path.join(base_dir, "Runtime")
    
    if os.path.exists(runtime_path):
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(runtime_path)
        os.environ["PATH"] = runtime_path + os.pathsep + os.environ.get("PATH", "")
        return True
    return False

def inject_portable_cuda():
    """Scans all possible pip install locations to link the portable CUDA DLLs."""
    try:
        package_dirs = site.getsitepackages()
        if hasattr(site, 'getusersitepackages'):
            package_dirs.append(site.getusersitepackages())
        for base_dir in package_dirs:
            cuda_paths = [
                os.path.join(base_dir, "nvidia", "cuda_runtime", "bin"),
                os.path.join(base_dir, "nvidia", "cublas", "bin")
            ]
            for path in cuda_paths:
                if os.path.exists(path) and hasattr(os, 'add_dll_directory'):
                    os.add_dll_directory(path)
    except Exception: pass

# Try local bundle first, fallback to pip installation
if not inject_bundled_cuda():
    inject_portable_cuda()

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

# Important: Import these AFTER DLL injection
from llama_cpp import Llama
import serenity_resources as res


# --- 2. THE APPLICATION ---
class SerenityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SerenityPC Lite")
        self.root.geometry("1200x800")
        self.root.configure(bg=res.THEME["bg_color"])
        
        self.config_path = "config.json"
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.media_dir = os.path.join(self.base_dir, "Media")
        self.level_map = ["low", "mid", "high", "empath", "deep_cook", "worldbuilder"]
        
        self.load_settings()
        self.llm = None
        self.is_loading = False
        self.is_halted = False
        self.current_level = "low"
        self.history = []

        self.setup_ui()
        self.update_avatar(res.AVATAR_FILENAMES["off"])
        self.update_vram_meter()

    def load_settings(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f: self.config = json.load(f)
        else:
            self.config = {"lite_mode": False} # Default to GPU enabled

    def save_settings(self):
        with open(self.config_path, 'w') as f: json.dump(self.config, f, indent=2)

    def setup_ui(self):
        # --- Sidebar ---
        self.sidebar = tk.Frame(self.root, bg=res.THEME["widget_bg_color"], width=320)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        self.avatar_label = tk.Label(self.sidebar, bg=res.THEME["widget_bg_color"])
        self.avatar_label.pack(pady=20)

        self.status_label = tk.Label(self.sidebar, text="System: Offline", bg=res.THEME["widget_bg_color"], fg="#888888", font=("Arial", 10))
        self.status_label.pack(pady=5)
        
        self.vram_label = tk.Label(self.sidebar, text="VRAM: --- MB", bg=res.THEME["widget_bg_color"], fg=res.THEME["electric_blue"], font=("Arial", 9, "bold"))
        self.vram_label.pack(pady=2)

        tk.Label(self.sidebar, text="Thinking Log", bg=res.THEME["widget_bg_color"], fg=res.THEME["electric_blue"], font=("Arial", 9, "bold")).pack(pady=(20, 0))
        self.think_log = tk.Text(self.sidebar, bg="#0a0a0a", fg="#00ffcc", font=("Consolas", 8), height=15, borderwidth=0)
        self.think_log.pack(padx=15, pady=10, fill="x")
        self.think_log.config(state="disabled")

        self.begin_btn = tk.Button(self.sidebar, text="Begin", command=self.handle_model_toggle, bg=res.THEME["button_bg_color"], fg="white", font=("Arial", 10, "bold"), pady=12)
        self.begin_btn.pack(side="bottom", fill="x", pady=30, padx=20)

        # --- Main Area ---
        self.main_frame = tk.Frame(self.root, bg=res.THEME["bg_color"])
        self.main_frame.pack(side="right", expand=True, fill="both")

        # Top Control Bar
        control_frame = tk.Frame(self.main_frame, bg=res.THEME["widget_bg_color"])
        control_frame.pack(side="top", fill="x")

        self.lite_var = tk.BooleanVar(value=self.config.get("lite_mode", False))
        self.lite_btn = tk.Checkbutton(control_frame, text="Lite Mode (CPU)", variable=self.lite_var, command=self.toggle_lite, bg=res.THEME["widget_bg_color"], fg="#ff6666", selectcolor="#1e1e1e", font=("Arial", 9))
        self.lite_btn.pack(side="left", padx=10, pady=10)

        # Right side of control bar
        right_ctrl = tk.Frame(control_frame, bg=res.THEME["widget_bg_color"])
        right_ctrl.pack(side="right", padx=20)

        # THE SECRET TRIGGER
        self.secret_trigger = tk.Frame(right_ctrl, bg=res.THEME["widget_bg_color"], width=20, height=20)
        self.secret_trigger.pack(side="right")
        self.secret_trigger.bind("<Double-Button-1>", self.activate_worldbuilder)

        # THE SLIDER (Capped at 4 to hide Worldbuilder)
        self.level_slider = tk.Scale(right_ctrl, from_=0, to=4, orient="horizontal", 
                                     showvalue=False, command=self.on_slider_move, 
                                     bg=res.THEME["widget_bg_color"], highlightthickness=0, 
                                     length=200)
        self.level_slider.pack(side="right", pady=10)
        
        self.level_indicator = tk.Label(right_ctrl, text=res.PERSONA_DISPLAY_INFO[1][0], bg=res.THEME["widget_bg_color"], fg=res.THERMO_COLORS[1], font=("Arial", 10, "bold"))
        self.level_indicator.pack(side="right", padx=10)

        # --- Packing Order Fix ---
        # 1. Pack the Input Container at the absolute bottom FIRST
        input_container = tk.Frame(self.main_frame, bg=res.THEME["bg_color"])
        input_container.pack(side="bottom", fill="x", padx=20, pady=20)
        
        self.input_field = tk.Text(input_container, bg=res.THEME["trim_color"], fg="#FFFFFF", font=("Arial", 11), height=3, wrap="word", insertbackground="white", borderwidth=0)
        self.input_field.pack(side="left", expand=True, fill="x", padx=(0, 10))
        self.send_btn = tk.Button(input_container, text="Send", command=self.send_message, bg=res.THEME["button_bg_color"], fg="white", font=("Arial", 10, "bold"), padx=20)
        self.send_btn.pack(side="right", fill="y")
        self.input_field.bind("<Return>", self._handle_enter)

        # 2. Pack the Chat Display LAST so it dynamically fills the remaining middle cavity
        self.chat_display = scrolledtext.ScrolledText(self.main_frame, bg=res.CHAT_BG_COLORS[1], fg=res.CHAT_FG_COLORS[1], font=("Segoe UI", 11), state="disabled", wrap="word", borderwidth=0)
        self.chat_display.pack(expand=True, fill="both", padx=20, pady=10)

    # --- RESTORED SLIDER FUNCTION ---
    def on_slider_move(self, value):
        idx = int(float(value)) + 1
        self.current_level = self.level_map[idx-1]
        self.level_indicator.config(text=res.PERSONA_DISPLAY_INFO[idx][0], fg=res.THERMO_COLORS[idx])
        self.chat_display.config(bg=res.CHAT_BG_COLORS[idx], fg=res.CHAT_FG_COLORS[idx])
        
        # Safely update input field colors if mapped
        input_color = res.INPUT_FG_COLORS.get(idx, "#FFFFFF")
        self.input_field.config(fg=input_color)
        
        if not self.is_loading and not self.llm:
            idle_avatar = res.PERSONA_IDLE_MAP.get(idx, "serenity_off.png")
            self.update_avatar(res.AVATAR_FILENAMES.get(idle_avatar, "serenity_off.png"))

    def activate_worldbuilder(self, event):
        """Forces the slider to expand and jump to the hidden Worldbuilder persona."""
        self.level_slider.configure(to=5) # Unlocks the 6th position
        self.level_slider.set(5)
        self.log_think("WORLD DATA OVERRIDE: Level 6 (Worldbuilder) Unlocked.")
        self.on_slider_move(5)

    def update_vram_meter(self):
        if not self.lite_var.get() and self.llm:
            try:
                raw_out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'], creationflags=0x08000000).decode('utf-8').strip()
                output = raw_out.split('\n')[0].split(', ')
                self.vram_label.config(text=f"VRAM: {output[0]} / {output[1]} MB", fg="#00ffcc")
            except: self.vram_label.config(text="VRAM: N/A", fg="#888888")
        else: self.vram_label.config(text="VRAM: Idle", fg="#888888")
        self.root.after(2000, self.update_vram_meter)

    def handle_model_toggle(self):
        if self.llm:
            self.llm = None
            gc.collect()
            self.begin_btn.config(text="Begin", bg=res.THEME["button_bg_color"])
            self.status_label.config(text="System: Offline", fg="#888888")
            self.update_avatar(res.AVATAR_FILENAMES["off"])
        else: self.initialize_model()

    def toggle_lite(self):
        self.config["lite_mode"] = self.lite_var.get()
        self.save_settings()

    def initialize_model(self):
        if self.is_loading: return
        self.begin_btn.config(text="Loading...", state="disabled")
        self.update_avatar(res.AVATAR_FILENAMES["listening"])
        
        def _load():
            self.is_loading = True
            try:
                idx = self.level_map.index(self.current_level) + 1
                model_rel = self.config.get("model_paths", {}).get("low", "Models/gemma-3-4b-it-q4_0.gguf")
                path = os.path.normpath(os.path.join(self.base_dir, model_rel))
                n_ctx = res.CONTEXT_SIZE_MAP.get(idx, 4096)
                
                # GPU Layer Calculation
                n_gpu = 0 if self.lite_var.get() else 35 # Force offload for Gemma-3 4B
                
                import psutil
                cpu_cores = psutil.cpu_count(logical=False) or 4
                
                self.log_think(f"Booting Gemma-3 Engine ({'GPU' if n_gpu > 0 else 'CPU'})...")
                self.llm = Llama(
                    model_path=path,
                    n_ctx=n_ctx,
                    n_gpu_layers=n_gpu,
                    n_threads=max(1, cpu_cores - 1),
                    type_k_v=2,
                    flash_attn=False,
                    verbose=True
                )
                
                self.root.after(0, lambda: self.status_label.config(text="System: Online", fg="#00ffcc"))
                self.root.after(0, lambda: self.begin_btn.config(text="Offload", bg="#ff6666", state="normal"))
                self.log_think("Hardware acceleration confirmed.")
            except Exception as e:
                self.log_think(f"LOAD ERROR: {str(e)}")
                self.root.after(0, lambda: self.begin_btn.config(text="Begin", state="normal"))
            finally: self.is_loading = False
                
        threading.Thread(target=_load, daemon=True).start()

    def _stream_with_filter(self, llm_generator):
        buffer = ""
        is_hiding = False
        known_tags = ["[DEEPLOG:", "[SUGGEST_DEEP_THOUGHT]", "[PRIME_MEMORY:"]
        
        for chunk in llm_generator:
            if self.is_halted: break
            text = chunk["choices"][0]["text"]
            for char in text:
                if is_hiding:
                    buffer += char
                    if char == ']':
                        self.root.after(0, lambda b=buffer: self.log_think(f"Insight: {b}"))
                        is_hiding = False
                        buffer = ""
                else:
                    buffer += char
                    if '[' in buffer:
                        b_idx = buffer.index('[')
                        prefix = buffer[:b_idx]
                        potential_tag = buffer[b_idx:]
                        if any(tag.startswith(potential_tag) for tag in known_tags):
                            if prefix: self.root.after(0, lambda p=prefix: self.chat_display_op("append", p))
                            buffer = potential_tag
                            if len(potential_tag) >= 5: is_hiding = True
                        else:
                            self.root.after(0, lambda b=buffer: self.chat_display_op("append", b))
                            buffer = ""
                    else:
                        self.root.after(0, lambda b=buffer: self.chat_display_op("append", b))
                        buffer = ""

    def _standard_stream(self, text, max_t, idx):
        sys_prompt = res.PERSONA_PROMPTS.get(idx, "")
        prompt = f"<bos><start_of_turn>system\n{sys_prompt}<end_of_turn>\n<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n"
        try:
            self.root.after(0, lambda: self.chat_display_op("start", "Serenity"))
            self._stream_with_filter(self.llm(prompt, max_tokens=max_t, stream=True, stop=["<end_of_turn>"]))
        finally: self.root.after(0, self.reset_system_ui)

    def send_message(self):
        if self.is_loading: return
        text = self.input_field.get("1.0", tk.END).strip()
        if not text: return
        self.input_field.delete("1.0", tk.END)
        self.update_chat("You", text)
        self.is_loading = True
        self.send_btn.config(text="Halt", bg="#ff4d4d")
        
        idx = self.level_map.index(self.current_level) + 1
        max_t = self.config.get("max_tokens_config", {}).get(self.current_level, 1024)
        
        threading.Thread(target=self._standard_stream, args=(text, max_t, idx), daemon=True).start()

    def reset_system_ui(self):
        self.status_label.config(text="System: Online", fg="#00ffcc")
        self.send_btn.config(text="Send", bg=res.THEME["button_bg_color"])
        self.is_loading = False

    def update_chat(self, s, m):
        self.chat_display_op("start", s)
        self.chat_display_op("append", f"{m}\n\n")

    def chat_display_op(self, op, content):
        self.chat_display.config(state="normal")
        if op == "start": self.chat_display.insert(tk.END, f"{content}: ", "bold")
        else: self.chat_display.insert(tk.END, content); self.chat_display.see(tk.END)
        self.chat_display.config(state="disabled")

    def update_avatar(self, name):
        try:
            img = Image.open(os.path.join(self.media_dir, name)).resize((280, 280), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.avatar_label.config(image=photo); self.avatar_label.image = photo
        except: pass

    def log_think(self, t):
        self.think_log.config(state="normal"); self.think_log.insert(tk.END, f"> {t}\n"); self.think_log.see(tk.END); self.think_log.config(state="disabled")

    def _handle_enter(self, event):
        if not (event.state & 0x0001): self.send_message(); return "break"

if __name__ == "__main__":
    root = tk.Tk(); app = SerenityApp(root); root.mainloop()