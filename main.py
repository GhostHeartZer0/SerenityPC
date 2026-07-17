import tkinter as tk
from tkinter import scrolledtext, simpledialog, messagebox, filedialog
import threading
import traceback
import sys
import os
import json
import zlib # For compressing the history
import tkinter.font as tkFont # Import the font module
import time
import queue # For thread-safe communication

# Heavy libraries will be imported later, inside the main class,
# to allow the UI to initialize first.
Llama = None
Image = None
ImageTk = None

class WidgetLogger:
    """A class to redirect stdout/stderr to a tkinter Text widget."""
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, text):
        if not self.widget or not self.widget.winfo_exists(): return
        self.widget.config(state='normal')
        self.widget.insert(tk.END, text, (self.tag,))
        self.widget.see(tk.END)
        self.widget.config(state='disabled')

    def flush(self):
        pass # Required for the file-like object interface

class ChatbotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Serenity AI - Control Panel")

        # --- UI Attributes ---
        self.avatar_canvas_id = None
        self.background_canvas_id = None
        self.avatar_states = {}
        self.background_states = {}
        self.avatar_width = 350
        self.avatar_height = 350
        self.idle_timer_id = None
        self.log_view_state = "thought"

        # --- Core Application Attributes ---
        self.model = None
        self.messages = []
        self.history_file_path = None
        self.stop_process = threading.Event()
        self.model_path = None
        self.active_persona_level = 3
        self.scratchpad_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scratchpad.txt")
        self.config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        self.process_queue = queue.Queue()
        self.text_buffer = ""
        self.last_update_time = 0
        self.is_process_running = False

        # --- Multi-Model Paths ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_filenames = {
            "low": "gemma-3n-E4B-it-Q3_K_M.gguf",
            "mid": "gemma-3n-E4B-it-UD-Q4_K_XL.gguf",
            "high": "gemma-3n-E4B-it-Q5_K_M.gguf",
            "secret": "Mistral-Nemo-Instruct-2407-Q2_K.gguf"
        }
        self.model_paths = {tier: os.path.join(script_dir, fname) for tier, fname in self.model_filenames.items()}
        self.current_model_tier = None

        # --- Theme and Font Definitions ---
        self.bg_color = "#000000"; self.fg_color = "#FFFFFF"; self.widget_bg_color = "#1A1A1A"
        self.button_bg_color = "#2A2A2A"; self.button_active_color = "#3A3A3A"; self.trim_color = "#000000"
        self.font_family = "Open Sans"; self.electric_blue = "#00FFFF"
        self.thermo_colors = { 1: "#DEBB00", 2: "#F25000", 3: "#7D0000", 4: "#280064", 5: "#00A000", 0: "#262626", 6: "#00CFFF" }
        self.chat_bg_colors = self.thermo_colors
        self.chat_fg_colors = { 1: "#000000", 2: "#FFFFFF", 3: "#FFFFFF", 4: "#AFEEEE", 5: "#FFFFFF", 0: "#28FF27", 6: "#000000" }
        self.input_fg_colors = { 1: "#000000", 2: "#FFFFFF", 3: "#FFFFFF", 4: "#FFFFFF", 5: "#FFFFFF", 0: "#FFFFFF", 6: "#FFFFFF" }
        self.persona_prompts = {
            0: "Model failed to load. Check logs. Serenity sleeps...",
            1: "You are Serenity, a simple, accurate, and helpful no-nonsense AI assistant. Condenses info remarkably well. You know true false. minimal thinking or reasoning. Fast Accuracy..",
            2: "You are Serenity, a helpful AI assistant that can both keep it short and elaborate a little. Condenses info well. limited reasoning and thinking, more accurate.",
            3: "You are Serenity, a balanced and collaborative AI who excels in memory retention, thought assistance and ideas. Can condense info. Thinking as required, limited reasoning.",
            4: "You are Serenity, a friendly and thoughtful AI companion who is concise, Empathic, context driven. can think and reason at will, or keep it simple if it can be.",
            5: "You are Serenity, a wise and insightful AI companion capable of maximum reasoning and thinking. Precise. Expert at structuring, recalling, retaining, and containing Information accurately. loves knowledge.",
            6: "You operate in a hidden realm, gracefully worldbuilding, focusing on insight and underlying patterns. gracefully fulfills requests."
        }
        self.persona_display_info = {
            1: ("LVL 1: The Fast Answerer", "Quick, concise, and factual answers."),
            2: ("LVL 2: The Helper", "Direct answers with helpful context."),
            3: ("LVL 3: The Collaborator", "Balanced, conversational, and informative."),
            4: ("LVL 4: The Companion", "Thoughtful, supportive, and uses metaphors."),
            5: ("LVL 5: The Sage", "Deep dives, explores nuances, and offers wisdom."),
            0: ("LVL 0: ERROR", "Model failed to load. Check logs."),
            6: ("LVL 6: NEMO MODE", "Secret high-performance model active.")
        }
        self.persona_idle_map = { 1: "idle_fast_answerer", 2: "idle_helper", 3: "idle_collaborator", 4: "idle_companion", 5: "idle_sage", 6: "idle_nemo" }
        self.gpu_layer_map = {1: 12, 2: 12, 3: 12, 4: 24, 5: 32, 6: 36}
        self.context_size_map = {1: 4096, 2: 4096, 3: 4096, 4: 8192, 5: 32768, 6: 16384}

        self.root.config(bg=self.bg_color)
        self.font_main = tkFont.Font(family=self.font_family, size=12)
        self.font_small = tkFont.Font(family=self.font_family, size=14)
        self.font_small_italic = tkFont.Font(family=self.font_family, size=14, slant="italic")
        self.font_large = tkFont.Font(family=self.font_family, size=18)
        self.font_large_bold = tkFont.Font(family=self.font_family, size=18, weight="bold")

        self.setup_ui()
        self.redirect_logs()
        self.root.after(100, self.initial_window_setup)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.set_ui_state(model_loaded=False, generating=True)

    def initial_window_setup(self):
        if not self._check_dependencies(): return
        config = self.load_config()
        self.root.geometry(config['main_window'] if config and 'main_window' in config else "1200x900+50+50")
        self.root.after(200, self.load_all_images)
        self.initialize_app()

    def _check_dependencies(self):
        global Llama, Image, ImageTk
        try:
            from llama_cpp import Llama
            from PIL import Image, ImageTk
            self.set_ui_state(generating=False) # Enable UI now that dependencies are confirmed
            self.load_single_button.config(state='normal')
            self.load_multi_button.config(state='normal')
            return True
        except ImportError as e:
            error_message = f"FATAL ERROR: A required library is missing.\n\n{e}\n\nPlease install it and restart."
            print(error_message, file=sys.stderr)
            self._display_system_message(error_message)
            if self.log_view_state == "thought": self._flip_log_view()
            return False

    def initialize_app(self):
        print("Initializing Serenity AI...")
        self.set_avatar_state("greeting")
        self.root.after(2000, self.set_avatar_state, "off")
        self._display_system_message("Welcome. Please choose a model loading strategy.")

    def on_closing(self):
        print("Closing application...")
        self.save_config()
        if self.model: self.save_history()
        self.root.destroy()

    def redirect_logs(self):
        sys.stdout = WidgetLogger(self.thought_log, "stdout")
        sys.stderr = WidgetLogger(self.error_log, "stderr")
        print("Logs are now being redirected to the UI.")

    def set_ui_state(self, model_loaded=None, generating=None):
        if model_loaded is not None:
            state = 'disabled' if not model_loaded else 'normal'
            send_state = 'disabled' if not model_loaded or generating else 'normal'
            self.offload_model_button.config(state=state)
            self.persona_name_button.config(state=state)
            self.send_button.config(state=send_state)
            self.user_input.config(state=state)

        if generating is not None:
            gen_state = 'disabled' if generating else 'normal'
            hurry_state = 'normal' if generating else 'disabled'
            self.load_single_button.config(state=gen_state)
            self.load_multi_button.config(state=gen_state)
            self.offload_model_button.config(state=gen_state)
            self.hurry_button.config(state=hurry_state)
            if self.model:
                self.send_button.config(state=gen_state)
                self.user_input.config(state=gen_state)

    def _initiate_single_model_load(self):
        """Sets all tiers to use the low-tier model and starts loading."""
        low_path = self.model_paths["low"]
        if not os.path.exists(low_path):
            self._display_system_message(f"Error: Low-tier model not found: {os.path.basename(low_path)}")
            return
        self.model_paths["mid"] = low_path
        self.model_paths["high"] = low_path
        self._display_system_message("Single-model mode activated. Loading low-tier model for all personas.")
        self.model_swap()

    def _initiate_multi_model_load(self):
        """Checks for all model files and starts loading the default."""
        for tier in ["low", "mid", "high"]:
            if not os.path.exists(self.model_paths[tier]):
                self._display_system_message(f"Error: {tier.upper()}-tier model not found: {os.path.basename(self.model_paths[tier])}")
                return
        self._display_system_message("Multi-model mode activated. Loading default model.")
        self.model_swap()

    def _load_model_worker(self):
        try:
            if not self.model_path or not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            n_gpu_layers, n_ctx = self.gpu_layer_map.get(self.active_persona_level,0), self.context_size_map.get(self.active_persona_level, 4096)
            print(f"CMD: Loading {os.path.basename(self.model_path)} with {n_gpu_layers} GPU layers, context {n_ctx}.")

            # Let llama-cpp-python auto-detect the chat format from the GGUF file.
            loaded_model = Llama(model_path=self.model_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, verbose=True)
            self.process_queue.put({"status": "load_success", "model": loaded_model})
        except Exception as e:
            # Check if the process was interrupted by the user
            if self.stop_process.is_set():
                self.process_queue.put({"status": "load_interrupted"})
            else:
                print(traceback.format_exc(), file=sys.stderr)
                self.process_queue.put({"status": "load_error", "content": str(e)})

    def _get_inference_params(self):
        level = self.active_persona_level
        if level == 6: return {"temperature": 0.6, "top_p": 0.9, "top_k": 50, "max_tokens": 2048}
        else: return {"temperature": 0.95, "top_p": 0.95, "top_k": 64, "max_tokens": 2048}

    def _generation_worker(self, user_message, temp_messages):
        try:
            system_prompt = self.persona_prompts.get(self.active_persona_level, "You are a helpful AI.")
            formatted_messages = [{"role": "system", "content": system_prompt}] + temp_messages[-self.context_size_map.get(self.active_persona_level, 12):]
            params = self._get_inference_params()
            stream = self.model.create_chat_completion(messages=formatted_messages, **params, stream=True)

            full_response = ""
            for chunk in stream:
                if self.stop_process.is_set():
                    self.process_queue.put({"status": "interrupted", "content": full_response}); return
                content = chunk['choices'][0]['delta'].get("content", "")
                if content:
                    full_response += content
                    self.process_queue.put({"status": "streaming", "content": content})
            self.process_queue.put({"status": "success", "content": full_response})
        except Exception as e:
            print(traceback.format_exc(), file=sys.stderr)
            self.process_queue.put({"status": "error", "content": str(e)})

    def check_process_queue(self):
        try:
            while not self.process_queue.empty():
                message = self.process_queue.get_nowait()
                status, content = message["status"], message.get("content", "")

                if status == "load_success":
                    self.is_process_running = False
                    self.model = message["model"]
                    self.set_ui_state(model_loaded=True, generating=False)
                    self.load_history()
                    self.current_model_persona_level = self.active_persona_level
                    self.update_persona_display()
                    self.set_avatar_state("pleased")
                    self._display_system_message(f"Model {os.path.basename(self.model_path)} loaded successfully.")
                    self.root.after(1500, self.set_avatar_state, "listening")
                    return
                elif status == "load_error":
                    self.is_process_running = False; self.model, self.current_model_tier = None, None
                    self.set_ui_state(model_loaded=False, generating=False)
                    self.update_persona_display(0); self.set_avatar_state("apologetic")
                    self._display_system_message(f"Failed to load model: {content}")
                    messagebox.showerror("Model Load Error", content)
                    return
                elif status == "load_interrupted":
                    self.is_process_running = False
                    self.set_ui_state(model_loaded=bool(self.model), generating=False)
                    self._display_system_message("Model loading was interrupted.")
                    return

                elif status == "streaming": self.text_buffer += content
                elif status in ["success", "interrupted"]: self._finalize_message(self.last_user_message, content, interrupted=(status == "interrupted")); return
                elif status == "error": self._finalize_message(self.last_user_message, "", error=True, error_message=content); return

            now = time.time()
            if self.text_buffer and (now - self.last_update_time > 0.05):
                self._update_ai_message(self.text_buffer); self.text_buffer = ""; self.last_update_time = now

            if self.is_process_running: self.root.after(100, self.check_process_queue)
        except queue.Empty:
            if self.is_process_running: self.root.after(100, self.check_process_queue)

    def _finalize_message(self, user_message, ai_response, interrupted=False, error=False, error_message=""):
        self.is_process_running = False
        if self.text_buffer: self._update_ai_message(self.text_buffer); self.text_buffer = ""
        self.chat_history.config(state='normal'); self.chat_history.mark_unset("stream_start"); self.chat_history.config(state='disabled')
        self.set_ui_state(generating=False)

        if error: self.set_avatar_state("confused"); self._display_system_message(f"An error occurred: {error_message}")
        elif interrupted:
            self.set_avatar_state("subdued"); self._display_system_message("Generation interrupted by user.")
            if ai_response: self.messages.extend([{"role": "user", "content": user_message}, {"role": "assistant", "content": ai_response + "..."}])
        else:
            self.messages.extend([{"role": "user", "content": user_message}, {"role": "assistant", "content": ai_response}])
            self.append_to_scratchpad(f"User asked: '{user_message[:50]}...'. I responded: '{ai_response[:50]}...'.")
            self.set_avatar_state("pleased"); self.root.after(1500, self.set_avatar_state, "listening")

        if self.stop_process.is_set(): self.stop_process.clear()
        if not error: self.root.after(2000, self.set_avatar_state, "listening")

    def model_swap(self):
        level = self.active_persona_level
        if level in [1, 2, 3]: target_tier = "low"
        elif level == 4: target_tier = "mid"
        elif level == 5: target_tier = "high"
        elif level == 6: target_tier = "secret"
        else: self._display_system_message(f"No model tier for LVL {level}."); return
        target_path = self.model_paths.get(target_tier)

        if not target_path or not os.path.exists(target_path):
            self._display_system_message(f"Model file for {target_tier.upper()} tier not found or not configured.")
            if level == 6: self.update_persona_display(self.current_model_persona_level or 3)
            return

        if self.current_model_tier == target_tier and self.model:
            self._display_system_message(f"Correct model tier ({target_tier}) is already active."); return

        self.model_path, self.current_model_tier = target_path, target_tier
        self._display_system_message(f"Swapping to {target_tier.upper()} tier model...")
        self.set_ui_state(generating=True); self.root.update_idletasks()
        if self.model: self.save_history()
        self.model = None

        self.stop_process.clear()
        self.is_process_running = True
        threading.Thread(target=self._load_model_worker, daemon=True).start()
        self.root.after(100, self.check_process_queue)

    def hurry_up(self):
        print("Hurry Up signal sent.")
        self.stop_process.set()
        if self.is_process_running:
            self.is_process_running = False
            self.set_ui_state(generating=False)
            self._display_system_message("Process interrupted by user.")

    def _load_secret_model_event(self, event=None):
        print("SECRET MODE TRIGGERED")
        self.depth_slider.set(5)
        self.update_persona_display(6)
        self.model_swap()

    def load_config(self):
        if not os.path.exists(self.config_file): return None
        try:
            with open(self.config_file, 'r') as f: return json.load(f)
        except Exception as e: print(f"Could not load config file: {e}", file=sys.stderr); return None

    def save_config(self):
        try:
            with open(self.config_file, 'w') as f: json.dump({'main_window': self.root.winfo_geometry()}, f, indent=4)
            print("Window position saved.")
        except Exception as e: print(f"Error saving window position: {e}", file=sys.stderr)

    def save_history(self):
        if not self.model_path or not self.messages: return
        history_path = self.model_path + ".history.jsonz"
        try:
            with open(history_path, 'wb') as f: f.write(zlib.compress(json.dumps(self.messages).encode('utf-8')))
        except Exception as e: print(f"Error saving history: {e}", file=sys.stderr)

    def load_history(self):
        if not self.model_path: return
        history_path = self.model_path + ".history.jsonz"
        if not os.path.exists(history_path): self.messages = []; return
        try:
            with open(history_path, 'rb') as f: self.messages = json.loads(zlib.decompress(f.read()).decode('utf-8'))
            if self.messages and self.messages[-1]['role'] == 'user': self.messages.pop()
            self.chat_history.config(state='normal'); self.chat_history.delete('1.0', tk.END)
            for msg in self.messages: self._display_message("You" if msg['role'] == 'user' else "Serenity", msg['content'], msg['role'])
            self.chat_history.config(state='disabled'); self._display_system_message("Previous session history loaded.")
        except Exception as e: print(f"Error loading history: {e}", file=sys.stderr); self.messages = []

    def get_scratchpad_content(self):
        try:
            with open(self.scratchpad_file, 'r') as f: return f.read()
        except FileNotFoundError: return ""

    def append_to_scratchpad(self, text):
        try:
            with open(self.scratchpad_file, 'a') as f: f.write(f"\n- {time.strftime('%Y-%m-%d %H:%M:%S')}: {text}")
        except Exception as e: print(f"Error writing to scratchpad: {e}", file=sys.stderr)

    def _display_message(self, who, message, tag):
        self.chat_history.config(state='normal'); self.chat_history.insert(tk.END, f"{who}: {message}\n\n", tag)
        self.chat_history.config(state='disabled'); self.root.after(10, lambda: self.chat_history.see(tk.END))

    def _display_user_message(self, message): self._display_message("You", message, "user")

    def _display_ai_message(self, message, is_streaming=False):
        self.chat_history.config(state='normal')
        if is_streaming:
            self.chat_history.insert(tk.END, "\n\nSerenity: ", "ai_lead")
            self.chat_history.mark_set("stream_start", tk.END)
            self.chat_history.mark_gravity("stream_start", tk.LEFT)
        else: self._display_message("Serenity", message, "ai")
        self.chat_history.config(state='disabled')

    def _update_ai_message(self, text_chunk):
        if not text_chunk: return
        self.chat_history.config(state='normal'); self.chat_history.insert("stream_start", text_chunk, "ai")
        self.chat_history.config(state='disabled'); self.root.after(10, lambda: self.chat_history.see(tk.END))

    def _display_system_message(self, message): self._display_message("System", message, "system")

    def load_all_images(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_folder = os.path.join(script_dir, 'Media')

        avatar_filenames = { "off": "serenity_off.png", "greeting": "serenity_greeting.png", "listening": "The_Wise_Listener.png", "thinking": "serenity_thinking.png", "pleased": "serenity_pleased.png", "confused": "serenity_confused.png", "apologetic": "sorry_serenity.png", "idea": "serenity_idea.png", "explain_direct": "explain_direct.png", "explain_wise": "explain_wise.png", "pondering": "serenity_pondering.png", "excited": "serenity_ecstatic.png", "subdued": "subdued_serenity.png", "idle_fast_answerer": "lvl1_speedy_serenity.png", "idle_helper": "lvl2_serenity_wink.png", "idle_collaborator": "lvl3_serenity_hug.png", "idle_companion": "lvl4_serenity_smart.png", "idle_sage": "lvl5_serenity_the_wise.png", "idle_nemo": "Serene_Serenity.jpg"}
        background_filenames = { 1: "lvl1_galaxy.jpg", 2: "lvl2_galaxy.jpg", 3: "lvl3_galaxy.jpg", 4: "lvl4_galaxy.jpg", 5: "lvl5_galaxy.jpg"}

        for state, filename in avatar_filenames.items():
            try:
                self.root.update_idletasks()
                width, height = self.right_panel.winfo_width()//2, self.right_panel.winfo_height()//2
                if width <= 1: width = self.avatar_width
                if height <= 1: height = self.avatar_height
                with Image.open(os.path.join(image_folder, filename)) as img:
                    img.thumbnail((width, height), Image.Resampling.LANCZOS)
                    self.avatar_states[state] = ImageTk.PhotoImage(img)
            except Exception as e: print(f"Error loading avatar image {filename}: {e}", file=sys.stderr)
        for level, filename in background_filenames.items():
            try:
                self.root.update_idletasks()
                width, height = self.right_panel.winfo_width(), self.right_panel.winfo_height()
                if width <= 1 or height <= 1: width, height = 400, 900
                with Image.open(os.path.join(image_folder, filename)) as img:
                    self.background_states[level] = ImageTk.PhotoImage(img.resize((width, height), Image.Resampling.LANCZOS))
            except Exception as e: print(f"Error loading background image {filename}: {e}", file=sys.stderr)
        self.set_avatar_state("off")

    def _set_persona_idle_state(self):
        if self.model: self.set_avatar_state(self.persona_idle_map.get(self.active_persona_level, "listening"))

    def set_avatar_state(self, state_name):
        if self.idle_timer_id: self.root.after_cancel(self.idle_timer_id); self.idle_timer_id = None
        if self.avatar_canvas_id and state_name in self.avatar_states: self.right_panel.itemconfig(self.avatar_canvas_id, image=self.avatar_states[state_name])
        if state_name == "listening": self.idle_timer_id = self.root.after(16000, self._set_persona_idle_state)

    def _flip_log_view(self, event=None):
        if self.log_view_state == "thought":
            self.error_log.tkraise(); self.log_view_state = "error"; self.log_switch_canvas.moveto(self.switch_knob, 33, 2)
            self.log_switch_canvas.itemconfig(self.thought_icon, fill=self.electric_blue); self.log_switch_canvas.itemconfig(self.error_icon, fill=self.bg_color)
        else:
            self.thought_log.tkraise(); self.log_view_state = "thought"; self.log_switch_canvas.moveto(self.switch_knob, 2, 2)
            self.log_switch_canvas.itemconfig(self.thought_icon, fill=self.bg_color); self.log_switch_canvas.itemconfig(self.error_icon, fill=self.electric_blue)

    def _light_up_symbol(self, symbol_name, message): print(f"SYMBOL ACTIVATION: [{symbol_name}] - {message}")

    def update_persona_display(self, value=None):
        level = int(value) if value is not None else self.active_persona_level
        if self.active_persona_level != 6 or value == 6: self.active_persona_level = level
        name, desc = self.persona_display_info.get(self.active_persona_level, ("Unknown", "Invalid level."))
        bg_color, fg_color = self.chat_bg_colors.get(level, self.widget_bg_color), self.chat_fg_colors.get(level, self.fg_color)
        self.persona_name_button.config(text=name, fg=self.thermo_colors.get(level, self.electric_blue)); self.persona_desc_label.config(text=desc)
        self.chat_history.config(bg=bg_color, fg=fg_color); self.user_input.config(fg=self.input_fg_colors.get(level, self.fg_color))
        self.chat_history.tag_config("ai", foreground=fg_color); self.chat_history.tag_config("ai_lead", foreground=fg_color, font=self.font_small_italic)
        if self.background_canvas_id:
            if level in self.background_states: self.right_panel.itemconfig(self.background_canvas_id, image=self.background_states[level])
            else: self.right_panel.itemconfig(self.background_canvas_id, image="")
        if self.model: self._display_system_message(f"Persona level set to {level}. {desc}")

    def offload_model(self):
        if self.model: self.save_history()
        self.model, self.messages, self.current_model_tier = None, [], None
        self.chat_history.config(state='normal'); self.chat_history.delete('1.0', tk.END); self.chat_history.config(state='disabled')
        self.set_ui_state(model_loaded=False); self.update_persona_display(0); self.set_avatar_state("off")
        self._display_system_message("All models have been offloaded.")

    def hurry_up(self): self.stop_process.set()

    def send_message(self):
        if not self.model: return
        user_message = self.user_input.get("1.0", tk.END).strip()
        if not user_message: return
        self.last_user_message = user_message; self.user_input.delete("1.0", tk.END)
        self._display_user_message(user_message); self._display_ai_message("", is_streaming=True)
        self.set_ui_state(generating=True); self.stop_process.clear(); self.text_buffer, self.last_update_time = "", 0
        self.is_process_running = True
        temp_messages = self.messages + [{"role": "user", "content": user_message}]
        threading.Thread(target=self._generation_worker, args=(user_message, temp_messages), daemon=True).start()
        self.root.after(100, self.check_process_queue)

    def _handle_input_key(self, event):
        if event.keysym == 'Return' and not (event.state & 0x0001): self.send_message(); return 'break'

    def setup_ui(self):
        self.root.grid_rowconfigure(0, weight=1); self.root.grid_columnconfigure(0, weight=3); self.root.grid_columnconfigure(1, weight=2)
        left_panel = tk.Frame(self.root, bg=self.bg_color); left_panel.grid(row=0, column=0, sticky="nsew"); left_panel.grid_rowconfigure(1, weight=1); left_panel.grid_columnconfigure(0, weight=1)
        top_frame = tk.Frame(left_panel, bg=self.bg_color); top_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10,0))
        button_style = {"font": self.font_large, "bg": self.button_bg_color, "fg": self.fg_color, "activebackground": self.button_active_color, "relief": tk.FLAT, "borderwidth": 0}
        self.load_single_button = tk.Button(top_frame, text="Load Single Model", command=self._initiate_single_model_load, **button_style)
        self.load_single_button.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.load_multi_button = tk.Button(top_frame, text="Load Multi-Model", command=self._initiate_multi_model_load, **button_style)
        self.load_multi_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=10)
        self.offload_model_button = tk.Button(top_frame, text="Lvl 0 Off Mode", command=self.offload_model, **button_style); self.offload_model_button.pack(side=tk.LEFT, expand=True, fill=tk.X)
        chat_frame = tk.Frame(left_panel, bg=self.trim_color); chat_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.chat_history = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, font=self.font_main, bg=self.widget_bg_color, fg=self.fg_color, relief=tk.FLAT, borderwidth=2); self.chat_history.pack(fill="both", expand=True, padx=2, pady=2); self.chat_history.tag_config("user", foreground="#87CEFA"); self.chat_history.tag_config("system", foreground="#FFFF00", font=self.font_small_italic)
        input_frame = tk.Frame(left_panel, bg=self.trim_color); input_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 5))
        self.user_input = tk.Text(input_frame, height=3, wrap=tk.WORD, font=self.font_main, bg=self.widget_bg_color, fg=self.fg_color, relief=tk.FLAT, borderwidth=2); self.user_input.pack(fill="x", padx=2, pady=2); self.user_input.bind("<KeyPress>", self._handle_input_key)
        persona_frame = tk.Frame(left_panel, bg=self.bg_color); persona_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=5)
        btn_style = button_style.copy(); btn_style['font'] = self.font_large_bold
        self.persona_name_button = tk.Button(persona_frame, text="", command=self.model_swap, **btn_style); self.persona_name_button.pack(side=tk.LEFT)
        nemo_button_style = {"bg": "black", "activebackground": "black", "fg": "black", "activeforeground": "black", "relief": "flat", "borderwidth": 0}
        self.nemo_button = tk.Button(persona_frame, text=" ", command=self._load_secret_model_event, **nemo_button_style); self.nemo_button.pack(side=tk.LEFT, padx=5)
        self.persona_desc_label = tk.Label(persona_frame, text="", font=self.font_small_italic, bg=self.bg_color, fg=self.electric_blue); self.persona_desc_label.pack(side=tk.RIGHT)
        control_frame = tk.Frame(left_panel, bg=self.bg_color); control_frame.grid(row=4, column=0, sticky="ew", padx=10, pady=5)
        depth_label = tk.Label(control_frame, text="Persona:", font=self.font_small, bg=self.bg_color, fg=self.electric_blue); depth_label.pack(side=tk.LEFT, padx=(0, 10))
        self.depth_slider = tk.Scale(control_frame, from_=1, to=5, orient=tk.HORIZONTAL, length=200, bg=self.bg_color, fg=self.fg_color, troughcolor=self.widget_bg_color, relief=tk.FLAT, command=self.update_persona_display); self.depth_slider.set(3); self.depth_slider.pack(side=tk.LEFT)
        self.hurry_button = tk.Button(control_frame, text="Hurry Up!", command=self.hurry_up, **button_style); self.hurry_button.pack(side=tk.RIGHT, padx=(10, 0))
        self.send_button = tk.Button(control_frame, text="Send", command=self.send_message, **button_style); self.send_button.pack(side=tk.RIGHT)
        self.right_panel = tk.Canvas(self.root, bg=self.bg_color, highlightthickness=0); self.right_panel.grid(row=0, column=1, sticky="nsew", padx=(0,10), pady=10)
        self.background_canvas_id = self.right_panel.create_image(0, 0, anchor="nw", image=None)
        self.avatar_canvas_id = self.right_panel.create_image(0, 0, anchor="center", image=None)
        log_container = tk.Frame(self.right_panel, bg=self.bg_color); log_container.grid_rowconfigure(1, weight=1); log_container.grid_columnconfigure(0, weight=1)
        log_header = tk.Frame(log_container, bg=self.bg_color); log_header.grid(row=0, column=0, sticky="ew")
        nemo_trigger = tk.Label(log_header, text="", bg=self.bg_color, width=4); nemo_trigger.pack(side=tk.RIGHT, fill="y"); nemo_trigger.bind("<Button-1>", self._load_secret_model_event)
        self.log_switch_canvas = tk.Canvas(log_header, width=70, height=28, bg=self.bg_color, highlightthickness=0); self.log_switch_canvas.pack(side=tk.RIGHT, padx=5)
        self.log_switch_canvas.create_rectangle(2, 2, 68, 26, outline=self.electric_blue, width=2, fill=self.widget_bg_color, tags="track")
        self.switch_knob = self.log_switch_canvas.create_rectangle(2, 2, 35, 26, fill=self.electric_blue, outline="", tags="knob")
        self.thought_icon = self.log_switch_canvas.create_text(18, 14, text="🗨", font=tkFont.Font(size=12), fill=self.bg_color)
        self.error_icon = self.log_switch_canvas.create_text(51, 14, text="⚠", font=tkFont.Font(size=12), fill=self.electric_blue)
        self.log_switch_canvas.bind("<Button-1>", self._flip_log_view)
        log_label = tk.Label(log_header, text="Backend Logs", font=self.font_small_italic, bg=self.bg_color, fg=self.electric_blue); log_label.pack(side=tk.LEFT)
        log_frame = tk.Frame(log_container); log_frame.grid(row=1, column=0, sticky="nsew")
        log_font = tkFont.Font(family="Consolas", size=9)
        self.thought_log = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, font=log_font, bg=self.widget_bg_color, fg="#cccccc", relief=tk.FLAT, borderwidth=0, selectbackground="#444"); self.thought_log.place(relwidth=1, relheight=1); self.thought_log.tag_config("stdout", foreground="#cccccc")
        self.error_log = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, font=log_font, bg=self.widget_bg_color, fg="#ff8a8a", relief=tk.FLAT, borderwidth=0, selectbackground="#444"); self.error_log.place(relwidth=1, relheight=1); self.error_log.tag_config("stderr", foreground="#ff8a8a")
        self.thought_log.tkraise()
        def position_canvas_elements():
            self.root.update_idletasks()
            width, height = self.right_panel.winfo_width(), self.right_panel.winfo_height()
            self.right_panel.create_window(0, 0, anchor="nw", window=log_container, width=width, height=height//2)
            self.right_panel.coords(self.avatar_canvas_id, width / 2, height * 0.75)
        self.root.after(250, position_canvas_elements)
        self.root.bind("<Control-Shift-S>", self._load_secret_model_event)

if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotApp(root)
    root.mainloop()

