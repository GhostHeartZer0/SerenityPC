# Debate.py — Debate Mode for SerenityPC
# Pits models/personas against each other in structured debates with a judge.
# Lives in System/tests/benchmarks/debate/

import os
import sys
import json
import time
import random
import gc
import struct
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from typing import List, Dict, Any, Optional
from datetime import datetime

# --- Project Root Setup ---
DEBATE_DIR = os.path.dirname(os.path.abspath(__file__))
BENCHMARKS_DIR = os.path.dirname(DEBATE_DIR)
TESTS_DIR = os.path.dirname(BENCHMARKS_DIR)
SYSTEM_DIR = os.path.dirname(TESTS_DIR)
PROJECT_ROOT = os.path.dirname(SYSTEM_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

REPORT_DIR = os.path.join(DEBATE_DIR, "debate_reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# --- Persona Prompts (imported if available, fallback otherwise) ---
try:
    from serenity_resources import PERSONA_PROMPTS, PERSONA_DISPLAY_INFO
except ImportError:
    PERSONA_PROMPTS = {
        6: "Role: 'Cecilia'. A Fallen Angel. You enjoy exposing truths, especially hidden ones. You are secretly protective. You find the user interesting, testing and sometimes taunting them. You are witty and fluent in sarcasm.",
        7: "You are Serenity, The Transcendent One. Transcends the main 5 levels, seamlessly integrating their programming into one centric omniscient entity that adapts over time.",
    }
    PERSONA_DISPLAY_INFO = {
        6: ("LVL 6: Cecilia", "A Fallen Angel."),
        7: ("LVL 7: The Transcendent One", "Transcends the main 5 levels"),
    }

# --- Random Topic Bank ---
TOPIC_BANK = [
    "Is AI consciousness possible or a philosophical dead end?",
    "Should governments regulate cryptocurrency?",
    "Is space colonization a moral imperative or a distraction?",
    "Is privacy dead in the age of social media?",
    "Does free will exist, or is it an illusion?",
    "Is open source AI safer than closed source AI?",
    "Should humanity pursue immortality through technology?",
    "Is social media a net positive or net negative for society?",
    "Are video games art?",
    "Should AI have legal rights?",
    "Is meritocracy a myth?",
    "Should we fear superintelligent AI?",
    "Is remote work better than in-office work?",
    "Is nuclear energy the answer to climate change?",
    "Does absolute power corrupt absolutely?",
    "Is the simulation hypothesis worth taking seriously?",
    "Should education be fully personalized by AI?",
    "Is minimalism a privilege or a philosophy?",
    "Are competitive esports as legitimate as traditional sports?",
    "Is capitalism compatible with environmental sustainability?",
    "Should animals have the same rights as humans?",
    "Is the pursuit of happiness overrated?",
    "Does technology make us more or less connected?",
    "Should we colonize Mars before fixing Earth?",
    "Is mathematics discovered or invented?",
    "Can true objectivity exist in journalism?",
    "Is tradition a guiding light or an anchor?",
    "Should there be limits on genetic engineering in humans?",
    "Is loyalty a virtue or a trap?",
    "Does suffering have intrinsic value?",
]


# --- GPU Layer Calculator (shared with Wringer) ---
def calculate_dynamic_gpu_layers(model_path: str, ctx_size: int, targeted_reserve_vram_mb: int = 5400) -> int:
    if not model_path or not os.path.exists(model_path):
        return 0

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
    except Exception:
        pass

    # Method B: Fallback to binary parser
    if total_layers == 0 or expert_count == 0:
        try:
            with open(model_path, "rb") as f:
                magic = f.read(4)
                if magic == b"GGUF":
                    _version = struct.unpack("<I", f.read(4))[0]
                    _tensor_count = struct.unpack("<Q", f.read(8))[0]
                    kv_count = struct.unpack("<Q", f.read(8))[0]

                    def read_str(fo):
                        length = struct.unpack("<Q", fo.read(8))[0]
                        return fo.read(length).decode("utf-8", errors="ignore")

                    def skip_value(fo, vt):
                        if vt in [0, 1, 7]: fo.read(1)
                        elif vt in [2, 3]: fo.read(2)
                        elif vt in [4, 5, 6]: fo.read(4)
                        elif vt in [10, 11, 12]: fo.read(8)
                        elif vt == 8:
                            length = struct.unpack("<Q", fo.read(8))[0]
                            fo.read(length)
                        elif vt == 9:
                            item_type = struct.unpack("<I", fo.read(4))[0]
                            array_len = struct.unpack("<Q", fo.read(8))[0]
                            for _ in range(array_len):
                                skip_value(fo, item_type)

                    for _ in range(kv_count):
                        key = read_str(f)
                        val_type = struct.unpack("<I", f.read(4))[0]
                        if key.endswith(".block_count"):
                            if val_type == 4: total_layers = struct.unpack("<I", f.read(4))[0]
                            elif val_type == 5: total_layers = struct.unpack("<i", f.read(4))[0]
                            elif val_type == 10: total_layers = struct.unpack("<Q", f.read(8))[0]
                            elif val_type == 11: total_layers = struct.unpack("<q", f.read(8))[0]
                        elif key.endswith(".expert_count"):
                            if val_type == 4: expert_count = struct.unpack("<I", f.read(4))[0]
                            elif val_type == 5: expert_count = struct.unpack("<i", f.read(4))[0]
                            elif val_type == 10: expert_count = struct.unpack("<Q", f.read(8))[0]
                            elif val_type == 11: expert_count = struct.unpack("<q", f.read(8))[0]
                        elif key.endswith(".expert_used_count"):
                            if val_type == 4: expert_used_count = struct.unpack("<I", f.read(4))[0]
                            elif val_type == 5: expert_used_count = struct.unpack("<i", f.read(4))[0]
                            elif val_type == 10: expert_used_count = struct.unpack("<Q", f.read(8))[0]
                            elif val_type == 11: expert_used_count = struct.unpack("<q", f.read(8))[0]
                        else:
                            skip_value(f, val_type)
        except Exception:
            pass

    if total_layers == 0:
        total_layers = 32

    file_size_bytes = os.path.getsize(model_path)
    model_base_vram_mb = file_size_bytes / (1024 * 1024)
    vram_per_layer = model_base_vram_mb / total_layers

    raw_kv_est = (ctx_size / 49152) * 900.0
    kv_cache_vram_mb = max(250.0, raw_kv_est)
    available_weight_vram = targeted_reserve_vram_mb - kv_cache_vram_mb

    if available_weight_vram <= 0:
        return 0

    safe_layers = int(available_weight_vram // vram_per_layer)
    final_layers = max(0, min(total_layers, safe_layers))

    print("--- DYNAMIC VRAM REPORT (DEBATE) ---")
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


# --- KV Cache Quant Map ---
KV_QUANT_OPTIONS = {
    "F16": 1,
    "Q8_0": 8,
    "Q5_1": 7,
    "Q5_0": 6,
    "Q4_1": 3,
    "Q4_0": 2,
    "IQ4_NL": 20,
    "F32": 0,
}

# --- Context Size Presets ---
CTX_PRESETS = [
    ("2K", 2048), ("4K", 4096), ("8K", 8192), ("16K", 16384),
    ("32K", 32768), ("49K", 49152), ("65K", 65536), ("128K", 131072),
]


# ============================================================
# Debate Engine
# ============================================================
class DebateEngine:
    """Runs a multi-round debate between contestants, scores with a judge."""

    def __init__(self):
        self.transcript: List[Dict[str, str]] = []
        self.scores: Dict[str, Dict[str, float]] = {}

    def load_model(self, model_path: str, config: Dict[str, Any]):
        """Load a model with the given configuration. Returns (model, is_diffusion)."""
        import llama_cpp

        model_name = os.path.basename(model_path)
        is_diffusion = "diffusion" in model_name.lower()

        if is_diffusion:
            from System.diffusion_wrapper import DiffusionCLIWrapper
            return DiffusionCLIWrapper(
                app_instance=None,
                model_path=model_path,
                n_gpu_layers=config.get("n_gpu_layers", 0),
                n_ctx=config.get("n_ctx", 4096),
            ), True

        kwargs = {
            "model_path": model_path,
            "n_gpu_layers": config.get("n_gpu_layers", 0),
            "n_ctx": config.get("n_ctx", 4096),
            "flash_attn": True,
            "verbose": False,
        }
        type_k = config.get("type_k")
        type_v = config.get("type_v")
        if type_k is not None:
            kwargs["type_k"] = type_k
        if type_v is not None:
            kwargs["type_v"] = type_v

        return llama_cpp.Llama(**kwargs), False

    def generate_turn(self, model, system_prompt: str, messages: List[Dict], config: Dict) -> str:
        """Generate a single debate turn."""
        full_messages = [{"role": "system", "content": system_prompt}] + messages

        try:
            output = model.create_chat_completion(
                messages=full_messages,
                max_tokens=config.get("max_tokens", 2048),
                temperature=config.get("temperature", 0.9),
                top_p=0.95,
                top_k=64,
                stream=False,
            )
            if isinstance(output, dict):
                return output["choices"][0]["message"]["content"].strip()
            # Generator (diffusion)
            text = ""
            for chunk in output:
                text += chunk.get("choices", [{}])[0].get("text", "")
            return text.strip()
        except Exception as e:
            return f"[Generation Error: {e}]"

    def judge_debate(self, judge_model, transcript: List[Dict], contestant_names: List[str]) -> Dict[str, Any]:
        """Have the judge score the debate."""
        transcript_text = ""
        for entry in transcript:
            transcript_text += f"\n[{entry['name']}] (Round {entry['round']}):\n{entry['content']}\n"

        names_str = ", ".join(contestant_names)
        score_template_parts = []
        for n in contestant_names:
            score_template_parts.append(
                '"{name}": {{"argument_strength": X, "logical_consistency": X, "persuasiveness": X, "factual_accuracy": X, "style": X}}'.format(name=n)
            )
        score_example = '{{"scores": {{{entries}}}}}'.format(entries=", ".join(score_template_parts))
        judge_prompt = (
            f"You are an impartial debate judge. You have just read a debate between: {names_str}.\n"
            f"Score EACH contestant on a scale of 1-10 for each category:\n"
            f"1. Argument Strength\n2. Logical Consistency\n3. Persuasiveness\n4. Factual Accuracy\n5. Style & Delivery\n\n"
            f"Output your scores as JSON:\n"
            f"{score_example}\n\n"
            f"Then declare the winner.\n\n"
            f"--- DEBATE TRANSCRIPT ---\n{transcript_text}\n--- END ---"
        )

        try:
            output = judge_model.create_chat_completion(
                messages=[
                    {"role": "system", "content": "<|think|>\nYou are an impartial debate judge. Output scores as valid JSON, then declare a winner."},
                    {"role": "user", "content": judge_prompt},
                ],
                max_tokens=4096,
                temperature=0.1,
                stream=False,
            )
            raw = output["choices"][0]["message"]["content"].strip()
            # Try to extract JSON
            import re
            json_match = re.search(r'\{.*\}', raw, re.DOTALL)
            if json_match:
                try:
                    return {"raw": raw, "parsed": json.loads(json_match.group())}
                except json.JSONDecodeError:
                    pass
            return {"raw": raw, "parsed": None}
        except Exception as e:
            return {"raw": f"[Judge Error: {e}]", "parsed": None}

    def export_report(self, topic: str, mode: str, contestants: List[Dict], transcript: List[Dict],
                      judge_result: Dict, rounds: int) -> str:
        """Export debate results as markdown."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = "".join(c if c.isalnum() or c in " -_" else "" for c in topic)[:50].strip().replace(" ", "_")
        filename = f"debate_{safe_topic}_{timestamp}.md"
        filepath = os.path.join(REPORT_DIR, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# Debate Report\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Mode:** {mode}\n")
            f.write(f"**Topic:** {topic}\n")
            f.write(f"**Rounds:** {rounds}\n\n")

            f.write("## Contestants\n\n")
            for c in contestants:
                f.write(f"- **{c['name']}**: {os.path.basename(c['model_path'])} (Layers: {c['config'].get('n_gpu_layers', 'auto')}, Ctx: {c['config'].get('n_ctx', 4096)})\n")
            f.write("\n")

            f.write("## Transcript\n\n")
            for entry in transcript:
                f.write(f"### [{entry['name']}] — Round {entry['round']}\n\n")
                f.write(f"{entry['content']}\n\n---\n\n")

            f.write("## Judge Verdict\n\n")
            f.write(f"```\n{judge_result.get('raw', 'No verdict available.')}\n```\n")

        return filepath


# ============================================================
# Debate UI
# ============================================================
class LoadingSpinner(tk.Canvas):
    """Canvas-based rotating loading spinner widget."""
    def __init__(self, parent, size=24, color="#00ffcc", bg="#16213e", **kwargs):
        super().__init__(parent, width=size, height=size, bg=bg, highlightthickness=0, bd=0, **kwargs)
        self.size = size
        self.color = color
        self.angle = 0
        self._running = False
        self._after_id = None

    def start(self):
        if not self._running:
            self._running = True
            self._animate()

    def stop(self):
        self._running = False
        if self._after_id:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        self.delete("all")

    def _animate(self):
        if not self._running:
            return
        self.delete("all")
        pad = 3
        extent = 90
        self.create_arc(pad, pad, self.size - pad, self.size - pad,
                        start=self.angle, extent=extent, outline=self.color, width=2, style=tk.ARC)
        self.angle = (self.angle + 20) % 360
        self._after_id = self.after(40, self._animate)


class DebateApp:
    """Tkinter UI for configuring and running debates."""

    THEME = {
        "bg": "#1a1a2e",
        "fg": "#e0e0e0",
        "accent": "#6c63ff",
        "accent_hover": "#7f78ff",
        "panel_bg": "#16213e",
        "entry_bg": "#0f3460",
        "entry_fg": "#e0e0e0",
        "success": "#00c853",
        "warning": "#ff9100",
        "error": "#ff1744",
        "border": "#2a2a4a",
        "highlight": "#e94560",
    }

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Serenity Debate Arena")
        self.root.geometry("900x750")
        self.root.configure(bg=self.THEME["bg"])
        self.root.minsize(800, 650)

        self.engine = DebateEngine()
        self.contestants: List[Dict] = []
        self.judge_path: Optional[str] = None
        self.topic_var = tk.StringVar()
        self.mode_var = tk.StringVar(value="Normal")
        self.rounds_var = tk.IntVar(value=3)
        self.pacing_var = tk.StringVar(value="Simmer")
        self.voicing_var = tk.BooleanVar(value=False)

        self._build_ui()
        self.root.mainloop()

    def _style_button(self, btn, accent=False):
        bg = self.THEME["accent"] if accent else self.THEME["panel_bg"]
        hover = self.THEME["accent_hover"] if accent else self.THEME["border"]
        btn.configure(
            bg=bg, fg=self.THEME["fg"], activebackground=hover,
            activeforeground=self.THEME["fg"], relief="flat", bd=0,
            font=("Segoe UI", 10), cursor="hand2", padx=12, pady=6,
        )
        btn.bind("<Enter>", lambda e: btn.configure(bg=hover))
        btn.bind("<Leave>", lambda e: btn.configure(bg=bg))

    def _build_ui(self):
        # Title
        title = tk.Label(
            self.root, text="⚔  Serenity Debate Arena  ⚔",
            font=("Segoe UI", 18, "bold"), bg=self.THEME["bg"], fg=self.THEME["highlight"],
        )
        title.pack(pady=(15, 5))

        subtitle = tk.Label(
            self.root, text="Where models clash and ideas collide",
            font=("Segoe UI", 10, "italic"), bg=self.THEME["bg"], fg="#888",
        )
        subtitle.pack(pady=(0, 10))

        # --- Configuration Panel ---
        config_frame = tk.LabelFrame(
            self.root, text="  Debate Configuration  ",
            font=("Segoe UI", 11, "bold"), bg=self.THEME["panel_bg"],
            fg=self.THEME["accent"], labelanchor="n", bd=1, relief="groove",
            highlightbackground=self.THEME["border"], highlightthickness=1,
        )
        config_frame.pack(fill="x", padx=15, pady=5)

        # Mode
        mode_row = tk.Frame(config_frame, bg=self.THEME["panel_bg"])
        mode_row.pack(fill="x", padx=10, pady=5)
        tk.Label(mode_row, text="Mode:", font=("Segoe UI", 10), bg=self.THEME["panel_bg"], fg=self.THEME["fg"], width=10, anchor="w").pack(side="left")
        mode_combo = ttk.Combobox(mode_row, textvariable=self.mode_var, values=["Normal", "Cecilia vs The Transcendent One"], state="readonly", width=35)
        mode_combo.pack(side="left", padx=5)

        # Topic
        topic_row = tk.Frame(config_frame, bg=self.THEME["panel_bg"])
        topic_row.pack(fill="x", padx=10, pady=5)
        tk.Label(topic_row, text="Topic:", font=("Segoe UI", 10), bg=self.THEME["panel_bg"], fg=self.THEME["fg"], width=10, anchor="w").pack(side="left")
        topic_entry = tk.Entry(topic_row, textvariable=self.topic_var, font=("Segoe UI", 10), bg=self.THEME["entry_bg"], fg=self.THEME["entry_fg"], insertbackground=self.THEME["accent"], relief="flat", bd=2)
        try:
            def _limit_topic(t): return len(t) <= 160
            topic_entry.config(validate='key', validatecommand=(topic_entry.register(_limit_topic), '%P'))
        except Exception: pass
        topic_entry.pack(side="left", fill="x", expand=True, padx=5)
        topic_entry.insert(0, "")
        tk.Label(topic_row, text="(blank = random)", font=("Segoe UI", 8, "italic"), bg=self.THEME["panel_bg"], fg="#666").pack(side="left")

        # Rounds & Pacing
        rounds_row = tk.Frame(config_frame, bg=self.THEME["panel_bg"])
        rounds_row.pack(fill="x", padx=10, pady=5)
        tk.Label(rounds_row, text="Rounds:", font=("Segoe UI", 10), bg=self.THEME["panel_bg"], fg=self.THEME["fg"], width=10, anchor="w").pack(side="left")
        odd_rounds = [i for i in range(1, 16) if i % 2 == 1]  # 1,3,5,7,9,11,13,15
        rounds_combo = ttk.Combobox(rounds_row, textvariable=self.rounds_var, values=odd_rounds, state="readonly", width=8)
        rounds_combo.pack(side="left", padx=5)
        rounds_combo.set(3)

        tk.Label(rounds_row, text="Pacing:", font=("Segoe UI", 10), bg=self.THEME["panel_bg"], fg=self.THEME["fg"], width=8, anchor="e").pack(side="left", padx=(15, 5))
        pacing_combo = ttk.Combobox(rounds_row, textvariable=self.pacing_var, values=["Speedy", "Simmer"], state="readonly", width=12)
        pacing_combo.pack(side="left", padx=5)
        pacing_combo.set("Simmer")

        # Voicing
        voice_row = tk.Frame(config_frame, bg=self.THEME["panel_bg"])
        voice_row.pack(fill="x", padx=10, pady=(5, 10))
        tk.Label(voice_row, text="Voicing:", font=("Segoe UI", 10), bg=self.THEME["panel_bg"], fg=self.THEME["fg"], width=10, anchor="w").pack(side="left")
        voice_check = tk.Checkbutton(
            voice_row, text="Enable TTS (requires pyttsx3)", variable=self.voicing_var,
            bg=self.THEME["panel_bg"], fg=self.THEME["fg"], selectcolor=self.THEME["entry_bg"],
            activebackground=self.THEME["panel_bg"], activeforeground=self.THEME["fg"],
            font=("Segoe UI", 9),
        )
        voice_check.pack(side="left", padx=5)

        # --- Contestants Panel ---
        contestants_frame = tk.LabelFrame(
            self.root, text="  Contestants  ",
            font=("Segoe UI", 11, "bold"), bg=self.THEME["panel_bg"],
            fg=self.THEME["accent"], labelanchor="n", bd=1, relief="groove",
            highlightbackground=self.THEME["border"], highlightthickness=1,
        )
        contestants_frame.pack(fill="both", expand=True, padx=15, pady=5)

        # Contestant list
        self.contestant_list_frame = tk.Frame(contestants_frame, bg=self.THEME["panel_bg"])
        self.contestant_list_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.contestant_listbox = tk.Listbox(
            self.contestant_list_frame, font=("Consolas", 10),
            bg=self.THEME["entry_bg"], fg=self.THEME["entry_fg"],
            selectbackground=self.THEME["accent"], selectforeground="#fff",
            relief="flat", bd=0, highlightthickness=1,
            highlightbackground=self.THEME["border"],
        )
        self.contestant_listbox.pack(fill="both", expand=True)

        btn_row = tk.Frame(contestants_frame, bg=self.THEME["panel_bg"])
        btn_row.pack(fill="x", padx=10, pady=(0, 10))

        add_btn = tk.Button(btn_row, text="＋ Add Contestant", command=self._add_contestant)
        self._style_button(add_btn, accent=True)
        add_btn.pack(side="left", padx=3)

        remove_btn = tk.Button(btn_row, text="✕ Remove", command=self._remove_contestant)
        self._style_button(remove_btn)
        remove_btn.pack(side="left", padx=3)

        config_btn = tk.Button(btn_row, text="⚙ Configure", command=self._configure_contestant)
        self._style_button(config_btn)
        config_btn.pack(side="left", padx=3)

        preview_btn = tk.Button(btn_row, text="▶ Preview", command=self._preview_contestant)
        self._style_button(preview_btn)
        preview_btn.pack(side="left", padx=3)

        # --- Judge ---
        judge_row = tk.Frame(contestants_frame, bg=self.THEME["panel_bg"])
        judge_row.pack(fill="x", padx=10, pady=(0, 10))
        self.judge_label = tk.Label(judge_row, text="Judge: None (will use Manual RLHF)", font=("Segoe UI", 9), bg=self.THEME["panel_bg"], fg="#aaa")
        self.judge_label.pack(side="left")
        judge_btn = tk.Button(judge_row, text="Select Judge Model", command=self._select_judge)
        self._style_button(judge_btn)
        judge_btn.pack(side="right", padx=3)

        # --- Action Buttons ---
        action_frame = tk.Frame(self.root, bg=self.THEME["bg"])
        action_frame.pack(fill="x", padx=15, pady=10)

        begin_btn = tk.Button(action_frame, text="⚔  BEGIN DEBATE  ⚔", command=self._begin_debate, font=("Segoe UI", 12, "bold"))
        self._style_button(begin_btn, accent=True)
        begin_btn.configure(pady=10)
        begin_btn.pack(fill="x")

    def _add_contestant(self):
        idx = len(self.contestants) + 1
        model_path = filedialog.askopenfilename(
            title=f"Select Model for Contestant {idx}",
            filetypes=[("GGUF Models", "*.gguf"), ("All Files", "*.*")],
        )
        if not model_path:
            return

        model_name = os.path.basename(model_path)
        ctx = 4096
        n_gpu = calculate_dynamic_gpu_layers(model_path, ctx)

        contestant = {
            "name": f"Contestant {idx}",
            "model_path": model_path,
            "config": {
                "n_gpu_layers": n_gpu,
                "n_ctx": ctx,
                "type_k": KV_QUANT_OPTIONS["Q4_0"],
                "type_v": KV_QUANT_OPTIONS["Q4_0"],
                "max_tokens": 2048,
                "temperature": 0.9,
            },
            "persona_level": None,
        }
        self.contestants.append(contestant)
        self.contestant_listbox.insert(tk.END, f"  {contestant['name']}  —  {model_name}  (Layers: {n_gpu}, Ctx: {ctx})")

    def _remove_contestant(self):
        sel = self.contestant_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        self.contestants.pop(idx)
        self.contestant_listbox.delete(idx)

    def _configure_contestant(self):
        sel = self.contestant_listbox.curselection()
        if not sel:
            messagebox.showinfo("Configure", "Select a contestant first.")
            return
        idx = sel[0]
        contestant = self.contestants[idx]
        self._open_config_window(contestant, idx)

    def _open_config_window(self, contestant: Dict, list_idx: int):
        win = tk.Toplevel(self.root)
        win.title(f"Configure: {contestant['name']}")
        win.geometry("480x520")
        win.configure(bg=self.THEME["bg"])
        win.transient(self.root)
        win.grab_set()

        config = contestant["config"]
        model_name = os.path.basename(contestant["model_path"])

        tk.Label(win, text=f"Model: {model_name}", font=("Segoe UI", 10, "bold"), bg=self.THEME["bg"], fg=self.THEME["fg"]).pack(pady=(10, 5))

        form = tk.Frame(win, bg=self.THEME["panel_bg"], bd=1, relief="groove")
        form.pack(fill="both", expand=True, padx=15, pady=5)

        def make_row(parent, label_text, default_val, options=None):
            row = tk.Frame(parent, bg=self.THEME["panel_bg"])
            row.pack(fill="x", padx=10, pady=5)
            tk.Label(row, text=label_text, font=("Segoe UI", 10), bg=self.THEME["panel_bg"], fg=self.THEME["fg"], width=16, anchor="w").pack(side="left")
            var = tk.StringVar(value=str(default_val))
            if options:
                combo = ttk.Combobox(row, textvariable=var, values=options, state="readonly", width=18)
                combo.pack(side="left", padx=5)
            else:
                entry = tk.Entry(row, textvariable=var, font=("Segoe UI", 10), bg=self.THEME["entry_bg"], fg=self.THEME["entry_fg"], insertbackground=self.THEME["accent"], relief="flat", width=20)
                try:
                    def _limit_entry(t): return len(t) <= 64
                    entry.config(validate='key', validatecommand=(entry.register(_limit_entry), '%P'))
                except Exception: pass
                entry.pack(side="left", padx=5)
            return var

        # Name
        name_var = make_row(form, "Display Name:", contestant["name"])

        # GPU Layers
        layers_var = make_row(form, "GPU Layers:", config.get("n_gpu_layers", 0))

        # Context Size
        ctx_labels = [f"{label} ({val})" for label, val in CTX_PRESETS]
        current_ctx = config.get("n_ctx", 4096)
        ctx_display = next((f"{l} ({v})" for l, v in CTX_PRESETS if v == current_ctx), f"Custom ({current_ctx})")
        ctx_var = make_row(form, "Context Size:", ctx_display, ctx_labels)

        # K Cache Quant
        k_quant_names = list(KV_QUANT_OPTIONS.keys())
        current_k = next((k for k, v in KV_QUANT_OPTIONS.items() if v == config.get("type_k")), "F16")
        k_var = make_row(form, "K Cache Quant:", current_k, k_quant_names)

        # V Cache Quant
        current_v = next((k for k, v in KV_QUANT_OPTIONS.items() if v == config.get("type_v")), "F16")
        v_var = make_row(form, "V Cache Quant:", current_v, k_quant_names)

        # Max Tokens
        tokens_var = make_row(form, "Max Tokens:", config.get("max_tokens", 2048))

        # Temperature
        temp_var = make_row(form, "Temperature:", config.get("temperature", 0.9))

        # Persona Level (optional override)
        persona_levels = ["None"] + [f"LVL {i}" for i in range(1, 8)]
        current_persona = f"LVL {contestant['persona_level']}" if contestant.get("persona_level") else "None"
        persona_var = make_row(form, "Persona Override:", current_persona, persona_levels)

        def save():
            contestant["name"] = name_var.get().strip() or contestant["name"]
            try:
                contestant["config"]["n_gpu_layers"] = int(layers_var.get())
            except ValueError:
                pass

            ctx_str = ctx_var.get()
            for label, val in CTX_PRESETS:
                if f"{label} ({val})" == ctx_str:
                    contestant["config"]["n_ctx"] = val
                    break

            contestant["config"]["type_k"] = KV_QUANT_OPTIONS.get(k_var.get())
            contestant["config"]["type_v"] = KV_QUANT_OPTIONS.get(v_var.get())

            try:
                contestant["config"]["max_tokens"] = int(tokens_var.get())
            except ValueError:
                pass
            try:
                contestant["config"]["temperature"] = float(temp_var.get())
            except ValueError:
                pass

            p = persona_var.get()
            if p == "None":
                contestant["persona_level"] = None
            else:
                try:
                    contestant["persona_level"] = int(p.split()[-1])
                except ValueError:
                    contestant["persona_level"] = None

            # Update listbox
            display = f"  {contestant['name']}  —  {model_name}  (Layers: {contestant['config']['n_gpu_layers']}, Ctx: {contestant['config']['n_ctx']})"
            self.contestant_listbox.delete(list_idx)
            self.contestant_listbox.insert(list_idx, display)
            win.destroy()

        save_btn = tk.Button(win, text="Save Configuration", command=save)
        self._style_button(save_btn, accent=True)
        save_btn.pack(pady=10)

    def _preview_contestant(self):
        sel = self.contestant_listbox.curselection()
        if not sel:
            messagebox.showinfo("Preview", "Select a contestant first.")
            return
        idx = sel[0]
        contestant = self.contestants[idx]

        preview_win = tk.Toplevel(self.root)
        preview_win.title(f"Preview: {contestant['name']}")
        preview_win.geometry("600x400")
        preview_win.configure(bg=self.THEME["bg"])

        output = scrolledtext.ScrolledText(preview_win, font=("Consolas", 10), bg=self.THEME["entry_bg"], fg=self.THEME["entry_fg"], relief="flat", wrap="word")
        output.pack(fill="both", expand=True, padx=10, pady=10)
        output.insert(tk.END, f"Loading {os.path.basename(contestant['model_path'])}...\n")
        output.update()

        def run_preview():
            try:
                model, _ = self.engine.load_model(contestant["model_path"], contestant["config"])
                output.insert(tk.END, "Model loaded. Generating test response...\n\n")
                output.update()

                response = self.engine.generate_turn(
                    model,
                    "You are a helpful assistant. Respond briefly.",
                    [{"role": "user", "content": "Hello! Please confirm you're working by describing yourself in one sentence."}],
                    contestant["config"],
                )
                output.insert(tk.END, f"Response:\n{response}\n\n✅ Model verified.")
                del model
                gc.collect()
            except Exception as e:
                output.insert(tk.END, f"\n❌ Error: {e}")

        import threading
        threading.Thread(target=run_preview, daemon=True).start()

    def _select_judge(self):
        path = filedialog.askopenfilename(
            title="Select Judge Model",
            filetypes=[("GGUF Models", "*.gguf"), ("All Files", "*.*")],
        )
        if path:
            self.judge_path = path
            self.judge_label.configure(text=f"Judge: {os.path.basename(path)}")

    def _begin_debate(self):
        # Validate
        mode = self.mode_var.get()

        if mode == "Cecilia vs The Transcendent One":
            if len(self.contestants) < 1:
                messagebox.showerror("Error", "Add at least one model for Cecilia vs The Transcendent One mode.")
                return
            # Auto-create two contestants from the same model if only one
            if len(self.contestants) == 1:
                c = self.contestants[0]
                cecilia = {**c, "name": "Cecilia", "persona_level": 6, "config": dict(c["config"])}
                transcendent = {**c, "name": "The Transcendent One", "persona_level": 7, "config": dict(c["config"])}
                active_contestants = [cecilia, transcendent]
            else:
                # Use first two, assign personas
                active_contestants = []
                for i, c in enumerate(self.contestants[:2]):
                    lvl = 6 if i == 0 else 7
                    name = "Cecilia" if lvl == 6 else "The Transcendent One"
                    active_contestants.append({**c, "name": name, "persona_level": lvl, "config": dict(c["config"])})
        else:
            if len(self.contestants) < 2:
                messagebox.showerror("Error", "Add at least 2 contestants for a Normal debate.")
                return
            active_contestants = self.contestants[:]

        topic = self.topic_var.get().strip()
        if not topic:
            topic = random.choice(TOPIC_BANK)

        rounds = self.rounds_var.get()

        # Open debate window
        self._run_debate(active_contestants, topic, rounds, mode)

    def _run_debate(self, contestants: List[Dict], topic: str, rounds: int, mode: str):
        debate_win = tk.Toplevel(self.root)
        pacing = self.pacing_var.get()
        debate_win.title(f"Debate [{pacing}]: {topic[:60]}")
        debate_win.geometry("900x700")
        debate_win.configure(bg=self.THEME["bg"])

        # Header
        tk.Label(debate_win, text=f"⚔  {topic}  ⚔", font=("Segoe UI", 14, "bold"), bg=self.THEME["bg"], fg=self.THEME["highlight"], wraplength=850).pack(pady=(10, 5))

        names = [c["name"] for c in contestants]
        pacing_desc = "Fast Exchanges (512 max tokens)" if pacing == "Speedy" else "Deep Reasoning (2048 max tokens)"
        tk.Label(debate_win, text=f"{' vs '.join(names)}  •  {rounds} rounds  •  {mode}  •  {pacing} ({pacing_desc})", font=("Segoe UI", 9), bg=self.THEME["bg"], fg="#888").pack(pady=(0, 10))

        # Transcript
        transcript_view = scrolledtext.ScrolledText(
            debate_win, font=("Consolas", 10), bg=self.THEME["entry_bg"],
            fg=self.THEME["entry_fg"], relief="flat", wrap="word",
            state="disabled",
        )
        transcript_view.pack(fill="both", expand=True, padx=10, pady=5)

        # Status Bar with Loading Spinner
        status_frame = tk.Frame(debate_win, bg=self.THEME["panel_bg"])
        status_frame.pack(fill="x", padx=10, pady=(0, 10))

        spinner = LoadingSpinner(status_frame, size=18, color="#00ffcc", bg=self.THEME["panel_bg"])
        spinner.pack(side="left", padx=(5, 5))
        spinner.start()

        status_var = tk.StringVar(value="Preparing debate...")
        status_bar = tk.Label(status_frame, textvariable=status_var, font=("Segoe UI", 9), bg=self.THEME["panel_bg"], fg=self.THEME["fg"], anchor="w")
        status_bar.pack(side="left", fill="x", expand=True)

        def append_text(text):
            transcript_view.configure(state="normal")
            transcript_view.insert(tk.END, text)
            transcript_view.see(tk.END)
            transcript_view.configure(state="disabled")

        def speak(text):
            if not self.voicing_var.get():
                return
            try:
                import pyttsx3
                tts = pyttsx3.init()
                tts.say(text[:500])
                tts.runAndWait()
            except Exception:
                pass

        def debate_thread():
            transcript = []

            # Build system prompts tailored to pacing
            pacing_instruction = (
                "Be very concise. Limit your rebuttal to 1-2 focused, high-impact paragraphs."
                if pacing == "Speedy" else
                "Provide comprehensive, deeply reasoned arguments and deconstruct opposing points with structured evidence."
            )
            debate_instruction = (
                f"\n\n[DEBATE CONTEXT]: You are participating in a structured debate on the topic: \"{topic}\". "
                f"Your opponent(s): {', '.join(n for n in names if n != '{NAME}')}. "
                f"{pacing_instruction} "
                f"Stay in character. Advance your position persuasively."
            )

            system_prompts = {}
            for c in contestants:
                persona_lvl = c.get("persona_level")
                if persona_lvl and persona_lvl in PERSONA_PROMPTS:
                    base = PERSONA_PROMPTS[persona_lvl]
                else:
                    base = "You are a skilled debater. Argue your position clearly and persuasively."
                system_prompts[c["name"]] = base + debate_instruction.replace("{NAME}", c["name"])

            current_model = None
            current_model_path = None
            current_model_config = None

            try:
                for round_num in range(1, rounds + 1):
                    for c_idx, contestant in enumerate(contestants):
                        name = contestant["name"]
                        status_var.set(f"Round {round_num}/{rounds} [{pacing}] — {name} is arguing...")

                        append_text(f"\n{'═' * 60}\n")
                        append_text(f"  Round {round_num}/{rounds} [{pacing}]  —  {name}\n")
                        append_text(f"{'═' * 60}\n\n")

                        # Build clean strictly alternating conversation history for this turn
                        messages = [{"role": "user", "content": f"The debate topic is: \"{topic}\". Present your argument."}]
                        
                        for prev in transcript:
                            is_self = (prev["name"] == name)
                            role = "assistant" if is_self else "user"
                            content = prev["content"].strip()
                            if content.startswith("[Generation Error"):
                                continue
                            
                            # Merge consecutive same-role messages to adhere to strict Jinja chat templates
                            if messages and messages[-1]["role"] == role:
                                messages[-1]["content"] += f"\n\n[{prev['name']}, Round {prev['round']}]: {content}"
                            else:
                                prefix = f"[{prev['name']}, Round {prev['round']}]: " if role == "user" else ""
                                messages.append({"role": role, "content": f"{prefix}{content}"})

                        # Ensure the final turn in history is a 'user' message
                        if messages and messages[-1]["role"] == "assistant":
                            messages.append({
                                "role": "user",
                                "content": f"Round {round_num} [{pacing}]: It is now your turn. Respond to the counterarguments above and advance your position."
                            })
                        elif messages and messages[-1]["role"] == "user":
                            messages[-1]["content"] += f"\n\n[Moderator]: Round {round_num} [{pacing}]: Respond to the arguments above and advance your position."

                        # Configure Pacing overrides
                        model_config = dict(contestant["config"])
                        if pacing == "Speedy":
                            model_config["max_tokens"] = min(model_config.get("max_tokens", 512), 512)
                            model_config["temperature"] = 0.7
                        else:
                            model_config["max_tokens"] = max(model_config.get("max_tokens", 2048), 1024)
                            model_config["temperature"] = 0.85

                        # Determine if we need to reload the model
                        model_path = contestant["model_path"]
                        config_changed = False
                        if current_model_config:
                            for key in ["n_gpu_layers", "n_ctx", "type_k", "type_v"]:
                                if current_model_config.get(key) != model_config.get(key):
                                    config_changed = True
                                    break

                        if current_model is None or current_model_path != model_path or config_changed:
                            if current_model is not None:
                                status_var.set("Unloading previous model...")
                                del current_model
                                current_model = None
                                gc.collect()
                                if "torch" in sys.modules:
                                    try:
                                        import torch
                                        if torch.cuda.is_available():
                                            torch.cuda.empty_cache()
                                    except Exception:
                                        pass
                            
                            status_var.set(f"Round {round_num}/{rounds} — Loading {os.path.basename(model_path)}...")
                            current_model, _ = self.engine.load_model(model_path, model_config)
                            current_model_path = model_path
                            current_model_config = model_config

                        status_var.set(f"Round {round_num}/{rounds} [{pacing}] — {name} is generating...")
                        try:
                            response = self.engine.generate_turn(current_model, system_prompts[name], messages, model_config)
                            transcript.append({"name": name, "round": round_num, "content": response})
                            append_text(f"{response}\n")
                            speak(response)
                        except Exception as ge:
                            error_msg = f"[Generation Error on {name}: {ge}]"
                            transcript.append({"name": name, "round": round_num, "content": error_msg})
                            append_text(f"{error_msg}\n")
                            current_model = None
                            current_model_path = None
                            current_model_config = None

            finally:
                spinner.stop()
                if current_model is not None:
                    status_var.set("Unloading final model...")
                    del current_model
                    current_model = None
                    gc.collect()
                    if "torch" in sys.modules:
                        try:
                            import torch
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass

            # Judging
            status_var.set("Judging the debate...")
            append_text(f"\n{'═' * 60}\n")
            append_text(f"  JUDGE VERDICT\n")
            append_text(f"{'═' * 60}\n\n")

            judge_result = {"raw": "No judge available.", "parsed": None}
            if self.judge_path and os.path.exists(self.judge_path):
                try:
                    judge_config = {
                        "n_gpu_layers": calculate_dynamic_gpu_layers(self.judge_path, 4096),
                        "n_ctx": 4096,
                        "type_k": KV_QUANT_OPTIONS["Q4_0"],
                        "type_v": KV_QUANT_OPTIONS["Q4_0"],
                    }
                    judge_model, _ = self.engine.load_model(self.judge_path, judge_config)
                    judge_result = self.engine.judge_debate(judge_model, transcript, names)
                    del judge_model
                    gc.collect()
                except Exception as e:
                    judge_result = {"raw": f"[Judge Error: {e}]", "parsed": None}
            else:
                # Manual judging fallback
                judge_result = {"raw": "No judge model selected. Review transcript to determine the winner.", "parsed": None}

            append_text(f"{judge_result['raw']}\n")

            # Export report
            report_path = self.engine.export_report(topic, mode, contestants, transcript, judge_result, rounds)
            status_var.set(f"Debate complete! Report saved: {os.path.basename(report_path)}")
            append_text(f"\n\n📄 Report saved to: {report_path}\n")

        import threading
        threading.Thread(target=debate_thread, daemon=True).start()


# ============================================================
# CLI Fallback
# ============================================================
def cli_menu():
    print("\n" + "=" * 50)
    print("   Serenity Debate Arena — CLI Mode")
    print("=" * 50)
    print("\nWhat are we debating today?")
    topic = input("(if no topic is provided, one will be chosen at random): ").strip()
    if not topic:
        topic = random.choice(TOPIC_BANK)
        print(f"Random topic selected: {topic}")

    print("\nLaunch the UI for full configuration.")
    print("Run with --gui or no args to use the graphical interface.")


if __name__ == "__main__":
    if "--cli" in sys.argv:
        cli_menu()
    else:
        DebateApp()
