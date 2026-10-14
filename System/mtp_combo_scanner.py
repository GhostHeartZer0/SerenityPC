# System/mtp_combo_scanner.py
# MTP & Drafter Model Scanner, Compatibility Matcher, and Layer Recommender for SerenityPC.

import os
import struct
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Any, Dict, List, Tuple

def read_gguf_header_fast(path: str) -> Dict[str, Any]:
    """Dependency-free GGUF KV header parser reading metadata in < 5ms."""
    scalar = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
    out = {}
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return out
            f.read(12)  # version (u32) + tensor_count (u64)

            def rd(fmt):
                return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

            def val(vt):
                if vt in scalar:
                    return rd(scalar[vt])
                if vt == 8:
                    return f.read(rd("<Q")).decode("utf-8", errors="ignore")
                if vt == 9:
                    item_type, n = rd("<I"), rd("<Q")
                    if item_type == 8:
                        for _ in range(n):
                            f.seek(rd("<Q"), 1)
                        return n
                    return [val(item_type) for _ in range(n)]
                raise ValueError(f"Unknown GGUF value type: {vt}")

            num_kv = rd("<Q")
            for _ in range(num_kv):
                key = f.read(rd("<Q")).decode("utf-8", errors="ignore").rstrip("\x00")
                out[key] = val(rd("<I"))
    except Exception:
        pass
    return out

def scan_model_directories(scan_roots: List[str]) -> List[Dict[str, Any]]:
    """Recursively scans directories for .gguf models and extracts architecture and vocabulary."""
    discovered = []
    seen = set()
    for root in scan_roots:
        if not root or not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if not f.lower().endswith(".gguf"):
                    continue
                if any(x in f.lower() for x in ["mmproj", "split", ".tmp"]):
                    continue
                full_path = os.path.normpath(os.path.join(dirpath, f))
                norm = os.path.normcase(full_path)
                if norm in seen:
                    continue
                seen.add(norm)
                try:
                    size_gb = round(os.path.getsize(full_path) / (1024 ** 3), 2)
                except Exception:
                    size_gb = 0.0

                kv = read_gguf_header_fast(full_path)
                arch = kv.get("general.architecture", "unknown")
                if arch == "unknown":
                    for k in kv:
                        if k.endswith(".block_count"):
                            arch = k.split(".")[0]
                            break

                vocab_size = kv.get("tokenizer.ggml.tokens") or kv.get(f"{arch}.vocab_size") or 0
                context_len = kv.get(f"{arch}.context_length", 8192)

                discovered.append({
                    "filename": f,
                    "path": full_path,
                    "size_gb": size_gb,
                    "arch": arch,
                    "vocab_size": vocab_size,
                    "context_len": context_len,
                    "is_assistant_head": arch == "gemma4_assistant" or "assistant" in f.lower()
                })
    return discovered

def compute_recommended_layers(size_gb: float, vram_gb: float = 6.0) -> int:
    """Computes optimal GPU layer offload budget to prevent PCIe shared memory spill."""
    if size_gb <= 4.0:
        return -1  # Full GPU offload for small models (E2B, E4B, 3B)
    elif size_gb <= 8.0:
        return 16  # 12B models on 6GB VRAM
    elif size_gb <= 18.0:
        return 8   # 26B/31B models on 6GB VRAM (Maximum safe ceiling)
    else:
        return 4   # Extra large models

def find_acceptable_combos(models: List[Dict[str, Any]], vram_gb: float = 6.0) -> List[Dict[str, Any]]:
    """Pairs large target models with verified compatible lightweight drafter models."""
    targets = [m for m in models if m["size_gb"] >= 5.0 and not m["is_assistant_head"]]
    drafters = [m for m in models if m["size_gb"] <= 4.5 and not m["is_assistant_head"]]

    combos = []
    for tgt in targets:
        best_drafter = None
        for d in drafters:
            # Architecture family match
            same_arch = (tgt["arch"] == d["arch"]) or (tgt["arch"].startswith("gemma") and d["arch"].startswith("gemma"))
            same_vocab = (tgt["vocab_size"] == d["vocab_size"]) if (tgt["vocab_size"] and d["vocab_size"]) else True
            if same_arch and same_vocab:
                if best_drafter is None or d["size_gb"] < best_drafter["size_gb"]:
                    best_drafter = d

        rec_layers = compute_recommended_layers(tgt["size_gb"], vram_gb)
        combos.append({
            "target": tgt,
            "drafter": best_drafter,
            "target_rec_layers": rec_layers,
            "drafter_rec_layers": 0,  # Drafters run on CPU to protect VRAM
            "fallback": "Prompt Lookup Decoding"
        })
    return combos

def open_mtp_combos_dialog(parent_win, app):
    """Renders the interactive MTP & Drafter Combos Manager modal."""
    from serenity_resources import THEME
    from System.serenity_utils import ToolTip

    win = tk.Toplevel(parent_win)
    win.title("⚡ MTP & Speculative Drafter Combos Manager")
    win.geometry("920x620")
    win.config(bg=THEME["bg_color"])
    win.transient(parent_win)
    win.attributes("-topmost", False)

    header_frame = tk.Frame(win, bg=THEME["bg_color"], pady=8, padx=12)
    header_frame.pack(fill=tk.X)

    tk.Label(header_frame, text="⚡ Speculative MTP & Drafter Alignment Engine",
             font=app.fonts["large"], fg=THEME["electric_blue"], bg=THEME["bg_color"]).pack(anchor="w")
    tk.Label(header_frame, text="Pairs primary inference cores with lightweight companion models for 1.5x–2.0x decode acceleration.",
             font=app.fonts["ui_small"], fg=THEME["fg_color"], bg=THEME["bg_color"]).pack(anchor="w")

    # Catch-all status badge
    badge_f = tk.Frame(header_frame, bg=THEME["widget_bg_color"], padx=8, pady=4)
    badge_f.pack(anchor="w", pady=(6, 0))
    tk.Label(badge_f, text="Catch-all Fallback active: ", font=app.fonts["ui_small"], fg=THEME["fg_color"], bg=THEME["widget_bg_color"]).pack(side=tk.LEFT)
    tk.Label(badge_f, text="Prompt Lookup Decoding (Instant, 0 extra RAM)", font=app.fonts["bold"], fg="#00ffcc", bg=THEME["widget_bg_color"]).pack(side=tk.LEFT)

    # Treeview frame
    tree_frame = tk.Frame(win, bg=THEME["bg_color"], padx=12, pady=6)
    tree_frame.pack(fill=tk.BOTH, expand=True)

    columns = ("target", "target_size", "drafter", "drafter_size", "rec_layers", "status")
    tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=12)
    tree.heading("target", text="Primary Target Model")
    tree.heading("target_size", text="Size (GB)")
    tree.heading("drafter", text="Assigned Companion Drafter")
    tree.heading("drafter_size", text="Drafter Size")
    tree.heading("rec_layers", text="Rec. GPU Layers")
    tree.heading("status", text="Speculative Alignment")

    tree.column("target", width=280)
    tree.column("target_size", width=70, anchor="center")
    tree.column("drafter", width=280)
    tree.column("drafter_size", width=80, anchor="center")
    tree.column("rec_layers", width=95, anchor="center")
    tree.column("status", width=120, anchor="center")

    scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscrollcommand=scroll.set)
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scroll.pack(side=tk.RIGHT, fill=tk.Y)

    combos_data = []

    def refresh_combos():
        nonlocal combos_data
        for item in tree.get_children():
            tree.delete(item)

        scan_dirs = ["S:/LLM", os.path.dirname(getattr(app, "model_path", "")) or ""]
        if hasattr(app, "model_paths"):
            scan_dirs.extend([os.path.dirname(p) for p in app.model_paths.values() if p])
        scan_dirs = list(set([d for d in scan_dirs if d and os.path.exists(d)]))

        models = scan_model_directories(scan_dirs)
        vram_budget = float(app.config.get("vram_threshold", 6.0))
        combos_data = find_acceptable_combos(models, vram_budget)
        mtp_mapping = app.config.get("mtp_mapping", {})

        for idx, c in enumerate(combos_data):
            tgt_name = c["target"]["filename"]
            tgt_path = c["target"]["path"]
            mapped_drafter = mtp_mapping.get(tgt_path)

            if mapped_drafter and os.path.exists(mapped_drafter):
                d_name = os.path.basename(mapped_drafter)
                d_size = f"{round(os.path.getsize(mapped_drafter)/(1024**3), 2)} GB"
                status = "✅ Paired (Custom)"
            elif c["drafter"]:
                d_name = c["drafter"]["filename"]
                d_size = f"{c['drafter']['size_gb']} GB"
                status = "⚡ Compatible"
            else:
                d_name = "[None - Prompt Lookup]"
                d_size = "0.0 GB"
                status = "🛡️ Catch-all Active"

            tree.insert("", tk.END, iid=str(idx), values=(
                tgt_name,
                f"{c['target']['size_gb']} GB",
                d_name,
                d_size,
                f"{c['target_rec_layers']} L (GPU)",
                status
            ))

    def on_auto_pair_all():
        mtp_mapping = app.config.get("mtp_mapping", {})
        paired_count = 0
        for c in combos_data:
            if c["drafter"]:
                mtp_mapping[c["target"]["path"]] = c["drafter"]["path"]
                paired_count += 1
        app.config["mtp_mapping"] = mtp_mapping
        app.config["speculative_drafting"] = True
        if hasattr(app, "save_config"):
            app.save_config()
        refresh_combos()
        messagebox.showinfo("Auto-Pair Complete", f"Successfully paired {paired_count} model combos and enabled Speculative MTP Drafting!", parent=win)

    def on_pair_selected():
        selected = tree.selection()
        if not selected:
            messagebox.showwarning("Select Combo", "Please select a model row first.", parent=win)
            return
        idx = int(selected[0])
        c = combos_data[idx]
        if not c["drafter"]:
            messagebox.showinfo("Catch-all Fallback", "No GGUF companion found for this model. SerenityPC will automatically engage Prompt Lookup Decoding on inference.", parent=win)
            return
        mtp_mapping = app.config.get("mtp_mapping", {})
        mtp_mapping[c["target"]["path"]] = c["drafter"]["path"]
        app.config["mtp_mapping"] = mtp_mapping
        app.config["speculative_drafting"] = True
        if hasattr(app, "save_config"):
            app.save_config()
        refresh_combos()
        messagebox.showinfo("Pair Saved", f"Paired {c['target']['filename']} with {c['drafter']['filename']}!", parent=win)

    def on_apply_rec_layers():
        selected = tree.selection()
        if not selected:
            messagebox.showwarning("Select Combo", "Please select a model row first.", parent=win)
            return
        idx = int(selected[0])
        c = combos_data[idx]
        rec = c["target_rec_layers"]
        tgt_name = c["target"]["filename"]

        # If model corresponds to an active tier in app, update layer config
        applied_tiers = []
        if hasattr(app, "model_paths") and hasattr(app, "gpu_layer_config"):
            for tier, p in app.model_paths.items():
                if p and os.path.basename(p) == tgt_name:
                    app.gpu_layer_config[tier] = rec
                    applied_tiers.append(tier)
            if hasattr(app, "save_config"):
                app.save_config()

        tiers_txt = f" (for tiers: {', '.join(applied_tiers)})" if applied_tiers else ""
        messagebox.showinfo("Recommended Layers", f"Optimal safe layer count for {tgt_name} is {rec} layers{tiers_txt}.\nPrevents PCIe Shared VRAM spill on 6GB GPUs.", parent=win)

    # Action Button Bar
    action_bar = tk.Frame(win, bg=THEME["bg_color"], pady=10, padx=12)
    action_bar.pack(fill=tk.X)

    btn_pair = tk.Button(action_bar, text="🔗 Pair Selected Drafter", command=on_pair_selected,
                         bg=THEME["button_active_color"], fg=THEME["fg_color"], font=app.fonts["ui_button"], padx=8, pady=3)
    btn_pair.pack(side=tk.LEFT, padx=(0, 6))
    ToolTip(btn_pair, "Assign matched companion drafter to selected model in config.", app=app)

    btn_auto = tk.Button(action_bar, text="⚡ Auto-Pair All Compatible", command=on_auto_pair_all,
                         bg=THEME["widget_bg_color"], fg="#00ffcc", font=app.fonts["ui_button"], padx=8, pady=3)
    btn_auto.pack(side=tk.LEFT, padx=6)
    ToolTip(btn_auto, "Automatically map all large models to their optimal companion drafters.", app=app)

    btn_rec = tk.Button(action_bar, text="🛡️ Apply Rec. GPU Layers", command=on_apply_rec_layers,
                        bg=THEME["widget_bg_color"], fg=THEME["electric_blue"], font=app.fonts["ui_button"], padx=8, pady=3)
    btn_rec.pack(side=tk.LEFT, padx=6)
    ToolTip(btn_rec, "Sets optimal layer offload budget (e.g. 8 layers for 26B on 6GB VRAM) to avoid PCIe spill.", app=app)

    btn_rescan = tk.Button(action_bar, text="🔄 Rescan Library", command=refresh_combos,
                           bg=THEME["widget_bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_button"], padx=8, pady=3)
    btn_rescan.pack(side=tk.RIGHT, padx=6)
    ToolTip(btn_rescan, "Rescans model directories for newly added models.", app=app)

    refresh_combos()
    return win
