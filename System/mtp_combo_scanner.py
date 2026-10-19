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

def has_baked_in_mtp(path: str, kv: Dict[str, Any] = None) -> bool:
    """Detects whether a GGUF model contains native baked-in Multi-Token Prediction (MTP) layers."""
    if not path:
        return False
    if kv is None:
        if not os.path.exists(path):
            return False
        kv = read_gguf_header_fast(path)
    for k, v in kv.items():
        if k.endswith(".nextn_predict_layers") or k == "general.nextn_predict_layers":
            try:
                if int(v) > 0:
                    return True
            except (ValueError, TypeError):
                pass
    f_lower = os.path.basename(path).lower()
    if any(tag in f_lower for tag in ["q8mtp", "q4mtp", "baked_mtp", "-mtp-native", "_mtp_native"]):
        return True
    return False

def is_mtp_assistant(path: str, kv: Dict[str, Any] = None) -> bool:
    """Detects whether a GGUF model is a dedicated MTP assistant/auxiliary model."""
    if not path:
        return False
    if kv is None:
        if not os.path.exists(path):
            return False
        kv = read_gguf_header_fast(path)
    arch = str(kv.get("general.architecture", "")).replace("-", "_").lower()
    if arch.endswith("_assistant") or "assistant" in arch or "mtp" in arch:
        return True
    f_lower = os.path.basename(path).lower()
    return any(k in f_lower for k in ["-assistant", "_assistant", "assistant-mtp", "-mtp-", "_mtp_"])

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
                baked_in = has_baked_in_mtp(full_path, kv)
                is_assistant = is_mtp_assistant(full_path, kv)

                discovered.append({
                    "filename": f,
                    "path": full_path,
                    "size_gb": size_gb,
                    "arch": arch,
                    "vocab_size": vocab_size,
                    "context_len": context_len,
                    "is_assistant_head": is_assistant,
                    "has_baked_in_mtp": baked_in
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
    """Pairs large target models with verified compatible lightweight drafter models (Baked-in > Dedicated MTP > Bundled/Adjacent > Prompt Lookup)."""
    targets = [m for m in models if m["size_gb"] >= 5.0 and not m["is_assistant_head"]]
    assistants = [m for m in models if m["is_assistant_head"]]
    drafters = [m for m in models if m["size_gb"] <= 4.5 and not m["is_assistant_head"]]

    combos = []
    for tgt in targets:
        rec_layers = compute_recommended_layers(tgt["size_gb"], vram_gb)
        if tgt.get("has_baked_in_mtp"):
            combos.append({
                "target": tgt,
                "drafter": None,
                "has_baked_in_mtp": True,
                "is_mtp_assistant": False,
                "target_rec_layers": rec_layers,
                "drafter_rec_layers": 0,
                "fallback": "Native Baked-in MTP"
            })
            continue

        best_drafter = None
        is_native_mtp = False

        # 1. First priority: Check dedicated MTP assistant heads matching target
        tgt_dir = os.path.dirname(os.path.abspath(tgt["path"]))
        for a in assistants:
            a_dir = os.path.dirname(os.path.abspath(a["path"]))
            same_dir = (tgt_dir == a_dir)
            same_family = (
                (tgt["arch"].startswith("gemma") and ("gemma" in a["arch"].lower() or "gemma" in a["filename"].lower()))
                or (tgt["arch"].startswith("qwen") and ("qwen" in a["arch"].lower() or "qwen" in a["filename"].lower()))
            )
            if same_dir and same_family:
                best_drafter = a
                is_native_mtp = True
                break
            elif same_family and best_drafter is None:
                best_drafter = a
                is_native_mtp = True

        # 2. Second priority: Standalone small companion drafters
        if best_drafter is None:
            for d in drafters:
                same_arch = (tgt["arch"] == d["arch"]) or (tgt["arch"].startswith("gemma") and d["arch"].startswith("gemma"))
                same_vocab = (tgt["vocab_size"] == d["vocab_size"]) if (tgt["vocab_size"] and d["vocab_size"]) else True
                if same_arch and same_vocab:
                    if best_drafter is None or d["size_gb"] < best_drafter["size_gb"]:
                        best_drafter = d

        combos.append({
            "target": tgt,
            "drafter": best_drafter,
            "has_baked_in_mtp": False,
            "is_mtp_assistant": is_native_mtp,
            "target_rec_layers": rec_layers,
            "drafter_rec_layers": 0,  # Drafters run on CPU to protect VRAM
            "fallback": "Native MTP Assistant" if is_native_mtp else "Prompt Lookup Decoding"
        })
    return combos

def open_mtp_combos_dialog(parent_win, app):
    """Renders the interactive MTP & Drafter Combos Manager modal."""
    from serenity_resources import THEME
    from System.serenity_utils import ToolTip
    from tkinter import filedialog

    win = tk.Toplevel(parent_win)
    win.title("⚡ MTP & Speculative Drafter Combos Manager")
    win.geometry("960x680")
    win.config(bg=THEME["bg_color"])
    win.transient(parent_win)
    win.attributes("-topmost", False)

    header_frame = tk.Frame(win, bg=THEME["bg_color"], pady=8, padx=12)
    header_frame.pack(fill=tk.X)

    tk.Label(header_frame, text="⚡ Speculative MTP & Drafter Alignment Engine",
             font=app.fonts["large"], fg=THEME["electric_blue"], bg=THEME["bg_color"]).pack(anchor="w")
    tk.Label(header_frame, text="Pairs primary inference cores with companion drafters. Priority: Native Baked-in > Bundled/Adjacent > Prompt Lookup.",
             font=app.fonts["ui_small"], fg=THEME["fg_color"], bg=THEME["bg_color"]).pack(anchor="w")

    # Status / Priority badges
    badge_f = tk.Frame(header_frame, bg=THEME["widget_bg_color"], padx=8, pady=4)
    badge_f.pack(anchor="w", pady=(6, 0))
    tk.Label(badge_f, text="Priority Hierarchy: ", font=app.fonts["ui_small"], fg=THEME["fg_color"], bg=THEME["widget_bg_color"]).pack(side=tk.LEFT)
    tk.Label(badge_f, text="⚡ 1. Baked-in MTP   →   🔗 2. Bundled Drafter GGUF   →   🛡️ 3. Prompt Lookup", font=app.fonts["bold"], fg="#00ffcc", bg=THEME["widget_bg_color"]).pack(side=tk.LEFT)

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
    tree.column("status", width=140, anchor="center")

    scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscrollcommand=scroll.set)
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scroll.pack(side=tk.RIGHT, fill=tk.Y)

    combos_data = []
    custom_scanned_dirs = []

    def refresh_combos():
        nonlocal combos_data
        for item in tree.get_children():
            tree.delete(item)

        # 1. Search model's containing folder FIRST, then parent, then model_paths, then custom, then S:/LLM
        scan_dirs = []
        cur_p = getattr(app, "model_path", "") or ""
        if cur_p and os.path.exists(cur_p):
            scan_dirs.append(os.path.dirname(cur_p))
            scan_dirs.append(os.path.dirname(os.path.dirname(cur_p)))

        if hasattr(app, "model_paths") and isinstance(app.model_paths, dict):
            for p in app.model_paths.values():
                if p and os.path.exists(p):
                    scan_dirs.append(os.path.dirname(p))

        if custom_scanned_dirs:
            scan_dirs.extend(custom_scanned_dirs)

        if hasattr(app, "dirs") and "Models" in app.dirs:
            scan_dirs.append(app.dirs["Models"])
        scan_dirs.append("S:/LLM")

        # Deduplicate while preserving first-seen order (folder of active model searched first)
        seen_d = set()
        ordered_dirs = []
        for d in scan_dirs:
            if d and os.path.exists(d):
                norm = os.path.normcase(os.path.abspath(d))
                if norm not in seen_d:
                    seen_d.add(norm)
                    ordered_dirs.append(d)

        models = scan_model_directories(ordered_dirs)
        vram_budget = float(app.config.get("vram_threshold", 6.0))
        combos_data = find_acceptable_combos(models, vram_budget)
        mtp_mapping = app.config.get("mtp_mapping", {})

        for idx, c in enumerate(combos_data):
            tgt_name = c["target"]["filename"]
            tgt_path = c["target"]["path"]
            mapped_drafter = mtp_mapping.get(tgt_path)

            if c.get("has_baked_in_mtp"):
                d_name = "[Native Baked-in MTP]"
                d_size = f"{c['target']['size_gb']} GB"
                status = "⚡ Baked-in MTP (Native)"
            elif mapped_drafter and os.path.exists(mapped_drafter):
                d_name = os.path.basename(mapped_drafter)
                d_size = f"{round(os.path.getsize(mapped_drafter)/(1024**3), 2)} GB"
                status = "✅ Paired (Custom)"
            elif c.get("is_mtp_assistant") and c["drafter"]:
                d_name = c["drafter"]["filename"]
                d_size = f"{c['drafter']['size_gb']} GB"
                status = "⚡ MTP Head (Native)"
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
            if c.get("has_baked_in_mtp"):
                continue
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
        if c.get("has_baked_in_mtp"):
            messagebox.showinfo("Native Baked-in MTP", f"{c['target']['filename']} already features native baked-in MTP layers!\nNo external companion drafter is needed.", parent=win)
            return
        if not c["drafter"]:
            messagebox.showinfo("Catch-all Fallback", "No automatic GGUF companion found for this model. Use '📂 Manual Pair Drafter...' to choose a companion, or Prompt Lookup Decoding will engage.", parent=win)
            return
        mtp_mapping = app.config.get("mtp_mapping", {})
        mtp_mapping[c["target"]["path"]] = c["drafter"]["path"]
        app.config["mtp_mapping"] = mtp_mapping
        app.config["speculative_drafting"] = True
        if hasattr(app, "save_config"):
            app.save_config()
        refresh_combos()
        messagebox.showinfo("Pair Saved", f"Paired {c['target']['filename']} with {c['drafter']['filename']}!", parent=win)

    def on_manual_pair_drafter():
        selected = tree.selection()
        if not selected:
            messagebox.showwarning("Select Model", "Please select a primary model row first to assign a drafter to.", parent=win)
            return
        idx = int(selected[0])
        c = combos_data[idx]
        init_dir = os.path.dirname(c["target"]["path"]) if os.path.exists(c["target"]["path"]) else (app.dirs.get("Models") or "S:/LLM")
        manual_path = filedialog.askopenfilename(
            parent=win,
            title=f"Select Companion Drafter for {c['target']['filename']}",
            filetypes=[("GGUF Models", "*.gguf"), ("All Files", "*.*")],
            initialdir=init_dir
        )
        if manual_path and os.path.exists(manual_path):
            if os.path.normcase(os.path.abspath(manual_path)) == os.path.normcase(os.path.abspath(c["target"]["path"])):
                messagebox.showerror("Invalid Pairing", "A model cannot be paired with itself as its own drafter.", parent=win)
                return
            mtp_mapping = app.config.get("mtp_mapping", {})
            mtp_mapping[c["target"]["path"]] = manual_path
            app.config["mtp_mapping"] = mtp_mapping
            app.config["speculative_drafting"] = True
            if hasattr(app, "save_config"):
                app.save_config()
            refresh_combos()
            messagebox.showinfo("Manual Pair Saved", f"Successfully mapped companion drafter:\n{os.path.basename(manual_path)}\nto primary model:\n{c['target']['filename']}", parent=win)

    def on_scan_custom_folder():
        chosen_dir = filedialog.askdirectory(parent=win, title="Select Directory to Scan for Models & Drafters")
        if chosen_dir and os.path.exists(chosen_dir):
            if chosen_dir not in custom_scanned_dirs:
                custom_scanned_dirs.insert(0, chosen_dir)
            refresh_combos()

    def on_apply_rec_layers():
        selected = tree.selection()
        if not selected:
            messagebox.showwarning("Select Combo", "Please select a model row first.", parent=win)
            return
        idx = int(selected[0])
        c = combos_data[idx]
        rec = c["target_rec_layers"]
        tgt_name = c["target"]["filename"]

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

    # 2-Row Action Button Bar so buttons fit cleanly without window overflow
    action_bar = tk.Frame(win, bg=THEME["bg_color"], pady=8, padx=12)
    action_bar.pack(fill=tk.X)

    # Row 1: Primary Pairing Actions
    r1 = tk.Frame(action_bar, bg=THEME["bg_color"])
    r1.pack(fill=tk.X, pady=(0, 4))

    btn_pair = tk.Button(r1, text="🔗 Pair Selected Drafter", command=on_pair_selected,
                         bg=THEME["button_active_color"], fg=THEME["fg_color"], font=app.fonts["ui_button"], padx=8, pady=3)
    btn_pair.pack(side=tk.LEFT, padx=(0, 6))
    ToolTip(btn_pair, "Assign automatically matched companion drafter to selected model.", app=app)

    btn_manual = tk.Button(r1, text="📂 Manual Pair Drafter...", command=on_manual_pair_drafter,
                           bg=THEME["widget_bg_color"], fg=THEME.get("accent_highlight", "#00ffcc"), font=app.fonts["ui_button"], padx=8, pady=3)
    btn_manual.pack(side=tk.LEFT, padx=6)
    ToolTip(btn_manual, "Manually browse and assign any compatible companion GGUF to the selected model.", app=app)

    btn_auto = tk.Button(r1, text="⚡ Auto-Pair All Compatible", command=on_auto_pair_all,
                         bg=THEME["widget_bg_color"], fg="#00ffcc", font=app.fonts["ui_button"], padx=8, pady=3)
    btn_auto.pack(side=tk.LEFT, padx=6)
    ToolTip(btn_auto, "Automatically map all large models to their optimal companion drafters.", app=app)

    btn_rec = tk.Button(r1, text="🛡️ Apply Rec. GPU Layers", command=on_apply_rec_layers,
                        bg=THEME["widget_bg_color"], fg=THEME["electric_blue"], font=app.fonts["ui_button"], padx=8, pady=3)
    btn_rec.pack(side=tk.LEFT, padx=6)
    ToolTip(btn_rec, "Sets optimal layer offload budget (e.g. 8 layers for 26B on 6GB VRAM) to avoid PCIe spill.", app=app)

    # Row 2: Library & Folder Scanning Actions
    r2 = tk.Frame(action_bar, bg=THEME["bg_color"])
    r2.pack(fill=tk.X, pady=(2, 0))

    btn_folder = tk.Button(r2, text="📁 Scan Folder...", command=on_scan_custom_folder,
                           bg=THEME["widget_bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_button"], padx=8, pady=3)
    btn_folder.pack(side=tk.LEFT, padx=(0, 6))
    ToolTip(btn_folder, "Scan a specific custom directory for models and companion drafters.", app=app)

    btn_rescan = tk.Button(r2, text="🔄 Rescan Library", command=refresh_combos,
                           bg=THEME["widget_bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_button"], padx=8, pady=3)
    btn_rescan.pack(side=tk.RIGHT, padx=6)
    ToolTip(btn_rescan, "Rescans all model directories (searching active model directory first).", app=app)

    refresh_combos()
    return win
