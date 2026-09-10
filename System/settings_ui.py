import os
import json
import sys
import tkinter as tk
from tkinter import messagebox, filedialog, ttk, simpledialog
from serenity_resources import THEME, THEMES
try:
    from System.serenity_utils import ToolTip, TutorialOverlay, bind_entry_limit
except ImportError:
    from serenity_utils import ToolTip, TutorialOverlay, bind_entry_limit

def run_auto_detect(app, window=None):
    """
    Attempts to automatically calculate optimal GPU layers based on VRAM,
    model complexity (MoE vs Dense), and non-linear KV-cache requirements.
    [RLHF]: Incorporates stability feedback from historical session loads.
    """
    app._log_and_display("Analyzing hardware/model complexity (RLHF Alpha)...")
    
    # Access globals from main module
    main_module = sys.modules.get('__main__')
    system_monitor_loaded = getattr(main_module, 'SYSTEM_MONITOR_LOADED', False)
    nvidia_ml = getattr(main_module, 'nvidia_ml', None)
    
    # 1. Determine VRAM source
    manual_vram_mb = app.state.get("virtual_vram", 0)
    vram_gb = None
    if manual_vram_mb > 0:
        vram_gb = manual_vram_mb / 1024
        app._log_and_display(f"Using Manual VRAM Target: {vram_gb:.2f}GB")
    elif system_monitor_loaded and nvidia_ml is not None and getattr(app, 'gpu_handle', None):
        try:
            mem = nvidia_ml.nvmlDeviceGetMemoryInfo(app.gpu_handle)
            vram_gb = mem.total / 1024**3
        except Exception: pass

    if vram_gb is None:
         app._log_and_display("Hardware detection offline. Defaulting to CPU.")
         vram_gb = 0

    # 2. RLHF Stability Feedback
    rlhf_penalty = 0
    rlhf_path = os.path.join(app.dirs["System"], "rlhf_stability.json")
    if os.path.exists(rlhf_path):
        try:
            with open(rlhf_path, 'r') as f:
                rlhf_data = json.load(f)
                rlhf_penalty = rlhf_data.get("vram_global_penalty", 0)
        except: pass

    # 3. Model-Aware Tier Scaling
    tiers = ["fast", "search", "low", "med", "high", "transcendent", "secret", "deep_cook", 
             "vision_video", "vision_video_deep", "vision_multimodal"]
    
    recommendations = {}
    for tier in tiers:
        path = app.model_paths.get(tier, "").lower()
        
        # APEX GUARD: Force everything to -1 for E-series
        if any(x in path for x in ["e2b", "e4b", "tiny"]):
            recommendations[tier] = -1
            continue

        # MOE/Large Model GUARD: Cap at 14 layers for 6GB stability (Verified limit)
        if vram_gb < 7 and any(x in path for x in ["26b", "31b", "moe"]):
                recommendations[tier] = 14
                continue
        
        # 4. Standard Linear Calculation (Layers per GB)
        base_ratio = 4.5
        calc = int((vram_gb - (rlhf_penalty / 1024)) * base_ratio)
        
        # 5. Cap to sensible limits
        final = max(0, min(64, calc))
        recommendations[tier] = final

    app._log_and_display(f"Auto-detection complete. Applied {rlhf_penalty} layers RLHF safety margin.")
    return recommendations


try:
    from System.settings_tabs import (
        build_models_tab, build_inference_tab, build_agents_tab,
        build_additional_tab, build_users_tab, build_personalize_tab,
        bind_radio_contrast
    )
except ImportError:
    from settings_tabs import (
        build_models_tab, build_inference_tab, build_agents_tab,
        build_additional_tab, build_users_tab, build_personalize_tab,
        bind_radio_contrast
    )


def open_settings_window(app, is_generating=None):
    """
    Opens the reorganized SerenityPC settings window (V1.7.0).
    Features a 6-tabbed layout:
    1. Models & Params
    2. Inference
    3. Agents
    4. Additional Settings (including Status Bar / Loading Bar)
    5. Users & Security
    6. Personalize
    Supports selective generation-state guarding (allows settings to open during generation
    while disabling only model weights and layer offloads).
    """
    try:
        win = tk.Toplevel(app.root)
        win.title("Model Settings")
        if app.icon_path:
            try: win.iconbitmap(app.icon_path)
            except: pass
        win.geometry(app.config.get("settings_window_geometry", "860x950"))
        win.config(bg=THEME["bg_color"])
        win.transient(app.root)
        win.attributes("-topmost", False)

        if not hasattr(app, "fonts") or not isinstance(app.fonts, dict):
            app.fonts = {
                "ui_button": ("Segoe UI", 9),
                "ui_small": ("Segoe UI", 8),
                "ui_label": ("Segoe UI", 9),
                "bold": ("Segoe UI", 9, "bold"),
                "log": ("Consolas", 9),
                "log_bold": ("Consolas", 9, "bold"),
                "title": ("Segoe UI", 11, "bold")
            }

        if is_generating is None:
            is_generating = bool(getattr(app, 'state', {}).get('running', False))

        generation_sensitive_widgets = []

        # --- 1. Fixed Top Action Bar ---
        btn_frame = tk.Frame(win, bg=THEME["bg_color"], pady=4)
        btn_frame.pack(side=tk.TOP, fill=tk.X, padx=10)

        tut_btn = tk.Button(btn_frame, text="🚀 Tutorial Walkthrough", 
                            command=lambda: (win.destroy(), getattr(app, 'start_tutorial_walkthrough', lambda: None)()),
                            bg=THEME["widget_bg_color"], fg=THEME.get("accent_highlight", "#00ffcc"),
                            font=app.fonts["ui_button"], relief=tk.FLAT)
        tut_btn.pack(side=tk.LEFT, padx=2)
        ToolTip(tut_btn, "Launch the interactive translucent tutorial walkthrough for Serenity PC.", app=app)

        # Generation banner if generation is currently active
        if is_generating:
            lbl_gen_banner = tk.Label(btn_frame, text="[⚡ Running — Model paths & allocations locked]",
                                      bg=THEME["bg_color"], fg="#ffaa00", font=app.fonts["ui_small"])
            lbl_gen_banner.pack(side=tk.LEFT, padx=8)

        # --- 2. Top Tab Navigation Bar ---
        tab_nav_frame = tk.Frame(win, bg=THEME["bg_color"], pady=3)
        tab_nav_frame.pack(side=tk.TOP, fill=tk.X, padx=10)

        # --- 3. Scrollable Content Container ---
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
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def on_closing():
            try:
                geom = win.winfo_geometry()
                if geom and "x" in geom and not geom.startswith("1x1"):
                    app.config["settings_window_geometry"] = geom
                    if hasattr(app, "save_config"):
                        app.save_config()
            except Exception: pass
            try:
                canvas.unbind_all("<MouseWheel>")
                print("[UI] Settings listener detached.")
            except: pass
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", on_closing)

        # --- 4. Setup State Variables & Bindings ---
        labels, ents, ctx_ents, n_batch_ents = {}, {}, {}, {}
        temp_ents, top_p_ents, min_p_ents, top_k_ents = {}, {}, {}, {}
        rep_ents, freq_ents, pres_ents, stop_ents = {}, {}, {}, {}

        # STT Input Devices
        from System.stt_manager import STTManager
        stt_devs = STTManager.get_input_devices()
        dev_names = ["Default Input Device"] + [f"{d['id']}: {d['name'][:24]}" for d in stt_devs]
        dev_id_map = {"Default Input Device": None}
        for d in stt_devs:
            dev_id_map[f"{d['id']}: {d['name'][:24]}"] = d["id"]
        curr_dev_idx = app.config.get("stt_device_index", None)
        curr_dev_label = "Default Input Device"
        if curr_dev_idx is not None:
            for k, v in dev_id_map.items():
                if v == curr_dev_idx:
                    curr_dev_label = k
                    break

        # Theme and Texture Mapping
        THEME_MAP = {
            "Apex (Default)": "apex",
            "Goth / Obsidian Dark": "goth",
            "Crystal Cavern": "crystal_cavern",
            "Yellow Blacket": "yellow_blacket",
            "Natural (Earth / Moss)": "natural",
            "Matrix (Cyber Green)": "matrix",
            "Persona (Level Dynamic)": "persona"
        }
        THEME_REV_MAP = {v: k for k, v in THEME_MAP.items()}
        curr_theme_key = app.config.get("theme", "apex")
        if curr_theme_key == "default": curr_theme_key = "apex"

        TEXTURE_MAP = {
            "Default Original": "default",
            "Frosted Glass": "frosted_glass",
            "Gloss": "gloss",
            "Metallic": "metallic",
            "Muted": "muted",
            "Iridescent": "iridescent",
            "Pearlescent": "pearlescent"
        }
        TEXTURE_REV_MAP = {v: k for k, v in TEXTURE_MAP.items()}
        curr_tex_key = app.config.get("texture_style", "default")

        TEXT_SCALE_PRESETS = [
            ("85% (Compact)", 85),
            ("100% (Standard)", 100),
            ("115% (Medium)", 115),
            ("125% (Large)", 125),
            ("140% (X-Large)", 140),
            ("160% (Huge)", 160),
            ("180% (Massive)", 180),
            ("200% (Maximum)", 200)
        ]
        SCALE_MAP = {lbl: val for lbl, val in TEXT_SCALE_PRESETS}
        SCALE_REV_MAP = {val: lbl for lbl, val in TEXT_SCALE_PRESETS}
        curr_text_scale = int(app.config.get("text_scale", 100))

        UI_FONT_OPTIONS = [
            "Segoe UI", "Verdana", "Times New Roman", "Cambria",
            "Comic Sans MS", "Gothic A1", "Modiableic",
            "Comfortaa", "YU Gothic UI", "Segoe UI Variable",
            "Modern Antiqua", "Quicksand", "UnifrakturMaguntia",
        ]
        MONO_FONT_OPTIONS = [
            "Doto", "Inconsolata", "Noto Sans Mono", "Lucida Sans Console",
            "Palatino Linotype", "Tajawal", "MV Boli", "Didact Gothic",
        ]

        init_reasoning = app.config.get("reasoning_strength", app.config.get("muse_reasoning_strength", "medium"))
        if init_reasoning == "minimal": init_reasoning = "low"
        elif init_reasoning == "maximum": init_reasoning = "xhigh"

        auto_lock_sec = app.vault_manager.get_auto_lock_seconds() if hasattr(app, 'vault_manager') else 0

        vars_dict = {
            "labels": labels, "ents": ents, "ctx_ents": ctx_ents, "n_batch_ents": n_batch_ents,
            "temp_ents": temp_ents, "top_p_ents": top_p_ents, "min_p_ents": min_p_ents, "top_k_ents": top_k_ents,
            "rep_ents": rep_ents, "freq_ents": freq_ents, "pres_ents": pres_ents, "stop_ents": stop_ents,
            "active_template": tk.StringVar(value=""),
            "template_mode": tk.StringVar(value="modify"),
            "dynamic_params_var": tk.BooleanVar(value=app.config.get("dynamic_params_enabled", True)),
            "hao_var": tk.StringVar(value=app.config.get("hao_preset", "exps=CPU")),
            "repeat_mode_var": tk.StringVar(value=app.config.get("repeat_detection_mode", "lazy")),
            "ratio_var": tk.IntVar(value=app.config.get("max_token_ratio", 4)),
            "overfill_behavior_var": tk.StringVar(value=app.config.get("overfill_behavior_mode", app.config.get("budget_recovery_mode", "wrapup"))),
            "halt_behavior_var": tk.StringVar(value=app.config.get("halt_behavior_mode", "off")),
            "stream_var": tk.StringVar(value=app.state.get("streaming_mode", "Buffered")),
            "v_behavior": tk.StringVar(value=app.state.get("deep_cook_behavior", "oneshot")),
            "swa_var": tk.StringVar(value=app.config.get("swa_kv_cache", "Auto")),
            "k_cache_var": tk.StringVar(value=app.config.get("k_cache_type", "q8_0").lower()),
            "v_cache_var": tk.StringVar(value=app.config.get("v_cache_type", "q8_0").lower()),
            "history_mode_var": tk.StringVar(value=app.config.get("history_mode", "TurboVec" if app.config.get("turbovec_mode") == "on" else ("Off" if app.config.get("turbovec_mode") == "off" else "Keyword"))),
            "history_usage_var": tk.StringVar(value=app.config.get("history_usage", "all")),
            "history_lookup_var": tk.StringVar(value=app.config.get("history_lookup_mode", "targeted")),
            "reasoning_var": tk.StringVar(value=init_reasoning),
            "delegation_enabled_var": tk.BooleanVar(value=app.config.get("delegation_enabled", False)),
            "delegation_model_mode_var": tk.StringVar(value=app.config.get("delegation_model_mode", "lvl6_7_model")),
            "subagent_density_var": tk.StringVar(value=app.config.get("subagent_selection_mode", "minimal")),
            "cecilia_mode_var": tk.StringVar(value=app.config.get("cecilia_delegation_mode", "shadow_wizard")),
            "chain_preset_var": tk.StringVar(value=app.config.get("delegation_chain_preset", "standard")),
            "handoff_target_var": tk.StringVar(value=app.config.get("delegation_handoff_target", "lvl3_compiler")),
            "offline_mode_var": tk.BooleanVar(value=app.config.get("offline_mode", False)),
            "auto_vram_var": tk.BooleanVar(value=app.config.get("auto_vram_offload", False)),
            "spec_draft_var": tk.BooleanVar(value=app.config.get("speculative_drafting", False)),
            "ghost_var": tk.BooleanVar(value=app.config.get("ghost_mode", False)),
            "thinking_var": tk.BooleanVar(value=app.config.get("thinking_checkbox", True)),
            "benchmark_var": tk.BooleanVar(value=app.config.get("benchmark_enabled", False)),
            "inline_md_var": tk.BooleanVar(value=app.config.get("inline_markdown", True)),
            "format_prompt_md_var": tk.BooleanVar(value=app.config.get("format_prompt_markdown", False)),
            "monitor_graph_var": tk.BooleanVar(value=app.config.get("monitor_graph_mode", False)),
            "show_tooltips_var": tk.BooleanVar(value=app.config.get("show_tooltips", True)),
            "resp_len_var": tk.StringVar(value=app.config.get("response_length", "natural")),
            "media_var": tk.IntVar(value=app.config.get("media_rendering", 1)),
            "dev_names": dev_names,
            "stt_dev_var": tk.StringVar(value=curr_dev_label),
            "stt_lang_var": tk.StringVar(value=app.config.get("stt_language", "en-US")),
            "multimedia_handling_var": tk.StringVar(value=app.config.get("multimedia_handling", app.config.get("image_handling", "auto"))),
            "dmn_enabled_var": tk.BooleanVar(value=app.config.get("dmn_enabled", True)),
            "sc_val": tk.IntVar(value=getattr(app, 'sub_chunk_size', 8)),
            "scroll_lock_var": tk.BooleanVar(value=app.config.get("scroll_lock_enabled", False)),
            "status_mode_var": tk.StringVar(value=app.config.get("status_bar_mode", "hybrid")),
            "anim_style_var": tk.StringVar(value=app.config.get("status_bar_anim_style", "spinner")),
            "sb_dmn_var": tk.BooleanVar(value=app.config.get("status_bar_dmn_idle", True)),
            "sb_fallback_var": tk.BooleanVar(value=app.config.get("status_bar_fallback_info", True)),
            "sb_linger_var": tk.DoubleVar(value=float(app.config.get("status_bar_linger_sec", 5.0))),
            "user_profiles_list": app.list_user_profiles() if hasattr(app, 'list_user_profiles') else ["Default"],
            "username_var": tk.StringVar(value=app.get_active_username() if hasattr(app, 'get_active_username') else app.config.get("username", "Default")),
            "show_def_var": tk.BooleanVar(value=app.config.get("show_default_profile", True)),
            "show_pub_var": tk.BooleanVar(value=app.config.get("show_public_profile", True)),
            "user_pref_name_var": tk.StringVar(value=app.config.get("user_preferred_name", "")),
            "user_addr_style_var": tk.StringVar(value=app.config.get("user_address_style", "Direct / Plain")),
            "auto_lock_var": tk.StringVar(value=str(auto_lock_sec)),
            "THEME_MAP": THEME_MAP,
            "theme_display_var": tk.StringVar(value=THEME_REV_MAP.get(curr_theme_key, "Apex (Default)")),
            "TEXTURE_MAP": TEXTURE_MAP,
            "tex_display_var": tk.StringVar(value=TEXTURE_REV_MAP.get(curr_tex_key, "Default Original")),
            "tex_int_var": tk.IntVar(value=int(float(app.config.get("texture_intensity", 1.0)) * 100)),
            "dark_mode_var": tk.BooleanVar(value=app.config.get("dark_mode", False)),
            "SCALE_MAP": SCALE_MAP,
            "text_scale_display_var": tk.StringVar(value=SCALE_REV_MAP.get(curr_text_scale, f"{curr_text_scale}%")),
            "text_scale_val_var": tk.IntVar(value=curr_text_scale),
            "UI_FONT_OPTIONS": UI_FONT_OPTIONS,
            "MONO_FONT_OPTIONS": MONO_FONT_OPTIONS,
            "ui_font_var": tk.StringVar(value=app.config.get("ui_font", "Segoe UI")),
            "mono_font_var": tk.StringVar(value=app.config.get("mono_font", "Consolas")),
            "open_scaling_center_fn": lambda: open_text_scaling_center(app, win)
        }
        vars_dict["budget_recovery_var"] = vars_dict["overfill_behavior_var"]
        vars_dict["turbovec_mode_var"] = vars_dict["history_mode_var"]

        # Password Modal Handlers
        def _open_set_password_modal():
            from System.vault_manager import DISCLAIMER_WARNING_TEXT
            pwd_win = tk.Toplevel(win)
            pwd_win.title("Set Master Vault Password")
            pwd_win.geometry("540x520")
            pwd_win.config(bg=THEME["bg_color"])
            pwd_win.transient(win)
            pwd_win.grab_set()

            disc_frame = tk.Frame(pwd_win, bg="#330000", bd=2, relief=tk.RIDGE)
            disc_frame.pack(fill=tk.X, padx=12, pady=10)
            tk.Label(disc_frame, text=DISCLAIMER_WARNING_TEXT, bg="#330000", fg="#ffcc00",
                     font=app.fonts["log_bold"], justify=tk.LEFT).pack(padx=8, pady=8)

            fields_frame = tk.Frame(pwd_win, bg=THEME["bg_color"])
            fields_frame.pack(fill=tk.X, padx=16, pady=4)

            is_already_enabled = hasattr(app, 'vault_manager') and app.vault_manager.is_lock_enabled()
            curr_pwd_var = tk.StringVar()
            new_pwd_var = tk.StringVar()
            confirm_pwd_var = tk.StringVar()

            row_idx = 0
            if is_already_enabled:
                tk.Label(fields_frame, text="Current Password:", bg=THEME["bg_color"], fg=THEME["fg_color"]).grid(row=row_idx, column=0, sticky="w", pady=4)
                curr_entry = tk.Entry(fields_frame, textvariable=curr_pwd_var, show="*", width=24,
                                      bg=THEME["widget_bg_color"], fg=THEME["fg_color"],
                                      insertbackground=THEME.get("electric_blue", THEME["fg_color"]))
                bind_entry_limit(curr_entry, max_len=64)
                curr_entry.grid(row=row_idx, column=1, padx=6, pady=4)
                row_idx += 1

            tk.Label(fields_frame, text="New Master Password:", bg=THEME["bg_color"], fg=THEME["fg_color"]).grid(row=row_idx, column=0, sticky="w", pady=4)
            new_entry = tk.Entry(fields_frame, textvariable=new_pwd_var, show="*", width=24,
                                 bg=THEME["widget_bg_color"], fg=THEME["fg_color"],
                                 insertbackground=THEME.get("electric_blue", THEME["fg_color"]))
            bind_entry_limit(new_entry, max_len=64)
            new_entry.grid(row=row_idx, column=1, padx=6, pady=4)
            row_idx += 1

            tk.Label(fields_frame, text="Confirm Password:", bg=THEME["bg_color"], fg=THEME["fg_color"]).grid(row=row_idx, column=0, sticky="w", pady=4)
            conf_entry = tk.Entry(fields_frame, textvariable=confirm_pwd_var, show="*", width=24,
                                  bg=THEME["widget_bg_color"], fg=THEME["fg_color"],
                                  insertbackground=THEME.get("electric_blue", THEME["fg_color"]))
            bind_entry_limit(conf_entry, max_len=64)
            conf_entry.grid(row=row_idx, column=1, padx=6, pady=4)

            def _apply_new_pwd():
                new_p = new_pwd_var.get().strip()
                conf_p = confirm_pwd_var.get().strip()
                curr_p = curr_pwd_var.get().strip() if is_already_enabled else None
                if len(new_p) < 4:
                    messagebox.showerror("Error", "Password must be at least 4 characters long.", parent=pwd_win)
                    return
                if new_p != conf_p:
                    messagebox.showerror("Error", "New password and confirmation do not match.", parent=pwd_win)
                    return
                if not messagebox.askyesno("CONFIRM ENCRYPTION", "ARE YOU ABSOLUTELY SURE?\n\nIf you lose this password, ALL history files will be PERMANENTLY lost.\n\nProceed with AES-256-GCM history migration?", parent=pwd_win):
                    return
                success, msg = app.vault_manager.set_password(new_p, curr_p)
                if success:
                    messagebox.showinfo("Vault Configured", msg, parent=pwd_win)
                    if "refresh_vault_status" in vars_dict: vars_dict["refresh_vault_status"]()
                    pwd_win.destroy()
                else:
                    messagebox.showerror("Vault Error", msg, parent=pwd_win)

            btn_box = tk.Frame(pwd_win, bg=THEME["bg_color"])
            btn_box.pack(fill=tk.X, padx=16, pady=12)
            tk.Button(btn_box, text="Encrypt & Set Password", command=_apply_new_pwd,
                      bg=THEME["button_active_color"], fg=THEME["fg_color"], font=app.fonts["ui_button"]).pack(side=tk.LEFT, padx=4)
            tk.Button(btn_box, text="Cancel", command=pwd_win.destroy,
                      bg=THEME["button_bg_color"], fg=THEME["fg_color"]).pack(side=tk.RIGHT, padx=4)

        def _open_disable_vault_modal():
            if not hasattr(app, 'vault_manager') or not app.vault_manager.is_lock_enabled():
                messagebox.showinfo("Info", "Vault lock is already disabled.", parent=win)
                return
            dis_win = tk.Toplevel(win)
            dis_win.title("Disable Vault Lock")
            dis_win.geometry("400x200")
            dis_win.config(bg=THEME["bg_color"])
            dis_win.transient(win)
            dis_win.grab_set()

            tk.Label(dis_win, text="Enter Master Password to Decrypt All Histories:", 
                     bg=THEME["bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_button"]).pack(padx=12, pady=10)
            pwd_ent = tk.Entry(dis_win, show="*", width=24, bg=THEME["widget_bg_color"], fg=THEME["fg_color"],
                               insertbackground=THEME.get("electric_blue", THEME["fg_color"]))
            bind_entry_limit(pwd_ent, max_len=64)
            pwd_ent.pack(padx=12, pady=6)
            pwd_ent.focus_set()

            def _do_disable():
                pwd = pwd_ent.get().strip()
                if not pwd: return
                success, msg = app.vault_manager.disable_lock(pwd)
                if success:
                    messagebox.showinfo("Vault Disabled", msg, parent=dis_win)
                    if "refresh_vault_status" in vars_dict: vars_dict["refresh_vault_status"]()
                    dis_win.destroy()
                else:
                    messagebox.showerror("Verification Failed", msg, parent=dis_win)

            tk.Button(dis_win, text="Decrypt & Disable", command=_do_disable,
                      bg="#660000", fg="white", font=app.fonts["ui_button"]).pack(pady=12)

        vars_dict["_open_set_password_modal"] = _open_set_password_modal
        vars_dict["_open_disable_vault_modal"] = _open_disable_vault_modal

        # --- 5. Construct All 6 Tabs ---
        tab_models = build_models_tab(scrollable_frame, app, win, vars_dict, generation_sensitive_widgets)
        tab_inference = build_inference_tab(scrollable_frame, app, win, vars_dict)
        tab_agents = build_agents_tab(scrollable_frame, app, win, vars_dict)
        tab_additional = build_additional_tab(scrollable_frame, app, win, vars_dict)
        tab_users = build_users_tab(scrollable_frame, app, win, vars_dict)
        tab_personalize = build_personalize_tab(scrollable_frame, app, win, vars_dict)

        tab_frames = {
            "models": tab_models,
            "inference": tab_inference,
            "agents": tab_agents,
            "additional": tab_additional,
            "users": tab_users,
            "personalize": tab_personalize
        }
        tab_buttons = {}

        def _switch_tab(tab_key):
            for k, f in tab_frames.items():
                if k == tab_key:
                    f.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
                else:
                    f.pack_forget()
            for k, b in tab_buttons.items():
                if k == tab_key:
                    b.config(bg=THEME["button_active_color"], fg=THEME.get("accent_highlight", "#00ffcc"))
                else:
                    b.config(bg=THEME["button_bg_color"], fg=THEME["fg_color"])
            canvas.yview_moveto(0)
            canvas.update_idletasks()
            canvas.configure(scrollregion=canvas.bbox("all"))

        # Build Navigation Buttons
        tab_defs = [
            ("models", "Models & Params"),
            ("inference", "Inference"),
            ("agents", "Agents"),
            ("additional", "Additional Settings"),
            ("users", "Users & Security"),
            ("personalize", "Personalize")
        ]
        for t_key, t_title in tab_defs:
            btn = tk.Button(tab_nav_frame, text=t_title, command=lambda k=t_key: _switch_tab(k),
                            bg=THEME["button_bg_color"], fg=THEME["fg_color"],
                            font=app.fonts["ui_button"], relief=tk.FLAT, padx=8, pady=2)
            btn.pack(side=tk.LEFT, padx=2)
            tab_buttons[t_key] = btn

        _switch_tab("models")

        # Apply generation guard if active
        if is_generating:
            for w in generation_sensitive_widgets:
                try: w.config(state="disabled")
                except: pass

        # --- 6. Save and Apply Logic ---
        def _apply_settings(close_window=False):
            app.config["media_rendering"] = vars_dict["media_var"].get()
            app.state["deep_cook_behavior"] = vars_dict["v_behavior"].get()
            if app.state["deep_cook_behavior"] == "oneshot":
                app.state["deep_cook"] = False
            app._sync_deep_cook_ui()

            app.config["k_cache_type"] = vars_dict["k_cache_var"].get()
            app.config["v_cache_type"] = vars_dict["v_cache_var"].get()
            app.config["history_lookup_mode"] = vars_dict["history_lookup_var"].get()
            app.config["history_usage"] = vars_dict["history_usage_var"].get()
            app.config["history_mode"] = vars_dict["history_mode_var"].get()
            hm = vars_dict["history_mode_var"].get().lower()
            app.config["turbovec_mode"] = "on" if hm == "turbovec" else ("off" if hm == "off" else "fallback")
            app.config["dynamic_params_enabled"] = vars_dict["dynamic_params_var"].get()
            app.config["ghost_mode"] = vars_dict["ghost_var"].get()

            if hasattr(app, 'ghost_button') and app.ghost_button:
                app.ghost_button.config(text=app._get_ghost_mode_label(), fg=app._get_ghost_mode_color())
            if hasattr(app, 'history_usage_button') and app.history_usage_button:
                app.history_usage_button.config(text=app._get_history_usage_label(), fg=app._get_history_usage_color())

            try:
                for pth in [os.path.join(app.script_dir, "Live", "System", "params.json"),
                            os.path.join("Live", "System", "params.json")]:
                    if os.path.exists(pth):
                        with open(pth, "r") as f: p_data = json.load(f)
                        p_data["k_cache_type"] = vars_dict["k_cache_var"].get()
                        p_data["v_cache_type"] = vars_dict["v_cache_var"].get()
                        with open(pth, "w") as f: json.dump(p_data, f, indent=4)
            except Exception as pe:
                print(f"[UI] Warning: Could not write KV cache types to Live params: {pe}")

            app.config["hao_preset"] = vars_dict["hao_var"].get()
            app.config["swa_kv_cache"] = vars_dict["swa_var"].get()
            app.config["auto_vram_offload"] = vars_dict["auto_vram_var"].get()

            old_spec = app.config.get("speculative_drafting", False)
            app.config["speculative_drafting"] = vars_dict["spec_draft_var"].get()
            draft_toggled = (old_spec != vars_dict["spec_draft_var"].get())

            app.config["thinking_checkbox"] = vars_dict["thinking_var"].get()
            app.config["benchmark_enabled"] = vars_dict["benchmark_var"].get()
            app.config["inline_markdown"] = vars_dict["inline_md_var"].get()
            if "format_prompt_md_var" in vars_dict:
                app.config["format_prompt_markdown"] = vars_dict["format_prompt_md_var"].get()
            app.config["overfill_behavior_mode"] = vars_dict["overfill_behavior_var"].get()
            app.config["budget_recovery_mode"] = vars_dict["overfill_behavior_var"].get()
            app.config["halt_behavior_mode"] = vars_dict["halt_behavior_var"].get()
            app.config["monitor_graph_mode"] = vars_dict["monitor_graph_var"].get()
            app.config["show_tooltips"] = vars_dict["show_tooltips_var"].get()
            app.state["streaming_mode"] = vars_dict["stream_var"].get()
            app.config["max_token_ratio"] = vars_dict["ratio_var"].get()
            app.config["multimedia_handling"] = vars_dict["multimedia_handling_var"].get()
            app.config["image_handling"] = vars_dict["multimedia_handling_var"].get()
            app.config["reasoning_strength"] = vars_dict["reasoning_var"].get()
            app.config["muse_reasoning_strength"] = vars_dict["reasoning_var"].get()
            app.config["response_length"] = vars_dict["resp_len_var"].get()
            app.config["dmn_enabled"] = vars_dict["dmn_enabled_var"].get()

            dmn_val = vars_dict["dmn_ent"].get().strip() if "dmn_ent" in vars_dict else "05:00"
            app.config["dmn_timeout"] = dmn_val

            app.config["stt_device_index"] = dev_id_map.get(vars_dict["stt_dev_var"].get(), None)
            app.config["stt_language"] = vars_dict["stt_lang_var"].get()

            if hasattr(app, 'vault_manager'):
                try: app.vault_manager.set_auto_lock_seconds(int(vars_dict["auto_lock_var"].get()))
                except: pass

            if "vram_ent" in vars_dict:
                try: app.state["virtual_vram"] = int(float(vars_dict["vram_ent"].get()) * 1024)
                except: pass

            for t, e in ents.items():
                try: app.gpu_layer_config[t] = int(e.get())
                except: pass
            for t, e in ctx_ents.items():
                try: app.context_size_config[t] = int(e.get())
                except: pass
            for t, e in n_batch_ents.items():
                try: app.n_batch_config[t] = int(e.get())
                except: pass
            for t, e in temp_ents.items():
                try: app.temp_config[t] = float(e.get())
                except: pass
            for t, e in top_p_ents.items():
                try: app.top_p_config[t] = float(e.get())
                except: pass
            for t, e in min_p_ents.items():
                try: app.min_p_config[t] = float(e.get())
                except: pass
            for t, e in top_k_ents.items():
                try: app.top_k_config[t] = int(float(e.get()))
                except: pass
            for t, e in rep_ents.items():
                try: app.repeat_penalty_config[t] = float(e.get())
                except: pass
            for t, e in freq_ents.items():
                try: app.frequency_penalty_config[t] = float(e.get())
                except: pass

            app.config["repeat_detection_mode"] = vars_dict["repeat_mode_var"].get()
            app.config["offline_mode"] = vars_dict["offline_mode_var"].get()
            try:
                from System.network_guard import set_offline_mode
                set_offline_mode(vars_dict["offline_mode_var"].get())
            except Exception as e:
                print(f"[SETTINGS] Failed to set offline guard: {e}")

            theme_k = THEME_MAP.get(vars_dict["theme_display_var"].get(), "apex")
            tex_k = TEXTURE_MAP.get(vars_dict["tex_display_var"].get(), "default")
            d_mode = vars_dict["dark_mode_var"].get()
            tex_int = float(vars_dict["tex_int_var"].get()) / 100.0

            app.config["theme"] = theme_k
            app.config["texture_style"] = tex_k
            app.config["texture_intensity"] = tex_int
            app.config["dark_mode"] = d_mode

            new_un = vars_dict["username_var"].get().strip() or "Default"
            if new_un != app.config.get("username", "Default") and hasattr(app, "switch_user"):
                app.switch_user(new_un)
            else:
                app.config["username"] = new_un

            app.config["user_preferred_name"] = vars_dict["user_pref_name_var"].get().strip()
            app.config["user_address_style"] = vars_dict["user_addr_style_var"].get().strip()

            app.config["delegation_enabled"] = vars_dict["delegation_enabled_var"].get()
            app.config["cecilia_delegation_mode"] = vars_dict["cecilia_mode_var"].get()
            app.config["subagent_selection_mode"] = vars_dict["subagent_density_var"].get()
            app.config["delegation_model_mode"] = vars_dict["delegation_model_mode_var"].get()
            app.config["delegation_chain_preset"] = vars_dict["chain_preset_var"].get()
            app.config["delegation_handoff_target"] = vars_dict["handoff_target_var"].get()

            app.config["status_bar_mode"] = vars_dict["status_mode_var"].get()
            app.config["status_bar_anim_style"] = vars_dict["anim_style_var"].get()
            app.config["status_bar_dmn_idle"] = vars_dict["sb_dmn_var"].get()
            app.config["status_bar_fallback_info"] = vars_dict["sb_fallback_var"].get()
            app.config["status_bar_linger_sec"] = float(vars_dict["sb_linger_var"].get())
            app.config["scroll_lock_enabled"] = vars_dict["scroll_lock_var"].get()
            app.config["user_preferred_name"] = vars_dict["user_pref_name_var"].get().strip()
            app.config["user_address_style"] = vars_dict["user_addr_style_var"].get().strip()
            try:
                geom = win.winfo_geometry()
                if geom and "x" in geom and not geom.startswith("1x1"):
                    app.config["settings_window_geometry"] = geom
            except Exception: pass

            target_scale = vars_dict["text_scale_val_var"].get()
            if hasattr(app, "apply_text_scale"):
                app.apply_text_scale(target_scale, persist=True)
            else:
                app.config["text_scale"] = target_scale

            target_ui_font = vars_dict["ui_font_var"].get()
            target_mono_font = vars_dict["mono_font_var"].get()
            if hasattr(app, "apply_font_family"):
                app.apply_font_family(target_ui_font, target_mono_font, persist=True)
            else:
                app.config["ui_font"] = target_ui_font
                app.config["mono_font"] = target_mono_font

            try:
                from serenity_resources import apply_theme_to_global
                apply_theme_to_global(theme_k, tex_k, d_mode, getattr(app, "active_persona_level", 3), (app.model is not None), tex_int)
                if hasattr(app, "apply_current_theme"):
                    app.apply_current_theme()
                if hasattr(app, "_update_hw_indicator"):
                    app._update_hw_indicator()

                if win.winfo_exists():
                    win.config(bg=THEME["bg_color"])
                    container.config(bg=THEME["bg_color"])
                    canvas.config(bg=THEME["bg_color"])
                    scrollable_frame.config(bg=THEME["bg_color"])
                    btn_frame.config(bg=THEME["bg_color"])
                    tab_nav_frame.config(bg=THEME["bg_color"])
                    _switch_tab(list(tab_frames.keys())[0])
            except Exception as te:
                print(f"[SETTINGS] Failed to apply theme: {te}")

            app.save_config()
            if draft_toggled and hasattr(app, "swap_tier") and hasattr(app, "current_model_tier"):
                app.swap_tier(app.current_model_tier)

            if close_window:
                messagebox.showinfo("Success", "Settings saved!", parent=win)
                win.destroy()
            else:
                messagebox.showinfo("Success", "Settings applied successfully!", parent=win)
                try: win.lift()
                except Exception: pass

        # Top Action Buttons
        save_close_btn = tk.Button(btn_frame, text="Save & Close", command=lambda: _apply_settings(True),
                                   bg=THEME["button_active_color"], fg=THEME["fg_color"], font=app.fonts["ui_button"], relief=tk.FLAT)
        save_close_btn.pack(side=tk.RIGHT, padx=3)
        ToolTip(save_close_btn, "Save all settings to configuration file and close settings window.", app=app)

        apply_btn = tk.Button(btn_frame, text="Apply", command=lambda: _apply_settings(False),
                              bg=THEME.get("button_bg_color", "#202020"), fg=THEME.get("accent_highlight", "#00ffcc"), font=app.fonts["ui_button"], relief=tk.FLAT)
        apply_btn.pack(side=tk.RIGHT, padx=3)
        ToolTip(apply_btn, "Apply current settings immediately without closing the settings window.", app=app)

        btn_clr_h = tk.Button(btn_frame, text="Clear History", command=app.clear_current_history,
                              bg="#660000", fg="white", font=app.fonts["ui_button"], relief=tk.FLAT)
        btn_clr_h.pack(side=tk.RIGHT, padx=3)
        ToolTip(btn_clr_h, "Permanently delete active conversation history and reset memory.", app=app)

        def _reset_defaults():
            if messagebox.askyesno("Reset", "Restore system defaults for all layers and samplers?", parent=win):
                recs = run_auto_detect(app, win)
                for t in recs:
                    if t in ents:
                        ents[t].delete(0, tk.END)
                        ents[t].insert(0, str(recs[t]))
                messagebox.showinfo("Reset", "System recommendations applied to visible fields. Click 'Apply' or 'Save & Close' to persist.", parent=win)
                try: win.lift()
                except Exception: pass

        btn_auto_d = tk.Button(btn_frame, text="Auto-Detect", command=_reset_defaults,
                               bg=THEME["button_bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_button"], relief=tk.FLAT)
        btn_auto_d.pack(side=tk.RIGHT, padx=3)
        ToolTip(btn_auto_d, "Benchmark system hardware and auto-calculate GPU layer offloads across tiers.", app=app)
        generation_sensitive_widgets.append(btn_auto_d)

        btn_cancel = tk.Button(btn_frame, text="Cancel", command=win.destroy,
                               bg=THEME["button_bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_button"], relief=tk.FLAT)
        btn_cancel.pack(side=tk.RIGHT, padx=3)
        ToolTip(btn_cancel, "Discard unapplied changes and close settings window.", app=app)

    except Exception as e:
        import traceback
        err_msg = f"Settings Window Crash: {e}\n{traceback.format_exc()}"
        print(err_msg)
        try:
            os.makedirs("Logs", exist_ok=True)
            with open("Logs/ui_crash.txt", "w") as f: f.write(err_msg)
        except: pass
        if hasattr(app, "root") and getattr(app.root, "winfo_exists", lambda: False)():
            try:
                if app.root.state() != "withdrawn":
                    messagebox.showerror("UI Error", f"Settings window failed to open:\n{e}")
            except: pass

def open_text_scaling_center(app, parent_win=None):
    """
    Opens the dedicated 'Text size & Scaling Center' window for fine-grained typography,
    category font scaling, window-responsiveness toggles, and real-time live preview.
    """
    try:
        from tkinter import scrolledtext
        parent = parent_win if (parent_win and parent_win.winfo_exists()) else (app.root if hasattr(app, 'root') else None)
        center_win = tk.Toplevel(parent)
        center_win.title("Text size & Scaling Center")
        center_win.geometry("740x740")
        center_win.minsize(580, 540)
        center_win.config(bg=THEME["bg_color"])
        if parent:
            center_win.transient(parent)
        
        # Center on screen/parent
        try:
            if parent:
                x = parent.winfo_x() + (parent.winfo_width() // 2) - 370
                y = parent.winfo_y() + (parent.winfo_height() // 2) - 370
                center_win.geometry(f"740x740+{max(0, x)}+{max(0, y)}")
        except Exception: pass

        # Header
        hdr_frame = tk.Frame(center_win, bg=THEME["widget_bg_color"], pady=8, padx=12)
        hdr_frame.pack(fill=tk.X)
        
        lbl_title = tk.Label(hdr_frame, text="🔍 Text size & Scaling Center", bg=THEME["widget_bg_color"],
                             fg=THEME.get("accent_highlight", "#00ffcc"), font=app.fonts["large"])
        lbl_title.pack(anchor="w")
        
        lbl_sub = tk.Label(hdr_frame, text="Configure global scaling, font families, per-category sizes, and test responsiveness with live preview.",
                           bg=THEME["widget_bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_small"])
        lbl_sub.pack(anchor="w", pady=(2, 0))

        # Main Container
        content_frame = tk.Frame(center_win, bg=THEME["bg_color"], padx=10, pady=6)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Upper Controls Frame
        ctrl_lf = tk.LabelFrame(content_frame, text="Global Scale & Font Configuration", bg=THEME["bg_color"],
                                fg=THEME["electric_blue"], font=app.fonts["bold"], padx=10, pady=6)
        ctrl_lf.pack(fill=tk.X, pady=(0, 6))

        # 1. Scale Slider + Value Display
        scale_row = tk.Frame(ctrl_lf, bg=THEME["bg_color"])
        scale_row.pack(fill=tk.X, pady=(2, 2))
        
        curr_scale = int(app.config.get("text_scale", 100)) if (hasattr(app, 'config') and app.config) else 100
        scale_var = tk.IntVar(value=curr_scale)
        
        tk.Label(scale_row, text="Global Scale Factor:", bg=THEME["bg_color"], fg=THEME["fg_color"],
                 font=app.fonts["ui_label"]).pack(side=tk.LEFT)
                 
        scale_val_lbl = tk.Label(scale_row, text=f"{curr_scale}%", bg=THEME["bg_color"],
                                 fg=THEME.get("accent_highlight", "#00ffcc"), font=app.fonts["bold"], width=6)
        scale_val_lbl.pack(side=tk.RIGHT)
        
        def _on_scale_slider_move(val):
            pct = int(float(val))
            scale_val_lbl.config(text=f"{pct}%")
            scale_var.set(pct)
            if hasattr(app, 'apply_text_scale'):
                app.apply_text_scale(pct, persist=False)
            _update_preview_tags()

        scale_slider = tk.Scale(ctrl_lf, from_=70, to=250, orient=tk.HORIZONTAL, variable=scale_var,
                                command=_on_scale_slider_move, showvalue=False, bg=THEME["widget_bg_color"],
                                fg=THEME["fg_color"], activebackground=THEME["electric_blue"],
                                highlightthickness=0, bd=0)
        scale_slider.pack(fill=tk.X, pady=(0, 4))

        # Preset Quick Buttons
        preset_row = tk.Frame(ctrl_lf, bg=THEME["bg_color"])
        preset_row.pack(fill=tk.X, pady=(0, 4))
        tk.Label(preset_row, text="Presets:", bg=THEME["bg_color"], fg="#888888", font=app.fonts["ui_small"]).pack(side=tk.LEFT, padx=(0, 4))
        
        def _apply_scale_preset(pct):
            scale_var.set(pct)
            scale_slider.set(pct)
            scale_val_lbl.config(text=f"{pct}%")
            if hasattr(app, 'apply_text_scale'):
                app.apply_text_scale(pct, persist=False)
            _update_preview_tags()

        for p_pct, p_name in [(85, "Compact (85%)"), (100, "100%"), (115, "115%"), (125, "125%"), (140, "140%"), (160, "160%"), (180, "180%"), (200, "200%")]:
            b = tk.Button(preset_row, text=p_name, command=lambda p=p_pct: _apply_scale_preset(p),
                          bg=THEME["widget_bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_small"],
                          relief=tk.FLAT, padx=3, pady=1)
            b.pack(side=tk.LEFT, padx=2)

        # Font Pickers Grid
        fonts_grid = tk.Frame(ctrl_lf, bg=THEME["bg_color"])
        fonts_grid.pack(fill=tk.X, pady=4)
        fonts_grid.grid_columnconfigure(1, weight=1)
        fonts_grid.grid_columnconfigure(3, weight=1)
        
        UI_FONT_OPTIONS = [
            "Segoe UI", "Verdana", "Tahoma", "Cambria",
            "Comic Sans MS", "Gothic A1", "Modiableic",
            "Comfortaa", "YU Gothic UI", "Segoe UI Variable",
            "Modern Antiqua", "Quicksand", "UnifrakturMaguntia",
        ]
        MONO_FONT_OPTIONS = [
            "Doto", "Inconsolata", "Noto Sans Mono", "Lucida Sans Console",
            "Palatino Linotype", "Tajawal", "MV Boli", "Didact Gothic",
        ]

        curr_ui_font = app.config.get("ui_font", "Segoe UI") if (hasattr(app, 'config') and app.config) else "Segoe UI"
        curr_mono_font = app.config.get("mono_font", "Consolas") if (hasattr(app, 'config') and app.config) else "Consolas"
        ui_font_var = tk.StringVar(value=curr_ui_font if curr_ui_font in UI_FONT_OPTIONS else "Segoe UI")
        mono_font_var = tk.StringVar(value=curr_mono_font if curr_mono_font in MONO_FONT_OPTIONS else MONO_FONT_OPTIONS[0])

        tk.Label(fonts_grid, text="UI Font:", bg=THEME["bg_color"], fg=THEME["fg_color"],
                 font=app.fonts["ui_label"]).grid(row=0, column=0, sticky="w", padx=(0, 4))
        ui_font_combo = ttk.Combobox(fonts_grid, textvariable=ui_font_var, values=UI_FONT_OPTIONS, state="readonly", width=14)
        ui_font_combo.grid(row=0, column=1, sticky="ew", padx=(0, 10))

        tk.Label(fonts_grid, text="Code/Log Font:", bg=THEME["bg_color"], fg=THEME["fg_color"],
                 font=app.fonts["ui_label"]).grid(row=0, column=2, sticky="w", padx=(0, 4))
        mono_font_combo = ttk.Combobox(fonts_grid, textvariable=mono_font_var, values=MONO_FONT_OPTIONS, state="readonly", width=14)
        mono_font_combo.grid(row=0, column=3, sticky="ew")

        def _on_font_select(*args):
            if hasattr(app, "apply_font_family"):
                app.apply_font_family(ui_font_var.get(), mono_font_var.get(), persist=False)
            _update_preview_tags()

        ui_font_combo.bind("<<ComboboxSelected>>", _on_font_select)
        mono_font_combo.bind("<<ComboboxSelected>>", _on_font_select)

        # Window Responsiveness Toggle
        opts_row = tk.Frame(ctrl_lf, bg=THEME["bg_color"])
        opts_row.pack(fill=tk.X, pady=(4, 2))
        
        resp_var = tk.BooleanVar(value=app.config.get("responsive_font_scaling", True) if (hasattr(app, 'config') and app.config) else True)
        
        def _on_resp_toggle():
            if hasattr(app, 'config') and app.config:
                app.config["responsive_font_scaling"] = resp_var.get()
            if hasattr(app, 'apply_text_scale'):
                app.apply_text_scale(scale_var.get(), persist=False)
            _update_preview_tags()

        cb_resp = tk.Checkbutton(opts_row, text="Window-Responsive Auto-Scale (enlarges text with window width)",
                                 variable=resp_var, command=_on_resp_toggle, bg=THEME["bg_color"],
                                 fg=THEME["fg_color"], selectcolor=THEME["widget_bg_color"],
                                 activebackground=THEME["bg_color"], activeforeground=THEME["electric_blue"],
                                 font=app.fonts["ui_label"])
        cb_resp.pack(side=tk.LEFT)

        # Category Fine-Tuning Frame
        cat_lf = tk.LabelFrame(content_frame, text="Per-Category Font Size Fine-Tuning (Offset)", bg=THEME["bg_color"],
                               fg=THEME["electric_blue"], font=app.fonts["bold"], padx=8, pady=4)
        cat_lf.pack(fill=tk.X, pady=(0, 6))

        saved_offsets = app.config.get("font_size_offsets", {}) if (hasattr(app, 'config') and app.config) else {}
        cat_vars = {
            "chat": tk.IntVar(value=saved_offsets.get("chat", 0)),
            "headers": tk.IntVar(value=saved_offsets.get("headers", 0)),
            "code_log": tk.IntVar(value=saved_offsets.get("code_log", 0)),
            "stats": tk.IntVar(value=saved_offsets.get("stats", 0)),
            "ui": tk.IntVar(value=saved_offsets.get("ui", 0)),
        }
        cat_labels = [
            ("chat", "Chat Body"),
            ("headers", "Markdown Headers"),
            ("code_log", "Code & Logs"),
            ("stats", "Telemetry & Stats"),
            ("ui", "UI Buttons & Labels")
        ]
        
        cat_grid = tk.Frame(cat_lf, bg=THEME["bg_color"])
        cat_grid.pack(fill=tk.X)
        
        def _on_cat_offset_change(*args):
            new_offsets = {k: var.get() for k, var in cat_vars.items()}
            if hasattr(app, 'config') and app.config:
                app.config["font_size_offsets"] = new_offsets
            if hasattr(app, 'apply_text_scale'):
                app.apply_text_scale(scale_var.get(), persist=False)
            _update_preview_tags()

        for idx, (cat_key, cat_name) in enumerate(cat_labels):
            col = idx % 3
            row = (idx // 3) * 2
            f = tk.Frame(cat_grid, bg=THEME["bg_color"])
            f.grid(row=row//2, column=col, sticky="ew", padx=6, pady=2)
            cat_grid.grid_columnconfigure(col, weight=1)
            
            lbl_f = tk.Frame(f, bg=THEME["bg_color"])
            lbl_f.pack(fill=tk.X)
            tk.Label(lbl_f, text=cat_name, bg=THEME["bg_color"], fg=THEME["fg_color"],
                     font=app.fonts["ui_small"]).pack(side=tk.LEFT)
            v_lbl = tk.Label(lbl_f, text=f"{cat_vars[cat_key].get():+d}pt", bg=THEME["bg_color"],
                             fg=THEME.get("accent_highlight", "#00ffcc"), font=app.fonts["stats"])
            v_lbl.pack(side=tk.RIGHT)
            
            def _make_cat_cmd(ck=cat_key, vl=v_lbl):
                def _cmd(val):
                    iv = int(float(val))
                    vl.config(text=f"{iv:+d}pt")
                    _on_cat_offset_change()
                return _cmd
                
            s = tk.Scale(f, from_=-4, to=8, orient=tk.HORIZONTAL, variable=cat_vars[cat_key],
                         command=_make_cat_cmd(cat_key, v_lbl), showvalue=False, bg=THEME["widget_bg_color"],
                         fg=THEME["fg_color"], activebackground=THEME["electric_blue"],
                         highlightthickness=0, bd=0)
            s.pack(fill=tk.X)

        # Live Preview Box
        prev_lf = tk.LabelFrame(content_frame, text="Live Text Preview", bg=THEME["bg_color"],
                                fg=THEME["electric_blue"], font=app.fonts["bold"], padx=6, pady=4)
        prev_lf.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        prev_text = scrolledtext.ScrolledText(prev_lf, bg=THEME["chat_bg_color"], fg=THEME["chat_fg_color"],
                                              wrap=tk.WORD, height=6, relief=tk.FLAT, font=app.fonts["main"])
        prev_text.pack(fill=tk.BOTH, expand=True)

        def _update_preview_tags():
            if not prev_text.winfo_exists(): return
            prev_text.tag_config("user_lead", font=app.fonts["bold"], foreground="#87CEFA")
            prev_text.tag_config("user", font=app.fonts["italic"], foreground="#007acc")
            prev_text.tag_config("ai_lead", font=app.fonts["bold"], foreground="#FFD700")
            prev_text.tag_config("md_header_1", font=app.fonts["md_header_1"], foreground="#00ffcc")
            prev_text.tag_config("md_bold", font=app.fonts["md_bold"])
            prev_text.tag_config("md_thought", font=app.fonts["md_thought"], foreground="#808080")
            prev_text.tag_config("md_code", font=app.fonts["md_code"], foreground="#E06C75", background="#1e1e1e")
            prev_text.tag_config("stats", font=app.fonts["stats"], foreground="#00ffcc")

        def _fill_preview_content():
            prev_text.config(state='normal')
            prev_text.delete('1.0', tk.END)
            prev_text.insert(tk.END, "You: ", ("user_lead",))
            prev_text.insert(tk.END, "Can you show me a sample code and status readout?\n\n", ("user",))
            prev_text.insert(tk.END, "Serenity: ", ("ai_lead",))
            prev_text.insert(tk.END, "# SerenityPC System Ready\n", ("md_header_1",))
            prev_text.insert(tk.END, "Thinking: Scanning active context and evaluating optimal tensor layers...\n", ("md_thought",))
            prev_text.insert(tk.END, "Typography scaling is active across all widgets and markdown tags.\n", ("md_bold",))
            prev_text.insert(tk.END, "def run_inference():\n    return 'Optimal speed: 42.5 t/s'\n", ("md_code",))
            prev_text.insert(tk.END, "\n[Stats] GPU: 48°C | VRAM: 3.8/6.0 GB | Speed: 42.5 t/s | Mode: APEX\n", ("stats",))
            prev_text.config(state='disabled')
            _update_preview_tags()

        _fill_preview_content()

        # Footer Action Buttons
        btn_bar = tk.Frame(center_win, bg=THEME["widget_bg_color"], pady=6, padx=10)
        btn_bar.pack(fill=tk.X, side=tk.BOTTOM)

        def _save_and_close():
            target_scale = scale_var.get()
            target_ui = ui_font_var.get()
            target_mono = mono_font_var.get()
            new_offsets = {k: var.get() for k, var in cat_vars.items()}
            if hasattr(app, 'config') and app.config:
                app.config["text_scale"] = target_scale
                app.config["ui_font"] = target_ui
                app.config["mono_font"] = target_mono
                app.config["responsive_font_scaling"] = resp_var.get()
                app.config["font_size_offsets"] = new_offsets
                if hasattr(app, 'save_config'):
                    app.save_config()
            if hasattr(app, 'apply_font_family'):
                app.apply_font_family(target_ui, target_mono, persist=True)
            if hasattr(app, 'apply_text_scale'):
                app.apply_text_scale(target_scale, persist=True)
            center_win.destroy()

        def _reset_all():
            scale_var.set(100)
            scale_slider.set(100)
            scale_val_lbl.config(text="100%")
            ui_font_var.set("Segoe UI")
            mono_font_var.set("Consolas")
            resp_var.set(True)
            for k in cat_vars:
                cat_vars[k].set(0)
            if hasattr(app, 'config') and app.config:
                app.config["font_size_offsets"] = {k: 0 for k in cat_vars}
                app.config["responsive_font_scaling"] = True
            if hasattr(app, 'apply_font_family'):
                app.apply_font_family("Segoe UI", "Consolas", persist=False)
            if hasattr(app, 'apply_text_scale'):
                app.apply_text_scale(100, persist=False)
            _on_cat_offset_change()

        tk.Button(btn_bar, text="Save & Close", command=_save_and_close,
                  bg=THEME["button_active_color"], fg=THEME["fg_color"],
                  font=app.fonts["ui_button"], relief=tk.FLAT).pack(side=tk.RIGHT, padx=4)

        tk.Button(btn_bar, text="Reset Defaults", command=_reset_all,
                  bg=THEME["button_bg_color"], fg=THEME["fg_color"],
                  font=app.fonts["ui_button"], relief=tk.FLAT).pack(side=tk.LEFT, padx=4)

        tk.Button(btn_bar, text="Cancel", command=center_win.destroy,
                  bg=THEME["button_bg_color"], fg=THEME["fg_color"],
                  font=app.fonts["ui_button"], relief=tk.FLAT).pack(side=tk.RIGHT, padx=4)

    except Exception as e:
        import traceback
        err_msg = f"Scaling Center Crash: {e}\n{traceback.format_exc()}"
        print(err_msg)
        messagebox.showerror("Scaling Center Error", f"Failed to open Text size & Scaling Center:\n{e}")
