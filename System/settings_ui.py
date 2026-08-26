import os
import json
import sys
import tkinter as tk
from tkinter import messagebox, filedialog, ttk, simpledialog
from serenity_resources import THEME
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


def open_settings_window(app):
    try:
        win = tk.Toplevel(app.root)
        win.title("Model Settings")
        if app.icon_path:
            try: win.iconbitmap(app.icon_path)
            except: pass
        win.geometry(app.config.get("settings_window_geometry", "800x950"))
        win.config(bg=THEME["bg_color"])
        win.transient(app.root)
        win.attributes("-topmost", False)
        
        # --- Fixed Top Action Bar ---
        btn_frame = tk.Frame(win, bg=THEME["bg_color"], pady=5)
        btn_frame.pack(side=tk.TOP, fill=tk.X, padx=10)
        
        tut_btn = tk.Button(btn_frame, text="🚀 Tutorial Walkthrough", 
                            command=lambda: (win.destroy(), getattr(app, 'start_tutorial_walkthrough', lambda: None)()),
                            bg=THEME["widget_bg_color"], fg=THEME.get("accent_highlight", "#00ffcc"), font=app.fonts["ui_button"], relief=tk.FLAT)
        tut_btn.pack(side=tk.LEFT, padx=2)
        ToolTip(tut_btn, "Launch the interactive translucent tutorial walkthrough for Serenity PC.", app=app)
        
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
        
        def _bind_targeted_settings_scroll(target_widget):
            def _on_enter(e):
                canvas.bind_all("<MouseWheel>", _on_mousewheel)
                canvas.bind_all("<Button-4>", lambda evt: canvas.yview_scroll(-1, "units") if canvas.winfo_exists() else None)
                canvas.bind_all("<Button-5>", lambda evt: canvas.yview_scroll(1, "units") if canvas.winfo_exists() else None)
            def _on_leave(e):
                try:
                    x, y = win.winfo_pointerxy()
                    wx = win.winfo_rootx()
                    wy = win.winfo_rooty()
                    ww = win.winfo_width()
                    wh = win.winfo_height()
                    if wx <= x <= wx + ww and wy <= y <= wy + wh:
                        return
                except Exception:
                    pass
                canvas.unbind_all("<MouseWheel>")
                canvas.unbind_all("<Button-4>")
                canvas.unbind_all("<Button-5>")
            target_widget.bind("<Enter>", _on_enter, add="+")
            target_widget.bind("<Leave>", _on_leave, add="+")

        def _on_mousewheel(event):
            if canvas.winfo_exists():
                canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        _bind_targeted_settings_scroll(win)
        _bind_targeted_settings_scroll(canvas)
        _bind_targeted_settings_scroll(scrollable_frame)
        
        def on_closing():
            try:
                canvas.unbind_all("<MouseWheel>")
                canvas.unbind_all("<Button-4>")
                canvas.unbind_all("<Button-5>")
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
        
        lbl_dc = tk.Label(left_header, text="Deep Cook:", bg=THEME["bg_color"], fg=THEME["electric_blue"])
        lbl_dc.pack(anchor="w")
        ToolTip(lbl_dc, "Select whether Deep Cook cycles operate in one-shot or persistent toggle mode.", app=app)
        v_behavior = tk.StringVar(value=app.state.get("deep_cook_behavior", "oneshot"))
        behavior_frame = tk.Frame(left_header, bg=THEME["bg_color"])
        behavior_frame.pack(anchor="w", padx=10)
        for val, txt in [("oneshot", "One-Shot"), ("toggle", "Toggle Mode")]:
             rb = tk.Radiobutton(behavior_frame, text=txt, variable=v_behavior, value=val, bg=THEME["bg_color"], fg=THEME["fg_color"], 
                            selectcolor=THEME["widget_bg_color"])
             rb.pack(side=tk.LEFT, padx=5)
             ToolTip(rb, f"Set Deep Cook behavior to {txt}.", app=app)
        
        vram_frame = tk.Frame(left_header, bg=THEME["bg_color"])
        vram_frame.pack(anchor="w", pady=(5, 0))
        lbl_vram = tk.Label(vram_frame, text="VRAM (GB):", bg=THEME["bg_color"], fg=THEME["electric_blue"])
        lbl_vram.pack(side=tk.LEFT)
        vram_ent = tk.Entry(vram_frame, bg=THEME["widget_bg_color"], fg=THEME["fg_color"], 
                            insertbackground=THEME.get("electric_blue", THEME["fg_color"]), width=6)
        bind_entry_limit(vram_ent, max_len=6)
        vram_ent.pack(side=tk.LEFT, padx=5)
        ToolTip(vram_ent, "Target VRAM threshold in Gigabytes for GPU layer calculation.", app=app)
        
        dmn_frame = tk.Frame(left_header, bg=THEME["bg_color"])
        dmn_frame.pack(anchor="w", pady=(5, 0))
        lbl_dmn = tk.Label(dmn_frame, text="DMN Timeout (min:sec):", bg=THEME["bg_color"], fg=THEME["electric_blue"])
        lbl_dmn.pack(side=tk.LEFT)
        ToolTip(lbl_dmn, "Default Mode Network idle countdown before Serenity begins background reflections.", app=app)
        dmn_ent = tk.Entry(dmn_frame, bg=THEME["widget_bg_color"], fg=THEME["fg_color"], 
                           insertbackground=THEME.get("electric_blue", THEME["fg_color"]), width=7)
        bind_entry_limit(dmn_ent, max_len=8)
        dmn_val = app.config.get("dmn_timeout", "05:00")
        dmn_ent.insert(0, str(dmn_val))
        dmn_ent.pack(side=tk.LEFT, padx=5)
        ToolTip(dmn_ent, "Idle duration (mm:ss) before triggering Default Mode Network simmer reflections.", app=app)
        


        if app._is_rgb_supported():
            toggle_frame = tk.Frame(left_header, bg=THEME["bg_color"])
            toggle_frame.pack(anchor="w", pady=(10, 0))
    
            show_rgb_var = tk.BooleanVar(value=app.config.get("show_rgb_button", True))
            def _toggle_rgb():
                app.config["show_rgb_button"] = show_rgb_var.get()
                if show_rgb_var.get():
                    if hasattr(app, 'rgb_button') and app.rgb_button.winfo_exists():
                        app.rgb_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, before=app.send_button)
                else:
                    if hasattr(app, 'rgb_button') and app.rgb_button.winfo_exists():
                        app.rgb_button.pack_forget()
                app.save_config()
    
            cb_rgb = tk.Checkbutton(toggle_frame, text="Show RGB Button", variable=show_rgb_var, command=_toggle_rgb,
                           bg=THEME["bg_color"], fg=THEME["electric_blue"], selectcolor=THEME["widget_bg_color"])
            cb_rgb.pack(anchor="w")
            ToolTip(cb_rgb, "Toggle visibility of RGB lighting controls button.", app=app)

        lbl_img = tk.Label(left_header, text="Multimedia Handling Mode:", bg=THEME["bg_color"], fg=THEME["electric_blue"])
        lbl_img.pack(anchor="w", pady=(5, 0))
        ToolTip(lbl_img, "Choose between automated handling, dedicated vision model, or native multimodal (images/audio/video).", app=app)
        multimedia_handling_var = tk.StringVar(value=app.config.get("multimedia_handling", app.config.get("image_handling", "auto")))
        multimedia_handling_frame = tk.Frame(left_header, bg=THEME["bg_color"])
        multimedia_handling_frame.pack(anchor="w", padx=10)
        for opt in ["auto", "vision", "native"]:
            rb = tk.Radiobutton(multimedia_handling_frame, text=opt.capitalize(), variable=multimedia_handling_var, value=opt,
                           bg=THEME["bg_color"], fg=THEME["fg_color"], selectcolor=THEME["widget_bg_color"])
            rb.pack(side=tk.LEFT, padx=5)
            ToolTip(rb, f"Use {opt} multimedia (images, audio, video) handling mode.", app=app)

        # Active Model Projector Mapping Selector
        proj_sel_frame = tk.Frame(left_header, bg=THEME["bg_color"])
        proj_sel_frame.pack(anchor="w", padx=10, pady=(2, 2))
        
        curr_m_path = app.model_path or ""
        curr_proj_cand = app.config.get("mmproj_mapping", {}).get(curr_m_path, "")
        if not curr_proj_cand and hasattr(app, "_find_projector_for_model"):
            curr_proj_cand = app._find_projector_for_model(curr_m_path, interactive=False) or ""
        
        lbl_proj_disp = tk.Label(proj_sel_frame, text=f"Projector: {os.path.basename(curr_proj_cand) if curr_proj_cand else 'None (Auto-Scanned)'}", 
                                 bg=THEME["bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_small"])
        
        def _pick_active_projector():
            from tkinter import filedialog
            chosen = filedialog.askopenfilename(
                title="Select Vision/Audio Projector (.mmproj / GGUF)",
                filetypes=[("Projector Files", "*.gguf;*.mmproj;*.bin"), ("All Files", "*.*")],
                parent=win
            )
            if chosen and os.path.exists(chosen):
                if app.model_path:
                    app.config.setdefault("mmproj_mapping", {})[app.model_path] = chosen
                    if hasattr(app, 'save_config'): app.save_config()
                lbl_proj_disp.config(text=f"Projector: {os.path.basename(chosen)}")

        btn_proj_pick = tk.Button(proj_sel_frame, text="Set Projector", command=_pick_active_projector, 
                                  bg=THEME["widget_bg_color"], fg=THEME["electric_blue"], font=app.fonts["ui_small"])
        btn_proj_pick.pack(side=tk.LEFT, padx=(0, 5))
        lbl_proj_disp.pack(side=tk.LEFT)
        ToolTip(btn_proj_pick, "Manually map a multimodal projector (.mmproj / GGUF) to the currently loaded model.", app=app)

        lbl_muse = tk.Label(left_header, text="Muse Reasoning:", bg=THEME["bg_color"], fg=THEME["electric_blue"])
        lbl_muse.pack(anchor="w", pady=(5, 0))
        ToolTip(lbl_muse, "Reasoning effort level for Muse-Glimmer thought cycles.", app=app)
        muse_reasoning_var = tk.StringVar(value=app.config.get("muse_reasoning_strength", "xhigh"))
        muse_reasoning_frame = tk.Frame(left_header, bg=THEME["bg_color"])
        muse_reasoning_frame.pack(anchor="w", padx=10)
        for opt in ["off", "low", "medium", "high", "xhigh"]:
            rb = tk.Radiobutton(muse_reasoning_frame, text=opt.capitalize(), variable=muse_reasoning_var, value=opt,
                           bg=THEME["bg_color"], fg=THEME["fg_color"], selectcolor=THEME["widget_bg_color"])
            rb.pack(side=tk.LEFT, padx=2)
            ToolTip(rb, f"Set Muse reasoning strength to {opt}.", app=app)

        # Checkboxes in left column
        offline_mode_var = tk.BooleanVar(value=app.config.get("offline_mode", False))
        auto_vram_var = tk.BooleanVar(value=app.config.get("auto_vram_offload", False))
        spec_draft_var = tk.BooleanVar(value=app.config.get("speculative_drafting", False))
        ghost_var = tk.BooleanVar(value=app.config.get("ghost_mode", False))
        thinking_var = tk.BooleanVar(value=app.config.get("thinking_checkbox", True))
        benchmark_var = tk.BooleanVar(value=app.config.get("benchmark_enabled", False))
        inline_md_var = tk.BooleanVar(value=app.config.get("inline_markdown", True))
        monitor_graph_var = tk.BooleanVar(value=app.config.get("monitor_graph_mode", False))
        show_tooltips_var = tk.BooleanVar(value=app.config.get("show_tooltips", True))

        auto_vram_f = tk.Frame(left_header, bg=THEME["bg_color"])
        auto_vram_f.pack(anchor="w", pady=(10, 0))
        cb_off = tk.Checkbutton(auto_vram_f, text="Fully Offline Mode (Block Net)", variable=offline_mode_var,
                       bg=THEME["bg_color"], fg="#ff8800", selectcolor=THEME["widget_bg_color"])
        cb_off.pack(anchor="w", pady=2)
        ToolTip(cb_off, "Blocks all outbound internet traffic while allowing local loopback.", app=app)

        cb_vram = tk.Checkbutton(auto_vram_f, text="Dynamic Auto-Offload", variable=auto_vram_var,
                       bg=THEME["bg_color"], fg=THEME["electric_blue"], selectcolor=THEME["widget_bg_color"])
        cb_vram.pack(anchor="w", pady=2)
        ToolTip(cb_vram, "Automatically flushes inactive model layers from VRAM to prevent memory exhaustion.", app=app)

        # TODO (Reference: TODO.txt Line 37): For the restructured settings window, ensure MTP Speculative Drafting enforces the 4GB minimum model size limit.
        cb_spec = tk.Checkbutton(auto_vram_f, text="Speculative MTP Drafting (>= 4GB Models)", variable=spec_draft_var,
                       bg=THEME["bg_color"], fg=THEME["electric_blue"], selectcolor=THEME["widget_bg_color"])
        cb_spec.pack(anchor="w", pady=2)
        ToolTip(cb_spec, "Accelerates token generation using assistant drafter speculative decoding (automatically bypassed for models < 4GB).", app=app)

        cb_ghost = tk.Checkbutton(auto_vram_f, text="Ghost Mode", variable=ghost_var,
                       bg=THEME["bg_color"], fg=THEME["electric_blue"], selectcolor=THEME["widget_bg_color"])
        cb_ghost.pack(anchor="w", pady=2)
        ToolTip(cb_ghost, "Disables chat history persistence to disk for private sessions.", app=app)

        cb_think = tk.Checkbutton(auto_vram_f, text="Thinking Process", variable=thinking_var,
                       bg=THEME["bg_color"], fg=THEME["electric_blue"], selectcolor=THEME["widget_bg_color"])
        cb_think.pack(anchor="w", pady=2)
        ToolTip(cb_think, "Controls whether internal model thought logs and reasoning blocks are captured.", app=app)

        cb_bench = tk.Checkbutton(auto_vram_f, text="Loading Benchmark", variable=benchmark_var,
                       bg=THEME["bg_color"], fg=THEME["electric_blue"], selectcolor=THEME["widget_bg_color"])
        cb_bench.pack(anchor="w", pady=2)
        ToolTip(cb_bench, "Runs a quick memory throughput benchmark upon model initialization.", app=app)

        cb_md = tk.Checkbutton(auto_vram_f, text="Inline Markdown", variable=inline_md_var,
                       bg=THEME["bg_color"], fg=THEME["electric_blue"], selectcolor=THEME["widget_bg_color"])
        cb_md.pack(anchor="w", pady=2)
        ToolTip(cb_md, "Enables real-time formatting for bold, italics, tables, and math equations.", app=app)

        cb_mon = tk.Checkbutton(auto_vram_f, text="Monitor Graph vs Line", variable=monitor_graph_var,
                       bg=THEME["bg_color"], fg=THEME["electric_blue"], selectcolor=THEME["widget_bg_color"])
        cb_mon.pack(anchor="w", pady=2)
        ToolTip(cb_mon, "Switches hardware telemetry display between graphs and text lines.", app=app)

        cb_tips = tk.Checkbutton(auto_vram_f, text="Enable Hover Tooltips / Help", variable=show_tooltips_var,
                       bg=THEME["bg_color"], fg=THEME["electric_blue"], selectcolor=THEME["widget_bg_color"])
        cb_tips.pack(anchor="w", pady=2)
        ToolTip(cb_tips, "Displays helpful linger-hover information boxes across UI controls.", app=app)

        lbl_templ = tk.Label(center_header, text="Templating Engine:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["bold"])
        lbl_templ.pack(anchor="n")
        ToolTip(lbl_templ, "32 instant slots to Save, Write, or Modify parameter presets across tiers.", app=app)
        template_mode = tk.StringVar(value="modify")
        active_template = tk.StringVar(value="")
        
        t_action_frame = tk.Frame(center_header, bg=THEME["bg_color"])
        t_action_frame.pack(anchor="n", pady=2)
        t_action_rbs = []
        for val, txt in [("save", "Save"), ("write", "Write"), ("modify", "Modify")]:
            rb_t = tk.Radiobutton(t_action_frame, text=txt, variable=template_mode, value=val, indicatoron=False, 
                           bg=THEME["widget_bg_color"], fg=THEME["fg_color"], selectcolor=THEME["electric_blue"],
                           activebackground=THEME["electric_blue"], activeforeground="#000000")
            rb_t.pack(side=tk.LEFT, padx=2)
            ToolTip(rb_t, f"Templating action mode: {txt} tier settings.", app=app)
            t_action_rbs.append((rb_t, val))

        def _update_t_action_colors(*args):
            cur = template_mode.get()
            for rb, v in t_action_rbs:
                if rb.winfo_exists():
                    rb.config(fg="#000000" if cur == v else THEME["fg_color"])
        template_mode.trace_add("write", _update_t_action_colors)
        _update_t_action_colors()

        t_grid = tk.Frame(center_header, bg=THEME["bg_color"])
        t_grid.pack(anchor="n", pady=5)
        
        template_buttons = []
        t_slot_rbs = []
        for i in range(8):
            for j in range(4):
                slot_id = f"T{(i*4)+j+1}"
                t_name = app.config.get("custom_templates", {}).get(slot_id, {}).get("name", slot_id)
                b = tk.Radiobutton(t_grid, text=t_name, variable=active_template, value=slot_id, indicatoron=False, width=12, 
                                   bg=THEME["widget_bg_color"], fg=THEME["electric_blue"], selectcolor=THEME["electric_blue"],
                                   activebackground=THEME["electric_blue"], activeforeground="#000000")
                b.grid(row=i, column=j, padx=2, pady=2)
                setattr(b, "slot_id", slot_id)
                ToolTip(b, f"Template Slot {slot_id} ({t_name}). Click to apply or modify.", app=app)
                template_buttons.append(b)
                t_slot_rbs.append((b, slot_id))

        def _update_template_slot_colors(*args):
            cur = active_template.get()
            for rb, s_id in t_slot_rbs:
                if rb.winfo_exists():
                    rb.config(fg="#000000" if cur == s_id else THEME["electric_blue"])
        active_template.trace_add("write", _update_template_slot_colors)
        _update_template_slot_colors()

        def _on_template_select(*args):
            mode = template_mode.get()
            t_id = active_template.get()
            if not t_id: return
            if mode == "modify":
                t_win = tk.Toplevel(win)
                t_win.title(f"Modify {t_id}")
                t_win.geometry("300x480")
                t_win.config(bg=THEME["bg_color"])
                t_win.transient(win)
                t_win.attributes("-topmost", False)
                current = app.config.get("custom_templates", {}).get(t_id, {})
                tk.Label(t_win, text="Name:", bg=THEME["bg_color"], fg=THEME["electric_blue"]).pack(anchor="w", padx=10, pady=(10,0))
                name_ent = tk.Entry(t_win, bg=THEME["widget_bg_color"], fg=THEME["fg_color"],
                                    insertbackground=THEME.get("electric_blue", THEME["fg_color"]))
                bind_entry_limit(name_ent, max_len=24)
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
                    e = tk.Entry(grid_f, bg=THEME["widget_bg_color"], fg=THEME["fg_color"], 
                                 insertbackground=THEME.get("electric_blue", THEME["fg_color"]), width=8)
                    bind_entry_limit(e, max_len=10)
                    e.insert(0, str(current.get(key, default)))
                    e.grid(row=r, column=c+1, padx=(0,10), pady=2)
                    fields[key] = e
                def _save_mod():
                    t_data: dict[str, object] = {"name": name_ent.get()}
                    for k, e in fields.items():
                        try: t_data[k] = float(e.get()) if '.' in e.get() else int(e.get())
                        except: t_data[k] = current.get(k, 0)
                    if "custom_templates" not in app.config: app.config["custom_templates"] = {}
                    app.config["custom_templates"][t_id] = t_data
                    for btn in template_buttons:
                        if btn.slot_id == t_id: btn.config(text=t_data["name"])
                    app.save_config()
                    t_win.destroy()
                    try: win.lift()
                    except Exception: pass
                tk.Button(t_win, text="Save & Close", command=_save_mod, bg=THEME["button_active_color"], fg=THEME["fg_color"]).pack(pady=15)
        active_template.trace_add("write", _on_template_select)

        right_header = tk.Frame(header_settings, bg=THEME["bg_color"])
        right_header.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(20, 0))
        lbl_sc = tk.Label(right_header, text="Video Processing Sub-Chunk Size:", bg=THEME["bg_color"], fg=THEME["electric_blue"])
        lbl_sc.pack(anchor="w")
        ToolTip(lbl_sc, "Sub-chunk frame size for processing video analysis.", app=app)
        sc_frame = tk.Frame(right_header, bg=THEME["bg_color"])
        sc_frame.pack(fill=tk.X, padx=5)
        sc_val = tk.IntVar(value=getattr(app, 'sub_chunk_size', 8))
        sc_scale = tk.Scale(sc_frame, from_=1, to=128, orient=tk.HORIZONTAL, variable=sc_val, 
                 bg=THEME["bg_color"], fg=THEME["fg_color"], highlightthickness=0, resolution=1)
        sc_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ToolTip(sc_scale, "Adjust frame batch size for multimodal video analysis.", app=app)
        
        def _reset_sc(): sc_val.set(8)
        btn_reset_sc = tk.Button(sc_frame, text="Reset", command=_reset_sc)
        btn_reset_sc.pack(side=tk.RIGHT, padx=5)
        ToolTip(btn_reset_sc, "Reset video sub-chunk size to default (8 frames).", app=app)

        labels, ents, ctx_ents, n_batch_ents, temp_ents, top_p_ents, min_p_ents, top_k_ents, rep_ents, freq_ents, pres_ents, stop_ents = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
        
        def _on_tier_box_click(tier_name):
            mode = template_mode.get()
            t_id = active_template.get()
            if not t_id: return
            if mode == "save":
                t_data = app.config.get("custom_templates", {}).get(t_id, {})
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
                if "custom_templates" not in app.config: app.config["custom_templates"] = {}
                app.config["custom_templates"][t_id] = t_data
                app.save_config()
                messagebox.showinfo("Templating", f"Saved {tier_name.upper()} settings to {t_data['name']}!", parent=win)
                try: win.lift()
                except Exception: pass
            elif mode == "write":
                t_data = app.config.get("custom_templates", {}).get(t_id, {})
                if not t_data: return
                for k, d in [("temp", temp_ents), ("top_p", top_p_ents), ("min_p", min_p_ents), ("rep", rep_ents), ("pres", pres_ents), ("freq", freq_ents), ("top_k", top_k_ents), 
                ("batch", n_batch_ents), ("layers", ents), ("ctx", ctx_ents), ("stop", stop_ents)]:
                    if k in t_data and tier_name in d: 
                        d[tier_name].delete(0, tk.END)
                        d[tier_name].insert(0, str(t_data[k]))
                messagebox.showinfo("Templating", f"Applied {t_data['name']} to {tier_name.upper()}!", parent=win)
                try: win.lift()
                except Exception: pass

        def _create_tier_block(parent, tier_name, row=0, col=0, is_vision=False):
            key = f"vision_{tier_name}" if is_vision else tier_name
            lvl_map = {"fast": "1", "search": "2", "low": "3", "med": "4", "high": "5", "transcendent": "6", "secret": "7"}
            title_suffix = f" (Lvl {lvl_map[tier_name]})" if tier_name in lvl_map else ""
            lf = tk.LabelFrame(parent, text=f"Engine: {tier_name.upper()}{title_suffix}", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["bold"], pady=5)
            lf.grid(row=row, column=col, sticky="nsew", padx=10, pady=5)
            
            def _bind_click(w):
                w.bind("<Button-1>", lambda e: _on_tier_box_click(key), add="+")
                for c in w.winfo_children(): _bind_click(c)
            
            r1 = tk.Frame(lf, bg=THEME["bg_color"]); r1.pack(fill=tk.X, padx=5)
            tk.Button(r1, text="Set Path", command=lambda t=key: app._set_path(t, labels, win)).pack(side=tk.LEFT)
            labels[key] = tk.Label(r1, text=os.path.basename(app.model_paths.get(key, "") or "Not Set"), bg=THEME["bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_small"])
            labels[key].pack(side=tk.LEFT, padx=5)
            
            r1b = tk.Frame(lf, bg=THEME["bg_color"]); r1b.pack(fill=tk.X, padx=5, pady=2)
            tk.Label(r1b, text="Layers:", bg=THEME["bg_color"], fg=THEME["fg_color"]).pack(side=tk.LEFT)
            ents[key] = tk.Entry(r1b, width=4, bg=THEME["widget_bg_color"], fg=THEME["fg_color"],
                                 insertbackground=THEME.get("electric_blue", THEME["fg_color"]))
            bind_entry_limit(ents[key], max_len=5)
            ents[key].insert(0, str(app.gpu_layer_config.get(key, -1)))
            ents[key].pack(side=tk.LEFT, padx=2)

            tk.Label(r1b, text="Ctx:", bg=THEME["bg_color"], fg=THEME["fg_color"]).pack(side=tk.LEFT, padx=(5, 0))
            ctx_ents[key] = tk.Entry(r1b, width=6, bg=THEME["widget_bg_color"], fg=THEME["fg_color"],
                                     insertbackground=THEME.get("electric_blue", THEME["fg_color"]))
            bind_entry_limit(ctx_ents[key], max_len=8)
            ctx_ents[key].insert(0, str(app.context_size_config.get(key, 4096)))
            ctx_ents[key].pack(side=tk.LEFT, padx=2)

            tk.Label(r1b, text="Batch:", bg=THEME["bg_color"], fg=THEME["fg_color"]).pack(side=tk.LEFT, padx=(5, 0))
            n_batch_ents[key] = tk.Entry(r1b, width=5, bg=THEME["widget_bg_color"], fg=THEME["fg_color"],
                                         insertbackground=THEME.get("electric_blue", THEME["fg_color"]))
            bind_entry_limit(n_batch_ents[key], max_len=6)
            n_batch_ents[key].insert(0, str(app.n_batch_config.get(key, 512)))
            n_batch_ents[key].pack(side=tk.LEFT, padx=2)

            r2 = tk.Frame(lf, bg=THEME["bg_color"]); r2.pack(fill=tk.X, padx=5)
            for l, d, c, df in [("Temp", temp_ents, app.temp_config, 0.8), ("Top-P", top_p_ents, app.top_p_config, 0.95), ("Min-P", min_p_ents, app.min_p_config, 0.05), ("Top-K", top_k_ents, app.top_k_config, 40)]:
                tk.Label(r2, text=f"{l}:", bg=THEME["bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_small"]).pack(side=tk.LEFT, padx=(2, 0))
                d[key] = tk.Entry(r2, width=5, bg=THEME["widget_bg_color"], fg=THEME["fg_color"],
                                  insertbackground=THEME.get("electric_blue", THEME["fg_color"]))
                bind_entry_limit(d[key], max_len=8)
                d[key].insert(0, f"{c.get(key, df):g}")
                d[key].pack(side=tk.LEFT, padx=2)
                
            r2b = tk.Frame(lf, bg=THEME["bg_color"]); r2b.pack(fill=tk.X, padx=5)
            for l, d, c, df in [("Rep", rep_ents, app.repeat_penalty_config, 1.1), ("Freq", freq_ents, app.frequency_penalty_config, 0.0), ("Pres", pres_ents, app.presence_penalty_config, 0.0)]:
                tk.Label(r2b, text=f"{l}:", bg=THEME["bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_small"]).pack(side=tk.LEFT, padx=(2, 0))
                d[key] = tk.Entry(r2b, width=5, bg=THEME["widget_bg_color"], fg=THEME["fg_color"],
                                   insertbackground=THEME.get("electric_blue", THEME["fg_color"]))
                bind_entry_limit(d[key], max_len=8)
                d[key].insert(0, f"{c.get(key, df):g}")
                d[key].pack(side=tk.LEFT, padx=2)

            if is_vision:
                pk = f"{key}_projector"
                r4 = tk.Frame(lf, bg=THEME["bg_color"]); r4.pack(fill=tk.X, padx=5, pady=2)
                tk.Button(r4, text="Projector", command=lambda k=pk: app._set_path(k, labels, win, True)).pack(side=tk.LEFT)
                labels[pk] = tk.Label(r4, text=os.path.basename(app.model_paths.get(pk, "") or "Not Set"), bg=THEME["bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_small"])
                labels[pk].pack(side=tk.LEFT, padx=5)
            _bind_click(lf)

        media_frame = tk.Frame(main, bg=THEME["bg_color"])
        media_frame.pack(fill=tk.X, pady=10)
        lbl_med = tk.Label(media_frame, text="Rich Media Rendering:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["bold"])
        lbl_med.pack(side=tk.LEFT, padx=10)
        ToolTip(lbl_med, "Choose how rich media (images, plots, markdown) is displayed: None, Inline, or in a separate Popup.", app=app)
        media_var = tk.IntVar(value=app.config.get("media_rendering", 1))
        for v, t in [(0, "None"), (1, "Inline"), (2, "Popup")]:
            rb_m = tk.Radiobutton(media_frame, text=t, variable=media_var, value=v, bg=THEME["bg_color"], fg=THEME["fg_color"], selectcolor=THEME["widget_bg_color"])
            rb_m.pack(side=tk.LEFT, padx=5)
            ToolTip(rb_m, f"Set media rendering mode to {t}.", app=app)

        over_lf = tk.LabelFrame(main, text="Global Engine & Memory Overrides", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["bold"], pady=5)
        over_lf.pack(fill=tk.X, padx=10, pady=5)
        
        over_subgrid = tk.Frame(over_lf, bg=THEME["bg_color"])
        over_subgrid.pack(fill=tk.X, padx=5, pady=5)
        over_subgrid.grid_columnconfigure(0, weight=1)
        over_subgrid.grid_columnconfigure(1, weight=1)

        left_over_col = tk.Frame(over_subgrid, bg=THEME["bg_color"])
        left_over_col.grid(row=0, column=0, sticky="nsew", padx=5)

        right_over_col = tk.Frame(over_subgrid, bg=THEME["bg_color"])
        right_over_col.grid(row=0, column=1, sticky="nsew", padx=5)

        # Left Sub-Column Controls
        dynamic_params_var = tk.BooleanVar(value=app.config.get("dynamic_params_enabled", True))
        cb_dyn = tk.Checkbutton(left_over_col, text="Dynamic Param Auto-Tune (Coding/Math/Creative)", variable=dynamic_params_var,
                       bg=THEME["bg_color"], fg=THEME["electric_blue"], selectcolor=THEME["widget_bg_color"],
                       activebackground=THEME["bg_color"], activeforeground=THEME["fg_color"], font=app.fonts["ui_label"])
        cb_dyn.pack(anchor="w", padx=5, pady=(2,0))
        ToolTip(cb_dyn, "Automatically optimizes sampling temperature and top-p when coding, math, or creative writing intent is detected.", app=app)

        def _bind_radio_contrast(var, rb_list):
            def _sync(*args):
                val = var.get()
                for rb, item_val in rb_list:
                    if rb.winfo_exists():
                        rb.config(fg="#000000" if val == item_val else THEME["fg_color"])
            var.trace_add("write", _sync)
            _sync()

        hao_var = tk.StringVar(value=app.config.get("hao_preset", "exps=CPU"))
        lbl_hao = tk.Label(left_over_col, text="HAO Preset:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["ui_label"])
        lbl_hao.pack(anchor="w", padx=5, pady=(5,0))
        ToolTip(lbl_hao, "Hardware Allocation Optimizer: configure MoE expert offloading strategies.", app=app)
        hao_f = tk.Frame(left_over_col, bg=THEME["bg_color"]); hao_f.pack(anchor="w", padx=10)
        hao_rbs = []
        for o in ["None", "exps=CPU"]:
            rb = tk.Radiobutton(hao_f, text=o, variable=hao_var, value=o, 
                           bg=THEME["widget_bg_color"], fg=THEME["fg_color"],
                           selectcolor=THEME["electric_blue"], indicatoron=False,
                           activebackground=THEME["electric_blue"], activeforeground="#000000",
                           width=10, font=app.fonts["ui_small"])
            rb.pack(side=tk.LEFT, padx=2)
            ToolTip(rb, f"Set HAO preset to {o}.", app=app)
            hao_rbs.append((rb, o))
        _bind_radio_contrast(hao_var, hao_rbs)

        swa_var = tk.StringVar(value=app.config.get("swa_kv_cache", "Auto"))
        lbl_swa = tk.Label(left_over_col, text="SWA Offload:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["ui_label"])
        lbl_swa.pack(anchor="w", padx=5, pady=(5,0))
        ToolTip(lbl_swa, "Sliding Window Attention KV cache offloading for models supporting SWA.", app=app)
        swa_f = tk.Frame(left_over_col, bg=THEME["bg_color"]); swa_f.pack(anchor="w", padx=10)
        swa_rbs = []
        for o in ["Auto", "CPU Only"]:
            rb = tk.Radiobutton(swa_f, text=o, variable=swa_var, value=o, 
                           bg=THEME["widget_bg_color"], fg=THEME["fg_color"],
                           selectcolor=THEME["electric_blue"], indicatoron=False,
                           activebackground=THEME["electric_blue"], activeforeground="#000000",
                           width=10, font=app.fonts["ui_small"])
            rb.pack(side=tk.LEFT, padx=2)
            ToolTip(rb, f"Set SWA offload mode to {o}.", app=app)
            swa_rbs.append((rb, o))
        _bind_radio_contrast(swa_var, swa_rbs)

        stream_var = tk.StringVar(value=app.state.get("streaming_mode", "Buffered"))
        lbl_stream = tk.Label(left_over_col, text="Streaming Behavior:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["ui_label"])
        lbl_stream.pack(anchor="w", padx=5, pady=(5,0))
        ToolTip(lbl_stream, "Select token streaming mode: Real-time, Buffered, Experimental Chunking, or Mass Dump.", app=app)
        stream_f = tk.Frame(left_over_col, bg=THEME["bg_color"]); stream_f.pack(anchor="w", padx=10)
        stream_rbs = []
        for o in ["Real-time", "Buffered", "Experimental Chunking", "Mass Dump"]:
            rb = tk.Radiobutton(stream_f, text=o, variable=stream_var, value=o, 
                           bg=THEME["widget_bg_color"], fg=THEME["fg_color"],
                           selectcolor=THEME["electric_blue"], indicatoron=False,
                           activebackground=THEME["electric_blue"], activeforeground="#000000",
                           width=0, font=app.fonts["ui_small"])
            rb.pack(side=tk.LEFT, padx=5, pady=2)
            ToolTip(rb, f"Use {o} response streaming strategy.", app=app)
            stream_rbs.append((rb, o))
        _bind_radio_contrast(stream_var, stream_rbs)

        ratio_var = tk.IntVar(value=app.config.get("max_token_ratio", 4))
        lbl_ratio = tk.Label(left_over_col, text="Response Headroom (ctx/N):", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["ui_label"])
        lbl_ratio.pack(anchor="w", padx=5, pady=(5,0))
        ToolTip(lbl_ratio, "Maximum token generation headroom ratio relative to context size.", app=app)
        ratio_f = tk.Frame(left_over_col, bg=THEME["bg_color"]); ratio_f.pack(anchor="w", padx=10)
        ratio_rbs = []
        for val, lbl in [(16, "U-Fast (16)"), (8, "Fast (8)"), (4, "Balanced (4)"), (2, "Deep (2)")]:
            rb = tk.Radiobutton(ratio_f, text=lbl, variable=ratio_var, value=val, 
                           bg=THEME["widget_bg_color"], fg=THEME["fg_color"],
                           selectcolor=THEME["electric_blue"], indicatoron=False,
                           activebackground=THEME["electric_blue"], activeforeground="#000000",
                           width=0, font=app.fonts["ui_small"])
            rb.pack(side=tk.LEFT, padx=5, pady=2)
            ToolTip(rb, f"Set response headroom to {lbl}.", app=app)
            ratio_rbs.append((rb, val))
        _bind_radio_contrast(ratio_var, ratio_rbs)

        repeat_mode_var = tk.StringVar(value=app.config.get("repeat_detection_mode", "lazy"))
        lbl_repeat = tk.Label(left_over_col, text="Repeat Loop Detection:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["ui_label"])
        lbl_repeat.pack(anchor="w", padx=5, pady=(5,0))
        ToolTip(lbl_repeat, "Configurable loop detector (Hyper, Lazy, Off) to prevent repetitive token generation stalls.", app=app)
        repeat_f = tk.Frame(left_over_col, bg=THEME["bg_color"]); repeat_f.pack(anchor="w", padx=10)
        repeat_rbs = []
        for o in ["hyper", "lazy", "off"]:
            rb = tk.Radiobutton(repeat_f, text=o.capitalize(), variable=repeat_mode_var, value=o, 
                           bg=THEME["widget_bg_color"], fg=THEME["fg_color"],
                           selectcolor=THEME["electric_blue"], indicatoron=False,
                           activebackground=THEME["electric_blue"], activeforeground="#000000",
                           width=8, font=app.fonts["ui_small"])
            rb.pack(side=tk.LEFT, padx=3, pady=2)
            ToolTip(rb, f"Set repeat loop detection to {o}.", app=app)
            repeat_rbs.append((rb, o))
        _bind_radio_contrast(repeat_mode_var, repeat_rbs)

        # Right Sub-Column Controls (4 Dropdowns)
        UNIVERSAL_KV_TYPES = ["fp16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl", "f32"]
        k_val = app.config.get("k_cache_type", "q8_0").lower()
        if k_val not in UNIVERSAL_KV_TYPES: k_val = "q8_0"
        v_val = app.config.get("v_cache_type", "q8_0").lower()
        if v_val not in UNIVERSAL_KV_TYPES: v_val = "q8_0"

        k_cache_var = tk.StringVar(value=k_val)
        v_cache_var = tk.StringVar(value=v_val)
        
        kv_frame = tk.Frame(right_over_col, bg=THEME["bg_color"])
        kv_frame.pack(anchor="w", padx=10, pady=5)
        
        lbl_k = tk.Label(kv_frame, text="K Cache Format:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["ui_label"])
        lbl_k.grid(row=0, column=0, sticky="w", pady=2)
        ToolTip(lbl_k, "Quantized Key cache format (Q8_0, Q4_0, FP16, etc.) for VRAM savings.", app=app)
        k_cache_dropdown = ttk.Combobox(kv_frame, textvariable=k_cache_var, values=UNIVERSAL_KV_TYPES, state="readonly", width=14)
        k_cache_dropdown.grid(row=0, column=1, padx=5, pady=2)
        ToolTip(k_cache_dropdown, "Select Key KV cache quantization type.", app=app)
        
        lbl_v = tk.Label(kv_frame, text="V Cache Format:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["ui_label"])
        lbl_v.grid(row=1, column=0, sticky="w", pady=2)
        ToolTip(lbl_v, "Quantized Value cache format (Q8_0, Q4_0, FP16, etc.) for VRAM savings.", app=app)
        v_cache_dropdown = ttk.Combobox(kv_frame, textvariable=v_cache_var, values=UNIVERSAL_KV_TYPES, state="readonly", width=14)
        v_cache_dropdown.grid(row=1, column=1, padx=5, pady=2)
        ToolTip(v_cache_dropdown, "Select Value KV cache quantization type.", app=app)

        lbl_hl = tk.Label(kv_frame, text="History Lookup Mode:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["ui_label"])
        lbl_hl.grid(row=2, column=0, sticky="w", pady=2)
        ToolTip(lbl_hl, "Scope of long-term history retrieval (targeted, model, level, all).", app=app)
        history_lookup_var = tk.StringVar(value=app.config.get("history_lookup_mode", "targeted"))
        history_lookup_dropdown = ttk.Combobox(kv_frame, textvariable=history_lookup_var, values=["targeted", "model", "level", "all"], state="readonly", width=14)
        history_lookup_dropdown.grid(row=2, column=1, padx=5, pady=2)
        ToolTip(history_lookup_dropdown, "Choose filter scope for past conversation retrieval.", app=app)

        lbl_hu = tk.Label(kv_frame, text="History Usage Mode:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["ui_label"])
        lbl_hu.grid(row=3, column=0, sticky="w", pady=2)
        ToolTip(lbl_hu, "Whether past conversation histories are injected into active context.", app=app)
        history_usage_var = tk.StringVar(value=app.config.get("history_usage", "all"))
        history_usage_dropdown = ttk.Combobox(kv_frame, textvariable=history_usage_var, values=["all", "current_window", "off"], state="readonly", width=14)
        history_usage_dropdown.grid(row=3, column=1, padx=5, pady=2)
        ToolTip(history_usage_dropdown, "Choose how much conversational history is loaded into memory.", app=app)

        lbl_br = tk.Label(kv_frame, text="Budget Recovery Mode:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["ui_label"])
        lbl_br.grid(row=4, column=0, sticky="w", pady=2)
        ToolTip(lbl_br, "Context budget overflow recovery strategy (wrapup, autocont, respond, off).", app=app)
        budget_recovery_var = tk.StringVar(value=app.config.get("budget_recovery_mode", "wrapup"))
        budget_recovery_dropdown = ttk.Combobox(kv_frame, textvariable=budget_recovery_var, values=["off", "respond", "wrapup", "autocont"], state="readonly", width=14)
        budget_recovery_dropdown.grid(row=4, column=1, padx=5, pady=2)
        ToolTip(budget_recovery_dropdown, "Strategy for wrapping up response when approaching context limit.", app=app)

        lbl_tv = tk.Label(kv_frame, text="TurboVec Mode:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["ui_label"])
        lbl_tv.grid(row=5, column=0, sticky="w", pady=2)
        ToolTip(lbl_tv, "Vector database embedding acceleration for long-term semantic history recall.", app=app)
        turbovec_mode_var = tk.StringVar(value=app.config.get("turbovec_mode", "fallback"))
        turbovec_mode_dropdown = ttk.Combobox(kv_frame, textvariable=turbovec_mode_var, values=["on", "fallback", "off"], state="readonly", width=14)
        turbovec_mode_dropdown.grid(row=5, column=1, padx=5, pady=2)
        ToolTip(turbovec_mode_dropdown, "Enable TurboVec vector index recall or fallback.", app=app)

        # --- THEME & TEXTURE OVERHAUL CONTROLS ---
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
        theme_display_var = tk.StringVar(value=THEME_REV_MAP.get(curr_theme_key, "Apex (Default)"))

        lbl_thm = tk.Label(kv_frame, text="Theme Palette:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["ui_label"])
        lbl_thm.grid(row=6, column=0, sticky="w", pady=2)
        ToolTip(lbl_thm, "Select visual color theme palette for Serenity PC.", app=app)
        theme_dropdown = ttk.Combobox(kv_frame, textvariable=theme_display_var, values=list(THEME_MAP.keys()), state="readonly", width=16)
        theme_dropdown.grid(row=6, column=1, padx=5, pady=2)
        ToolTip(theme_dropdown, "Switch active theme palette.", app=app)

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
        tex_display_var = tk.StringVar(value=TEXTURE_REV_MAP.get(curr_tex_key, "Default Original"))

        lbl_tex = tk.Label(kv_frame, text="Texture Style:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["ui_label"])
        lbl_tex.grid(row=7, column=0, sticky="w", pady=2)
        ToolTip(lbl_tex, "Select material texture style for interface elements.", app=app)
        tex_dropdown = ttk.Combobox(kv_frame, textvariable=tex_display_var, values=list(TEXTURE_MAP.keys()), state="readonly", width=16)
        tex_dropdown.grid(row=7, column=1, padx=5, pady=2)
        ToolTip(tex_dropdown, "Switch active texture finish style.", app=app)

        lbl_tex_int = tk.Label(kv_frame, text="Texture Intensity:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["ui_label"])
        lbl_tex_int.grid(row=8, column=0, sticky="w", pady=2)
        ToolTip(lbl_tex_int, "Adjust the depth and opacity weighting of active texture styling (0% - 100%).", app=app)
        curr_tex_int = int(float(app.config.get("texture_intensity", 1.0)) * 100)
        tex_int_var = tk.IntVar(value=curr_tex_int)
        tex_int_scale = tk.Scale(kv_frame, from_=0, to=100, orient=tk.HORIZONTAL, variable=tex_int_var, 
                                 bg=THEME["bg_color"], fg=THEME["fg_color"], highlightthickness=0, resolution=5, length=120)
        tex_int_scale.grid(row=8, column=1, padx=5, pady=2, sticky="ew")
        ToolTip(tex_int_scale, "Live texture intensity modifier.", app=app)

        dark_mode_var = tk.BooleanVar(value=app.config.get("dark_mode", False))
        cb_dark = tk.Checkbutton(kv_frame, text="Dark Mode (OLED Blackout)", variable=dark_mode_var,
                       bg=THEME["bg_color"], fg=THEME["electric_blue"], selectcolor=THEME["widget_bg_color"], font=app.fonts["ui_label"])
        cb_dark.grid(row=9, column=0, columnspan=2, sticky="w", pady=2)
        ToolTip(cb_dark, "Blacks out the interface to pure OLED #000000 to save power and maximize neon text contrast.", app=app)

        # STT Audio Input Settings
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

        stt_dev_var = tk.StringVar(value=curr_dev_label)
        lbl_stt = tk.Label(kv_frame, text="STT Mic Input:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["ui_label"])
        lbl_stt.grid(row=10, column=0, sticky="w", pady=2)
        ToolTip(lbl_stt, "Select local audio microphone device for Speech-To-Text dictation.", app=app)
        stt_dev_dropdown = ttk.Combobox(kv_frame, textvariable=stt_dev_var, values=dev_names, state="readonly", width=14)
        stt_dev_dropdown.grid(row=10, column=1, padx=5, pady=2)
        ToolTip(stt_dev_dropdown, "Select microphone input device for STT recording.", app=app)

        lbl_sttl = tk.Label(kv_frame, text="STT Language:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["ui_label"])
        lbl_sttl.grid(row=11, column=0, sticky="w", pady=2)
        ToolTip(lbl_sttl, "Recognition language code for offline Speech-To-Text.", app=app)
        stt_lang_var = tk.StringVar(value=app.config.get("stt_language", "en-US"))
        stt_lang_dropdown = ttk.Combobox(kv_frame, textvariable=stt_lang_var, values=["en-US", "en-GB", "es-ES", "fr-FR", "de-DE", "ja-JP", "zh-CN"], state="readonly", width=14)
        stt_lang_dropdown.grid(row=11, column=1, padx=5, pady=2)
        ToolTip(stt_lang_dropdown, "Select spoken language code for voice dictation.", app=app)

        # --- TEXT & FONT SCALING ---
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
        default_preset_lbl = SCALE_REV_MAP.get(curr_text_scale, f"{curr_text_scale}%")
        text_scale_display_var = tk.StringVar(value=default_preset_lbl)
        text_scale_val_var = tk.IntVar(value=curr_text_scale)

        lbl_scale = tk.Label(kv_frame, text="Text Size / Scale:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["ui_label"])
        lbl_scale.grid(row=12, column=0, sticky="w", pady=2)
        ToolTip(lbl_scale, "Adjust text and font size scaling (70% - 250%). Use Ctrl + / Ctrl - for quick zoom.", app=app)
        
        scale_dropdown = ttk.Combobox(kv_frame, textvariable=text_scale_display_var, values=list(SCALE_MAP.keys()), state="readonly", width=14)
        scale_dropdown.grid(row=12, column=1, padx=5, pady=2)
        ToolTip(scale_dropdown, "Select text size scale preset.", app=app)
        
        def _on_scale_dropdown_select(*args):
            sel = text_scale_display_var.get()
            if sel in SCALE_MAP:
                val = SCALE_MAP[sel]
                text_scale_val_var.set(val)
                if hasattr(app, 'apply_text_scale'):
                    app.apply_text_scale(val, persist=True)

        scale_dropdown.bind("<<ComboboxSelected>>", _on_scale_dropdown_select)

        # --- FONT FAMILY SELECTIONS ---
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

        curr_ui_font = app.config.get("ui_font", "Segoe UI")
        curr_mono_font = app.config.get("mono_font", "Consolas")
        ui_font_var = tk.StringVar(value=curr_ui_font if curr_ui_font in UI_FONT_OPTIONS else "Segoe UI")
        mono_font_var = tk.StringVar(value=curr_mono_font if curr_mono_font in MONO_FONT_OPTIONS else MONO_FONT_OPTIONS[0])

        lbl_ui_font = tk.Label(kv_frame, text="UI Font:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["ui_label"])
        lbl_ui_font.grid(row=13, column=0, sticky="w", pady=2)
        ToolTip(lbl_ui_font, "Select font family for chat, buttons, menus, and labels.", app=app)

        ui_font_dropdown = ttk.Combobox(kv_frame, textvariable=ui_font_var, values=UI_FONT_OPTIONS, state="readonly", width=14)
        ui_font_dropdown.grid(row=13, column=1, padx=5, pady=2)
        ToolTip(ui_font_dropdown, "Choose UI typography font family.", app=app)

        lbl_mono_font = tk.Label(kv_frame, text="Code / Log Font:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["ui_label"])
        lbl_mono_font.grid(row=14, column=0, sticky="w", pady=2)
        ToolTip(lbl_mono_font, "Select font family for code blocks, telemetry, and backend logs.", app=app)

        mono_font_dropdown = ttk.Combobox(kv_frame, textvariable=mono_font_var, values=MONO_FONT_OPTIONS, state="readonly", width=14)
        mono_font_dropdown.grid(row=14, column=1, padx=5, pady=2)
        ToolTip(mono_font_dropdown, "Choose monospace font family.", app=app)

        def _on_font_family_change(*args):
            u_fam = ui_font_var.get()
            m_fam = mono_font_var.get()
            if hasattr(app, "apply_font_family"):
                app.apply_font_family(u_fam, m_fam, persist=True)

        ui_font_dropdown.bind("<<ComboboxSelected>>", _on_font_family_change)
        mono_font_dropdown.bind("<<ComboboxSelected>>", _on_font_family_change)

        btn_scaling_center = tk.Button(kv_frame, text="🔍 Open Text size & Scaling Center",
                                       command=lambda: open_text_scaling_center(app, win),
                                       bg=THEME["widget_bg_color"], fg=THEME.get("accent_highlight", "#00ffcc"),
                                       font=app.fonts["ui_button"], relief=tk.FLAT)
        btn_scaling_center.grid(row=15, column=0, columnspan=2, pady=(6, 2), sticky="ew")
        ToolTip(btn_scaling_center, "Open comprehensive Text size & Scaling Center for live preview and custom category scaling.", app=app)

        # --- USER PROFILE & HISTORY SEPARATION ---
        user_section = tk.Frame(main, bg=THEME["bg_color"], highlightbackground=THEME["electric_blue"], highlightthickness=1, bd=0)
        user_section.pack(fill=tk.X, padx=10, pady=(15, 5))

        user_header = tk.Frame(user_section, bg=THEME["widget_bg_color"])
        user_header.pack(fill=tk.X, padx=0, pady=0)
        lbl_uhdr = tk.Label(user_header, text="👤 User Profiles", bg=THEME["widget_bg_color"], 
                 fg=THEME["electric_blue"], font=app.fonts["bold"])
        lbl_uhdr.pack(side=tk.LEFT, padx=8, pady=4)
        ToolTip(lbl_uhdr, "Manage user profiles to keep isolated conversation histories and DMN memory states.", app=app)

        user_body = tk.Frame(user_section, bg=THEME["bg_color"])
        user_body.pack(fill=tk.X, padx=10, pady=6)

        lbl_uact = tk.Label(user_body, text="Active Username:", bg=THEME["bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_label"])
        lbl_uact.pack(side=tk.LEFT, padx=(0, 5))
        ToolTip(lbl_uact, "Select or type a username to switch workspaces or create a new user profile.", app=app)
        
        user_profiles_list = app.list_user_profiles() if hasattr(app, 'list_user_profiles') else ["Default"]
        username_var = tk.StringVar(value=app.get_active_username() if hasattr(app, 'get_active_username') else app.config.get("username", "Default"))
        user_combo = ttk.Combobox(user_body, textvariable=username_var, values=user_profiles_list, width=16)
        user_combo.pack(side=tk.LEFT, padx=5)
        ToolTip(user_combo, "Select an existing profile or type a new name to create a workspace.", app=app)

        def _verify_vault_unlock(target_un):
            if target_un not in ("Default", "Public") and hasattr(app, 'vault_manager') and app.vault_manager and app.vault_manager.is_lock_enabled() and app.vault_manager.is_locked():
                pwd = simpledialog.askstring("Vault Authentication", f"Profile '{target_un}' is locked with Vault Security.\nEnter Master Password:", show="*", parent=win)
                if pwd is None:
                    return False
                if not app.vault_manager.unlock(pwd):
                    messagebox.showerror("Access Denied", "Incorrect master password for locked profile.", parent=win)
                    return False
            return True

        def _apply_switch_user():
            target_un = username_var.get().strip()
            if not target_un: return
            if not _verify_vault_unlock(target_un):
                return
            if hasattr(app, 'switch_user'):
                app.switch_user(target_un)
                user_profiles_list = app.list_user_profiles()
                user_combo['values'] = user_profiles_list
                messagebox.showinfo("User Profile Switched", f"Active user profile set to '{target_un}'.")

        btn_switch_u = tk.Button(user_body, text="Switch / Create Profile", command=_apply_switch_user,
                  bg=THEME["button_bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_button"], relief=tk.FLAT)
        btn_switch_u.pack(side=tk.LEFT, padx=8)
        ToolTip(btn_switch_u, "Switch active profile and load its separate history archive and DMN state.", app=app)

        user_identity_frame = tk.Frame(user_section, bg=THEME["bg_color"])
        user_identity_frame.pack(fill=tk.X, padx=10, pady=(0, 6))

        lbl_pref_name = tk.Label(user_identity_frame, text="Preferred Name / Call Sign:", bg=THEME["bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_label"])
        lbl_pref_name.pack(side=tk.LEFT, padx=(0, 5))
        ToolTip(lbl_pref_name, "Name that Serenity, Sage, and Cecilia will use when addressing you.", app=app)
        
        user_pref_name_var = tk.StringVar(value=app.config.get("user_preferred_name", ""))
        user_pref_name_entry = tk.Entry(user_identity_frame, textvariable=user_pref_name_var, 
                                        bg=THEME["widget_bg_color"], fg=THEME["fg_color"], 
                                        insertbackground=THEME.get("electric_blue", THEME["fg_color"]),
                                        font=app.fonts.get("ui_entry", app.fonts.get("ui_label", app.fonts.get("main"))), width=14)
        bind_entry_limit(user_pref_name_entry, max_len=32)
        user_pref_name_entry.pack(side=tk.LEFT, padx=(0, 15))
        ToolTip(user_pref_name_entry, "Set your custom name or call sign for this profile.", app=app)

        lbl_addr_style = tk.Label(user_identity_frame, text="Addressing Style:", bg=THEME["bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_label"])
        lbl_addr_style.pack(side=tk.LEFT, padx=(0, 5))
        ToolTip(lbl_addr_style, "Configure how personas refer to you in dialogue.", app=app)

        ADDRESS_STYLES = ["Direct / Plain", "Warm / Familiar", "Formal / Respectful", "Silent / Unnamed"]
        curr_addr_style = app.config.get("user_address_style", "Direct / Plain")
        user_addr_style_var = tk.StringVar(value=curr_addr_style if curr_addr_style in ADDRESS_STYLES else "Direct / Plain")
        addr_style_combo = ttk.Combobox(user_identity_frame, textvariable=user_addr_style_var, values=ADDRESS_STYLES, state="readonly", width=16)
        addr_style_combo.pack(side=tk.LEFT, padx=0)
        ToolTip(addr_style_combo, "Select how Serenity/Cecilia should address you.", app=app)

        user_opts_frame = tk.Frame(user_section, bg=THEME["bg_color"])
        user_opts_frame.pack(fill=tk.X, padx=10, pady=(0, 6))

        is_logged_in = (app.get_active_username() not in ("Default", "Public")) if hasattr(app, 'get_active_username') else False

        show_def_var = tk.BooleanVar(value=app.config.get("show_default_profile", True))
        show_pub_var = tk.BooleanVar(value=app.config.get("show_public_profile", True))

        def _on_toggle_profile_vis():
            app.config["show_default_profile"] = show_def_var.get()
            app.config["show_public_profile"] = show_pub_var.get()
            app.save_config()
            user_combo['values'] = app.list_user_profiles()

        cb_def = tk.Checkbutton(user_opts_frame, text="Show 'Default' in Profiles", variable=show_def_var,
                                command=_on_toggle_profile_vis, bg=THEME["bg_color"], fg=THEME["fg_color"],
                                selectcolor=THEME["widget_bg_color"], font=app.fonts["ui_small"],
                                state="normal" if is_logged_in else "disabled")
        cb_def.pack(side=tk.LEFT, padx=(5, 15))
        ToolTip(cb_def, "Toggle visibility of Default profile in switcher (requires logged in user).", app=app)

        cb_pub = tk.Checkbutton(user_opts_frame, text="Show 'Public' in Profiles", variable=show_pub_var,
                                command=_on_toggle_profile_vis, bg=THEME["bg_color"], fg=THEME["fg_color"],
                                selectcolor=THEME["widget_bg_color"], font=app.fonts["ui_small"],
                                state="normal" if is_logged_in else "disabled")
        cb_pub.pack(side=tk.LEFT, padx=5)
        ToolTip(cb_pub, "Toggle visibility of Public profile in switcher (requires logged in user).", app=app)

        # --- MULTI-AGENT DELEGATION & SUBAGENTS SECTION ---
        delegation_section = tk.Frame(main, bg=THEME["bg_color"], highlightbackground=THEME["electric_blue"], highlightthickness=1, bd=0)
        delegation_section.pack(fill=tk.X, padx=10, pady=(10, 5))

        del_header = tk.Frame(delegation_section, bg=THEME["widget_bg_color"])
        del_header.pack(fill=tk.X, padx=0, pady=0)
        lbl_del_hdr = tk.Label(del_header, text="👥 Multi-Agent Delegation & Subagents (Lvls 6 & 7)", bg=THEME["widget_bg_color"], 
                               fg=THEME["electric_blue"], font=app.fonts["bold"])
        lbl_del_hdr.pack(side=tk.LEFT, padx=8, pady=4)
        ToolTip(lbl_del_hdr, "Configure subagent task delegation, handoffs, and orchestration across Levels 1-6.", app=app)

        del_body = tk.Frame(delegation_section, bg=THEME["bg_color"])
        del_body.pack(fill=tk.X, padx=10, pady=6)

        del_left = tk.Frame(del_body, bg=THEME["bg_color"])
        del_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        delegation_enabled_var = tk.BooleanVar(value=app.config.get("delegation_enabled", False))
        cb_del_enable = tk.Checkbutton(del_left, text="Enable Delegation & Subagents (Lvls 6 & 7)", variable=delegation_enabled_var,
                                       bg=THEME["bg_color"], fg=THEME["fg_color"], selectcolor=THEME["widget_bg_color"],
                                       font=app.fonts["ui_label"])
        cb_del_enable.pack(anchor="w", pady=(0, 4))
        ToolTip(cb_del_enable, "Allows Transcendent (Lvl 6) and Cecilia (Lvl 7) to task subagents and orchestrate handoffs.", app=app)

        lbl_cecilia_mode = tk.Label(del_left, text="Cecilia Mode (Lvl 7):", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["ui_label"])
        lbl_cecilia_mode.pack(anchor="w", pady=(2, 0))
        ToolTip(lbl_cecilia_mode, "Choose between Shadow Wizard (full subagent orchestration) or Divine Judgement (direct omniscience).", app=app)

        cecilia_mode_var = tk.StringVar(value=app.config.get("cecilia_delegation_mode", "shadow_wizard"))
        cecilia_mode_frame = tk.Frame(del_left, bg=THEME["bg_color"])
        cecilia_mode_frame.pack(anchor="w", pady=(1, 4))
        for c_val, c_lbl in [("shadow_wizard", "Shadow Wizard (Subagent Orchestration)"), ("divine_judgement", "Divine Judgement (Direct Omniscience)")]:
            rb_c = tk.Radiobutton(cecilia_mode_frame, text=c_lbl, variable=cecilia_mode_var, value=c_val,
                                  bg=THEME["bg_color"], fg=THEME["fg_color"], selectcolor=THEME["widget_bg_color"],
                                  font=app.fonts["ui_small"])
            rb_c.pack(anchor="w", pady=1)
            ToolTip(rb_c, f"Set Cecilia operating mode to {c_lbl}.", app=app)

        lbl_sub_density = tk.Label(del_left, text="Subagent Density / Selection:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["ui_label"])
        lbl_sub_density.pack(anchor="w", pady=(2, 0))
        ToolTip(lbl_sub_density, "Control whether the orchestrator utilizes only the minimum required subagents or engages all desired agents.", app=app)
        subagent_density_var = tk.StringVar(value=app.config.get("subagent_selection_mode", "minimal"))
        density_frame = tk.Frame(del_left, bg=THEME["bg_color"])
        density_frame.pack(anchor="w", pady=(1, 4))
        for d_val, d_lbl in [("minimal", "Uses minimum required subagents"), ("all", "Use as many subagents as you want")]:
            rb_d = tk.Radiobutton(density_frame, text=d_lbl, variable=subagent_density_var, value=d_val,
                                  bg=THEME["bg_color"], fg=THEME["fg_color"], selectcolor=THEME["widget_bg_color"],
                                  font=app.fonts["ui_small"])
            rb_d.pack(anchor="w", pady=1)
            ToolTip(rb_d, f"Configure subagent density to: {d_lbl}.", app=app)

        del_right = tk.Frame(del_body, bg=THEME["bg_color"])
        del_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=15)

        lbl_engine_mode = tk.Label(del_right, text="Execution Engine Model:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["ui_label"])
        lbl_engine_mode.pack(anchor="w")
        ToolTip(lbl_engine_mode, "Choose whether to run all subagents with the active Lvl 6/7 model or dynamically swap to each subagent's assigned model.", app=app)
        delegation_model_mode_var = tk.StringVar(value=app.config.get("delegation_model_mode", "lvl6_7_model"))
        model_mode_frame = tk.Frame(del_right, bg=THEME["bg_color"])
        model_mode_frame.pack(anchor="w", pady=(1, 4))
        for m_val, m_lbl in [("lvl6_7_model", "Use model selected for Lvl 6 / 7"), ("per_subagent_model", "Use model selected for specific subagent")]:
            rb_m = tk.Radiobutton(model_mode_frame, text=m_lbl, variable=delegation_model_mode_var, value=m_val,
                                  bg=THEME["bg_color"], fg=THEME["fg_color"], selectcolor=THEME["widget_bg_color"],
                                  font=app.fonts["ui_small"])
            rb_m.pack(anchor="w", pady=1)
            ToolTip(rb_m, f"Engine execution mode: {m_lbl}.", app=app)

        lbl_chain = tk.Label(del_right, text="Delegation Chain Preset:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["ui_label"])
        lbl_chain.pack(anchor="w")
        ToolTip(lbl_chain, "Select default subagent execution and reporting pipeline.", app=app)
        chain_preset_var = tk.StringVar(value=app.config.get("delegation_chain_preset", "standard"))
        chain_combo = ttk.Combobox(del_right, textvariable=chain_preset_var, 
                                   values=["standard (L2 Search -> L3 Store -> L5 Reason -> L6/7 Approve)", "direct_strike (L2 Search -> L6/7 Approve)"], 
                                   state="readonly", width=36)
        chain_combo.pack(anchor="w", pady=(2, 6))
        ToolTip(chain_combo, "Select pipeline order for subagent handoffs.", app=app)

        lbl_handoff = tk.Label(del_right, text="Handoff Reporting Target:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["ui_label"])
        lbl_handoff.pack(anchor="w")
        ToolTip(lbl_handoff, "Where subagents report their intermediary outputs.", app=app)
        handoff_target_var = tk.StringVar(value=app.config.get("delegation_handoff_target", "lvl3_compiler"))
        handoff_combo = ttk.Combobox(del_right, textvariable=handoff_target_var,
                                     values=["lvl3_compiler (Staging & Aggregation)", "taskmaster_direct (Direct to Lvl 6/7)"],
                                     state="readonly", width=36)
        handoff_combo.pack(anchor="w", pady=(2, 4))
        ToolTip(handoff_combo, "Select whether intermediate results stage in Level 3 memory or return directly to orchestrator.", app=app)

        # --- LOADING BAR & STATUS AREA CONFIGURATION ---
        status_bar_section = tk.Frame(main, bg=THEME["bg_color"], highlightbackground=THEME["electric_blue"], highlightthickness=1, bd=0)
        status_bar_section.pack(fill=tk.X, padx=10, pady=(10, 5))

        sb_header = tk.Frame(status_bar_section, bg=THEME["widget_bg_color"])
        sb_header.pack(fill=tk.X, padx=0, pady=0)
        lbl_sb_hdr = tk.Label(sb_header, text="⏳ Loading Bar & Status Area", bg=THEME["widget_bg_color"], 
                 fg=THEME["electric_blue"], font=app.fonts["bold"])
        lbl_sb_hdr.pack(side=tk.LEFT, padx=8, pady=4)
        ToolTip(lbl_sb_hdr, "Customize loading bar behavior, transition styles, and idle displays.", app=app)

        sb_body = tk.Frame(status_bar_section, bg=THEME["bg_color"])
        sb_body.pack(fill=tk.X, padx=10, pady=6)

        opts_frame = tk.Frame(sb_body, bg=THEME["bg_color"])
        opts_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        lbl_disp_opts = tk.Label(opts_frame, text="Display Options:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["bold"])
        lbl_disp_opts.pack(anchor="w", pady=(0, 2))
        ToolTip(lbl_disp_opts, "Select active loading indicator display mode.", app=app)

        STATUS_MODES = [
            ("hybrid", "Hybrid / Smart Feature (State-Aware with Finish t/s)"),
            ("tasks", "Active Generation Tasks (Prefill, Reasoning, Streaming)"),
            ("percentage", "Percentage Gauge (Load % & TTFT / Estimated Duration)"),
            ("animation", "Selectable Animation (Custom Canvas Animation)"),
            ("prayer", "Serenity Prayer (Smooth Line Fading Transition)")
        ]
        status_mode_var = tk.StringVar(value=app.config.get("status_bar_mode", "hybrid"))
        for val, lbl in STATUS_MODES:
            rb_sb = tk.Radiobutton(opts_frame, text=lbl, variable=status_mode_var, value=val,
                           bg=THEME["bg_color"], fg=THEME["fg_color"], selectcolor=THEME["widget_bg_color"],
                           font=app.fonts["ui_small"])
            rb_sb.pack(anchor="w", padx=4, pady=1)
            ToolTip(rb_sb, f"Switch status bar display mode to {lbl}.", app=app)

        toggles_frame = tk.Frame(sb_body, bg=THEME["bg_color"])
        toggles_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=15)

        lbl_anim_s = tk.Label(toggles_frame, text="Animation Style:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["ui_label"])
        lbl_anim_s.pack(anchor="w")
        ToolTip(lbl_anim_s, "Select animation visual style when Animation mode is active.", app=app)
        anim_style_var = tk.StringVar(value=app.config.get("status_bar_anim_style", "spinner"))
        anim_combo = ttk.Combobox(toggles_frame, textvariable=anim_style_var, values=["spinner", "pulse", "orbit"], state="readonly", width=12)
        anim_combo.pack(anchor="w", pady=(2, 6))
        ToolTip(anim_combo, "Choose between spinner, pulse, or orbit canvas animation.", app=app)

        lbl_sb_tog = tk.Label(toggles_frame, text="Status Area Toggles:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["bold"])
        lbl_sb_tog.pack(anchor="w", pady=(4, 2))
        ToolTip(lbl_sb_tog, "Configure status bar idle behavior and fallback metadata.", app=app)

        sb_dmn_var = tk.BooleanVar(value=app.config.get("status_bar_dmn_idle", True))
        cb_dmn_t = tk.Checkbutton(toggles_frame, text="Swaps to DMN timer showing idle time", variable=sb_dmn_var,
                       bg=THEME["bg_color"], fg=THEME["fg_color"], selectcolor=THEME["widget_bg_color"],
                       font=app.fonts["ui_small"])
        cb_dmn_t.pack(anchor="w", pady=1)
        ToolTip(cb_dmn_t, "Displays DMN idle timer countdown when resting.", app=app)

        sb_fallback_var = tk.BooleanVar(value=app.config.get("status_bar_fallback_info", True))
        cb_fb_t = tk.Checkbutton(toggles_frame, text="Defaults back to active level & KV quant/ctx info", variable=sb_fallback_var,
                       bg=THEME["bg_color"], fg=THEME["fg_color"], selectcolor=THEME["widget_bg_color"],
                       font=app.fonts["ui_small"])
        cb_fb_t.pack(anchor="w", pady=1)
        ToolTip(cb_fb_t, "Shows persona level and hardware KV cache status when idle.", app=app)

        # --- SECURITY & VAULT ENCRYPTION PANEL ---
        vault_section = tk.Frame(main, bg=THEME["bg_color"], highlightbackground=THEME["electric_blue"], highlightthickness=1, bd=0)
        vault_section.pack(fill=tk.X, padx=10, pady=(15, 10))

        vault_header = tk.Frame(vault_section, bg=THEME["widget_bg_color"])
        vault_header.pack(fill=tk.X, padx=0, pady=0)
        tk.Label(vault_header, text="🔐 Secure Vault Encryption", bg=THEME["widget_bg_color"], 
                 fg=THEME["electric_blue"], font=app.fonts["bold"]).pack(side=tk.LEFT, padx=8, pady=4)

        vault_status_lbl = tk.Label(vault_header, text="", font=app.fonts["ui_button"], bg=THEME["widget_bg_color"])
        vault_status_lbl.pack(side=tk.RIGHT, padx=8, pady=4)

        def _refresh_vault_status_ui():
            if hasattr(app, 'vault_manager') and app.vault_manager.is_lock_enabled():
                if app.vault_manager.is_locked():
                    vault_status_lbl.config(text="● LOCKED", fg="#ff4444")
                else:
                    vault_status_lbl.config(text="● ACTIVE (Unlocked)", fg="#00ff88")
            else:
                vault_status_lbl.config(text="○ DISABLED (Plaintext)", fg="#888888")

        _refresh_vault_status_ui()

        vault_body = tk.Frame(vault_section, bg=THEME["bg_color"], padx=8, pady=6)
        vault_body.pack(fill=tk.X)

        # Master Password Action Buttons
        v_btn_row = tk.Frame(vault_body, bg=THEME["bg_color"])
        v_btn_row.pack(fill=tk.X, pady=(2, 6))

        def _open_set_password_modal():
            from System.vault_manager import DISCLAIMER_WARNING_TEXT
            pwd_win = tk.Toplevel(win)
            pwd_win.title("Set Master Vault Password")
            pwd_win.geometry("540x520")
            pwd_win.config(bg=THEME["bg_color"])
            pwd_win.transient(win)
            pwd_win.grab_set()

            # Loud All-Caps Disclaimer Box
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

            def _apply_new_password():
                new_p = new_pwd_var.get().strip()
                conf_p = confirm_pwd_var.get().strip()
                curr_p = curr_pwd_var.get().strip() if is_already_enabled else None

                if len(new_p) < 4:
                    messagebox.showerror("Error", "Password must be at least 4 characters long.", parent=pwd_win)
                    return
                if new_p != conf_p:
                    messagebox.showerror("Error", "New password and confirmation do not match.", parent=pwd_win)
                    return

                if not messagebox.askyesno("CONFIRM ENCRYPTION", 
                                           "ARE YOU ABSOLUTELY SURE?\n\nIf you lose this password, ALL history files will be PERMANENTLY lost.\n\nProceed with AES-256-GCM history migration?",
                                           parent=pwd_win):
                    return

                success, msg = app.vault_manager.set_password(new_p, curr_p)
                if success:
                    messagebox.showinfo("Vault Configured", msg, parent=pwd_win)
                    _refresh_vault_status_ui()
                    pwd_win.destroy()
                else:
                    messagebox.showerror("Vault Error", msg, parent=pwd_win)

            btn_box = tk.Frame(pwd_win, bg=THEME["bg_color"])
            btn_box.pack(fill=tk.X, padx=16, pady=12)
            tk.Button(btn_box, text="Encrypt & Set Password", command=_apply_new_password,
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
                    _refresh_vault_status_ui()
                    dis_win.destroy()
                else:
                    messagebox.showerror("Verification Failed", msg, parent=dis_win)

            tk.Button(dis_win, text="Decrypt & Disable", command=_do_disable,
                      bg="#660000", fg="white", font=app.fonts["ui_button"]).pack(pady=12)

        btn_set_pwd = tk.Button(v_btn_row, text="🔑 Set / Change Master Password", command=_open_set_password_modal,
                  bg=THEME["button_bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_small"])
        btn_set_pwd.pack(side=tk.LEFT, padx=4)
        ToolTip(btn_set_pwd, "Set or change the AES-256-GCM Master Password for encrypting conversation histories.", app=app)

        btn_dis_vault = tk.Button(v_btn_row, text="🔓 Disable Vault Lock", command=_open_disable_vault_modal,
                  bg=THEME["button_bg_color"], fg="#ffaa00", font=app.fonts["ui_small"])
        btn_dis_vault.pack(side=tk.LEFT, padx=4)
        ToolTip(btn_dis_vault, "Decrypt all histories back to plaintext and remove master password protection.", app=app)

        def _lock_now():
            if hasattr(app, 'vault_manager') and app.vault_manager.is_lock_enabled():
                app.vault_manager.lock()
                _refresh_vault_status_ui()
                if hasattr(app, 'show_vault_unlock_modal'):
                    app.show_vault_unlock_modal()
            else:
                messagebox.showinfo("Vault Inactive", "Enable Master Password first to lock the application.", parent=win)

        btn_lock_app = tk.Button(v_btn_row, text="🔒 Lock App Now", command=_lock_now,
                  bg="#441111", fg="#ff8888", font=app.fonts["ui_small"])
        btn_lock_app.pack(side=tk.RIGHT, padx=4)
        ToolTip(btn_lock_app, "Instantly lock Serenity PC and encrypt active history memory.", app=app)

        # Inactivity Auto-Lock Control (Slider + Typeable input + Quick presets)
        auto_lock_sec = app.vault_manager.get_auto_lock_seconds() if hasattr(app, 'vault_manager') else 0
        auto_lock_var = tk.IntVar(value=auto_lock_sec)
        auto_lock_min_var = tk.DoubleVar(value=round(auto_lock_sec / 60.0, 1))

        inactivity_frame = tk.Frame(vault_body, bg=THEME["bg_color"])
        inactivity_frame.pack(fill=tk.X, pady=(4, 2))

        lbl_alock = tk.Label(inactivity_frame, text="Auto-Lock Inactivity:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["ui_label"])
        lbl_alock.pack(side=tk.LEFT, padx=(0, 6))
        ToolTip(lbl_alock, "Automatically locks vault after a period of user inactivity (0 = disabled).", app=app)

        sec_entry = tk.Entry(inactivity_frame, textvariable=auto_lock_var, width=6, 
                             bg=THEME["widget_bg_color"], fg=THEME["fg_color"],
                             insertbackground=THEME.get("electric_blue", THEME["fg_color"]))
        bind_entry_limit(sec_entry, max_len=6)
        sec_entry.pack(side=tk.LEFT, padx=4)
        ToolTip(sec_entry, "Auto-lock timeout in seconds.", app=app)
        tk.Label(inactivity_frame, text="sec", bg=THEME["bg_color"], fg="#888888").pack(side=tk.LEFT, padx=(0, 8))

        slider_lock = tk.Scale(inactivity_frame, from_=0, to=60, orient=tk.HORIZONTAL, resolution=0.5,
                               variable=auto_lock_min_var, bg=THEME["bg_color"], fg=THEME["fg_color"],
                               troughcolor=THEME["widget_bg_color"], highlightthickness=0, length=140)
        slider_lock.pack(side=tk.LEFT, padx=4)
        ToolTip(slider_lock, "Slide to set auto-lock inactivity timer in minutes.", app=app)
        tk.Label(inactivity_frame, text="min", bg=THEME["bg_color"], fg="#888888").pack(side=tk.LEFT, padx=(0, 8))

        def _on_sec_entry_change(*args):
            try:
                s = int(auto_lock_var.get())
                auto_lock_min_var.set(round(s / 60.0, 1))
            except: pass

        def _on_min_slider_change(*args):
            try:
                m = float(auto_lock_min_var.get())
                auto_lock_var.set(int(m * 60))
            except: pass

        auto_lock_var.trace_add("write", _on_sec_entry_change)
        auto_lock_min_var.trace_add("write", _on_min_slider_change)

        # Quick Preset Buttons
        def _set_preset(s):
            auto_lock_var.set(s)
            auto_lock_min_var.set(round(s / 60.0, 1))

        presets_frame = tk.Frame(vault_body, bg=THEME["bg_color"])
        presets_frame.pack(fill=tk.X, pady=(2, 4))
        lbl_pres = tk.Label(presets_frame, text="Presets:", bg=THEME["bg_color"], fg="#777777", font=app.fonts["ui_small"])
        lbl_pres.pack(side=tk.LEFT, padx=(0, 4))
        ToolTip(lbl_pres, "Quick auto-lock timer duration presets.", app=app)
        for p_sec, p_lbl in [(0, "Off"), (15, "15s"), (30, "30s"), (45, "45s"), (300, "5m"), (900, "15m"), (1800, "30m")]:
            btn_p = tk.Button(presets_frame, text=p_lbl, command=lambda s=p_sec: _set_preset(s),
                      bg=THEME["widget_bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_small"], relief=tk.FLAT, padx=3, pady=0)
            btn_p.pack(side=tk.LEFT, padx=2)
            ToolTip(btn_p, f"Set auto-lock inactivity timer to {p_lbl}.", app=app)

        tk.Label(main, text="Text/Inline Engines:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["bold"]).pack(anchor="w", padx=10, pady=(15, 5))
        tier_grid = tk.Frame(main, bg=THEME["bg_color"])
        tier_grid.pack(fill=tk.X, pady=10)
        tier_grid.grid_columnconfigure(0, weight=1); tier_grid.grid_columnconfigure(1, weight=1)
        tiers = ["fast", "search", "low", "med", "high", "transcendent", "secret", "deep_cook"]
        for i, tier in enumerate(tiers):
            r, c = divmod(i, 2)
            _create_tier_block(tier_grid, tier, r, c)


        tk.Label(main, text="Vision Engines:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["bold"]).pack(anchor="w", padx=10, pady=(15, 5))
        v_grid = tk.Frame(main, bg=THEME["bg_color"]); v_grid.pack(fill=tk.X, pady=5)
        v_grid.grid_columnconfigure(0, weight=1); v_grid.grid_columnconfigure(1, weight=1)
        for i, vt in enumerate(["video", "video_deep", "multimodal"]):
            r, c = divmod(i, 2)
            _create_tier_block(v_grid, vt, r, c, True)

        def _apply_settings(close_window=False):
            app.config["media_rendering"] = media_var.get()
            app.state["deep_cook_behavior"] = v_behavior.get()
            if app.state["deep_cook_behavior"] == "oneshot":
                app.state["deep_cook"] = False
            app._sync_deep_cook_ui()
            app.config["k_cache_type"] = k_cache_var.get()
            app.config["v_cache_type"] = v_cache_var.get()
            app.config["history_lookup_mode"] = history_lookup_var.get()
            app.config["history_usage"] = history_usage_var.get()
            app.config["turbovec_mode"] = turbovec_mode_var.get()
            app.config["dynamic_params_enabled"] = dynamic_params_var.get()
            app.config["ghost_mode"] = ghost_var.get()
            if hasattr(app, 'ghost_button') and app.ghost_button:
                app.ghost_button.config(text=app._get_ghost_mode_label(), fg=app._get_ghost_mode_color())
            if hasattr(app, 'history_usage_button') and app.history_usage_button:
                app.history_usage_button.config(text=app._get_history_usage_label(), fg=app._get_history_usage_color())
            if hasattr(app, "soft_reload_turbovec"):
                app.soft_reload_turbovec()
            elif getattr(app, 'turbo_vec', None):
                import threading
                threading.Thread(
                    target=app.turbo_vec.ingest_needed_files, 
                    args=(app.model_path, app.active_persona_level, history_lookup_var.get()),
                    daemon=True
                ).start()
            try:
                for path in [
                    os.path.join(app.script_dir, "Live", "System", "params.json"),
                    os.path.join("Live", "System", "params.json")
                ]:
                    if os.path.exists(path):
                        with open(path, "r") as f:
                            p_data = json.load(f)
                        p_data["k_cache_type"] = k_cache_var.get()
                        p_data["v_cache_type"] = v_cache_var.get()
                        with open(path, "w") as f:
                            json.dump(p_data, f, indent=4)
            except Exception as pe:
                print(f"[UI] Warning: Could not write KV cache types to Live params: {pe}")
            app.config["hao_preset"] = hao_var.get()

            app.config["swa_kv_cache"] = swa_var.get()
            app.config["auto_vram_offload"] = auto_vram_var.get()
            old_spec = app.config.get("speculative_drafting", False)
            app.config["speculative_drafting"] = spec_draft_var.get()
            draft_toggled = (old_spec != spec_draft_var.get())
            app.config["ghost_mode"] = ghost_var.get()
            app.config["thinking_checkbox"] = thinking_var.get()
            app.config["benchmark_enabled"] = benchmark_var.get()
            app.config["inline_markdown"] = inline_md_var.get()
            app.config["budget_recovery_mode"] = budget_recovery_var.get()
            app.config["monitor_graph_mode"] = monitor_graph_var.get()
            app.config["show_tooltips"] = show_tooltips_var.get()
            app.state["streaming_mode"] = stream_var.get()
            app.config["max_token_ratio"] = ratio_var.get()
            app.config["multimedia_handling"] = multimedia_handling_var.get()
            app.config["image_handling"] = multimedia_handling_var.get()
            app.config["muse_reasoning_strength"] = muse_reasoning_var.get()
            app.config["dmn_timeout"] = dmn_ent.get().strip()
            app.config["stt_device_index"] = dev_id_map.get(stt_dev_var.get(), None)
            app.config["stt_language"] = stt_lang_var.get()
            if hasattr(app, 'vault_manager'):
                try: app.vault_manager.set_auto_lock_seconds(int(auto_lock_var.get()))
                except: pass
            try: app.state["virtual_vram"] = int(float(vram_ent.get()) * 1024)
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
            app.config["repeat_detection_mode"] = repeat_mode_var.get()
            app.config["offline_mode"] = offline_mode_var.get()
            try:
                from System.network_guard import set_offline_mode
                set_offline_mode(offline_mode_var.get())
            except Exception as e:
                print(f"[SETTINGS] Failed to set offline guard: {e}")

            theme_k = THEME_MAP.get(theme_display_var.get(), "apex")
            tex_k = TEXTURE_MAP.get(tex_display_var.get(), "default")
            d_mode = dark_mode_var.get()
            tex_int = float(tex_int_var.get()) / 100.0
            app.config["theme"] = theme_k
            app.config["texture_style"] = tex_k
            app.config["texture_intensity"] = tex_int
            app.config["dark_mode"] = d_mode

            # User Profile & Status Bar config
            new_un = username_var.get().strip() or "Default"
            if new_un != app.config.get("username", "Default") and hasattr(app, "switch_user"):
                if not _verify_vault_unlock(new_un):
                    new_un = app.config.get("username", "Default")
                    username_var.set(new_un)
                else:
                    app.switch_user(new_un)
            else:
                app.config["username"] = new_un
            
            app.config["user_preferred_name"] = user_pref_name_var.get().strip()
            app.config["user_address_style"] = user_addr_style_var.get().strip()

            # Multi-Agent Delegation Config
            app.config["delegation_enabled"] = delegation_enabled_var.get()
            app.config["cecilia_delegation_mode"] = cecilia_mode_var.get()
            app.config["subagent_selection_mode"] = subagent_density_var.get()
            app.config["delegation_model_mode"] = delegation_model_mode_var.get()
            app.config["delegation_chain_preset"] = chain_preset_var.get()
            app.config["delegation_handoff_target"] = handoff_target_var.get()
            
            app.config["status_bar_mode"] = status_mode_var.get()
            app.config["status_bar_anim_style"] = anim_style_var.get()
            app.config["status_bar_dmn_idle"] = sb_dmn_var.get()
            app.config["status_bar_fallback_info"] = sb_fallback_var.get()

            target_scale = text_scale_val_var.get()
            if hasattr(app, "apply_text_scale"):
                app.apply_text_scale(target_scale, persist=True)
            else:
                app.config["text_scale"] = target_scale

            target_ui_font = ui_font_var.get()
            target_mono_font = mono_font_var.get()
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
                
                # Refresh settings window colors immediately
                if win.winfo_exists():
                    win.config(bg=THEME["bg_color"])
                    container.config(bg=THEME["bg_color"])
                    canvas.config(bg=THEME["bg_color"])
                    scrollable_frame.config(bg=THEME["bg_color"])
                    btn_frame.config(bg=THEME["bg_color"])
                    for rb_group in [hao_rbs, swa_rbs, stream_rbs, ratio_rbs, repeat_rbs, t_action_rbs, t_slot_rbs]:
                        for rb, v in rb_group:
                            if rb.winfo_exists():
                                rb.config(bg=THEME["widget_bg_color"], selectcolor=THEME["electric_blue"], activebackground=THEME["electric_blue"], activeforeground="#000000")
                    _update_t_action_colors()
                    _update_template_slot_colors()
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

        save_close_btn = tk.Button(btn_frame, text="Save & Close", command=lambda: _apply_settings(True),
                                   bg=THEME["button_active_color"], fg=THEME["fg_color"], font=app.fonts["ui_button"], relief=tk.FLAT)
        save_close_btn.pack(side=tk.RIGHT, padx=4)
        ToolTip(save_close_btn, "Save all settings to configuration file and close settings window.", app=app)

        apply_btn = tk.Button(btn_frame, text="Apply", command=lambda: _apply_settings(False),
                              bg=THEME.get("button_bg_color", "#202020"), fg=THEME.get("accent_highlight", "#00ffcc"), font=app.fonts["ui_button"], relief=tk.FLAT)
        apply_btn.pack(side=tk.RIGHT, padx=4)
        ToolTip(apply_btn, "Apply current settings immediately without closing the settings window.", app=app)

        btn_clr_h = tk.Button(btn_frame, text="Clear History", command=app.clear_current_history, bg="#660000", fg="white", font=app.fonts["ui_button"], relief=tk.FLAT)
        btn_clr_h.pack(side=tk.RIGHT, padx=4)
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
        
        btn_auto_d = tk.Button(btn_frame, text="Auto-Detect", command=_reset_defaults, bg=THEME["button_bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_button"], relief=tk.FLAT)
        btn_auto_d.pack(side=tk.RIGHT, padx=4)
        ToolTip(btn_auto_d, "Benchmark system hardware and auto-calculate GPU layer offloads across tiers.", app=app)

        btn_cancel = tk.Button(btn_frame, text="Cancel", command=win.destroy, bg=THEME["button_bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_button"], relief=tk.FLAT)
        btn_cancel.pack(side=tk.RIGHT, padx=4)
        ToolTip(btn_cancel, "Discard unapplied changes and close settings window.", app=app)

    except Exception as e:
        import traceback
        err_msg = f"Settings Window Crash: {e}\n{traceback.format_exc()}"
        print(err_msg)
        try:
            os.makedirs("Logs", exist_ok=True)
            with open("Logs/ui_crash.txt", "w") as f: f.write(err_msg)
        except: pass
        messagebox.showerror("UI Error", f"Settings window failed to open:\n{e}")

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
