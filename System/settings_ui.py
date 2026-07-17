import os
import json
import sys
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from serenity_resources import THEME

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
    elif system_monitor_loaded and getattr(app, 'gpu_handle', None):
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
    tiers = ["fast", "search", "low", "med", "high", "secret", "Live", "deep_cook", 
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
        win.attributes("-topmost", False)
        
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
        v_behavior = tk.StringVar(value=app.state.get("deep_cook_behavior", "oneshot"))
        behavior_frame = tk.Frame(left_header, bg=THEME["bg_color"])
        behavior_frame.pack(anchor="w", padx=10)
        for val, txt in [("oneshot", "One-Shot"), ("toggle", "Toggle Mode")]:
             tk.Radiobutton(behavior_frame, text=txt, variable=v_behavior, value=val, bg=THEME["bg_color"], fg=THEME["fg_color"], 
                            selectcolor=THEME["widget_bg_color"]).pack(side=tk.LEFT, padx=5)
        
        vram_frame = tk.Frame(left_header, bg=THEME["bg_color"])
        vram_frame.pack(anchor="w", pady=(5, 0))
        tk.Label(vram_frame, text="VRAM (GB):", bg=THEME["bg_color"], fg=THEME["electric_blue"]).pack(side=tk.LEFT)
        vram_ent = tk.Entry(vram_frame, bg=THEME["widget_bg_color"], fg=THEME["fg_color"], width=6)
        vram_ent.insert(0, f"{app.state.get('virtual_vram', 0)/1024:g}" if app.state.get('virtual_vram', 0) > 0 else "0")
        vram_ent.pack(side=tk.LEFT, padx=5)
        
        g_frame = tk.Frame(left_header, bg=THEME["bg_color"])
        g_frame.pack(anchor="w", pady=(10, 0))
        tk.Label(g_frame, text="Glitch FX:", bg=THEME["bg_color"], fg=THEME["electric_blue"]).pack(side=tk.LEFT)
        glitch_anim_var = tk.StringVar(value=app.config.get("glitch_animation", "warp"))
        for opt in ["warp", "vortex", "off"]:
            tk.Radiobutton(g_frame, text=opt, variable=glitch_anim_var, value=opt,
                           bg=THEME["bg_color"], fg=THEME["fg_color"], selectcolor=THEME["widget_bg_color"]).pack(side=tk.LEFT, padx=2)

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
    
            tk.Checkbutton(toggle_frame, text="Show RGB Button", variable=show_rgb_var, command=_toggle_rgb,
                           bg=THEME["bg_color"], fg=THEME["electric_blue"], selectcolor=THEME["widget_bg_color"]).pack(anchor="w")

        tk.Label(left_header, text="Image Handling Mode:", bg=THEME["bg_color"], fg=THEME["electric_blue"]).pack(anchor="w", pady=(5, 0))
        image_handling_var = tk.StringVar(value=app.config.get("image_handling", "auto"))
        image_handling_frame = tk.Frame(left_header, bg=THEME["bg_color"])
        image_handling_frame.pack(anchor="w", padx=10)
        for opt in ["auto", "vision", "native"]:
            tk.Radiobutton(image_handling_frame, text=opt.capitalize(), variable=image_handling_var, value=opt,
                           bg=THEME["bg_color"], fg=THEME["fg_color"], selectcolor=THEME["widget_bg_color"]).pack(side=tk.LEFT, padx=5)

        # 3 Checkboxes relocated from Global Overrides to left column
        auto_vram_var = tk.BooleanVar(value=app.config.get("auto_vram_offload", False))
        spec_draft_var = tk.BooleanVar(value=app.config.get("speculative_drafting", True))
        ghost_var = tk.BooleanVar(value=app.config.get("ghost_mode", False))
        thinking_var = tk.BooleanVar(value=app.config.get("thinking_checkbox", True))
        benchmark_var = tk.BooleanVar(value=app.config.get("benchmark_enabled", False))
        inline_md_var = tk.BooleanVar(value=app.config.get("inline_markdown", True))
        monitor_graph_var = tk.BooleanVar(value=app.config.get("monitor_graph_mode", False))

        auto_vram_f = tk.Frame(left_header, bg=THEME["bg_color"])
        auto_vram_f.pack(anchor="w", pady=(10, 0))
        tk.Checkbutton(auto_vram_f, text="Dynamic Auto-Offload", variable=auto_vram_var,
                       bg=THEME["bg_color"], fg=THEME["electric_blue"], selectcolor=THEME["widget_bg_color"]).pack(anchor="w", pady=2)
        tk.Checkbutton(auto_vram_f, text="Speculative MTP Drafting", variable=spec_draft_var,
                       bg=THEME["bg_color"], fg=THEME["electric_blue"], selectcolor=THEME["widget_bg_color"]).pack(anchor="w", pady=2)
        tk.Checkbutton(auto_vram_f, text="Ghost Mode", variable=ghost_var,
                       bg=THEME["bg_color"], fg=THEME["electric_blue"], selectcolor=THEME["widget_bg_color"]).pack(anchor="w", pady=2)
        tk.Checkbutton(auto_vram_f, text="Thinking Process", variable=thinking_var,
                       bg=THEME["bg_color"], fg=THEME["electric_blue"], selectcolor=THEME["widget_bg_color"]).pack(anchor="w", pady=2)
        tk.Checkbutton(auto_vram_f, text="Loading Benchmark", variable=benchmark_var,
                       bg=THEME["bg_color"], fg=THEME["electric_blue"], selectcolor=THEME["widget_bg_color"]).pack(anchor="w", pady=2)
        tk.Checkbutton(auto_vram_f, text="Inline Markdown", variable=inline_md_var,
                       bg=THEME["bg_color"], fg=THEME["electric_blue"], selectcolor=THEME["widget_bg_color"]).pack(anchor="w", pady=2)
        tk.Checkbutton(auto_vram_f, text="Monitor Graph vs Line", variable=monitor_graph_var,
                       bg=THEME["bg_color"], fg=THEME["electric_blue"], selectcolor=THEME["widget_bg_color"]).pack(anchor="w", pady=2)

        tk.Label(center_header, text="Templating Engine (32 Slots):", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=("Open Sans", 9, "bold")).pack(anchor="n")
        template_mode = tk.StringVar(value="modify")
        active_template = tk.StringVar(value="")
        
        t_action_frame = tk.Frame(center_header, bg=THEME["bg_color"])
        t_action_frame.pack(anchor="n", pady=2)
        for val, txt in [("save", "Save"), ("write", "Write"), ("modify", "Modify")]:
            tk.Radiobutton(t_action_frame, text=txt, variable=template_mode, value=val, indicatoron=0, 
                           bg=THEME["widget_bg_color"], fg=THEME["fg_color"], selectcolor=THEME["button_active_color"]).pack(side=tk.LEFT, padx=2)

        t_grid = tk.Frame(center_header, bg=THEME["bg_color"])
        t_grid.pack(anchor="n", pady=5)
        
        template_buttons = []
        for i in range(8):
            for j in range(4):
                slot_id = f"T{(i*4)+j+1}"
                t_name = app.config.get("custom_templates", {}).get(slot_id, {}).get("name", slot_id)
                b = tk.Radiobutton(t_grid, text=t_name, variable=active_template, value=slot_id, indicatoron=0, width=12, 
                                   bg=THEME["widget_bg_color"], fg=THEME["electric_blue"], selectcolor=THEME["button_active_color"])
                b.grid(row=i, column=j, padx=2, pady=2)
                b.slot_id = slot_id
                template_buttons.append(b)

        def _on_template_select(*args):
            mode = template_mode.get()
            t_id = active_template.get()
            if not t_id: return
            if mode == "modify":
                t_win = tk.Toplevel(win)
                t_win.title(f"Modify {t_id}")
                t_win.geometry("300x480")
                t_win.config(bg=THEME["bg_color"])
                t_win.attributes("-topmost", False)
                current = app.config.get("custom_templates", {}).get(t_id, {})
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
                    if "custom_templates" not in app.config: app.config["custom_templates"] = {}
                    app.config["custom_templates"][t_id] = t_data
                    for btn in template_buttons:
                        if btn.slot_id == t_id: btn.config(text=t_data["name"])
                    app.save_config()
                    t_win.destroy()
                tk.Button(t_win, text="Save & Close", command=_save_mod, bg=THEME["button_active_color"], fg=THEME["fg_color"]).pack(pady=15)
        active_template.trace_add("write", _on_template_select)

        right_header = tk.Frame(header_settings, bg=THEME["bg_color"])
        right_header.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(20, 0))
        tk.Label(right_header, text="Video Processing Sub-Chunk Size:", bg=THEME["bg_color"], fg=THEME["electric_blue"]).pack(anchor="w")
        sc_frame = tk.Frame(right_header, bg=THEME["bg_color"])
        sc_frame.pack(fill=tk.X, padx=5)
        sc_val = tk.IntVar(value=getattr(app, 'sub_chunk_size', 8))
        tk.Scale(sc_frame, from_=1, to=128, orient=tk.HORIZONTAL, variable=sc_val, 
                 bg=THEME["bg_color"], fg=THEME["fg_color"], highlightthickness=0, resolution=1).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        def _reset_sc(): sc_val.set(8)
        tk.Button(sc_frame, text="Reset", command=_reset_sc).pack(side=tk.RIGHT, padx=5)

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
                messagebox.showinfo("Templating", f"Saved {tier_name.upper()} settings to {t_data['name']}!")
            elif mode == "write":
                t_data = app.config.get("custom_templates", {}).get(t_id, {})
                if not t_data: return
                for k, d in [("temp", temp_ents), ("top_p", top_p_ents), ("min_p", min_p_ents), ("rep", rep_ents), ("pres", pres_ents), ("freq", freq_ents), ("top_k", top_k_ents), 
                ("batch", n_batch_ents), ("layers", ents), ("ctx", ctx_ents), ("stop", stop_ents)]:
                    if k in t_data: 
                        d[tier_name].delete(0, tk.END)
                        d[tier_name].insert(0, str(t_data[k]))
                messagebox.showinfo("Templating", f"Applied {t_data['name']} to {tier_name.upper()}!")

        def _create_tier_block(parent, tier_name, row=0, col=0, is_vision=False):
            key = f"vision_{tier_name}" if is_vision else tier_name
            lvl_map = {"fast": "1", "search": "2", "low": "3", "med": "4", "high": "5", "secret": "6", "Live": "7"}
            title_suffix = f" (Lvl {lvl_map[tier_name]})" if tier_name in lvl_map else ""
            lf = tk.LabelFrame(parent, text=f"Engine: {tier_name.upper()}{title_suffix}", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=("Open Sans", 10, "bold"), pady=5)
            lf.grid(row=row, column=col, sticky="nsew", padx=10, pady=5)
            
            def _bind_click(w):
                w.bind("<Button-1>", lambda e: _on_tier_box_click(key), add="+")
                for c in w.winfo_children(): _bind_click(c)
            
            r1 = tk.Frame(lf, bg=THEME["bg_color"]); r1.pack(fill=tk.X, padx=5)
            tk.Button(r1, text="Set Path", command=lambda t=key: app._set_path(t, labels, win)).pack(side=tk.LEFT)
            labels[key] = tk.Label(r1, text=os.path.basename(app.model_paths.get(key, "") or "Not Set"), bg=THEME["bg_color"], fg=THEME["fg_color"], font=("Open Sans", 8))
            labels[key].pack(side=tk.LEFT, padx=5)
            
            r1b = tk.Frame(lf, bg=THEME["bg_color"]); r1b.pack(fill=tk.X, padx=5, pady=2)
            tk.Label(r1b, text="Layers:", bg=THEME["bg_color"], fg=THEME["fg_color"]).pack(side=tk.LEFT)
            ents[key] = tk.Entry(r1b, width=4); ents[key].insert(0, str(app.gpu_layer_config.get(key, -1))); ents[key].pack(side=tk.LEFT, padx=2)
            tk.Label(r1b, text="Ctx:", bg=THEME["bg_color"], fg=THEME["fg_color"]).pack(side=tk.LEFT, padx=(5, 0))
            ctx_ents[key] = tk.Entry(r1b, width=6); ctx_ents[key].insert(0, str(app.context_size_config.get(key, 4096))); ctx_ents[key].pack(side=tk.LEFT, padx=2)
            tk.Label(r1b, text="Batch:", bg=THEME["bg_color"], fg=THEME["fg_color"]).pack(side=tk.LEFT, padx=(5, 0))
            n_batch_ents[key] = tk.Entry(r1b, width=5); n_batch_ents[key].insert(0, str(app.n_batch_config.get(key, 512))); n_batch_ents[key].pack(side=tk.LEFT, padx=2)

            r2 = tk.Frame(lf, bg=THEME["bg_color"]); r2.pack(fill=tk.X, padx=5)
            for l, d, c, df in [("Temp", temp_ents, app.temp_config, 0.8), ("Top-P", top_p_ents, app.top_p_config, 0.95), ("Min-P", min_p_ents, app.min_p_config, 0.05), ("Top-K", top_k_ents, app.top_k_config, 40)]:
                tk.Label(r2, text=f"{l}:", bg=THEME["bg_color"], fg=THEME["fg_color"], font=("Open Sans", 8)).pack(side=tk.LEFT, padx=(2, 0))
                d[key] = tk.Entry(r2, width=5); d[key].insert(0, f"{c.get(key, df):g}"); d[key].pack(side=tk.LEFT, padx=2)
                
            r2b = tk.Frame(lf, bg=THEME["bg_color"]); r2b.pack(fill=tk.X, padx=5)
            for l, d, c, df in [("Rep", rep_ents, app.repeat_penalty_config, 1.1), ("Freq", freq_ents, app.frequency_penalty_config, 0.0), ("Pres", pres_ents, app.presence_penalty_config, 0.0)]:
                tk.Label(r2b, text=f"{l}:", bg=THEME["bg_color"], fg=THEME["fg_color"], font=("Open Sans", 8)).pack(side=tk.LEFT, padx=(2, 0))
                d[key] = tk.Entry(r2b, width=5); d[key].insert(0, f"{c.get(key, df):g}"); d[key].pack(side=tk.LEFT, padx=2)

            r3 = tk.Frame(lf, bg=THEME["bg_color"]); r3.pack(fill=tk.X, padx=5)
            tk.Label(r3, text="Stop:", bg=THEME["bg_color"], fg=THEME["fg_color"], font=("Open Sans", 8)).pack(side=tk.LEFT)
            stop_ents[key] = tk.Entry(r3, font=("Open Sans", 8), bg=THEME["widget_bg_color"], fg=THEME["fg_color"], width=50)
            stop_ents[key].insert(0, app.stop_strings_config.get(key, "")); stop_ents[key].pack(side=tk.LEFT, padx=5)
            
            if is_vision:
                pk = f"{key}_projector"
                r4 = tk.Frame(lf, bg=THEME["bg_color"]); r4.pack(fill=tk.X, padx=5, pady=2)
                tk.Button(r4, text="Projector", command=lambda k=pk: app._set_path(k, labels, win, True)).pack(side=tk.LEFT)
                labels[pk] = tk.Label(r4, text=os.path.basename(app.model_paths.get(pk, "") or "Not Set"), bg=THEME["bg_color"], fg=THEME["fg_color"], font=("Open Sans", 8))
                labels[pk].pack(side=tk.LEFT, padx=5)
            _bind_click(lf)

        media_frame = tk.Frame(main, bg=THEME["bg_color"])
        media_frame.pack(fill=tk.X, pady=10)
        tk.Label(media_frame, text="Rich Media Rendering:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=("Open Sans", 10, "bold")).pack(side=tk.LEFT, padx=10)
        media_var = tk.IntVar(value=app.config.get("media_rendering", 1))
        for v, t in [(0, "None"), (1, "Inline"), (2, "Popup")]:
            tk.Radiobutton(media_frame, text=t, variable=media_var, value=v, bg=THEME["bg_color"], fg=THEME["fg_color"], selectcolor=THEME["widget_bg_color"]).pack(side=tk.LEFT, padx=5)

        tier_grid = tk.Frame(main, bg=THEME["bg_color"])
        tier_grid.pack(fill=tk.X, pady=10)
        tier_grid.grid_columnconfigure(0, weight=1); tier_grid.grid_columnconfigure(1, weight=1)
        tiers = ["fast", "search", "low", "med", "high", "secret", "Live", "deep_cook"]
        for i, tier in enumerate(tiers):
            r, c = divmod(i, 2)
            _create_tier_block(tier_grid, tier, r, c)

        over_lf = tk.LabelFrame(tier_grid, text="Global Overrides", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=("Open Sans", 10, "bold"), pady=5)
        over_lf.grid(row=4, column=0, columnspan=2, sticky="nsew", padx=10, pady=5)
        
        k_cache_var = tk.StringVar(value=app.config.get("k_cache_type", "q8_0"))
        v_cache_var = tk.StringVar(value=app.config.get("v_cache_type", "q4_0"))
        
        kv_frame = tk.Frame(over_lf, bg=THEME["bg_color"])
        kv_frame.pack(anchor="w", padx=10, pady=5)
        
        tk.Label(kv_frame, text="K Cache Format:", bg=THEME["bg_color"], fg=THEME["electric_blue"]).grid(row=0, column=0, sticky="w", pady=2)
        k_cache_dropdown = ttk.Combobox(kv_frame, textvariable=k_cache_var, values=["fp16", "q8_0", "q6_0", "q5_1", "q5_0", "q4_1", "q4_0", "turbo3_tcq", "turbo2_tcq"], state="readonly", width=12)
        k_cache_dropdown.grid(row=0, column=1, padx=5, pady=2)
        
        tk.Label(kv_frame, text="V Cache Format:", bg=THEME["bg_color"], fg=THEME["electric_blue"]).grid(row=1, column=0, sticky="w", pady=2)
        v_cache_dropdown = ttk.Combobox(kv_frame, textvariable=v_cache_var, values=["fp16", "q8_0", "q6_0", "q5_1", "q5_0", "q4_1", "q4_0", "turbo3_tcq", "turbo2_tcq"], state="readonly", width=12)
        v_cache_dropdown.grid(row=1, column=1, padx=5, pady=2)

        tk.Label(kv_frame, text="History Lookup:", bg=THEME["bg_color"], fg=THEME["electric_blue"]).grid(row=2, column=0, sticky="w", pady=2)
        history_lookup_var = tk.StringVar(value=app.config.get("history_lookup_mode", "targeted"))
        history_lookup_dropdown = ttk.Combobox(kv_frame, textvariable=history_lookup_var, values=["targeted", "model", "level", "all"], state="readonly", width=12)
        history_lookup_dropdown.grid(row=2, column=1, padx=5, pady=2)

        tk.Label(kv_frame, text="History Usage:", bg=THEME["bg_color"], fg=THEME["electric_blue"]).grid(row=3, column=0, sticky="w", pady=2)
        history_usage_var = tk.StringVar(value=app.config.get("history_usage", "all"))
        history_usage_dropdown = ttk.Combobox(kv_frame, textvariable=history_usage_var, values=["all", "current_window", "off"], state="readonly", width=12)
        history_usage_dropdown.grid(row=3, column=1, padx=5, pady=2)

        hao_var = tk.StringVar(value=app.config.get("hao_preset", "exps=CPU"))
        tk.Label(over_lf, text="HAO Preset:", bg=THEME["bg_color"], fg=THEME["electric_blue"]).pack(anchor="w", padx=5, pady=(5,0))
        hao_f = tk.Frame(over_lf, bg=THEME["bg_color"]); hao_f.pack(anchor="w", padx=10)
        for o in ["None", "exps=CPU"]:
            tk.Radiobutton(hao_f, text=o, variable=hao_var, value=o, 
                           bg=THEME["widget_bg_color"], fg=THEME["fg_color"],
                           selectcolor=THEME["electric_blue"], indicatoron=False,
                           activebackground=THEME["electric_blue"], width=10).pack(side=tk.LEFT, padx=2)

        swa_var = tk.StringVar(value=app.config.get("swa_kv_cache", "Auto"))
        tk.Label(over_lf, text="SWA Offload:", bg=THEME["bg_color"], fg=THEME["electric_blue"]).pack(anchor="w", padx=5, pady=(5,0))
        swa_f = tk.Frame(over_lf, bg=THEME["bg_color"]); swa_f.pack(anchor="w", padx=10)
        for o in ["Auto", "CPU Only"]:
            tk.Radiobutton(swa_f, text=o, variable=swa_var, value=o, 
                           bg=THEME["widget_bg_color"], fg=THEME["fg_color"],
                           selectcolor=THEME["electric_blue"], indicatoron=False,
                           activebackground=THEME["electric_blue"], width=10).pack(side=tk.LEFT, padx=2)

        stream_var = tk.StringVar(value=app.state.get("streaming_mode", "Buffered"))
        tk.Label(over_lf, text="Streaming Behavior:", bg=THEME["bg_color"], fg=THEME["electric_blue"]).pack(anchor="w", padx=5, pady=(5,0))
        stream_f = tk.Frame(over_lf, bg=THEME["bg_color"]); stream_f.pack(anchor="w", padx=10)
        for o in ["Real-time", "Buffered", "Experimental Chunking", "Mass Dump"]:
            tk.Radiobutton(stream_f, text=o, variable=stream_var, value=o, 
                           bg=THEME["widget_bg_color"], fg=THEME["fg_color"],
                           selectcolor=THEME["electric_blue"], indicatoron=False,
                           activebackground=THEME["electric_blue"], width=0).pack(side=tk.LEFT, padx=5, pady=2)

        ratio_var = tk.IntVar(value=app.config.get("max_token_ratio", 4))
        tk.Label(over_lf, text="Response Headroom (ctx/N):", bg=THEME["bg_color"], fg=THEME["electric_blue"]).pack(anchor="w", padx=5, pady=(5,0))
        ratio_f = tk.Frame(over_lf, bg=THEME["bg_color"]); ratio_f.pack(anchor="w", padx=10)
        for val, lbl in [(16, "U-Fast (16)"), (8, "Fast (8)"), (4, "Balanced (4)"), (2, "Deep (2)")]:
            tk.Radiobutton(ratio_f, text=lbl, variable=ratio_var, value=val, 
                           bg=THEME["widget_bg_color"], fg=THEME["fg_color"],
                           selectcolor=THEME["electric_blue"], indicatoron=False,
                           activebackground=THEME["electric_blue"], width=0).pack(side=tk.LEFT, padx=5, pady=2)


        tk.Label(main, text="Vision Engines:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=("Open Sans", 10, "bold")).pack(anchor="w", padx=10, pady=(15, 5))
        v_grid = tk.Frame(main, bg=THEME["bg_color"]); v_grid.pack(fill=tk.X, pady=5)
        v_grid.grid_columnconfigure(0, weight=1); v_grid.grid_columnconfigure(1, weight=1)
        for i, vt in enumerate(["video", "video_deep", "multimodal"]):
            r, c = divmod(i, 2)
            _create_tier_block(v_grid, vt, r, c, True)

        def _save():
            app.config["media_rendering"] = media_var.get()
            app.state["deep_cook_behavior"] = v_behavior.get()
            if app.state["deep_cook_behavior"] == "oneshot":
                app.state["deep_cook"] = False
            app._sync_deep_cook_ui()
            app.config["glitch_animation"] = glitch_anim_var.get()
            app.config["k_cache_type"] = k_cache_var.get()
            app.config["v_cache_type"] = v_cache_var.get()
            app.config["history_lookup_mode"] = history_lookup_var.get()
            app.config["history_usage"] = history_usage_var.get()
            app.config["ghost_mode"] = ghost_var.get()
            if hasattr(app, 'ghost_button') and app.ghost_button:
                app.ghost_button.config(text=app._get_ghost_mode_label(), fg=app._get_ghost_mode_color())
            if hasattr(app, 'history_usage_button') and app.history_usage_button:
                app.history_usage_button.config(text=app._get_history_usage_label(), fg=app._get_history_usage_color())
            if getattr(app, 'turbo_vec', None):
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
            app.config["speculative_drafting"] = spec_draft_var.get()
            app.config["ghost_mode"] = ghost_var.get()
            app.config["thinking_checkbox"] = thinking_var.get()
            app.config["benchmark_enabled"] = benchmark_var.get()
            app.config["inline_markdown"] = inline_md_var.get()
            app.config["monitor_graph_mode"] = monitor_graph_var.get()
            app.state["streaming_mode"] = stream_var.get()
            app.config["max_token_ratio"] = ratio_var.get()
            app.config["image_handling"] = image_handling_var.get()
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
            for t, e in stop_ents.items():
                app.stop_strings_config[t] = e.get()
            app.save_config()
            messagebox.showinfo("Success", "Settings saved!")
            win.destroy()

        tk.Button(btn_frame, text="Save & Apply", command=_save, bg=THEME["button_active_color"], fg=THEME["fg_color"]).pack(side=tk.RIGHT, padx=5)
        tk.Button(btn_frame, text="Clear History", command=app.clear_current_history, bg="#660000", fg="white").pack(side=tk.RIGHT, padx=5)
        
        def _reset_defaults():
            if messagebox.askyesno("Reset", "Restore system defaults for all layers and samplers?"):
                recs = run_auto_detect(app, win)
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
        try:
            os.makedirs("Logs", exist_ok=True)
            with open("Logs/ui_crash.txt", "w") as f: f.write(err_msg)
        except: pass
        messagebox.showerror("UI Error", f"Settings window failed to open:\n{e}")
