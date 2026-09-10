"""
settings_tabs.py
Tab view constructors for SerenityPC Settings Window Reorganization.
Provides modular, clean builders for all 6 settings tabs:
1. Models & Params (Balanced columns, parameter inputs, templates below, auto-tune)
2. Inference (HAO-style push-radios, Overfill Behavior, separate Halt options, History mode)
3. Agents (Single column delegation pipeline)
4. Additional Settings (Offline mode, hardware inputs, and Loading/Status bar)
5. Users & Security (User profile switcher, logout, and Secure Vault panel)
6. Personalize (Theme previews, side-by-side fonts, texture finish, dark mode, scaling)
"""

import os
import json
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from serenity_resources import THEME, THEMES
try:
    from System.serenity_utils import ToolTip, bind_entry_limit
except ImportError:
    from serenity_utils import ToolTip, bind_entry_limit


def bind_radio_contrast(var, rb_list):
    """Syncs high-contrast text color on active push-radio selection."""
    def _sync(*args):
        val = var.get()
        for rb, item_val in rb_list:
            if rb.winfo_exists():
                rb.config(fg="#000000" if val == item_val else THEME.get("fg_color", "#ffaa44"))
    var.trace_add("write", _sync)
    _sync()


def build_models_tab(parent, app, win, vars_dict, generation_sensitive_widgets):
    """
    Builds Tab 1: Models & Params.
    - Model selection up top with balanced columns (uniform column weights, shortened label).
    - Parameter selections (Layers, Ctx, Batch, Temp, Top-P, Min-P, Top-K, Rep, Freq, Pres, Stop).
    - Dynamic Param Auto-Tune moved here.
    - Templating Engine moved below with 8x4 slots, visual active slot indication,
      and write mode auto-switching back to 'modify'.
    """
    frame = tk.Frame(parent, bg=THEME["bg_color"])

    labels = vars_dict["labels"]
    ents = vars_dict["ents"]
    ctx_ents = vars_dict["ctx_ents"]
    n_batch_ents = vars_dict["n_batch_ents"]
    temp_ents = vars_dict["temp_ents"]
    top_p_ents = vars_dict["top_p_ents"]
    min_p_ents = vars_dict["min_p_ents"]
    top_k_ents = vars_dict["top_k_ents"]
    rep_ents = vars_dict["rep_ents"]
    freq_ents = vars_dict["freq_ents"]
    pres_ents = vars_dict["pres_ents"]
    stop_ents = vars_dict["stop_ents"]
    active_template = vars_dict["active_template"]
    template_mode = vars_dict["template_mode"]

    def _apply_template_to_tier(tier_name):
        t_id = active_template.get()
        if not t_id:
            messagebox.showwarning("Templating", "Select a template slot first!", parent=win)
            return
        t_data = app.config.get("custom_templates", {}).get(t_id, {})
        if not t_data:
            messagebox.showwarning("Templating", f"Slot {t_id} is empty.", parent=win)
            return
        for k, d in [("temp", temp_ents), ("top_p", top_p_ents), ("min_p", min_p_ents),
                     ("rep", rep_ents), ("pres", pres_ents), ("freq", freq_ents),
                     ("top_k", top_k_ents), ("batch", n_batch_ents), ("layers", ents),
                     ("ctx", ctx_ents), ("stop", stop_ents)]:
            if k in t_data and tier_name in d:
                d[tier_name].delete(0, tk.END)
                d[tier_name].insert(0, str(t_data[k]))
        template_mode.set("modify")
        messagebox.showinfo("Templating", f"Applied {t_data.get('name', t_id)} to {tier_name.upper()}!\nMode returned to Modify.", parent=win)
        try: win.lift()
        except Exception: pass

    def _create_tier_block(p, tier_name, row=0, col=0, is_vision=False):
        key = f"vision_{tier_name}" if is_vision else tier_name
        lvl_map = {"fast": "1", "search": "2", "low": "3", "med": "4", "high": "5", "transcendent": "6", "secret": "7"}
        title_suffix = f" (Lvl {lvl_map[tier_name]})" if tier_name in lvl_map else ""
        lf = tk.LabelFrame(p, text=f"Engine: {tier_name.upper()}{title_suffix}", bg=THEME["bg_color"],
                           fg=THEME["electric_blue"], font=app.fonts["bold"], pady=4)
        lf.grid(row=row, column=col, sticky="nsew", padx=6, pady=4)

        # Row 1: Set Path & Label
        r1 = tk.Frame(lf, bg=THEME["bg_color"])
        r1.pack(fill=tk.X, padx=5, pady=1)
        btn_p = tk.Button(r1, text="Set Path", command=lambda t=key: app._set_path(t, labels, win),
                          bg=THEME["widget_bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_small"])
        btn_p.pack(side=tk.LEFT)
        generation_sensitive_widgets.append(btn_p)

        full_p = app.model_paths.get(key, "") or "Not Set"
        b_name = os.path.basename(full_p)
        disp_txt = b_name if len(b_name) <= 22 else b_name[:11] + "..." + b_name[-8:]
        lbl = tk.Label(r1, text=disp_txt, bg=THEME["bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_small"], anchor="w")
        lbl.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)
        labels[key] = lbl
        ToolTip(lbl, f"Full Path: {full_p}", app=app)

        # Copy Template Button for Write mode
        btn_copy = tk.Button(r1, text="📋 Copy", command=lambda t=key: _apply_template_to_tier(t),
                             bg=THEME["widget_bg_color"], fg=THEME.get("accent_highlight", "#00ffcc"),
                             font=app.fonts["ui_small"], padx=3, pady=0)
        btn_copy.pack(side=tk.RIGHT)
        ToolTip(btn_copy, f"Copy selected template values into {tier_name.upper()}.", app=app)

        # Row 1b: Hardware Allocations
        r1b = tk.Frame(lf, bg=THEME["bg_color"])
        r1b.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(r1b, text="Layers:", bg=THEME["bg_color"], fg=THEME["fg_color"]).pack(side=tk.LEFT)
        ents[key] = tk.Entry(r1b, width=4, bg=THEME["widget_bg_color"], fg=THEME["fg_color"],
                             insertbackground=THEME.get("electric_blue", THEME["fg_color"]))
        bind_entry_limit(ents[key], max_len=5)
        ents[key].insert(0, str(app.gpu_layer_config.get(key, -1)))
        ents[key].pack(side=tk.LEFT, padx=2)
        generation_sensitive_widgets.append(ents[key])

        tk.Label(r1b, text="Ctx:", bg=THEME["bg_color"], fg=THEME["fg_color"]).pack(side=tk.LEFT, padx=(4, 0))
        ctx_ents[key] = tk.Entry(r1b, width=6, bg=THEME["widget_bg_color"], fg=THEME["fg_color"],
                                 insertbackground=THEME.get("electric_blue", THEME["fg_color"]))
        bind_entry_limit(ctx_ents[key], max_len=8)
        ctx_ents[key].insert(0, str(app.context_size_config.get(key, 4096)))
        ctx_ents[key].pack(side=tk.LEFT, padx=2)
        generation_sensitive_widgets.append(ctx_ents[key])

        tk.Label(r1b, text="Batch:", bg=THEME["bg_color"], fg=THEME["fg_color"]).pack(side=tk.LEFT, padx=(4, 0))
        n_batch_ents[key] = tk.Entry(r1b, width=5, bg=THEME["widget_bg_color"], fg=THEME["fg_color"],
                                     insertbackground=THEME.get("electric_blue", THEME["fg_color"]))
        bind_entry_limit(n_batch_ents[key], max_len=6)
        n_batch_ents[key].insert(0, str(app.n_batch_config.get(key, 512)))
        n_batch_ents[key].pack(side=tk.LEFT, padx=2)
        generation_sensitive_widgets.append(n_batch_ents[key])

        # Row 2: Samplers 1
        r2 = tk.Frame(lf, bg=THEME["bg_color"])
        r2.pack(fill=tk.X, padx=5, pady=1)
        for l, d, c, df in [("Temp", temp_ents, app.temp_config, 0.8), ("Top-P", top_p_ents, app.top_p_config, 0.95),
                            ("Min-P", min_p_ents, app.min_p_config, 0.05), ("Top-K", top_k_ents, app.top_k_config, 40)]:
            tk.Label(r2, text=f"{l}:", bg=THEME["bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_small"]).pack(side=tk.LEFT, padx=(2, 0))
            d[key] = tk.Entry(r2, width=5, bg=THEME["widget_bg_color"], fg=THEME["fg_color"],
                              insertbackground=THEME.get("electric_blue", THEME["fg_color"]))
            bind_entry_limit(d[key], max_len=8)
            d[key].insert(0, f"{c.get(key, df):g}")
            d[key].pack(side=tk.LEFT, padx=1)

        # Row 2b: Samplers 2
        r2b = tk.Frame(lf, bg=THEME["bg_color"])
        r2b.pack(fill=tk.X, padx=5, pady=1)
        for l, d, c, df in [("Rep", rep_ents, app.repeat_penalty_config, 1.1),
                            ("Freq", freq_ents, app.frequency_penalty_config, 0.0),
                            ("Pres", pres_ents, app.presence_penalty_config, 0.0)]:
            tk.Label(r2b, text=f"{l}:", bg=THEME["bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_small"]).pack(side=tk.LEFT, padx=(2, 0))
            d[key] = tk.Entry(r2b, width=5, bg=THEME["widget_bg_color"], fg=THEME["fg_color"],
                              insertbackground=THEME.get("electric_blue", THEME["fg_color"]))
            bind_entry_limit(d[key], max_len=8)
            d[key].insert(0, f"{c.get(key, df):g}")
            d[key].pack(side=tk.LEFT, padx=1)

        if is_vision:
            pk = f"{key}_projector"
            r4 = tk.Frame(lf, bg=THEME["bg_color"])
            r4.pack(fill=tk.X, padx=5, pady=2)
            btn_proj = tk.Button(r4, text="Projector", command=lambda k=pk: app._set_path(k, labels, win, True),
                                 bg=THEME["widget_bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_small"])
            btn_proj.pack(side=tk.LEFT)
            generation_sensitive_widgets.append(btn_proj)
            p_name = os.path.basename(app.model_paths.get(pk, "") or "Not Set")
            p_disp = p_name if len(p_name) <= 22 else p_name[:11] + "..." + p_name[-8:]
            labels[pk] = tk.Label(r4, text=p_disp, bg=THEME["bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_small"], anchor="w")
            labels[pk].pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)

    # 1. Text & Inline Engines
    tk.Label(frame, text="Text & Inline Engines:", bg=THEME["bg_color"], fg=THEME["electric_blue"],
             font=app.fonts["bold"]).pack(anchor="w", padx=6, pady=(6, 2))
    tier_grid = tk.Frame(frame, bg=THEME["bg_color"])
    tier_grid.pack(fill=tk.X, padx=4, pady=2)
    tier_grid.grid_columnconfigure(0, weight=1, uniform="model_col")
    tier_grid.grid_columnconfigure(1, weight=1, uniform="model_col")

    tiers = ["fast", "search", "low", "med", "high", "transcendent", "secret", "deep_cook"]
    for i, tier in enumerate(tiers):
        r, c = divmod(i, 2)
        _create_tier_block(tier_grid, tier, r, c)

    # 2. Vision Engines
    tk.Label(frame, text="Vision Engines:", bg=THEME["bg_color"], fg=THEME["electric_blue"],
             font=app.fonts["bold"]).pack(anchor="w", padx=6, pady=(10, 2))
    v_grid = tk.Frame(frame, bg=THEME["bg_color"])
    v_grid.pack(fill=tk.X, padx=4, pady=2)
    v_grid.grid_columnconfigure(0, weight=1, uniform="model_col")
    v_grid.grid_columnconfigure(1, weight=1, uniform="model_col")
    for i, vt in enumerate(["video", "video_deep", "multimodal"]):
        r, c = divmod(i, 2)
        _create_tier_block(v_grid, vt, r, c, True)

    # 3. Dynamic Param Auto-Tune (Moved from Inference)
    auto_tune_f = tk.Frame(frame, bg=THEME["bg_color"])
    auto_tune_f.pack(fill=tk.X, padx=8, pady=(8, 4))
    cb_dyn = tk.Checkbutton(auto_tune_f, text="Dynamic Param Auto-Tune (Coding/Math/Creative)",
                            variable=vars_dict["dynamic_params_var"], bg=THEME["bg_color"],
                            fg=THEME["electric_blue"], selectcolor=THEME["widget_bg_color"],
                            font=app.fonts["ui_label"])
    cb_dyn.pack(anchor="w")
    ToolTip(cb_dyn, "Automatically optimizes sampling temperature and top-p when coding, math, or creative writing intent is detected.", app=app)

    # 4. Templating Engine (Moved Below)
    templ_lf = tk.LabelFrame(frame, text="Templating Engine (32 Parameter Slots)", bg=THEME["bg_color"],
                             fg=THEME["electric_blue"], font=app.fonts["bold"], padx=8, pady=6)
    templ_lf.pack(fill=tk.X, padx=6, pady=(10, 10))

    t_mode_f = tk.Frame(templ_lf, bg=THEME["bg_color"])
    t_mode_f.pack(anchor="w", pady=(0, 4))
    tk.Label(t_mode_f, text="Action Mode:", bg=THEME["bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_label"]).pack(side=tk.LEFT, padx=(0, 6))

    t_action_rbs = []
    for val, txt in [("save", "Save"), ("write", "Write"), ("modify", "Modify")]:
        rb = tk.Radiobutton(t_mode_f, text=txt, variable=template_mode, value=val, indicatoron=False,
                            bg=THEME["widget_bg_color"], fg=THEME["fg_color"], selectcolor=THEME["electric_blue"],
                            activebackground=THEME["electric_blue"], activeforeground="#000000",
                            width=8, font=app.fonts["ui_small"])
        rb.pack(side=tk.LEFT, padx=3)
        ToolTip(rb, f"Templating action mode: {txt} tier settings.", app=app)
        t_action_rbs.append((rb, val))
    bind_radio_contrast(template_mode, t_action_rbs)
    vars_dict["t_action_rbs"] = t_action_rbs

    lbl_slot_info = tk.Label(templ_lf, text="Select slot to inspect or apply with '📋 Copy' button:",
                             bg=THEME["bg_color"], fg=THEME.get("accent_highlight", "#00ffcc"), font=app.fonts["ui_small"])
    lbl_slot_info.pack(anchor="w", pady=(2, 4))

    t_grid = tk.Frame(templ_lf, bg=THEME["bg_color"])
    t_grid.pack(anchor="center", pady=4)

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
            ToolTip(b, f"Template Slot {slot_id} ({t_name}). Click to select.", app=app)
            template_buttons.append(b)
            t_slot_rbs.append((b, slot_id))

    bind_radio_contrast(active_template, t_slot_rbs)
    vars_dict["t_slot_rbs"] = t_slot_rbs

    def _on_template_click(*args):
        mode = template_mode.get()
        t_id = active_template.get()
        if not t_id: return
        cur_name = app.config.get("custom_templates", {}).get(t_id, {}).get("name", t_id)
        lbl_slot_info.config(text=f"Selected: {t_id} ({cur_name}) | Mode: {mode.upper()} — Press '📋 Copy' on any engine above to apply.")
        if mode == "modify":
            _open_template_modify_dialog(win, app, t_id, template_buttons, template_mode)
    active_template.trace_add("write", _on_template_click)

    return frame


def _open_template_modify_dialog(parent_win, app, t_id, template_buttons, template_mode=None):
    """Opens modal to edit template slot parameters with mode-switch actions."""
    t_win = tk.Toplevel(parent_win)
    t_win.title(f"Modify Template {t_id}")
    t_win.geometry("340x520")
    t_win.config(bg=THEME["bg_color"])
    t_win.transient(parent_win)

    current = app.config.get("custom_templates", {}).get(t_id, {})
    tk.Label(t_win, text=f"Template {t_id} Name:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["bold"]).pack(anchor="w", padx=10, pady=(10, 0))
    name_ent = tk.Entry(t_win, bg=THEME["widget_bg_color"], fg=THEME["fg_color"],
                        insertbackground=THEME.get("electric_blue", THEME["fg_color"]))
    bind_entry_limit(name_ent, max_len=24)
    name_ent.insert(0, current.get("name", t_id))
    name_ent.pack(fill=tk.X, padx=10, pady=2)

    param_list = [("Temp:", "temp", 0.8), ("Top P:", "top_p", 0.9), ("Min P:", "min_p", 0.05),
                  ("Rep Pen:", "rep", 1.1), ("Pres Pen:", "pres", 0.0), ("Freq Pen:", "freq", 0.0),
                  ("Top K:", "top_k", 40), ("Batch:", "batch", 512), ("Layers:", "layers", -1), ("Ctx Size:", "ctx", 8192)]
    fields = {}
    grid_f = tk.Frame(t_win, bg=THEME["bg_color"])
    grid_f.pack(fill=tk.X, padx=10, pady=6)
    for idx, (label, key, default) in enumerate(param_list):
        r, c = divmod(idx, 2)
        c *= 2
        tk.Label(grid_f, text=label, bg=THEME["bg_color"], fg=THEME["electric_blue"], width=8, anchor="w").grid(row=r, column=c, padx=(0, 2), pady=2)
        e = tk.Entry(grid_f, bg=THEME["widget_bg_color"], fg=THEME["fg_color"], width=8,
                     insertbackground=THEME.get("electric_blue", THEME["fg_color"]))
        bind_entry_limit(e, max_len=10)
        e.insert(0, str(current.get(key, default)))
        e.grid(row=r, column=c + 1, padx=(0, 6), pady=2)
        fields[key] = e

    def _save(target_mode=None):
        t_data = {"name": name_ent.get().strip() or t_id}
        for k, e in fields.items():
            try: t_data[k] = float(e.get()) if '.' in e.get() else int(e.get())
            except Exception: t_data[k] = current.get(k, 0)
        if "custom_templates" not in app.config: app.config["custom_templates"] = {}
        app.config["custom_templates"][t_id] = t_data
        for btn in template_buttons:
            if getattr(btn, "slot_id", None) == t_id:
                btn.config(text=t_data["name"])
        app.save_config()
        if target_mode and template_mode:
            template_mode.set(target_mode)
        t_win.destroy()

    btn_row = tk.Frame(t_win, bg=THEME["bg_color"])
    btn_row.pack(pady=12)
    tk.Button(btn_row, text="Save & Write", command=lambda: _save("write"),
              bg=THEME["widget_bg_color"], fg=THEME.get("accent_highlight", "#00ffcc"), font=app.fonts["ui_small"]).pack(side=tk.LEFT, padx=3)
    tk.Button(btn_row, text="Save & Save", command=lambda: _save("save"),
              bg=THEME["widget_bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_small"]).pack(side=tk.LEFT, padx=3)
    tk.Button(btn_row, text="Save & Close", command=lambda: _save("modify"),
              bg=THEME["button_active_color"], fg=THEME["fg_color"], font=app.fonts["ui_small"]).pack(side=tk.LEFT, padx=3)


def build_inference_tab(parent, app, win, vars_dict):
    """
    Builds Tab 2: Inference.
    - Two balanced columns.
    - Standardized HAO push-radio button style across entire tab.
    - Column 1: HAO, Repeat Loop Detection, Response Headroom, Overfill Behavior,
      separate Halt Options, Streaming Behavior, Deep Cook Toggle.
    - Column 2: SWA, K Cache, V Cache, History Mode, History Usage, History Lookup, Muse Reasoning.
    """
    frame = tk.Frame(parent, bg=THEME["bg_color"])

    grid_f = tk.Frame(frame, bg=THEME["bg_color"])
    grid_f.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
    grid_f.grid_columnconfigure(0, weight=1, uniform="inf_col")
    grid_f.grid_columnconfigure(1, weight=1, uniform="inf_col")

    col1 = tk.Frame(grid_f, bg=THEME["bg_color"])
    col1.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
    col2 = tk.Frame(grid_f, bg=THEME["bg_color"])
    col2.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

    radio_groups = []

    def _make_push_radios(p, title, tooltip, var, options, width=10):
        lbl = tk.Label(p, text=title, bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["ui_label"])
        lbl.pack(anchor="w", pady=(6, 1))
        ToolTip(lbl, tooltip, app=app)
        row_f = tk.Frame(p, bg=THEME["bg_color"])
        row_f.pack(anchor="w", pady=(0, 4))
        rbs = []
        for opt in options:
            val, disp = (opt, opt) if isinstance(opt, str) else opt
            rb = tk.Radiobutton(row_f, text=str(disp), variable=var, value=val, indicatoron=False,
                                bg=THEME["widget_bg_color"], fg=THEME["fg_color"], selectcolor=THEME["electric_blue"],
                                activebackground=THEME["electric_blue"], activeforeground="#000000",
                                font=app.fonts["ui_small"], width=width)
            rb.pack(side=tk.LEFT, padx=2, pady=1)
            ToolTip(rb, f"Select {disp}.", app=app)
            rbs.append((rb, val))
        bind_radio_contrast(var, rbs)
        radio_groups.append(rbs)
        return rbs

    # --- COLUMN 1 ---
    # 1. HAO Preset
    _make_push_radios(col1, "HAO Preset:", "Hardware Allocation Optimizer MoE strategy.", vars_dict["hao_var"], ["None", "exps=CPU"], width=11)

    # 2. Repeat Loop Detection
    _make_push_radios(col1, "Repeat Loop Detection:", "Prevent repetitive token generation loops.", vars_dict["repeat_mode_var"], [("hyper", "Hyper"), ("lazy", "Lazy"), ("off", "Off")], width=8)

    # 3. Response Headroom (ctx/N)
    _make_push_radios(col1, "Response Headroom (ctx/N):", "Maximum token generation headroom relative to context.", vars_dict["ratio_var"], [(16, "U-Fast (16)"), (8, "Fast (8)"), (4, "Balanced (4)"), (2, "Deep (2)")], width=11)

    # 4. Overfill Behavior (Natural context budget limit reached)
    overfill_var = vars_dict.get("overfill_behavior_var", vars_dict.get("budget_recovery_var"))
    if overfill_var is None:
        overfill_var = tk.StringVar(value=getattr(app, "config", {}).get("overfill_behavior_mode", "wrapup"))
        vars_dict["overfill_behavior_var"] = overfill_var
    _make_push_radios(col1, "Overfill Behavior (Token Limit):", "Action taken when token context budget limit is reached naturally.", overfill_var, ["off", "respond", "wrapup", "autocont"], width=9)

    # 5. Halt Options (Clicking Halt button behavior)
    halt_var = vars_dict.get("halt_behavior_var")
    if halt_var is None:
        halt_var = tk.StringVar(value=getattr(app, "config", {}).get("halt_behavior_mode", "off"))
        vars_dict["halt_behavior_var"] = halt_var
    _make_push_radios(col1, "Halt Options (Button Action):", "Behavior when clicking the Halt button ('off' triggers immediate EOS cutoff).", halt_var, ["off", "wrapup", "autocont", "respond"], width=9)

    # 6. Streaming Behavior
    _make_push_radios(col1, "Streaming Behavior:", "Token delivery pacing and chunking mode.", vars_dict["stream_var"], ["Real-time", "Buffered", "Experimental Chunking", "Mass Dump"], width=13)

    # 7. Deep Cook Toggle
    lbl_dc = tk.Label(col1, text="Deep Cook Mode:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["ui_label"])
    lbl_dc.pack(anchor="w", pady=(6, 1))
    ToolTip(lbl_dc, "Select Deep Cook recursive cycle behavior: One-Shot trigger or persistent toggle.", app=app)
    dc_f = tk.Frame(col1, bg=THEME["bg_color"])
    dc_f.pack(anchor="w", pady=(0, 4))
    dc_rbs = []
    for val, txt in [("oneshot", "One-Shot"), ("toggle", "Toggle Mode")]:
        rb = tk.Radiobutton(dc_f, text=txt, variable=vars_dict["v_behavior"], value=val, indicatoron=False,
                            bg=THEME["widget_bg_color"], fg=THEME["fg_color"], selectcolor=THEME["electric_blue"],
                            activebackground=THEME["electric_blue"], activeforeground="#000000",
                            font=app.fonts["ui_small"], width=12)
        rb.pack(side=tk.LEFT, padx=2)
        ToolTip(rb, f"Set Deep Cook behavior to {txt}.", app=app)
        dc_rbs.append((rb, val))
    bind_radio_contrast(vars_dict["v_behavior"], dc_rbs)
    radio_groups.append(dc_rbs)

    # --- COLUMN 2 ---
    # 1. SWA Offload
    _make_push_radios(col2, "SWA Offload:", "Sliding Window Attention KV cache offloading.", vars_dict["swa_var"], ["Auto", "CPU Only"], width=10)

    # 2. K Cache Format
    _make_push_radios(col2, "K Cache Format:", "Quantized Key cache format for VRAM savings.", vars_dict["k_cache_var"], ["q8_0", "q5_1", "q4_0", "fp16"], width=8)

    # 3. V Cache Format
    _make_push_radios(col2, "V Cache Format:", "Quantized Value cache format for VRAM savings.", vars_dict["v_cache_var"], ["q8_0", "q5_1", "q4_0", "fp16"], width=8)

    # 4. History Mode
    hist_var = vars_dict.get("history_mode_var", vars_dict.get("turbovec_mode_var"))
    if hist_var is None:
        hist_var = tk.StringVar(value=getattr(app, "config", {}).get("history_mode", "TurboVec"))
        vars_dict["history_mode_var"] = hist_var
    _make_push_radios(col2, "History Mode:", "Conversation history indexing and search engine.", hist_var, ["TurboVec", "Keyword", "Off"], width=9)

    # 5. History Usage
    _make_push_radios(col2, "History Usage Mode:", "Whether past histories are injected into active context.", vars_dict["history_usage_var"], ["all", "current_window", "off"], width=11)

    # 6. History Lookup
    _make_push_radios(col2, "History Lookup Scope:", "Scope of conversation search retrieval.", vars_dict["history_lookup_var"], ["targeted", "model", "level", "all"], width=9)

    # 7. Muse Reasoning
    _make_push_radios(col2, "Muse Reasoning Strength:", "Reasoning effort level for Gemma-4, Muse-Glimmer, and thinking models.", vars_dict["reasoning_var"], ["off", "low", "medium", "high", "xhigh"], width=7)

    vars_dict["inference_radio_groups"] = radio_groups
    return frame


def build_agents_tab(parent, app, win, vars_dict):
    """
    Builds Tab 3: Agents.
    Single column top-to-bottom layout:
    1. Enable toggle
    2. Subagent Model
    3. Subagent Density
    4. Cecilia Mode
    5. Delegation Chain
    6. Handoff Reporting
    """
    frame = tk.Frame(parent, bg=THEME["bg_color"], padx=10, pady=8)

    # 1. Enable Toggle
    cb_enable = tk.Checkbutton(frame, text="Enable Delegation & Subagents (Lvls 6 & 7)",
                               variable=vars_dict["delegation_enabled_var"], bg=THEME["bg_color"],
                               fg=THEME["electric_blue"], selectcolor=THEME["widget_bg_color"],
                               font=app.fonts["bold"])
    cb_enable.pack(anchor="w", pady=(4, 8))
    ToolTip(cb_enable, "Allows Transcendent (Lvl 6) and Cecilia (Lvl 7) to task subagents and orchestrate handoffs.", app=app)

    # 2. Subagent Model
    lbl_m = tk.Label(frame, text="Subagent Model Selection:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["ui_label"])
    lbl_m.pack(anchor="w", pady=(4, 2))
    ToolTip(lbl_m, "Choose whether to run all subagents with the active Lvl 6/7 model or dynamically swap.", app=app)
    m_frame = tk.Frame(frame, bg=THEME["bg_color"])
    m_frame.pack(anchor="w", pady=(0, 6))
    for val, txt in [("lvl6_7_model", "Use model selected for Lvl 6 / 7"), ("per_subagent_model", "Use model selected for specific subagent")]:
        rb = tk.Radiobutton(m_frame, text=txt, variable=vars_dict["delegation_model_mode_var"], value=val,
                            bg=THEME["bg_color"], fg=THEME["fg_color"], selectcolor=THEME["widget_bg_color"], font=app.fonts["ui_small"])
        rb.pack(anchor="w", pady=1)
        ToolTip(rb, f"Engine execution mode: {txt}.", app=app)

    # 3. Subagent Density
    lbl_d = tk.Label(frame, text="Subagent Density / Selection:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["ui_label"])
    lbl_d.pack(anchor="w", pady=(4, 2))
    ToolTip(lbl_d, "Control whether the orchestrator utilizes only the minimum required subagents or engages all desired agents.", app=app)
    d_frame = tk.Frame(frame, bg=THEME["bg_color"])
    d_frame.pack(anchor="w", pady=(0, 6))
    for val, txt in [("minimal", "Uses minimum required subagents"), ("all", "Use as many subagents as you want")]:
        rb = tk.Radiobutton(d_frame, text=txt, variable=vars_dict["subagent_density_var"], value=val,
                            bg=THEME["bg_color"], fg=THEME["fg_color"], selectcolor=THEME["widget_bg_color"], font=app.fonts["ui_small"])
        rb.pack(anchor="w", pady=1)
        ToolTip(rb, f"Configure subagent density to: {txt}.", app=app)

    # 4. Cecilia Mode
    lbl_c = tk.Label(frame, text="Cecilia Mode (Lvl 7):", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["ui_label"])
    lbl_c.pack(anchor="w", pady=(4, 2))
    ToolTip(lbl_c, "Choose between Shadow Wizard (full subagent orchestration) or Divine Judgement (direct omniscience).", app=app)
    c_frame = tk.Frame(frame, bg=THEME["bg_color"])
    c_frame.pack(anchor="w", pady=(0, 6))
    for val, txt in [("shadow_wizard", "Shadow Wizard (Subagent Orchestration)"), ("divine_judgement", "Divine Judgement (Direct Omniscience)")]:
        rb = tk.Radiobutton(c_frame, text=txt, variable=vars_dict["cecilia_mode_var"], value=val,
                            bg=THEME["bg_color"], fg=THEME["fg_color"], selectcolor=THEME["widget_bg_color"], font=app.fonts["ui_small"])
        rb.pack(anchor="w", pady=1)
        ToolTip(rb, f"Set Cecilia operating mode to: {txt}.", app=app)

    # 5. Delegation Chain
    lbl_chain = tk.Label(frame, text="Delegation Chain Preset:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["ui_label"])
    lbl_chain.pack(anchor="w", pady=(4, 2))
    ToolTip(lbl_chain, "Select pipeline order for subagent handoffs.", app=app)
    chain_combo = ttk.Combobox(frame, textvariable=vars_dict["chain_preset_var"],
                               values=["standard (L2 Search -> L3 Store -> L5 Reason -> L6/7 Approve)",
                                       "direct_strike (L2 Search -> L6/7 Approve)"],
                               state="readonly", width=48)
    chain_combo.pack(anchor="w", pady=(0, 6))
    ToolTip(chain_combo, "Select pipeline order for subagent handoffs.", app=app)

    # 6. Handoff Reporting
    lbl_hand = tk.Label(frame, text="Handoff Reporting Target:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["ui_label"])
    lbl_hand.pack(anchor="w", pady=(4, 2))
    ToolTip(lbl_hand, "Where subagents report their intermediary outputs.", app=app)
    hand_combo = ttk.Combobox(frame, textvariable=vars_dict["handoff_target_var"],
                              values=["lvl3_compiler (Staging & Aggregation)", "taskmaster_direct (Direct to Lvl 6/7)"],
                              state="readonly", width=48)
    hand_combo.pack(anchor="w", pady=(0, 8))
    ToolTip(hand_combo, "Select intermediate results staging destination.", app=app)

    return frame


def build_additional_tab(parent, app, win, vars_dict):
    """
    Builds Tab 4: Additional Settings.
    Single column layout:
    - Checkboxes from Offline Mode to Enable ToolTips.
    - Configurable response target length.
    - Hardware & Multimedia controls.
    - Status Bar (Loading Bar & Status Area) in one dedicated section.
    """
    frame = tk.Frame(parent, bg=THEME["bg_color"], padx=10, pady=8)

    # --- Section 1: System Toggles ---
    lbl_togs = tk.Label(frame, text="System & Interface Toggles:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["bold"])
    lbl_togs.pack(anchor="w", pady=(2, 4))

    tog_grid = tk.Frame(frame, bg=THEME["bg_color"])
    tog_grid.pack(fill=tk.X, pady=2)
    tog_grid.grid_columnconfigure(0, weight=1)
    tog_grid.grid_columnconfigure(1, weight=1)

    scroll_lock_var = vars_dict.get("scroll_lock_var")
    if scroll_lock_var is None:
        scroll_lock_var = tk.BooleanVar(value=app.config.get("scroll_lock_enabled", False))
        vars_dict["scroll_lock_var"] = scroll_lock_var

    sb_linger_var = vars_dict.get("sb_linger_var")
    if sb_linger_var is None:
        sb_linger_var = tk.DoubleVar(value=float(app.config.get("status_bar_linger_sec", 5.0)))
        vars_dict["sb_linger_var"] = sb_linger_var

    format_prompt_md_var = vars_dict.get("format_prompt_md_var")
    if format_prompt_md_var is None:
        format_prompt_md_var = tk.BooleanVar(value=app.config.get("format_prompt_markdown", False))
        vars_dict["format_prompt_md_var"] = format_prompt_md_var

    toggles_list = [
        ("Offline Mode (Block Net)", vars_dict["offline_mode_var"], "Blocks all outbound internet traffic while allowing local loopback."),
        ("Dynamic Auto-Offload", vars_dict["auto_vram_var"], "Automatically flushes inactive model layers from VRAM to prevent memory exhaustion."),
        ("Speculative MTP Drafting (>= 4GB Models)", vars_dict["spec_draft_var"], "Accelerates token generation using assistant drafter speculative decoding (automatically bypassed for models < 4GB)."),
        ("Ghost Mode", vars_dict["ghost_var"], "Disables chat history persistence to disk for private sessions."),
        ("Thinking Process", vars_dict["thinking_var"], "Controls whether internal model thought logs and reasoning blocks are captured."),
        ("Loading Benchmark", vars_dict["benchmark_var"], "Runs a quick memory throughput benchmark upon model initialization."),
        ("Inline Markdown", vars_dict["inline_md_var"], "Enables real-time formatting for bold, italics, tables, and math equations."),
        ("Format Prompts Markdown", vars_dict["format_prompt_md_var"], "Enables markdown formatting for user prompts. Disabled by default to preserve math notation (e.g. 3*3*5*5)."),
        ("Monitor Graph vs Line", vars_dict["monitor_graph_var"], "Switches hardware telemetry display between graphs and text lines."),
        ("Scroll Lock to Lines of Text", vars_dict["scroll_lock_var"], "Locks chat viewport strictly to latest lines of text during generation."),
        ("Enable Hover Tooltips / Help", vars_dict["show_tooltips_var"], "Displays helpful linger-hover information boxes across UI controls.")
    ]

    for idx, (txt, var, tip) in enumerate(toggles_list):
        r, c = divmod(idx, 2)
        cb = tk.Checkbutton(tog_grid, text=txt, variable=var, bg=THEME["bg_color"],
                            fg="#ff8800" if "Offline" in txt else THEME["fg_color"],
                            selectcolor=THEME["widget_bg_color"], font=app.fonts["ui_small"])
        cb.grid(row=r, column=c, sticky="w", padx=6, pady=2)
        ToolTip(cb, tip, app=app)

    # Target Response Length
    lbl_resp = tk.Label(frame, text="Configurable Target Response Length:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["ui_label"])
    lbl_resp.pack(anchor="w", pady=(8, 2))
    ToolTip(lbl_resp, "Configurable target response length (natural uses proportional conversational pacing).", app=app)
    resp_f = tk.Frame(frame, bg=THEME["bg_color"])
    resp_f.pack(anchor="w", pady=(0, 6))
    for opt in ["natural", "mini", "short", "medium", "long", "matched"]:
        rb = tk.Radiobutton(resp_f, text=opt.capitalize(), variable=vars_dict["resp_len_var"], value=opt,
                            bg=THEME["bg_color"], fg=THEME["fg_color"], selectcolor=THEME["widget_bg_color"], font=app.fonts["ui_small"])
        rb.pack(side=tk.LEFT, padx=3)
        ToolTip(rb, f"Set target response length to {opt}.", app=app)

    # --- Section 2: Hardware & Media ---
    lbl_hw = tk.Label(frame, text="Hardware, Multimedia & Projector:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["bold"])
    lbl_hw.pack(anchor="w", pady=(10, 4))

    hw_f = tk.Frame(frame, bg=THEME["bg_color"])
    hw_f.pack(fill=tk.X, pady=2)

    # Rich Media
    r_med = tk.Frame(hw_f, bg=THEME["bg_color"])
    r_med.pack(fill=tk.X, pady=2)
    tk.Label(r_med, text="Rich Media Rendering:", bg=THEME["bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_label"]).pack(side=tk.LEFT, padx=(0, 6))
    for v, t in [(0, "None"), (1, "Inline"), (2, "Popup")]:
        rb = tk.Radiobutton(r_med, text=t, variable=vars_dict["media_var"], value=v,
                            bg=THEME["bg_color"], fg=THEME["fg_color"], selectcolor=THEME["widget_bg_color"], font=app.fonts["ui_small"])
        rb.pack(side=tk.LEFT, padx=4)
        ToolTip(rb, f"Set media rendering mode to {t}.", app=app)

    # STT Mic & Language
    r_stt = tk.Frame(hw_f, bg=THEME["bg_color"])
    r_stt.pack(fill=tk.X, pady=2)
    tk.Label(r_stt, text="STT Mic Input:", bg=THEME["bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_label"]).pack(side=tk.LEFT, padx=(0, 4))
    stt_combo = ttk.Combobox(r_stt, textvariable=vars_dict["stt_dev_var"], values=vars_dict["dev_names"], state="readonly", width=22)
    stt_combo.pack(side=tk.LEFT, padx=(0, 10))
    ToolTip(stt_combo, "Select local audio microphone device for Speech-To-Text dictation.", app=app)

    tk.Label(r_stt, text="STT Language:", bg=THEME["bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_label"]).pack(side=tk.LEFT, padx=(0, 4))
    sttl_combo = ttk.Combobox(r_stt, textvariable=vars_dict["stt_lang_var"], values=["en-US", "en-GB", "es-ES", "fr-FR", "de-DE", "ja-JP", "zh-CN"], state="readonly", width=10)
    sttl_combo.pack(side=tk.LEFT)
    ToolTip(sttl_combo, "Select spoken language code for voice dictation.", app=app)

    # Multimedia Mode & Projector
    r_mm = tk.Frame(hw_f, bg=THEME["bg_color"])
    r_mm.pack(fill=tk.X, pady=2)
    tk.Label(r_mm, text="Multimedia Handling:", bg=THEME["bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_label"]).pack(side=tk.LEFT, padx=(0, 6))
    for opt in ["auto", "vision", "native"]:
        rb = tk.Radiobutton(r_mm, text=opt.capitalize(), variable=vars_dict["multimedia_handling_var"], value=opt,
                            bg=THEME["bg_color"], fg=THEME["fg_color"], selectcolor=THEME["widget_bg_color"], font=app.fonts["ui_small"])
        rb.pack(side=tk.LEFT, padx=4)
        ToolTip(rb, f"Use {opt} multimedia handling mode.", app=app)

    # DMN Timeout & VRAM
    r_dmn = tk.Frame(hw_f, bg=THEME["bg_color"])
    r_dmn.pack(fill=tk.X, pady=2)
    tk.Label(r_dmn, text="DMN Timeout (min:sec):", bg=THEME["bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_label"]).pack(side=tk.LEFT, padx=(0, 4))
    dmn_ent = tk.Entry(r_dmn, bg=THEME["widget_bg_color"], fg=THEME["fg_color"],
                       insertbackground=THEME.get("electric_blue", THEME["fg_color"]), width=7)
    bind_entry_limit(dmn_ent, max_len=8)
    dmn_ent.insert(0, str(app.config.get("dmn_timeout", "05:00")))
    dmn_ent.pack(side=tk.LEFT, padx=(0, 4))
    vars_dict["dmn_ent"] = dmn_ent
    ToolTip(dmn_ent, "Idle duration (mm:ss) before triggering Default Mode Network simmer reflections.", app=app)

    cb_dmn = tk.Checkbutton(r_dmn, text="Active", variable=vars_dict["dmn_enabled_var"],
                            bg=THEME["bg_color"], fg=THEME["electric_blue"], selectcolor=THEME["widget_bg_color"], font=app.fonts["ui_small"])
    cb_dmn.pack(side=tk.LEFT, padx=(0, 15))

    tk.Label(r_dmn, text="VRAM Target (GB):", bg=THEME["bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_label"]).pack(side=tk.LEFT, padx=(0, 4))
    vram_ent = tk.Entry(r_dmn, bg=THEME["widget_bg_color"], fg=THEME["fg_color"],
                        insertbackground=THEME.get("electric_blue", THEME["fg_color"]), width=6)
    bind_entry_limit(vram_ent, max_len=6)
    vram_mb = getattr(app, "state", {}).get("virtual_vram", 0)
    vram_ent.insert(0, str(vram_mb / 1024) if vram_mb else "0")
    vram_ent.pack(side=tk.LEFT)
    vars_dict["vram_ent"] = vram_ent
    ToolTip(vram_ent, "Target VRAM threshold in Gigabytes for GPU layer calculation.", app=app)

    # Video Sub-Chunk Size
    r_chunk = tk.Frame(hw_f, bg=THEME["bg_color"])
    r_chunk.pack(fill=tk.X, pady=2)
    tk.Label(r_chunk, text="Video Sub-Chunk Size:", bg=THEME["bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_label"]).pack(side=tk.LEFT, padx=(0, 6))
    sc_scale = tk.Scale(r_chunk, from_=1, to=128, orient=tk.HORIZONTAL, variable=vars_dict["sc_val"],
                        bg=THEME["bg_color"], fg=THEME["fg_color"], highlightthickness=0, resolution=1, length=180)
    sc_scale.pack(side=tk.LEFT, padx=4)
    ToolTip(sc_scale, "Adjust frame batch size for multimodal video analysis.", app=app)
    tk.Button(r_chunk, text="Reset", command=lambda: vars_dict["sc_val"].set(8),
              bg=THEME["widget_bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_small"]).pack(side=tk.LEFT, padx=4)

    # --- Section 3: Status Bar & Loading Area ---
    sb_lf = tk.LabelFrame(frame, text="⏳ Loading Bar & Status Area", bg=THEME["bg_color"],
                          fg=THEME["electric_blue"], font=app.fonts["bold"], padx=8, pady=6)
    sb_lf.pack(fill=tk.X, pady=(10, 6))

    lbl_sb_mode = tk.Label(sb_lf, text="Display Options:", bg=THEME["bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_label"])
    lbl_sb_mode.pack(anchor="w", pady=(0, 2))
    STATUS_MODES = [
        ("hybrid", "Hybrid / Smart Feature (State-Aware with Finish t/s)"),
        ("tasks", "Active Generation Tasks (Prefill, Reasoning, Streaming)"),
        ("percentage", "Percentage Gauge (Load % & TTFT / Estimated Duration)"),
        ("animation", "Selectable Animation (Custom Canvas Animation)"),
        ("prayer", "Serenity Prayer (Smooth Line Fading Transition)")
    ]
    for val, lbl in STATUS_MODES:
        rb = tk.Radiobutton(sb_lf, text=lbl, variable=vars_dict["status_mode_var"], value=val,
                            bg=THEME["bg_color"], fg=THEME["fg_color"], selectcolor=THEME["widget_bg_color"], font=app.fonts["ui_small"])
        rb.pack(anchor="w", pady=1)
        ToolTip(rb, f"Switch status bar mode to: {lbl}.", app=app)

    r_anim = tk.Frame(sb_lf, bg=THEME["bg_color"])
    r_anim.pack(fill=tk.X, pady=(4, 2))
    tk.Label(r_anim, text="Animation Style:", bg=THEME["bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_label"]).pack(side=tk.LEFT, padx=(0, 6))
    anim_combo = ttk.Combobox(r_anim, textvariable=vars_dict["anim_style_var"], values=["spinner", "pulse", "orbit"], state="readonly", width=12)
    anim_combo.pack(side=tk.LEFT)
    ToolTip(anim_combo, "Choose between spinner, pulse, or orbit canvas animation.", app=app)

    cb_dmn_idle = tk.Checkbutton(sb_lf, text="Swaps to DMN timer showing idle time", variable=vars_dict["sb_dmn_var"],
                                 bg=THEME["bg_color"], fg=THEME["fg_color"], selectcolor=THEME["widget_bg_color"], font=app.fonts["ui_small"])
    cb_dmn_idle.pack(anchor="w", pady=1)
    ToolTip(cb_dmn_idle, "Displays DMN idle timer countdown when resting.", app=app)

    cb_fallback = tk.Checkbutton(sb_lf, text="Defaults back to active level & KV quant/ctx info", variable=vars_dict["sb_fallback_var"],
                                 bg=THEME["bg_color"], fg=THEME["fg_color"], selectcolor=THEME["widget_bg_color"], font=app.fonts["ui_small"])
    cb_fallback.pack(anchor="w", pady=1)
    ToolTip(cb_fallback, "Shows persona level and hardware KV cache status when idle.", app=app)

    r_linger = tk.Frame(sb_lf, bg=THEME["bg_color"])
    r_linger.pack(fill=tk.X, pady=(4, 2))
    tk.Label(r_linger, text="Status Linger Time (sec):", bg=THEME["bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_label"]).pack(side=tk.LEFT, padx=(0, 6))
    linger_scale = tk.Scale(r_linger, from_=1.0, to=15.0, resolution=0.5, orient=tk.HORIZONTAL, variable=vars_dict["sb_linger_var"],
                            bg=THEME["bg_color"], fg=THEME["fg_color"], highlightthickness=0, length=160)
    linger_scale.pack(side=tk.LEFT, padx=4)
    ToolTip(linger_scale, "Duration (seconds) completion stats linger on the status bar before transitioning to idle.", app=app)

    return frame


def build_users_tab(parent, app, win, vars_dict):
    """
    Builds Tab 5: Users & Security.
    Top to bottom:
    1. Active User dropdown & Switch / Create Profile button
    2. Logout button (resets to Default, locks vault if active)
    3. Default / Public toggles
    4. Name & addressing style
    5. Secure Vault section below
    """
    frame = tk.Frame(parent, bg=THEME["bg_color"], padx=10, pady=8)

    # 1. Active User & Logout
    u_row = tk.Frame(frame, bg=THEME["bg_color"])
    u_row.pack(fill=tk.X, pady=4)
    tk.Label(u_row, text="Active Username:", bg=THEME["bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_label"]).pack(side=tk.LEFT, padx=(0, 6))
    user_combo = ttk.Combobox(u_row, textvariable=vars_dict["username_var"], values=vars_dict["user_profiles_list"], width=16)
    user_combo.pack(side=tk.LEFT, padx=(0, 8))
    vars_dict["user_combo"] = user_combo

    def _apply_switch():
        target = vars_dict["username_var"].get().strip()
        if not target: return
        if hasattr(app, "switch_user"):
            res = app.switch_user(target)
            if res is False:
                vars_dict["username_var"].set(app.get_active_username() if hasattr(app, 'get_active_username') else "Default")
                return
            user_combo['values'] = app.list_user_profiles() if hasattr(app, 'list_user_profiles') else ["Default"]
            if "user_pref_name_var" in vars_dict:
                vars_dict["user_pref_name_var"].set(app.config.get("user_preferred_name", ""))
            if "user_addr_style_var" in vars_dict:
                vars_dict["user_addr_style_var"].set(app.config.get("user_address_style", "Direct / Plain"))
            if "theme_display_var" in vars_dict:
                curr_th = app.config.get("theme", "apex")
                vars_dict["theme_display_var"].set(vars_dict.get("THEME_REV_MAP", {}).get(curr_th, curr_th))
            if "dark_mode_var" in vars_dict:
                vars_dict["dark_mode_var"].set(app.config.get("dark_mode", False))
            messagebox.showinfo("User Profile", f"Active user profile set to '{target}'.", parent=win)

    def _apply_logout():
        if hasattr(app, 'vault_manager') and app.vault_manager and app.vault_manager.is_lock_enabled():
            app.vault_manager.lock()
        if hasattr(app, 'switch_user'):
            app.switch_user("Default")
        vars_dict["username_var"].set("Default")
        user_combo['values'] = app.list_user_profiles() if hasattr(app, 'list_user_profiles') else ["Default"]
        if "refresh_vault_status" in vars_dict:
            vars_dict["refresh_vault_status"]()
        messagebox.showinfo("Logged Out", "Logged out. Switched to 'Default' profile.", parent=win)

    btn_switch = tk.Button(u_row, text="Switch / Create Profile", command=_apply_switch,
                           bg=THEME["button_bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_button"], relief=tk.FLAT)
    btn_switch.pack(side=tk.LEFT, padx=4)

    btn_logout = tk.Button(u_row, text="Logout", command=_apply_logout,
                           bg=THEME["button_bg_color"], fg="#ff4444", font=app.fonts["ui_button"], relief=tk.FLAT)
    btn_logout.pack(side=tk.LEFT, padx=6)
    ToolTip(btn_logout, "Log out of current profile and safely return to Default.", app=app)

    # 3. Default / Public Toggles
    vis_row = tk.Frame(frame, bg=THEME["bg_color"])
    vis_row.pack(fill=tk.X, pady=4)
    is_logged_in = (app.get_active_username() not in ("Default", "Public")) if hasattr(app, 'get_active_username') else False

    def _on_toggle_vis():
        app.config["show_default_profile"] = vars_dict["show_def_var"].get()
        app.config["show_public_profile"] = vars_dict["show_pub_var"].get()
        app.save_config()
        if hasattr(app, 'list_user_profiles'):
            user_combo['values'] = app.list_user_profiles()

    cb_def = tk.Checkbutton(vis_row, text="Show 'Default' in Profiles", variable=vars_dict["show_def_var"],
                            command=_on_toggle_vis, bg=THEME["bg_color"], fg=THEME["fg_color"],
                            selectcolor=THEME["widget_bg_color"], font=app.fonts["ui_small"],
                            state="normal" if is_logged_in else "disabled")
    cb_def.pack(side=tk.LEFT, padx=(0, 12))

    cb_pub = tk.Checkbutton(vis_row, text="Show 'Public' in Profiles", variable=vars_dict["show_pub_var"],
                            command=_on_toggle_vis, bg=THEME["bg_color"], fg=THEME["fg_color"],
                            selectcolor=THEME["widget_bg_color"], font=app.fonts["ui_small"],
                            state="normal" if is_logged_in else "disabled")
    cb_pub.pack(side=tk.LEFT)

    # 4. Preferred Name & Addressing Style
    id_row = tk.Frame(frame, bg=THEME["bg_color"])
    id_row.pack(fill=tk.X, pady=4)
    tk.Label(id_row, text="Preferred Name / Call Sign:", bg=THEME["bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_label"]).pack(side=tk.LEFT, padx=(0, 4))
    name_ent = tk.Entry(id_row, textvariable=vars_dict["user_pref_name_var"], bg=THEME["widget_bg_color"], fg=THEME["fg_color"],
                        insertbackground=THEME.get("electric_blue", THEME["fg_color"]), width=14)
    bind_entry_limit(name_ent, max_len=32)
    name_ent.pack(side=tk.LEFT, padx=(0, 12))

    tk.Label(id_row, text="Addressing Style:", bg=THEME["bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_label"]).pack(side=tk.LEFT, padx=(0, 4))
    addr_combo = ttk.Combobox(id_row, textvariable=vars_dict["user_addr_style_var"],
                              values=["Direct / Plain", "Warm / Familiar", "Formal / Respectful", "Silent / Unnamed"],
                              state="readonly", width=18)
    addr_combo.pack(side=tk.LEFT)

    # 5. Secure Vault Section Below
    vault_lf = tk.LabelFrame(frame, text="🔐 Secure Vault Encryption Panel", bg=THEME["bg_color"],
                             fg=THEME["electric_blue"], font=app.fonts["bold"], padx=8, pady=6)
    vault_lf.pack(fill=tk.X, pady=(12, 6))

    v_status_lbl = tk.Label(vault_lf, text="", bg=THEME["bg_color"], font=app.fonts["ui_button"])
    v_status_lbl.pack(anchor="w", pady=(0, 4))

    def _refresh_vault():
        if hasattr(app, 'vault_manager') and app.vault_manager and app.vault_manager.is_lock_enabled():
            if app.vault_manager.is_locked():
                v_status_lbl.config(text="● LOCKED (Encrypted)", fg="#ff4444")
            else:
                v_status_lbl.config(text="● ACTIVE (Unlocked)", fg="#00ff88")
        else:
            v_status_lbl.config(text="○ DISABLED (Plaintext)", fg="#888888")

    vars_dict["refresh_vault_status"] = _refresh_vault
    _refresh_vault()

    v_btn_row = tk.Frame(vault_lf, bg=THEME["bg_color"])
    v_btn_row.pack(fill=tk.X, pady=4)

    def _set_pwd():
        if hasattr(vars_dict.get("_open_set_password_modal"), "__call__"):
            vars_dict["_open_set_password_modal"]()

    def _disable_pwd():
        if hasattr(vars_dict.get("_open_disable_vault_modal"), "__call__"):
            vars_dict["_open_disable_vault_modal"]()

    tk.Button(v_btn_row, text="Set / Change Master Password", command=_set_pwd,
              bg=THEME["widget_bg_color"], fg=THEME.get("accent_highlight", "#00ffcc"), font=app.fonts["ui_button"]).pack(side=tk.LEFT, padx=4)
    tk.Button(v_btn_row, text="Disable Encryption / Decrypt", command=_disable_pwd,
              bg=THEME["widget_bg_color"], fg="#ff8888", font=app.fonts["ui_button"]).pack(side=tk.LEFT, padx=4)

    # Inactivity Timer Presets
    timer_row = tk.Frame(vault_lf, bg=THEME["bg_color"])
    timer_row.pack(fill=tk.X, pady=4)
    tk.Label(timer_row, text="Auto-Lock Inactivity:", bg=THEME["bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_label"]).pack(side=tk.LEFT, padx=(0, 4))
    for p_sec, p_lbl in [(0, "Off"), (15, "15s"), (30, "30s"), (45, "45s"), (300, "5m"), (900, "15m"), (1800, "30m")]:
        btn = tk.Button(timer_row, text=p_lbl, command=lambda s=p_sec: vars_dict["auto_lock_var"].set(str(s)),
                        bg=THEME["widget_bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_small"], padx=3, pady=0)
        btn.pack(side=tk.LEFT, padx=2)

    return frame


def build_personalize_tab(parent, app, win, vars_dict):
    """
    Builds Tab 6: Personalize.
    Top to bottom:
    1. Theme choices with visible color preview swatches up top.
    2. Font Selections below that (UI, Log beside each other).
    3. Texture Style & Intensity below that (beside each other).
    4. Dark Mode toggle below that (renamed simply 'Dark Mode').
    5. Text size and Global Scale, with the rest of the scaling center below.
    """
    frame = tk.Frame(parent, bg=THEME["bg_color"], padx=10, pady=8)

    # 1. Theme Choices with Color Examples Up Top
    lbl_thm = tk.Label(frame, text="Theme Palette & Visual Preview:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["bold"])
    lbl_thm.pack(anchor="w", pady=(2, 4))
    ToolTip(lbl_thm, "Select visual color theme palette with live preview.", app=app)

    theme_grid = tk.Frame(frame, bg=THEME["bg_color"])
    theme_grid.pack(fill=tk.X, pady=(0, 8))

    themes_to_show = [
        ("Apex (Default)", "apex"),
        ("Goth / Obsidian Dark", "goth"),
        ("Crystal Cavern", "crystal_cavern"),
        ("Yellow Blacket", "yellow_blacket"),
        ("Natural (Earth / Moss)", "natural"),
        ("Matrix (Cyber Green)", "matrix"),
        ("Persona (Level Dynamic)", "persona")
    ]

    for idx, (disp, key) in enumerate(themes_to_show):
        r, c = divmod(idx, 4)
        t_data = THEMES.get(key, THEME)
        card = tk.Frame(theme_grid, bg=t_data.get("widget_bg_color", "#18181c"),
                        highlightthickness=1, highlightbackground=t_data.get("accent_highlight", "#ff8800"), padx=4, pady=4)
        card.grid(row=r, column=c, padx=3, pady=3, sticky="nsew")

        # Color preview chips
        swatch_f = tk.Frame(card, bg=t_data.get("bg_color", "#000000"), width=24, height=14)
        swatch_f.pack(side=tk.LEFT, padx=(0, 4))
        dot = tk.Frame(swatch_f, bg=t_data.get("accent_highlight", "#ff8800"), width=6, height=6)
        dot.place(relx=0.5, rely=0.5, anchor="center")

        rb = tk.Radiobutton(card, text=disp.split(" (")[0], variable=vars_dict["theme_display_var"], value=disp,
                            bg=t_data.get("widget_bg_color", "#18181c"), fg=t_data.get("fg_color", "#ffffff"),
                            selectcolor=t_data.get("button_active_color", "#382e24"), font=app.fonts["ui_small"])
        rb.pack(side=tk.LEFT)
        ToolTip(card, f"Switch palette to {disp}.", app=app)

    # 2. Font Selections Beside Each Other
    lbl_fonts = tk.Label(frame, text="Typography & Font Selection:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["bold"])
    lbl_fonts.pack(anchor="w", pady=(6, 2))

    font_row = tk.Frame(frame, bg=THEME["bg_color"])
    font_row.pack(fill=tk.X, pady=(0, 6))
    font_row.grid_columnconfigure(0, weight=1)
    font_row.grid_columnconfigure(1, weight=1)

    if "ui_font_var" not in vars_dict:
        vars_dict["ui_font_var"] = tk.StringVar(value=getattr(app, "config", {}).get("ui_font", "Segoe UI"))
    if "mono_font_var" not in vars_dict:
        vars_dict["mono_font_var"] = tk.StringVar(value=getattr(app, "config", {}).get("mono_font", "Consolas"))
    if "UI_FONT_OPTIONS" not in vars_dict:
        vars_dict["UI_FONT_OPTIONS"] = ["Segoe UI", "Verdana", "Arial"]
    if "MONO_FONT_OPTIONS" not in vars_dict:
        vars_dict["MONO_FONT_OPTIONS"] = ["Consolas", "Courier New"]

    f_left = tk.Frame(font_row, bg=THEME["bg_color"])
    f_left.grid(row=0, column=0, sticky="w", padx=(0, 10))
    tk.Label(f_left, text="UI Font:", bg=THEME["bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_label"]).pack(side=tk.LEFT, padx=(0, 4))
    ui_combo = ttk.Combobox(f_left, textvariable=vars_dict["ui_font_var"], values=vars_dict["UI_FONT_OPTIONS"], state="readonly", width=18)
    ui_combo.pack(side=tk.LEFT)

    f_right = tk.Frame(font_row, bg=THEME["bg_color"])
    f_right.grid(row=0, column=1, sticky="w")
    tk.Label(f_right, text="Code / Log Font:", bg=THEME["bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_label"]).pack(side=tk.LEFT, padx=(0, 4))
    mono_combo = ttk.Combobox(f_right, textvariable=vars_dict["mono_font_var"], values=vars_dict["MONO_FONT_OPTIONS"], state="readonly", width=18)
    mono_combo.pack(side=tk.LEFT)

    def _on_font_change(*args):
        u_fam = vars_dict["ui_font_var"].get()
        m_fam = vars_dict["mono_font_var"].get()
        if hasattr(app, "apply_font_family"):
            app.apply_font_family(u_fam, m_fam, persist=True)
    ui_combo.bind("<<ComboboxSelected>>", _on_font_change)
    mono_combo.bind("<<ComboboxSelected>>", _on_font_change)

    # 3. Texture Style & Intensity Beside Each Other
    lbl_tex = tk.Label(frame, text="Texture Finish & Intensity:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["bold"])
    lbl_tex.pack(anchor="w", pady=(6, 2))

    tex_row = tk.Frame(frame, bg=THEME["bg_color"])
    tex_row.pack(fill=tk.X, pady=(0, 6))
    tex_row.grid_columnconfigure(0, weight=1)
    tex_row.grid_columnconfigure(1, weight=1)

    t_left = tk.Frame(tex_row, bg=THEME["bg_color"])
    t_left.grid(row=0, column=0, sticky="w", padx=(0, 10))
    tk.Label(t_left, text="Texture Style:", bg=THEME["bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_label"]).pack(side=tk.LEFT, padx=(0, 4))
    tex_combo = ttk.Combobox(t_left, textvariable=vars_dict["tex_display_var"], values=list(vars_dict["TEXTURE_MAP"].keys()), state="readonly", width=18)
    tex_combo.pack(side=tk.LEFT)

    t_right = tk.Frame(tex_row, bg=THEME["bg_color"])
    t_right.grid(row=0, column=1, sticky="w")
    tk.Label(t_right, text="Intensity:", bg=THEME["bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_label"]).pack(side=tk.LEFT, padx=(0, 4))
    tex_scale = tk.Scale(t_right, from_=0, to=100, orient=tk.HORIZONTAL, variable=vars_dict["tex_int_var"],
                         bg=THEME["bg_color"], fg=THEME["fg_color"], highlightthickness=0, resolution=5, length=120)
    tex_scale.pack(side=tk.LEFT)

    # 4. Dark Mode Toggle (Renamed simply 'Dark Mode')
    cb_dark = tk.Checkbutton(frame, text="Dark Mode", variable=vars_dict["dark_mode_var"],
                             bg=THEME["bg_color"], fg=THEME["electric_blue"], selectcolor=THEME["widget_bg_color"],
                             font=app.fonts["ui_label"])
    cb_dark.pack(anchor="w", pady=(4, 6))
    ToolTip(cb_dark, "Pure OLED blackout (#000000) for maximum neon text contrast and power efficiency.", app=app)

    # 5. Text Size & Global Scale with Scaling Center
    lbl_scale = tk.Label(frame, text="Text Size & Global Scale:", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=app.fonts["bold"])
    lbl_scale.pack(anchor="w", pady=(6, 2))

    scale_row = tk.Frame(frame, bg=THEME["bg_color"])
    scale_row.pack(fill=tk.X, pady=(0, 6))
    tk.Label(scale_row, text="Scale Preset:", bg=THEME["bg_color"], fg=THEME["fg_color"], font=app.fonts["ui_label"]).pack(side=tk.LEFT, padx=(0, 4))
    scale_combo = ttk.Combobox(scale_row, textvariable=vars_dict["text_scale_display_var"], values=list(vars_dict["SCALE_MAP"].keys()), state="readonly", width=18)
    scale_combo.pack(side=tk.LEFT, padx=(0, 12))

    def _on_scale_select(*args):
        sel = vars_dict["text_scale_display_var"].get()
        if sel in vars_dict["SCALE_MAP"]:
            val = vars_dict["SCALE_MAP"][sel]
            vars_dict["text_scale_val_var"].set(val)
            if hasattr(app, 'apply_text_scale'):
                app.apply_text_scale(val, persist=True)
    scale_combo.bind("<<ComboboxSelected>>", _on_scale_select)

    btn_sc = tk.Button(frame, text="🔍 Open Text size & Scaling Center",
                       command=lambda: getattr(vars_dict.get("open_scaling_center_fn"), "__call__", lambda: None)(),
                       bg=THEME["widget_bg_color"], fg=THEME.get("accent_highlight", "#00ffcc"),
                       font=app.fonts["ui_button"], relief=tk.FLAT, padx=6, pady=4)
    btn_sc.pack(anchor="w", pady=(4, 8))
    ToolTip(btn_sc, "Open comprehensive Text size & Scaling Center for fine-tuned per-category font sizing and live preview.", app=app)

    return frame
