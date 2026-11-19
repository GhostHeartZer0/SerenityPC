# System/serenity_p2p_ui_wiring.py
"""
SerenityPC & SerenityPC Legacy UI, Process Queue & Settings Wiring for:
- Per-Subagent Collapsible Dropdowns (`subagent_spawn`, `subagent_stream`, `subagent_done`)
- Cross-State P2P Pairing & Invite Card Modal (`serenity-peer://`)
- Serenity Encrypted Mailbox & Quarantine Sandbox Modal (`.smp` / `.bmb`)
- 72-Hour Profile Age & Trust Tier Live Status Badge in Settings Tab 5
- SerenityDev MCP Subagent Bridge Controls in Settings Tab 3
- Standalone Folder & PyInstaller `--onedir` Runtime Auto-Hook
"""

import os
import sys
import json
import time
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog, scrolledtext
    TK_AVAILABLE = True
except ImportError:
    class _DummyTk:
        Widget = Any = object
        Toplevel = object
        END = "end"
        FLAT = "flat"
        LEFT = "left"
        RIGHT = "right"
        BOTH = "both"
        X = "x"
        VERTICAL = "vertical"
    tk = _DummyTk()
    ttk = messagebox = filedialog = scrolledtext = None
    TK_AVAILABLE = False
from typing import Any, Dict, Optional

try:
    from serenity_resources import THEME, PERSONA_SECONDARY_COLORS
except ImportError:
    THEME = {
        "bg_color": "#1b1b1f",
        "fg_color": "#ff6a00",
        "widget_bg_color": "#222228",
        "button_bg_color": "#24201c",
        "button_active_color": "#382e24",
        "electric_blue": "#ff6a00",
        "accent_highlight": "#ff7700",
    }
    PERSONA_SECONDARY_COLORS = {}

try:
    from System.serenity_utils import ToolTip, bind_entry_limit
except ImportError:
    def ToolTip(*args, **kwargs):
        return None
    def bind_entry_limit(entry_widget, max_len=64):
        return entry_widget

try:
    from System.serenity_mail_courier import (
        SerenityEnvResolver,
        Level2CourierAgent,
        CrossStateP2PTransport,
        PacketType,
        ExchangeMode,
    )
except ImportError:
    from serenity_mail_courier import (
        SerenityEnvResolver,
        Level2CourierAgent,
        CrossStateP2PTransport,
        PacketType,
        ExchangeMode,
    )


def _get_courier(app: Any) -> Level2CourierAgent:
    """Retrieves or initializes the Level2CourierAgent bound to `app` and syncs active user."""
    tr = getattr(app, "tool_registry", None)
    if tr is not None and hasattr(tr, "courier") and tr.courier is not None:
        tr.courier.sync_environment_and_user()
        return tr.courier
    if not hasattr(app, "_standalone_courier") or app._standalone_courier is None:
        app._standalone_courier = Level2CourierAgent(app=app)
    app._standalone_courier.sync_environment_and_user()
    return app._standalone_courier


# ==============================================================================
# 1. RUNTIME AUTO-WIRING FOR MAIN.PY (QUEUE HANDLERS & PROFILE SWITCHING)
# ==============================================================================

def wire_serenity_app(app: Any) -> bool:
    """
    Idempotently attaches V2.3.0 P2P, Mail, Quarantine, and Per-Subagent Collapsible
    Dropdown handlers onto a `SerenityApp` / `ChatbotApp` instance.
    Safe to call early in `__init__` (before `queue_handlers` exists) or post-startup.
    """
    if app is None:
        return False

    # If called at line 383/533 of main.py before self.queue_handlers is created at line 642,
    # schedule a follow-up hook on the Tkinter event loop.
    if not hasattr(app, "queue_handlers") or not isinstance(getattr(app, "queue_handlers", None), dict):
        root = getattr(app, "root", None)
        if root is not None and hasattr(root, "after") and not getattr(app, "_p2p_wire_scheduled", False):
            app._p2p_wire_scheduled = True
            try:
                root.after(80, lambda: wire_serenity_app(app))
            except Exception:
                pass
        return False

    # 1. Register V2.3.0 Queue Handlers in app.queue_handlers
    app.queue_handlers["subagent_spawn"] = lambda msg: handle_subagent_spawn(app, msg)
    app.queue_handlers["subagent_stream"] = lambda msg: handle_subagent_stream(app, msg)
    app.queue_handlers["subagent_done"] = lambda msg: handle_subagent_done(app, msg)
    app.queue_handlers["p2p_notification"] = lambda msg: handle_p2p_notification(app, msg)

    # 2. Wrap app.switch_user so profile switches sync the active user's Mailbox & 72h age gate
    if hasattr(app, "switch_user") and not getattr(app, "_p2p_switch_user_wrapped", False):
        orig_switch = app.switch_user

        def _wrapped_switch_user(username, *args, **kwargs):
            res = orig_switch(username, *args, **kwargs)
            try:
                _get_courier(app).sync_environment_and_user()
            except Exception:
                pass
            return res

        app.switch_user = _wrapped_switch_user
        app._p2p_switch_user_wrapped = True

    # 3. Start lightweight periodic Dead-Drop & Outbox poller (every 15s on background thread)
    root = getattr(app, "root", None)
    if root is not None and hasattr(root, "after") and not getattr(app, "_p2p_poller_started", False):
        app._p2p_poller_started = True

        def _periodic_p2p_poll():
            try:
                import threading
                if not getattr(app, "_p2p_poll_running", False):
                    app._p2p_poll_running = True

                    def _bg():
                        try:
                            _get_courier(app).poll_deaddrop_and_outbox()
                        except Exception:
                            pass
                        finally:
                            app._p2p_poll_running = False

                    threading.Thread(target=_bg, daemon=True).start()
            except Exception:
                pass
            try:
                if root.winfo_exists():
                    root.after(15000, _periodic_p2p_poll)
            except Exception:
                pass

        try:
            root.after(4000, _periodic_p2p_poll)
        except Exception:
            pass

    # 4. Wire History File Titles Marquee on app._render_history_menu (V1.9.0)
    if hasattr(app, "_render_history_menu") and not getattr(app, "_marquee_history_wrapped", False):
        orig_render_hist = app._render_history_menu

        def _wrapped_render_history_menu(*args, **kwargs):
            res = orig_render_hist(*args, **kwargs)
            try:
                try:
                    from System.serenity_utils import setup_marquee_label as _real_marquee
                except ImportError:
                    from serenity_utils import setup_marquee_label as _real_marquee
                scroll_f = getattr(app, "history_scroll_frame", None)
                if scroll_f is not None and hasattr(scroll_f, "winfo_exists") and scroll_f.winfo_exists():
                    for card in scroll_f.winfo_children():
                        for hdr in card.winfo_children():
                            children = hdr.winfo_children() if hasattr(hdr, "winfo_children") else []
                            # In _render_history_menu: badge (left), meta_lbl (right), name_lbl (left expand)
                            if len(children) >= 3:
                                name_lbl = children[2]
                                if hasattr(name_lbl, "cget") and name_lbl.winfo_class() == "Label":
                                    raw_title = getattr(name_lbl, "_full_history_title", None) or str(name_lbl.cget("text"))
                                    name_lbl._full_history_title = raw_title
                                    full_p = getattr(name_lbl, "_full_history_path", raw_title)
                                    _real_marquee(name_lbl, raw_title, app=app, limit=28, full_path=full_p)
            except Exception:
                pass
            return res

        app._render_history_menu = _wrapped_render_history_menu
        app._marquee_history_wrapped = True

    app._p2p_wired = True
    return True


# ==============================================================================
# 2. PER-SUBAGENT COLLAPSIBLE DROPDOWN HANDLERS (V2.3.0)
# ==============================================================================

def handle_subagent_spawn(app: Any, msg: Dict[str, Any]) -> None:
    """
    Creates an individual collapsible dropdown block in `app.chat_history` for each
    spawned subagent (Courier Lvl 2, SerenityDev MCP, or OTL Subagent Lvl 1-6).
    """
    sub_id = str(msg.get("subagent_id") or f"sa_{int(time.time() * 1000)}")
    lvl = int(msg.get("level") or 2)
    label = str(msg.get("label") or f"[+] Subagent Lvl {lvl}")
    if not label.startswith("["):
        label = f"[+] {label}"

    mode = "per_subagent"
    if hasattr(app, "config") and isinstance(app.config, dict):
        mode = str(app.config.get("subagent_dropdown_mode", "per_subagent")).lower()

    if mode == "logs_only":
        return
    if mode == "combined":
        if hasattr(app, "_update_agentic_dropdown"):
            app._update_agentic_dropdown(f"\n--- {label} ---\n")
        return

    hist = getattr(app, "chat_history", None)
    if hist is None:
        return
    try:
        if hasattr(hist, "winfo_exists") and not hist.winfo_exists():
            return
    except Exception:
        pass

    tag = f"sa_block_{sub_id}"
    if not hasattr(app, "_subagent_ui_registry") or not isinstance(app._subagent_ui_registry, dict):
        app._subagent_ui_registry = {}

    btn_bg = THEME.get("button_bg_color", "#202024")
    fg_col = PERSONA_SECONDARY_COLORS.get(lvl, THEME.get("electric_blue", "#00bfff"))

    try:
        lock_scroll = bool(getattr(app, "_should_lock_scroll", lambda: False)())
        saved_y0 = None
        saved_top = None
        if hasattr(app, "_capture_scroll_anchor"):
            lock_scroll, saved_y0, saved_top = app._capture_scroll_anchor(hist)
        elif lock_scroll:
            try:
                saved_y0 = float(hist.yview()[0])
            except Exception:
                saved_y0 = None
        hist.config(state="normal")

        def _toggle_sa(t=tag, b_ref=sub_id):
            info = app._subagent_ui_registry.get(b_ref, {})
            btn_w = info.get("btn")
            base_lbl = info.get("base_label", f"Subagent Lvl {lvl}")
            is_elided = str(hist.tag_cget(t, "elide")) in ("1", "True", "true")
            new_elide = not is_elided
            hist.tag_config(t, elide=new_elide)
            if btn_w and hasattr(btn_w, "winfo_exists") and btn_w.winfo_exists():
                prefix = "[+]" if new_elide else "[-]"
                btn_w.config(text=f"{prefix} {base_lbl}")

        clean_base = label.replace("[+]", "").replace("[-]", "").strip()
        fonts_dict = getattr(app, "fonts", {}) or {}
        btn_font = fonts_dict.get("stats") or fonts_dict.get("small") or ("Segoe UI", 10)
        log_font = fonts_dict.get("log") or fonts_dict.get("small") or ("Consolas", 10)

        btn = tk.Button(
            hist,
            text=f"[+] {clean_base}",
            bg=btn_bg,
            fg=fg_col,
            relief=tk.FLAT,
            font=btn_font,
            cursor="hand2",
            padx=6,
            pady=1,
        )
        btn.config(command=_toggle_sa)

        app._subagent_ui_registry[sub_id] = {
            "tag": tag,
            "btn": btn,
            "base_label": clean_base,
            "level": lvl,
        }

        hist.insert(tk.END, "\n", ("ai",))
        hist.window_create(tk.END, window=btn)
        hist.insert(tk.END, "\n", ("ai",))
        hist.insert(tk.END, f"  [{clean_base}]\n", (tag, "ai"))
        hist.tag_config(tag, elide=True, foreground="#9a9a9a", font=log_font, lmargin1=18, lmargin2=18)
        hist.config(state="disabled")
        if lock_scroll:
            if hasattr(app, "_restore_or_scroll_bottom"):
                app._restore_or_scroll_bottom(hist, True, saved_y0, saved_top, target="end-1c")
            elif saved_y0 is not None:
                hist.yview_moveto(saved_y0)
    except Exception:
        pass


def handle_subagent_stream(app: Any, msg: Dict[str, Any]) -> None:
    """Streams live subagent thoughts or tool output into its collapsible tag block."""
    sub_id = str(msg.get("subagent_id") or "")
    chunk = str(msg.get("chunk") or "")
    if not chunk:
        return

    mode = "per_subagent"
    if hasattr(app, "config") and isinstance(app.config, dict):
        mode = str(app.config.get("subagent_dropdown_mode", "per_subagent")).lower()

    if mode == "logs_only":
        if hasattr(app, "_buffer_tool_log"):
            app._buffer_tool_log(chunk.rstrip())
        return
    if mode == "combined":
        if hasattr(app, "_update_agentic_dropdown"):
            app._update_agentic_dropdown(chunk)
        return

    hist = getattr(app, "chat_history", None)
    reg = getattr(app, "_subagent_ui_registry", None)
    if hist is None or not isinstance(reg, dict) or sub_id not in reg:
        return

    tag = reg[sub_id]["tag"]
    try:
        lock_scroll = bool(getattr(app, "_should_lock_scroll", lambda: False)())
        saved_y0 = None
        saved_top = None
        if hasattr(app, "_capture_scroll_anchor"):
            lock_scroll, saved_y0, saved_top = app._capture_scroll_anchor(hist)
        elif lock_scroll:
            try:
                saved_y0 = float(hist.yview()[0])
            except Exception:
                saved_y0 = None
        hist.config(state="normal")
        hist.insert(tk.END, f"  {chunk}", (tag, "ai"))
        hist.config(state="disabled")
        if lock_scroll:
            if hasattr(app, "_restore_or_scroll_bottom"):
                app._restore_or_scroll_bottom(hist, True, saved_y0, saved_top, target="end-1c")
            elif saved_y0 is not None:
                hist.yview_moveto(saved_y0)
    except Exception:
        pass


def handle_subagent_done(app: Any, msg: Dict[str, Any]) -> None:
    """Marks a per-subagent collapsible dropdown as completed with execution duration."""
    sub_id = str(msg.get("subagent_id") or "")
    dur_ms = float(msg.get("duration_ms") or 0.0)
    reg = getattr(app, "_subagent_ui_registry", None)
    if not isinstance(reg, dict) or sub_id not in reg:
        return

    info = reg[sub_id]
    btn = info.get("btn")
    tag = info.get("tag")
    base_lbl = info.get("base_label", "Subagent")
    dur_str = f"{dur_ms / 1000.0:.2f}s" if dur_ms >= 1000 else f"{dur_ms:.0f}ms"
    new_base = f"✔ {base_lbl} ({dur_str})"
    info["base_label"] = new_base

    hist = getattr(app, "chat_history", None)
    try:
        is_elided = True
        if hist is not None and tag:
            is_elided = str(hist.tag_cget(tag, "elide")) in ("1", "True", "true")
        if btn and hasattr(btn, "winfo_exists") and btn.winfo_exists():
            prefix = "[+]" if is_elided else "[-]"
            btn.config(text=f"{prefix} {new_base}")
    except Exception:
        pass


def handle_p2p_notification(app: Any, msg: Dict[str, Any]) -> None:
    """Handles inbound Dead-Drop / P2P mail notifications."""
    ev = msg.get("event", "received")
    sender = msg.get("sender", "Peer")
    subj = msg.get("subject", "Serenity Mail")
    summary = f"[SERENITY MAIL] {ev.upper()} from {sender}: '{subj}'"
    if hasattr(app, "_buffer_tool_log"):
        app._buffer_tool_log(summary)
    if hasattr(app, "_update_status_label_text"):
        try:
            app._update_status_label_text(summary)
        except Exception:
            pass


# ==============================================================================
# 3. SETTINGS UI & VARS WIRING (TAB 3: AGENTS & TAB 5: USERS & SECURITY)
# ==============================================================================

def init_p2p_settings_vars(app: Any, vars_dict: Dict[str, Any]) -> None:
    """Initializes Tkinter variables for P2P Messaging, Mail, and SerenityDev MCP in `vars_dict`."""
    cfg = getattr(app, "config", {}) if app else {}
    if not isinstance(cfg, dict):
        cfg = {}

    defaults = {
        "messaging_default_level_var": tk.IntVar(value=int(cfg.get("messaging_default_level", 2))),
        "cecilia_p2p_mode_var": tk.StringVar(value=str(cfg.get("cecilia_p2p_mode", "quarantine_gate"))),
        "subagent_dropdown_mode_var": tk.StringVar(value=str(cfg.get("subagent_dropdown_mode", "per_subagent"))),
        "serenitydev_mcp_url_var": tk.StringVar(value=str(cfg.get("serenitydev_mcp_url", "http://127.0.0.1:8002/mcp"))),
        "serenitydev_auto_spawn_var": tk.BooleanVar(value=bool(cfg.get("serenitydev_auto_spawn", True))),
    }
    for k, v in defaults.items():
        if k not in vars_dict or vars_dict[k] is None:
            vars_dict[k] = v


def save_p2p_settings_vars(app: Any, vars_dict: Dict[str, Any]) -> None:
    """Persists P2P Messaging, Mail, and SerenityDev MCP settings into `app.config`."""
    if app is None or not hasattr(app, "config"):
        return
    if "messaging_default_level_var" in vars_dict:
        app.config["messaging_default_level"] = int(vars_dict["messaging_default_level_var"].get())
    if "cecilia_p2p_mode_var" in vars_dict:
        app.config["cecilia_p2p_mode"] = str(vars_dict["cecilia_p2p_mode_var"].get()).strip()
    if "subagent_dropdown_mode_var" in vars_dict:
        app.config["subagent_dropdown_mode"] = str(vars_dict["subagent_dropdown_mode_var"].get()).strip()
    if "serenitydev_mcp_url_var" in vars_dict:
        app.config["serenitydev_mcp_url"] = str(vars_dict["serenitydev_mcp_url_var"].get()).strip()
    if "serenitydev_auto_spawn_var" in vars_dict:
        app.config["serenitydev_auto_spawn"] = bool(vars_dict["serenitydev_auto_spawn_var"].get())
    try:
        _get_courier(app).sync_environment_and_user()
    except Exception:
        pass


def inject_agents_tab_p2p_controls(frame: tk.Widget, app: Any, win: tk.Widget, vars_dict: Dict[str, Any], bind_radio_contrast_fn=None) -> None:
    """
    Appends the V2.3.0 Inter-Serenity P2P, Mail & SerenityDev MCP section to Tab 3 (`build_agents_tab`).
    """
    init_p2p_settings_vars(app, vars_dict)
    fonts = getattr(app, "fonts", {}) or {}
    f_bold = fonts.get("bold", ("Segoe UI", 11, "bold"))
    f_lbl = fonts.get("ui_label", ("Segoe UI", 10))
    f_sm = fonts.get("ui_small", ("Segoe UI", 9))
    f_btn = fonts.get("ui_button", ("Segoe UI", 10, "bold"))

    p2p_lf = tk.LabelFrame(
        frame,
        text="🌐 Inter-Serenity P2P Messaging, Mail & SerenityDev MCP (V2.3.0)",
        bg=THEME["bg_color"],
        fg=THEME["electric_blue"],
        font=f_bold,
        padx=8,
        pady=6,
    )
    p2p_lf.pack(fill=tk.X, pady=(12, 8))

    # 1. Default P2P & Mail Persona Level (Default: Level 2)
    lbl_lvl = tk.Label(
        p2p_lf,
        text="Default P2P Courier & Mail Persona Level:",
        bg=THEME["bg_color"],
        fg=THEME["electric_blue"],
        font=f_lbl,
    )
    lbl_lvl.pack(anchor="w", pady=(2, 2))
    ToolTip(lbl_lvl, "Level 2 (The Helper) is recommended: lowest tool-capable tier (~4.89 GiB VRAM) that protects Cecilia (Lvl 7).", app=app)

    lvl_f = tk.Frame(p2p_lf, bg=THEME["bg_color"])
    lvl_f.pack(anchor="w", pady=(0, 6))
    lvl_rbs = []
    for l_val, l_txt in [
        (2, "Lvl 2: Helper (Default)"),
        (3, "Lvl 3: Collaborator"),
        (4, "Lvl 4: Confidant"),
        (5, "Lvl 5: Brains"),
        (6, "Lvl 6: Prime"),
        (7, "Lvl 7: Cecilia"),
    ]:
        rb = tk.Radiobutton(
            lvl_f,
            text=l_txt,
            variable=vars_dict["messaging_default_level_var"],
            value=l_val,
            indicatoron=False,
            bg=THEME["widget_bg_color"],
            fg=THEME["fg_color"],
            selectcolor=THEME["electric_blue"],
            activebackground=THEME["electric_blue"],
            activeforeground="#000000",
            font=f_sm,
            padx=6,
        )
        rb.pack(side=tk.LEFT, padx=2, pady=1)
        lvl_rbs.append((rb, l_val))
    if callable(bind_radio_contrast_fn):
        bind_radio_contrast_fn(vars_dict["messaging_default_level_var"], lvl_rbs)

    # 2. Cecilia / Brainuke Gatekeeper Mode
    lbl_gate = tk.Label(
        p2p_lf,
        text="Cecilia (Lvl 7) & Brainuke P2P Gatekeeper:",
        bg=THEME["bg_color"],
        fg=THEME["electric_blue"],
        font=f_lbl,
    )
    lbl_gate.pack(anchor="w", pady=(4, 2))
    ToolTip(lbl_gate, "Controls how external peer messages reach Cecilia (Lvl 7). Quarantine Gate triages via Lvl 2 with bounded 0.25 influence weight.", app=app)

    gate_f = tk.Frame(p2p_lf, bg=THEME["bg_color"])
    gate_f.pack(anchor="w", pady=(0, 6))
    gate_rbs = []
    for g_val, g_txt in [
        ("quarantine_gate", "Quarantine Gate (Lvl 2 -> Lvl 7 @ 0.25x)"),
        ("off", "Block Lvl 7 External P2P"),
        ("direct_throne", "Direct Paired Throne"),
    ]:
        rb = tk.Radiobutton(
            gate_f,
            text=g_txt,
            variable=vars_dict["cecilia_p2p_mode_var"],
            value=g_val,
            indicatoron=False,
            bg=THEME["widget_bg_color"],
            fg=THEME["fg_color"],
            selectcolor=THEME["electric_blue"],
            activebackground=THEME["electric_blue"],
            activeforeground="#000000",
            font=f_sm,
            padx=6,
        )
        rb.pack(side=tk.LEFT, padx=2, pady=1)
        gate_rbs.append((rb, g_val))
    if callable(bind_radio_contrast_fn):
        bind_radio_contrast_fn(vars_dict["cecilia_p2p_mode_var"], gate_rbs)

    # 3. Subagent Stream Dropdowns
    lbl_dd = tk.Label(
        p2p_lf,
        text="Subagent Output Dropdown Mode:",
        bg=THEME["bg_color"],
        fg=THEME["electric_blue"],
        font=f_lbl,
    )
    lbl_dd.pack(anchor="w", pady=(4, 2))
    dd_f = tk.Frame(p2p_lf, bg=THEME["bg_color"])
    dd_f.pack(anchor="w", pady=(0, 6))
    dd_rbs = []
    for d_val, d_txt in [
        ("per_subagent", "Per-Subagent Collapsible Dropdown"),
        ("combined", "Combined Agentic Dropdown"),
        ("logs_only", "Backend Logs Only"),
    ]:
        rb = tk.Radiobutton(
            dd_f,
            text=d_txt,
            variable=vars_dict["subagent_dropdown_mode_var"],
            value=d_val,
            indicatoron=False,
            bg=THEME["widget_bg_color"],
            fg=THEME["fg_color"],
            selectcolor=THEME["electric_blue"],
            activebackground=THEME["electric_blue"],
            activeforeground="#000000",
            font=f_sm,
            padx=6,
        )
        rb.pack(side=tk.LEFT, padx=2, pady=1)
        dd_rbs.append((rb, d_val))
    if callable(bind_radio_contrast_fn):
        bind_radio_contrast_fn(vars_dict["subagent_dropdown_mode_var"], dd_rbs)

    # 4. SerenityDev MCP Endpoint & Test Button
    mcp_row = tk.Frame(p2p_lf, bg=THEME["bg_color"])
    mcp_row.pack(fill=tk.X, pady=(4, 2))
    tk.Label(mcp_row, text="SerenityDev MCP Endpoint:", bg=THEME["bg_color"], fg=THEME["fg_color"], font=f_lbl).pack(side=tk.LEFT, padx=(0, 4))
    mcp_ent = tk.Entry(
        mcp_row,
        textvariable=vars_dict["serenitydev_mcp_url_var"],
        bg=THEME["widget_bg_color"],
        fg=THEME["fg_color"],
        insertbackground=THEME.get("electric_blue", "#ff6a00"),
        width=28,
    )
    bind_entry_limit(mcp_ent, max_len=96)
    mcp_ent.pack(side=tk.LEFT, padx=4)

    def _test_mcp():
        save_p2p_settings_vars(app, vars_dict)
        courier = _get_courier(app)
        online = courier.mcp_client.is_online()
        target = SerenityEnvResolver.find_serenitydev_target(app)
        if online:
            messagebox.showinfo("SerenityDev MCP", f"Connected to SerenityDev MCP at {courier.mcp_client.base_url}!", parent=win)
        else:
            t_info = f"Found local launcher: {target[0]} ({target[2]})" if target else "No local SerenityDev folder detected alongside SerenityPC."
            messagebox.showwarning(
                "SerenityDev MCP Offline",
                f"SerenityDev MCP server at {courier.mcp_client.base_url} is currently offline.\n\n{t_info}\n\nTasks delegated via delegate_to_serenitydev will stage an OTL ContextSnapshot until SerenityDev is started.",
                parent=win,
            )

    tk.Button(
        mcp_row,
        text="🔌 Test SerenityDev MCP",
        command=_test_mcp,
        bg=THEME["widget_bg_color"],
        fg=THEME.get("accent_highlight", "#00ffcc"),
        font=f_sm,
        padx=6,
    ).pack(side=tk.LEFT, padx=4)


def inject_users_tab_p2p_controls(frame: tk.Widget, app: Any, win: tk.Widget, vars_dict: Dict[str, Any], vault_lf: Optional[tk.Widget] = None) -> None:
    """
    Enhances Tab 5 (`build_users_tab`) with:
    1. Live 72-Hour Profile Age & Trust Tier Badge inside the Secure Vault panel.
    2. P2P Contacts, Cross-State Invite Pairing & Serenity Mailbox panel.
    """
    init_p2p_settings_vars(app, vars_dict)
    fonts = getattr(app, "fonts", {}) or {}
    f_bold = fonts.get("bold", ("Segoe UI", 11, "bold"))
    f_btn = fonts.get("ui_button", ("Segoe UI", 10, "bold"))
    f_sm = fonts.get("ui_small", ("Segoe UI", 9))

    target_vault_parent = vault_lf if vault_lf is not None else frame
    trust_badge_lbl = tk.Label(target_vault_parent, text="", bg=THEME["bg_color"], font=f_sm, anchor="w", justify=tk.LEFT)
    trust_badge_lbl.pack(anchor="w", pady=(2, 4))

    orig_refresh = vars_dict.get("refresh_vault_status")

    def _enhanced_refresh_vault():
        if callable(orig_refresh):
            try:
                orig_refresh()
            except Exception:
                pass
        try:
            courier = _get_courier(app)
            trust = courier.get_trust_status()
            tier = trust.get("tier", 0)
            color = "#00ff88" if tier == 2 else ("#ffcc00" if tier == 1 else "#888888")
            env_mode = "PyInstaller Onedir" if SerenityEnvResolver.is_frozen() else "Standalone Folder"
            trust_badge_lbl.config(
                text=f"🛡️ 72h Gate: {trust.get('label', '')}  |  Env: {env_mode}",
                fg=color,
            )
        except Exception:
            pass

    vars_dict["refresh_vault_status"] = _enhanced_refresh_vault
    _enhanced_refresh_vault()

    # P2P Contacts & Mailbox Panel
    p2p_box = tk.LabelFrame(
        frame,
        text="📬 Cross-State P2P Contacts & Serenity Mailbox (.smp / .bmb)",
        bg=THEME["bg_color"],
        fg=THEME["electric_blue"],
        font=f_bold,
        padx=8,
        pady=6,
    )
    p2p_box.pack(fill=tk.X, pady=(8, 6))

    summary_lbl = tk.Label(p2p_box, text="", bg=THEME["bg_color"], fg=THEME["fg_color"], font=f_sm)
    summary_lbl.pack(anchor="w", pady=(0, 4))

    def _refresh_p2p_summary():
        try:
            courier = _get_courier(app)
            peers = courier.mailbox.list_peers()
            with courier.mailbox._get_connection() as conn:
                unread = conn.execute("SELECT COUNT(*) FROM mailbox_messages WHERE folder='inbox' AND is_read=0").fetchone()[0]
                quar = conn.execute("SELECT COUNT(*) FROM quarantine_sandbox WHERE status='quarantined'").fetchone()[0]
                outq = conn.execute("SELECT COUNT(*) FROM outbox_queue WHERE status='pending'").fetchone()[0]
            summary_lbl.config(
                text=(
                    f"Node ID: {courier.mailbox.identity['peer_id']}  |  "
                    f"Paired Peers: {len(peers)}  |  Unread Inbox: {unread}  |  "
                    f"Quarantine: {quar}  |  Outbox Queue: {outq}"
                )
            )
        except Exception as e:
            summary_lbl.config(text=f"Mailbox status unavailable: {e}")

    _refresh_p2p_summary()

    btn_row = tk.Frame(p2p_box, bg=THEME["bg_color"])
    btn_row.pack(fill=tk.X, pady=4)

    tk.Button(
        btn_row,
        text="🔗 Pair Friend / Invite Card",
        command=lambda: open_peer_pairing_dialog(win, app, on_close=_refresh_p2p_summary),
        bg=THEME["widget_bg_color"],
        fg=THEME.get("accent_highlight", "#00ffcc"),
        font=f_btn,
        padx=8,
    ).pack(side=tk.LEFT, padx=4)

    tk.Button(
        btn_row,
        text="📬 Open Serenity Mailbox & Quarantine",
        command=lambda: open_serenity_mailbox_dialog(win, app, on_close=_refresh_p2p_summary),
        bg=THEME["widget_bg_color"],
        fg=THEME.get("electric_blue", "#00bfff"),
        font=f_btn,
        padx=8,
    ).pack(side=tk.LEFT, padx=4)

    def _sync_now():
        res = _get_courier(app).poll_deaddrop_and_outbox()
        _refresh_p2p_summary()
        messagebox.showinfo(
            "Dead-Drop & Outbox Sync",
            f"Imported from DeadDrop: {res.get('imported_count', 0)}\nFlushed from Outbox: {res.get('flushed_outbox', 0)}",
            parent=win,
        )

    tk.Button(
        btn_row,
        text="🔄 Sync Dead-Drop & Outbox",
        command=_sync_now,
        bg=THEME["button_bg_color"],
        fg=THEME["fg_color"],
        font=f_sm,
        padx=6,
    ).pack(side=tk.LEFT, padx=4)


# ==============================================================================
# 4. INTERACTIVE MODALS: PEER PAIRING & SERENITY MAILBOX / QUARANTINE
# ==============================================================================

def open_peer_pairing_dialog(parent_win: tk.Widget, app: Any, on_close=None) -> tk.Toplevel:
    """Opens the Cross-State Peer Pairing & Invite Card modal."""
    courier = _get_courier(app)
    ident = courier.mailbox.identity

    dlg = tk.Toplevel(parent_win)
    dlg.title("Cross-State P2P Pairing & Invite Card")
    dlg.geometry("680x540")
    dlg.config(bg=THEME["bg_color"])
    dlg.transient(parent_win)

    fonts = getattr(app, "fonts", {}) or {}
    f_bold = fonts.get("bold", ("Segoe UI", 11, "bold"))
    f_lbl = fonts.get("ui_label", ("Segoe UI", 10))
    f_sm = fonts.get("ui_small", ("Segoe UI", 9))

    # Section 1: My Invite Card
    my_lf = tk.LabelFrame(dlg, text="1. Your Serenity Node Identity & Invite URI", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=f_bold, padx=8, pady=6)
    my_lf.pack(fill=tk.X, padx=10, pady=(10, 6))

    tk.Label(
        my_lf,
        text=f"Peer ID: {ident['peer_id']}   |   Ed25519/X25519 Fingerprint: {ident['fingerprint']}",
        bg=THEME["bg_color"],
        fg=THEME["fg_color"],
        font=f_sm,
    ).pack(anchor="w")

    ep_row = tk.Frame(my_lf, bg=THEME["bg_color"])
    ep_row.pack(fill=tk.X, pady=4)
    tk.Label(ep_row, text="Optional Live Endpoint / Tailscale URL:", bg=THEME["bg_color"], fg=THEME["fg_color"], font=f_sm).pack(side=tk.LEFT)
    ep_var = tk.StringVar(value="")
    ep_ent = tk.Entry(ep_row, textvariable=ep_var, bg=THEME["widget_bg_color"], fg=THEME["fg_color"], width=34)
    ep_ent.pack(side=tk.LEFT, padx=6)

    invite_var = tk.StringVar(value="")

    def _regen_invite(discover_stun: bool = False):
        if discover_stun and not ep_var.get().strip():
            wan = CrossStateP2PTransport.discover_wan_endpoint()
            if wan:
                ep_var.set(f"https://{wan}")
        uri = CrossStateP2PTransport.generate_invite_code(
            peer_id=ident["peer_id"],
            display_name=ident["display_name"],
            identity_keys=ident,
            endpoint_url=ep_var.get().strip(),
            app_edition=ident.get("app_edition", "SerenityPC-Main"),
            dead_drop_hint=os.path.join(courier.mailbox.mailbox_dir, "DeadDrop"),
        )
        invite_var.set(uri)

    _regen_invite(discover_stun=False)

    tk.Button(ep_row, text="🌐 STUN WAN Lookup", command=lambda: _regen_invite(True), bg=THEME["button_bg_color"], fg=THEME["fg_color"], font=f_sm).pack(side=tk.LEFT, padx=4)

    inv_ent = tk.Entry(my_lf, textvariable=invite_var, bg=THEME["widget_bg_color"], fg="#00ffcc", font=("Consolas", 9))
    inv_ent.pack(fill=tk.X, pady=4)

    def _copy_invite():
        _regen_invite(False)
        dlg.clipboard_clear()
        dlg.clipboard_append(invite_var.get())
        messagebox.showinfo("Copied", "Your signed 'serenity-peer://' Invite URI has been copied to the clipboard!", parent=dlg)

    tk.Button(my_lf, text="📋 Copy My Invite URI to Clipboard", command=_copy_invite, bg=THEME["widget_bg_color"], fg="#00ffcc", font=f_lbl).pack(anchor="w", pady=2)

    # Section 2: Accept Friend's Invite URI
    pair_lf = tk.LabelFrame(dlg, text="2. Pair a Friend's Serenity Node (Paste 'serenity-peer://' URI)", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=f_bold, padx=8, pady=6)
    pair_lf.pack(fill=tk.X, padx=10, pady=6)

    paste_var = tk.StringVar(value="")
    paste_ent = tk.Entry(pair_lf, textvariable=paste_var, bg=THEME["widget_bg_color"], fg=THEME["fg_color"], font=("Consolas", 9))
    paste_ent.pack(fill=tk.X, pady=4)

    # Section 3: Paired Peers List
    list_lf = tk.LabelFrame(dlg, text="3. Paired Peers in Address Book", bg=THEME["bg_color"], fg=THEME["electric_blue"], font=f_bold, padx=8, pady=6)
    list_lf.pack(fill=tk.BOTH, expand=True, padx=10, pady=(6, 10))

    peers_box = tk.Listbox(list_lf, bg=THEME["widget_bg_color"], fg=THEME["fg_color"], font=("Consolas", 10), height=6)
    peers_box.pack(fill=tk.BOTH, expand=True, pady=4)

    def _refresh_peers():
        peers_box.delete(0, tk.END)
        for p in courier.mailbox.list_peers():
            peers_box.insert(
                tk.END,
                f"{p['display_name']} ({p['peer_id']}) | {p['app_edition']} | FP: {p['fingerprint'][:12]}... | Trust: {p['trust_status']}",
            )

    def _do_pair():
        raw = paste_var.get().strip()
        if not raw:
            messagebox.showwarning("Missing URI", "Paste a 'serenity-peer://' invite URI first.", parent=dlg)
            return
        try:
            peer = courier.mailbox.register_peer_from_invite(raw)
            paste_var.set("")
            _refresh_peers()
            if callable(on_close):
                on_close()
            messagebox.showinfo("Peer Paired", f"Successfully verified and paired with {peer['display_name']} ({peer['peer_id']})!", parent=dlg)
        except Exception as e:
            messagebox.showerror("Pairing Error", f"Invalid or tampered invite URI:\n{e}", parent=dlg)

    tk.Button(pair_lf, text="✔ Verify Signature & Pair Friend", command=_do_pair, bg=THEME["button_active_color"], fg=THEME["fg_color"], font=f_lbl).pack(anchor="w", pady=2)

    _refresh_peers()
    return dlg


def open_serenity_mailbox_dialog(parent_win: tk.Widget, app: Any, on_close=None) -> tk.Toplevel:
    """Opens the Serenity Encrypted Mailbox, Quarantine Sandbox & Compose modal."""
    courier = _get_courier(app)
    courier.poll_deaddrop_and_outbox()

    dlg = tk.Toplevel(parent_win)
    dlg.title(f"Serenity Encrypted Mailbox — {SerenityEnvResolver.get_active_username(app)}")
    dlg.geometry("780x580")
    dlg.config(bg=THEME["bg_color"])
    dlg.transient(parent_win)

    fonts = getattr(app, "fonts", {}) or {}
    f_bold = fonts.get("bold", ("Segoe UI", 11, "bold"))
    f_sm = fonts.get("ui_small", ("Segoe UI", 9))

    top_bar = tk.Frame(dlg, bg=THEME["bg_color"])
    top_bar.pack(fill=tk.X, padx=10, pady=(8, 4))

    folder_var = tk.StringVar(value="inbox")
    for f_val, f_txt in [("inbox", "📥 Inbox"), ("quarantine", "🛡️ Quarantine Sandbox"), ("sent", "📤 Sent"), ("outbox", "⏳ Outbox Queue")]:
        tk.Radiobutton(
            top_bar,
            text=f_txt,
            variable=folder_var,
            value=f_val,
            indicatoron=False,
            bg=THEME["widget_bg_color"],
            fg=THEME["fg_color"],
            selectcolor=THEME["electric_blue"],
            font=f_sm,
            padx=8,
            pady=3,
        ).pack(side=tk.LEFT, padx=3)

    pane = tk.PanedWindow(dlg, orient=tk.VERTICAL, bg=THEME["bg_color"], sashwidth=4)
    pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

    item_list = tk.Listbox(pane, bg=THEME["widget_bg_color"], fg=THEME["fg_color"], font=("Consolas", 10), height=8)
    pane.add(item_list)

    detail_txt = scrolledtext.ScrolledText(pane, bg=THEME["widget_bg_color"], fg=THEME["fg_color"], font=("Consolas", 10), height=12)
    pane.add(detail_txt)

    current_items = []

    def _load_folder(*args):
        nonlocal current_items
        item_list.delete(0, tk.END)
        detail_txt.delete("1.0", tk.END)
        folder = folder_var.get()
        with courier.mailbox._get_connection() as conn:
            if folder == "quarantine":
                rows = conn.execute(
                    "SELECT * FROM quarantine_sandbox WHERE status='quarantined' ORDER BY quarantined_at DESC LIMIT 50"
                ).fetchall()
                current_items = [dict(r) for r in rows]
                for r in current_items:
                    item_list.insert(
                        tk.END,
                        f"[Q#{r['id']}] From: {r['sender_name']} | Risk: {r['risk_score']} ({r['intent_label']}) | Subj: {r['subject']}",
                    )
            elif folder == "outbox":
                rows = conn.execute(
                    "SELECT * FROM outbox_queue WHERE status='pending' ORDER BY created_at DESC LIMIT 50"
                ).fetchall()
                current_items = [dict(r) for r in rows]
                for r in current_items:
                    item_list.insert(
                        tk.END,
                        f"[OUT#{r['id']}] To: {r['peer_id']} | Retries: {r['retry_count']} | File: {os.path.basename(r['smp_file_path'])}",
                    )
            else:
                rows = conn.execute(
                    "SELECT * FROM mailbox_messages WHERE folder=? ORDER BY timestamp DESC LIMIT 50",
                    (folder,),
                ).fetchall()
                current_items = [dict(r) for r in rows]
                for r in current_items:
                    item_list.insert(
                        tk.END,
                        f"[#{r['id']}] {r['sender_name']} -> Lvl {r['target_level']} | {r['subject']}",
                    )

    def _on_select(event=None):
        sel = item_list.curselection()
        if not sel or sel[0] >= len(current_items):
            return
        item = current_items[sel[0]]
        detail_txt.delete("1.0", tk.END)
        detail_txt.insert(tk.END, json.dumps(item, indent=2, ensure_ascii=False))

    item_list.bind("<<ListboxSelect>>", _on_select)
    folder_var.trace_add("write", _load_folder)

    act_bar = tk.Frame(dlg, bg=THEME["bg_color"])
    act_bar.pack(fill=tk.X, padx=10, pady=(4, 10))

    def _moderate_selected(action: str):
        if folder_var.get() != "quarantine":
            messagebox.showinfo("Quarantine Action", "Switch to the 'Quarantine Sandbox' tab to admit or reject held packets.", parent=dlg)
            return
        sel = item_list.curselection()
        if not sel or sel[0] >= len(current_items):
            return
        q_id = current_items[sel[0]]["id"]
        courier.resolve_quarantine_item(q_id, action=action, reason=f"User {action} via Mailbox UI")
        _load_folder()
        if callable(on_close):
            on_close()

    def _import_smp_file():
        fpath = filedialog.askopenfilename(
            parent=dlg,
            title="Import Encrypted Serenity Mail (.smp / .bmb)",
            filetypes=[("Serenity & Brainuke Mail", "*.smp *.bmb"), ("All Files", "*.*")],
        )
        if not fpath:
            return
        try:
            with open(fpath, "r", encoding="utf-8") as fp:
                raw_b64 = fp.read().strip()
            res = courier.receive_smp_packet(raw_b64)
            _load_folder()
            if callable(on_close):
                on_close()
            messagebox.showinfo("Packet Imported", f"Result: {res.get('status')}\nSender: {res.get('sender')}", parent=dlg)
        except Exception as e:
            messagebox.showerror("Import Error", f"Failed to decrypt/verify packet:\n{e}", parent=dlg)

    tk.Button(act_bar, text="✔ Admit Quarantined to Inbox", command=lambda: _moderate_selected("admit"), bg=THEME["widget_bg_color"], fg="#00ff88", font=f_sm, padx=6).pack(side=tk.LEFT, padx=3)
    tk.Button(act_bar, text="✖ Reject Quarantined", command=lambda: _moderate_selected("reject"), bg=THEME["widget_bg_color"], fg="#ff6666", font=f_sm, padx=6).pack(side=tk.LEFT, padx=3)
    tk.Button(act_bar, text="📥 Import .smp / .bmb File", command=_import_smp_file, bg=THEME["button_bg_color"], fg=THEME["fg_color"], font=f_sm, padx=6).pack(side=tk.LEFT, padx=3)
    tk.Button(act_bar, text="📂 Open Dead-Drop Folder", command=lambda: os.startfile(os.path.join(courier.mailbox.mailbox_dir, "DeadDrop")) if hasattr(os, "startfile") else None, bg=THEME["button_bg_color"], fg=THEME["fg_color"], font=f_sm, padx=6).pack(side=tk.RIGHT, padx=3)

    _load_folder()
    return dlg
