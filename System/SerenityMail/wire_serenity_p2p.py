# wire_serenity_p2p.py
"""
Idempotent Installer & Environment Verifier for Serenity P2P Messaging, Mail & V2.3.0 MCP.
Works across:
  - SerenityPC (Main) & SerenityPC Legacy (v1.8.0+)
  - Standalone Source Folder (`main.py` + `System/`)
  - PyInstaller `--onedir` distribution (`SerenityPC.exe` + `_internal/System/`)

Usage:
  python wire_serenity_p2p.py [--target-dir PATH]
"""

import os
import sys
import shutil
import argparse
import py_compile


def backup_and_write(filepath: str, new_content: str) -> bool:
    """Creates a `.bak` backup if one doesn't exist and writes `new_content` atomically."""
    bak_path = filepath + ".p2p.bak"
    if not os.path.exists(bak_path):
        shutil.copy2(filepath, bak_path)
    with open(filepath, "w", encoding="utf-8") as fp:
        fp.write(new_content)
    if filepath.endswith(".py"):
        py_compile.compile(filepath, doraise=True)
    return True


def patch_main_py(main_path: str) -> str:
    if not os.path.isfile(main_path):
        return "skipped (main.py not found)"
    with open(main_path, "r", encoding="utf-8") as fp:
        code = fp.read()
    if "wire_serenity_app" in code:
        return "already wired"

    anchor = '            "cleanup": lambda msg: self._run_hygiene_on_main_thread()\n        }'
    alt_anchor = '            "cleanup": lambda msg: self._run_hygiene_on_main_thread()\r\n        }'
    hook_code = (
        "\n        try:\n"
        "            from System.serenity_p2p_ui_wiring import wire_serenity_app\n"
        "            wire_serenity_app(self)\n"
        "        except Exception as _p2p_e:\n"
        "            print(f\"[P2P] Wiring notice: {_p2p_e}\")"
    )

    if anchor in code:
        code = code.replace(anchor, anchor + hook_code, 1)
        backup_and_write(main_path, code)
        return "patched (explicit queue_handlers hook added)"
    elif alt_anchor in code:
        code = code.replace(alt_anchor, alt_anchor + hook_code.replace("\n", "\r\n"), 1)
        backup_and_write(main_path, code)
        return "patched (explicit CRLF queue_handlers hook added)"
    return "auto-hooked via GemmaToolRegistry.__init__"


def patch_settings_tabs_py(st_path: str) -> str:
    if not os.path.isfile(st_path):
        return "skipped (settings_tabs.py not found)"
    with open(st_path, "r", encoding="utf-8") as fp:
        code = fp.read()
    if "inject_agents_tab_p2p_controls" in code and "inject_users_tab_p2p_controls" in code:
        return "already wired"

    modified = False
    for nl in ("\n", "\r\n"):
        a_anchor = f"    bind_radio_contrast(norm_fmt_var, norm_fmt_rbs){nl}{nl}    return frame"
        if a_anchor in code and "inject_agents_tab_p2p_controls" not in code:
            a_rep = (
                f"    bind_radio_contrast(norm_fmt_var, norm_fmt_rbs){nl}{nl}"
                f"    try:{nl}"
                f"        from System.serenity_p2p_ui_wiring import inject_agents_tab_p2p_controls{nl}"
                f"        inject_agents_tab_p2p_controls(frame, app, win, vars_dict, bind_radio_contrast_fn=bind_radio_contrast){nl}"
                f"    except Exception as _p2p_err:{nl}"
                f"        print(f\"[SETTINGS] P2P Agents tab hook notice: {{_p2p_err}}\"){nl}{nl}"
                f"    return frame"
            )
            code = code.replace(a_anchor, a_rep, 1)
            modified = True

        u_anchor = (
            f"    for p_sec, p_lbl in [(0, \"Off\"), (15, \"15s\"), (30, \"30s\"), (45, \"45s\"), (300, \"5m\"), (900, \"15m\"), (1800, \"30m\")]:{nl}"
            f"        btn = tk.Button(timer_row, text=p_lbl, command=lambda s=p_sec: vars_dict[\"auto_lock_var\"].set(str(s)),{nl}"
            f"                        bg=THEME[\"widget_bg_color\"], fg=THEME[\"fg_color\"], font=app.fonts[\"ui_small\"], padx=3, pady=0){nl}"
            f"        btn.pack(side=tk.LEFT, padx=2){nl}{nl}"
            f"    return frame"
        )
        if u_anchor in code and "inject_users_tab_p2p_controls" not in code:
            u_rep = u_anchor.replace(
                f"{nl}    return frame",
                f"{nl}    try:{nl}"
                f"        from System.serenity_p2p_ui_wiring import inject_users_tab_p2p_controls{nl}"
                f"        inject_users_tab_p2p_controls(frame, app, win, vars_dict, vault_lf=vault_lf){nl}"
                f"    except Exception as _p2p_err:{nl}"
                f"        print(f\"[SETTINGS] P2P Users tab hook notice: {{_p2p_err}}\"){nl}{nl}"
                f"    return frame",
                1,
            )
            code = code.replace(u_anchor, u_rep, 1)
            modified = True

    if modified:
        backup_and_write(st_path, code)
        return "patched (Tab 3 & Tab 5 controls injected)"
    return "skipped (custom layout)"


def patch_settings_ui_py(sui_path: str) -> str:
    if not os.path.isfile(sui_path):
        return "skipped (settings_ui.py not found)"
    with open(sui_path, "r", encoding="utf-8") as fp:
        code = fp.read()
    if "init_p2p_settings_vars" in code and "save_p2p_settings_vars" in code:
        return "already wired"

    modified = False
    for nl in ("\n", "\r\n"):
        v_anchor = f'        vars_dict["turbovec_mode_var"] = vars_dict["history_mode_var"]'
        if v_anchor in code and "init_p2p_settings_vars" not in code:
            v_rep = (
                f"{v_anchor}{nl}"
                f"        try:{nl}"
                f"            from System.serenity_p2p_ui_wiring import init_p2p_settings_vars{nl}"
                f"            init_p2p_settings_vars(app, vars_dict){nl}"
                f"        except Exception:{nl}"
                f"            pass"
            )
            code = code.replace(v_anchor, v_rep, 1)
            modified = True

        s_anchor = f"            app.save_config(){nl}            if draft_toggled"
        if s_anchor in code and "save_p2p_settings_vars" not in code:
            s_rep = (
                f"            try:{nl}"
                f"                from System.serenity_p2p_ui_wiring import save_p2p_settings_vars{nl}"
                f"                save_p2p_settings_vars(app, vars_dict){nl}"
                f"            except Exception:{nl}"
                f"                pass{nl}"
                f"{s_anchor}"
            )
            code = code.replace(s_anchor, s_rep, 1)
            modified = True

    if modified:
        backup_and_write(sui_path, code)
        return "patched (init & save hooks added)"
    return "skipped"


def patch_spec_file(spec_path: str) -> str:
    if not os.path.isfile(spec_path):
        return "skipped (SerenityPC.spec not found)"
    with open(spec_path, "r", encoding="utf-8") as fp:
        code = fp.read()
    if "System.serenity_mail_courier" in code:
        return "already wired"

    anchor = "for optional_mod in ['numpy', 'requests']:"
    if anchor in code:
        rep = (
            "hiddenimports += [\n"
            "    'sqlite3',\n"
            "    '_sqlite3',\n"
            "    '_cffi_backend',\n"
            "    'System.serenity_mail_courier',\n"
            "    'System.serenity_p2p_ui_wiring',\n"
            "    'System.tool_registry',\n"
            "]\n"
            "for optional_mod in ['numpy', 'requests', 'cryptography']:"
        )
        code = code.replace(anchor, rep, 1)
        backup_and_write(spec_path, code)
        return "patched (onedir cryptography & System.serenity_mail_courier hiddenimports added)"
    return "skipped"


def main():
    parser = argparse.ArgumentParser(description="Wire Serenity P2P Mail, Courier & V2.3.0 MCP into SerenityPC.")
    parser.add_argument("--target-dir", default=".", help="Path to SerenityPC root folder (standalone or onedir).")
    args = parser.parse_args()

    target_root = os.path.abspath(args.target_dir)
    print(f"=== Serenity P2P & V2.3.0 MCP Environment Wirer ===")
    print(f"Target Root: {target_root}")

    # Detect standalone vs onedir layout
    internal_sys = os.path.join(target_root, "_internal", "System")
    direct_sys = os.path.join(target_root, "System")
    is_onedir_dist = os.path.isdir(internal_sys)

    print(f"Detected Layout: {'PyInstaller --onedir Distribution (_internal/System)' if is_onedir_dist else 'Standalone Source Folder (System/)'}")

    # Ensure writable directories exist at target_root (never inside _internal)
    for d in ("Users", os.path.join("Users", "Default", "Mailbox", "DeadDrop"), "History", "Logs", ".tmp"):
        os.makedirs(os.path.join(target_root, d), exist_ok=True)

    # Copy core modules if running from extracted bundle
    script_here = os.path.dirname(os.path.abspath(__file__))
    for sys_target in ([direct_sys, internal_sys] if is_onedir_dist else [direct_sys]):
        os.makedirs(sys_target, exist_ok=True)
        for mod_name in ("serenity_mail_courier.py", "tool_registry.py", "serenity_p2p_ui_wiring.py"):
            src_cand = os.path.join(script_here, "System", mod_name)
            if not os.path.isfile(src_cand):
                src_cand = os.path.join(script_here, mod_name)
            dst_cand = os.path.join(sys_target, mod_name)
            if os.path.isfile(src_cand) and os.path.abspath(src_cand) != os.path.abspath(dst_cand):
                shutil.copy2(src_cand, dst_cand)
                print(f"  [INSTALLED] {dst_cand}")

    active_sys = internal_sys if (is_onedir_dist and not os.path.isfile(os.path.join(direct_sys, "settings_tabs.py"))) else direct_sys
    print(f"  [MAIN.PY]          {patch_main_py(os.path.join(target_root, 'main.py'))}")
    print(f"  [SETTINGS_TABS.PY] {patch_settings_tabs_py(os.path.join(active_sys, 'settings_tabs.py'))}")
    print(f"  [SETTINGS_UI.PY]   {patch_settings_ui_py(os.path.join(active_sys, 'settings_ui.py'))}")
    print(f"  [SERENITYPC.SPEC]  {patch_spec_file(os.path.join(target_root, 'SerenityPC.spec'))}")
    print("=== Wiring Complete! ===")


if __name__ == "__main__":
    main()
