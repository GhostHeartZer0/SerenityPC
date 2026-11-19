import json
import os
import sys
import tkinter as tk
from tkinter import ttk
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import main as app_mod
from System import settings_tabs
from System.serenity_utils import FileAndWidgetLogger, WidgetLogger
from System.tool_registry import GemmaToolRegistry


def _make_stub_app(tmp_path):
    app = app_mod.ChatbotApp.__new__(app_mod.ChatbotApp)
    app.root = None
    app.script_dir = str(tmp_path)
    app.dirs = {
        "History": str(tmp_path / "History"),
        "Logs": str(tmp_path / "Logs"),
        "System": str(tmp_path / "System"),
        "Users": str(tmp_path / "Users"),
    }
    for d in app.dirs.values():
        os.makedirs(d, exist_ok=True)
    app.config = {
        "history_usage": "all",
        "history_mode": "Keyword",
        "reasoning_strength": "medium",
        "muse_reasoning_strength": "medium",
        "response_length": "natural",
        "k_cache_type": "q8_0",
        "v_cache_type": "q8_0",
        "max_token_ratio": 4,
        "show_extended_kv_types": False,
        "expanded_file_access": True,
        "quick_settings_items": ["reasoning_strength", "response_length", "kv_cache", "headroom"],
    }
    app.active_persona_level = 2
    app.messages = []
    app.vault_manager = None
    app.fonts = {
        "ui_small": ("Segoe UI", 9),
        "ui_label": ("Segoe UI", 10),
        "ui_button": ("Segoe UI", 10),
        "bold": ("Segoe UI", 10, "bold"),
        "stats": ("Consolas", 8),
    }
    app.save_config = lambda: None
    app._log_and_display = lambda msg: None
    app.get_user_history_dir = lambda: app.dirs["History"]
    return app


def test_fix1_media_synthesis_section_not_duplicated():
    src = (ROOT_DIR / "System" / "settings_tabs.py").read_text(encoding="utf-8")
    assert src.count("Image, GIF & Video Synthesis") == 1


def test_fix2_quick_settings_two_row_label_and_configurable_popup(tmp_path):
    main_src = (ROOT_DIR / "main.py").read_text(encoding="utf-8")
    assert '"Quick\\nSettings"' in main_src

    root = tk.Tk()
    root.withdraw()
    try:
        app = _make_stub_app(tmp_path)
        app.root = root
        app.btn_sizes = tk.Button(root, text="Quick\nSettings")
        app.btn_sizes.pack()

        # Default 4 items: Reasoning Strength, Response Length, KV Cache, Headroom
        app.toggle_sizes_popup()
        assert app.sizes_popup is not None and app.sizes_popup.winfo_exists()

        labels = []
        combos = []
        def _walk(w):
            for c in w.winfo_children():
                if isinstance(c, tk.Label):
                    labels.append(c.cget("text"))
                elif isinstance(c, ttk.Combobox):
                    combos.append(c)
                _walk(c)
        _walk(app.sizes_popup)
        assert len(combos) == 4
        assert "KV Cache" in labels
        assert "Headroom" in labels
        assert "History" not in labels

        # Change KV cache & Headroom via comboboxes
        combos[2].set("q4_0")
        combos[3].set("Deep (2)")
        app._sizes_save_and_close()
        assert app.config["k_cache_type"] == "q4_0"
        assert app.config["v_cache_type"] == "q4_0"
        assert app.config["max_token_ratio"] == 2

        # Reconfigure to include History Mode and exclude Reasoning Strength
        app.config["quick_settings_items"] = ["kv_cache", "history_mode"]
        app.config["show_extended_kv_types"] = True
        app.toggle_sizes_popup()
        labels2, combos2 = [], []
        def _walk2(w):
            for c in w.winfo_children():
                if isinstance(c, tk.Label):
                    labels2.append(c.cget("text"))
                elif isinstance(c, ttk.Combobox):
                    combos2.append(c)
                _walk2(c)
        _walk2(app.sizes_popup)
        assert len(combos2) == 2
        assert "KV Cache" in labels2 and "History" in labels2
        assert "turbo3_tcq" in list(combos2[0].cget("values"))
        app._sizes_save_and_close()
    finally:
        root.destroy()


def test_fix3_lvl2_no_irrelevant_history_and_no_40k_prefill_bloat(tmp_path):
    app = _make_stub_app(tmp_path)
    hist_dir = app.dirs["History"]

    # Create massive unrelated past chat (~160k chars -> ~40k tokens)
    huge_unrelated = [
        {"role": "user", "content": "Explain medieval French bakery regulations in detail."},
        {"role": "assistant", "content": ("Baguette flour yeast oven regulation. " * 4000)},
    ]
    fp = os.path.join(hist_dir, "old_huge_chat.history.jsonz")
    with open(fp, "wb") as f:
        f.write(zlib.compress(json.dumps(huge_unrelated).encode("utf-8")))

    # Level 2 short unrelated sentence -> MUST NOT pull irrelevant history or 40k tokens
    app.active_persona_level = 2
    app.messages = [{"role": "user", "content": "What is the capital of Japan?"}]
    ctx = app._get_contextual_history_messages()
    assert len(ctx) == 1
    assert ctx[0]["content"] == "What is the capital of Japan?"

    # Level 2 relevant query -> pulls truncated snippet strictly capped well under 40k
    app.messages = [{"role": "user", "content": "Tell me about medieval bakery regulations."}]
    ctx_rel = app._get_contextual_history_messages()
    assert len(ctx_rel) >= 2
    total_chars = sum(len(str(m.get("content", ""))) for m in ctx_rel)
    assert total_chars < 3000

    # Level 3 with huge unrelated archive -> still strictly capped under 6500 chars
    app.active_persona_level = 3
    app.messages = [{"role": "user", "content": "Quick hello"}]
    ctx_lvl3 = app._get_contextual_history_messages()
    total_chars_lvl3 = sum(len(str(m.get("content", ""))) for m in ctx_lvl3)
    assert total_chars_lvl3 < 6500


def test_fix4_expanded_file_access_default_and_local_file_read(tmp_path):
    main_src = (ROOT_DIR / "main.py").read_text(encoding="utf-8")
    assert '"expanded_file_access": False' in main_src
    assert "[LOCAL FILE ACCESS]:" in main_src

    app = _make_stub_app(tmp_path)
    # Create a file outside workspace/Logs (explicitly opted into expanded_file_access in stub)
    sample = tmp_path / "notes.txt"
    sample.write_text("Hello local file access!", encoding="utf-8")

    # Default (expanded_file_access=False) blocks outside workspace when no roots configured
    app.config["expanded_file_access"] = False
    reg_closed = GemmaToolRegistry(app)
    res_blocked = reg_closed.registry.execute("read_file", {"path": str(sample.parent.parent)})
    assert "SECURITY RESTRICTION" in res_blocked

    # Opt-in (expanded_file_access=True) allows non-sensitive local file reads
    app.config["expanded_file_access"] = True
    reg = GemmaToolRegistry(app)
    res_read = reg.registry.execute("read_file", {"path": str(sample)})
    assert "Hello local file access!" in res_read
    assert "SECURITY RESTRICTION" not in res_read

    # Protected secret files remain strictly blocked
    env_file = tmp_path / ".env"
    env_file.write_text("SECRET=123", encoding="utf-8")
    res_env = reg.registry.execute("read_file", {"path": str(env_file)})
    assert "SECURITY RESTRICTION" in res_env


def test_fix5_chat_template_collapses_multiline_and_stderr_logger(tmp_path):
    root = tk.Tk()
    root.withdraw()
    try:
        app = _make_stub_app(tmp_path)
        app.root = root
        app.error_log = tk.Text(root)
        app.error_log.pack()

        # Route FileAndWidgetLogger with chat_template_handler just like main.py
        log_file = str(tmp_path / "Logs" / "err.log")
        logger = FileAndWidgetLogger(
            app.error_log,
            log_file,
            "stderr",
            chat_template_handler=lambda txt: app._buffer_log(txt, is_chat_template=True),
        )

        # Simulate multi-line llama.cpp chat_template dump over stderr
        logger.write("llm_load_print_meta: chat_template = {%- for message in messages %}\n")
        logger.write("    {{ message['content'] }}\n")
        logger.write("{%- endfor %}\n")
        root.update()

        # Should create ONE collapsed ct_* tag with elide=True and ONE toggle button
        ct_tags = [t for t in app.error_log.tag_names() if t.startswith("ct_")]
        assert len(ct_tags) == 1
        ct_tag = ct_tags[0]
        assert str(app.error_log.tag_cget(ct_tag, "elide")) in ("1", "True", "true")

        # Verify all 3 lines are inside the collapsed tag
        ranges = app.error_log.tag_ranges(ct_tag)
        tagged_text = "".join(
            app.error_log.get(ranges[i], ranges[i + 1]) for i in range(0, len(ranges), 2)
        )
        assert "chat_template" in tagged_text
        assert "message['content']" in tagged_text
        assert "endfor" in tagged_text

        # Clicking toggle button expands, clicking again re-collapses
        btns = [w for w in app.error_log.winfo_children() if isinstance(w, tk.Button)]
        assert len(btns) == 1
        assert "Show Chat Template" in btns[0].cget("text")
        btns[0].invoke()
        assert str(app.error_log.tag_cget(ct_tag, "elide")) in ("0", "False", "false", "")
        assert "Collapse Chat Template" in btns[0].cget("text")
        btns[0].invoke()
        assert str(app.error_log.tag_cget(ct_tag, "elide")) in ("1", "True", "true")
    finally:
        root.destroy()
