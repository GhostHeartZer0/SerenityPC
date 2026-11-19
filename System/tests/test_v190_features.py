import json
import os
import queue
import sys
import tkinter as tk
import types
import warnings
import zlib
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
SYS_DIR = os.path.join(ROOT_DIR, "System")
if SYS_DIR not in sys.path:
    sys.path.insert(0, SYS_DIR)

import main
from System.agentic_monitor import AgenticMonitor, format_auto_label, is_text_readable, next_auto_label_for_dir
from System.kv_manager import TurboVecIndex
from System.semantic_vault import SemanticVault
from System.serenity_p2p_ui_wiring import handle_subagent_spawn, handle_subagent_stream
from System.serenity_utils import WidgetLogger
from System.tool_registry import GemmaToolRegistry


def _make_stub_app(tmp_path):
    app = main.ChatbotApp.__new__(main.ChatbotApp)
    app.script_dir = str(tmp_path)
    app.dirs = {
        "System": str(tmp_path / "System"),
        "History": str(tmp_path / "Users" / "Tester" / "History"),
    }
    os.makedirs(app.dirs["System"], exist_ok=True)
    os.makedirs(app.dirs["History"], exist_ok=True)
    app.get_user_history_dir = lambda: app.dirs["History"]
    app.get_active_username = lambda: "Tester"
    app.config = {
        "history_usage": "all",
        "history_lookup_mode": "targeted",
        "scroll_lock_enabled": False,
        "media_rendering": 0,
        "inline_markdown": False,
        "show_cecilia_slider": False,
        "expanded_file_access": False,
        "chat_templates": {},
    }
    app.state = {"running": False, "deep_cook": False, "dmn_backbone": {}, "was_halted": False}
    app.messages = []
    app.active_persona_level = 3
    app.current_model_tier = "low"
    app.model_path = r"C:\models\gemma-4-26B.gguf"
    app.model_paths = {"low": app.model_path}
    app.model = None
    app.chat_history = None
    app.error_log = None
    app.process_queue = queue.Queue()
    app.fonts = {"stats": ("Segoe UI", 9), "log": ("Consolas", 9)}
    app.chunk_counter = 0
    app._history_content_cache = {}
    app.get_history_path = lambda *a, **kw: None
    app._save_dmn_backbone = lambda: None
    app._log_and_display = lambda msg: None
    app.stop_process = types.SimpleNamespace(is_set=lambda: False)
    return app


def test_step1_blocking_inference_accepts_stream_callback():
    import inspect
    sig = inspect.signature(main.ChatbotApp._run_blocking_inference)
    assert "stream_callback" in sig.parameters


def test_step2_widget_logger_preserves_chat_template_and_routes_traceback():
    fake_stderr = MagicMock()
    logger = WidgetLogger(None, tag="stdout")
    logger.original_stream = MagicMock()

    with patch.object(sys, "stderr", fake_stderr):
        logger.write("chat_template: {{ bos_token }}")
        logger.original_stream.write.assert_called()
        logger.write("Traceback (most recent call last):\n  File x.py")
        fake_stderr.write.assert_called_once()


def test_step2_buffer_log_collapses_chat_template_in_error_log():
    root = tk.Tk()
    root.withdraw()
    try:
        app = main.ChatbotApp.__new__(main.ChatbotApp)
        err_w = tk.Text(root, height=10, width=60)
        app.error_log = err_w
        app.error_log_buffer = []
        app.root = root
        app.fonts = {"stats": ("Segoe UI", 9)}
        app._buffer_log("[CHAT TEMPLATE] {% for msg in messages %}\n{{ msg.content }}\n{% endfor %}", is_chat_template=True)
        tag_names = err_w.tag_names()
        tpl_tags = [t for t in tag_names if t.startswith("ct_")]
        assert tpl_tags, "Expected collapsible ct_ tag"
        assert str(err_w.tag_cget(tpl_tags[0], "elide")) in ("1", "True", "true")
    finally:
        root.destroy()


def test_step3_analyze_charts_no_warning(tmp_path):
    from benchmarks.wringer import analyze
    rep = tmp_path / "demo-model_report.md"
    rep.write_text(
        "# Report\n\n## lvl1\n- **Quality Score**: 8.5/10 (85.0%)\nDecode: 22.0 t/s\n\n"
        "## lvl2\n- **Quality Score**: 9.5/10 (95.0%)\nDecode: 18.0 t/s\n",
        encoding="utf-8",
    )
    results = analyze.collect_results(str(tmp_path))
    assert "demo-model" in results
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        c_path = analyze.generate_consolidated_chart(results, str(tmp_path), open_chart=False)
        v_path = analyze.generate_vertical_pct_chart(results, str(tmp_path), open_chart=False)
    assert os.path.isfile(c_path)
    assert os.path.isfile(v_path)
    tight_warnings = [w for w in caught if "tight_layout" in str(w.message).lower()]
    assert not tight_warnings


def test_step4_expanded_file_tools_and_terminal_drawer(tmp_path):
    work_dir = tmp_path / "project"
    work_dir.mkdir()
    app = _make_stub_app(tmp_path)
    app.config["expanded_file_access"] = True
    app.config["expanded_file_roots"] = str(work_dir)
    app.active_persona_level = 7

    with patch("System.tool_registry.SerenityEnvResolver.get_writable_root", return_value=str(work_dir)), \
         patch("System.tool_registry.SerenityEnvResolver.get_bundle_root", return_value=str(work_dir)):
        reg = GemmaToolRegistry(app)
        reg_keys = reg.registry.list_keys()
        assert "write_file" in reg_keys
        assert "list_directory" in reg_keys
        assert "organize_files" in reg_keys
        assert "control_serenitydev" in reg_keys
        assert "run_terminal_command" in reg_keys
        lvl7_names = {d["function"]["name"] for d in reg.get_definitions(7)}
        assert {"write_file", "list_directory", "organize_files", "control_serenitydev", "run_terminal_command"}.issubset(lvl7_names)

        site_index = work_dir / "site" / "index.html"
        res_w = reg.registry.execute("write_file", {"path": str(site_index), "content": "<h1>Hi</h1>"})
        assert "Success:" in res_w and site_index.read_text(encoding="utf-8") == "<h1>Hi</h1>"

        res_ls = reg.registry.execute("list_directory", {"path": str(work_dir / "site")})
        assert "index.html" in res_ls

        moved_index = work_dir / "dist" / "index.html"
        res_mv = reg.registry.execute(
            "organize_files",
            {"action": "move", "source_path": str(site_index), "destination_path": str(moved_index)},
        )
        assert "Success:" in res_mv and moved_index.is_file()

        # Sensitive credential/crypto files stay strictly blocked even with expanded access
        res_env = reg.registry.execute("write_file", {"path": str(work_dir / ".env"), "content": "SECRET=1"})
        assert "SECURITY RESTRICTION" in res_env
        res_encz = reg.registry.execute("write_file", {"path": str(work_dir / "x.encz"), "content": "bad"})
        assert "SECURITY RESTRICTION" in res_encz

    # Terminal drawer toggle expands and collapses cleanly
    root = tk.Tk()
    root.withdraw()
    try:
        hist = tk.Text(root, height=24)
        drawer = tk.Frame(root)
        body = tk.Frame(drawer)
        btn = tk.Button(drawer, text=">_ Terminal ▲")
        hist.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        drawer.pack(side=tk.BOTTOM, fill=tk.X, expand=False)
        app.chat_history = hist
        app.terminal_drawer_frame = drawer
        app.terminal_body_frame = body
        app.terminal_toggle_btn = btn
        app.terminal_entry = tk.Entry(body)
        app._orig_chat_height = 24
        app.active_tab = "active"
        app.toggle_terminal_drawer(expand=True)
        assert app._terminal_expanded is True
        assert int(hist.cget("height")) == 4
        assert "▼" in btn.cget("text")
        app.toggle_terminal_drawer(expand=False)
        assert app._terminal_expanded is False
        assert int(hist.cget("height")) == 24
        assert "▲" in btn.cget("text")
    finally:
        root.destroy()


def test_step5_status_tier_template_resolution(tmp_path):
    app = _make_stub_app(tmp_path)
    app.active_persona_level = 5
    app.current_model_tier = "high"
    app.model_paths = {"high": r"C:\models\big.gguf"}
    app.config["custom_templates"] = {"T1": {"temp": 0.42, "ctx": 8192}}
    app.temp_config = {}
    app.context_size_config = {}
    app.save_config = lambda: None
    assert app._resolve_active_tier() == (5, "high")
    assert app._apply_template_to_tier_config("T1", "high") is True
    assert app.temp_config["high"] == 0.42
    assert app.context_size_config["high"] == 8192


def test_step6_scroll_lock_zero_drift_across_streaming_and_subagents(tmp_path):
    root = tk.Tk()
    root.geometry("420x220")
    root.withdraw()
    try:
        app = _make_stub_app(tmp_path)
        app.root = root
        app.active_tab = "active"
        hist = tk.Text(root, height=10, width=50)
        hist.pack(fill=tk.BOTH, expand=True)
        hist.insert("1.0", "\n".join(f"line {i}" for i in range(120)))
        hist.config(state="disabled")
        root.update_idletasks()

        hist.yview_moveto(0.25)
        root.update_idletasks()
        pinned_top = hist.index("@0,0")
        app.chat_history = hist
        app.config["scroll_lock_enabled"] = True
        app._user_scrolled_up = True
        app.thinking_display = None
        app.set_avatar_state = lambda *a, **kw: None
        app._subagent_ui_registry = {}
        app._animate_text_fade = lambda *a, **kw: None

        assert app._is_chat_at_bottom() is False
        assert app._should_lock_scroll() is True

        for i in range(12):
            app._update_ai_message(f"chunk_{i}\n")
        app._append_to_chat("extra system line\n", "ai")
        handle_subagent_spawn(app, {"subagent_id": "s1", "level": 3, "label": "Lvl 3 Worker"})
        handle_subagent_stream(app, {"subagent_id": "s1", "chunk": "subagent progress\n"})
        app._handle_deep_cook_ui_start({"ctype": "cycle", "cnum": 1, "title": "Cycle 1"})
        app._handle_deep_cook_ui_stream({"ctype": "cycle", "cnum": 1, "content": "draft body\n"})
        root.update_idletasks()

        assert hist.index("@0,0") == pinned_top
    finally:
        root.destroy()


def test_step7_history_all_remembering_and_lvl7_isolation(tmp_path):
    app = _make_stub_app(tmp_path)
    hist_dir = app.dirs["History"]

    def _write_hist(fname, msgs):
        fp = os.path.join(hist_dir, fname)
        with open(fp, "wb") as f:
            f.write(zlib.compress(json.dumps(msgs).encode("utf-8")))
        return fp

    _write_hist("chat_old_lvl3.history.jsonz", [
        {"role": "user", "content": "My project codename is ObsidianFalcon.", "level": 3, "persona": "Serenity"},
        {"role": "assistant", "content": "Got it, codename ObsidianFalcon noted.", "level": 3, "persona": "Serenity"},
    ])
    _write_hist("secret_lvl7.history.jsonz", [
        {"role": "user", "content": "Cecilia private vow", "level": 7, "persona": "Cecilia"},
        {"role": "assistant", "content": "Kept solely inside Level 7 Cecilia archive.", "level": 7, "persona": "Cecilia"},
    ])

    app.active_persona_level = 3
    app.config["history_usage"] = "all"
    app.config["show_cecilia_slider"] = False

    # 1. History tab listing isolates Lvl 7
    entries = app._get_all_history_entries()
    fnames = [e["filename"] for e in entries]
    assert "chat_old_lvl3.history.jsonz" in fnames
    assert "secret_lvl7.history.jsonz" not in fnames

    # 2. Cross-session History: All context recalls ObsidianFalcon without leaking Lvl 7
    app.messages = [
        {"role": "user", "content": "Private prompt", "level": 7, "persona": "Cecilia"},
        {"role": "assistant", "content": "Private Cecilia reply", "level": 7, "persona": "Cecilia"},
        {"role": "user", "content": "Current lvl3 turn"},
    ]
    ctx = app._get_contextual_history_messages()
    joined = " ".join(str(m.get("content", "")) for m in ctx)
    assert "ObsidianFalcon" in joined
    assert "Private Cecilia reply" not in joined
    assert "Level 7 Cecilia" not in joined

    # 3. TurboVec isolation across lookup_mode='all' even if ingested at lvl 7 first
    tvec = TurboVecIndex(hist_dir, mode="keyword")
    tvec.ingest_needed_files(None, 7, "all")
    hits_lvl3 = tvec.search_snippets("Cecilia ObsidianFalcon archive", top_k=5, active_level=3, lookup_mode="all")
    excerpts_lvl3 = " ".join(h["excerpt"] for h in hits_lvl3)
    assert "ObsidianFalcon" in excerpts_lvl3
    assert "Level 7 Cecilia" not in excerpts_lvl3

    hits_lvl7 = tvec.search_snippets("Cecilia archive", top_k=5, active_level=7, lookup_mode="all")
    assert any("Level 7 Cecilia" in h["excerpt"] for h in hits_lvl7)


def test_step8_rlhf_downvote_comment_and_vault_learning(tmp_path):
    app = _make_stub_app(tmp_path)
    vault = SemanticVault(db_path=str(tmp_path / "vault.db"))
    app.semantic_vault = vault

    app._save_rlhf_log("Write PowerShell grep", "Use grep -P foo", -1, comment="Use Select-String on Windows")
    log_file = tmp_path / "System" / "rlhf_logs.json"
    saved = json.loads(log_file.read_text(encoding="utf-8"))
    assert saved[0]["rating"] == -1
    assert saved[0]["comment"] == "Use Select-String on Windows"

    notes = vault.get_grounding_for_prompt("Windows Select-String", limit=4)
    assert any(
        n["concept_family"] == "rlhf_correction" and "Select-String" in n["insight"]
        for n in notes
    )


def test_step9_brainuke_dmn_slop_dedup_and_decay(tmp_path):
    vault = SemanticVault(db_path=str(tmp_path / "brainuke.db"))
    assert vault.store_insight("[PC Core Offline] error fallback") == -1
    assert vault.store_insight("As an AI language model I apologize for the delay") == -1
    assert vault.store_insight("short") == -1

    id1 = vault.store_insight("User prefers concise verification scripts.", concept_family="general", importance_weight=1.0)
    assert id1 > 0
    assert vault.store_insight("User prefers concise verification scripts.", concept_family="general") == -1

    vault.store_insight("[RLHF AVOID MISTAKE]: Never skip pytest.", concept_family="rlhf_correction", importance_weight=1.5)
    decayed = vault.apply_memory_decay(decay_rate=0.10, min_weight=0.25)
    assert decayed >= 1

    rows = {r["concept_family"]: r["importance_weight"] for r in vault.get_recent_grounding(limit=10)}
    assert rows["general"] == pytest.approx(0.9, rel=1e-3)
    assert rows["rlhf_correction"] == pytest.approx(1.5, rel=1e-3)


def test_step10_auto_label_testbank_monitor_and_clean_banks(tmp_path):
    fixed_dt = datetime(2026, 11, 16)
    assert format_auto_label(1, dt=fixed_dt) == "001_2026-11-16"
    d = tmp_path / "banks"
    d.mkdir()
    assert next_auto_label_for_dir(str(d), dt=fixed_dt) == "001_2026-11-16"
    (d / "001_2026-11-16_demo.json").write_text("{}", encoding="utf-8")
    assert next_auto_label_for_dir(str(d), suffix="math", dt=fixed_dt) == "002_2026-11-16_math"

    # Active-model routed /testbank
    app = _make_stub_app(tmp_path)
    app.model = object()
    app._run_blocking_inference = MagicMock(
        return_value=json.dumps({
            "lvl1": [{"prompt": "Explain Python generators concisely.", "reference": "Generators yield lazily."}],
            "lvl2": [{"prompt": "Contrast threads and asyncio.", "reference": "Cooperative vs preemptive."}],
        })
    )
    msg_tb = app._run_testbank_command("Python async")
    assert "active model" in msg_tb
    created = [f for f in os.listdir(tmp_path / "benchmarks" / "wringer" / "custom_test_banks") if f.endswith(".json")]
    assert created and created[0].startswith("001_")

    # Monitor scans clean real workspace banks without false corruption
    real_banks = os.path.join(ROOT_DIR, "benchmarks", "wringer", "custom_test_banks")
    mon = AgenticMonitor(app=app, workspace_dir=ROOT_DIR, custom_banks_dir=real_banks)
    rep = mon.scan_once()
    assert not rep["corrupt_banks"]
    for req in ("custom1.json", "Wiki Trivia.json", "zim_cross_domain_mix.json", "test01.json"):
        assert req in rep["valid_banks"]
        assert is_text_readable(req)
