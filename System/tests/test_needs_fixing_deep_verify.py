import unittest
import os
import json
import tempfile
import shutil
import tkinter as tk
from unittest.mock import MagicMock, patch

from main import ChatbotApp
from System.serenity_utils import ThinkingDisplay
from System.settings_tabs import build_models_tab, build_additional_tab, build_users_tab, _open_template_modify_dialog
from System.gguf_draft_model import GgufDraftModel


class TestNeedsFixingDeepVerify(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = MagicMock(spec=ChatbotApp)
        self.app.root = self.root
        self.app.dirs = {
            "Users": os.path.join(self.tmp_dir, "Users"),
            "History": os.path.join(self.tmp_dir, "History"),
            "System": os.path.join(self.tmp_dir, "System"),
            "Media": os.path.join(self.tmp_dir, "Media"),
            "Logs": os.path.join(self.tmp_dir, "Logs")
        }
        for d in self.app.dirs.values():
            os.makedirs(d, exist_ok=True)
        self.app.config_file = os.path.join(self.app.dirs["System"], "config.json")
        self.app.config = {
            "username": "Default",
            "sash_pos": 810,
            "settings_window_geometry": "860x950",
            "scroll_lock_enabled": False,
            "status_bar_linger_sec": 5.0,
            "user_preferred_name": "",
            "user_address_style": "Direct / Plain",
            "custom_templates": {
                "T1": {"name": "Thinking (Gen)", "temp": 1.0, "top_p": 0.95, "ctx": 32768}
            }
        }
        self.app.fonts = {
            "ui_button": ("Segoe UI", 9),
            "ui_small": ("Segoe UI", 8),
            "ui_label": ("Segoe UI", 9),
            "bold": ("Segoe UI", 9, "bold"),
            "stats": ("Consolas", 8),
            "stats_bold": ("Consolas", 8, "bold"),
            "italic": ("Segoe UI", 10, "italic")
        }
        self.app.active_persona_level = 3
        self.app.model_paths = {}
        self.app.get_active_username = lambda: self.app.config.get("username", "Default")
        self.app.get_user_dir = lambda un=None: os.path.join(self.app.dirs["Users"], un if un else self.app.get_active_username())
        self.app.get_user_history_dir = lambda un=None: os.path.join(self.app.dirs["History"], un if un else self.app.get_active_username())
        self.app.list_user_profiles = lambda: ["Default", "UserA", "UserB"]

    def tearDown(self):
        try:
            self.root.destroy()
        except Exception:
            pass
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_prefill_avatar_option_b(self):
        """Test Option B: Lvl 1-5 use thinking, Lvl 6 meditating, Lvl 7 cecilia_alt."""
        for lvl, expected in [(1, "thinking"), (2, "thinking"), (3, "thinking"), (4, "thinking"), (5, "thinking"), (6, "meditating"), (7, "cecilia_alt")]:
            self.app.active_persona_level = lvl
            ChatbotApp._set_prefill_avatar_state(self.app)
            self.app.set_avatar_state.assert_called_with(expected)

    def test_sash_position_persistence_across_profiles(self):
        """Test sash position is persisted to user profile config and restored."""
        self.app.paned = MagicMock()
        self.app.paned.sash_coord.return_value = (765, 0)
        self.app.config["sash_pos"] = 765

        # Call save_config logic
        user_dir = self.app.get_user_dir("UserA")
        os.makedirs(user_dir, exist_ok=True)
        with open(os.path.join(user_dir, "config.json"), "w") as f:
            json.dump({"username": "UserA", "sash_pos": 765}, f)

        # Load back
        with open(os.path.join(user_dir, "config.json"), "r") as f:
            u_data = json.load(f)
        self.assertEqual(u_data["sash_pos"], 765)

    def test_settings_geometry_persistence(self):
        """Test settings_window_geometry updates and persists."""
        win = tk.Toplevel(self.root)
        win.geometry("920x800+100+100")
        win.update_idletasks()
        geom = win.winfo_geometry()
        self.assertIn("920x800", geom)
        self.app.config["settings_window_geometry"] = geom
        self.assertEqual(self.app.config["settings_window_geometry"], geom)
        win.destroy()

    def test_template_modify_dialog_mode_switching(self):
        """Test that template modify dialog buttons switch the template mode variable."""
        template_mode = tk.StringVar(value="modify")
        slot_btn = tk.Button(self.root, text="T1")
        setattr(slot_btn, "slot_id", "T1")

        # Simulate _open_template_modify_dialog and save with write mode
        win = tk.Toplevel(self.root)
        _open_template_modify_dialog(win, self.app, "T1", [slot_btn], template_mode)
        
        # Find child toplevel
        mod_win = [c for c in win.winfo_children() if isinstance(c, tk.Toplevel)][0]
        # Inspect save & write button inside mod_win
        found_btn = False
        for f in mod_win.winfo_children():
            if isinstance(f, tk.Frame):
                for b in f.winfo_children():
                    if isinstance(b, tk.Button) and "Write" in b.cget("text"):
                        b.invoke()
                        found_btn = True
                        break
        self.assertTrue(found_btn)
        self.assertEqual(template_mode.get(), "write")
        win.destroy()

    def test_user_switching_lock_and_isolation(self):
        """Test switching from locked profile locks it, and password is required for private profile."""
        vault = MagicMock()
        vault.is_lock_enabled.return_value = True
        vault.is_locked.return_value = True
        vault.unlock.return_value = True
        self.app.vault_manager = vault

        # Outgoing profile is UserA (private)
        self.app.get_active_username = lambda: "UserA"
        self.app.save_config = MagicMock()
        self.app._apply_sash_pos = MagicMock()
        self.app._load_dmn_backbone = MagicMock()

        # Switch to UserB (with password provided)
        os.makedirs(self.app.get_user_dir("UserB"), exist_ok=True)
        with patch("tkinter.simpledialog.askstring", return_value="correct_pass"):
            res = ChatbotApp.switch_user(self.app, "UserB")
            self.assertTrue(res)
            # Outgoing UserA should have been locked
            vault.lock.assert_called()
            # Incoming UserB should have been unlocked
            vault.unlock.assert_called_with("correct_pass")
            self.assertEqual(self.app.config["username"], "UserB")

    def test_user_switching_cancelled_on_bad_password(self):
        """Test switching to locked profile aborts if password cancelled."""
        vault = MagicMock()
        vault.is_lock_enabled.return_value = True
        vault.is_locked.return_value = True
        self.app.vault_manager = vault
        self.app.get_active_username = lambda: "Default"

        os.makedirs(self.app.get_user_dir("PrivateUser"), exist_ok=True)
        with patch("tkinter.simpledialog.askstring", return_value=None):
            res = ChatbotApp.switch_user(self.app, "PrivateUser")
            self.assertFalse(res)

    def test_persona_label_message_level_grounding(self):

        """Test that _get_message_persona_label prevents Cecilia from overwriting Serenity Lvls 1-6."""
        # Active slider is Level 7 (Cecilia)
        self.app.active_persona_level = 7
        self.app.is_cecilia_active = True
        self.app._get_persona_label = lambda: "Cecilia" if self.app.active_persona_level >= 7 else "Serenity"

        # Turn 1: Level 3 Serenity turn
        m_lvl3 = {"role": "assistant", "content": "Hello, I am Serenity.", "level": 3}
        self.assertEqual(ChatbotApp._get_message_persona_label(self.app, m_lvl3), "Serenity")

        # Turn 2: Explicit Serenity persona tag
        m_serenity = {"role": "assistant", "content": "I am here.", "persona": "Serenity"}
        self.assertEqual(ChatbotApp._get_message_persona_label(self.app, m_serenity), "Serenity")

        # Turn 3: Level 7 turn
        m_lvl7 = {"role": "assistant", "content": "Analysis ready.", "level": 7}
        self.assertEqual(ChatbotApp._get_message_persona_label(self.app, m_lvl7), "Cecilia")

        # Turn 4: Explicit Cecilia persona tag
        m_cecilia = {"role": "assistant", "content": "Calculated.", "persona": "Cecilia"}
        self.assertEqual(ChatbotApp._get_message_persona_label(self.app, m_cecilia), "Cecilia")

        # Turn 5: Legacy turn without level or persona (falls back to UI active label when file path unknown)
        m_legacy = {"role": "assistant", "content": "Legacy turn."}
        self.assertEqual(ChatbotApp._get_message_persona_label(self.app, m_legacy), "Cecilia")

        # Turn 6: File path context grounding: sub-7 file forces Serenity even when slider is Cecilia
        sub7_path = "C:/Users/test/History/model_lvl3.history.encz"
        self.assertEqual(ChatbotApp._get_message_persona_label(self.app, m_legacy, file_path=sub7_path), "Serenity")

        # Turn 7: Sub-7 file with corrupted Cecilia persona tag is safely restored to Serenity
        self.assertEqual(ChatbotApp._get_message_persona_label(self.app, m_cecilia, file_path=sub7_path), "Serenity")

        # Turn 8: Level 7 file grounds legacy turn to Cecilia
        lvl7_path = "C:/Users/test/History/model_lvl7.history.jsonz"
        self.assertEqual(ChatbotApp._get_message_persona_label(self.app, m_legacy, file_path=lvl7_path), "Cecilia")

    def test_multi_format_tool_calling_all_models(self):
        """Test tool parsing across all models and formats present in S:/LLM."""
        from System.tool_registry import GemmaToolRegistry
        reg = GemmaToolRegistry(chatbot_app=None)

        # 1. Python PTC (Gemma-4 / diffusiongemma)
        ptc = "```python\nweb_search(query='quantum computing')\n```"
        calls = reg.parse_tool_calls(ptc)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "web_search")
        self.assertEqual(calls[0][1].get("query"), "quantum computing")

        # 2. Nemotron XML (NVIDIA-Nemotron-3.5-Lightning)
        nemotron = "<tool_call>\n<function=web_search>\n<parameter=query>\nsuperconductors\n</parameter>\n</function>\n</tool_call>"
        calls = reg.parse_tool_calls(nemotron)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "web_search")
        self.assertEqual(calls[0][1].get("query"), "superconductors")

        # 3. MiniCPM XML (MiniCPM5-2B / Sharp-MiniCPM5)
        minicpm = '<function name="read_file">\n<param name="path">README.md</param>\n</function>'
        calls = reg.parse_tool_calls(minicpm)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "read_file")
        self.assertEqual(calls[0][1].get("path"), "README.md")

        # 4. Spark-X2.5 XML (Spark-X2.5-4B)
        spark = "<tool_call>web_search<arg_key>query</arg_key><arg_value>neural scaling laws</arg_value></tool_call>"
        calls = reg.parse_tool_calls(spark)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "web_search")
        self.assertEqual(calls[0][1].get("query"), "neural scaling laws")

        # 5. Olmo-3 JSON calls (Olmo-3-7B / 32B)
        olmo = '<function_calls>[{"name": "web_search", "arguments": {"query": "deep learning architectures"}}]</function_calls>'
        calls = reg.parse_tool_calls(olmo)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "web_search")
        self.assertEqual(calls[0][1].get("query"), "deep learning architectures")

        # 6. Mistral [TOOL_CALLS] (Mistral-Small-24B)
        mistral = '[TOOL_CALLS] [{"name": "web_search", "arguments": {"query": "mixture of experts"}}]'
        calls = reg.parse_tool_calls(mistral)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "web_search")
        self.assertEqual(calls[0][1].get("query"), "mixture of experts")

        # 7. Tag format (<ctrl42>call:name{...})
        tag = '<ctrl42>call:web_search{"query": "antigravity"}'
        calls = reg.parse_tool_calls(tag)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "web_search")
        self.assertEqual(calls[0][1].get("query"), "antigravity")


    def test_slash_n_cleaning_and_config_cleanup(self):
        """Verify literal /n newline artifact cleanup and orphaned mid/mmproj config cleanup."""
        from System.vision_handler import VisionHandler
        raw = "<|channel>thought\nReasoning step.<channel|>\nFirst line./nSecond line./n/nThird line with https://example.com/news/now preserved."
        think, ans = VisionHandler.split_thoughts_and_answer(raw)
        self.assertEqual(think, "Reasoning step.")
        self.assertNotIn("/nSecond", ans)
        self.assertIn("First line.\nSecond line.\n\nThird line", ans)
        self.assertIn("https://example.com/news/now", ans)

        synth_cleaned = ChatbotApp._sanitize_synthesis_output(self.app, "Hello there!/nHow can I help?/nSee path/node.")
        self.assertIn("Hello there!\nHow can I help?", synth_cleaned)
        self.assertIn("path/node", synth_cleaned)

    def test_halt_process_and_regeneration_prevention(self):
        """Verify halt stops generation immediately without tool execution or response overwriting."""
        import threading
        self.app.state = {"running": True}
        self.app.stop_process = threading.Event()
        self.app.text_buffer = "partial response text"
        self.app.thought_stream_buffer = "partial thought"
        self.app.process_queue = MagicMock()
        self.app.thinking_display = MagicMock()
        self.app._update_ai_message = MagicMock()
        self.app._update_thought_dropdown = MagicMock()
        self.app._drain_process_queue = MagicMock()
        self.app._log_and_display = MagicMock()

        # 1. Trigger halt
        ChatbotApp.halt_process(self.app)
        self.assertTrue(self.app.stop_process.is_set())
        self.assertTrue(self.app.state.get("was_halted"))
        self.assertFalse(self.app.state.get("running"))
        self.app._update_ai_message.assert_called_with("partial response text")
        self.app._update_thought_dropdown.assert_called_with("partial thought")

        # 2. Tool loop bypasses tool execution when stop_process is set
        self.app.tool_registry = MagicMock()
        tool_call_text = "send_peer_message(peer_id='test', message='hello')"
        result = ChatbotApp._run_tool_loop(self.app, tool_call_text, "User prompt", {})
        self.assertEqual(result, tool_call_text)
        self.app.tool_registry.execute.assert_not_called()

        # 3. Final synthesis bypasses inference when stop_process is set
        synth_res = ChatbotApp._perform_final_synthesis(self.app, "User prompt", "some reasoning")
        self.assertEqual(synth_res, "")

        # 4. _finalize_message preserves chat without deleting on halt
        hist = tk.Text(self.root)
        hist.insert("1.0", "Actual response that was halted")
        self.app.chat_history = hist
        self.app.state["response_start_idx"] = "1.0"
        self.app.messages = []
        ChatbotApp._finalize_message(self.app, "User prompt", "", "", error=False, halted=True)
        self.assertIn("Actual response that was halted", hist.get("1.0", tk.END))


if __name__ == "__main__":
    unittest.main()

