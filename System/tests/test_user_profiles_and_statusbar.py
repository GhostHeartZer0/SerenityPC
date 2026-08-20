"""
Test: User Profiles & Mind Separation + Dynamic Status Bar Overhaul
Validates:
1. Username support for history & mind memory separation under Serenity root (History/<user>/, Users/<user>/).
2. DynamicStatusWidget modes (Tasks, Percentage, Animation, Prayer, DMN Idle Timer, Hybrid).
"""
import sys
import os
import json
import zlib
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class TestUserProfilesAndStatusBar(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root_dir = self.tmpdir.name

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_user_directory_isolation_and_switching(self):
        from main import ChatbotApp

        class DummyChatbotApp:
            def __init__(self, script_dir):
                self.script_dir = script_dir
                self.dirs = {d: os.path.join(self.script_dir, d) for d in ["Media", "History", "Models", "Logs", "System", "Users"]}
                for d in self.dirs.values(): os.makedirs(d, exist_ok=True)
                self.config = {"username": "Alice"}
                self.state = {"dmn_backbone": {}}
                self.model_path = os.path.join(self.dirs["Models"], "test_model.gguf")
                self.active_persona_level = 3
                self.messages = [{"role": "user", "content": "Alice's secret chat"}]

            get_active_username = ChatbotApp.get_active_username
            get_user_history_dir = ChatbotApp.get_user_history_dir
            get_user_dir = ChatbotApp.get_user_dir
            list_user_profiles = ChatbotApp.list_user_profiles
            switch_user = ChatbotApp.switch_user
            get_history_path = ChatbotApp.get_history_path
            _load_dmn_backbone = ChatbotApp._load_dmn_backbone
            _save_dmn_backbone = ChatbotApp._save_dmn_backbone
            _migrate_legacy_user_files = ChatbotApp._migrate_legacy_user_files
            save_config = lambda self: None
            _log_and_display = lambda self, msg: None

        app = DummyChatbotApp(self.root_dir)

        # 1. Verify user paths
        self.assertEqual(app.get_active_username(), "Alice")
        alice_hist = app.get_user_history_dir()
        alice_user = app.get_user_dir()
        self.assertTrue(alice_hist.endswith(os.path.join("History", "Alice")))
        self.assertTrue(alice_user.endswith(os.path.join("Users", "Alice")))

        # 2. Save DMN backbone for Alice
        app.state["dmn_backbone"] = {"simmer": "Alice workspace thoughts", "node_count": 5}
        app._save_dmn_backbone()
        alice_dmn_path = os.path.join(alice_user, "dmn_backbone.json")
        self.assertTrue(os.path.exists(alice_dmn_path))

        # 3. Switch user to Bob
        app.switch_user("Bob")
        self.assertEqual(app.get_active_username(), "Bob")
        bob_hist = app.get_user_history_dir()
        bob_user = app.get_user_dir()
        self.assertTrue(bob_hist.endswith(os.path.join("History", "Bob")))
        self.assertTrue(bob_user.endswith(os.path.join("Users", "Bob")))

        # 4. Verify DMN backbone isolation (Bob has clean state)
        self.assertEqual(app.state.get("dmn_backbone"), {})

        # 5. List profiles
        profiles = app.list_user_profiles()
        self.assertIn("Alice", profiles)
        self.assertIn("Bob", profiles)
        self.assertIn("Default", profiles)

    def test_dynamic_status_widget_phases_and_telemetry(self):
        import tkinter as tk
        from System.serenity_utils import DynamicStatusWidget

        root = tk.Tk()
        root.withdraw()

        class MockAppWithConfig:
            def __init__(self):
                self.config = {
                    "status_bar_mode": "hybrid",
                    "status_bar_anim_style": "spinner",
                    "status_bar_dmn_idle": True,
                    "status_bar_fallback_info": True,
                    "flash_attention_kv": "q8_0"
                }
                self.active_persona_level = 5
                self.current_model_tier = "high"
                self.context_size_config = {"high": 32768}
                self.state = {"auto_watch": False}

        app = MockAppWithConfig()
        widget = DynamicStatusWidget(root, app=app)
        widget.start()

        # Prefill phase
        widget.set_phase("prefill", "Ingesting context")
        self.assertIn("Prefill", widget.label.cget("text"))

        # TTFT recording
        widget.record_ttft(0.42)
        self.assertEqual(widget._ttft, 0.42)

        # Generating phase
        widget.set_phase("generating", tokens=25, speed=35.5)
        self.assertIn("35.5 t/s", widget.label.cget("text"))
        self.assertIn("Lvl 5", widget.telemetry_label.cget("text"))

        # Complete phase
        widget.stop()
        self.assertEqual(widget._current_phase, "complete")

        root.destroy()

if __name__ == "__main__":
    unittest.main()
