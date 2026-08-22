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
import time
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

    def test_legacy_migration_and_safe_username(self):
        from main import ChatbotApp
        
        # Test safe get_active_username when config attribute is absent or None
        class BareApp:
            get_active_username = ChatbotApp.get_active_username
            get_user_dir = ChatbotApp.get_user_dir
            _load_dmn_backbone = ChatbotApp._load_dmn_backbone
            _migrate_legacy_user_files = ChatbotApp._migrate_legacy_user_files
        
        bare = BareApp()
        self.assertEqual(bare.get_active_username(), "Default")
        bare.config = None
        self.assertEqual(bare.get_active_username(), "Default")
        
        # Test migration using shutil without NameError
        hist_dir = os.path.join(self.root_dir, "History")
        users_dir = os.path.join(self.root_dir, "Users")
        os.makedirs(hist_dir, exist_ok=True)
        os.makedirs(users_dir, exist_ok=True)
        bare.dirs = {"History": hist_dir, "Users": users_dir, "System": self.root_dir}
        bare.state = {}
        
        legacy_file = os.path.join(hist_dir, "sample.history.jsonz")
        with open(legacy_file, "wb") as f:
            f.write(b"dummy history data")
            
        bare._migrate_legacy_user_files()
        migrated_file = os.path.join(hist_dir, "Default", "sample.history.jsonz")
        self.assertTrue(os.path.exists(migrated_file))
        self.assertFalse(os.path.exists(legacy_file))

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

        # DMN Idle: When not in DMN state (countdown mode)
        app.state["dmn_active"] = False
        widget._update_idle_display()
        self.assertIn("Next DMN in", widget.label.cget("text"))

        # DMN Idle: When in DMN state (time in DMN state mode)
        app.state["dmn_active"] = True
        app.state["dmn_entry_time"] = time.time() - 45 # 45 seconds in DMN
        widget._update_idle_display()
        self.assertIn("Time in DMN", widget.label.cget("text"))
        self.assertIn("00:45", widget.label.cget("text"))

        root.destroy()

    def test_prayer_animation_cycling_and_profile_exclusions(self):
        import tkinter as tk
        from System.serenity_utils import DynamicStatusWidget
        from main import ChatbotApp

        # 1. Test profile exclusions
        class DummyApp:
            def __init__(self, d):
                self.dirs = {"Users": os.path.join(d, "Users"), "History": os.path.join(d, "History")}
                for p in self.dirs.values(): os.makedirs(p, exist_ok=True)
                os.makedirs(os.path.join(self.dirs["Users"], "backups_repair"), exist_ok=True)
                os.makedirs(os.path.join(self.dirs["History"], "jsonz to txt"), exist_ok=True)
                os.makedirs(os.path.join(self.dirs["History"], "backups"), exist_ok=True)
                os.makedirs(os.path.join(self.dirs["Users"], "NormalUser"), exist_ok=True)
            def get_active_username(self): return "NormalUser"
            list_user_profiles = ChatbotApp.list_user_profiles

        d_app = DummyApp(self.root_dir)
        profiles = d_app.list_user_profiles()
        self.assertIn("NormalUser", profiles)
        self.assertNotIn("backups_repair", profiles)
        self.assertNotIn("jsonz to txt", profiles)
        self.assertNotIn("backups", profiles)

        # 2. Test Prayer animation cycling
        root = tk.Tk()
        root.withdraw()
        app = type("MockApp", (), {"config": {"status_bar_mode": "prayer", "status_bar_dmn_idle": True, "status_bar_fallback_info": True}})()
        widget = DynamicStatusWidget(root, app=app)
        widget.start()
        self.assertEqual(widget._current_phase, "idle")
        self.assertTrue(widget._is_active)

        # Step prayer animation through full cycle
        for _ in range(80):
            widget._start_prayer_animation()
        # Verify prayer text is populated and alpha cycled
        self.assertTrue(len(widget.prayer_label.cget("text")) > 0)
        self.assertTrue(widget._prayer_direction in [-1, 0, 1, 2])

        widget.stop()
        root.destroy()

    def test_default_and_public_profiles_handling(self):
        from main import ChatbotApp
        from System.vault_manager import VaultManager

        class MockApp:
            def __init__(self, script_dir):
                self.script_dir = script_dir
                self.dirs = {d: os.path.join(self.script_dir, d) for d in ["Media", "History", "Models", "Logs", "System", "Users"]}
                for d in self.dirs.values(): os.makedirs(d, exist_ok=True)
                self.config = {"username": "Default"}
                self.state = {"dmn_backbone": {}}
                self.model_path = os.path.join(self.dirs["Models"], "test_model.gguf")
                self.active_persona_level = 1
                self.messages = []
                self.vault_manager = VaultManager(history_dir=self.dirs["History"], state_dir=self.dirs["System"])

            get_active_username = ChatbotApp.get_active_username
            get_user_history_dir = ChatbotApp.get_user_history_dir
            get_user_dir = ChatbotApp.get_user_dir
            list_user_profiles = ChatbotApp.list_user_profiles
            switch_user = ChatbotApp.switch_user
            get_history_path = ChatbotApp.get_history_path
            _load_dmn_backbone = ChatbotApp._load_dmn_backbone
            _save_dmn_backbone = ChatbotApp._save_dmn_backbone
            save_config = lambda self: None
            _log_and_display = lambda self, msg: None

        app = MockApp(self.root_dir)

        # 1. Verify Default profile uses .jsonz extension
        app.switch_user("Default")
        def_path = app.get_history_path()
        self.assertTrue(def_path.endswith(".jsonz"))
        self.assertIn(os.path.join("History", "Default"), def_path)

        # 2. Verify Public profile uses .jsonz extension
        app.switch_user("Public")
        pub_path = app.get_history_path()
        self.assertTrue(pub_path.endswith(".jsonz"))
        self.assertIn(os.path.join("History", "Public"), pub_path)

        # 3. Verify Private profile uses .encz when vault lock is enabled
        app.vault_manager.set_password("Secret1234")
        app.switch_user("PrivateUser")
        priv_path = app.get_history_path()
        self.assertTrue(priv_path.endswith(".encz"))
        self.assertIn(os.path.join("History", "PrivateUser"), priv_path)

        # 4. Profile listing includes Default and Public
        profiles = app.list_user_profiles()
        self.assertIn("Default", profiles)
        self.assertIn("Public", profiles)
        self.assertIn("PrivateUser", profiles)

    def test_settings_locked_profile_swap_verification(self):
        from System.vault_manager import VaultManager
        v_dir = os.path.join(self.root_dir, "SettingsVault")
        os.makedirs(v_dir, exist_ok=True)
        vm = VaultManager(history_dir=v_dir, state_dir=v_dir)
        vm.set_password("MyPass123!")
        vm.lock()
        self.assertTrue(vm.is_lock_enabled())
        self.assertTrue(vm.is_locked())

        # Verify correct vs incorrect password unlock
        self.assertFalse(vm.unlock("wrong"))
        self.assertTrue(vm.is_locked())
        self.assertTrue(vm.unlock("MyPass123!"))
        self.assertFalse(vm.is_locked())

if __name__ == "__main__":
    unittest.main()
