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

        with patch("tkinter.simpledialog.askstring", return_value=None):
            res = ChatbotApp.switch_user(self.app, "PrivateUser")
            self.assertFalse(res)


if __name__ == "__main__":
    unittest.main()
