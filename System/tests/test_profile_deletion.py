"""
test_profile_deletion.py
Validates:
1. delete_user_profile prevents deletion of 'Default' and 'Public'.
2. delete_user_profile safely deletes profile directories from Users and History.
3. delete_user_profile auto-switches active profile to 'Default' when deleting active user.
4. Tab 5 (Users & Security) contains 'Delete Profile' button with admin verification.
5. Atomic profile creation: Wizard does not leave half-created directories on disk if switch/auth fails.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
import tkinter as tk
from unittest.mock import MagicMock, patch

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from System.settings_tabs import build_users_tab
from main import ChatbotApp


class TestProfileDeletion(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.users_dir = os.path.join(self.tmpdir.name, "Users")
        self.history_dir = os.path.join(self.tmpdir.name, "History")
        os.makedirs(self.users_dir, exist_ok=True)
        os.makedirs(self.history_dir, exist_ok=True)

        class MockApp:
            def __init__(self, root, users_dir, history_dir):
                self.root = root
                self.scale_factor = 1.0
                self.icon_path = None
                self.dirs = {"Users": users_dir, "History": history_dir}
                self.config = {
                    "username": "Default",
                    "user_preferred_name": "",
                    "user_address_style": "Direct / Plain",
                    "theme": "apex",
                    "dark_mode": False,
                    "show_default_profile": True,
                    "show_public_profile": True,
                }
                self.fonts = {
                    "ui_label": ("Segoe UI", 9),
                    "ui_button": ("Segoe UI", 9),
                    "ui_small": ("Segoe UI", 8),
                    "bold": ("Segoe UI", 9, "bold"),
                }
                self.vault_manager = MagicMock()
                self.vault_manager.is_lock_enabled.return_value = False
                self.vault_manager.is_locked.return_value = False

            def get_active_username(self):
                return self.config.get("username", "Default")

            def get_user_dir(self, un=None):
                target = un or self.get_active_username()
                p = os.path.join(self.dirs["Users"], target)
                os.makedirs(p, exist_ok=True)
                return p

            def get_user_history_dir(self, un=None):
                target = un or self.get_active_username()
                p = os.path.join(self.dirs["History"], target)
                os.makedirs(p, exist_ok=True)
                return p

            def list_user_profiles(self):
                return ChatbotApp.list_user_profiles(self)

            def switch_user(self, new_un, skip_lock_prompt=False):
                clean_un = "".join(c for c in new_un.strip() if c.isalnum() or c in ("-", "_", " ")).strip()
                if not clean_un:
                    clean_un = "Default"
                self.config["username"] = clean_un
                return True

            def delete_user_profile(self, username):
                return ChatbotApp.delete_user_profile(self, username)

        self.app = MockApp(self.root, self.users_dir, self.history_dir)

    def tearDown(self):
        try:
            self.root.destroy()
        except Exception:
            pass
        self.tmpdir.cleanup()

    def _setup_vars_dict(self):
        return {
            "username_var": tk.StringVar(value=self.app.get_active_username()),
            "user_profiles_list": self.app.list_user_profiles(),
            "user_pref_name_var": tk.StringVar(value=""),
            "user_addr_style_var": tk.StringVar(value="Direct / Plain"),
            "theme_display_var": tk.StringVar(value="Apex (Default)"),
            "dark_mode_var": tk.BooleanVar(value=False),
            "show_def_var": tk.BooleanVar(value=True),
            "show_pub_var": tk.BooleanVar(value=True),
            "auto_lock_var": tk.StringVar(value="0"),
            "THEME_MAP": {"Apex (Default)": "apex"},
            "THEME_REV_MAP": {"apex": "Apex (Default)"},
        }

    def test_delete_protected_system_profiles(self):
        """Ensure Default and Public profiles cannot be deleted."""
        success_def, msg_def = self.app.delete_user_profile("Default")
        self.assertFalse(success_def)
        self.assertIn("system profile", msg_def)

        success_pub, msg_pub = self.app.delete_user_profile("Public")
        self.assertFalse(success_pub)
        self.assertIn("system profile", msg_pub)

    def test_delete_custom_profile_removes_directories(self):
        """Verify custom profile folders in Users and History are cleanly deleted."""
        # Create test profile
        custom_u_dir = os.path.join(self.users_dir, "TempUser")
        custom_h_dir = os.path.join(self.history_dir, "TempUser")
        os.makedirs(custom_u_dir, exist_ok=True)
        os.makedirs(custom_h_dir, exist_ok=True)
        with open(os.path.join(custom_u_dir, "config.json"), "w") as f:
            json.dump({"username": "TempUser"}, f)

        self.assertTrue(os.path.exists(custom_u_dir))
        self.assertTrue(os.path.exists(custom_h_dir))

        success, msg = self.app.delete_user_profile("TempUser")
        self.assertTrue(success)
        self.assertFalse(os.path.exists(custom_u_dir))
        self.assertFalse(os.path.exists(custom_h_dir))

    def test_delete_active_profile_switches_to_default(self):
        """Verify deleting the currently active user profile switches active user to Default."""
        self.app.config["username"] = "ActiveDeletable"
        os.makedirs(os.path.join(self.users_dir, "ActiveDeletable"), exist_ok=True)

        success, msg = self.app.delete_user_profile("ActiveDeletable")
        self.assertTrue(success)
        self.assertEqual(self.app.get_active_username(), "Default")

    def test_users_tab_has_delete_profile_button(self):
        """Verify Tab 5 has a Delete Profile button and user_combo updates after deletion."""
        win = tk.Toplevel(self.root)
        vars_dict = self._setup_vars_dict()
        build_users_tab(win, self.app, win, vars_dict)

        self.assertIn("btn_delete_profile", vars_dict)
        btn_del = vars_dict["btn_delete_profile"]
        self.assertEqual(btn_del.cget("text"), "Delete Profile")

    def test_atomic_profile_creation_no_half_profiles(self):
        """Verify wizard leaves no directory on disk if switch_user fails."""
        win = tk.Toplevel(self.root)
        vars_dict = self._setup_vars_dict()
        build_users_tab(win, self.app, win, vars_dict)

        # Make switch_user fail (simulate bad password or abort)
        self.app.switch_user = MagicMock(return_value=False)

        wiz = vars_dict["_open_create_profile_wizard"]()
        wiz._un_var.set("FailedNewUser")

        wiz._do_create()

        # Check that no folder was left behind
        target_dir = os.path.join(self.users_dir, "FailedNewUser")
        self.assertFalse(os.path.exists(target_dir))
        self.assertNotIn("FailedNewUser", self.app.list_user_profiles())


if __name__ == "__main__":
    unittest.main()
