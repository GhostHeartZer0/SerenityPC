"""
test_profile_wizard.py
Validates:
1. Tab 5 (Users & Security) contains distinct 'Switch Profile' and 'Create Profile' buttons.
2. Setup Wizard modal dialog instantiates correctly with themed inputs for:
   - Profile Username
   - Preferred Name / Call Sign
   - Addressing Style
   - Color Theme
   - Dark Mode toggle
3. Profile setup wizard sanitizes username, handles empty validation, writes config.json to user directory,
   and switches active profile while updating UI variables.
"""

import os
import sys
import json
import tempfile
import unittest
import tkinter as tk
from unittest.mock import MagicMock, patch

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from System.settings_tabs import build_users_tab


class TestProfileWizard(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.tmpdir = tempfile.TemporaryDirectory()
        self.users_dir = os.path.join(self.tmpdir.name, "Users")
        os.makedirs(self.users_dir, exist_ok=True)

        class MockApp:
            def __init__(self, root, users_dir):
                self.root = root
                self.scale_factor = 1.0
                self.icon_path = None
                self.dirs = {"Users": users_dir}
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
                self._profiles = ["Default"]

            def get_active_username(self):
                return self.config.get("username", "Default")

            def get_user_dir(self, un=None):
                target = un or self.get_active_username()
                p = os.path.join(self.dirs["Users"], target)
                os.makedirs(p, exist_ok=True)
                return p

            def list_user_profiles(self):
                return list(self._profiles)

            def switch_user(self, new_un):
                clean_un = "".join(c for c in new_un.strip() if c.isalnum() or c in ("-", "_", " ")).strip()
                if not clean_un:
                    clean_un = "Default"
                self.config["username"] = clean_un
                if clean_un not in self._profiles:
                    self._profiles.append(clean_un)
                    self._profiles.sort()
                u_dir = self.get_user_dir(clean_un)
                cfg_path = os.path.join(u_dir, "config.json")
                if os.path.exists(cfg_path):
                    try:
                        with open(cfg_path, "r", encoding="utf-8") as f:
                            self.config.update(json.load(f))
                    except Exception:
                        pass
                return True

            def save_config(self):
                pass

        self.app = MockApp(self.root, self.users_dir)

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
            "THEME_MAP": {
                "Apex (Default)": "apex",
                "Goth / Obsidian Dark": "goth",
                "Crystal Cavern": "crystal_cavern",
                "Yellow Blacket": "yellow_blacket",
                "Natural (Earth / Moss)": "natural",
                "Matrix (Cyber Green)": "matrix",
                "Persona (Level Dynamic)": "persona",
            },
            "THEME_REV_MAP": {
                "apex": "Apex (Default)",
                "goth": "Goth / Obsidian Dark",
                "crystal_cavern": "Crystal Cavern",
                "yellow_blacket": "Yellow Blacket",
                "natural": "Natural (Earth / Moss)",
                "matrix": "Matrix (Cyber Green)",
                "persona": "Persona (Level Dynamic)",
            },
        }

    def test_split_buttons_exist_in_users_tab(self):
        """Verify Tab 5 contains separate 'Switch Profile' and 'Create Profile' buttons."""
        win = tk.Toplevel(self.root)
        vars_dict = self._setup_vars_dict()
        tab = build_users_tab(win, self.app, win, vars_dict)

        self.assertIn("btn_switch_profile", vars_dict)
        self.assertIn("btn_create_profile", vars_dict)
        self.assertIn("btn_logout", vars_dict)

        self.assertEqual(vars_dict["btn_switch_profile"].cget("text"), "Switch Profile")
        self.assertEqual(vars_dict["btn_create_profile"].cget("text"), "Create Profile")
        self.assertEqual(vars_dict["btn_logout"].cget("text"), "Logout")

    def test_open_create_profile_wizard_instantiation(self):
        """Verify Setup Wizard modal opens with required form variables."""
        win = tk.Toplevel(self.root)
        vars_dict = self._setup_vars_dict()
        build_users_tab(win, self.app, win, vars_dict)

        open_wiz = vars_dict["_open_create_profile_wizard"]
        wiz = open_wiz()
        self.assertIsInstance(wiz, tk.Toplevel)
        self.assertEqual(wiz.title(), "Create Profile Wizard")

        # Verify exposed form variables
        self.assertTrue(hasattr(wiz, "_un_var"))
        self.assertTrue(hasattr(wiz, "_pref_var"))
        self.assertTrue(hasattr(wiz, "_addr_var"))
        self.assertTrue(hasattr(wiz, "_th_var"))
        self.assertTrue(hasattr(wiz, "_dark_var"))
        wiz.destroy()

    def test_wizard_validation_empty_username(self):
        """Verify Wizard blocks creation when username is empty or whitespace."""
        win = tk.Toplevel(self.root)
        vars_dict = self._setup_vars_dict()
        build_users_tab(win, self.app, win, vars_dict)

        wiz = vars_dict["_open_create_profile_wizard"]()
        wiz._un_var.set("   ")

        with patch("tkinter.messagebox.showerror") as mock_err:
            wiz._do_create()
            mock_err.assert_called_once()
            # Wizard should not have closed
            self.assertTrue(wiz.winfo_exists())
        wiz.destroy()

    def test_wizard_successful_profile_creation_and_switch(self):
        """Verify Wizard creates profile directory, writes config.json, and switches active user."""
        win = tk.Toplevel(self.root)
        vars_dict = self._setup_vars_dict()
        build_users_tab(win, self.app, win, vars_dict)

        wiz = vars_dict["_open_create_profile_wizard"]()
        wiz._un_var.set("GhostHeart")
        wiz._pref_var.set("Commander Zer0")
        wiz._addr_var.set("Warm / Familiar")
        wiz._th_var.set("Goth / Obsidian Dark")
        wiz._dark_var.set(True)

        with patch("tkinter.messagebox.showinfo") as mock_info:
            wiz._do_create()
            mock_info.assert_called_once()

        # Check that user directory and config.json were created
        ghost_dir = os.path.join(self.users_dir, "GhostHeart")
        self.assertTrue(os.path.exists(ghost_dir))
        cfg_file = os.path.join(ghost_dir, "config.json")
        self.assertTrue(os.path.exists(cfg_file))

        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg_data = json.load(f)

        self.assertEqual(cfg_data["username"], "GhostHeart")
        self.assertEqual(cfg_data["user_preferred_name"], "Commander Zer0")
        self.assertEqual(cfg_data["user_address_style"], "Warm / Familiar")
        self.assertEqual(cfg_data["theme"], "goth")
        self.assertTrue(cfg_data["dark_mode"])

        # Check app active user & UI vars synced
        self.assertEqual(self.app.get_active_username(), "GhostHeart")
        self.assertEqual(vars_dict["username_var"].get(), "GhostHeart")
        self.assertEqual(vars_dict["user_pref_name_var"].get(), "Commander Zer0")
        self.assertEqual(vars_dict["user_addr_style_var"].get(), "Warm / Familiar")
        self.assertEqual(vars_dict["theme_display_var"].get(), "Goth / Obsidian Dark")
        self.assertTrue(vars_dict["dark_mode_var"].get())
        self.assertIn("GhostHeart", vars_dict["user_combo"]["values"])


if __name__ == "__main__":
    unittest.main()
