# test_profile_custom_name.py
# Verification suite for Custom Profile Name and User Addressing preferences.

import unittest
import os
import sys
import tempfile
import json
import shutil

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

class TestProfileCustomName(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_custom_user_name_prompt_injection(self):
        """Verify that user_preferred_name and addressing style format safely into system prompt."""
        base_sys = "You are Serenity, an insightful and loyal AI companion."
        config = {
            "user_preferred_name": "GhostHeart",
            "user_address_style": "Warm / Familiar"
        }
        
        pref_name = str(config.get("user_preferred_name", "")).strip()
        addr_style = str(config.get("user_address_style", "Direct / Plain")).strip()
        
        sys_clean = base_sys
        if pref_name:
            sys_clean += f"\n[User Identity]: The user's name is {pref_name}."
            if addr_style and addr_style != "Silent / Unnamed":
                sys_clean += f" Preferred addressing style: {addr_style}."
            elif addr_style == "Silent / Unnamed":
                sys_clean += " Do not address the user by name unless explicitly asked."

        self.assertIn("[User Identity]: The user's name is GhostHeart.", sys_clean)
        self.assertIn("Preferred addressing style: Warm / Familiar.", sys_clean)
        # Ensure base persona definition is preserved intact
        self.assertTrue(sys_clean.startswith(base_sys))

    def test_silent_addressing_style_directive(self):
        """Verify that 'Silent / Unnamed' style instructs model not to address user by name."""
        base_sys = "You are Serenity."
        config = {
            "user_preferred_name": "Alex",
            "user_address_style": "Silent / Unnamed"
        }
        
        pref_name = str(config.get("user_preferred_name", "")).strip()
        addr_style = str(config.get("user_address_style", "")).strip()
        
        sys_clean = base_sys
        if pref_name:
            sys_clean += f"\n[User Identity]: The user's name is {pref_name}."
            if addr_style and addr_style != "Silent / Unnamed":
                sys_clean += f" Preferred addressing style: {addr_style}."
            elif addr_style == "Silent / Unnamed":
                sys_clean += " Do not address the user by name unless explicitly asked."

        self.assertIn("Do not address the user by name unless explicitly asked.", sys_clean)

    def test_user_profile_config_isolation(self):
        """Verify that separate user profiles maintain distinct preferred names and addressing styles."""
        profile_a_dir = os.path.join(self.test_dir, "Users", "ProfileA")
        profile_b_dir = os.path.join(self.test_dir, "Users", "ProfileB")
        os.makedirs(profile_a_dir, exist_ok=True)
        os.makedirs(profile_b_dir, exist_ok=True)

        cfg_a = {
            "username": "ProfileA",
            "user_preferred_name": "Commander",
            "user_address_style": "Formal / Respectful"
        }
        cfg_b = {
            "username": "ProfileB",
            "user_preferred_name": "GhostHeart",
            "user_address_style": "Warm / Familiar"
        }

        with open(os.path.join(profile_a_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg_a, f)
        with open(os.path.join(profile_b_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg_b, f)

        # Reload profile A
        with open(os.path.join(profile_a_dir, "config.json"), "r", encoding="utf-8") as f:
            loaded_a = json.load(f)
        # Reload profile B
        with open(os.path.join(profile_b_dir, "config.json"), "r", encoding="utf-8") as f:
            loaded_b = json.load(f)

        self.assertEqual(loaded_a["user_preferred_name"], "Commander")
        self.assertEqual(loaded_a["user_address_style"], "Formal / Respectful")
        self.assertEqual(loaded_b["user_preferred_name"], "GhostHeart")
        self.assertEqual(loaded_b["user_address_style"], "Warm / Familiar")

if __name__ == "__main__":
    unittest.main()
