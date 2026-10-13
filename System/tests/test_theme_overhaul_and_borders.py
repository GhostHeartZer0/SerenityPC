import unittest
import sys
import os
import tkinter as tk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from serenity_resources import THEMES, apply_theme_to_global, THEME, THERMO_COLORS, PERSONA_SECONDARY_COLORS, desaturate_color

class TestThemeOverhaulAndBorders(unittest.TestCase):
    def test_goth_and_obsidian_dark_split(self):
        self.assertIn("goth", THEMES)
        self.assertIn("obsidian_dark", THEMES)
        
        goth = THEMES["goth"]
        self.assertEqual(goth["bg_color"], "#000000")
        self.assertEqual(goth["trim_color"], "#8a0303") # deep blood red
        self.assertEqual(goth["accent_highlight"], "#FF1A14")
        self.assertEqual(goth["fg_color"], "#FF1A14")   # neon crimson
        self.assertEqual(goth["chat_fg_color"], "#D60808")
        
        obsidian = THEMES["obsidian_dark"]
        self.assertEqual(obsidian["bg_color"], "#0e0d13") # dark granite
        self.assertEqual(obsidian["widget_bg_color"], "#161420")
        self.assertEqual(obsidian["trim_color"], "#4d2d7d") # candied blue-violet
        self.assertEqual(obsidian["electric_blue"], "#6D00FF") # electric violet
        self.assertEqual(obsidian["fg_color"], "#6D00FF")
        self.assertEqual(obsidian["chat_fg_color"], "#6D11FF")

    def test_lvl3_red_contrast(self):
        # THERMO_COLORS[3] must be high contrast (vivid crimson), not dark red #7D0000
        self.assertNotEqual(THERMO_COLORS[3], "#7D0000")
        self.assertEqual(THERMO_COLORS[3], "#FF3355")

    def test_persona_secondary_colors(self):
        for lvl in range(8):
            self.assertIn(lvl, PERSONA_SECONDARY_COLORS)
            self.assertTrue(PERSONA_SECONDARY_COLORS[lvl].startswith("#"))

    def test_borders_in_theme_application(self):
        # 1. Plan A: Non-persona themes keep their own palette and do not bleed when model loaded
        apply_theme_to_global(theme_name="apex", dark_mode=False, active_level=3, model_loaded=True)
        self.assertEqual(THEME["accent_highlight"], "#ff7700")
        self.assertEqual(THEME["bg_color"], "#1b1b1f")
        self.assertEqual(THEME["trim_color"], "#b84700")
        # Persona thermo colors are provided for prompt & response borders
        self.assertEqual(THEME["_persona_color"], THERMO_COLORS[3])
        self.assertEqual(THEME["_persona_secondary_color"], PERSONA_SECONDARY_COLORS[3])

        # 2. Persona theme dynamically adapts to active level
        apply_theme_to_global(theme_name="persona", dark_mode=False, active_level=3, model_loaded=True)
        self.assertEqual(THEME["accent_highlight"], THERMO_COLORS[3])
        self.assertEqual(THEME["accent_secondary"], PERSONA_SECONDARY_COLORS[3])

        # 3. When dark mode enabled, backgrounds must be pure black while theme accents stay intact
        apply_theme_to_global(theme_name="obsidian_dark", dark_mode=True, active_level=2, model_loaded=True)
        self.assertEqual(THEME["bg_color"], "#000000")
        self.assertEqual(THEME["widget_bg_color"], "#000000")
        self.assertEqual(THEME["chat_bg_color"], "#000000")
        self.assertEqual(THEME["accent_highlight"], "#6D00FF")
        self.assertEqual(THEME["_persona_color"], THERMO_COLORS[2])

    def test_prompt_and_thought_color_distinction(self):
        # Prompt uses (a) accent_secondary, response uses (t) chat_fg_color, thoughts are desaturated (t)
        apply_theme_to_global(theme_name="apex", dark_mode=False)
        self.assertEqual(THEME["accent_secondary"], "#b84700")
        self.assertEqual(THEME["chat_fg_color"], "#ff7700")
        thought = desaturate_color(THEME["chat_fg_color"])
        self.assertNotEqual(thought, THEME["chat_fg_color"])
        self.assertTrue(thought.startswith("#"))

    def test_settings_retheming_helper(self):
        from System.settings_ui import apply_theme_to_settings_window
        apply_theme_to_global("matrix", dark_mode=False)
        root = tk.Tk()
        root.withdraw()
        try:
            f = tk.Frame(root)
            lbl = tk.Label(f, text="Test Label", fg="#ffee00")
            btn = tk.Button(f, text="Click", bg="#0a0a00", fg="#ffee00")
            ent = tk.Entry(f, bg="#000000", fg="#ffee00")
            apply_theme_to_settings_window(f)
            # Must reflect matrix theme (#00ff41), not yellow blacket (#ffee00)
            self.assertEqual(lbl.cget("fg"), THEME["fg_color"])
            self.assertEqual(btn.cget("fg"), THEME["fg_color"])
            self.assertEqual(ent.cget("fg"), THEME["fg_color"])
        finally:
            root.destroy()

if __name__ == "__main__":
    unittest.main()
