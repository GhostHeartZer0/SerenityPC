"""
Test: Dynamic Text & Font Size Scaling, Presets, and Zoom Controls
Validates:
1. BASE_FONT_SPECS structure and font dictionary initialization.
2. apply_text_scale dynamically scales all tkFont.Font sizes.
3. Proportional font hierarchy (small <= main <= bold <= large) preserved at all scales.
4. Scale boundary enforcement (clamped between 70% and 250%).
5. zoom_in, zoom_out, and zoom_reset methods.
"""
import sys
import os
import unittest
import tkinter as tk
import tkinter.font as tkFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from main import ChatbotApp, BASE_FONT_SPECS

class TestTextScaling(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tk.Tk()
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.root.destroy()
        except Exception:
            pass

    def setUp(self):
        self.app = ChatbotApp.__new__(ChatbotApp)
        self.app.root = self.root
        self.app.config = {"text_scale": 100}
        self.app.fonts = {}
        for k, spec in BASE_FONT_SPECS.items():
            self.app.fonts[k] = tkFont.Font(**spec)

    def test_base_font_specs_coverage(self):
        """Verify all essential font keys exist in BASE_FONT_SPECS."""
        required_keys = [
            "main", "small", "italic", "large", "bold",
            "ui_button", "ui_label", "ui_small",
            "log", "log_bold", "stats", "stats_bold",
            "md_bold", "md_italic", "md_code", "md_thought",
            "md_header", "md_header_1", "md_header_2", "md_header_3"
        ]
        for key in required_keys:
            self.assertIn(key, BASE_FONT_SPECS)
            self.assertIn(key, self.app.fonts)

    def test_default_100_percent_scale(self):
        """Verify base font sizes at default 100% scale."""
        self.app.apply_text_scale(100, persist=False)
        self.assertEqual(self.app.fonts["small"].cget("size"), 12)
        self.assertEqual(self.app.fonts["main"].cget("size"), 13)
        self.assertEqual(self.app.fonts["bold"].cget("size"), 13)
        self.assertEqual(self.app.fonts["large"].cget("size"), 14)
        self.assertEqual(self.app.fonts["ui_button"].cget("size"), 13)
        self.assertEqual(self.app.fonts["stats"].cget("size"), 8)
        self.assertEqual(self.app.fonts["log"].cget("size"), 8)

    def test_scaled_up_150_percent(self):
        """Verify font sizes scale proportionally at 150%."""
        self.app.apply_text_scale(150, persist=True)
        self.assertEqual(self.app.config.get("text_scale"), 150)
        self.assertEqual(self.app.fonts["small"].cget("size"), int(round(12 * 1.5)))
        self.assertEqual(self.app.fonts["main"].cget("size"), int(round(13 * 1.5)))
        self.assertEqual(self.app.fonts["bold"].cget("size"), int(round(13 * 1.5)))
        self.assertEqual(self.app.fonts["large"].cget("size"), int(round(14 * 1.5)))
        self.assertEqual(self.app.fonts["ui_button"].cget("size"), int(round(13 * 1.5)))

    def test_font_hierarchy_across_all_scales(self):
        """Verify small <= main <= bold <= large holds true across various scale factors."""
        for scale in [70, 85, 100, 115, 125, 140, 160, 200, 250]:
            self.app.apply_text_scale(scale, persist=False)
            small_sz = self.app.fonts["small"].cget("size")
            main_sz = self.app.fonts["main"].cget("size")
            bold_sz = self.app.fonts["bold"].cget("size")
            large_sz = self.app.fonts["large"].cget("size")
            self.assertLessEqual(small_sz, main_sz, f"Failed at {scale}%: small={small_sz}, main={main_sz}")
            self.assertLessEqual(main_sz, bold_sz, f"Failed at {scale}%: main={main_sz}, bold={bold_sz}")
            self.assertLessEqual(bold_sz, large_sz, f"Failed at {scale}%: bold={bold_sz}, large={large_sz}")

    def test_zoom_in_out_reset(self):
        """Verify zoom_in (+10%), zoom_out (-10%), and zoom_reset."""
        self.app.config["text_scale"] = 100
        self.app.apply_text_scale(100, persist=True)
        
        # Zoom in
        self.app.zoom_in()
        self.assertEqual(self.app.config["text_scale"], 110)
        
        # Zoom out twice
        self.app.zoom_out()
        self.assertEqual(self.app.config["text_scale"], 100)
        self.app.zoom_out()
        self.assertEqual(self.app.config["text_scale"], 90)
        
        # Zoom reset
        self.app.zoom_reset()
        self.assertEqual(self.app.config["text_scale"], 100)

    def test_boundary_clamping(self):
        """Verify scales outside 70%-250% are clamped properly."""
        self.app.apply_text_scale(10, persist=True)
        self.assertEqual(self.app.config["text_scale"], 70)
        
        self.app.apply_text_scale(500, persist=True)
        self.assertEqual(self.app.config["text_scale"], 250)

    def test_font_family_switching(self):
        """Verify apply_font_family updates UI and monospace font families appropriately."""
        self.app.apply_font_family(ui_family="Times New Roman", mono_family="Courier New", persist=False)
        self.assertEqual(self.app.fonts["main"].cget("family"), "Times New Roman")
        self.assertEqual(self.app.fonts["ui_button"].cget("family"), "Times New Roman")
        self.assertEqual(self.app.fonts["log"].cget("family"), "Courier New")
        self.assertEqual(self.app.fonts["stats"].cget("family"), "Courier New")
        self.assertEqual(self.app.fonts["md_code"].cget("family"), "Courier New")
        
        # Test switching to Comic Sans
        self.app.apply_font_family(ui_family="Comic Sans MS", mono_family="Consolas", persist=True)
        self.assertEqual(self.app.fonts["main"].cget("family"), "Comic Sans MS")
        self.assertEqual(self.app.config.get("ui_font"), "Comic Sans MS")
        self.assertEqual(self.app.config.get("mono_font"), "Consolas")

if __name__ == "__main__":
    unittest.main()
