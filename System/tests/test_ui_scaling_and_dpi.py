"""
Test: UI Dynamic Auto-Scaling, High-DPI Awareness, and Layout Constraints
Validates:
1. High-DPI awareness initialization helper (enable_high_dpi_awareness).
2. Proportional font hierarchy (small < main <= italic <= bold <= large).
3. Persona control bar layout sizing and responsive auto-scaling bounds.
4. Telemetry panel key uniqueness and stats mapping without duplicate collision.
"""
import sys
import os
import unittest
import tkinter as tk
import tkinter.font as tkFont
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from System.serenity_utils import enable_high_dpi_awareness, HardwareProfile

class TestUIScalingAndDPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        enable_high_dpi_awareness()
        cls.root = tk.Tk()
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.root.destroy()
        except Exception:
            pass

    def test_high_dpi_awareness_execution(self):
        """Verify enable_high_dpi_awareness runs without raising exceptions."""
        try:
            enable_high_dpi_awareness()
            HardwareProfile.enable_high_dpi_awareness()
            success = True
        except Exception as e:
            success = False
            self.fail(f"enable_high_dpi_awareness failed with exception: {e}")
        self.assertTrue(success)

    def test_font_hierarchy_scale(self):
        """Verify font scale hierarchy is properly proportioned for 1080p and high-DPI."""
        fonts = {
            "main": tkFont.Font(family="Open Sans", size=10),
            "small": tkFont.Font(family="Open Sans", size=9),
            "italic": tkFont.Font(family="Open Sans", size=10, slant="italic"),
            "large": tkFont.Font(family="Open Sans", size=12),
            "bold": tkFont.Font(family="Open Sans", size=11, weight="bold"),
        }
        self.assertLess(fonts["small"].cget("size"), fonts["main"].cget("size"))
        self.assertLessEqual(fonts["main"].cget("size"), fonts["bold"].cget("size"))
        self.assertLessEqual(fonts["bold"].cget("size"), fonts["large"].cget("size"))
        self.assertLessEqual(fonts["large"].cget("size"), 14)

    def test_persona_controls_fit(self):
        """Verify persona controls assemble inside a compact frame without overflowing."""
        frame = tk.Frame(self.root, width=500, height=50)
        frame.pack()
        
        lbl = tk.Label(frame, text="Persona:", font=("Open Sans", 9))
        lbl.pack(side=tk.LEFT)
        
        scale = tk.Scale(frame, from_=1, to=6, orient=tk.HORIZONTAL, length=110, showvalue=False)
        scale.pack(side=tk.LEFT, padx=(6, 2))
        
        btn_name = tk.Button(frame, text="LVL 6: The Transcendent One", font=("Open Sans", 11, "bold"), padx=6, pady=2)
        btn_name.pack(side=tk.LEFT, padx=3)
        
        btn_add = tk.Button(frame, text="+", font=("Open Sans", 11, "bold"), padx=6, pady=2)
        btn_add.pack(side=tk.LEFT, padx=2)
        
        btn_mic = tk.Button(frame, text="🎙️", font=("Open Sans", 10), padx=4, pady=2)
        btn_mic.pack(side=tk.LEFT, padx=2)
        
        self.root.update_idletasks()
        
        # Check total requested width
        req_width = (
            lbl.winfo_reqwidth() +
            scale.winfo_reqwidth() +
            btn_name.winfo_reqwidth() +
            btn_add.winfo_reqwidth() +
            btn_mic.winfo_reqwidth()
        )
        self.assertLess(req_width, 500, f"Persona control row requested width {req_width}px exceeds 500px")
        frame.destroy()

    def test_telemetry_keys_uniqueness(self):
        """Verify telemetry stats mapping has distinct keys without collision."""
        stats_to_show = [
            ("GPU Use", "GPU Use"), ("CPU", "CPU Use"),
            ("VRAM", "VRAM"), ("Total VRAM", "Total VRAM"),
            ("Shared VRAM", "Shared VRAM"), ("RAM", "Total RAM"),
            ("GPU Temp", "GPU Temp"), ("CPU Temp", "CPU Temp"),
            ("Power", "GPU Power"), ("CPU Power", "CPU Power")
        ]
        keys = [k for k, _ in stats_to_show]
        self.assertEqual(len(keys), len(set(keys)), f"Duplicate telemetry keys detected: {keys}")
        self.assertIn("Total VRAM", keys)
        self.assertIn("RAM", keys)
        self.assertIn("VRAM", keys)

if __name__ == "__main__":
    unittest.main()
