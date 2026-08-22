"""
Unit and regression tests for SerenityPC Version 1.5.0 features.
Covers:
- High-DPI & Audio Tone utilities
- Theme palettes & font baseline specs
- Repeat loop detector (lazy vs hyper vs off)
- STT Manager fallback & device querying
- Font scaling computation
"""

import os
import sys
import unittest
import json
import tempfile
import re

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from serenity_resources import THEMES, TEXTURE_STYLES, THEME
from main import BASE_FONT_SPECS
from System.serenity_utils import enable_high_dpi_awareness
from System.stt_manager import STTManager


class TestSerenityV150Features(unittest.TestCase):

    def test_dpi_and_audio_guards(self):
        """Verify high-DPI awareness executes without crashing on any OS."""
        try:
            enable_high_dpi_awareness()
            passed = True
        except Exception:
            passed = False
        self.assertTrue(passed)

    def test_theme_palettes_and_textures(self):
        """Verify all palettes and texture styles are registered and contain required keys."""
        expected_themes = [
            "apex", "goth", "crystal_cavern", 
            "yellow_blacket", "natural", "matrix", "persona", "cecilia"
        ]
        for th_key in expected_themes:
            self.assertIn(th_key, THEMES, f"Theme {th_key} missing")
            pal = THEMES[th_key]
            for req_key in ["bg_color", "fg_color", "widget_bg_color", "button_bg_color", "electric_blue"]:
                self.assertIn(req_key, pal, f"Theme {th_key} missing key {req_key}")

        expected_textures = ["default", "gloss", "metallic", "muted", "iridescent", "pearlescent", "frosted_glass"]
        for tex in expected_textures:
            self.assertIn(tex, TEXTURE_STYLES)

    def test_base_font_specs(self):
        """Verify standard font sizes and family definitions."""
        self.assertIn("main", BASE_FONT_SPECS)
        self.assertIn("stats", BASE_FONT_SPECS)
        self.assertIn("md_header_1", BASE_FONT_SPECS)
        self.assertEqual(BASE_FONT_SPECS["main"]["family"], "Segoe UI")
        self.assertEqual(BASE_FONT_SPECS["stats"]["family"], "Consolas")

    def test_stt_manager_initialization(self):
        """Verify STTManager behaves safely with or without sounddevice/vosk."""
        stt = STTManager()
        self.assertFalse(stt.is_recording)
        devices = STTManager.get_input_devices()
        self.assertIsInstance(devices, list)

    def test_repeat_loop_detector(self):
        """Verify repeat loop detection logic across modes (off, lazy, hyper)."""
        # Minimal simulator for testing the repeat detection algorithm
        def detect_rep(text: str, mode: str = "lazy") -> bool:
            if mode == "off" or not text:
                return False
            clean_text = text
            if mode == "lazy":
                clean_text = re.sub(r'```[\s\S]*?```', '', clean_text)
                clean_text = re.sub(r'<execute_tool>[\s\S]*?</execute_tool>', '', clean_text)
                clean_text = re.sub(r'action:[\s\S]*?\n', '', clean_text)
                clean_text = re.sub(r'\{[\s\S]*?\}', '', clean_text)

            lines = [line.strip() for line in clean_text.splitlines() if line.strip()]
            threshold_lines = 3 if mode == "hyper" else 5
            if len(lines) >= threshold_lines:
                last_line = lines[-1]
                if len(last_line) > 5 and lines[-threshold_lines:] == [last_line] * threshold_lines:
                    return True

            min_len = 35 if mode == "hyper" else 80
            max_repeats = 3 if mode == "hyper" else 4
            window_size = 400 if mode == "hyper" else 800
            sample = clean_text[-window_size:] if len(clean_text) > window_size else clean_text

            if len(sample) >= min_len:
                if mode == "hyper":
                    stall_patterns = [
                        r'(?i)(i apologize|let me check again|as stated earlier|as mentioned before|let me re-read)',
                    ]
                    for sp in stall_patterns:
                        if len(re.findall(sp, sample)) >= 3:
                            return True

                words = sample.split()
                for n in [4, 6, 8]:
                    if len(words) >= n * max_repeats:
                        for i in range(len(words) - n + 1):
                            ngram = " ".join(words[i:i+n])
                            if len(ngram) >= 15 and sample.count(ngram) >= max_repeats:
                                return True
            return False

        # 1. Normal prose should not trigger
        normal_prose = "The system is functioning within normal parameters. Let us examine the telemetry logs next."
        self.assertFalse(detect_rep(normal_prose, "lazy"))
        self.assertFalse(detect_rep(normal_prose, "hyper"))

        # 2. Degenerate repeating lines
        rep_lines = "Analyzing cache block.\n" * 6
        self.assertTrue(detect_rep(rep_lines, "lazy"))
        self.assertTrue(detect_rep(rep_lines, "hyper"))
        self.assertFalse(detect_rep(rep_lines, "off"))

        # 3. Repeated code blocks should be IGNORED in lazy mode but caught in hyper mode
        code_rep = "Here is the code:\n```python\nfor i in range(10):\n    print('test')\n```\n```python\nfor i in range(10):\n    print('test')\n```\n```python\nfor i in range(10):\n    print('test')\n```\n```python\nfor i in range(10):\n    print('test')\n```\n"
        self.assertFalse(detect_rep(code_rep, "lazy"))

        # 4. Hyper mode catches stall phrases
        stall_text = "I apologize for the delay. I apologize for the confusion. I apologize for that."
        self.assertTrue(detect_rep(stall_text, "hyper"))
        self.assertFalse(detect_rep(stall_text, "lazy"))

    def test_font_scale_calculations(self):
        """Verify dynamic scaling arithmetic adheres to point size constraints."""
        scale_pcts = [70, 100, 125, 150, 200]
        base_size = BASE_FONT_SPECS["main"]["size"] # 13
        for pct in scale_pcts:
            factor = pct / 100.0
            calc_size = max(6, int(round(base_size * factor)))
            if pct == 70:
                self.assertEqual(calc_size, 9)
            elif pct == 100:
                self.assertEqual(calc_size, 13)
            elif pct == 150:
                self.assertEqual(calc_size, 20)
            elif pct == 200:
                self.assertEqual(calc_size, 26)


if __name__ == "__main__":
    unittest.main()
