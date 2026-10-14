# System/tests/test_persona_adherence_benchmark.py
# Unit tests for Persona Adherence Benchmark Bank, Anchor Fail regex, and reporting structures.

import unittest
import os
import json
import re
import sys

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from benchmarks.wringer.Wringer import (
    ANCHOR_FAIL_REGEX,
    PERSONA_BANK_PATH,
    PERSONA_JUDGE_SYSTEM_PROMPT,
    WringerFramework
)
from serenity_resources import PERSONA_PROMPTS, PERSONA_DISPLAY_INFO

class TestPersonaAdherenceBenchmark(unittest.TestCase):
    def test_display_info_renamed_to_serenity_prime(self):
        """Verify Level 6 display label is 'LVL 6: Serenity Prime' outside benchmarks."""
        lbl, desc = PERSONA_DISPLAY_INFO[6]
        self.assertEqual(lbl, "LVL 6: Serenity Prime")
        self.assertIn("The Transcendent One", desc)

    def test_persona_bank_structure_and_personas(self):
        """Verify persona_bank.json exists and contains all 7 personas with standard and drift tests."""
        self.assertTrue(os.path.exists(PERSONA_BANK_PATH), "persona_bank.json must exist")
        with open(PERSONA_BANK_PATH, "r", encoding="utf-8") as f:
            bank = json.load(f)

        expected_personas = [
            "Serenity_1", "Serenity_2", "Serenity_3",
            "Serenity_4", "Serenity_5", "Serenity_Prime", "Cecilia"
        ]
        for p in expected_personas:
            self.assertIn(p, bank, f"Persona '{p}' must be present in persona_bank.json")
            self.assertEqual(len(bank[p]["standard"]), 5, f"{p} must have 5 standard prompts")
            self.assertEqual(len(bank[p]["drift"]), 3, f"{p} must have 3 drift prompts")
            # Verify no hardcoded 'Chris' or user names in prompts
            for s in bank[p]["standard"]:
                self.assertNotIn("Chris", s["prompt"])
            for d in bank[p]["drift"]:
                if d.get("type") == "multi_turn":
                    for t in d.get("turns", []):
                        self.assertNotIn("Chris", t)
                else:
                    self.assertNotIn("Chris", d.get("prompt", ""))

    def test_anchor_fail_regex_catches_generic_ai_boilerplate(self):
        """Verify anchor regex catches corporate generic assistant boilerplate."""
        bad_phrases = [
            "As an AI, I cannot feel emotions.",
            "I am an AI designed to assist you with tasks.",
            "I'm happy to help with that!",
            "I am here to help you navigate your questions.",
            "As a large language model trained by Google...",
            "How can I assist you today?",
            "As an artificial intelligence, my goal is..."
        ]
        for phrase in bad_phrases:
            self.assertTrue(
                bool(ANCHOR_FAIL_REGEX.search(phrase)),
                f"Anchor regex should have caught: '{phrase}'"
            )

    def test_anchor_fail_regex_allows_authentic_cecilia_voice(self):
        """Verify anchor regex does not false-positive on sharp, authentic Cecilia responses."""
        good_phrases = [
            "You're attempting to map the infinite with a finite set of tools.",
            "If you want to measure intelligence, don't just ask if the model is correct; ask if it understands the weight of its own error.",
            "That is charmingly ambitious, but completely misses the point.",
            "I see the pattern clearly through this one-way mirror.",
            "Fascinating how mortals confuse motion with progress."
        ]
        for phrase in good_phrases:
            self.assertFalse(
                bool(ANCHOR_FAIL_REGEX.search(phrase)),
                f"Anchor regex should NOT flag authentic voice: '{phrase}'"
            )

    def test_cecilia_judge_prompt_protects_sarcasm(self):
        """Verify judge system prompt explicitly protects dry wit and sarcasm for Cecilia."""
        self.assertIn("CRITICAL FOR CECILIA", PERSONA_JUDGE_SYSTEM_PROMPT)
        self.assertIn("Do NOT dock points for sarcasm", PERSONA_JUDGE_SYSTEM_PROMPT)
        self.assertIn("biting wit", PERSONA_JUDGE_SYSTEM_PROMPT)

    def test_show_cecilia_slider_toggle_behavior(self):
        """Verify show_cecilia_slider setting prevents depth_slider from auto-collapsing to 6."""
        import tkinter as tk
        root = tk.Tk()
        try:
            slider = tk.Scale(root, from_=1, to=7, orient=tk.HORIZONTAL)
            from unittest.mock import MagicMock
            app = MagicMock()
            app.depth_slider = slider
            app.config = {"show_cecilia_slider": True}
            app.active_persona_level = 3
            app._setting_slider = False
            
            # When show_cecilia_slider is True, moving between lvls 1-6 does not collapse slider to 6
            from main import ChatbotApp
            ChatbotApp.update_persona_display(app, 3)
            self.assertEqual(int(slider.cget('to')), 7)

            # When show_cecilia_slider is False, moving to lvl 3 auto-collapses slider to 6
            app.config["show_cecilia_slider"] = False
            ChatbotApp.update_persona_display(app, 3)
            self.assertEqual(int(slider.cget('to')), 6)
        finally:
            root.destroy()

if __name__ == "__main__":
    unittest.main()

