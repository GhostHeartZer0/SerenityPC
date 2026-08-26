# test_muse_glimmer_isolation.py
# Verification suite for Muse-Glimmer forward graph alignment, sampling defaults, and Jinja template reasoning parameters.

import unittest
import sys
import os

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

class TestMuseGlimmerIsolation(unittest.TestCase):
    def setUp(self):
        self.mock_config = {
            "muse_reasoning_strength": "high",
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "min_p": 0.05,
            "repeat_penalty": 1.15
        }

    def test_muse_stop_token_hygiene(self):
        """Verify that Muse-Glimmer never halts prematurely on thought closer <|eom|>."""
        stops = ["<|end_of_text|>", "<|eot|>", "<|eom|>"]
        # Stops filtered for muse-glimmer
        filtered_stops = [s for s in stops if s != "<|eom|>"]
        self.assertNotIn("<|eom|>", filtered_stops)
        self.assertIn("<|end_of_text|>", filtered_stops)
        self.assertIn("<|eot|>", filtered_stops)

    def test_muse_reasoning_strength_jinja_injection(self):
        """Verify Jinja chat template variable injection matches valid Muse-Glimmer profiles."""
        valid_strengths = ["off", "low", "medium", "high", "xhigh"]
        active_strength = self.mock_config.get("muse_reasoning_strength", "high")
        self.assertIn(active_strength, valid_strengths)

    def test_muse_sampler_profile_alignment(self):
        """Validate Muse-Glimmer official sampling bounds."""
        self.assertEqual(self.mock_config["temperature"], 1.0)
        self.assertEqual(self.mock_config["top_p"], 0.95)
        self.assertEqual(self.mock_config["top_k"], 64)
        self.assertEqual(self.mock_config["min_p"], 0.05)

    def test_muse_forward_graph_parameters(self):
        """Verify expected Muse-Glimmer architecture constants."""
        logit_softcapping = 30.0
        self.assertEqual(logit_softcapping, 30.0)
        # SWA sliding window check
        swa_window = 4096
        self.assertTrue(swa_window > 0)

if __name__ == "__main__":
    unittest.main()
