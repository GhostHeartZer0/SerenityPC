import unittest
from unittest.mock import MagicMock
from System.mtp_combo_scanner import compute_recommended_layers, find_acceptable_combos

class TestMtpComboScanner(unittest.TestCase):
    def test_compute_recommended_layers(self):
        # Small models (E2B, 3B): full offload
        self.assertEqual(compute_recommended_layers(2.5, 6.0), -1)
        self.assertEqual(compute_recommended_layers(3.8, 6.0), -1)

        # Mid models (12B): 16 layers
        self.assertEqual(compute_recommended_layers(7.2, 6.0), 16)

        # Large models (26B/31B): 8 layers max ceiling for 6GB VRAM
        self.assertEqual(compute_recommended_layers(13.3, 6.0), 8)
        self.assertEqual(compute_recommended_layers(15.6, 6.0), 8)

        # Extra large models (> 18GB): 4 layers
        self.assertEqual(compute_recommended_layers(24.0, 6.0), 4)

    def test_find_acceptable_combos(self):
        models = [
            {
                "filename": "target-gemma-26B.gguf",
                "path": "S:/LLM/target-gemma-26B.gguf",
                "size_gb": 13.3,
                "arch": "gemma",
                "vocab_size": 262144,
                "context_len": 8192,
                "is_assistant_head": False
            },
            {
                "filename": "drafter-gemma-E2B.gguf",
                "path": "S:/LLM/drafter-gemma-E2B.gguf",
                "size_gb": 3.2,
                "arch": "gemma",
                "vocab_size": 262144,
                "context_len": 8192,
                "is_assistant_head": False
            },
            {
                "filename": "incompatible-llama-3B.gguf",
                "path": "S:/LLM/incompatible-llama-3B.gguf",
                "size_gb": 2.0,
                "arch": "llama",
                "vocab_size": 128256,
                "context_len": 8192,
                "is_assistant_head": False
            }
        ]

        combos = find_acceptable_combos(models, 6.0)
        self.assertEqual(len(combos), 1)
        combo = combos[0]
        self.assertEqual(combo["target"]["filename"], "target-gemma-26B.gguf")
        self.assertIsNotNone(combo["drafter"])
        self.assertEqual(combo["drafter"]["filename"], "drafter-gemma-E2B.gguf")
        self.assertEqual(combo["target_rec_layers"], 8)
        self.assertEqual(combo["drafter_rec_layers"], 0)
        self.assertEqual(combo["fallback"], "Prompt Lookup Decoding")

if __name__ == "__main__":
    unittest.main()
