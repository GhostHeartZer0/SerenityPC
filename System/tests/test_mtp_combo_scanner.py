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

    def test_baked_in_mtp_combo_priority(self):
        models = [
            {
                "filename": "Qwen3.8-27B-Coder390-EfficientThink-Q4LynnStyle-Q8MTP.gguf",
                "path": "S:/LLM/Qwen3.8/Qwen3.8-27B-Coder390-EfficientThink-Q4LynnStyle-Q8MTP.gguf",
                "size_gb": 16.5,
                "arch": "qwen35",
                "vocab_size": 152064,
                "context_len": 32768,
                "is_assistant_head": False,
                "has_baked_in_mtp": True
            },
            {
                "filename": "mtp-Qwen3.8-27B-Q4_0.gguf",
                "path": "S:/LLM/Qwen3.8/mtp-Qwen3.8-27B-Q4_0.gguf",
                "size_gb": 1.2,
                "arch": "llama",
                "vocab_size": 152064,
                "context_len": 4096,
                "is_assistant_head": True,
                "has_baked_in_mtp": False
            }
        ]
        combos = find_acceptable_combos(models, 6.0)
        self.assertEqual(len(combos), 1)
        combo = combos[0]
        self.assertTrue(combo["has_baked_in_mtp"])
        self.assertIsNone(combo["drafter"])
        self.assertEqual(combo["fallback"], "Native Baked-in MTP")

    def test_has_baked_in_mtp_detection(self):
        from System.mtp_combo_scanner import has_baked_in_mtp
        # From metadata
        self.assertTrue(has_baked_in_mtp("dummy.gguf", {"qwen35.nextn_predict_layers": 1}))
        # From filename
        self.assertTrue(has_baked_in_mtp("Model-Q8MTP.gguf", {}))
        # Negative
        self.assertFalse(has_baked_in_mtp("Model-Standard.gguf", {"block_count": 32}))

    def test_is_mtp_assistant_detection(self):
        from System.mtp_combo_scanner import is_mtp_assistant
        # Gemma 4 assistant architectures (hyphen and underscore)
        self.assertTrue(is_mtp_assistant("dummy.gguf", {"general.architecture": "gemma4-assistant"}))
        self.assertTrue(is_mtp_assistant("dummy.gguf", {"general.architecture": "gemma4_assistant"}))
        self.assertTrue(is_mtp_assistant("gemma-4-12B-it-MTP-Q8_0.gguf", {"general.architecture": "gemma4-assistant"}))
        # Primary target model
        self.assertFalse(is_mtp_assistant("gemma4-v2-Q4_K_M.gguf", {"general.architecture": "gemma4"}))

    def test_dedicated_mtp_head_combo_pairing(self):
        models = [
            {
                "filename": "gemma4-v2-Q4_K_M.gguf",
                "path": "S:/LLM/gemma4-v2-Q4_K_M.gguf",
                "size_gb": 7.5,
                "arch": "gemma4",
                "vocab_size": 262144,
                "context_len": 49152,
                "is_assistant_head": False,
                "has_baked_in_mtp": False
            },
            {
                "filename": "gemma-4-12B-it-MTP-Q8_0.gguf",
                "path": "S:/LLM/gemma-4-12B-it-MTP-Q8_0.gguf",
                "size_gb": 1.1,
                "arch": "gemma4-assistant",
                "vocab_size": 262144,
                "context_len": 4096,
                "is_assistant_head": True,
                "has_baked_in_mtp": False
            }
        ]
        combos = find_acceptable_combos(models, 6.0)
        self.assertEqual(len(combos), 1)
        combo = combos[0]
        self.assertEqual(combo["target"]["filename"], "gemma4-v2-Q4_K_M.gguf")
        self.assertIsNotNone(combo["drafter"])
        self.assertEqual(combo["drafter"]["filename"], "gemma-4-12B-it-MTP-Q8_0.gguf")
        self.assertTrue(combo["is_mtp_assistant"])
        self.assertEqual(combo["fallback"], "Native MTP Assistant")

if __name__ == "__main__":
    unittest.main()
