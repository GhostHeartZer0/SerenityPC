# test_mmproj_lookup.py
# Unit and integration test suite for mmproj lookup hierarchy, parameter matching, and fallback selection.

import unittest
import sys
import os

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

class MockApp:
    def __init__(self):
        self.config = {
            "mmproj_mapping": {}
        }
        self.model_paths = {
            "med": os.path.join(base_dir, "Models", "12B", "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf"),
            "high": os.path.join(base_dir, "Models", "26B-A4B", "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"),
            "fast": os.path.join(base_dir, "Models", "E4B", "gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"),
            "vision_multimodal": os.path.join(base_dir, "Models", "E4B", "gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf"),
            "vision_multimodal_projector": os.path.join(base_dir, "Models", "E4B", "mmproj-BF16.gguf")
        }
        self.model_path = self.model_paths["med"]
        self.current_model_tier = "med"
        self.save_config_called = False

    def save_config(self):
        self.save_config_called = True

    # Bind the exact lookup method from main.py
    from main import ChatbotApp
    _is_native_encoder_decoder_model = ChatbotApp._is_native_encoder_decoder_model
    _find_projector_for_model = ChatbotApp._find_projector_for_model
    _ensure_chat_handler = ChatbotApp._ensure_chat_handler

from unittest.mock import patch

class TestMmprojLookup(unittest.TestCase):
    def setUp(self):
        self.app = MockApp()
        self.app.model = object() # Dummy model instance

    @patch("os.path.exists")
    @patch("os.listdir")
    def test_e4b_model_matches_e4b_projector(self, mock_listdir, mock_exists):
        """Verify E4B model automatically matches adjacent or size-matched E4B projector."""
        mock_exists.side_effect = lambda p: "E4B" in p or "mmproj" in p or p.endswith(".gguf")
        mock_listdir.return_value = ["gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf", "mmproj-BF16.gguf"]
        proj = self.app._find_projector_for_model(self.app.model_paths["fast"], interactive=False)
        self.assertIsNotNone(proj)
        self.assertIn("E4B", proj)

    @patch("os.path.exists")
    @patch("os.listdir")
    def test_26b_model_matches_26b_projector(self, mock_listdir, mock_exists):
        """Verify 26B model automatically matches adjacent or size-matched 26B projector."""
        mock_exists.side_effect = lambda p: "26B-A4B" in p or "mmproj" in p or p.endswith(".gguf")
        mock_listdir.return_value = ["gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf", "mmproj-BF16.gguf"]
        proj = self.app._find_projector_for_model(self.app.model_paths["high"], interactive=False)
        self.assertIsNotNone(proj)
        self.assertIn("26B-A4B", proj)

    @patch("os.path.exists")
    @patch("os.listdir")
    @patch("os.walk")
    def test_12b_model_avoids_incompatible_e4b_projector(self, mock_walk, mock_listdir, mock_exists):
        """Verify 12B model does NOT blindly bind an incompatible 2560-dim E4B projector."""
        mock_exists.side_effect = lambda p: "12B" in p
        mock_listdir.return_value = ["gemma-4-12B-it-qat-UD-Q4_K_XL.gguf"]
        mock_walk.return_value = []
        proj = self.app._find_projector_for_model(self.app.model_paths["med"], interactive=False)
        self.assertIsNone(proj)

    def test_12b_detected_as_native_encoder_decoder(self):
        """Verify 12B model is detected as native encoder/decoder and passes chat handler check without mmproj."""
        is_native = self.app._is_native_encoder_decoder_model(self.app.model_paths["med"])
        self.assertTrue(is_native)
        self.app.model_path = self.app.model_paths["med"]
        self.assertTrue(self.app._ensure_chat_handler(interactive=False))

    @patch("os.path.exists")
    def test_persistent_mapping_takes_priority(self, mock_exists):
        """Verify user manual mapping in mmproj_mapping is prioritized."""
        mock_exists.return_value = True
        fake_custom_path = os.path.join(base_dir, "Models", "26B-A4B", "mmproj-BF16.gguf")
        self.app.config["mmproj_mapping"][self.app.model_paths["med"]] = fake_custom_path
        proj = self.app._find_projector_for_model(self.app.model_paths["med"], interactive=False)
        self.assertEqual(proj, fake_custom_path)

if __name__ == "__main__":
    unittest.main()
