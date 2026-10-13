import unittest
import tkinter as tk
import os
import sys
import zlib
import json
from unittest.mock import MagicMock

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from main import ChatbotApp
from System.orchestration_manager import clean_model_display_name


class TestSubagentBannerAndHistoryCondensing(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()

    def tearDown(self):
        try:
            self.root.destroy()
        except Exception:
            pass

    def test_clean_model_display_name(self):
        """Verify model filenames are stripped of quants and extensions for clean banner display."""
        self.assertEqual(clean_model_display_name("C:\\Models\\Qwen3.8-27B-UD-Q4_K_XL.gguf"), "Qwen3.8-27B")
        self.assertEqual(clean_model_display_name("gemma-4-26B-A4B-it-UD-Q6_K_XL.gguf"), "gemma-4-26B-A4B")
        self.assertEqual(clean_model_display_name("Spark-X2.5-4B-Q4_K_M.gguf"), "Spark-X2.5-4B")
        self.assertEqual(clean_model_display_name("Llama-3.2-3B-Q4.gguf"), "Llama-3.2-3B")

    def test_subagent_banner_display_and_revert(self):
        """Verify that when a subagent is active, the banner shows 'Subagent: Lvl X, Model' and reverts when done."""
        class DummyApp:
            def __init__(self, root):
                self.root = root
                self.config = {"show_tooltips": True, "marquee_text_enabled": False}
                self.model = object()
                self.model_path = "C:\\Models\\Qwen3.8-27B-UD-Q4_K_XL.gguf"
                self.active_persona_level = 6
                self.state = {}
                self.system_status_label = tk.Label(root, text="")
                self.system_status_label.pack()
                self._status_timer = None
                self.scale_factor = 1.0

            get_proposed_model_path = ChatbotApp.get_proposed_model_path
            _revert_status_label = ChatbotApp._revert_status_label
            _update_status_label_text = ChatbotApp._update_status_label_text
            _handle_subagent_status = ChatbotApp._handle_subagent_status

        app = DummyApp(self.root)

        # 1. Base state: Loaded model
        app._revert_status_label()
        self.assertTrue(app.system_status_label.cget("text").startswith("Loaded: Qwen3.8-27B-UD-Q4_K_XL.gguf") or "..." in app.system_status_label.cget("text"))

        # 2. Subagent active: Lvl 5, Qwen3.8-27B
        app._handle_subagent_status({"active": True, "level": 5, "model": "Qwen3.8-27B"})
        displayed = app.system_status_label.cget("text")
        self.assertTrue(displayed.startswith("Subagent: Lvl 5, Qwen3.8-27B") or "..." in displayed)
        self.assertEqual(app.state.get("active_subagent"), {"level": 5, "model": "Qwen3.8-27B"})

        # 3. While subagent is active, calling _revert_status_label preserves the Subagent banner
        app._revert_status_label()
        self.assertTrue(app.system_status_label.cget("text").startswith("Subagent: Lvl 5, Qwen3.8-27B") or "..." in app.system_status_label.cget("text"))

        # 4. Subagent inactive: reverts to Loaded orchestrator model
        app._handle_subagent_status({"active": False})
        self.assertIsNone(app.state.get("active_subagent"))
        self.assertTrue(app.system_status_label.cget("text").startswith("Loaded: Qwen3.8-27B-UD-Q4_K_XL.gguf") or "..." in app.system_status_label.cget("text"))

    def test_format_levels_condensed(self):
        """Verify _format_levels_condensed groups repeating levels into count notation."""
        self.assertEqual(ChatbotApp._format_levels_condensed([7, 7, 7, 7, 7, 7, 7, 7]), "7(8)")
        self.assertEqual(ChatbotApp._format_levels_condensed([7, 7, 7, 5, 5, 2]), "7(3), 5(2), 2")
        self.assertEqual(ChatbotApp._format_levels_condensed([4, 6, 2]), "4, 6, 2")
        self.assertEqual(ChatbotApp._format_levels_condensed([7]), "7")
        self.assertEqual(ChatbotApp._format_levels_condensed([]), "")

    def test_history_path_concise_no_quantity_repeats(self):
        """Verify get_history_path does not repeat multiple instances of the same level in filenames."""
        app = MagicMock(spec=ChatbotApp)
        app.model_path = "C:\\Models\\Gemma-4.gguf"
        app.get_user_history_dir = lambda: "C:\\History"
        app.get_active_username = lambda: "Default"
        app.active_persona_level = 7
        app.config = {"history_level_format": "full"}
        app._format_level_pipeline = lambda p, m: ChatbotApp._format_level_pipeline(app, p, m)

        # When 8 consecutive Level 7 turns exist
        app._get_conversation_level_pipeline = lambda: [7, 7, 7, 7, 7, 7, 7, 7]
        path = ChatbotApp.get_history_path(app)
        self.assertIn("Gemma-4_lvl7.history", path)
        self.assertNotIn("7_7_7", path)

        # When multiple distinct levels exist
        app._get_conversation_level_pipeline = lambda: [7, 7, 5, 5, 2]
        path_multi = ChatbotApp.get_history_path(app)
        self.assertIn("Gemma-4_lvls7_5_2.history", path_multi)
        self.assertNotIn("7_7", path_multi)

    def test_pipeline_breakdown_condensed_in_loaded_history(self):
        """Verify _load_selected_history renders condensed repeats in pipeline breakdown header."""
        app = MagicMock(spec=ChatbotApp)
        text_widget = tk.Text(self.root)
        app.past_history_view = text_widget
        app.history_state = {}
        app._history_content_cache = {}
        app._clean_latex_artifacts = lambda x: x
        app._get_persona_label = lambda: "Serenity"

        # Simulate 8 level-7 turns
        msgs = [{"role": "assistant", "content": "turn", "level": 7} for _ in range(8)]
        app.vault_manager = MagicMock()
        app.vault_manager.read_history_messages = lambda path: msgs

        ChatbotApp._load_selected_history(app, "dummy_lvl7.history.jsonz")
        content = text_widget.get("1.0", tk.END)
        self.assertIn("PIPELINE BREAKDOWN: Lvl 7(8)", content)
        self.assertNotIn("Lvl 7 -> Lvl 7", content)


if __name__ == "__main__":
    unittest.main()
