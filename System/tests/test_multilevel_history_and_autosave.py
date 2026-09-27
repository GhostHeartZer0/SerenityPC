import unittest
import os
import shutil
import tempfile
import json
import zlib
import tkinter as tk
from unittest.mock import MagicMock, patch

from main import ChatbotApp
from System.kv_manager import KVManager


class TestMultilevelHistoryAndAutosave(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.history_dir = os.path.join(self.test_dir, "History")
        os.makedirs(self.history_dir, exist_ok=True)

        self.root = tk.Tk()
        self.root.withdraw()
        self.app = MagicMock(spec=ChatbotApp)
        self.app.root = self.root
        self.app.model_path = "C:/Models/TestModel.gguf"
        self.app.active_persona_level = 4
        self.app.config = {
            "history_level_format": "All",
            "history_autosave_mode": "End",
            "history_usage": "all"
        }
        self.app.get_user_history_dir.return_value = self.history_dir
        self.app.get_active_username.return_value = "Default"
        self.app.vault_manager = MagicMock()
        self.app.vault_manager.is_lock_enabled.return_value = False

    def tearDown(self):
        try:
            self.root.destroy()
        except Exception:
            pass
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_pipeline_formatting_modes(self):
        """Test level pipeline formatting across All, full, ordered, first, last."""
        pipeline = [4, 6, 2, 3, 2, 4]

        # 'All': all unique levels used without repeats
        all_res = ChatbotApp._format_level_pipeline(self.app, pipeline, mode="All")
        self.assertEqual(all_res, [4, 6, 2, 3])

        # 'full': full pipeline / stacktrace with repeats
        full_res = ChatbotApp._format_level_pipeline(self.app, pipeline, mode="full")
        self.assertEqual(full_res, [4, 6, 2, 3, 2, 4])

        # 'ordered': levels in numerical order 1-7
        ord_res = ChatbotApp._format_level_pipeline(self.app, pipeline, mode="ordered")
        self.assertEqual(ord_res, [2, 3, 4, 6])

        # 'first': first model/level invoked
        first_res = ChatbotApp._format_level_pipeline(self.app, pipeline, mode="first")
        self.assertEqual(first_res, [4])

        # 'last': last model/level invoked
        last_res = ChatbotApp._format_level_pipeline(self.app, pipeline, mode="last")
        self.assertEqual(last_res, [4])

    def test_get_conversation_level_pipeline_extraction(self):
        """Verify pipeline is correctly extracted from messages."""
        self.app.messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello", "level": 4},
            {"role": "user", "content": "Deep thought"},
            {"role": "assistant", "content": "Reflecting", "level": 6},
            {"role": "user", "content": "Check"},
            {"role": "assistant", "content": "Verified", "level": 2}
        ]
        pipe = ChatbotApp._get_conversation_level_pipeline(self.app)
        self.assertEqual(pipe, [4, 6, 2])

    def test_get_history_path_multi_level(self):
        """Verify history path reflects multi-level format."""
        self.app.messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Ans1", "level": 4},
            {"role": "user", "content": "More"},
            {"role": "assistant", "content": "Ans2", "level": 6},
            {"role": "user", "content": "Again"},
            {"role": "assistant", "content": "Ans3", "level": 2}
        ]
        self.app._get_conversation_level_pipeline = lambda: [4, 6, 2]
        self.app._format_level_pipeline = lambda p, m: ChatbotApp._format_level_pipeline(self.app, p, m)

        # Default 'All'
        self.app.config["history_level_format"] = "All"
        path = ChatbotApp.get_history_path(self.app)
        self.assertIn("TestModel_lvls4_6_2.history", path)

        # Single level
        self.app._get_conversation_level_pipeline = lambda: [4]
        path_single = ChatbotApp.get_history_path(self.app)
        self.assertIn("TestModel_lvl4.history", path_single)

    def test_autosave_modes_enforcement(self):
        """Verify autosave behavior under End, Close, Manual."""
        # Toggle test
        self.app._log_and_display = MagicMock()
        self.app.save_config = MagicMock()
        self.app.history_autosave_button = MagicMock()
        self.app._get_history_autosave_label = lambda: "label"
        self.app._get_history_autosave_color = lambda: "#00ff88"

        self.app.config["history_autosave_mode"] = "End"
        ChatbotApp.toggle_history_autosave(self.app)
        self.assertEqual(self.app.config["history_autosave_mode"], "Close")

        ChatbotApp.toggle_history_autosave(self.app)
        self.assertEqual(self.app.config["history_autosave_mode"], "Manual")

        ChatbotApp.toggle_history_autosave(self.app)
        self.assertEqual(self.app.config["history_autosave_mode"], "End")

    def test_turbovec_multilevel_history_parsing(self):
        """Verify TurboVecIndex parses and matches multi-level archives."""
        from System.kv_manager import TurboVecIndex

        sample_hist = [{"role": "user", "content": "Quantum physics topic"}, {"role": "assistant", "content": "Wave function collapse"}]
        compressed = zlib.compress(json.dumps(sample_hist).encode("utf-8"))

        f_multi = os.path.join(self.history_dir, "testmodel_lvls4_6_2.history.jsonz")
        with open(f_multi, "wb") as fp:
            fp.write(compressed)

        tv = TurboVecIndex(self.history_dir, mode="fallback")
        tv.ingest_needed_files(active_model_path="C:/Models/testmodel.gguf", active_level=6, lookup_mode="targeted")
        self.assertIn(f_multi, tv._ingested_files)

    def test_pipeline_breakdown_header_insertion(self):
        """Verify pipeline breakdown header appears at top of loaded history view."""
        text_widget = tk.Text(self.root)
        self.app.past_history_view = text_widget
        self.app.history_state = {}
        self.app._history_content_cache = {}
        self.app._clean_latex_artifacts = lambda x: x
        self.app._get_persona_label = lambda: "Serenity"

        sample_hist = [
            {"role": "user", "content": "User question"},
            {"role": "assistant", "content": "Assistant answer", "level": 4},
            {"role": "user", "content": "Second question"},
            {"role": "assistant", "content": "Second answer", "level": 6}
        ]
        sample_path = os.path.join(self.history_dir, "test_file_lvls4_6.history.jsonz")
        with open(sample_path, "wb") as fp:
            fp.write(zlib.compress(json.dumps(sample_hist).encode("utf-8")))

        ChatbotApp._load_selected_history(self.app, sample_path)
        content = text_widget.get("1.0", tk.END)
        self.assertIn("PIPELINE BREAKDOWN: Lvl 4 -> Lvl 6", content)


if __name__ == "__main__":
    unittest.main()
