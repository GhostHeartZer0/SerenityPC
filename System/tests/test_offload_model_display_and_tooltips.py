import unittest
import tkinter as tk
import os
import sys

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from System.serenity_utils import ToolTip
from main import ChatbotApp


class TestOffloadModelDisplayAndToolTips(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()

    def tearDown(self):
        try:
            self.root.destroy()
        except Exception:
            pass

    def test_tooltip_no_duplicate_stacking(self):
        """Verify attaching multiple ToolTips to a widget unbinds old instances and avoids stacking."""
        btn = tk.Button(self.root, text="Hover Target")
        btn.pack()

        tip1 = ToolTip(btn, "First tooltip", delay_ms=10)
        self.assertIs(getattr(btn, "_serenity_tooltip", None), tip1)

        # Attaching a second tooltip should unbind the first
        tip2 = ToolTip(btn, "Second tooltip", delay_ms=10)
        self.assertIs(getattr(btn, "_serenity_tooltip", None), tip2)
        self.assertEqual(tip1._bind_ids, [])

        # Update text dynamically without creating a new instance
        tip2.update_text("Updated text")
        self.assertEqual(tip2.text, "Updated text")

    def test_offload_model_display_and_revert(self):
        """Verify that when offloaded, the status label shows Selected: (proposed) instead of Loaded:."""
        class DummyApp:
            def __init__(self, root):
                self.root = root
                self.config = {"show_tooltips": True, "marquee_text_enabled": False}
                self.model = None
                self.model_path = ""
                self.active_persona_level = 3
                self.depth_slider = None
                self.state = {}
                self.model_paths = {
                    "low": "C:\\Models\\Llama-3.2-3B-Q4.gguf",
                    "high": "C:\\Models\\Qwen-2.5-72B-Q4.gguf",
                }
                self.system_status_label = tk.Label(root, text="")
                self.system_status_label.pack()
                self._status_timer = None
                self.fonts = {"ui_small": ("Segoe UI", 9)}

            get_proposed_model_path = ChatbotApp.get_proposed_model_path
            _revert_status_label = ChatbotApp._revert_status_label
            _update_status_label_text = ChatbotApp._update_status_label_text

        app = DummyApp(self.root)

        # 1. Initially offloaded -> Should show proposed model as Selected:
        app._revert_status_label()
        displayed = app.system_status_label.cget("text")
        self.assertTrue(displayed.startswith("Selected: Llama-3.2-3B-Q4.gguf") or "..." in displayed)
        self.assertNotIn("Loaded:", displayed)
        self.assertIn("Llama-3.2-3B-Q4.gguf", app.system_status_label._serenity_tooltip.text)

        # 2. Simulate model loaded -> Should show Loaded:
        app.model = object()
        app.model_path = "C:\\Models\\Llama-3.2-3B-Q4.gguf"
        app._revert_status_label()
        displayed_loaded = app.system_status_label.cget("text")
        self.assertTrue(displayed_loaded.startswith("Loaded: Llama-3.2-3B-Q4.gguf") or "..." in displayed_loaded)

        # 3. Simulate offload -> Should clear model and revert to Selected:
        app.model = None
        app.model_path = ""
        app._revert_status_label()
        displayed_offloaded = app.system_status_label.cget("text")
        self.assertTrue(displayed_offloaded.startswith("Selected: Llama-3.2-3B-Q4.gguf") or "..." in displayed_offloaded)
        self.assertNotIn("Loaded:", displayed_offloaded)

        # 4. Change persona level to level 5 (high tier) -> Proposed changes to Qwen
        app.active_persona_level = 5
        app._revert_status_label()
        displayed_lvl5 = app.system_status_label.cget("text")
        self.assertTrue(displayed_lvl5.startswith("Selected: Qwen-2.5-72B-Q4.gguf") or "..." in displayed_lvl5)


if __name__ == "__main__":
    unittest.main()
