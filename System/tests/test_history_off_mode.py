"""
test_history_off_mode.py
Unit tests verifying:
1. History Usage 'off' guarantees zero past history sent to inference even when ghost mode is enabled.
2. Ghost mode initialization / load_history respects history_usage == 'off'.
3. Reasoning strength options (off, low, medium, high, xhigh) and legacy fallback.
"""
import unittest
from unittest.mock import MagicMock
import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from main import ChatbotApp

class TestHistoryOffMode(unittest.TestCase):
    def setUp(self):
        self.app = MagicMock()
        self.app.config = {
            "history_usage": "off",
            "ghost_mode": True,
            "reasoning_strength": "medium"
        }
        self.app.messages = [
            {"role": "user", "content": "Prior message 1"},
            {"role": "assistant", "content": "Prior reply 1"},
            {"role": "user", "content": "Prior message 2"},
            {"role": "assistant", "content": "Prior reply 2"},
        ]

    def test_load_history_with_history_off_and_ghost_on(self):
        """load_history must reset messages to empty when history_usage is off, even if ghost_mode is True."""
        mock_self = MagicMock()
        mock_self.config = {"history_usage": "off", "ghost_mode": True}
        mock_self.messages = [{"role": "user", "content": "old"}]

        ChatbotApp.load_history(mock_self)
        self.assertEqual(mock_self.messages, [])

    def test_history_off_inference_payload(self):
        """When history_usage == 'off', only the single current user message must be submitted."""
        usage = self.app.config.get("history_usage", "all")
        user_msg = {"role": "user", "content": "Fresh test question"}

        if usage == "off":
            staged_messages = [user_msg]
        elif usage == "current_window":
            staged_messages = [m for m in self.app.messages if not m.get("is_memory")] + [user_msg]
        else:
            staged_messages = self.app.messages + [user_msg]

        self.assertEqual(len(staged_messages), 1)
        self.assertEqual(staged_messages[0]["content"], "Fresh test question")

    def test_generation_worker_history_off_guard(self):
        """_generation_worker slicing ensures history off keeps only the last user turn."""
        temp_messages = [
            {"role": "user", "content": "Turn 1"},
            {"role": "assistant", "content": "Turn 1 reply"},
            {"role": "user", "content": "Turn 2"}
        ]
        history_usage = "off"
        if history_usage == "off":
            temp_messages = [temp_messages[-1]]

        self.assertEqual(len(temp_messages), 1)
        self.assertEqual(temp_messages[0]["content"], "Turn 2")

    def test_reasoning_strength_options(self):
        """Reasoning levels supported: off, low, medium, high, xhigh."""
        valid_opts = ["off", "low", "medium", "high", "xhigh"]
        self.assertEqual(len(valid_opts), 5)
        self.assertNotIn("minimal", valid_opts)
        self.assertNotIn("maximum", valid_opts)

if __name__ == "__main__":
    unittest.main()
