import unittest
import tkinter as tk
from unittest.mock import MagicMock, patch
import sys
import os

# Add repo root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from main import ChatbotApp
from System.markdown_engine import MarkdownEngine

class TestThoughtsDropdownAndEcstaticTimer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tk.Tk()
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.root.destroy()
        except:
            pass

    def setUp(self):
        # Create minimal mock of ChatbotApp without booting full hardware backend
        self.app = MagicMock()
        self.app.root = self.root
        self.app.state = {
            "response_started": False,
            "thought_streamed": False,
            "current_think_tag": None,
            "current_think_btn": None,
            "current_dropdown_frame": None,
            "avatar_current": "listening"
        }
        self.app.config = {
            "media_rendering": 1,
            "thinking_checkbox": True,
            "reasoning_strength": "medium"
        }
        self.app.fonts = {
            "stats": ("Consolas", 9),
            "log": ("Consolas", 10),
            "md_thought": ("Consolas", 11, "italic")
        }
        self.app._get_persona_label = MagicMock(return_value="Serenity")
        self.app.chat_history = tk.Text(self.root)
        self.app._apply_markdown = lambda s, e, b=("ai",), is_thought=False: ChatbotApp._apply_markdown(self.app, s, e, b, is_thought)
        self.app._post_process_media = MagicMock()
        self.app._save_rlhf_log = MagicMock()
        self.app._is_chat_at_bottom = MagicMock(return_value=True)
        self.app.save_history = MagicMock()
        self.app.messages = []
        self.app.dirs = {"Media": "."}
        self.app.right_panel = MagicMock()
        self.app.avatar_pil_images = {}
        self.app.avatar_states = {}
        self.app._get_avatar_target_size = MagicMock(return_value=(200, 200))
        self.app.active_persona_level = 3
        self.app.idle_timer_id = None
        self.app.ecstatic_timer_id = None
        self.app._sanitize_synthesis_output = lambda x: x
        self.app.thought_dropdown_buttons = []
        self.app._append_to_chat = lambda text, tag="ai": (
            self.app.chat_history.config(state='normal'),
            self.app.chat_history.insert(tk.END, text, (tag,)),
            self.app.chat_history.config(state='disabled')
        )

    def tearDown(self):
        try:
            self.app.chat_history.destroy()
        except:
            pass

    def test_non_streamed_thoughts_placed_above_response_with_markdown(self):
        """Verify that when thoughts are not streamed live, dropdown is placed above response and response markdown is rendered."""
        hist = self.app.chat_history
        hist.config(state='normal')
        start_idx = hist.index(tk.END + "-1c")
        self.app.state["response_started"] = True
        self.app.state["response_start_idx"] = start_idx
        self.app.state["thought_streamed"] = False

        think_log = "Analyzing the query structure and calculating parameters."
        final_answer = "Here is the result: *italic* and **bold** text."

        # Call real _finalize_message
        ChatbotApp._finalize_message(self.app, "User prompt", think_log, final_answer, False)

        content = hist.get("1.0", tk.END)
        # Verify persona lead appears first
        self.assertIn("Serenity:", content)
        
        # Verify thoughts dropdown button exists in thought_dropdown_buttons
        self.assertTrue(len(self.app.thought_dropdown_buttons) > 0)
        btn = self.app.thought_dropdown_buttons[-1]
        self.assertEqual(btn.cget("text"), "[+] View Thinking Process")

        # Verify thoughts are elided by default
        think_tags = [t for t in hist.tag_names() if t.startswith("think_block_")]
        self.assertTrue(len(think_tags) > 0)
        think_tag = think_tags[0]
        is_elided = str(hist.tag_cget(think_tag, "elide")) in ["1", "True", "true"]
        self.assertTrue(is_elided)

        # Verify markdown was applied to the final answer (*italic* and **bold**)
        italic_ranges = hist.tag_ranges("md_italic")
        bold_ranges = hist.tag_ranges("md_bold")
        self.assertTrue(len(italic_ranges) >= 2, "md_italic should be applied to *italic*")
        self.assertTrue(len(bold_ranges) >= 2, "md_bold should be applied to **bold**")
        self.assertEqual(hist.get(italic_ranges[0], italic_ranges[1]), "italic")
        self.assertEqual(hist.get(bold_ranges[0], bold_ranges[1]), "bold")

    def test_empty_thoughts_do_not_create_dropdown(self):
        """Verify that empty or whitespace-only thoughts do not create a thought dropdown button."""
        hist = self.app.chat_history
        hist.config(state='normal')
        start_idx = hist.index(tk.END + "-1c")
        self.app.state["response_started"] = True
        self.app.state["response_start_idx"] = start_idx
        self.app.state["thought_streamed"] = False

        # Empty/whitespace think_log
        ChatbotApp._finalize_message(self.app, "Hello", "   \n\n  ", "Direct answer.", False)

        btn = self.app.state.get("current_think_btn")
        self.assertIsNone(btn)
        self.assertNotIn("[+] View Thinking Process", hist.get("1.0", tk.END))

    def test_ecstatic_avatar_auto_revert_timer(self):
        """Verify that serenity_ecstatic sets a 4.5s auto-revert timer to listening."""
        self.app.set_avatar_state = lambda s: ChatbotApp.set_avatar_state(self.app, s)
        self.app._revert_ecstatic_avatar = lambda: ChatbotApp._revert_ecstatic_avatar(self.app)
        self.app._parse_dmn_timeout_sec = MagicMock(return_value=0)
        self.app._set_persona_idle_state = MagicMock()

        # Set avatar state to ecstatic
        self.app.set_avatar_state("ecstatic")
        self.assertIsNotNone(self.app.ecstatic_timer_id)

        # Trigger the revert handler
        self.app._revert_ecstatic_avatar()
        self.assertIsNone(self.app.ecstatic_timer_id)
        self.assertEqual(self.app.state.get("avatar_current"), "listening")

    def test_ecstatic_timer_cancels_on_state_change(self):
        """Verify that changing state before 4.5s cancels any running ecstatic timer."""
        self.app.set_avatar_state = lambda s: ChatbotApp.set_avatar_state(self.app, s)
        self.app._revert_ecstatic_avatar = lambda: ChatbotApp._revert_ecstatic_avatar(self.app)
        self.app._parse_dmn_timeout_sec = MagicMock(return_value=0)
        self.app._set_persona_idle_state = MagicMock()

        self.app.set_avatar_state("ecstatic")
        timer_id = self.app.ecstatic_timer_id
        self.assertIsNotNone(timer_id)

        # Switch to pleased
        self.app.set_avatar_state("pleased")
        self.assertIsNone(self.app.ecstatic_timer_id)

if __name__ == "__main__":
    unittest.main()
