# test_chat_scroll_lock.py
# Verification suite for chat scrolling, scroll lock persistence, and long-window behavior.

import unittest
import sys
import os
import tkinter as tk
from tkinter import scrolledtext

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from main import ChatbotApp

class TestChatScrollLock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tk.Tk()
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.root.destroy()
        except Exception:
            pass

    def setUp(self):
        self.frame = tk.Frame(self.root)
        self.frame.pack()
        self.txt_chat = scrolledtext.ScrolledText(self.frame, height=10, width=40)
        self.txt_chat.pack()

        # Mock minimal ChatbotApp
        self.app = type("MockApp", (), {})()
        self.app.root = self.root
        self.app.chat_history = self.txt_chat
        self.app.config = {"scroll_lock_enabled": False}
        self.app.state = {"response_started": True, "current_think_tag": "think_1", "current_agentic_tag": "agentic_1"}
        self.app.active_persona_level = 3
        self.app.chunk_counter = 0
        self.app._user_scrolled_up = False
        self.app.thinking_display = None
        self.app.fonts = {"main": ("Segoe UI", 10), "bold": ("Segoe UI", 10, "bold"), "log": ("Consolas", 9), "stats": ("Consolas", 8)}

        # Bind methods from ChatbotApp
        self.app._is_chat_at_bottom = ChatbotApp._is_chat_at_bottom.__get__(self.app, type(self.app))
        self.app._check_user_scroll = ChatbotApp._check_user_scroll.__get__(self.app, type(self.app))
        self.app._update_ai_message = ChatbotApp._update_ai_message.__get__(self.app, type(self.app))
        self.app._replace_ai_message = ChatbotApp._replace_ai_message.__get__(self.app, type(self.app))
        self.app._append_to_chat = ChatbotApp._append_to_chat.__get__(self.app, type(self.app))
        self.app._update_thought_dropdown = ChatbotApp._update_thought_dropdown.__get__(self.app, type(self.app))
        self.app._update_agentic_dropdown = ChatbotApp._update_agentic_dropdown.__get__(self.app, type(self.app))
        self.app._ensure_thought_dropdown = lambda: None
        self.app._ensure_agentic_dropdown = lambda: None
        self.app._animate_text_fade = lambda *args, **kwargs: None
        self.app._get_persona_label = lambda: "Serenity"

    def tearDown(self):
        try:
            self.frame.destroy()
        except Exception:
            pass

    def test_short_window_bottom_detection(self):
        """Verify _is_chat_at_bottom works reliably on small documents."""
        self.txt_chat.delete("1.0", tk.END)
        self.txt_chat.insert(tk.END, "Short text line 1\nShort text line 2\n")
        self.root.update_idletasks()
        
        self.assertTrue(self.app._is_chat_at_bottom())

    def test_long_window_scroll_detection(self):
        """Verify _is_chat_at_bottom and _check_user_scroll detect scroll on 3000+ line documents without percentage degradation."""
        self.txt_chat.delete("1.0", tk.END)
        # Insert 3000 lines
        lines = [f"Chat history line {i}\n" for i in range(3000)]
        self.txt_chat.insert(tk.END, "".join(lines))
        self.txt_chat.see("end-1c")
        self.root.update_idletasks()

        # At bottom
        self.assertTrue(self.app._is_chat_at_bottom())
        self.app._check_user_scroll()
        self.assertFalse(self.app._user_scrolled_up)

        # Scroll up just 2 units (a minor mousewheel turn)
        self.txt_chat.yview_scroll(-2, "units")
        self.root.update_idletasks()

        # Must detect that bottom is NOT visible even on massive documents
        self.assertFalse(self.app._is_chat_at_bottom())
        self.app._check_user_scroll()
        self.assertTrue(self.app._user_scrolled_up)

    def test_streaming_does_not_steal_scroll_when_scrolled_up(self):
        """Verify _update_ai_message preserves viewport position when user is scrolled up on long document."""
        self.txt_chat.delete("1.0", tk.END)
        lines = [f"Message backlog line {i}\n" for i in range(1000)]
        self.txt_chat.insert(tk.END, "".join(lines))
        self.txt_chat.see("end-1c")
        self.root.update_idletasks()

        # Scroll up to middle
        self.txt_chat.yview_moveto(0.5)
        self.root.update_idletasks()
        self.app._check_user_scroll()
        self.assertTrue(self.app._user_scrolled_up)

        pos_before = self.txt_chat.yview()[0]

        # Stream new token chunks
        for i in range(10):
            self.app._update_ai_message(f" Token {i}")
            self.root.update_idletasks()

        pos_after = self.txt_chat.yview()[0]

        # Viewport position must not have jumped to bottom
        self.assertAlmostEqual(pos_before, pos_after, places=3)
        self.assertLess(pos_after, 0.9)

    def test_dropdown_streams_do_not_steal_scroll_when_scrolled_up(self):
        """Verify thought and agentic dropdown updates preserve scroll when user scrolled up."""
        self.txt_chat.delete("1.0", tk.END)
        lines = [f"Backlog {i}\n" for i in range(1000)]
        self.txt_chat.insert(tk.END, "".join(lines))
        self.txt_chat.see("end-1c")
        self.root.update_idletasks()

        # Scroll up
        self.txt_chat.yview_moveto(0.4)
        self.root.update_idletasks()
        self.app._check_user_scroll()
        self.assertTrue(self.app._user_scrolled_up)

        pos_before = self.txt_chat.yview()[0]

        # Stream thought and agentic updates
        self.app._update_thought_dropdown("Reasoning token 1")
        self.app._update_agentic_dropdown("Agent action 1")
        self.root.update_idletasks()

        pos_after = self.txt_chat.yview()[0]
        self.assertAlmostEqual(pos_before, pos_after, places=3)

    def test_scroll_lock_enabled_freezes_viewport(self):
        """Verify scroll_lock_enabled prevents autoscroll even when at bottom."""
        self.txt_chat.delete("1.0", tk.END)
        lines = [f"Line {i}\n" for i in range(200)]
        self.txt_chat.insert(tk.END, "".join(lines))
        self.txt_chat.see("end-1c")
        self.root.update_idletasks()

        self.app.config["scroll_lock_enabled"] = True
        self.app._user_scrolled_up = False

        # Move slightly up
        self.txt_chat.yview_scroll(-5, "units")
        self.root.update_idletasks()
        pos_before = self.txt_chat.yview()[0]

        self.app._update_ai_message("New generation token")
        self.root.update_idletasks()

        pos_after = self.txt_chat.yview()[0]
        self.assertAlmostEqual(pos_before, pos_after, places=3)

if __name__ == "__main__":
    unittest.main()
