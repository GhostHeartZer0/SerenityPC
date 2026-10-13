import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import serenity_resources
from serenity_resources import CHAT_FG_COLORS

class TestChatColorAndPrefix(unittest.TestCase):
    def test_chat_fg_colors_distinctness(self):
        self.assertIn(2, CHAT_FG_COLORS, "Lvl 2 color must be present in CHAT_FG_COLORS")
        self.assertEqual(CHAT_FG_COLORS[2], "#FFC299", "Lvl 2 color must be warm coral (#FFC299)")
        
        for lvl in range(8):
            self.assertIn(lvl, CHAT_FG_COLORS)
            self.assertNotEqual(CHAT_FG_COLORS[lvl].upper(), "#FFFFFF", f"Lvl {lvl} color should be distinct from white")

    def test_persona_labels_exist(self):
        persona_map = {
            0: "Base",
            1: "Assistant",
            2: "Serenity",
            3: "Architect",
            4: "Philosopher",
            5: "Oracle",
            6: "Cecilia",
            7: "Transcendent"
        }
        self.assertIn(2, persona_map)
        self.assertEqual(persona_map[2], "Serenity")

    def test_main_persona_label_resolution(self):
        from unittest.mock import MagicMock
        from main import ChatbotApp
        
        # Test levels 0 through 6 -> "Serenity", level 7 -> "Cecilia"
        dummy = MagicMock(spec=ChatbotApp)
        dummy.active_persona_level = 2
        dummy.is_cecilia_active = False
        self.assertEqual(ChatbotApp._get_persona_label(dummy), "Serenity")

        dummy.active_persona_level = 6
        self.assertEqual(ChatbotApp._get_persona_label(dummy), "Serenity")

        dummy.active_persona_level = 7
        self.assertEqual(ChatbotApp._get_persona_label(dummy), "Cecilia")

        # Cecilia override flag
        dummy.active_persona_level = 3
        dummy.is_cecilia_active = True
        self.assertEqual(ChatbotApp._get_persona_label(dummy), "Cecilia")

    def test_display_user_message_uses_prompt_tag(self):
        import tkinter as tk
        from unittest.mock import MagicMock
        from main import ChatbotApp

        root = tk.Tk()
        root.withdraw()
        try:
            txt = tk.Text(root)
            dummy = MagicMock(spec=ChatbotApp)
            dummy.chat_history = txt
            dummy.state = {}
            dummy.config = {"format_prompts_markdown": False}
            appended = []
            dummy._append_to_chat = lambda text, tag=None: appended.append((text, tag))

            ChatbotApp._display_user_message(dummy, "Hello world")
            self.assertTrue(any("Prompt: " in t[0] for t in appended))
            self.assertFalse(any("You: " in t[0] for t in appended))
        finally:
            root.destroy()

    def test_handle_input_key_multiline(self):
        from unittest.mock import MagicMock
        from main import ChatbotApp

        dummy = MagicMock(spec=ChatbotApp)
        
        # 1. Plain Return sends message and breaks
        evt_enter = MagicMock(keysym='Return', state=0)
        res = ChatbotApp._handle_input_key(dummy, evt_enter)
        self.assertEqual(res, 'break')
        dummy.send_message.assert_called_once()
        dummy.send_message.reset_mock()

        # 2. KP_Enter sends message and breaks
        evt_kp = MagicMock(keysym='KP_Enter', state=0)
        res = ChatbotApp._handle_input_key(dummy, evt_kp)
        self.assertEqual(res, 'break')
        dummy.send_message.assert_called_once()
        dummy.send_message.reset_mock()

        # 3. Shift+Return allows newline without sending
        evt_shift_enter = MagicMock(keysym='Return', state=0x0001)
        res = ChatbotApp._handle_input_key(dummy, evt_shift_enter)
        self.assertIsNone(res)
        dummy.send_message.assert_not_called()


if __name__ == "__main__":
    unittest.main()
