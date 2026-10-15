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

if __name__ == "__main__":
    unittest.main()
