import unittest
import tkinter as tk
from benchmarks.debate.Debate import DebateCore, DebateEngine, DebateApp, CTX_PRESETS, KV_QUANT_OPTIONS

class TestDebateArena(unittest.TestCase):
    def test_debate_core_and_engine_alias(self):
        core = DebateCore()
        self.assertIsInstance(core, DebateCore)
        engine = DebateEngine()
        self.assertIsInstance(engine, DebateCore)
        self.assertIs(DebateEngine, DebateCore)

    def test_debate_app_init_clean(self):
        root = tk.Tk()
        root.withdraw()
        old_mainloop = tk.Tk.mainloop
        tk.Tk.mainloop = lambda self: None
        try:
            app = DebateApp()
            self.assertIsInstance(app.engine, DebateCore)
            self.assertEqual(len(app.contestants), 0)
            app.root.destroy()
        finally:
            tk.Tk.mainloop = old_mainloop
            try:
                root.destroy()
            except Exception:
                pass

if __name__ == "__main__":
    unittest.main()
