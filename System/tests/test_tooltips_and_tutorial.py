"""
Test: ToolTips & Tutorial Walkthrough Overlay (V1.6.7)
Validates:
1. ToolTip behavior: timer scheduling, text resolution, app config toggle enforcement.
2. TutorialOverlay behavior: step-by-step navigation across all 9 areas, progress calculation, skip and completion persistence.
"""
import sys
import os
import unittest
import tkinter as tk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from System.serenity_utils import ToolTip, TutorialOverlay, TUTORIAL_SCREENS

class TestToolTipsAndTutorial(unittest.TestCase):
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

    def test_tooltip_config_toggle_and_resolution(self):
        class MockApp:
            def __init__(self, show_tooltips=True):
                self.config = {"show_tooltips": show_tooltips}

        btn = tk.Button(self.root, text="Test Button")
        
        # 0. Default delay is 1500ms
        tip_def = ToolTip(btn, "Default Delay")
        self.assertEqual(tip_def.delay_ms, 1500)

        # 1. Static text tooltip with enabled config
        app_enabled = MockApp(show_tooltips=True)
        tip1 = ToolTip(btn, "Test description", delay_ms=10, app=app_enabled)
        self.assertEqual(tip1.text_or_callable, "Test description")
        tip1._schedule()
        self.assertIsNotNone(tip1._after_id)
        tip1._unschedule()
        self.assertIsNone(tip1._after_id)

        # 2. Dynamic callable tooltip
        dyn_call = lambda: "Dynamic text 123"
        tip2 = ToolTip(btn, dyn_call, delay_ms=10, app=app_enabled)
        text_val = tip2.text_or_callable() if callable(tip2.text_or_callable) else tip2.text_or_callable
        self.assertEqual(text_val, "Dynamic text 123")

        # 3. Disabled config suppresses scheduling
        app_disabled = MockApp(show_tooltips=False)
        tip3 = ToolTip(btn, "Disabled tooltip", delay_ms=10, app=app_disabled)
        tip3._schedule()
        self.assertIsNone(tip3._after_id)

    def test_tutorial_screens_structure(self):
        # Must cover all 9 areas
        self.assertEqual(len(TUTORIAL_SCREENS), 9)
        required_keys = ["area", "badge", "title", "subtitle", "desc", "hint", "target_key"]
        for idx, screen in enumerate(TUTORIAL_SCREENS):
            for k in required_keys:
                self.assertIn(k, screen, f"Screen {idx} missing key {k}")
            self.assertTrue(len(screen["title"]) > 0)
            self.assertTrue(len(screen["desc"]) > 0)
        
        # Step 2 targets top bar
        self.assertEqual(TUTORIAL_SCREENS[1]["target_key"], "top_bar")
        # Step 8 is complete settings walkthrough
        self.assertIn("Settings Walkthrough", TUTORIAL_SCREENS[7]["title"])

    def test_tutorial_overlay_navigation_and_completion(self):
        class MockAppWithRoot:
            def __init__(self, root):
                self.root = root
                self.config = {"tutorial_completed": False}
                self.saved = False
                self.top_bar_frame = tk.Frame(root)
                self.load_model_button = tk.Button(root, text="Settings")
                self.user_input = tk.Text(root)
                self.mic_button = tk.Button(root, text="Mic")
                self.depth_slider = tk.Scale(root)
                self.top_bar_frame.pack()
                self.load_model_button.pack()
                self.user_input.pack()
                self.mic_button.pack()
                self.depth_slider.pack()

            def save_config(self):
                self.saved = True

        app = MockAppWithRoot(self.root)
        finished_flag = []
        overlay = TutorialOverlay(app, on_finish=lambda: finished_flag.append(True))

        self.assertEqual(overlay.current_step, 0)
        self.assertEqual(overlay.total_steps, 9)

        # Target widgets detection
        top_bar_widgets = overlay._get_target_widgets("top_bar")
        self.assertIn(app.top_bar_frame, top_bar_widgets)
        self.assertIn(app.load_model_button, top_bar_widgets)

        console_widgets = overlay._get_target_widgets("console")
        self.assertIn(app.user_input, console_widgets)
        self.assertIn(app.mic_button, console_widgets)

        persona_widgets = overlay._get_target_widgets("persona")
        self.assertIn(app.depth_slider, persona_widgets)

        settings_widgets = overlay._get_target_widgets("settings")
        self.assertIn(app.load_model_button, settings_widgets)

        # Next step
        overlay.next_step()
        self.assertEqual(overlay.current_step, 1)

        # Prev step
        overlay.prev_step()
        self.assertEqual(overlay.current_step, 0)

        # Skip immediately completes and persists
        overlay.skip()
        self.assertTrue(app.config.get("tutorial_completed"))
        self.assertTrue(app.saved)
        self.assertTrue(len(finished_flag) > 0)
        self.assertIsNone(overlay.win)

    def test_tutorial_buttons_visibility_all_steps(self):
        class MockAppWithRoot:
            def __init__(self, root):
                self.root = root
                self.config = {"tutorial_completed": False}
                self.saved = False
                self.user_input = tk.Text(root)
                self.user_input.pack()

            def save_config(self):
                self.saved = True

        self.root.deiconify()
        self.root.geometry("1179x796+100+100")
        self.root.update()
        try:
            app = MockAppWithRoot(self.root)
            overlay = TutorialOverlay(app)
            self.root.update()

            # Step through every single screen and verify bottom button bar is mapped and in bounds
            for step in range(len(TUTORIAL_SCREENS)):
                overlay.current_step = step
                overlay._render_step()
                self.root.update()
                self.assertTrue(overlay.btn_next.winfo_ismapped(), f"Step {step} Next button is not mapped/visible")
                self.assertTrue(overlay.btn_skip.winfo_ismapped(), f"Step {step} Skip button is not mapped/visible")
                self.assertTrue(overlay.btn_back.winfo_ismapped(), f"Step {step} Back button is not mapped/visible")
                self.assertGreater(overlay.btn_next.winfo_rooty(), 0)

            overlay.skip()
        finally:
            self.root.withdraw()

    def test_settings_window_renders_without_name_errors(self):
        from System.settings_ui import open_settings_window
        class MockFullApp:
            def __init__(self, root):
                self.root = root
                self.icon_path = None
                self.script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                self.config = {
                    "show_tooltips": True,
                    "stt_language": "en-US",
                    "stt_device_index": None,
                    "theme": "default",
                    "texture_style": "default",
                    "frosted_glass": False,
                    "k_cache_type": "q8_0",
                    "v_cache_type": "q8_0"
                }
                self.state = {"deep_cook_behavior": "oneshot", "streaming_mode": "Buffered"}
                self.model_paths = {}
                self.gpu_layer_config = {}
                self.context_size_config = {}
                self.n_batch_config = {}
                self.temp_config = {}
                self.top_p_config = {}
                self.min_p_config = {}
                self.top_k_config = {}
                self.repeat_penalty_config = {}
                self.frequency_penalty_config = {}
                self.presence_penalty_config = {}
                self.stop_strings_config = {}
                self.list_user_profiles = lambda: ["Default"]
                self.get_active_username = lambda: "Default"
                self._is_rgb_supported = lambda: False
                self.clear_current_history = lambda: None
                self.save_config = lambda: None

        from unittest.mock import patch
        with patch("System.stt_manager.STTManager.get_input_devices", return_value=[]):
            app = MockFullApp(self.root)
            open_settings_window(app)
        # Verify settings toplevel window created and exists
        toplevels = [w for w in self.root.winfo_children() if isinstance(w, tk.Toplevel)]
        self.assertTrue(len(toplevels) > 0)
        for t in toplevels:
            t.destroy()

if __name__ == "__main__":
    unittest.main()
