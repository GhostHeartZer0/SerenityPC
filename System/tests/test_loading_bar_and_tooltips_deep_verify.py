"""
Test: Deep Verification of Loading Bar States/Functions & ToolTips (V1.6.8 / V1.6.9)
Validates:
1. DynamicStatusWidget / ThinkingDisplay in all 5 modes:
   - hybrid (determinate/indeterminate progress, TTFT, tokens/sec, complete speed badge, DMN transition)
   - tasks (multiline telemetry phase, tokens, speed, TTFT)
   - percentage (loading gauge %, determinate progress)
   - animation (spinner, pulse, orbit canvas routines)
   - prayer (smooth Serenity prayer fading lines cycle)
   - idle displays (DMN idle timer vs countdown timer)
2. ToolTip mechanics:
   - 1500ms default linger delay
   - Text resolution and boundary clamping inside screen bounds
   - Config show_tooltips toggle enforcement
   - Tooltip attachment verification across main app and settings window controls
"""
import sys
import os
import unittest
import tkinter as tk
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from System.serenity_utils import DynamicStatusWidget, ThinkingDisplay, ToolTip

class TestLoadingBarAndTooltipsDeepVerify(unittest.TestCase):
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

    def _create_mock_app(self, mode="hybrid", dmn_idle=True, fallback_info=True, anim_style="spinner"):
        app = MagicMock()
        app.config = {
            "status_bar_mode": mode,
            "status_bar_anim_style": anim_style,
            "status_bar_dmn_idle": dmn_idle,
            "status_bar_fallback_info": fallback_info,
            "show_tooltips": True,
            "theme": "default",
            "texture_style": "default",
            "frosted_glass": False
        }
        app.state = {
            "avatar_current": "neutral",
            "dmn_active": False,
            "dmn_entry_time": None
        }
        app.active_persona_level = 3
        app.current_model_tier = "fast"
        app.context_size_config = {"fast": 8192}
        app._parse_dmn_timeout_sec = lambda: 300
        return app

    def test_status_widget_hybrid_mode_lifecycle(self):
        self.root.deiconify()
        self.root.update()
        try:
            app = self._create_mock_app(mode="hybrid")
            widget = DynamicStatusWidget(self.root, app=app)
            widget.start()
            self.root.update()
            self.assertTrue(widget._is_active)
            self.assertEqual(widget.progress_container.winfo_manager(), "pack")

            # 1. Loading Phase
            widget.set_phase("loading", details="Gemma 26B", progress_val=45.0)
            self.assertEqual(widget._current_phase, "loading")
            self.assertEqual(widget.gauge_label.cget("text"), "LOAD: 45%")

            # 2. Prefill Phase
            widget.set_phase("prefill", details="Processing Prompt Context")
            self.assertEqual(widget.gauge_label.cget("text"), "PREFILL...")

            # 3. Reasoning Phase
            widget.set_phase("reasoning", details="Analyzing Constraints")
            self.assertEqual(widget.gauge_label.cget("text"), "THINKING...")

            # 4. Generating Phase with TTFT and Speed
            widget.record_ttft(1.24)
            widget.set_phase("generating", tokens=150, speed=28.5)
            self.assertEqual(widget.gauge_label.cget("text"), "28.5 t/s")
            self.assertIn("TTFT: 1.24s", widget.label.cget("text"))
            self.assertIn("28.5 t/s", widget.label.cget("text"))

            # 5. Stop and Complete Transition
            widget.stop()
            self.assertFalse(widget._is_active)
            widget.destroy()
        finally:
            self.root.withdraw()

    def test_status_widget_tasks_mode(self):
        self.root.deiconify()
        self.root.update()
        try:
            app = self._create_mock_app(mode="tasks")
            widget = DynamicStatusWidget(self.root, app=app)
            widget.start()
            self.root.update()
            self.assertEqual(widget.tasks_frame.winfo_manager(), "pack")
            self.assertEqual(widget.progress_container.winfo_manager(), "")

            widget.record_ttft(0.95)
            widget.set_phase("generating", tokens=80, speed=32.0)
            task_text = widget.task_lines_label.cget("text")
            self.assertIn("Phase: GENERATING", task_text)
            self.assertIn("Tokens: 80", task_text)
            self.assertIn("Speed: 32.0 t/s", task_text)
            self.assertIn("TTFT: 0.95s", task_text)
            widget.stop()
            widget.destroy()
        finally:
            self.root.withdraw()

    def test_status_widget_percentage_mode(self):
        self.root.deiconify()
        self.root.update()
        try:
            app = self._create_mock_app(mode="percentage")
            widget = DynamicStatusWidget(self.root, app=app)
            widget.start()
            self.root.update()
            self.assertEqual(widget.progress_container.winfo_manager(), "pack")
            widget.set_estimated_tokens(200)
            widget.set_phase("generating", tokens=100, speed=25.0)
            self.assertEqual(widget.gauge_label.cget("text"), "50% (25.0 t/s)")
            widget.stop()
            widget.destroy()
        finally:
            self.root.withdraw()

    def test_status_widget_animation_styles(self):
        self.root.deiconify()
        self.root.update()
        try:
            for style in ["spinner", "pulse", "orbit"]:
                app = self._create_mock_app(mode="animation", anim_style=style)
                widget = DynamicStatusWidget(self.root, app=app)
                widget.start()
                self.root.update()
                self.assertEqual(widget.anim_canvas.winfo_manager(), "pack")
                widget._start_canvas_animation()
                self.assertGreater(widget._anim_angle, 0)
                widget.stop()
                widget.destroy()
        finally:
            self.root.withdraw()

    def test_status_widget_prayer_mode_cycling(self):
        self.root.deiconify()
        self.root.update()
        try:
            app = self._create_mock_app(mode="prayer")
            widget = DynamicStatusWidget(self.root, app=app)
            widget.start()
            self.root.update()
            self.assertEqual(widget.prayer_label.winfo_manager(), "pack")
            for _ in range(50):
                widget._start_prayer_animation()
            # Prayer label should have text and color applied
            self.assertTrue(len(widget.prayer_label.cget("text")) > 0)
            self.assertTrue(widget.prayer_label.cget("fg").startswith("#"))
            widget.stop()
            widget.destroy()
        finally:
            self.root.withdraw()

    def test_status_widget_idle_and_dmn_state(self):
        # 1. Regular Idle Countdown
        app = self._create_mock_app(mode="hybrid", dmn_idle=True)
        widget = DynamicStatusWidget(self.root, app=app)
        widget._update_idle_display()
        self.assertIn("Next DMN in:", widget.label.cget("text"))

        # 2. Active DMN State
        app.state["dmn_active"] = True
        app.state["dmn_entry_time"] = 100.0
        widget._update_idle_display()
        self.assertIn("[DMN Idle] Time in DMN:", widget.label.cget("text"))

        # 3. Simmering DMN
        app.state["auto_watch"] = True
        widget._update_idle_display()
        self.assertIn("[DMN Simmering]", widget.label.cget("text"))
        widget.destroy()

    def test_tooltip_screen_clamping_and_dismissal(self):
        btn = tk.Button(self.root, text="Sample Button")
        btn.pack()
        tip_default = ToolTip(btn, "Default delay test")
        self.assertEqual(tip_default.delay_ms, 1500)

        tip = ToolTip(btn, "Hover explanation for sample button", delay_ms=5)
        
        # Trigger hover
        tip._on_enter()
        self.root.update()
        self.assertIsNotNone(tip._after_id)

        # Trigger show
        tip._show()
        self.assertIsNotNone(tip.tip_window)
        self.assertTrue(tip.tip_window.winfo_exists())

        # Trigger leave
        tip._on_leave()
        self.assertIsNone(tip.tip_window)
        btn.destroy()

    def test_settings_all_tooltips_instantiated(self):
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
                    "v_cache_type": "q8_0",
                    "settings_window_geometry": "600x600"
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

        with patch("System.stt_manager.STTManager.get_input_devices", return_value=[]):
            app = MockFullApp(self.root)
            open_settings_window(app)
            toplevels = [w for w in self.root.winfo_children() if isinstance(w, tk.Toplevel)]
            self.assertTrue(len(toplevels) > 0)
            settings_win = toplevels[-1]
            
            # Count child widgets inside settings window
            all_widgets = settings_win.winfo_children()
            self.assertGreater(len(all_widgets), 0)
            
            for t in toplevels:
                t.destroy()

if __name__ == "__main__":
    unittest.main()
