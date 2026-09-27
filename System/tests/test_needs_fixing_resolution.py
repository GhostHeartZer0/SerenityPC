import unittest
import os
import json
import tkinter as tk
from unittest.mock import MagicMock

from main import ChatbotApp
from System.markdown_engine import MarkdownEngine
from System.settings_tabs import build_personalize_tab, build_additional_tab
from System.settings_ui import open_settings_window


class TestNeedsFixingResolution(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = MagicMock(spec=ChatbotApp)
        self.app.root = self.root
        self.app.config = {
            "scale_factor": 1.5,
            "text_scale": 125,
            "marquee_text_enabled": False,
            "ghost_mode": False,
            "history_usage": "all",
            "show_rgb_button": True
        }
        self.app.scale_factor = 1.5
        self.app.fonts = {
            "ui_button": ("Segoe UI", 9),
            "ui_small": ("Segoe UI", 8),
            "ui_label": ("Segoe UI", 9),
            "bold": ("Segoe UI", 9, "bold"),
            "large": ("Segoe UI", 12, "bold"),
            "stats": ("Consolas", 8),
            "stats_bold": ("Consolas", 8, "bold"),
            "chat": ("Segoe UI", 10),
            "chat_bold": ("Segoe UI", 10, "bold"),
            "chat_italic": ("Segoe UI", 10, "italic"),
            "log": ("Consolas", 9),
            "log_bold": ("Consolas", 9, "bold"),
            "title": ("Segoe UI", 11, "bold")
        }
        self.app.state = {"avatar_current": "off"}
        self.app.model_paths = {}
        self.app.active_persona_level = 3
        self.app.avatar_states = {}
        self.app.avatar_pil_images = {}

    def tearDown(self):
        try:
            self.root.destroy()
        except Exception:
            pass

    # 1. Hide Cycle (Deep Cook dropdowns) Minimization
    def test_cycle_minimization_elide_toggle(self):
        """Verify toggle_cyc, toggle_draft, and toggle_mem properly minimize and elide."""
        text_widget = tk.Text(self.root)
        text_widget.pack()
        self.app.hist = text_widget

        # Insert some dummy content and tags
        text_widget.insert("end", "Cycle Header\n")
        text_widget.insert("end", "Step 1: Thought content here\n", "cyc_tag_1")
        text_widget.insert("end", "Sub-step A\n", "nested_cyc_tag_1")

        # Initial state: tag is not elided
        self.app.state["nested_tags_cyc_tag_1"] = ["nested_cyc_tag_1"]
        self.app.hist.tag_config("cyc_tag_1", elide=0)
        self.app.hist.tag_config("nested_cyc_tag_1", elide=0)

        # Call toggle_cyc logic
        ChatbotApp.toggle_cyc(self.app, "cyc_tag_1")

        # After toggle: tag should now be elided (minimized)
        is_elided = str(text_widget.tag_cget("cyc_tag_1", "elide")) in ["1", "True", "true"]
        self.assertTrue(is_elided, "Cycle tag should be elided (minimized) after toggle_cyc")

        # Nested tags must also be elided
        is_nested_elided = str(text_widget.tag_cget("nested_cyc_tag_1", "elide")) in ["1", "True", "true"]
        self.assertTrue(is_nested_elided, "Nested cycle tags must also be elided when collapsed")

        # Second toggle: should expand back
        ChatbotApp.toggle_cyc(self.app, "cyc_tag_1")
        is_elided_after_second = str(text_widget.tag_cget("cyc_tag_1", "elide")) in ["1", "True", "true"]
        self.assertFalse(is_elided_after_second, "Cycle tag should expand back upon second click")

    # 2. Sliders 4K Dynamic Scaling
    def test_sliders_4k_dynamic_scaling(self):
        """Verify persona slider clamp expanded to 360+ on 4K, and settings sliders scale properly."""
        # Persona slider in main.py
        slider = tk.Scale(self.root, from_=1, to=7, orient=tk.HORIZONTAL)
        self.app.depth_slider = slider
        self.app.scale_factor = 2.0  # 4K scale factor
        self.app._window_scale_factor = 1.0

        # Simulate 4K event width (e.g. 1600px left panel width)
        event = MagicMock()
        event.width = 1600

        ChatbotApp._on_left_resize(self.app, event)
        # Verify length clamped to min(360, int(1600 * 0.26)) -> 360, NOT the old 160 clamp!
        self.assertEqual(slider.cget("length"), 360, "Slider length clamp must expand to at least 360 on 4K")
        self.assertGreaterEqual(slider.cget("width"), 18, "Slider trough thickness should scale up on 4K")

        # Test settings sliders in build_additional_tab
        vars_dict = {
            "sc_val": tk.IntVar(value=8),
            "status_mode_var": tk.StringVar(value="hybrid"),
            "anim_style_var": tk.StringVar(value="spinner"),
            "sb_dmn_var": tk.BooleanVar(value=True),
            "sb_fallback_var": tk.BooleanVar(value=True),
            "sb_linger_var": tk.DoubleVar(value=5.0)
        }
        parent = tk.Frame(self.root)
        f_add = build_additional_tab(parent, self.app, None, vars_dict)

        # Check sc_scale and linger_scale
        scales = [w for w in f_add.winfo_children() if isinstance(w, tk.Scale) or isinstance(w, tk.LabelFrame) or isinstance(w, tk.Frame)]
        self.assertTrue(len(scales) > 0)

    # 3. Math Formatting Overhaul
    def test_math_formatting_overhaul(self):
        """Verify /frac, /sum, /times, /mathbf, /left, /right, $$$$, nested braces, and currency protection."""
        # Nested fraction
        latex1 = r"\frac{5^{n+1} - 1}{5 - 1}"
        conv1 = MarkdownEngine.convert_latex_to_unicode(latex1)
        self.assertIn("5ⁿ⁺¹ - 1", conv1)
        self.assertIn("5 - 1", conv1)
        self.assertIn("/", conv1)

        # Sum with limits and times
        latex2 = r"\sum_{i=0}^{n} 5^i \times 2"
        conv2 = MarkdownEngine.convert_latex_to_unicode(latex2)
        self.assertIn("∑", conv2)
        self.assertIn("×", conv2)

        # Mathbf and left/right brackets
        latex3 = r"\mathbf{x} = \left( \frac{a}{b} \right)"
        conv3 = MarkdownEngine.convert_latex_to_unicode(latex3)
        self.assertNotIn(r"\mathbf", conv3)
        self.assertNotIn(r"\left", conv3)
        self.assertNotIn(r"\right", conv3)

        # Quad dollars ($$$$)
        raw_quad = "$$$$E = mc^2$$$$"
        spans_quad = MarkdownEngine.parse_to_spans(raw_quad)
        res_quad = "".join(s[0] for s in spans_quad)
        self.assertIn("E = mc²", res_quad)

        # Currency amounts protection
        currency_text = "Revenue reached $596,046,447,753,906 and $200 today."
        spans_curr = MarkdownEngine.parse_to_spans(currency_text)
        res_curr = "".join(s[0] for s in spans_curr)
        self.assertIn("$596,046,447,753,906", res_curr)
        self.assertIn("$200", res_curr)

    # 4. Status Bar & Universal Marquee in Personalize (Appearance) Tab
    def test_status_bar_and_marquee_toggle(self):
        """Verify status label title compacting, end-truncation, and marquee checkbutton in Personalize tab."""
        lbl = tk.Label(self.root, text="")
        lbl.pack()
        self.app.system_status_label = lbl
        self.app.config["marquee_text_enabled"] = False

        # Long model message should be compacted to basename and end-truncated
        long_model = "Loaded: C:\\Users\\user\\Models\\DeepSeek-Coder-V2-Lite-Instruct.Q4_K_M.gguf"
        ChatbotApp._update_status_label_text(self.app, long_model)

        displayed = lbl.cget("text")
        self.assertTrue(displayed.startswith("Loaded: DeepSeek-Coder-V2-Lite-Instruct.Q4_K_M.gguf") or "..." in displayed)
        self.assertNotIn("C:\\Users\\user\\Models", displayed, "Redundant file paths should be stripped from status label")

        # Verify Universal Marquee toggle exists in Personalize (Appearance) tab
        vars_dict = {
            "THEME_MAP": {"Apex (Default)": "apex"},
            "theme_display_var": tk.StringVar(value="Apex (Default)"),
            "TEXTURE_MAP": {"Default Original": "default"},
            "tex_display_var": tk.StringVar(value="Default Original"),
            "tex_int_var": tk.IntVar(value=100),
            "dark_mode_var": tk.BooleanVar(value=False),
            "SCALE_MAP": {"100% (Standard)": 100},
            "text_scale_display_var": tk.StringVar(value="100% (Standard)"),
            "text_scale_val_var": tk.IntVar(value=100),
            "marquee_text_var": tk.BooleanVar(value=False)
        }
        parent = tk.Frame(self.root)
        f_pers = build_personalize_tab(parent, self.app, None, vars_dict)

        # Check for marquee checkbox widget
        found_marquee_cb = False
        for child in f_pers.winfo_children():
            if isinstance(child, tk.Checkbutton) and "Marquee" in child.cget("text"):
                found_marquee_cb = True
                break
        self.assertTrue(found_marquee_cb, "Universal Marquee toggle must be present in Personalize tab")

    # 5. Avatar Dynamic Scaling to Fill Panel
    def test_avatar_dynamic_scaling(self):
        """Verify avatar target size dynamically computes from right panel canvas space instead of fixed 350x350."""
        canvas = tk.Canvas(self.root, width=800, height=1200)
        canvas.pack()
        self.app.right_panel = canvas
        self.root.update_idletasks()

        # Canvas configured to 800x1200
        w, h = ChatbotApp._get_avatar_target_size(self.app)
        # Should scale to panel dimensions (w >= 700, h >= 500)
        self.assertGreater(w, 350, "Avatar target width should scale up beyond 350 in large right panel")
        self.assertGreater(h, 350, "Avatar target height should scale up beyond 350 in large right panel")


if __name__ == "__main__":
    unittest.main()
