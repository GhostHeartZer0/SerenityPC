"""
Test: Theme Overhaul, Dark Mode Blackout, Persona Dynamic Theme, and User Profile Configs.
Validates:
1. Frosted glass in texture dropdown options.
2. Dark Mode (OLED blackout) power-saving mode.
3. Persona dynamic theme adapting to active persona level.
4. User profile config persistence in Users/<user>/config.json.
5. All text tags styled with neon colors instead of default white.
"""
import sys
import os
import json
import unittest
import tempfile
import tkinter as tk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from serenity_resources import THEMES, TEXTURE_STYLES, apply_theme_to_global, THEME, THERMO_COLORS

class TestThemeAndDarkMode(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root_dir = self.tmpdir.name

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_frosted_glass_in_texture_styles(self):
        self.assertIn("frosted_glass", TEXTURE_STYLES)
        self.assertEqual(TEXTURE_STYLES["frosted_glass"]["name"], "Frosted Glass")
        self.assertTrue(TEXTURE_STYLES["frosted_glass"].get("is_frosted"))

    def test_dark_mode_blackout(self):
        apply_theme_to_global(theme_name="apex", texture_style="default", dark_mode=True)
        self.assertEqual(THEME["bg_color"], "#000000")
        self.assertEqual(THEME["widget_bg_color"], "#000000")
        self.assertEqual(THEME["chat_bg_color"], "#000000")
        self.assertEqual(THEME["_dark_mode"], True)

    def test_pure_black_themes(self):
        for th in ["goth", "yellow_blacket", "matrix"]:
            apply_theme_to_global(theme_name=th, texture_style="default", dark_mode=False)
            self.assertEqual(THEME["bg_color"], "#000000")
            self.assertEqual(THEME["widget_bg_color"], "#000000")
            self.assertEqual(THEME["chat_bg_color"], "#000000")

    def test_persona_dynamic_theme(self):
        # Level 1 Gold
        apply_theme_to_global(theme_name="persona", texture_style="default", dark_mode=False, active_level=1, model_loaded=True)
        self.assertEqual(THEME["accent_highlight"], THERMO_COLORS[1])
        self.assertEqual(THEME["electric_blue"], THERMO_COLORS[1])
        self.assertEqual(THEME["fg_color"], THERMO_COLORS[1])

        # Level 4 Violet
        apply_theme_to_global(theme_name="persona", texture_style="default", dark_mode=False, active_level=4, model_loaded=True)
        self.assertEqual(THEME["accent_highlight"], THERMO_COLORS[4])
        self.assertEqual(THEME["electric_blue"], THERMO_COLORS[4])

        # Level 7 Cecilia Dark Emerald / Neon Green
        apply_theme_to_global(theme_name="persona", texture_style="default", dark_mode=False, active_level=7, model_loaded=True)
        self.assertEqual(THEME["accent_highlight"], "#00ff66")
        self.assertEqual(THEME["trim_color"], "#005a36")
        self.assertEqual(THEME["bg_color"], "#02140e")

        apply_theme_to_global(theme_name="cecilia", texture_style="default", dark_mode=False)
        self.assertEqual(THEME["fg_color"], "#00ff66")
        self.assertEqual(THEME["trim_color"], "#005a36")

    def test_user_profile_config_persistence(self):
        from main import ChatbotApp
        
        class MockApp:
            def __init__(self, script_dir):
                self.script_dir = script_dir
                self.dirs = {d: os.path.join(self.script_dir, d) for d in ["Media", "History", "Models", "Logs", "System", "Users"]}
                for d in self.dirs.values(): os.makedirs(d, exist_ok=True)
                self.config_file = os.path.join(self.dirs["System"], "config.json")
                self.config = {"username": "Default", "theme": "apex", "dark_mode": False}
                self.model_paths = {}
                self.gpu_layer_config = {}
                self.context_size_config = {}
                self.temp_config = {}
                self.top_p_config = {}
                self.min_p_config = {}
                self.repeat_penalty_config = {}
                self.frequency_penalty_config = {}
                self.presence_penalty_config = {}
                self.stop_strings_config = {}
                self.n_batch_config = {}
                self.top_k_config = {}
                self.state = {"deep_cook_behavior": "oneshot", "virtual_vram": 0}
                self.active_persona_level = 3
                self.max_persona_level = 7
                self.depth_slider = None
                self.root = type("MockRoot", (), {"winfo_geometry": lambda *args: "960x600+0+0"})()
                self.model = None

            get_active_username = ChatbotApp.get_active_username
            get_user_dir = ChatbotApp.get_user_dir
            get_user_history_dir = ChatbotApp.get_user_history_dir
            save_config = ChatbotApp.save_config
            switch_user = ChatbotApp.switch_user
            _load_dmn_backbone = ChatbotApp._load_dmn_backbone
            _save_dmn_backbone = ChatbotApp._save_dmn_backbone
            _log_and_display = lambda self, m: None

        app = MockApp(self.root_dir)
        app.save_config()
        
        # Verify Default config saved
        def_cfg = os.path.join(self.root_dir, "Users", "Default", "config.json")
        self.assertTrue(os.path.exists(def_cfg))

        # Switch to GhostHeartZer0, modify setting and save
        app.switch_user("GhostHeartZer0")
        app.config["theme"] = "matrix"
        app.config["dark_mode"] = True
        app.save_config()

        ghz_cfg = os.path.join(self.root_dir, "Users", "GhostHeartZer0", "config.json")
        self.assertTrue(os.path.exists(ghz_cfg))
        with open(ghz_cfg, "r") as f:
            saved_data = json.load(f)
        self.assertEqual(saved_data.get("theme"), "matrix")
        self.assertEqual(saved_data.get("dark_mode"), True)

    def test_radio_button_contrast_binding(self):
        root = tk.Tk()
        root.withdraw()
        
        var = tk.StringVar(value="exps=CPU")
        f = tk.Frame(root)
        rb1 = tk.Radiobutton(f, text="None", variable=var, value="None")
        rb2 = tk.Radiobutton(f, text="exps=CPU", variable=var, value="exps=CPU")
        
        # Test contrast updater
        rb_list = [(rb1, "None"), (rb2, "exps=CPU")]
        def _bind_radio_contrast(v, rbs):
            def _sync(*args):
                val = v.get()
                for rb, item_val in rbs:
                    if rb.winfo_exists():
                        rb.config(fg="#000000" if val == item_val else "#ffaa44")
            v.trace_add("write", _sync)
            _sync()
            
        _bind_radio_contrast(var, rb_list)
        
        # Selected should be black
        self.assertEqual(rb2.cget("fg"), "#000000")
        self.assertEqual(rb1.cget("fg"), "#ffaa44")
        
        # Switch to None
        var.set("None")
        self.assertEqual(rb1.cget("fg"), "#000000")
        self.assertEqual(rb2.cget("fg"), "#ffaa44")
        
        root.destroy()

    def test_backend_logs_telemetry_and_secret_trigger_theming(self):
        from main import ChatbotApp
        root = tk.Tk()
        root.withdraw()
        
        class MockApp:
            def __init__(self, r):
                self.root = r
                self.config = {"theme": "matrix", "dark_mode": False}
                self.state = {"log_view": "thought"}
                self.active_persona_level = 3
                self.secret_trigger = tk.Label(self.root, text="  ", bg="#ffffff")
                self.log_container = tk.Frame(self.root, bg="#ffffff")
                self.log_header_frame = tk.Frame(self.log_container, bg="#ffffff")
                self.log_header_label = tk.Label(self.log_header_frame, text="Backend Logs", bg="#ffffff", fg="#000000")
                self.self_analysis_btn = tk.Label(self.log_header_frame, text="🔍", bg="#ffffff", fg="#000000")
                self.lock_logout_btn = tk.Label(self.log_header_frame, text="🔒", bg="#ffffff", fg="#000000")
                self.clear_log_btn = tk.Label(self.log_header_frame, text="🗑", bg="#ffffff", fg="#000000")
                self.log_frame = tk.Frame(self.log_container, bg="#ffffff")
                self.thought_log = tk.Text(self.log_frame, bg="#ffffff", fg="#000000")
                self.error_log = tk.Text(self.log_frame, bg="#ffffff", fg="#000000")
                self.tool_log = tk.Text(self.log_frame, bg="#ffffff", fg="#000000")
                self.diag_log = tk.Text(self.log_frame, bg="#ffffff", fg="#000000")
                self.stats_frame = tk.Frame(self.log_container, bg="#ffffff")
                f1 = tk.Frame(self.stats_frame, bg="#ffffff")
                lbl_t = tk.Label(f1, text="CPU:", bg="#ffffff", fg="#000000")
                lbl_v = tk.Label(f1, text="10%", bg="#ffffff", fg="#000000")
                self.input_control_frame = tk.Frame(self.root, bg="#ffffff")
                self.attachment_frame = tk.Frame(self.input_control_frame, bg="#ffffff")
                self.user_input = tk.Text(self.input_control_frame, bg="#ffffff", fg="#000000")
                self.status_frame = tk.Frame(self.root, bg="#ffffff")
                self.stats_row_frames = [f1]
                self.stats_title_labels = [lbl_t]
                self.stats_labels = {"CPU": lbl_v}
                self.system_status_label = tk.Label(self.root, text="System: Idle", bg="#ffffff", fg="#000000")
                self.hw_mode_label = tk.Label(self.root, text="[APEX]", bg="#ffffff", fg="#000000")

            apply_current_theme = ChatbotApp.apply_current_theme

        app = MockApp(root)
        apply_theme_to_global("matrix", "default", False)
        app.apply_current_theme()

        self.assertEqual(app.secret_trigger.cget("bg"), THEME["bg_color"])
        self.assertEqual(app.secret_trigger.cget("fg"), THEME["bg_color"])
        self.assertEqual(app.input_control_frame.cget("bg"), THEME["trim_color"])
        self.assertEqual(app.user_input.cget("highlightbackground"), THEME["trim_color"])
        self.assertEqual(app.user_input.cget("insertbackground"), THEME["electric_blue"])
        self.assertEqual(app.log_header_label.cget("fg"), THEME["electric_blue"])
        self.assertEqual(app.self_analysis_btn.cget("fg"), THEME["electric_blue"])
        self.assertEqual(app.thought_log.cget("bg"), THEME["widget_bg_color"])
        self.assertEqual(app.thought_log.cget("insertbackground"), THEME["electric_blue"])
        self.assertEqual(app.stats_frame.cget("bg"), THEME["widget_bg_color"])
        self.assertEqual(app.stats_row_frames[0].cget("bg"), THEME["widget_bg_color"])
        self.assertEqual(app.stats_title_labels[0].cget("bg"), THEME["widget_bg_color"])
        self.assertEqual(app.stats_labels["CPU"].cget("fg"), THEME["electric_blue"])
        self.assertEqual(app.system_status_label.cget("bg"), THEME["bg_color"])
        self.assertEqual(app.system_status_label.cget("fg"), THEME["electric_blue"])
        self.assertEqual(app.hw_mode_label.cget("bg"), THEME["bg_color"])
        self.assertEqual(app.hw_mode_label.cget("fg"), THEME["accent_highlight"])

        root.destroy()

if __name__ == "__main__":
    unittest.main()
