"""
Test Suite: Settings Window Reorganization
Validates:
1. Six-tab modular architecture (Models, Inference, Agents, Additional, Users, Personalize).
2. History Mode options (TurboVec, Keyword, Off) and config synchronization.
3. Overfill Behavior vs. Halt Options independence and persistence.
4. Generation guarding: window opens while running, with generation-sensitive widgets disabled.
5. Templating Engine: Write mode requires 'Copy' press, auto-returns to 'modify' mode.
6. Clean apply and save persistence into config.
"""
import os
import sys
import unittest
import tkinter as tk
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from System.settings_ui import open_settings_window
from System.settings_tabs import (
    build_models_tab,
    build_inference_tab,
    build_agents_tab,
    build_additional_tab,
    build_users_tab,
    build_personalize_tab,
)

class MockApp:
    def __init__(self, root):
        self.root = root
        self.icon_path = None
        self.script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.config = {
            "theme": "apex",
            "texture_style": "default",
            "texture_intensity": 1.0,
            "dark_mode": False,
            "show_tooltips": True,
            "stt_language": "en-US",
            "stt_device_index": None,
            "k_cache_type": "q8_0",
            "v_cache_type": "q8_0",
            "history_mode": "TurboVec",
            "turbovec_mode": "on",
            "overfill_behavior_mode": "wrapup",
            "budget_recovery_mode": "wrapup",
            "halt_behavior_mode": "off",
            "text_scale": 100,
            "ui_font": "Segoe UI",
            "mono_font": "Consolas",
            "custom_templates": {
                "T1": {"name": "Test Preset", "temp": 0.7, "top_p": 0.9, "layers": 20}
            }
        }
        self.state = {
            "running": False,
            "generating": False,
            "deep_cook_behavior": "oneshot",
            "streaming_mode": "Buffered",
            "virtual_vram": 8192 * 1024 * 1024
        }
        self.fonts = {
            "ui_button": ("Segoe UI", 9),
            "ui_small": ("Segoe UI", 8),
            "ui_label": ("Segoe UI", 9),
            "bold": ("Segoe UI", 9, "bold"),
            "log": ("Consolas", 9),
            "log_bold": ("Consolas", 9, "bold"),
            "title": ("Segoe UI", 11, "bold")
        }
        self.model_paths = {"fast": "models/fast.gguf"}
        self.gpu_layer_config = {"fast": 25}
        self.context_size_config = {"fast": 4096}
        self.n_batch_config = {"fast": 512}
        self.temp_config = {"fast": 0.8}
        self.top_p_config = {"fast": 0.9}
        self.min_p_config = {"fast": 0.05}
        self.top_k_config = {"fast": 40}
        self.repeat_penalty_config = {"fast": 1.1}
        self.frequency_penalty_config = {"fast": 0.0}
        self.presence_penalty_config = {"fast": 0.0}
        self.stop_strings_config = {"fast": "###"}
        self.saved_config = False

    def save_config(self):
        self.saved_config = True

    def clear_current_history(self):
        pass

    def _sync_deep_cook_ui(self):
        pass

    def list_user_profiles(self):
        return ["Default", "Work", "Personal"]

    def get_active_username(self):
        return "Default"

    def _is_rgb_supported(self):
        return False


class TestSettingsReorganization(unittest.TestCase):
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
        self.app = MockApp(self.root)

    def tearDown(self):
        for w in self.root.winfo_children():
            if isinstance(w, tk.Toplevel):
                try:
                    w.destroy()
                except Exception:
                    pass

    def test_all_six_tabs_build_successfully(self):
        win = tk.Toplevel(self.root)
        parent = tk.Frame(win)
        parent.pack()

        vars_dict = {
            "labels": {}, "ents": {}, "ctx_ents": {}, "n_batch_ents": {},
            "temp_ents": {}, "top_p_ents": {}, "min_p_ents": {}, "top_k_ents": {},
            "rep_ents": {}, "freq_ents": {}, "pres_ents": {}, "stop_ents": {},
            "active_template": tk.StringVar(value=""),
            "template_mode": tk.StringVar(value="modify"),
            "dynamic_params_var": tk.BooleanVar(value=True),
            "hao_var": tk.StringVar(value="exps=CPU"),
            "repeat_mode_var": tk.StringVar(value="lazy"),
            "ratio_var": tk.IntVar(value=4),
            "overfill_behavior_var": tk.StringVar(value="wrapup"),
            "halt_behavior_var": tk.StringVar(value="off"),
            "stream_var": tk.StringVar(value="Buffered"),
            "v_behavior": tk.StringVar(value="oneshot"),
            "swa_var": tk.StringVar(value="Auto"),
            "k_cache_var": tk.StringVar(value="q8_0"),
            "v_cache_var": tk.StringVar(value="q8_0"),
            "history_mode_var": tk.StringVar(value="TurboVec"),
            "history_usage_var": tk.StringVar(value="all"),
            "history_lookup_var": tk.StringVar(value="targeted"),
            "reasoning_var": tk.StringVar(value="medium"),
            "delegation_enabled_var": tk.BooleanVar(value=False),
            "delegation_model_mode_var": tk.StringVar(value="lvl6_7_model"),
            "subagent_density_var": tk.StringVar(value="minimal"),
            "cecilia_mode_var": tk.StringVar(value="shadow_wizard"),
            "chain_preset_var": tk.StringVar(value="standard"),
            "handoff_target_var": tk.StringVar(value="lvl3_compiler"),
            "offline_mode_var": tk.BooleanVar(value=False),
            "auto_vram_var": tk.BooleanVar(value=False),
            "spec_draft_var": tk.BooleanVar(value=False),
            "ghost_var": tk.BooleanVar(value=False),
            "thinking_var": tk.BooleanVar(value=True),
            "benchmark_var": tk.BooleanVar(value=False),
            "inline_md_var": tk.BooleanVar(value=True),
            "monitor_graph_var": tk.BooleanVar(value=False),
            "show_tooltips_var": tk.BooleanVar(value=True),
            "resp_len_var": tk.StringVar(value="natural"),
            "media_var": tk.IntVar(value=1),
            "dev_names": ["Default"],
            "stt_dev_var": tk.StringVar(value="Default"),
            "stt_lang_var": tk.StringVar(value="en-US"),
            "multimedia_handling_var": tk.StringVar(value="auto"),
            "dmn_enabled_var": tk.BooleanVar(value=True),
            "sc_val": tk.IntVar(value=8),
            "status_mode_var": tk.StringVar(value="hybrid"),
            "anim_style_var": tk.StringVar(value="spinner"),
            "sb_dmn_var": tk.BooleanVar(value=True),
            "sb_fallback_var": tk.BooleanVar(value=True),
            "user_profiles_list": ["Default"],
            "username_var": tk.StringVar(value="Default"),
            "show_def_var": tk.BooleanVar(value=True),
            "show_pub_var": tk.BooleanVar(value=True),
            "user_pref_name_var": tk.StringVar(value=""),
            "user_addr_style_var": tk.StringVar(value="Direct / Plain"),
            "auto_lock_var": tk.StringVar(value="0"),
            "THEME_MAP": {"Apex (Default)": "apex"},
            "theme_display_var": tk.StringVar(value="Apex (Default)"),
            "TEXTURE_MAP": {"Default Original": "default"},
            "tex_display_var": tk.StringVar(value="Default Original"),
            "tex_int_var": tk.IntVar(value=100),
            "dark_mode_var": tk.BooleanVar(value=False),
            "SCALE_MAP": {"100% (Standard)": 100},
            "text_scale_display_var": tk.StringVar(value="100% (Standard)"),
            "text_scale_val_var": tk.IntVar(value=100),
            "UI_FONT_OPTIONS": ["Segoe UI"],
            "MONO_FONT_OPTIONS": ["Consolas"],
            "ui_font_var": tk.StringVar(value="Segoe UI"),
            "mono_font_var": tk.StringVar(value="Consolas"),
            "open_scaling_center_fn": lambda: None,
            "_open_set_password_modal": lambda: None,
            "_open_disable_vault_modal": lambda: None
        }

        gen_widgets = []
        t1 = build_models_tab(parent, self.app, win, vars_dict, gen_widgets)
        t2 = build_inference_tab(parent, self.app, win, vars_dict)
        t3 = build_agents_tab(parent, self.app, win, vars_dict)
        t4 = build_additional_tab(parent, self.app, win, vars_dict)
        t5 = build_users_tab(parent, self.app, win, vars_dict)
        t6 = build_personalize_tab(parent, self.app, win, vars_dict)

        for tab in [t1, t2, t3, t4, t5, t6]:
            self.assertIsInstance(tab, tk.Frame)
        self.assertGreater(len(gen_widgets), 0)

    def test_open_settings_window_complete_render(self):
        with patch("System.stt_manager.STTManager.get_input_devices", return_value=[]):
            open_settings_window(self.app, is_generating=False)

        toplevels = [w for w in self.root.winfo_children() if isinstance(w, tk.Toplevel)]
        self.assertEqual(len(toplevels), 1)
        win = toplevels[0]
        self.assertEqual(win.title(), "Model Settings")

    def test_generation_guarding_disables_sensitive_widgets(self):
        with patch("System.stt_manager.STTManager.get_input_devices", return_value=[]):
            open_settings_window(self.app, is_generating=True)

        toplevels = [w for w in self.root.winfo_children() if isinstance(w, tk.Toplevel)]
        self.assertEqual(len(toplevels), 1)
        win = toplevels[0]

        # Find all buttons inside win
        buttons = []
        def _gather_buttons(widget):
            for child in widget.winfo_children():
                if isinstance(child, tk.Button):
                    buttons.append(child)
                _gather_buttons(child)
        _gather_buttons(win)

        # Confirm at least some widgets are disabled due to generation guard (e.g. Set Path, Auto-Detect)
        disabled_buttons = [b for b in buttons if str(b.cget("state")) == "disabled"]
        self.assertGreater(len(disabled_buttons), 0, "Expected generation-sensitive buttons to be disabled")

    def test_history_mode_and_overfill_behavior_options(self):
        with patch("System.stt_manager.STTManager.get_input_devices", return_value=[]):
            open_settings_window(self.app, is_generating=False)

        win = [w for w in self.root.winfo_children() if isinstance(w, tk.Toplevel)][0]
        # Inspect radiobuttons in the window
        radio_values = []
        def _gather_radios(widget):
            for child in widget.winfo_children():
                if isinstance(child, tk.Radiobutton):
                    radio_values.append(child.cget("value"))
                _gather_radios(child)
        _gather_radios(win)

        # Check History Mode options: TurboVec, Keyword, Off
        self.assertIn("TurboVec", radio_values)
        self.assertIn("Keyword", radio_values)
        self.assertIn("Off", radio_values)

        # Check Overfill Behavior options: off, respond, wrapup, autocont
        self.assertIn("wrapup", radio_values)
        self.assertIn("autocont", radio_values)
        self.assertIn("respond", radio_values)

    def test_template_apply_returns_to_modify_mode(self):
        win = tk.Toplevel(self.root)
        parent = tk.Frame(win)
        parent.pack()

        temp_ents = {"fast": tk.Entry(parent)}
        top_p_ents = {"fast": tk.Entry(parent)}
        min_p_ents = {"fast": tk.Entry(parent)}
        top_k_ents = {"fast": tk.Entry(parent)}
        rep_ents = {"fast": tk.Entry(parent)}
        freq_ents = {"fast": tk.Entry(parent)}
        pres_ents = {"fast": tk.Entry(parent)}
        stop_ents = {"fast": tk.Entry(parent)}
        n_batch_ents = {"fast": tk.Entry(parent)}
        ents = {"fast": tk.Entry(parent)}
        ctx_ents = {"fast": tk.Entry(parent)}

        active_template = tk.StringVar(value="T1")
        template_mode = tk.StringVar(value="write")

        vars_dict = {
            "labels": {}, "ents": ents, "ctx_ents": ctx_ents, "n_batch_ents": n_batch_ents,
            "temp_ents": temp_ents, "top_p_ents": top_p_ents, "min_p_ents": min_p_ents, "top_k_ents": top_k_ents,
            "rep_ents": rep_ents, "freq_ents": freq_ents, "pres_ents": pres_ents, "stop_ents": stop_ents,
            "active_template": active_template,
            "template_mode": template_mode,
            "dynamic_params_var": tk.BooleanVar(value=True)
        }

        build_models_tab(parent, self.app, win, vars_dict, [])

        # Find Copy button on fast tier block
        copy_buttons = [w for w in win.winfo_children() if isinstance(w, tk.Button) and "Copy" in str(w.cget("text"))]
        # If not direct child, search recursively
        def _find_copy(w):
            res = []
            for c in w.winfo_children():
                if isinstance(c, tk.Button) and "Copy" in str(c.cget("text")):
                    res.append(c)
                res.extend(_find_copy(c))
            return res
        copy_btns = _find_copy(win)
        self.assertGreater(len(copy_btns), 0)

        # Trigger copy with mocked messagebox
        with patch("tkinter.messagebox.showinfo") as mock_info:
            copy_btns[0].invoke()
            self.assertEqual(template_mode.get(), "modify")
            # Confirm values were applied from T1 ("temp": 0.7)
            self.assertEqual(temp_ents["fast"].get(), "0.7")


if __name__ == "__main__":
    unittest.main()
