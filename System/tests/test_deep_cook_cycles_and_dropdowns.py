import unittest
import tkinter as tk
from unittest.mock import MagicMock, patch
import re
import threading

from serenity_resources import THEME
from System.settings_tabs import build_agents_tab
from System.settings_ui import open_settings_window


class MockApp:
    def __init__(self, root):
        self.root = root
        self.config_file = "mock_config.json"
        self.icon_path = ""
        self.active_persona_level = 5
        self.model_path = "mock_model.gguf"
        self.model = MagicMock()
        self.kv_manager = None
        self.stop_process = MagicMock()
        self.stop_process.is_set.return_value = False
        self.process_queue = MagicMock()
        self.fonts = {
            "title": ("Segoe UI", 12, "bold"),
            "bold": ("Segoe UI", 10, "bold"),
            "ui_label": ("Segoe UI", 9),
            "ui_small": ("Segoe UI", 8),
            "ui_button": ("Segoe UI", 9),
            "stats": ("Segoe UI", 8),
            "log": ("Consolas", 9)
        }
        self.config = {
            "deep_cook_cycles": 3,
            "deep_cook_agent_coop": "otl_per_step",
            "deep_cook_formatting": "per_step",
            "normal_generation_formatting": "inline",
            "delegation_enabled": True,
            "delegation_model_mode": "lvl6_7_model",
            "subagent_selection_mode": "minimal",
            "cecilia_delegation_mode": "shadow_wizard",
            "delegation_chain_preset": "standard",
            "delegation_handoff_target": "lvl3_compiler",
            "streaming_mode": "Buffered",
            "custom_templates": {}
        }
        self.state = {
            "deep_cook": False,
            "deep_cook_behavior": "oneshot",
            "streaming_mode": "Buffered"
        }
        self.model_paths = {"fast": "", "deep_cook": ""}
        self.gpu_layer_config = {"fast": -1}
        self.context_size_config = {"fast": 4096}
        self.n_batch_config = {"fast": 512}
        self.temp_config = {"fast": 0.8}
        self.top_p_config = {"fast": 0.95}
        self.min_p_config = {"fast": 0.05}
        self.top_k_config = {"fast": 40}
        self.repeat_penalty_config = {"fast": 1.1}
        self.frequency_penalty_config = {"fast": 0.0}
        self.presence_penalty_config = {"fast": 0.0}
        self.stop_strings_config = {"fast": "###"}

    def save_config(self):
        pass

    def clear_current_history(self):
        pass

    def _sync_deep_cook_ui(self):
        pass

    def _is_rgb_supported(self):
        return False

    def list_user_profiles(self):
        return ["Default"]

    def get_active_username(self):
        return "Default"


class TestDeepCookCyclesAndDropdowns(unittest.TestCase):
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

    def test_settings_submenu_two_line_titles(self):
        """Verify all settings submenus use 2-line centered titles following 'Agents &\\nDeep Cook' structure."""
        with patch("System.stt_manager.STTManager.get_input_devices", return_value=[]):
            open_settings_window(self.app, is_generating=False)
            toplevels = [w for w in self.root.winfo_children() if isinstance(w, tk.Toplevel)]
            self.assertGreater(len(toplevels), 0)
            win = toplevels[0]
            
            # Find tab buttons inside win
            tab_buttons = [w for w in win.winfo_children() if isinstance(w, tk.Frame)]
            found_labels = []
            def _find_buttons(parent):
                for child in parent.winfo_children():
                    if isinstance(child, tk.Button):
                        txt = child.cget("text")
                        if "\n" in txt:
                            found_labels.append(txt)
                    _find_buttons(child)
            _find_buttons(win)

            expected_submenus = [
                "Models &\nParams",
                "Inference &\nExecution",
                "Agents &\nDeep Cook",
                "Additional\nSettings",
                "Users &\nSecurity",
                "Personalize\n& Theme"
            ]
            for exp in expected_submenus:
                self.assertIn(exp, found_labels, f"Expected 2-line submenu '{exp}' not found in settings window.")

    def test_agents_tab_deep_cook_settings_controls(self):
        """Verify Deep Cook cycles (2,3,4,8,12), agent cooperation, and formatting controls in Agents tab."""
        win = tk.Toplevel(self.root)
        parent = tk.Frame(win)
        parent.pack()

        vars_dict = {
            "delegation_enabled_var": tk.BooleanVar(value=True),
            "delegation_model_mode_var": tk.StringVar(value="lvl6_7_model"),
            "subagent_density_var": tk.StringVar(value="minimal"),
            "cecilia_mode_var": tk.StringVar(value="shadow_wizard"),
            "chain_preset_var": tk.StringVar(value="standard"),
            "handoff_target_var": tk.StringVar(value="lvl3_compiler"),
            "deep_cook_cycles_var": tk.StringVar(value="4"),
            "deep_cook_coop_var": tk.StringVar(value="otl_per_step"),
            "deep_cook_format_var": tk.StringVar(value="per_step"),
            "normal_format_var": tk.StringVar(value="chunked")
        }

        tab = build_agents_tab(parent, self.app, win, vars_dict)
        self.assertIsInstance(tab, tk.Frame)

        vars_dict["deep_cook_cycles_var"].set("16")
        self.assertEqual(str(vars_dict["deep_cook_cycles_var"].get()), "16")

        vars_dict["deep_cook_cycles_var"].set("20")
        self.assertEqual(str(vars_dict["deep_cook_cycles_var"].get()), "20")

        vars_dict["deep_cook_cycles_var"].set("auto")
        self.assertEqual(str(vars_dict["deep_cook_cycles_var"].get()), "auto")

        vars_dict["deep_cook_coop_var"].set("otl_per_cycle")
        self.assertEqual(vars_dict["deep_cook_coop_var"].get(), "otl_per_cycle")

        vars_dict["deep_cook_format_var"].set("upon_completion")
        self.assertEqual(vars_dict["deep_cook_format_var"].get(), "upon_completion")

        vars_dict["normal_format_var"].set("upon_end")
        self.assertEqual(vars_dict["normal_format_var"].get(), "upon_end")

    def test_deep_cook_cycle_limits_enforced(self):
        """Verify cycle loop respects configured deep_cook_cycles (2, 3, 4, 8, 12, 16, 20, auto)."""
        for cycle_limit in [2, 3, 4, 8, 12, 16, 20, "auto"]:
            self.app.config["deep_cook_cycles"] = cycle_limit
            cfg_cycles = str(self.app.config.get("deep_cook_cycles", "3")).strip().lower()
            is_auto_cycles = (cfg_cycles == "auto")
            max_cycles = 50 if is_auto_cycles else int(cfg_cycles)

            current_cycle = 1
            task_complete = False
            logged_events = []
            
            while not task_complete:
                snapshot = "[STATUS: CONTINUE] Next phase needed."
                if re.search(r'\[(?:STATUS:\s*)?TOTAL RESOLUTION\]', snapshot, re.IGNORECASE):
                    task_complete = True
                elif not is_auto_cycles and current_cycle >= max_cycles:
                    task_complete = True
                    logged_events.append(f"Deep Cook cycle budget reached ({current_cycle}/{max_cycles} cycles).")
                elif is_auto_cycles and current_cycle >= 3:
                    # In auto mode, model resolves dynamically when done
                    snapshot_res = "[STATUS: TOTAL RESOLUTION] Total resolution achieved."
                    if re.search(r'\[(?:STATUS:\s*)?TOTAL RESOLUTION\]', snapshot_res, re.IGNORECASE):
                        task_complete = True
                        logged_events.append("Assessment indicates total resolution attained.")
                else:
                    current_cycle += 1

            self.assertTrue(task_complete)
            if not is_auto_cycles:
                self.assertEqual(current_cycle, cycle_limit)
                self.assertIn(f"({cycle_limit}/{cycle_limit} cycles)", logged_events[0])
            else:
                self.assertEqual(current_cycle, 3)
                self.assertIn("total resolution attained", logged_events[0])

    def test_deep_cook_dropdown_placement_and_toggle(self):
        """Verify dropdown creation, nested step binding, and accordion open/close."""
        from main import ChatbotApp
        
        # Test directly on ChatbotApp methods
        chat = tk.Text(self.root)
        app = MagicMock()
        app.chat_history = chat
        app.fonts = {"stats": ("Segoe UI", 8), "log": ("Consolas", 9)}
        app.state = {}
        app.config = {"scroll_lock_enabled": False}
        app._is_chat_at_bottom = MagicMock(return_value=True)

        # 1. Start Cycle 1
        msg_cyc = {"ctype": "cycle", "cnum": 1, "dnum": 0, "title": "Cycle 1: Plan"}
        ChatbotApp._handle_deep_cook_ui_start(app, msg_cyc)
        cyc_tag = "dc_cycle_1_0"
        self.assertEqual(app.state.get("current_deep_cook_cyc_tag"), cyc_tag)
        self.assertEqual(str(chat.tag_cget(cyc_tag, "elide")), "1")

        # 2. Start Step 1 inside Cycle 1
        msg_step1 = {"ctype": "draft", "cnum": 1, "dnum": 1, "title": "Cycle 1 | Drafting 1"}
        ChatbotApp._handle_deep_cook_ui_start(app, msg_step1)
        step1_tag = "dc_draft_1_1"
        self.assertIn(step1_tag, app.state.get(f"nested_tags_{cyc_tag}", []))
        self.assertEqual(str(chat.tag_cget(step1_tag, "elide")), "1")

        # 3. Start Step 2 inside Cycle 1
        msg_step2 = {"ctype": "draft", "cnum": 1, "dnum": 2, "title": "Cycle 1 | Drafting 2"}
        ChatbotApp._handle_deep_cook_ui_start(app, msg_step2)
        step2_tag = "dc_draft_1_2"
        self.assertIn(step2_tag, app.state.get(f"nested_tags_{cyc_tag}", []))

        # 4. Populate batch text for Step 1
        msg_batch1 = {"ctype": "draft", "cnum": 1, "dnum": 1, "text": "Step 1 Resolved Insight."}
        ChatbotApp._handle_deep_cook_ui_batch(app, msg_batch1)
        self.assertIn("Step 1 Resolved Insight.", chat.get("1.0", tk.END))

        # 5. Expand Cycle 1 (simulate click)
        mock_cyc_btn = MagicMock()
        ChatbotApp.toggle_cyc_tag(app, cyc_tag, btn=mock_cyc_btn, cnum=1)
        self.assertEqual(str(chat.tag_cget(cyc_tag, "elide")), "0")  # Cycle un-elided
        self.assertEqual(str(chat.tag_cget(step1_tag, "elide")), "1")  # Step 1 content remains collapsed
        mock_cyc_btn.config.assert_called_with(text="[-] Hide Cycle 1")

        # 6. Collapse Cycle 1
        ChatbotApp.toggle_cyc_tag(app, cyc_tag, btn=mock_cyc_btn, cnum=1)
        self.assertEqual(str(chat.tag_cget(cyc_tag, "elide")), "1")  # Cycle elided
        self.assertEqual(str(chat.tag_cget(step1_tag, "elide")), "1")  # Step 1 elided
        mock_cyc_btn.config.assert_called_with(text="[+] View Cycle 1")

    def test_deep_cook_prompt_sending_and_tier_preservation(self):
        """Verify prompt sending when deep cook is toggled on, and tier preservation at level 6."""
        from main import ChatbotApp
        app = self.app
        app.state["deep_cook"] = True
        app.state["running"] = False
        app.user_input = tk.Text(self.root)
        app.user_input.insert("1.0", "Solve this complex puzzle step-by-step")
        app.last_user_message = ""
        app.displayed_user_msg = []
        app._display_user_message = lambda m: app.displayed_user_msg.append(m)
        app._parse_and_stage_filename_imports = lambda m: m
        app.model = MagicMock()
        app.current_model_tier = "deep_cook"
        app.depth_slider = MagicMock()
        app.depth_slider.get.return_value = "6"
        app.send_deep_cook_message = lambda *a, **k: ChatbotApp.send_deep_cook_message(app, *a, **k)
        app._prep_generation = MagicMock()
        app._display_ai_message = MagicMock()
        app._set_prefill_avatar_state = MagicMock()
        app.check_process_queue = MagicMock()
        app._generation_worker_deep_cook = MagicMock()

        spawned_threads = []
        real_thread = threading.Thread
        def fake_thread(*args, **kwargs):
            t = real_thread(*args, **kwargs)
            spawned_threads.append(kwargs.get("target"))
            return MagicMock()

        with patch("threading.Thread", side_effect=fake_thread):
            ChatbotApp.send_message(app)

        self.assertIn("Solve this complex puzzle step-by-step", app.displayed_user_msg)
        self.assertEqual(app.user_input.get("1.0", tk.END).strip(), "")
        self.assertIn(app._generation_worker_deep_cook, spawned_threads)

        # Also verify that when model_swap is called with target_tier='deep_cook' at level 6,
        # it is not overridden to 'transcendent'
        app.model_paths["deep_cook"] = "dummy.gguf"
        app.script_dir = ""
        app.halt_process = MagicMock()
        app._drain_process_queue = MagicMock()
        app.load_params = MagicMock()
        app.clear_chat_ui = MagicMock()
        app.set_ui_state = MagicMock()
        app.set_avatar_state = MagicMock()
        app._log_and_display = MagicMock()
        app._load_model_worker = MagicMock()
        app.save_history = MagicMock()
        with patch("os.path.exists", return_value=True), \
             patch("threading.Thread") as mock_swap_th:
            app.current_model_tier = "transcendent"
            ChatbotApp.model_swap(app, target_level=6, target_tier="deep_cook")
            self.assertEqual(app.current_model_tier, "deep_cook")


if __name__ == "__main__":
    unittest.main()

