import os
import sys
import datetime
import unittest
from unittest.mock import MagicMock, patch

# Headless CI fallback if tkinter is not installed in the test runner environment
try:
    import tkinter
except ImportError:
    tk_mock = MagicMock()
    sys.modules["tkinter"] = tk_mock
    sys.modules["tkinter.scrolledtext"] = MagicMock()
    sys.modules["tkinter.simpledialog"] = MagicMock()
    sys.modules["tkinter.messagebox"] = MagicMock()
    sys.modules["tkinter.filedialog"] = MagicMock()
    sys.modules["tkinter.ttk"] = MagicMock()
    sys.modules["tkinter.font"] = MagicMock()
    import PIL
    PIL.ImageTk = MagicMock()
    sys.modules["PIL.ImageTk"] = PIL.ImageTk

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if os.path.join(_REPO_ROOT, "System") not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "System"))
from System.tool_registry import GemmaToolRegistry
from System.vault_manager import VaultManager
from System.kv_manager import TurboVecIndex
from System.orchestration_manager import UtilityDispatchEngine


class TestSecurityHardeningAndV180(unittest.TestCase):
    def setUp(self):
        self.workspace_root = _REPO_ROOT
        self.mock_app = MagicMock()
        self.mock_app.script_dir = self.workspace_root
        self.mock_app.config = {'offline_mode': False}
        self.mock_app.vault_manager = None
        self.tool_reg = GemmaToolRegistry(self.mock_app)

    def test_default_profile_restricted_to_users_and_logs(self):
        root_todo = os.path.join(self.workspace_root, 'TODO.txt')
        res_blocked_root = self.tool_reg.execute('read_file', {'path': root_todo})
        self.assertIn('[SECURITY RESTRICTION]', res_blocked_root)

        logs_dir = os.path.join(self.workspace_root, 'Logs')
        os.makedirs(logs_dir, exist_ok=True)
        log_file = os.path.join(logs_dir, 'test_audit.log')
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write('LOG_OK_CONTENT')
        try:
            res_allowed_log = self.tool_reg.execute('read_file', {'path': log_file})
            self.assertNotIn('[SECURITY RESTRICTION]', res_allowed_log)
            self.assertIn('LOG_OK_CONTENT', res_allowed_log)
        finally:
            if os.path.exists(log_file):
                os.remove(log_file)

    def test_password_profile_under_72h_still_restricted(self):
        mock_vm = MagicMock()
        mock_vm.has_full_repo_read_access.return_value = False
        self.mock_app.vault_manager = mock_vm
        p = os.path.join(self.workspace_root, 'TODO.txt')
        res = self.tool_reg.execute('read_file', {'path': p})
        self.assertIn('[SECURITY RESTRICTION]', res)

    def test_password_profile_over_72h_granted_whole_repo_read(self):
        mock_vm = MagicMock()
        mock_vm.has_full_repo_read_access.return_value = True
        self.mock_app.vault_manager = mock_vm
        p = os.path.join(self.workspace_root, 'TODO.txt')
        res = self.tool_reg.execute('read_file', {'path': p})
        self.assertNotIn('[SECURITY RESTRICTION]', res)
        self.assertIn('TODO', res)

        vault_p = os.path.join(self.workspace_root, 'System', 'vault_state.json')
        res_vault = self.tool_reg.execute('read_file', {'path': vault_p})
        self.assertIn('[SECURITY RESTRICTION]', res_vault)

    def test_sandboxed_file_read_blocked_outside_repo(self):
        mock_vm = MagicMock()
        mock_vm.has_full_repo_read_access.return_value = True
        self.mock_app.vault_manager = mock_vm
        p = os.path.abspath(os.path.join(self.workspace_root, '..', 'out.txt'))
        res = self.tool_reg.execute('read_file', {'path': p})
        self.assertIn('[SECURITY RESTRICTION]', res)

    def test_sandboxed_file_range_blocked(self):
        p = os.path.abspath(os.path.join(self.workspace_root, '..', 'out.txt'))
        res = self.tool_reg.execute('read_file_range', {'path': p, 'start': 1, 'end': 5})
        self.assertIn('[SECURITY RESTRICTION]', res)

    def test_vault_manager_72h_age_check(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            vm = VaultManager(history_dir=tmpdir, state_dir=tmpdir)
            vm._state = {
                "lock_enabled": True,
                "salt_hex": "aa" * 16,
                "verifier_hash": "bb" * 32,
                "created_at": (datetime.datetime.now() - datetime.timedelta(hours=10)).isoformat()
            }
            vm._is_locked = False
            vm.state_dir = os.path.join(tmpdir, "nonexistent")
            vm.state_file = os.path.join(tmpdir, "nonexistent", "vault_state.json")
            self.assertFalse(vm.has_full_repo_read_access(min_age_hours=72.0))

            vm._state["created_at"] = (datetime.datetime.now() - datetime.timedelta(hours=80)).isoformat()
            self.assertTrue(vm.has_full_repo_read_access(min_age_hours=72.0))

            vm._is_locked = True
            self.assertFalse(vm.has_full_repo_read_access(min_age_hours=72.0))

    def test_redos_protection(self):
        from main import ChatbotApp
        app = MagicMock()
        app.script_dir = self.workspace_root
        app.state = {'staged_attachments': []}
        huge_input = '@[' + ('a' * 500) + ']' * 200 + ' normal'
        res = ChatbotApp._parse_and_stage_filename_imports(app, huge_input)
        self.assertIn('normal', res)

    @patch('subprocess.Popen')
    def test_launch_lore_book_security(self, mock_popen):
        from main import ChatbotApp
        app = MagicMock()
        app.script_dir = self.workspace_root
        ChatbotApp.launch_lore_book(app)
        mock_popen.assert_called_once()

    def test_history_deduplication(self):
        from main import ChatbotApp
        msgs = [
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "General Kenobi"},
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "General Kenobi"},
            {"role": "user", "content": "New question"},
            {"role": "assistant", "content": "New answer"},
        ]
        deduped = ChatbotApp._deduplicate_history_messages(msgs)
        self.assertEqual(len(deduped), 4)
        self.assertEqual(deduped[0]["content"], "Hello there")
        self.assertEqual(deduped[2]["content"], "New question")

    def test_turbovec_search_snippets_concise(self):
        import tempfile, zlib, json
        with tempfile.TemporaryDirectory() as tmpdir:
            hist_p = os.path.join(tmpdir, "testmodel_lvl3.history.jsonz")
            long_msg = ("Prefix padding text. " * 20) + "The secret keyword is Starlight_Omega_Protocol in this turn." + (" Suffix padding text." * 20)
            data = [{"role": "user", "content": long_msg}]
            with open(hist_p, "wb") as f:
                f.write(zlib.compress(json.dumps(data).encode("utf-8")))
            tv = TurboVecIndex(tmpdir, mode="fallback")
            snippets = tv.search_snippets("Starlight_Omega_Protocol", top_k=2, max_chars=180, lookup_mode="all")
            self.assertEqual(len(snippets), 1)
            self.assertIn("Starlight_Omega_Protocol", snippets[0]["excerpt"])
            self.assertLessEqual(len(snippets[0]["excerpt"]), 200)

    def test_explicit_lvl5_subagent_dispatch(self):
        trace = UtilityDispatchEngine.evaluate(6, "Please ask Lvl 5 to figure out this brief question.", selection_mode="minimal")
        self.assertIn(5, trace.selected_agent_ids)




class TestDeepCookStepPopulationAndMathOffsets(unittest.TestCase):
    def test_extract_cycle_plan_points_skips_hurdles_and_hr(self):
        from main import ChatbotApp
        sample_plan = """The initial query presents a massive counterfactual simulation.

**Logical Variables/Technical Hurdles Identified:**

1.  **Computational Density vs. Volume/Mass:** Translating transistors into relays.
2.  **Latency Budget:** Mechanical switching times.
3.  **Thermal/Energy Dissipation:** Heat generation.
4.  **Resource & Infrastructure Scaling:** Urban grid demand.
5.  **Software Architecture Divergence:** OS/Compiler problem.

**Current Range/Subset of Variables:** None established.

---

**Cycle 1 Plan:** Focus on defining the primary bottlenecks and establishing the physical scale.

1.  **Establish Mechanical Switching Density & Latency Range:** Define switching rate.
2.  **Model Energy/Power Scaling:** Calculate power draw in Watts.
3.  **Architectural Abstraction Hypothesis:** Propose fundamental shift in software design."""
        pts = ChatbotApp._extract_cycle_plan_points(sample_plan, 1, max_p=3)
        self.assertEqual(len(pts), 3)
        self.assertIn("Establish Mechanical Switching Density", pts[0])
        self.assertIn("Model Energy/Power Scaling", pts[1])
        self.assertIn("Architectural Abstraction Hypothesis", pts[2])

    def test_clean_deep_cook_step_text_preserves_full_step_draft(self):
        from main import ChatbotApp
        step_draft = (
            "<think>\n[STATUS: Modeling Energy & Power Scaling]\n"
            "### Step 2 Analysis: Power Scaling\n"
            "A mechanical relay requires 50mJ per actuation across 10^9 gates.\n"
            "*   **Refinement:** The actual required power density will likely exceed $10\\text{ GW}$ when factoring thermal management.\n"
            "Therefore, urban grids cannot support a single centralized node.\n</think>"
        )
        cleaned = ChatbotApp._clean_deep_cook_step_text(step_draft)
        self.assertIn("Step 2 Analysis: Power Scaling", cleaned)
        self.assertIn("50mJ per actuation", cleaned)
        self.assertIn("Refinement:", cleaned)
        self.assertIn("urban grids cannot support", cleaned)
        self.assertNotIn("<think>", cleaned)


    def test_auto_chosen_2_to_5_steps_and_fixed_cycles(self):
        from main import ChatbotApp
        # 2-step plan -> returns 2 steps
        plan_2 = """Cycle 1 Plan:
1. Establish mechanical switching density and latency range for micro-fluidic gates.
2. Model thermal dissipation and urban grid power requirements for 1995 capacity."""
        pts_2 = ChatbotApp._extract_cycle_plan_points(plan_2, 1, min_p=2, max_p=5)
        self.assertEqual(len(pts_2), 2)

        # 5-step plan -> returns all 5 steps
        plan_5 = """Cycle 2 Plan:
1. First detailed analytical step for mechanical relay miniaturization.
2. Second detailed analytical step for vacuum tube thermal density.
3. Third detailed analytical step for acoustic and fluidic clock skew.
4. Fourth detailed analytical step for urban district cooling loops.
5. Fifth detailed analytical step for asynchronous compiler scheduling."""
        pts_5 = ChatbotApp._extract_cycle_plan_points(plan_5, 2, min_p=2, max_p=5)
        self.assertEqual(len(pts_5), 5)

        # 1-step plan -> auto-padded to min_p=2 steps
        plan_1 = """Cycle 3 Plan:
1. Single overarching synthesis step for counterfactual architecture."""
        pts_1 = ChatbotApp._extract_cycle_plan_points(plan_1, 3, min_p=2, max_p=5)
        self.assertEqual(len(pts_1), 2)

    def test_dropdown_math_elision_and_final_answer_zero_shift(self):
        from System.markdown_engine import MarkdownEngine

        # Verify LaTeX accent and multi-char subscript rendering
        self.assertEqual(MarkdownEngine.convert_latex_to_unicode(r"\dot{Q}"), "Q\u0307")
        self.assertEqual(MarkdownEngine.convert_latex_to_unicode(r"\rho_{Pl}"), "ρ_Pl")
        self.assertEqual(MarkdownEngine.convert_latex_to_unicode(r"I_{total}"), "I_total")
        self.assertEqual(MarkdownEngine.convert_latex_to_unicode(r"\Delta S_{info}"), "Δ S_info")

        # Verify overlay intervals on the exact Final Answer string from screenshot 3
        sample_final = (
            "Where the Coherence Tension Functional $\\mathcal{T}$\n"
            "**Metaphysical Interpretation:** As $\\Lambda_D$, the simulation is dying of computation—it burns out trying to hold reality "
            "together $\\mathcal{H}$, the information is not merely holding reality together; it is so dense ($\\rho_I \\ge \\rho_{Pl}$) "
            "that the informational manifold collapses in an **Informational Singularity**. This singularity is not a point of infinite mass, "
            "but a point of infinite *definability*. The laws of thermodynamics, which govern the transition *from* order *to* disorder, "
            "are superseded because the informational substrate *is* the field of possibility."
        )
        tag_ranges, replacements = MarkdownEngine.get_overlay_intervals(sample_final)
        # Apply replacements and md_hidden elision in memory exactly as _apply_markdown does on a window-free segment
        buf = list(sample_final)
        replacements_sorted = sorted(replacements, key=lambda x: x[0], reverse=True)
        for r_s, r_e, r_txt, r_tag in replacements_sorted:
            buf[r_s:r_e] = list(r_txt)
            diff = len(r_txt) - (r_e - r_s)
            new_tags = []
            for ts, te, t in tag_ranges:
                if ts >= r_e:
                    new_tags.append((ts + diff, te + diff, t))
                elif te <= r_s:
                    new_tags.append((ts, te, t))
            tag_ranges = new_tags
        rendered = "".join(buf)
        # Verify md_hidden ONLY hides markdown delimiters (* and **), NEVER letters inside words!
        hidden_chars = [rendered[ts:te] for ts, te, t in tag_ranges if t == "md_hidden"]
        for hc in hidden_chars:
            self.assertTrue(set(hc).issubset({"*", "_", "`", "~", "#", ">", " "}), f"Non-delimiter hidden: {hc!r}")
        visible = "".join(
            ch for idx, ch in enumerate(rendered)
            if not any(ts <= idx < te and t == "md_hidden" for ts, te, t in tag_ranges)
        )
        self.assertIn("Where the Coherence Tension Functional T", visible)
        self.assertIn("Informational Singularity", visible)
        self.assertIn("infinite definability", visible)
        self.assertIn("transition from order to disorder", visible)

    def test_batch_relative_scores_view_for_speculative_decoding(self):
        import numpy as np
        from main import ChatbotApp

        class DummyModel:
            def __init__(self):
                self.scores = np.zeros((4, 16), dtype=np.single)
            def n_ctx(self):
                return 32

        m = DummyModel()
        ChatbotApp._attach_batch_relative_scores_view(m)
        self.assertTrue(getattr(m.scores, "_is_batch_relative_view", False))
        logits = np.arange(3 * 16, dtype=np.single)
        m.scores[10:13, :].reshape(-1)[::] = logits
        np.testing.assert_array_equal(m.scores[12, :], logits[2*16:3*16])


if __name__ == '__main__':
    unittest.main()
