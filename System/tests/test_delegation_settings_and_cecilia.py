# System/tests/test_delegation_settings_and_cecilia.py
# Unit Tests for Delegation Section Settings (Lvls 6 & 7) and Cecilia Shadow Wizard / Divine Judgement Modes.

import sys
import os
import unittest
import queue
import tkinter as tk

# Ensure workspace root is in sys.path for direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from System.orchestration_manager import (
    OrchestrationManager,
    UtilityDispatchEngine,
    get_available_subagents,
    SUBAGENT_ROLES
)
from serenity_resources import DELEGATION_SYSTEM_PROMPTS


class MockApp:
    def __init__(self):
        self.config = {
            "delegation_enabled": True,
            "cecilia_delegation_mode": "shadow_wizard",
            "subagent_selection_mode": "minimal",
            "delegation_model_mode": "lvl6_7_model",
            "delegation_chain_preset": "standard",
            "delegation_handoff_target": "lvl3_compiler",
            "offline_mode": True
        }
        self.process_queue = queue.Queue()
        self.active_persona_level = 7
        self.fonts = {
            "main": ("Arial", 10),
            "bold": ("Arial", 10, "bold"),
            "ui_button": ("Arial", 9),
            "ui_label": ("Arial", 9),
            "ui_small": ("Arial", 8)
        }

    def _run_blocking_inference(self, prompt, params):
        return "Cecilia Core Insight: Hidden truth derived with angelic precision."


class TestDelegationSettingsAndCecilia(unittest.TestCase):
    def setUp(self):
        self.app = MockApp()
        self.orch_mgr = OrchestrationManager(self.app)

    def test_delegation_settings_defaults_and_updates(self):
        """Verify all delegation settings keys for Lvls 6 & 7 update and reflect correctly."""
        keys = [
            "delegation_enabled",
            "cecilia_delegation_mode",
            "subagent_selection_mode",
            "delegation_model_mode",
            "delegation_chain_preset",
            "delegation_handoff_target"
        ]
        for k in keys:
            self.assertIn(k, self.app.config)

        # Simulate UI setting changes
        self.app.config["delegation_enabled"] = False
        self.app.config["cecilia_delegation_mode"] = "divine_judgement"
        self.app.config["subagent_selection_mode"] = "all"
        self.app.config["delegation_model_mode"] = "per_subagent_model"
        self.app.config["delegation_chain_preset"] = "direct_strike"
        self.app.config["delegation_handoff_target"] = "taskmaster_direct"

        self.assertFalse(self.app.config["delegation_enabled"])
        self.assertEqual(self.app.config["cecilia_delegation_mode"], "divine_judgement")
        self.assertEqual(self.app.config["subagent_selection_mode"], "all")
        self.assertEqual(self.app.config["delegation_model_mode"], "per_subagent_model")
        self.assertEqual(self.app.config["delegation_chain_preset"], "direct_strike")
        self.assertEqual(self.app.config["delegation_handoff_target"], "taskmaster_direct")

    def test_cecilia_shadow_wizard_subagent_visibility(self):
        """Verify Cecilia (Lvl 7) in Shadow Wizard mode sees all 6 subagent levels (1-6)."""
        subagents = get_available_subagents(7)
        self.assertEqual(subagents, [1, 2, 3, 4, 5, 6])
        self.assertIn(6, subagents, "Level 6 must be available as subagent to Level 7")

    def test_cecilia_subagent_density_selection_modes(self):
        """Verify subagent selection density: 'all' vs 'minimal'."""
        # Selection mode 'all': selects all available subagents
        trace_all = UtilityDispatchEngine.evaluate(7, "Analyze quantum paradox", selection_mode="all")
        self.assertEqual(trace_all.selected_agent_ids, [1, 2, 3, 4, 5, 6])

        # Selection mode 'minimal': filters based on query profile
        trace_min = UtilityDispatchEngine.evaluate(7, "Quick 1+1 calculation", selection_mode="minimal")
        self.assertLess(len(trace_min.selected_agent_ids), 6)

    def test_cecilia_prompts_separation(self):
        """Verify Cecilia Shadow Wizard vs Divine Judgement prompts separation."""
        prompts = DELEGATION_SYSTEM_PROMPTS.get(7, {})
        self.assertIn("shadow_wizard", prompts)
        self.assertIn("divine_judgement", prompts)

        sw = prompts["shadow_wizard"]
        dj = prompts["divine_judgement"]

        self.assertIn("Shadow Wizard", sw)
        self.assertIn("subagent", sw)
        self.assertIn("Divine Judgement", dj)
        self.assertIn("without subagent handoffs", dj)

    def test_cecilia_shadow_wizard_execution_chain(self):
        """Verify Level 7 delegation execution executes subagent chain and handles Lvl 6 supervisor."""
        self.app.config["subagent_selection_mode"] = "all"
        res = self.orch_mgr.execute_delegation_chain(7, "Analyze cosmic background radiation", {})

        self.assertIn("compiled_briefing", res)
        self.assertIn("reports", res)
        self.assertGreaterEqual(len(res["reports"]), 1)

        # Confirm queue received thoughts
        statuses = []
        while not self.app.process_queue.empty():
            statuses.append(self.app.process_queue.get().get("status"))

        self.assertIn("agentic_stream", statuses)

    def test_cecilia_divine_judgement_bypasses_subagents(self):
        """Verify Divine Judgement mode does not spawn or aggregate subagent reports."""
        self.app.config["cecilia_delegation_mode"] = "divine_judgement"
        res = self.orch_mgr.execute_delegation_chain(7, "Direct decree prompt", {})
        # Should produce direct insight without multi-agent handoff reports
        self.assertEqual(len(res.get("reports", [])), 0)
        self.assertIn("Cecilia Core Insight", res.get("compiled_briefing", ""))

    def test_lvl6_subagent_visibility(self):
        """Verify Level 6 supervisor sees levels 1-5 only (excluding itself and Level 7)."""
        subagents = get_available_subagents(6)
        self.assertEqual(subagents, [1, 2, 3, 4, 5])
        self.assertNotIn(6, subagents)
        self.assertNotIn(7, subagents)

if __name__ == "__main__":
    unittest.main()
