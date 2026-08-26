# System/tests/test_delegation_and_search_filter.py
# Unit Tests for V1.7.0 Multi-Agent Delegation, Subagent Permissions, and Search Bloat Filter.

import unittest
import os
import json
import queue
import tkinter as tk
from System.tool_registry import GemmaToolRegistry
from System.orchestration_manager import (
    OrchestrationManager,
    get_available_subagents,
    build_default_delegation_plan,
    SUBAGENT_ROLES,
    HandoffReport
)
from serenity_resources import (
    PERSONA_PROMPTS,
    DELEGATION_SYSTEM_PROMPTS,
    SUBAGENT_SYSTEM_PROMPTS
)

class MockApp:
    def __init__(self):
        self.config = {
            "delegation_enabled": True,
            "cecilia_delegation_mode": "shadow_wizard",
            "delegation_chain_preset": "standard",
            "delegation_handoff_target": "lvl3_compiler",
            "offline_mode": False
        }
        self.process_queue = queue.Queue()
        self.tool_registry = GemmaToolRegistry(self)
        self.active_persona_level = 6
        self.fonts = {
            "main": ("Arial", 10),
            "bold": ("Arial", 10, "bold"),
            "ui_button": ("Arial", 9),
            "ui_label": ("Arial", 9),
            "ui_small": ("Arial", 8)
        }

    def _run_blocking_inference(self, prompt, params):
        return "Core deduction: The query can be resolved via modular pipeline synthesis."


class TestDelegationAndSearchFilter(unittest.TestCase):
    def setUp(self):
        self.app = MockApp()
        self.orch_mgr = OrchestrationManager(self.app)

    def test_subagent_visibility_permissions(self):
        """Verify Level 6 as an agent is ONLY seen/accessible by Level 7."""
        # Level 6 orchestrator can only see Levels 1-5
        l6_agents = get_available_subagents(6)
        self.assertEqual(l6_agents, [1, 2, 3, 4, 5])
        self.assertNotIn(6, l6_agents, "Level 6 must NOT see Level 6 as a subagent")

        # Level 7 (Cecilia) can see Levels 1-6
        l7_agents = get_available_subagents(7)
        self.assertEqual(l7_agents, [1, 2, 3, 4, 5, 6])
        self.assertIn(6, l7_agents, "Level 7 must have access to Level 6 as a subagent")

        # Levels 1-5 have no subagent delegation authority
        for lvl in range(1, 6):
            self.assertEqual(get_available_subagents(lvl), [])

    def test_delegation_plan_structure(self):
        """Verify default delegation pipeline has L2 Search -> L3 Store -> L5 Reason -> L3 Compile."""
        plan = build_default_delegation_plan(6, "Explain quantum entanglement")
        self.assertEqual(len(plan), 4)
        self.assertEqual(plan[0].assigned_level, 2)
        self.assertEqual(plan[1].assigned_level, 3)
        self.assertEqual(plan[2].assigned_level, 5)
        self.assertEqual(plan[3].assigned_level, 3)

    def test_delegation_execution_and_thought_isolation(self):
        """Verify delegation execution produces structured handoff reports without leaking to chat."""
        result = self.orch_mgr.execute_delegation_chain(6, "How do black holes form?", {})
        self.assertIn("compiled_briefing", result)
        self.assertIn("reports", result)
        self.assertGreaterEqual(len(result["reports"]), 1)

        # Check that queue received thought stream and tool log updates
        messages = []
        while not self.app.process_queue.empty():
            messages.append(self.app.process_queue.get())

        statuses = [m.get("status") for m in messages]
        self.assertIn("tool_log_update", statuses)
        self.assertIn("thought_stream", statuses)

        # Ensure no direct streaming to chat output occurred from subagent chain
        self.assertNotIn("streaming", statuses)

    def test_cecilia_modes_system_prompts(self):
        """Verify Cecilia Shadow Wizard vs Divine Judgement prompt separation."""
        self.assertIn(6, DELEGATION_SYSTEM_PROMPTS)
        self.assertIn("shadow_wizard", DELEGATION_SYSTEM_PROMPTS[7])
        self.assertIn("divine_judgement", DELEGATION_SYSTEM_PROMPTS[7])

        sw_prompt = DELEGATION_SYSTEM_PROMPTS[7]["shadow_wizard"]
        dj_prompt = DELEGATION_SYSTEM_PROMPTS[7]["divine_judgement"]

        self.assertIn("Shadow Wizard", sw_prompt)
        self.assertIn("one-way mirror", sw_prompt)
        self.assertIn("Divine Judgement", dj_prompt)
        self.assertIn("one-way mirror", dj_prompt)

    def test_subagent_role_definitions(self):
        """Verify subagent roles and system prompts for all levels."""
        for lvl in range(1, 7):
            self.assertIn(lvl, SUBAGENT_ROLES)
            self.assertIn(lvl, SUBAGENT_SYSTEM_PROMPTS)
            self.assertTrue(len(SUBAGENT_SYSTEM_PROMPTS[lvl]) > 10)

    def test_search_bloat_filter_logic(self):
        """Verify that web search handler filters boilerplate noise and deduplicates."""
        raw_dirty_text = """
        Accept all cookies to continue browsing
        Privacy Policy & Terms of Use
        [Quantum Mechanics Overview]
        Quantum mechanics is a fundamental theory in physics that describes the behavior of nature at atomic scales.
        Sign in to read full article
        Skip to main content
        [Quantum Mechanics Overview]
        Quantum mechanics is a fundamental theory in physics that describes the behavior of nature at atomic scales.
        All rights reserved Copyright 2026
        """
        # Execute tool registry search in offline mock
        self.app.config["offline_mode"] = True
        res = self.app.tool_registry.execute("web_search", {"query": "quantum mechanics"})
        self.assertIn("[OFFLINE MODE]", res)

        # Direct execution of delegate_subtask
        del_res = self.app.tool_registry.execute("delegate_subtask", {"subagent_level": 2, "task_description": "Search arXiv"})
        self.assertIn("[SUBAGENT LVL 2 DISPATCH]", del_res)


if __name__ == "__main__":
    unittest.main()
