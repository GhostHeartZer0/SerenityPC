# System/tests/test_dynamic_dispatch.py
# Unit Tests for Context-Driven Dynamic Agent Orchestration, Stateful Offload/Load, and Observability.

import unittest
from unittest.mock import MagicMock
import queue
import json
from System.orchestration_manager import (
    IntentProfiler,
    UtilityDispatchEngine,
    SerializedAgentContext,
    OrchestrationManager,
    get_available_subagents,
    SUBAGENT_ROLES
)
from System.tool_registry import GemmaToolRegistry

class MockApp:
    def __init__(self):
        self.config = {
            "delegation_enabled": True,
            "cecilia_delegation_mode": "shadow_wizard",
            "delegation_chain_preset": "dynamic_utility",
            "offline_mode": True
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
        return "Calculated deduction: The algorithm converges in O(n log n) time."

class TestDynamicDispatch(unittest.TestCase):
    def setUp(self):
        self.app = MockApp()
        self.mgr = OrchestrationManager(self.app)

    def test_intent_profiler_factual(self):
        """Verify factual intent extraction."""
        p = IntentProfiler.profile("What is the latest release date of Python 3.14 on python.org?")
        self.assertGreaterEqual(p.factual_need, 0.5)
        self.assertLess(p.emotional_intensity, 0.3)

    def test_intent_profiler_reasoning(self):
        """Verify deep reasoning intent extraction."""
        p = IntentProfiler.profile("Explain step-by-step why quicksort has a worst-case time complexity of O(n^2) and how to optimize it.")
        self.assertGreaterEqual(p.reasoning_complexity, 0.5)
        self.assertLess(p.emotional_intensity, 0.3)

    def test_intent_profiler_emotional(self):
        """Verify affective/emotional intent extraction."""
        p = IntentProfiler.profile("I feel so overwhelmed, lonely, and stressed with work lately...")
        self.assertGreaterEqual(p.emotional_intensity, 0.6)
        self.assertLess(p.reasoning_complexity, 0.4)

    def test_intent_profiler_speed(self):
        """Verify quick/speed intent extraction."""
        p = IntentProfiler.profile("Quick summary tldr")
        self.assertGreaterEqual(p.speed_preference, 0.5)

    def test_dynamic_dispatch_factual_query(self):
        """Verify factual query selects Lvl 2 Searcher and excludes Lvl 4 Confidant."""
        trace = UtilityDispatchEngine.evaluate(6, "Search for recent gravitational wave discoveries in 2026")
        self.assertIn(2, trace.selected_agent_ids, "Lvl 2 (Searcher) must be selected for factual queries")
        self.assertNotIn(4, trace.selected_agent_ids, "Lvl 4 (Confidant) must be skipped for purely factual queries")

    def test_dynamic_dispatch_emotional_query(self):
        """Verify emotional query selects Lvl 4 Confidant and skips Lvl 2 Searcher."""
        trace = UtilityDispatchEngine.evaluate(6, "I feel sad, lonely, and really need someone to talk to about my day.")
        self.assertIn(4, trace.selected_agent_ids, "Lvl 4 (Confidant) must be selected for emotional queries")
        self.assertNotIn(2, trace.selected_agent_ids, "Lvl 2 (Searcher) must be skipped when no factual lookup is needed")

    def test_subagent_permission_invariant(self):
        """Verify Level 6 is never selected as a subagent under Level 6 orchestrator."""
        trace_l6 = UtilityDispatchEngine.evaluate(6, "Solve this complex overarching multi-disciplinary problem")
        self.assertNotIn(6, trace_l6.selected_agent_ids, "Level 6 must NOT be available as a subagent to Level 6")

        trace_l7 = UtilityDispatchEngine.evaluate(7, "Solve this complex overarching multi-disciplinary problem")
        l6_candidate = next((s for s in trace_l7.candidate_scores if s.agent_id == 6), None)
        self.assertIsNotNone(l6_candidate)
        self.assertNotIn("Access Restricted", l6_candidate.rationale)

    def test_stateful_context_offload_and_load(self):
        """Verify minimal context serialization, anti-OOM budget clamp, and load reconstitution."""
        ctx = SerializedAgentContext(
            trace_id="test_trace_123",
            origin_agent=2,
            target_agent=5,
            token_budget=128, # Small budget for testing clamp
            staged_facts=["Fact 1: Quasars emit vast energy.", "Fact 2: Found at centers of galaxies."],
            reasoning_steps=["Step 1: Energy powered by supermassive black hole accretion."],
            raw_payload="A" * 5000 # 5000 characters of raw data
        )

        offloaded = ctx.offload()
        self.assertIsInstance(offloaded, str)
        self.assertLess(len(offloaded), 1000, "Serialized offload must clamp payload to budget")

        loaded = SerializedAgentContext.load(offloaded)
        self.assertEqual(loaded.trace_id, "test_trace_123")
        self.assertEqual(loaded.origin_agent, 2)
        self.assertEqual(loaded.target_agent, 5)
        self.assertEqual(len(loaded.staged_facts), 2)
        self.assertEqual(len(loaded.reasoning_steps), 1)

    def test_observability_decision_trace_logging(self):
        """Verify decision trace is logged to diagnostics, tool log, and thought stream without chat output leakage."""
        res = self.mgr.execute_delegation_chain(6, "Why do black holes emit Hawking radiation?", {})
        self.assertIn("compiled_briefing", res)
        self.assertIn("trace", res)
        self.assertGreater(len(res["reports"]), 0)

        # Collect queue events
        events = []
        while not self.app.process_queue.empty():
            events.append(self.app.process_queue.get())

        statuses = [e.get("status") for e in events]
        self.assertIn("diag_log_update", statuses, "Decision trace must be logged to diagnostics log")
        self.assertIn("thought_stream", statuses, "Intermediate progression must be logged to thought stream")
        self.assertIn("tool_log_update", statuses, "Subagent taskings must be logged to tool log")
        self.assertNotIn("streaming", statuses, "Subagent communications must NOT leak into chat response buffer")

    def test_subagent_density_modes(self):
        """Verify minimal required vs all desired selection modes."""
        trace_min = UtilityDispatchEngine.evaluate(6, "Hello there", selection_mode="minimal")
        self.assertEqual(len(trace_min.selected_agent_ids), 1, "Minimal mode should only pick fallback/best agent for generic greetings")

        trace_all = UtilityDispatchEngine.evaluate(6, "Hello there", selection_mode="all")
        self.assertEqual(trace_all.selected_agent_ids, [1, 2, 3, 4, 5], "All mode should dispatch all 5 subagents for Lvl 6")

    def test_delegation_dynamic_model_swap(self):
        """Verify dynamic model swap is invoked when delegation_model_mode is per_subagent_model."""
        self.app.config["delegation_model_mode"] = "per_subagent_model"
        self.app.model_paths = {
            "fast": "S:/LLM/fast.gguf",
            "search": "S:/LLM/search.gguf",
            "low": "S:/LLM/collab.gguf",
            "med": "S:/LLM/confidant.gguf",
            "high": "S:/LLM/brains.gguf",
            "transcendent": "S:/LLM/transcendent.gguf"
        }
        self.app.model_path = "S:/LLM/transcendent.gguf"
        self.app.model_swap = MagicMock()

        self.mgr.execute_delegation_chain(6, "Solve complex physics calculus", {})
        # Should have called model_swap during subagent steps and at finish to restore
        self.assertTrue(self.app.model_swap.called, "model_swap should be called in per_subagent_model mode")


if __name__ == "__main__":
    unittest.main()

