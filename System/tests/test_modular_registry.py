# System/tests/test_modular_registry.py
# Unit tests for Modular Registry and Dynamic Parameter Auto-Adjustment Engine

import unittest
import os
import sys

# Ensure repository root is in sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from System.modular_registry import ModularRegistry, DynamicParamRegistry
from System.tool_registry import GemmaToolRegistry

class TestModularRegistry(unittest.TestCase):
    def setUp(self):
        self.reg = ModularRegistry(name="TestRegistry")

    def test_registration_and_execution(self):
        @self.reg.register("add", priority=10)
        def add(a, b):
            return a + b

        self.assertTrue(self.reg.has("add"))
        self.assertEqual(self.reg.execute("add", 2, 3), 5)
        self.assertEqual(self.reg.get_metadata("add")["priority"], 10)

    def test_missing_key_raises(self):
        with self.assertRaises(KeyError):
            self.reg.execute("non_existent_key")

    def test_list_keys(self):
        self.reg.register("k1", lambda: 1)
        self.reg.register("k2", lambda: 2)
        self.assertIn("k1", self.reg.list_keys())
        self.assertIn("k2", self.reg.list_keys())


class TestDynamicParamRegistry(unittest.TestCase):
    def setUp(self):
        self.dynamic_reg = DynamicParamRegistry()
        self.base_params = {
            "temperature": 0.8,
            "top_p": 0.95,
            "min_p": 0.05,
            "repeat_penalty": 1.0,
            "top_k": 64,
            "max_tokens": 2048,
        }

    def test_coding_prompt_adjustment(self):
        coding_prompt = "Can you write a python function to refactor this binary search algorithm?"
        adjusted, domain = self.dynamic_reg.adjust_params(coding_prompt, self.base_params)
        
        self.assertIsNotNone(domain)
        self.assertIn("Coding", domain)
        self.assertLessEqual(adjusted["temperature"], 0.4)
        self.assertGreaterEqual(adjusted["min_p"], 0.10)
        self.assertLessEqual(adjusted["top_p"], 0.90)

    def test_code_fenced_block_adjustment(self):
        fenced_prompt = "Look at this code:\n```\ndef foo(): pass\n```\nWhat is wrong with it?"
        adjusted, domain = self.dynamic_reg.adjust_params(fenced_prompt, self.base_params)
        
        self.assertIsNotNone(domain)
        self.assertIn("Coding", domain)
        self.assertLessEqual(adjusted["temperature"], 0.4)

    def test_math_prompt_adjustment(self):
        math_prompt = "Calculate the derivative of x^3 * sin(x) using the product rule."
        adjusted, domain = self.dynamic_reg.adjust_params(math_prompt, self.base_params)
        
        self.assertIsNotNone(domain)
        self.assertIn("Math", domain)
        self.assertLessEqual(adjusted["temperature"], 0.4)

    def test_creative_prompt_adjustment(self):
        creative_prompt = "Write a sci-fi cyberpunk novel opening with rich dialogue between two rogue androids."
        adjusted, domain = self.dynamic_reg.adjust_params(creative_prompt, self.base_params)
        
        self.assertIsNotNone(domain)
        self.assertIn("Creative", domain)
        self.assertGreaterEqual(adjusted["temperature"], 0.85)
        self.assertGreaterEqual(adjusted["top_p"], 0.95)

    def test_factual_prompt_adjustment(self):
        factual_prompt = "Who is Ada Lovelace and what did she contribute to computing?"
        adjusted, domain = self.dynamic_reg.adjust_params(factual_prompt, self.base_params)
        
        self.assertIsNotNone(domain)
        self.assertIn("Factual", domain)
        self.assertLessEqual(adjusted["temperature"], 0.70)

    def test_neutral_prompt_keeps_base(self):
        neutral_prompt = "Hello there! How are you doing today?"
        adjusted, domain = self.dynamic_reg.adjust_params(neutral_prompt, self.base_params)
        
        self.assertIsNone(domain)
        self.assertEqual(adjusted["temperature"], 0.8)
        self.assertEqual(adjusted["top_p"], 0.95)


class TestGemmaToolRegistry(unittest.TestCase):
    def setUp(self):
        self.tool_reg = GemmaToolRegistry(chatbot_app=None)

    def test_modular_tool_registration(self):
        keys = self.tool_reg.registry.list_keys()
        self.assertIn("web_search", keys)
        self.assertIn("get_system_stats", keys)
        self.assertIn("read_file", keys)
        self.assertIn("control_rgb", keys)
        self.assertIn("generate_image", keys)

    def test_system_stats_execution(self):
        result = self.tool_reg.execute("get_system_stats", {})
        self.assertIn("cpu", result)
        self.assertIn("ram", result)

    def test_read_file_not_found(self):
        result = self.tool_reg.execute("read_file", {"path": "non_existent_file_12345.txt"})
        self.assertEqual(result, "Error: File not found.")

    def test_offline_mode_excludes_web_search(self):
        from System.network_guard import set_offline_mode
        try:
            set_offline_mode(True)
            defs = self.tool_reg.get_definitions(level=6)
            tool_names = [t["function"]["name"] for t in defs]
            self.assertNotIn("web_search", tool_names)
            self.assertIn("get_system_stats", tool_names)
            self.assertIn("read_file", tool_names)
        finally:
            set_offline_mode(False)


if __name__ == "__main__":
    unittest.main()
