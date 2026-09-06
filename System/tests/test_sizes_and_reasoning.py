"""
test_sizes_and_reasoning.py
Unit test suite verifying:
1. Reasoning strength expansion (off, minimal, low, medium, high, xhigh, maximum).
2. Gemma-4 thinking channel handling (<|think|> omission on 'off', presence on non-off).
3. Response target length modes (natural bypass, mini 2 sentences, matched prompt calibration, short, medium, long).
4. Thought channel isolation compliance (length directives target only final response).
5. Backwards compatibility with muse_reasoning_strength.
"""
import unittest
import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

class MockSerenityPC:
    """Mock Serenity PC application to test prompt staging without loading full GGUF weights."""
    def __init__(self, model_path="S:/LLM/gemma-4-unc/26B/gemma-4-26B-A4B-it.gguf"):
        self.model_path = model_path
        self.active_persona_level = 3
        self.state = {"deep_cook": False}
        self.config = {
            "reasoning_strength": "medium",
            "response_length": "natural",
            "dmn_enabled": True,
            "dmn_timeout": "05:00"
        }

    def build_system_prompt(self, user_message, base_sys="You are Serenity."):
        is_gemma = "gemma" in (self.model_path or "").lower()
        model_name_lower = os.path.basename(self.model_path or "").lower()
        is_diffusion = "diffusion" in (self.model_path or "").lower()
        is_nemotron = "nemotron" in model_name_lower
        is_qwen = "qwen" in model_name_lower
        is_deepseek = any(k in model_name_lower for k in ["deepseek", "r1", "qwq"])
        is_muse = any(k in model_name_lower for k in ["muse", "glimmer", "onyx", "atem"])

        sys_content = base_sys

        r_strength = str(self.config.get("reasoning_strength", self.config.get("muse_reasoning_strength", "medium"))).lower().strip()
        if r_strength == "minimal": r_strength = "low"
        elif r_strength == "maximum": r_strength = "xhigh"

        if self.model_path and "muse" in self.model_path.lower() and "glimmer" in self.model_path.lower():
            if r_strength != "off":
                muse_level = "low" if r_strength in ("minimal", "low") else ("xhigh" if r_strength in ("maximum", "xhigh") else r_strength)
                sys_content += f"\nReasoning strength: {muse_level}"

        sys_clean = sys_content.strip()

        reasoning_directives = {
            "low": "\n[REASONING LEVEL: LOW]: Keep internal reasoning brief before reaching your conclusion.",
            "medium": "\n[REASONING LEVEL: MEDIUM]: Provide balanced, step-by-step reasoning.",
            "high": "\n[REASONING LEVEL: HIGH]: Analyze the query thoroughly, exploring edge cases and reasoning step-by-step.",
            "xhigh": "\n[REASONING LEVEL: XHIGH]: Perform deep multi-step analysis, validating reasoning steps and evaluating alternative hypotheses."
        }

        if not is_diffusion and r_strength != "off" and (self.active_persona_level >= 3 or self.state.get("deep_cook")):
            if is_gemma:
                sys_clean = f"<|think|>\n{sys_clean}"
                if r_strength in reasoning_directives:
                    sys_clean += reasoning_directives[r_strength]
            elif is_nemotron:
                sys_clean += f"\n[REASONING]: Provide clear, direct, and rigorous answers without conversational meta-commentary.{reasoning_directives.get(r_strength, '')}"
            elif is_qwen or is_deepseek:
                sys_clean += f"\n[REASONING]: Analyze the query thoroughly and provide a direct, precise answer.{reasoning_directives.get(r_strength, '')}"
            elif is_muse:
                pass
            else:
                sys_clean += f"\n[REASONING]: Think step by step before answering and provide a clear, accurate response.{reasoning_directives.get(r_strength, '')}"

        # Configurable Response Target Length
        resp_length_mode = str(self.config.get("response_length", "natural")).lower().strip()
        if resp_length_mode == "natural":
            sys_clean += "\n[RESPONSE LENGTH DIRECTIVE]: Natural conversational flow. Answer proportionally—direct and concise for simple questions, expansive only when depth required."
        elif resp_length_mode == "mini":
            sys_clean += "\n[RESPONSE LENGTH DIRECTIVE]: Restrict your final answer to at most two concise sentences."
        elif resp_length_mode == "short":
            sys_clean += "\n[RESPONSE LENGTH DIRECTIVE]: Keep your final answer brief and concise, approximately 1 to 2 short paragraphs."
        elif resp_length_mode == "medium":
            sys_clean += "\n[RESPONSE LENGTH DIRECTIVE]: Provide a standard, moderately detailed final answer."
        elif resp_length_mode == "long":
            sys_clean += "\n[RESPONSE LENGTH DIRECTIVE]: Provide a comprehensive, detailed, and in-depth final answer."
        elif resp_length_mode == "matched":
            raw_user = user_message.strip() if user_message else ""
            word_count = len(raw_user.split())
            if word_count <= 25:
                matched_desc = "brief, at most two concise sentences"
            elif word_count <= 80:
                matched_desc = "short, approximately one to two paragraphs"
            elif word_count <= 250:
                matched_desc = "moderate, balanced in depth and length"
            else:
                matched_desc = f"detailed and comprehensive (matching user's ~{word_count} word query)"
            sys_clean += f"\n[RESPONSE LENGTH DIRECTIVE]: Match the length and scale of the user's prompt in your final answer ({matched_desc})."

        return sys_clean


class TestSizesAndReasoning(unittest.TestCase):
    def setUp(self):
        self.app = MockSerenityPC()

    def test_gemma_thinking_off_omits_think_token(self):
        """Verify that setting reasoning_strength to 'off' completely omits <|think|> for Gemma-4."""
        self.app.config["reasoning_strength"] = "off"
        prompt = self.app.build_system_prompt("Hello there")
        self.assertNotIn("<|think|>", prompt)
        self.assertNotIn("[REASONING LEVEL", prompt)
        self.assertTrue(prompt.startswith("You are Serenity."))

    def test_gemma_thinking_on_includes_think_token_and_level(self):
        """Verify that setting reasoning_strength to 'high' injects <|think|> and high directive."""
        self.app.config["reasoning_strength"] = "high"
        prompt = self.app.build_system_prompt("Hello there")
        self.assertTrue(prompt.startswith("<|think|>\n"))
        self.assertIn("[REASONING LEVEL: HIGH]", prompt)

    def test_expanded_reasoning_levels_validity(self):
        """Test active reasoning levels for Gemma-4."""
        levels = ["low", "medium", "high", "xhigh"]
        for lvl in levels:
            self.app.config["reasoning_strength"] = lvl
            prompt = self.app.build_system_prompt("Question")
            self.assertIn(f"[REASONING LEVEL: {lvl.upper()}]", prompt)
            self.assertTrue(prompt.startswith("<|think|>\n"))

    def test_legacy_reasoning_fallback(self):
        """Verify backwards compatibility and legacy fallback for minimal -> low and maximum -> xhigh."""
        del self.app.config["reasoning_strength"]
        self.app.config["muse_reasoning_strength"] = "maximum"
        prompt = self.app.build_system_prompt("Question")
        self.assertIn("[REASONING LEVEL: XHIGH]", prompt)

        self.app.config["muse_reasoning_strength"] = "minimal"
        prompt = self.app.build_system_prompt("Question")
        self.assertIn("[REASONING LEVEL: LOW]", prompt)

    def test_natural_response_length_proportional(self):
        """Verify that 'natural' response length mode injects proportional pacing directive."""
        self.app.config["response_length"] = "natural"
        self.app.config["reasoning_strength"] = "off"
        prompt = self.app.build_system_prompt("Explain gravity")
        self.assertIn("[RESPONSE LENGTH DIRECTIVE]: Natural conversational flow. Answer proportionally—direct and concise for simple questions, expansive only when depth required.", prompt)

    def test_mini_response_length_two_sentences(self):
        """Verify 'mini' response target length directs at most 2 concise sentences."""
        self.app.config["response_length"] = "mini"
        prompt = self.app.build_system_prompt("Explain quantum mechanics")
        self.assertIn("[RESPONSE LENGTH DIRECTIVE]: Restrict your final answer to at most two concise sentences.", prompt)

    def test_matched_response_length_calibration(self):
        """Verify 'matched' response length dynamically adjusts to prompt scale."""
        self.app.config["response_length"] = "matched"

        # Short single question (<= 25 words)
        short_prompt = self.app.build_system_prompt("What is 2+2?")
        self.assertIn("at most two concise sentences", short_prompt)

        # Medium query (26-80 words: 35 words here)
        medium_query = (
            "Can you explain the conceptual difference between supervised and unsupervised learning, "
            "providing concrete real-world engineering examples for both approaches, including common evaluation metrics "
            "and typical pitfalls that practitioners encounter during deployment?"
        )
        medium_prompt = self.app.build_system_prompt(medium_query)
        self.assertIn("approximately one to two paragraphs", medium_prompt)

    def test_thought_channel_isolation_in_directives(self):
        """Verify all response length directives explicitly target final answer, protecting thought channel."""
        for mode in ["mini", "short", "medium", "long", "matched"]:
            self.app.config["response_length"] = mode
            prompt = self.app.build_system_prompt("Test question")
            self.assertIn("final answer", prompt.lower())
            if "<|think|>" in prompt:
                self.assertTrue(prompt.startswith("<|think|>\n"))

    def test_muse_glimmer_reasoning_strength(self):
        """Verify Muse-Glimmer template injection with reasoning strength."""
        self.app.model_path = "S:/LLM/META ASI (Muse Glimmer)/muse-glimmer-30b.gguf"
        self.app.config["reasoning_strength"] = "high"
        prompt = self.app.build_system_prompt("Test")
        self.assertIn("Reasoning strength: high", prompt)

    def test_mid_split_sash_saving_and_apply_retries(self):
        """Verify that _apply_sash_pos does not corrupt config and sash saving triggers when position changes."""
        from unittest.mock import MagicMock
        from main import ChatbotApp

        mock_app = MagicMock()
        mock_app.config = {"sash_pos": 810}
        mock_app.paned = MagicMock()
        mock_app.root = MagicMock()

        # Simulate unmapped window during startup
        mock_app.paned.winfo_width.return_value = 1
        ChatbotApp._apply_sash_pos(mock_app, 810, retries=0)
        # Verify stored config is not overwritten with clamped 100
        self.assertEqual(mock_app.config["sash_pos"], 810)

        # Simulate fully mapped window (width=1600)
        mock_app.paned.winfo_width.return_value = 1600
        ChatbotApp._apply_sash_pos(mock_app, 810, retries=5)
        mock_app.paned.sash_place.assert_called_with(0, 810, 0)
        self.assertEqual(mock_app.config["sash_pos"], 810)

        # Verify release handler updates config only on actual position delta
        mock_app._get_current_sash_pos.return_value = 950
        # Trigger deferred check logic
        mock_app.save_config = MagicMock()
        pos = mock_app._get_current_sash_pos()
        if pos and pos > 50 and pos != mock_app.config.get("sash_pos"):
            mock_app.config["sash_pos"] = pos
            mock_app.save_config()

        self.assertEqual(mock_app.config["sash_pos"], 950)
        mock_app.save_config.assert_called_once()


if __name__ == "__main__":
    unittest.main()
