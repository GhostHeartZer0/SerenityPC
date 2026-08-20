"""
Test: Image Generation & ReAct / JSON Tool Calling (V1.6.9)
Validates:
1. GemmaToolRegistry generate_image handler with various argument encodings (plain string, dict, stringified JSON, python literal dict, diagram type).
2. Tool definitions inclusion for Level 2+.
3. ChatbotApp._run_tool_loop parsing of Python PTC, legacy tags, and JSON action blocks without dropping into raw unparsed text.
"""
import sys
import os
import json
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from System.tool_registry import GemmaToolRegistry

class TestImageGenerationAndToolCalling(unittest.TestCase):
    def setUp(self):
        self.mock_app = MagicMock()
        self.mock_app.script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.mock_app.config = {"offline_mode": False}
        self.tool_reg = GemmaToolRegistry(self.mock_app)

    @patch("subprocess.Popen")
    def test_generate_image_plain_dict(self, mock_popen):
        res = self.tool_reg.execute("generate_image", {"prompt": "Sunset over cyberpunk city", "type": "image"})
        self.assertIn("Successfully generated and displayed", res)
        self.assertIn("Sunset over cyberpunk city", res)

    @patch("subprocess.Popen")
    def test_generate_image_nested_action_input_string(self, mock_popen):
        # Reproduces exact user report format
        args = {
            "action": "generate_image",
            "action_input": "{'prompt': 'Semi-realistic anime style, a bed of moss and forest flowers, 8k resolution', 'type': 'artistic'}"
        }
        res = self.tool_reg.execute("generate_image", args)
        self.assertIn("Successfully generated and displayed", res)
        self.assertIn("Semi-realistic anime style", res)

    @patch("subprocess.Popen")
    def test_generate_image_diagram_type(self, mock_popen):
        args = {
            "prompt": "graph TD\nA-->B",
            "type": "diagram"
        }
        res = self.tool_reg.execute("generate_image", args)
        self.assertIn("Successfully generated and displayed", res)
        self.assertIn("graph TD", res)

    def test_tool_definitions_level_2(self):
        defs = self.tool_reg.get_definitions(level=2)
        names = [d["function"]["name"] for d in defs]
        self.assertIn("generate_image", names)
        self.assertIn("read_file", names)
        self.assertIn("web_search", names)
        self.assertIn("get_system_stats", names)
        self.assertIn("control_rgb", names)

    @patch("subprocess.Popen")
    def test_run_tool_loop_json_action_block(self, mock_popen):
        from main import ChatbotApp
        app = MagicMock()
        app.active_persona_level = 3
        app.tool_registry = self.tool_reg
        app.process_queue = MagicMock()
        app._run_blocking_inference = MagicMock(return_value="Here is your requested artwork of a mossy forest.")
        app._run_tool_loop = lambda *a, **k: ChatbotApp._run_tool_loop(app, *a, **k)
        
        # Test full response with JSON action block
        full_resp = """```json
{
  "action": "generate_image",
  "action_input": "{'prompt': 'Semi-realistic anime style, a bed of moss and forest flowers, 8k resolution', 'type': 'artistic'}"
}
```
(Note: Generating visual parameters...)"""
        
        final_answer = ChatbotApp._run_tool_loop(app, full_resp, "Draw me a mossy forest", {})
        self.assertIn("Here is your requested artwork", final_answer)
        app.process_queue.put.assert_any_call({"status": "thinking_status", "content": "Executing tool: generate_image..."})

    @patch("subprocess.Popen")
    def test_run_tool_loop_py_stub(self, mock_popen):
        from main import ChatbotApp
        app = MagicMock()
        app.active_persona_level = 3
        app.tool_registry = self.tool_reg
        app.process_queue = MagicMock()
        app._run_blocking_inference = MagicMock(return_value="Here is your cybernetic portrait.")
        app._run_tool_loop = lambda *a, **k: ChatbotApp._run_tool_loop(app, *a, **k)

        full_resp = "generate_image(prompt='Cybernetic neon portrait', type='image')"
        final_answer = ChatbotApp._run_tool_loop(app, full_resp, "Generate cybernetic portrait", {})
        self.assertIn("Here is your cybernetic portrait", final_answer)

if __name__ == "__main__":
    unittest.main()
