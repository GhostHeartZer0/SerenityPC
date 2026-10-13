import unittest
from unittest.mock import patch, MagicMock
from System.diffusion_wrapper import DiffusionCLIWrapper

class TestDiffusionCLIWrapper(unittest.TestCase):
    def test_bos_eos_tokens(self):
        wrapper = DiffusionCLIWrapper(None, "diffusion-model.gguf", 33, 8192)
        self.assertEqual(wrapper.token_bos(), 2)
        self.assertEqual(wrapper.token_eos(), 1)
        self.assertEqual(wrapper.n_ctx, 4096)

    def test_chat_completion_prompt_format(self):
        wrapper = DiffusionCLIWrapper(None, "diffusion-model.gguf", 33, 4096)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "Tell me a story"}
        ]
        
        with patch.object(DiffusionCLIWrapper, '__call__', return_value=iter([{"choices": [{"text": "Once upon a time", "delta": {"content": "Once upon a time"}}]}])) as mock_call:
            res = wrapper.create_chat_completion(messages, stream=False)
            mock_call.assert_called_once()
            args = mock_call.call_args[0]
            prompt = args[1] if len(args) > 1 else args[0]
            self.assertIn("<start_of_turn>user\nHello<end_of_turn>\n", prompt)
            self.assertIn("<start_of_turn>assistant\nHi there!<end_of_turn>\n", prompt)
            self.assertIn("<start_of_turn>user\nTell me a story<end_of_turn>\n", prompt)
            self.assertTrue(prompt.endswith("<start_of_turn>model\n"))
            self.assertEqual(res["choices"][0]["message"]["content"], "Once upon a time")

    @patch("subprocess.Popen")
    @patch("shutil.which", return_value="llama-diffusion-cli.exe")
    def test_diffusion_cli_args(self, mock_which, mock_popen):
        mock_proc = MagicMock()
        mock_proc.stdout.read.side_effect = ["", ""]
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc

        wrapper = DiffusionCLIWrapper(None, "diffusion-model.gguf", 33, 8192)
        # Call with excess max_tokens to test clamping
        list(wrapper("Test prompt", max_tokens=2048))

        mock_popen.assert_called_once()
        cmd = mock_popen.call_args[0][0]
        
        # Check clamped max tokens
        self.assertIn("-n", cmd)
        idx_n = cmd.index("-n")
        self.assertEqual(cmd[idx_n + 1], "256")

        # Check context cap
        idx_c = cmd.index("-c")
        self.assertLessEqual(int(cmd[idx_c + 1]), 4096)

        # Check diffusion flags
        self.assertIn("--diffusion-eb-max-steps", cmd)
        self.assertEqual(cmd[cmd.index("--diffusion-eb-max-steps") + 1], "24")

        self.assertIn("--diffusion-eb-entropy-bound", cmd)
        self.assertEqual(cmd[cmd.index("--diffusion-eb-entropy-bound") + 1], "0.2")

        self.assertIn("--diffusion-steps", cmd)
        self.assertEqual(cmd[cmd.index("--diffusion-steps") + 1], "64")

        self.assertIn("--diffusion-kv-cache", cmd)
        self.assertEqual(cmd[cmd.index("--diffusion-kv-cache") + 1], "on")

if __name__ == "__main__":
    unittest.main()
