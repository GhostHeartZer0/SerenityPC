import re
import unittest
from unittest.mock import MagicMock, patch
from System.diffusion_wrapper import DiffusionCLIWrapper


class TestDiffusionWrapper(unittest.TestCase):
    def test_bos_eos_tokens(self):
        w = DiffusionCLIWrapper(None, "dummy_model.gguf", 0, 2048)
        self.assertEqual(w.token_bos(), 2)
        self.assertEqual(w.token_eos(), 1)

    def test_gemma_chat_template(self):
        w = DiffusionCLIWrapper(None, "dummy_model.gguf", 0, 2048)
        messages = [
            {"role": "system", "content": "<|think|>\nYou are Serenity."},
            {"role": "user", "content": "Hello"}
        ]
        with patch.object(DiffusionCLIWrapper, '__call__', return_value=[{"choices": [{"text": "Hi", "delta": {"content": "Hi"}}]}]):
            res = w.create_chat_completion(messages, stream=False)
            self.assertEqual(res["choices"][0]["message"]["content"], "Hi")

    def test_ansi_escape_frame_splitting(self):
        raw_pattern = re.compile(r'(\x1b\[\d+A|\x1b\[2J\x1b\[H|\x1b\[2J|\x1b\[H)')
        ansi_escape_pattern = re.compile(r'\x1b\[[0-9;?]*[A-Za-z]')

        # Synthetic multi-frame buffer matching llama-diffusion-cli visual output
        raw_stream = (
            '\x1b[?25l\r\n\r\n\x1b[23A\x1b[?2026h\r\n <|channel>thought\x1b[K\r\nStep 1 denoising\x1b[K\r\n'
            '\x1b[23A\r\n <|channel>thought\x1b[K\r\nStep 2 denoising\x1b[K\r\n'
            '\x1b[23A\r\n\x1b[J\x1b[?25h\r\n<|channel>thought\r\nFinal reasoning<channel|>Final answer\r\n'
            'total time: 2500ms\r\nthroughput: 120 tok/s\r\n'
        )

        parts = raw_pattern.split(raw_stream)
        frames = []
        while len(parts) > 2:
            frame_raw = parts.pop(0)
            _sep = parts.pop(0)
            clean = ansi_escape_pattern.sub('', frame_raw).strip()
            lines = [l for l in clean.splitlines() if not any(k in l.lower() for k in ['total time:', 'throughput:', 'diffusion step:'])]
            txt = '\n'.join(lines).strip()
            if txt:
                frames.append(txt)
        if parts:
            clean = ansi_escape_pattern.sub('', parts[0]).strip()
            lines = [l for l in clean.splitlines() if not any(k in l.lower() for k in ['total time:', 'throughput:', 'diffusion step:'])]
            txt = '\n'.join(lines).strip()
            if txt:
                frames.append(txt)

        self.assertEqual(len(frames), 3)
        self.assertIn("Step 1 denoising", frames[0])
        self.assertIn("Step 2 denoising", frames[1])
        self.assertIn("Final reasoning<channel|>Final answer", frames[2])
        self.assertNotIn("total time", frames[2])
        self.assertNotIn("throughput", frames[2])


if __name__ == "__main__":
    unittest.main()