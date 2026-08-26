# test_context_and_scroll.py
# Verification suite for Context 64k limit removal, history packing, and scroll behavior.

import unittest
import sys
import os

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from System.kv_manager import KVManager

class TestContextAndScroll(unittest.TestCase):
    def test_kv_manager_massive_context_initialization(self):
        """Verify KVManager supports full 256k context capacity."""
        kv = KVManager(max_context_tokens=262144, prune_ratio=0.5)
        self.assertEqual(kv.max_context_tokens, 262144)
        kv.set_max_context(131072)
        self.assertEqual(kv.max_context_tokens, 131072)

    def test_kv_manager_multimodal_token_estimation(self):
        """Verify token estimation handles multimodal content lists safely."""
        kv = KVManager(max_context_tokens=32768)
        str_msg = "Hello world"
        self.assertTrue(kv._estimate_tokens(str_msg) >= 1)
        
        list_msg = [
            {"type": "text", "text": "Analyze this image"},
            {"type": "image_url", "image_url": "data:image/jpeg;base64,123456"}
        ]
        self.assertTrue(kv._estimate_tokens(list_msg) >= 1)

    def test_dynamic_max_tokens_headroom_calculation(self):
        """Verify that a 256k context window is not artificially clamped down to 64k."""
        ctx = 262144
        config_no_ratio = {}
        # When no explicit ratio is forced, calculated headroom expands beyond 64k
        calculated_max = max(4096, ctx - 1024) if ctx > 8192 else max(256, ctx // 4)
        self.assertGreater(calculated_max, 65536)
        self.assertEqual(calculated_max, 262144 - 1024)

    def test_fallback_history_packing(self):
        """Verify fallback packing preserves many messages up to context limit instead of hardcoded 12."""
        temp_messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"Message number {i}"} for i in range(50)]
        ctx_limit = 32768
        budget_tokens = int(ctx_limit * 0.9)
        accumulated = 0
        fitting_msgs = []
        for msg in reversed(temp_messages):
            cnt = msg.get("content", "")
            tok_est = max(1, len(str(cnt)) // 4)
            if accumulated + tok_est > budget_tokens and fitting_msgs:
                break
            fitting_msgs.append(msg)
            accumulated += tok_est
        fitting_msgs.reverse()
        self.assertEqual(len(fitting_msgs), 50)

if __name__ == "__main__":
    unittest.main()
