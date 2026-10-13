import unittest
import os
import tempfile
import struct
import queue
from System.serenity_utils import patch_gguf_architecture

class TestGgufAndQueueHygiene(unittest.TestCase):
    def test_patch_gguf_architecture_pre_tokenizer(self):
        # Create a mock minimal GGUF binary with minicpm5 pre-tokenizer
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            temp_path = f.name
            f.write(b"GGUF")
            f.write(struct.pack("<I", 3)) # version
            f.write(struct.pack("<Q", 0)) # tensor count
            f.write(struct.pack("<Q", 2)) # kv count

            # KV 1: general.architecture = llama
            k1 = b"general.architecture"
            f.write(struct.pack("<Q", len(k1)))
            f.write(k1)
            f.write(struct.pack("<I", 8)) # string type
            val1 = b"llama"
            f.write(struct.pack("<Q", len(val1)))
            f.write(val1)

            # KV 2: tokenizer.ggml.pre = minicpm5
            k2 = b"tokenizer.ggml.pre"
            f.write(struct.pack("<Q", len(k2)))
            f.write(k2)
            f.write(struct.pack("<I", 8))
            val2 = b"minicpm5"
            f.write(struct.pack("<Q", len(val2)))
            f.write(val2)

        try:
            patched = patch_gguf_architecture(temp_path)
            self.assertTrue(patched)

            # Verify that minicpm5 was patched to default
            with open(temp_path, "rb") as f:
                content = f.read()
                self.assertIn(b"default", content)
                self.assertNotIn(b"minicpm5", content)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_gemma4_architecture_preserved(self):
        # Create a mock minimal GGUF binary with gemma4 architecture
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as f:
            temp_path = f.name
            f.write(b"GGUF")
            f.write(struct.pack("<I", 3)) # version
            f.write(struct.pack("<Q", 0)) # tensor count
            f.write(struct.pack("<Q", 2)) # kv count

            # KV 1: general.architecture = gemma4
            k1 = b"general.architecture"
            f.write(struct.pack("<Q", len(k1)))
            f.write(k1)
            f.write(struct.pack("<I", 8)) # string type
            val1 = b"gemma4"
            f.write(struct.pack("<Q", len(val1)))
            f.write(val1)

            # KV 2: gemma4.block_count = 32
            k2 = b"gemma4.block_count"
            f.write(struct.pack("<Q", len(k2)))
            f.write(k2)
            f.write(struct.pack("<I", 4)) # uint32
            f.write(struct.pack("<I", 32))

        try:
            # Calling with default new_arch="llama" must NOT overwrite gemma4
            patched = patch_gguf_architecture(temp_path)
            self.assertFalse(patched)

            with open(temp_path, "rb") as f:
                content = f.read()
                self.assertIn(b"gemma4", content)
                self.assertNotIn(b"llama", content)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_corrupted_gemma_auto_healed(self):
        # Create a mock GGUF that was previously corrupted to llama
        with tempfile.NamedTemporaryFile(prefix="gemma-4-", suffix=".gguf", delete=False) as f:
            temp_path = f.name
            f.write(b"GGUF")
            f.write(struct.pack("<I", 3)) # version
            f.write(struct.pack("<Q", 0)) # tensor count
            f.write(struct.pack("<Q", 3)) # kv count

            # KV 1: general.architecture = llama\x00 (corrupted from 6-byte gemma4)
            k1 = b"general.architecture"
            f.write(struct.pack("<Q", len(k1)))
            f.write(k1)
            f.write(struct.pack("<I", 8)) # string type
            val1 = b"llama\x00"
            f.write(struct.pack("<Q", len(val1)))
            f.write(val1)

            # KV 2: tokenizer.ggml.model = gemma4
            k2 = b"tokenizer.ggml.model"
            f.write(struct.pack("<Q", len(k2)))
            f.write(k2)
            f.write(struct.pack("<I", 8))
            val2 = b"gemma4"
            f.write(struct.pack("<Q", len(val2)))
            f.write(val2)

            # KV 3: llama.block_count\x00 (corrupted from 18-byte gemma4.block_count)
            k3 = b"llama.block_count\x00"
            f.write(struct.pack("<Q", len(k3)))
            f.write(k3)
            f.write(struct.pack("<I", 4)) # uint32
            f.write(struct.pack("<I", 32))

        try:
            # Calling patch_gguf_architecture should auto-heal to gemma4
            patched = patch_gguf_architecture(temp_path)
            self.assertTrue(patched)

            with open(temp_path, "rb") as f:
                content = f.read()
                self.assertIn(b"gemma4.block_count", content)
                self.assertNotIn(b"llama.block_count", content)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_queue_drain(self):
        q = queue.Queue()
        for i in range(10):
            q.put({"status": "streaming", "content": str(i)})
        self.assertEqual(q.qsize(), 10)

        # Drain
        while not q.empty():
            q.get_nowait()
            q.task_done()
        self.assertEqual(q.qsize(), 0)

if __name__ == "__main__":
    unittest.main()

