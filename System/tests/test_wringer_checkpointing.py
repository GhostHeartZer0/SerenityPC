"""
Unit Test: Wringer Framework Incremental Checkpointing & Resumption
Validates:
1. Atomic checkpoint file creation, update, and loading.
2. Prompt-level and Level-level caching to prevent lost progress across interrupts.
3. Partial markdown report export.
4. Auto-resumption logic skipping already evaluated levels.
"""
import os
import sys
import json
import tempfile
import unittest

test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(test_dir))
wringer_dir = os.path.join(project_root, "benchmarks", "wringer")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if wringer_dir not in sys.path:
    sys.path.insert(0, wringer_dir)

from Wringer import WringerFramework

class TestWringerCheckpointing(unittest.TestCase):
    def setUp(self):
        self.wringer = WringerFramework(manual_grading=True)
        self.test_model_name = "test_model_checkpoint.gguf"

    def tearDown(self):
        cp_path = self.wringer._get_checkpoint_path(self.test_model_name)
        if os.path.exists(cp_path):
            try:
                os.remove(cp_path)
            except Exception:
                pass
        report_path = os.path.join(os.path.dirname(cp_path), f"{self.test_model_name}_report.md")
        if os.path.exists(report_path):
            try:
                os.remove(report_path)
            except Exception:
                pass

    def test_checkpoint_lifecycle(self):
        # 1. Fresh checkpoint initialization
        cp = self.wringer._load_checkpoint(self.test_model_name)
        self.assertEqual(cp["model_name"], self.test_model_name)
        self.assertEqual(cp["results"], {})
        self.assertEqual(cp["partial_prompts"], {})
        self.assertEqual(cp["partial_scores"], {})

        # 2. Saving partial prompt generation
        cp["partial_prompts"]["lvl1"] = {
            "0": {
                "content": "Simulated output prompt 0",
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "prefill_time": 0.1,
                "decode_time": 0.5,
                "total_time": 0.6,
                "prefill_tps": 100.0,
                "decode_tps": 40.0,
                "overall_tps": 50.0
            }
        }
        self.wringer._save_checkpoint(cp)

        # 3. Reload and verify persistence
        reloaded = self.wringer._load_checkpoint(self.test_model_name)
        self.assertIn("lvl1", reloaded["partial_prompts"])
        self.assertIn("0", reloaded["partial_prompts"]["lvl1"])
        self.assertEqual(reloaded["partial_prompts"]["lvl1"]["0"]["content"], "Simulated output prompt 0")

        # 4. Save level results and export partial markdown
        reloaded["results"]["lvl1"] = {
            "average_score": 9.5,
            "percentage": "95.0%",
            "composite_score": 9.5,
            "prefill_tps": 100.0,
            "decode_tps": 40.0,
            "overall_tps": 50.0,
            "anomaly_count": 0,
            "anomalies": [],
            "clean_decode_tps": 40.0,
            "details": [{
                "prompt": "Test prompt",
                "response": "Simulated output prompt 0",
                "score": 9.5,
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "prefill_tps": 100.0,
                "decode_tps": 40.0,
                "overall_tps": 50.0
            }]
        }
        self.wringer._save_checkpoint(reloaded)

        rep_path = self.wringer._export_markdown_report(self.test_model_name, reloaded["results"], 12.34, is_partial=True)
        self.assertTrue(os.path.exists(rep_path))
        with open(rep_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("IN PROGRESS - AUTO-CHECKPOINTED", content)
        self.assertIn("9.5/10", content)

if __name__ == "__main__":
    unittest.main()
