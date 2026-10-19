import os
import sys
import unittest
import tempfile
import shutil
import json
import zlib

from Tools.repair_history import (
    extract_pipeline_levels,
    clean_baked_in_prefix,
    repair_file_messages,
    run_history_repair
)


class TestRepairHistory(unittest.TestCase):
    def test_extract_pipeline_levels(self):
        self.assertEqual(extract_pipeline_levels("model_lvl1.history.jsonz"), [1])
        self.assertEqual(extract_pipeline_levels("model_lvl7.history.encz"), [7])
        self.assertEqual(extract_pipeline_levels("model_lvls1_3_5.history.jsonz"), [1, 3, 5])
        self.assertEqual(extract_pipeline_levels("unknown_file.txt"), [])

    def test_clean_baked_in_prefix(self):
        # Assistant prefixes
        cleaned, changed = clean_baked_in_prefix("Serenity: Hello world", "assistant")
        self.assertTrue(changed)
        self.assertEqual(cleaned, "Hello world")

        cleaned, changed = clean_baked_in_prefix("Cecilia: Analysis complete", "assistant")
        self.assertTrue(changed)
        self.assertEqual(cleaned, "Analysis complete")

        # User prefixes
        cleaned, changed = clean_baked_in_prefix("You: What is gravity?", "user")
        self.assertTrue(changed)
        self.assertEqual(cleaned, "What is gravity?")

        # Untouched normal text
        cleaned, changed = clean_baked_in_prefix("Normal response without prefix", "assistant")
        self.assertFalse(changed)
        self.assertEqual(cleaned, "Normal response without prefix")

    def test_repair_file_messages_sub7(self):
        """Tests that sub-7 history file correctly restores Serenity and fixes corrupted Cecilia tags."""
        msgs = [
            {"role": "user", "content": "You: help me"},
            {"role": "assistant", "content": "Serenity: I am Serenity.", "persona": "Cecilia", "level": 7},
            {"role": "user", "content": "more info"},
            {"role": "cecilia", "content": "additional text"}
        ]
        repaired, changes, logs = repair_file_messages(msgs, "model_lvl3.history.jsonz")
        self.assertGreater(changes, 0)

        # Turn 0 (user)
        self.assertEqual(repaired[0]["content"], "help me")

        # Turn 1 (assistant restored to Serenity Lvl 3)
        self.assertEqual(repaired[1]["persona"], "Serenity")
        self.assertEqual(repaired[1]["level"], 3)
        self.assertEqual(repaired[1]["content"], "I am Serenity.")

        # Turn 3 (role 'cecilia' grounded to assistant)
        self.assertEqual(repaired[3]["role"], "assistant")
        self.assertEqual(repaired[3]["persona"], "Serenity")
        self.assertEqual(repaired[3]["level"], 3)

    def test_repair_file_messages_lvl7(self):
        """Tests that Level 7 history file correctly grounds to Cecilia."""
        msgs = [
            {"role": "user", "content": "calculate"},
            {"role": "assistant", "content": "result", "persona": "Serenity", "level": 3}
        ]
        repaired, changes, logs = repair_file_messages(msgs, "model_lvl7.history.jsonz")
        self.assertGreater(changes, 0)
        self.assertEqual(repaired[1]["persona"], "Cecilia")
        self.assertEqual(repaired[1]["level"], 7)

    def test_run_history_repair_e2e(self):
        """Tests full scan, backup creation, and in-place repair on temporary directory."""
        temp_dir = tempfile.mkdtemp()
        try:
            sample_file = os.path.join(temp_dir, "test_model_lvl2.history.jsonz")
            test_msgs = [
                {"role": "user", "content": "You: test"},
                {"role": "assistant", "content": "Cecilia: output", "persona": "Cecilia", "level": 7}
            ]
            with open(sample_file, "wb") as fp:
                fp.write(zlib.compress(json.dumps(test_msgs).encode("utf-8")))

            stats = run_history_repair(target_dirs=[temp_dir], dry_run=False)
            self.assertEqual(stats["files_scanned"], 1)
            self.assertEqual(stats["files_repaired"], 1)
            self.assertGreater(stats["total_changes"], 0)
            self.assertIsNotNone(stats["backup_dir"])

            # Verify repaired file on disk
            with open(sample_file, "rb") as fp:
                data = json.loads(zlib.decompress(fp.read()).decode("utf-8"))
            self.assertEqual(data[0]["content"], "test")
            self.assertEqual(data[1]["persona"], "Serenity")
            self.assertEqual(data[1]["level"], 2)
            self.assertEqual(data[1]["content"], "output")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
