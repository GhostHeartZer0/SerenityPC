import unittest
from benchmarks.wringer.Wringer import WringerFramework

class TestWringerImprovements(unittest.TestCase):
    def setUp(self):
        self.wringer = WringerFramework(manual_grading=True)

    def test_split_thoughts_and_answer(self):
        raw = "<think>\nThinking through the calculus problem...\nStep 1: derivative\n</think>\nThe answer is 42."
        thought, answer = self.wringer.split_thoughts_and_answer(raw)
        self.assertIn("derivative", thought)
        self.assertEqual(answer, "The answer is 42.")

    def test_judge_scoring_isolation(self):
        # When judge outputs reasoning mentioning other numbers followed by score
        judge_output_with_reasoning = "<think>\nThe model followed 2 guidelines out of 3. Deducting 1 point.\n</think>\nScore: 8/10"
        _, clean = self.wringer.split_thoughts_and_answer(judge_output_with_reasoning)
        import re
        out_of_10 = re.findall(r"([0-9]*\.?[0-9]+)\s*(?:/|out of)\s*10", clean)
        self.assertEqual(out_of_10[-1], "8")

    def test_answer_key_coverage(self):
        self.assertIn("describe yourself.", self.wringer.answer_key)
        self.assertTrue(len(self.wringer.answer_key["describe yourself."]) > 10)

    def test_dynamic_gpu_layer_safety(self):
        # Non-existent file returns 0 safely
        layers = self.wringer.calculate_dynamic_gpu_layers("non_existent_file.gguf", 2048)
        self.assertEqual(layers, 0)

    def test_judge_structured_json_parsing(self):
        raw_judge_with_thoughts = (
            "<think>\nThe response is great, followed 3 steps, answered in 1 paragraph.\n</think>\n"
            "```json\n"
            '{\n  "critique": "Accurate and coherent explanation.",\n  "score": 9\n}\n'
            "```"
        )
        _, final_text = self.wringer.split_thoughts_and_answer(raw_judge_with_thoughts)
        import re, json
        # Attempt 1: regex
        json_match = re.search(r'\{\s*"critique".*?"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*\}', final_text, re.DOTALL | re.IGNORECASE)
        self.assertIsNotNone(json_match)
        score = float(json_match.group(1))
        self.assertEqual(score, 9.0)

    def test_judge_score_extraction_variations(self):
        import re, json

        def extract_score(judge_final_text):
            score_json_match = re.search(r'["\']score["\']\s*:\s*["\']?\s*([0-9]+(?:\.[0-9]+)?)(?:\s*(?:/|out of)\s*10)?\s*["\']?', judge_final_text, re.IGNORECASE)
            if score_json_match:
                try:
                    return float(score_json_match.group(1))
                except ValueError:
                    pass
            try:
                blob_match = re.search(r'\{.*\}', judge_final_text, re.DOTALL)
                if blob_match:
                    data = json.loads(blob_match.group(0))
                    if "score" in data:
                        s_val = str(data["score"]).strip().strip("'\"")
                        frac_m = re.match(r'^([0-9]+(?:\.[0-9]+)?)(?:\s*(?:/|out of)\s*10)?$', s_val)
                        if frac_m:
                            return float(frac_m.group(1))
                        return float(s_val)
            except Exception:
                pass
            return None

        # 1. Quoted integer
        self.assertEqual(extract_score('{"critique": "Solid", "score": "10"}'), 10.0)
        # 2. Fractional score "8/10"
        self.assertEqual(extract_score('{"critique": "Solid", "score": "8/10"}'), 8.0)
        # 3. Unescaped quotes in critique
        self.assertEqual(extract_score('{"critique": "Does not match "45140..." exactly", "score": "1"}'), 1.0)
        # 4. Score key before critique
        self.assertEqual(extract_score('{"score": 9, "critique": "Great"}'), 9.0)
        # 5. Placeholder fails safely
        self.assertIsNone(extract_score('{"critique": "<your 1-2 sentence justification>", "score": "<integer 1 to 10>"}'))

    def test_calibrated_reference_answers(self):
        # Level 5 philosophy reference calibrated
        entropy_prompt = [p for p in self.wringer.answer_key.keys() if "entropy dictates" in p.lower()][0]
        self.assertIn("Criteria: Reward deep, nuanced synthesis", self.wringer.answer_key[entropy_prompt])
        
        # Level 6 worldbuilding reference calibrated
        world_prompt = [p for p in self.wringer.answer_key.keys() if "type-ii civilization" in p.lower()][0]
        self.assertIn("Criteria: Reward detailed macro-engineering concepts", self.wringer.answer_key[world_prompt])

        # Level 5 big addition reference calibrated with comma note
        addition_prompt = [p for p in self.wringer.answer_key.keys() if "45976542378932137" in p][0]
        self.assertIn("45,140,762,792,838,499,557", self.wringer.answer_key[addition_prompt])

if __name__ == "__main__":
    unittest.main()
