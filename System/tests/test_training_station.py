import os
import sys
import unittest
from pathlib import Path
import pytest

# Add project root and training station to sys.path
BASE_DIR = Path(__file__).resolve().parents[2]
STATION_DIR = BASE_DIR / "benchmarks" / "training station"
if not STATION_DIR.exists():
    pytest.skip("Training Station migrated to Environment M", allow_module_level=True)
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(STATION_DIR) not in sys.path:
    sys.path.insert(0, str(STATION_DIR))

from harvesters.kiwix_zim_harvester import KiwixZimHarvester
from harvesters.game_lore_harvester import GameLoreHarvester
from harvesters.dmn_dream_harvester import DmnDreamHarvester
from crucible.solver_rollout import SolverRollout
from crucible.wringer_gate import WringerGate
from crucible.style_transplanter import StyleTransplanter
from cartridges.game_assistants.gamerenity_core import GamerenityGuideManager

class TestTrainingStation(unittest.TestCase):

    def test_zim_harvester_discovery(self):
        harvester = KiwixZimHarvester("S:/Wikiwiki")
        zims = harvester.list_available_zims()
        self.assertIsInstance(zims, list)
        if zims:
            self.assertTrue(any("zim" in z["filename"].lower() for z in zims))

    def test_thought_isolation_in_solver_rollout(self):
        raw_output = "<think>\nComputing trajectory delta...\nDone.\n</think>\nFinal Answer: 42"
        res = SolverRollout.isolate_thought_and_response(raw_output)
        self.assertIn("Computing trajectory delta", res["thought"])
        self.assertEqual("Final Answer: 42", res["response"])
        self.assertNotIn("<think>", res["response"])
        self.assertNotIn("Computing trajectory delta", res["response"])

    def test_wringer_zpd_gate(self):
        gate = WringerGate(min_zpd=2.0, max_zpd=8.0)
        candidates = [
            {"prompt": "p1", "score": 9.5},  # too easy
            {"prompt": "p2", "score": 6.0},  # valid ZPD
            {"prompt": "p3", "score": 1.0},  # too hard/broken
            {"prompt": "p4", "score": 4.5},  # valid ZPD
        ]
        kept = gate.filter_zpd_candidates(candidates)
        self.assertEqual(len(kept), 2)
        self.assertEqual([c["prompt"] for c in kept], ["p2", "p4"])

    def test_game_lore_categorization_and_compilation(self):
        harvester = GameLoreHarvester()
        self.assertEqual(harvester.classify_text_category("Pulse engine jump to warp station"), "flight")
        self.assertEqual(harvester.classify_text_category("Crit rate 75% with elemental burst scaling"), "combat")
        self.assertEqual(harvester.classify_text_category("Hold Shift to sprint and press Space to glide"), "controls")

    def test_dmn_dream_harvester_read(self):
        d_harvester = DmnDreamHarvester()
        misses = d_harvester.harvest_rlhf_misses()
        self.assertIsInstance(misses, list)

if __name__ == "__main__":
    unittest.main()

