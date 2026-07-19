from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.simulation.contest_type import apply_contest_type_to_state, recommend_contest_type_profile
from slate_optimizer.simulation.payouts import PayoutBand


class TestContestTypeBrain(unittest.TestCase):
    def test_auto_recommends_massive_gpp_for_300_max_large_field(self) -> None:
        profile = recommend_contest_type_profile(
            contest_entries=50_000,
            num_lineups=300,
            payout_bands=[PayoutBand(1, 1, 100000.0), PayoutBand(2, 10000, 10.0)],
        )
        self.assertIn(profile.name, {"Massive 150/300-max GPP", "Top-heavy large-field GPP"})

    def test_apply_updates_selection_weights(self) -> None:
        state = {"contest_entries": 50_000, "num_candidates": 300}
        profile = apply_contest_type_to_state(state, "Massive 150/300-max GPP")

        self.assertEqual(profile.name, "Massive 150/300-max GPP")
        self.assertGreater(float(state["selection_duplication_weight"]), 0.20)
        self.assertTrue(state["use_contest_type_brain"])


if __name__ == "__main__":
    unittest.main()
