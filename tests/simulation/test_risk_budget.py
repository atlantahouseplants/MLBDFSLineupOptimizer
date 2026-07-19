from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.simulation.risk_budget import RISK_BUDGET_PROFILES, apply_risk_budget_to_state


class TestRiskBudget(unittest.TestCase):
    def test_profile_updates_multiple_weights(self) -> None:
        state = {"selection_duplication_weight": 0.0, "risk_profile": "Balanced leverage"}
        apply_risk_budget_to_state(state, "Extreme top-heavy GPP")

        self.assertEqual(state["risk_profile"], "Extreme top-heavy GPP")
        self.assertGreater(state["selection_duplication_weight"], 0.3)
        self.assertIn("Balanced leverage", RISK_BUDGET_PROFILES)


if __name__ == "__main__":
    unittest.main()
