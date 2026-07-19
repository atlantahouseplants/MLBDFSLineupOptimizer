from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from slate_optimizer.simulation.payouts import (
    parse_payout_ladder_dataframe,
    parse_payout_ladder_text,
    payout_from_percentiles,
    ranks_from_percentiles,
)


class TestPayouts(unittest.TestCase):
    def test_rank_mapping_and_money_parsing(self) -> None:
        bands = parse_payout_ladder_text("1,$100,000\n2-10,$5,000\n11-100,100")
        self.assertEqual(bands[0].payout, 100000)

        percentiles = np.array([100.0, 99.99, 99.0, 50.0])
        ranks = ranks_from_percentiles(percentiles, contest_entries=50000)
        payouts = payout_from_percentiles(percentiles, bands, contest_entries=50000)

        self.assertEqual(int(ranks[0]), 1)
        self.assertEqual(float(payouts[0]), 100000)
        self.assertGreaterEqual(float(payouts[1]), 5000)
        self.assertEqual(float(payouts[-1]), 0)

    def test_fanduel_style_dataframe_import(self) -> None:
        payout_df = pd.DataFrame(
            {
                "Place": ["1st", "2 - 3", "4th-10th", "11-100"],
                "Prize": ["$100,000", "$25,000", "$2,500", "$100"],
            }
        )
        bands = parse_payout_ladder_dataframe(payout_df)

        self.assertEqual(len(bands), 4)
        self.assertEqual(bands[0].min_rank, 1)
        self.assertEqual(bands[1].max_rank, 3)
        self.assertEqual(bands[2].payout, 2500)


if __name__ == "__main__":
    unittest.main()
