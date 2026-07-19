from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from dashboard.daily_workflow import _auto_detect_projection_paths


class TestDailyWorkflowUploads(unittest.TestCase):
    def test_auto_detects_ballpark_optimizer_projection_workbook(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bpp_dir = Path(tmpdir)
            optimizer_path = bpp_dir / "Ballpark DFS Optimizer  Ballpark Pal.xlsx"
            standard_path = bpp_dir / "BallparkPal_Batters.xlsx"
            source = pd.DataFrame(
                [
                    {
                        "Tm": "AAA",
                        "Players": "Hitter One",
                        "Points": 10.0,
                        "Bust": 0.20,
                        "Median": 8.0,
                        "Upside": 24.0,
                    }
                ]
            )
            with pd.ExcelWriter(optimizer_path) as writer:
                source.to_excel(writer, index=False, startrow=1)
                writer.sheets["Sheet1"].cell(row=1, column=1).value = "Ballpark DFS Optimizer | Ballpark Pal"
            pd.DataFrame([{"FullName": "Hitter One", "PointsFD": 10.0}]).to_excel(standard_path, index=False)

            detected = _auto_detect_projection_paths(bpp_dir, [])

        self.assertEqual([path.name for path in detected], [optimizer_path.name])


if __name__ == "__main__":
    unittest.main()
