from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import unittest

import numpy as np

from slate_optimizer.simulation import (
    antithetic_uniforms,
    control_variate_adjustment,
    stratified_uniforms,
)


class TestVarianceReduction(unittest.TestCase):
    def test_antithetic_uniforms(self) -> None:
        base = np.linspace(0.1, 0.9, 10).reshape(-1, 1)
        anti = antithetic_uniforms(base)
        self.assertEqual(anti.shape, (20, 1))
        self.assertTrue(np.allclose(anti[:10], base))
        self.assertTrue(np.allclose(anti[10:], 1 - base))

    def test_stratified_uniforms(self) -> None:
        rng = np.random.default_rng(0)
        samples = stratified_uniforms(num_strata=5, num_per_stratum=2, num_players=3, rng=rng)
        self.assertEqual(samples.shape, (10, 3))
        self.assertTrue(np.all((samples >= 0) & (samples <= 1)))
        # ensure stratification covers range
        self.assertGreater(samples.max(), 0.8)
        self.assertLess(samples.min(), 0.2)

    def test_control_variate_adjustment(self) -> None:
        estimates = np.array([1.0, 1.2, 0.8])
        control_est = np.array([0.9, 1.1, 0.7])
        target = 1.0
        adjusted = control_variate_adjustment(estimates, control_est, target)
        self.assertEqual(adjusted.shape, estimates.shape)
        self.assertLess(np.var(adjusted), np.var(estimates))


if __name__ == "__main__":
    unittest.main()
