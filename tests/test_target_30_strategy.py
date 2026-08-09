import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "target_30", ROOT / "scripts" / "build_target_30_report.py"
)
TARGET_30 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(TARGET_30)


class Target30OverlayTests(unittest.TestCase):
    def sample(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "signal_date": ["20260101", "20260102"],
                "nav": [101.0, 102.01],
                "net_return": [0.01, 0.01],
                "gross_return": [0.012, 0.012],
                "benchmark_return": [0.001, -0.001],
                "sell_fee": [10.0, 10.0],
                "base_slippage_cost": [10.0, 10.0],
                "impact_cost": [20.0, 20.0],
            }
        )

    def test_one_times_overlay_uses_original_cost_model(self):
        result = TARGET_30.apply_exposure_overlay(
            self.sample(), leverage=1.0, annual_financing_rate=0.04
        )
        expected_cost = 40.0 / 100.0
        self.assertAlmostEqual(result.loc[0, "leveraged_cost_rate"], expected_cost)
        self.assertAlmostEqual(result.loc[0, "financing_rate"], 0.0)
        self.assertAlmostEqual(
            result.loc[0, "strategy_return"],
            (1.0 - expected_cost) * 1.012 - 1.0,
        )

    def test_impact_scales_non_linearly_and_financing_is_charged(self):
        leverage = 1.25
        result = TARGET_30.apply_exposure_overlay(
            self.sample(), leverage=leverage, annual_financing_rate=0.04
        )
        expected = leverage * 0.2 + leverage ** 1.5 * 0.2
        self.assertAlmostEqual(result.loc[0, "leveraged_cost_rate"], expected)
        self.assertAlmostEqual(
            result.loc[0, "financing_rate"], (leverage - 1.0) * 0.04 / 252
        )

    def test_max_drawdown_includes_initial_nav(self):
        self.assertAlmostEqual(
            TARGET_30.max_drawdown(np.array([0.9, 1.0, 0.8])), -0.2
        )


if __name__ == "__main__":
    unittest.main()
