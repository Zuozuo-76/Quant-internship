from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline_code"))

from analyze_factors_backtest import (
    build_constrained_target,
    equal_weight_benchmark_return,
    estimate_trade_costs,
    factor_signal_scale,
    select_buffered_codes,
)
from import_trade_data import summarize_stock, trading_minutes
from build_kline_csv import FIELDS, build_daily_kline
from evaluate_factors import compute_factor_correlation, neutralize_cross_section
from train_lstm_pytorch import (
    TorchLSTM,
    make_walk_forward_folds,
    metrics_from_probability,
    sequence_summary_features,
)


class ImporterTests(unittest.TestCase):
    def test_trading_minute_grid(self):
        minutes = trading_minutes()
        self.assertEqual(len(minutes), 253)
        self.assertEqual(len(set(minutes)), 253)
        self.assertEqual(minutes[:3], [915, 916, 917])
        self.assertIn(925, minutes)
        self.assertIn(1130, minutes)
        self.assertIn(1300, minutes)
        self.assertEqual(minutes[-1], 1500)
        self.assertNotIn(1200, minutes)

    def test_tick_aggregation_and_adjustment(self):
        ticks = pd.DataFrame({
            "Time": [92500000, 93000000, 93030000],
            "Price": [1000, 1100, 900],
            "Volume": [100, 200, 300],
            "BSFlag": [2, 0, 1],
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "000001.csv"
            ticks.to_csv(path, index=False)
            daily, minute = summarize_stock(path, adjustment=2.0)
        self.assertEqual(daily["open"], 20.0)
        self.assertEqual(daily["high"], 22.0)
        self.assertEqual(daily["low"], 18.0)
        self.assertEqual(daily["close"], 18.0)
        self.assertEqual(daily["volume"], 600)
        self.assertEqual(daily["trade_count"], 3)
        self.assertEqual(daily["amount"], 5900.0)
        self.assertEqual(daily["buy_volume"], 200)
        self.assertEqual(daily["sell_volume"], 300)
        self.assertEqual(minute.loc[930, "open"], 22.0)
        self.assertEqual(minute.loc[930, "close"], 18.0)
        self.assertEqual(minute.loc[930, "volume"], 500)

    def test_daily_kline_csv_preserves_codes_and_field_order(self):
        dates = ["20250102", "20250103"]
        codes = ["000001", "600000"]
        field_values = {
            "open": [[10.0, 20.0], [11.0, 19.0]],
            "high": [[12.0, 21.0], [12.0, 20.0]],
            "low": [[9.0, 18.0], [10.0, 18.0]],
            "close": [[11.0, 19.0], [10.5, 18.5]],
            "volume": [[100, 200], [120, 210]],
            "trade_count": [[2, 3], [2, 4]],
            "amount": [[1050.0, 3900.0], [1290.0, 3885.0]],
            "buy_volume": [[60, 110], [70, 100]],
            "sell_volume": [[40, 90], [50, 110]],
            "buy_amount": [[630.0, 2145.0], [752.5, 1850.0]],
            "sell_amount": [[420.0, 1755.0], [537.5, 2035.0]],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "daily"
            input_dir.mkdir()
            for field in FIELDS:
                table = pd.DataFrame(field_values[field], columns=codes)
                table.insert(0, "date", dates)
                table.to_csv(input_dir / f"{field}.csv", index=False)
            output = root / "daily_kline.csv"
            summary = build_daily_kline(input_dir, output)
            result = pd.read_csv(output, dtype={"date": str, "code": str})

        self.assertEqual(summary["rows"], 4)
        self.assertEqual(result.columns.tolist(), ["date", "code", *FIELDS])
        self.assertEqual(result["code"].tolist(), ["000001", "600000", "000001", "600000"])
        self.assertEqual(result.iloc[0]["open"], 10.0)


class LSTMTests(unittest.TestCase):
    def test_pytorch_forward_backward_are_finite(self):
        torch.manual_seed(3)
        generator = torch.Generator().manual_seed(7)
        x = torch.randn(8, 5, 3, generator=generator)
        y = torch.randint(0, 2, (8,), generator=generator).float()
        model = TorchLSTM(input_size=3, hidden_size=4)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss = torch.nn.BCEWithLogitsLoss()(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        gradients = [parameter.grad for parameter in model.parameters()]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all().item() for gradient in gradients))
        before = model.lstm.weight_ih_l0.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.lstm.weight_ih_l0))

    def test_walk_forward_folds_have_non_overlapping_oos_windows(self):
        dates = pd.date_range("2025-01-01", periods=120, freq="D").strftime("%Y%m%d").tolist()
        folds = make_walk_forward_folds(
            dates,
            min_train_dates=60,
            validation_dates=20,
            test_dates=20,
            step_dates=20,
        )
        self.assertEqual(len(folds), 2)
        first_test = set(folds[0]["test_dates"])
        second_test = set(folds[1]["test_dates"])
        self.assertFalse(first_test & second_test)
        self.assertLess(folds[0]["train_dates"][-1], folds[0]["validation_dates"][0])
        self.assertLess(folds[0]["validation_dates"][-1], folds[0]["test_dates"][0])

    def test_classification_metrics_and_logistic_features(self):
        y = np.array([0, 0, 1, 1], dtype=float)
        probability = np.array([0.1, 0.8, 0.7, 0.2], dtype=float)
        metrics = metrics_from_probability(y, probability)
        self.assertEqual((metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]), (1, 1, 1, 1))
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 0.5)
        sequences = np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3)
        self.assertEqual(sequence_summary_features(sequences).shape, (2, 9))


class RiskAndExecutionTests(unittest.TestCase):
    def test_exploratory_factor_signal_scale_is_conservative(self):
        self.assertEqual(factor_signal_scale("amount_mean_sd_log"), 1.0)
        self.assertEqual(
            factor_signal_scale("avg_trade_size_surprise_20d"), 0.10
        )

    def test_rank_buffer_keeps_existing_names_inside_exit_band(self):
        ranked = pd.DataFrame({
            "code": [f"S{rank:02d}" for rank in range(1, 21)],
            "rank": list(range(1, 21)),
        })
        selected = select_buffered_codes(
            ranked,
            {"S11": 0.5, "S12": 0.5},
            portfolio_size=10,
            buffer_exit_rank=20,
        )
        self.assertIn("S11", selected)
        self.assertIn("S12", selected)
        self.assertEqual(len(selected), 10)
        self.assertNotIn("S09", selected)

    def test_industry_market_cap_neutralization_removes_linear_exposure(self):
        rng = np.random.default_rng(9)
        n = 90
        industry = pd.Series(np.repeat(["A", "B", "C"], n // 3))
        log_cap = rng.normal(23.0, 1.0, n)
        market_cap = pd.Series(np.exp(log_cap))
        industry_effect = industry.map({"A": -1.0, "B": 0.5, "C": 1.5}).to_numpy()
        values = pd.Series(2.5 * log_cap + industry_effect + rng.normal(0, 0.1, n))
        residual = neutralize_cross_section(values, industry, market_cap)
        self.assertLess(abs(np.corrcoef(residual, log_cap)[0, 1]), 1e-10)
        means = residual.groupby(industry).mean().abs()
        self.assertTrue(means.lt(1e-10).all())

    def test_factor_correlation_uses_date_code_aligned_panel(self):
        rows = []
        for index in range(20):
            for factor, multiplier in (("f1", 1.0), ("f2", 2.0)):
                rows.append({
                    "date": pd.Timestamp("2025-01-01"),
                    "code": f"{index:06d}",
                    "factor": factor,
                    "factor_z": multiplier * index,
                })
        panel = pd.DataFrame(rows)
        correlation = compute_factor_correlation(panel)
        self.assertAlmostEqual(correlation.loc["f1", "f2"], 1.0)

    def test_execution_constraints_freeze_blocked_positions(self):
        state = pd.DataFrame({
            "amount": [1e8, 1e8, 1e8],
            "suspended": [False, False, False],
            "limit_up": [False, False, True],
            "limit_down": [True, False, False],
            "buyable": [True, True, False],
            "sellable": [False, True, True],
        }, index=["A", "B", "C"])
        target, cash, diagnostics = build_constrained_target(
            ["B", "C"],
            {"A": 0.6, "B": 0.4},
            state,
            portfolio_size=2,
        )
        self.assertAlmostEqual(target["A"], 0.6)
        self.assertAlmostEqual(target["B"], 0.4)
        self.assertNotIn("C", target)
        self.assertAlmostEqual(cash, 0.0)
        self.assertEqual(diagnostics["blocked_buys"], 1)
        self.assertEqual(diagnostics["blocked_sells"], 1)

    def test_slippage_and_impact_costs_are_positive_and_auditable(self):
        state = pd.DataFrame({
            "amount": [20_000_000.0, 50_000_000.0],
            "suspended": [False, False],
            "limit_up": [False, False],
            "limit_down": [False, False],
            "buyable": [True, True],
            "sellable": [True, True],
        }, index=["A", "B"])
        costs, trades = estimate_trade_costs(
            10_000_000.0,
            {"A": 1.0},
            {"A": 0.2, "B": 0.8},
            state,
        )
        self.assertEqual(len(trades), 2)
        self.assertGreater(costs["base_slippage_cost"], 0)
        self.assertGreater(costs["impact_cost"], 0)
        self.assertGreater(costs["sell_fee"], 0)
        self.assertAlmostEqual(
            costs["total_cost"],
            costs["base_slippage_cost"] + costs["impact_cost"] + costs["sell_fee"],
        )
        optimistic, _ = estimate_trade_costs(
            10_000_000.0,
            {"A": 1.0},
            {"A": 0.2, "B": 0.8},
            state,
            base_slippage_bps=0.0,
            impact_coefficient=0.0,
            max_impact_rate=0.0,
        )
        pessimistic, _ = estimate_trade_costs(
            10_000_000.0,
            {"A": 1.0},
            {"A": 0.2, "B": 0.8},
            state,
            base_slippage_bps=10.0,
            impact_coefficient=0.015,
            max_impact_rate=0.03,
        )
        self.assertLess(optimistic["total_cost"], pessimistic["total_cost"])

    def test_equal_weight_benchmark_excludes_blocked_entry_names(self):
        date = pd.Timestamp("2025-01-02")
        returns = pd.DataFrame(
            [[0.1, -0.2, 0.3]], index=[date], columns=["A", "B", "C"]
        )
        state = {
            "amount": pd.DataFrame([[1e8, 1e8, 0.0]], index=[date], columns=returns.columns),
            "suspended": pd.DataFrame([[False, False, True]], index=[date], columns=returns.columns),
            "limit_up": pd.DataFrame([[False, True, False]], index=[date], columns=returns.columns),
            "limit_down": pd.DataFrame([[False, False, False]], index=[date], columns=returns.columns),
        }
        benchmark_return, n_stock = equal_weight_benchmark_return(returns, state, date)
        self.assertEqual(n_stock, 1)
        self.assertAlmostEqual(benchmark_return, 0.1)


if __name__ == "__main__":
    unittest.main()
