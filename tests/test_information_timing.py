"""Regression checks for information available at the decision time."""
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_factors_backtest import compute_weights, simulate, volatility_target_exposure
from train_lstm_pytorch import metrics_from_probability


class InformationTimingTests(unittest.TestCase):
    def test_future_prices_cannot_change_current_weights(self):
        rng = np.random.default_rng(42)
        dates = pd.bdate_range('2025-01-01', periods=85)
        codes = [f'{i:06}' for i in range(30)]
        prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, .02, (85, 30)), axis=0)), index=dates, columns=codes)
        panel = pd.DataFrame([(f, d, c, rng.normal()) for f in ['a', 'b', 'c', 'd', 'e'] for d in dates for c in codes], columns=['factor', 'date', 'code', 'factor_z'])
        today = dates[65]
        changed = prices.copy()
        changed.loc[dates[66]:] *= rng.uniform(.5, 1.5, (19, 30))
        # Both open-to-open and close-to-close use this same two-session alignment.
        for strategy in ['historical_20d_ic', 'historical_60d_ewma_icir']:
            for sparse in [False, True]:
                with self.subTest(strategy=strategy, sparse=sparse):
                    sample = panel.loc[~panel.date.isin(dates[[5, 17]])] if sparse else panel
                    before = compute_weights(sample, prices.shift(-2) / prices.shift(-1) - 1, strategy)
                    after = compute_weights(sample, changed.shift(-2) / changed.shift(-1) - 1, strategy)
                    mask = before.date.le(today)
                    pd.testing.assert_frame_equal(before.loc[mask].drop(columns='raw_ic'), after.loc[mask].drop(columns='raw_ic'))
                    self.assertTrue(before.loc[before.date.eq(today), 'weight'].notna().all())

    def test_ic_becomes_available_at_return_end(self):
        dates = pd.bdate_range('2025-01-01', periods=25)
        x = np.arange(30.)
        panel = pd.DataFrame([('a', d, str(c), v) for d in dates for c, v in enumerate(x)], columns=['factor', 'date', 'code', 'factor_z'])
        returns = pd.DataFrame(np.tile(x, (25, 1)), index=dates, columns=[str(c) for c in range(30)])
        before = compute_weights(panel, returns, 'historical_20d_ic')
        returns.iloc[20] *= -1
        after = compute_weights(panel, returns, 'historical_20d_ic')
        self.assertAlmostEqual(before.loc[21, 'weight'], after.loc[21, 'weight'])
        self.assertNotAlmostEqual(before.loc[22, 'weight'], after.loc[22, 'weight'])
        self.assertTrue(before.loc[:20, 'weight'].isna().all())

    def test_volatility_budget_uses_only_completed_returns(self):
        rng = np.random.default_rng(7)
        dates = pd.bdate_range('2025-01-01', periods=55)
        codes = [str(i) for i in range(30)]
        panel = pd.DataFrame([('a', d, c, rng.normal()) for d in dates for c in codes], columns=['factor', 'date', 'code', 'factor_z'])
        returns = pd.DataFrame(rng.normal(0, .02, (55, 30)), index=dates, columns=codes)
        returns.iloc[-2:] = np.nan
        state = {name: pd.DataFrame(value, index=dates, columns=codes) for name, value in [('amount', 1e9), ('suspended', False), ('limit_up', False), ('limit_down', False)]}
        with patch('analyze_factors_backtest.volatility_target_exposure', wraps=volatility_target_exposure) as budget:
            results, _, _, _ = simulate(panel, returns, pd.Series(dates, index=dates).shift(-1), state, 'historical_20d_ic', 'next_open_to_open', target_volatility=.15)
        self.assertGreater(len(results), 20)
        self.assertEqual(len(budget.call_args_list), len(results))
        for call, row in zip(budget.call_args_list, results.itertuples()):
            cutoff = dates[dates.get_loc(row.signal_date)-2]
            expected = results.loc[results.signal_date.le(cutoff), 'gross_return'].to_numpy()
            np.testing.assert_allclose(call.args[0], expected)

    def test_training_majority_can_lose_on_test_distribution(self):
        labels = np.array([1., 1., 1., 0.])
        training_probability = np.full(4, .2)
        metrics = metrics_from_probability(labels, training_probability, training_probability)
        self.assertEqual(metrics['accuracy'], .25)
        self.assertEqual(metrics['majority_baseline_accuracy'], .25)
        self.assertEqual(metrics['auc'], .5)

if __name__ == '__main__':
    unittest.main()
