#!/usr/bin/env python3
"""Deterministic synthetic daily-data smoke demo; outputs are not research evidence."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analyze_factors_backtest import (
    build_factors, clean_factor_panel, load_wide,
    returns_and_market_state, simulate, summarize_backtest_results,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('tmp/synthetic-demo'))
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    daily = root / 'processed' / 'daily'
    daily.mkdir(parents=True)
    rng = np.random.default_rng(42)
    dates = pd.bdate_range('2025-01-01', periods=110)
    codes = [f'{i:06d}' for i in range(1, 41)]
    close = 30 * np.exp(np.cumsum(rng.normal(0, .012, (110, 40)), axis=0))
    opening = close * np.exp(rng.normal(0, .004, close.shape))
    volume = rng.uniform(1e6, 5e6, close.shape)
    buy = volume * rng.uniform(.3, .6, close.shape)
    sell = volume * .9 - buy
    fields = dict(open=opening, close=close,
                  high=np.maximum(opening, close) * 1.01,
                  low=np.minimum(opening, close) * .99,
                  volume=volume, amount=volume*close,
                  trade_count=volume/200, buy_volume=buy, sell_volume=sell,
                  buy_amount=buy*close, sell_amount=sell*close)
    for field, values in fields.items():
        pd.DataFrame(values, index=dates.strftime('%Y%m%d'), columns=codes).to_csv(daily / f'{field}.csv', index_label='date')
    processed = root / 'processed'
    factors = build_factors(processed, root / 'factors')
    close_table = load_wide(processed, 'close')
    panel = clean_factor_panel(factors, close_table.shift(-1)/close_table-1)
    returns, trade_dates, state = returns_and_market_state(processed, 'next_open_to_open')
    results, holdings, weights, trades = simulate(panel, returns, trade_dates, state, 'historical_20d_ic', 'next_open_to_open')
    if results.empty or not np.isfinite(results['net_return']).all():
        raise RuntimeError('Synthetic demo did not produce finite daily results')
    for name, frame in [('results', results), ('holdings', holdings), ('weights', weights), ('trades', trades), ('summary', summarize_backtest_results(results))]:
        frame.to_csv(root / f'{name}.csv', index=False)
    (root / 'provenance.json').write_text(json.dumps({'source': 'SYNTHETIC ONLY', 'seed': 42, 'stocks': 40, 'dates': 110, 'scope': 'daily factors and execution-aware backtest; not raw-tick reconstruction or LSTM training'}, indent=2)+'\n')
    print(f'Synthetic demo completed: {root}')

if __name__ == '__main__':
    main()
