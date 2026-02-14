"""
Filter strategy (F).

Buy when price rises x% above n-period min, sell when falls x% below n-period max.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Generator
from itertools import product

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.backtester import (
    BacktestResult,
    calculate_positions_with_delay,
    apply_holding_period,
    backtest_strategy
)


DEFAULT_PARAMS = {
    "n": [3, 5, 10, 18],
    "x": [0.0005, 0.001, 0.0025, 0.005],
    "c": [0, 1, 2],
    "d": [0, 1, 2]
}


def generate_filter_signals(prices, n, x):
    """Buy: price > (1+x)*rolling_min. Sell: price < (1-x)*rolling_max."""
    df = pd.DataFrame({"Price": prices})
    rolling_max = df["Price"].rolling(n).max().values
    rolling_min = df["Price"].rolling(n).min().values
    
    signals = np.zeros(len(prices))
    
    signals[prices > (1 + x) * rolling_min] = 1
    signals[prices < (1 - x) * rolling_max] = -1
    
    # forward fill
    last = 0
    for i in range(len(signals)):
        if signals[i] == 0:
            signals[i] = last
        else:
            last = signals[i]
    
    return signals


def run_filter_strategy(prices, n, x, c, d, contrarian=False):
    """Run filter strategy with delay and holding period filters."""
    signals = generate_filter_signals(prices, n, x)
    
    if contrarian:
        signals = -signals
    
    positions = calculate_positions_with_delay(signals, d)
    positions = apply_holding_period(positions, c)
    
    return signals, positions


def get_param_combinations(params=None, include_contrarian=True):
    if params is None:
        params = DEFAULT_PARAMS
    
    contrarian_values = [False, True] if include_contrarian else [False]
    
    for n, x, c, d, contrarian in product(
        params["n"], params["x"], params["c"], params["d"],
        contrarian_values
    ):
        yield {"n": n, "x": x, "c": c, "d": d, "contrarian": contrarian}


def backtest_all_filter_strategies(
    prices, params=None, include_contrarian=True,
    transaction_cost=0.001, verbose=True
):
    """Backtest all filter parameterizations."""
    results = []
    param_list = list(get_param_combinations(params, include_contrarian))
    total = len(param_list)
    
    for i, p in enumerate(param_list):
        if verbose and (i + 1) % 50 == 0:
            print(f"  Filter: {i + 1}/{total}")
        
        signals, positions = run_filter_strategy(
            prices, n=p["n"], x=p["x"], c=p["c"], d=p["d"],
            contrarian=p["contrarian"]
        )
        
        suffix = "c" if p["contrarian"] else ""
        strategy_name = f"F{suffix}({p['x']},{p['n']},{p['c']},{p['d']})"
        
        result = backtest_strategy(
            prices=prices, positions=positions,
            strategy_name=strategy_name, params=p,
            transaction_cost=transaction_cost
        )
        results.append(result)
    
    return results


def count_strategies(include_contrarian=True):
    base = len(DEFAULT_PARAMS["n"]) * len(DEFAULT_PARAMS["x"]) * \
           len(DEFAULT_PARAMS["c"]) * len(DEFAULT_PARAMS["d"])
    return base * 2 if include_contrarian else base
