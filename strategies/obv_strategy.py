"""
On-balance volume strategy (OBV).

OBV accumulates volume on up days, subtracts on down days.
Uses MA crossover on OBV: buy when short OBV MA > long OBV MA.
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
    "s": [2, 3, 5, 7, 10],
    "l": [3, 5, 7, 10, 15],
    "b": [0, 0.025, 0.05, 0.1, 0.5],
    "c": [0, 1, 2],
    "d": [0, 1, 2]
}


def calculate_obv(prices, volumes):
    """Cumulative OBV: +volume on up, -volume on down."""
    price_diff = np.diff(prices, prepend=prices[0])
    obv = np.zeros(len(prices))
    
    for i in range(1, len(prices)):
        if price_diff[i] > 0:
            obv[i] = obv[i-1] + volumes[i]
        elif price_diff[i] < 0:
            obv[i] = obv[i-1] - volumes[i]
        else:
            obv[i] = obv[i-1]
    
    return obv


def generate_obv_signals(prices, volumes, s, l, b):
    """Buy: short OBV MA > long OBV MA * (1+b). Sell: opposite."""
    obv = calculate_obv(prices, volumes)
    
    df = pd.DataFrame({"OBV": obv})
    obv_short = df["OBV"].rolling(s, min_periods=1).mean().values
    obv_long = df["OBV"].rolling(l, min_periods=1).mean().values
    
    signals = np.zeros(len(prices))
    signals[obv_short > obv_long * (1 + b)] = 1
    signals[obv_short < obv_long * (1 - b)] = -1
    
    last = 0
    for i in range(len(signals)):
        if signals[i] == 0:
            signals[i] = last
        else:
            last = signals[i]
    
    return signals


def run_obv_strategy(prices, volumes, s, l, b, c, d, contrarian=False):
    signals = generate_obv_signals(prices, volumes, s, l, b)
    
    if contrarian:
        signals = -signals
    
    positions = calculate_positions_with_delay(signals, d)
    positions = apply_holding_period(positions, c)
    return signals, positions


def get_param_combinations(params=None, include_contrarian=True):
    if params is None:
        params = DEFAULT_PARAMS
    
    contrarian_values = [False, True] if include_contrarian else [False]
    
    for s, l, b, c, d, contrarian in product(
        params["s"], params["l"], params["b"],
        params["c"], params["d"], contrarian_values
    ):
        if s >= l:
            continue
        yield {"s": s, "l": l, "b": b, "c": c, "d": d, "contrarian": contrarian}


def backtest_all_obv_strategies(
    prices, volumes, params=None, include_contrarian=True,
    transaction_cost=0.001, verbose=True
):
    results = []
    param_list = list(get_param_combinations(params, include_contrarian))
    total = len(param_list)
    
    for i, p in enumerate(param_list):
        if verbose and (i + 1) % 100 == 0:
            print(f"  OBV: {i + 1}/{total}")
        
        signals, positions = run_obv_strategy(
            prices, volumes,
            s=p["s"], l=p["l"], b=p["b"],
            c=p["c"], d=p["d"], contrarian=p["contrarian"]
        )
        
        suffix = "c" if p["contrarian"] else ""
        strategy_name = f"OBV{suffix}({p['s']},{p['l']},{p['b']},{p['c']},{p['d']})"
        
        result = backtest_strategy(
            prices=prices, positions=positions,
            strategy_name=strategy_name, params=p,
            transaction_cost=transaction_cost
        )
        results.append(result)
    
    return results


def count_strategies(include_contrarian=True):
    count = sum(1 for _ in get_param_combinations(include_contrarian=False))
    return count * 2 if include_contrarian else count
