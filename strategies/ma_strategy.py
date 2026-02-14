"""
Moving average strategy (MA).

Single MA: buy when price > MA, sell when price < MA.
Double MA: buy when short MA > long MA, sell when short < long.

Params: s (short period), l (long period, 0=single), b (bandwidth), c (holding), d (delay)
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
    "b": [0, 0.00025, 0.0005, 0.001, 0.01],
    "c": [0, 1, 2],
    "d": [0, 1, 2]
}

def calculate_ma(prices, period):
    if period <= 1:
        return prices.copy()
    return pd.Series(prices).rolling(period).mean().values


def generate_single_ma_signals(prices, s, b):
    """Buy: price > MA*(1+b). Sell: price < MA*(1-b)."""
    ma = calculate_ma(prices, s)
    signals = np.zeros(len(prices))
    
    signals[prices > ma * (1 + b)] = 1
    signals[prices < ma * (1 - b)] = -1
    
    last = 0
    for i in range(len(signals)):
        if signals[i] == 0:
            signals[i] = last
        else:
            last = signals[i]
    
    return signals


def generate_double_ma_signals(prices, s, l, b):
    """Buy: short MA > long MA*(1+b). Sell: short MA < long MA*(1-b)."""
    ma_short = calculate_ma(prices, s)
    ma_long = calculate_ma(prices, l)
    
    signals = np.zeros(len(prices))
    
    signals[ma_short > ma_long * (1 + b)] = 1
    signals[ma_short < ma_long * (1 - b)] = -1
    
    last = 0
    for i in range(len(signals)):
        if signals[i] == 0:
            signals[i] = last
        else:
            last = signals[i]
    
    return signals


def run_ma_strategy(prices, s, l, b, c, d, contrarian=False):
    """Run MA strategy with all filters. l=0 for single MA."""
    if l == 0:
        signals = generate_single_ma_signals(prices, s, b)
    else:
        signals = generate_double_ma_signals(prices, s, l, b)
    
    if contrarian:
        signals = -signals
    
    positions = calculate_positions_with_delay(signals, d)
    positions = apply_holding_period(positions, c)
    
    return signals, positions


def get_param_combinations(params=None, include_contrarian=True):
    """Generate valid param combos. For double MA, s must be < l."""
    if params is None:
        params = DEFAULT_PARAMS
    
    contrarian_values = [False, True] if include_contrarian else [False]
    
    for s, l, b, c, d, contrarian in product(
        params["s"], params["l"], params["b"],
        params["c"], params["d"], contrarian_values
    ):
        if l != 0 and s >= l:
            continue
        
        yield {"s": s, "l": l, "b": b, "c": c, "d": d, "contrarian": contrarian}


def backtest_all_ma_strategies(
    prices, params=None, include_contrarian=True,
    transaction_cost=0.001, verbose=True
):
    """Backtest all MA parameterizations."""
    results = []
    param_list = list(get_param_combinations(params, include_contrarian))
    total = len(param_list)
    
    for i, p in enumerate(param_list):
        if verbose and (i + 1) % 100 == 0:
            print(f"  MA: {i + 1}/{total}")
        
        signals, positions = run_ma_strategy(
            prices, s=p["s"], l=p["l"], b=p["b"],
            c=p["c"], d=p["d"], contrarian=p["contrarian"]
        )
        
        suffix = "c" if p["contrarian"] else ""
        if p["l"] == 0:
            strategy_name = f"MA{suffix}({p['s']},{p['b']},{p['c']},{p['d']})"
        else:
            strategy_name = f"MA{suffix}({p['s']},{p['l']},{p['b']},{p['c']},{p['d']})"
        
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
