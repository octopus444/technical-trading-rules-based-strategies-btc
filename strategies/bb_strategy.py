"""
Bollinger bands strategy (BB).

Mean-reversion: buy when price < lower band (oversold), sell when > upper band.
Upper = MA(n) + k*std(n), Lower = MA(n) - k*std(n).
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
    "n": [3, 5, 7, 12, 18],
    "k": [0.25, 0.5, 1, 2],
    "c": [0, 1, 2],
    "d": [0, 1, 2]
}

def calculate_bollinger_bands(prices, n, k):
    """Returns (middle, upper, lower) bands."""
    df = pd.DataFrame({"Price": prices})
    middle = df["Price"].rolling(n).mean().values
    std = df["Price"].rolling(n).std().values
    return middle, middle + k * std, middle - k * std


def generate_bb_signals(prices, n, k):
    """
    Buy: price < lower band (oversold, expect reversion).
    Sell: price > upper band (overbought).
    Note: base BB is mean-reversion, contrarian=True makes it trend-following.
    """
    middle, upper, lower = calculate_bollinger_bands(prices, n, k)
    
    signals = np.zeros(len(prices))
    signals[prices < lower] = 1
    signals[prices > upper] = -1
    
    last = 0
    for i in range(len(signals)):
        if signals[i] == 0:
            signals[i] = last
        else:
            last = signals[i]
    
    return signals


def run_bb_strategy(prices, n, k, c, d, contrarian=False):
    signals = generate_bb_signals(prices, n, k)
    
    if contrarian:
        signals = -signals
    
    positions = calculate_positions_with_delay(signals, d)
    positions = apply_holding_period(positions, c)
    return signals, positions


def get_param_combinations(params=None, include_contrarian=True):
    if params is None:
        params = DEFAULT_PARAMS
    
    contrarian_values = [False, True] if include_contrarian else [False]
    
    for n, k, c, d, contrarian in product(
        params["n"], params["k"], params["c"], params["d"],
        contrarian_values
    ):
        yield {"n": n, "k": k, "c": c, "d": d, "contrarian": contrarian}


def backtest_all_bb_strategies(
    prices, params=None, include_contrarian=True,
    transaction_cost=0.001, verbose=True
):
    results = []
    param_list = list(get_param_combinations(params, include_contrarian))
    total = len(param_list)
    
    for i, p in enumerate(param_list):
        if verbose and (i + 1) % 50 == 0:
            print(f"  BB: {i + 1}/{total}")
        
        signals, positions = run_bb_strategy(
            prices, n=p["n"], k=p["k"], c=p["c"], d=p["d"],
            contrarian=p["contrarian"]
        )
        
        suffix = "c" if p["contrarian"] else ""
        strategy_name = f"BB{suffix}({p['n']},{p['k']},{p['c']},{p['d']})"
        
        result = backtest_strategy(
            prices=prices, positions=positions,
            strategy_name=strategy_name, params=p,
            transaction_cost=transaction_cost
        )
        results.append(result)
    
    return results


def count_strategies(include_contrarian=True):
    base = len(DEFAULT_PARAMS["n"]) * len(DEFAULT_PARAMS["k"]) * \
           len(DEFAULT_PARAMS["c"]) * len(DEFAULT_PARAMS["d"])
    return base * 2 if include_contrarian else base
