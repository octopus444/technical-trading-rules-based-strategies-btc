"""
Support & resistance strategy (SR).

Buy on breakout above n-period resistance (rolling max).
Sell on breakdown below n-period support (rolling min).
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
    "n": [3, 5, 10, 20, 30],
    "b": [0, 0.00025, 0.0005, 0.001, 0.0025, 0.005],
    "c": [0, 1, 2],
    "d": [0, 1, 2]
}


def generate_sr_signals(prices, n, b):
    """
    Buy: price > rolling_max*(1+b), sell: price < rolling_min*(1-b).
    shift(1) to avoid look-ahead.
    """
    df = pd.DataFrame({"Price": prices})
    
    rolling_max = df["Price"].rolling(n).max().shift(1).values
    rolling_min = df["Price"].rolling(n).min().shift(1).values
    
    signals = np.zeros(len(prices))
    signals[prices > rolling_max * (1 + b)] = 1
    signals[prices < rolling_min * (1 - b)] = -1
    
    last = 0
    for i in range(len(signals)):
        if signals[i] == 0:
            signals[i] = last
        else:
            last = signals[i]
    
    return signals


def run_sr_strategy(prices, n, b, c, d, contrarian=False):
    signals = generate_sr_signals(prices, n, b)
    
    if contrarian:
        signals = -signals
    
    positions = calculate_positions_with_delay(signals, d)
    positions = apply_holding_period(positions, c)
    return signals, positions


def get_param_combinations(params=None, include_contrarian=True):
    if params is None:
        params = DEFAULT_PARAMS
    
    contrarian_values = [False, True] if include_contrarian else [False]
    
    for n, b, c, d, contrarian in product(
        params["n"], params["b"], params["c"], params["d"],
        contrarian_values
    ):
        yield {"n": n, "b": b, "c": c, "d": d, "contrarian": contrarian}


def backtest_all_sr_strategies(
    prices, params=None, include_contrarian=True,
    transaction_cost=0.001, verbose=True
):
    """Backtest all S&R parameterizations."""
    results = []
    param_list = list(get_param_combinations(params, include_contrarian))
    total = len(param_list)
    
    for i, p in enumerate(param_list):
        if verbose and (i + 1) % 50 == 0:
            print(f"  SR: {i + 1}/{total}")
        
        signals, positions = run_sr_strategy(
            prices, n=p["n"], b=p["b"], c=p["c"], d=p["d"],
            contrarian=p["contrarian"]
        )
        
        suffix = "c" if p["contrarian"] else ""
        strategy_name = f"SR{suffix}({p['n']},{p['b']},{p['c']},{p['d']})"
        
        result = backtest_strategy(
            prices=prices, positions=positions,
            strategy_name=strategy_name, params=p,
            transaction_cost=transaction_cost
        )
        results.append(result)
    
    return results


def count_strategies(include_contrarian=True):
    base = len(DEFAULT_PARAMS["n"]) * len(DEFAULT_PARAMS["b"]) * \
           len(DEFAULT_PARAMS["c"]) * len(DEFAULT_PARAMS["d"])
    return base * 2 if include_contrarian else base
