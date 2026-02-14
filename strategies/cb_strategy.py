"""
Channel breakout strategy (CB).

Buy on breakout above n-period high after channel formation.
Sell on breakdown below n-period low.
Channel = price stays within x% range for n periods.
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
    "n": [5, 10, 15, 20, 30],
    "x": [0.01, 0.02, 0.04, 0.075],
    "b": [0, 0.00025, 0.0005, 0.001, 0.0015],
    "c": [0, 1, 2]
}


def detect_channel(prices, n, x):
    """Channel formed when (max-min)/min < x over n periods."""
    df = pd.DataFrame({"Price": prices})
    rolling_max = df["Price"].rolling(n).max()
    rolling_min = df["Price"].rolling(n).min()
    
    channel_width = (rolling_max - rolling_min) / rolling_min
    return (channel_width < x).values


def generate_cb_signals(prices, n, x, b):
    """Buy: channel + price > rolling_max*(1+b). Sell: channel + price < rolling_min*(1-b)."""
    df = pd.DataFrame({"Price": prices})
    
    rolling_max = df["Price"].rolling(n).max().shift(1).values
    rolling_min = df["Price"].rolling(n).min().shift(1).values
    
    channel_formed = detect_channel(prices, n, x)
    channel_formed = np.roll(channel_formed, 1)
    channel_formed[0] = False
    
    signals = np.zeros(len(prices))
    
    signals[channel_formed & (prices > rolling_max * (1 + b))] = 1
    signals[channel_formed & (prices < rolling_min * (1 - b))] = -1
    
    last = 0
    for i in range(len(signals)):
        if signals[i] == 0:
            signals[i] = last
        else:
            last = signals[i]
    
    return signals


def run_cb_strategy(prices, n, x, b, c, contrarian=False):
    """Run CB strategy. No d parameter, only holding period c."""
    signals = generate_cb_signals(prices, n, x, b)
    
    if contrarian:
        signals = -signals
    
    positions = apply_holding_period(signals, c)
    return signals, positions


def get_param_combinations(params=None, include_contrarian=True):
    if params is None:
        params = DEFAULT_PARAMS
    
    contrarian_values = [False, True] if include_contrarian else [False]
    
    for n, x, b, c, contrarian in product(
        params["n"], params["x"], params["b"], params["c"],
        contrarian_values
    ):
        yield {"n": n, "x": x, "b": b, "c": c, "contrarian": contrarian}


def backtest_all_cb_strategies(
    prices, params=None, include_contrarian=True,
    transaction_cost=0.001, verbose=True
):
    results = []
    param_list = list(get_param_combinations(params, include_contrarian))
    total = len(param_list)
    
    for i, p in enumerate(param_list):
        if verbose and (i + 1) % 50 == 0:
            print(f"  CB: {i + 1}/{total}")
        
        signals, positions = run_cb_strategy(
            prices, n=p["n"], x=p["x"], b=p["b"],
            c=p["c"], contrarian=p["contrarian"]
        )
        
        suffix = "c" if p["contrarian"] else ""
        strategy_name = f"CB{suffix}({p['n']},{p['x']},{p['b']},{p['c']})"
        
        result = backtest_strategy(
            prices=prices, positions=positions,
            strategy_name=strategy_name, params=p,
            transaction_cost=transaction_cost
        )
        results.append(result)
    
    return results


def count_strategies(include_contrarian=True):
    base = len(DEFAULT_PARAMS["n"]) * len(DEFAULT_PARAMS["x"]) * \
           len(DEFAULT_PARAMS["b"]) * len(DEFAULT_PARAMS["c"])
    return base * 2 if include_contrarian else base
