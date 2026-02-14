"""
RSI strategy.

Buy when RSI < 50-v (oversold), sell when RSI > 50+v (overbought).
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
    "v": [10, 20, 30, 40],
    "c": [0, 1, 2],
    "d": [0, 1, 2]
}

def calculate_rsi(prices, n):
    """RSI = 100 - 100/(1+RS), RS = avg_gain/avg_loss over n periods."""
    delta = np.diff(prices, prepend=prices[0])
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).rolling(n, min_periods=1).mean().values
    avg_loss = pd.Series(losses).rolling(n, min_periods=1).mean().values
    
    avg_loss = np.where(avg_loss == 0, 1e-10, avg_loss)
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def generate_rsi_signals(prices, n, v):
    rsi = calculate_rsi(prices, n)
    
    signals = np.zeros(len(prices))
    signals[rsi < (50 - v)] = 1
    signals[rsi > (50 + v)] = -1
    
    last = 0
    for i in range(len(signals)):
        if signals[i] == 0:
            signals[i] = last
        else:
            last = signals[i]
    
    return signals


def run_rsi_strategy(prices, n, v, c, d, contrarian=False):
    signals = generate_rsi_signals(prices, n, v)
    
    if contrarian:
        signals = -signals
    
    positions = calculate_positions_with_delay(signals, d)
    positions = apply_holding_period(positions, c)
    return signals, positions


def get_param_combinations(params=None, include_contrarian=True):
    if params is None:
        params = DEFAULT_PARAMS
    
    contrarian_values = [False, True] if include_contrarian else [False]
    
    for n, v, c, d, contrarian in product(
        params["n"], params["v"], params["c"], params["d"],
        contrarian_values
    ):
        yield {"n": n, "v": v, "c": c, "d": d, "contrarian": contrarian}


def backtest_all_rsi_strategies(
    prices, params=None, include_contrarian=True,
    transaction_cost=0.001, verbose=True
):
    results = []
    param_list = list(get_param_combinations(params, include_contrarian))
    total = len(param_list)
    
    for i, p in enumerate(param_list):
        if verbose and (i + 1) % 50 == 0:
            print(f"  RSI: {i + 1}/{total}")
        
        signals, positions = run_rsi_strategy(
            prices, n=p["n"], v=p["v"], c=p["c"], d=p["d"],
            contrarian=p["contrarian"]
        )
        
        suffix = "c" if p["contrarian"] else ""
        strategy_name = f"RSI{suffix}({p['n']},{p['v']},{p['c']},{p['d']})"
        
        result = backtest_strategy(
            prices=prices, positions=positions,
            strategy_name=strategy_name, params=p,
            transaction_cost=transaction_cost
        )
        results.append(result)
    
    return results


def count_strategies(include_contrarian=True):
    base = len(DEFAULT_PARAMS["n"]) * len(DEFAULT_PARAMS["v"]) * \
           len(DEFAULT_PARAMS["c"]) * len(DEFAULT_PARAMS["d"])
    return base * 2 if include_contrarian else base
