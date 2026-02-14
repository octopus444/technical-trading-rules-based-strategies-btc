"""
Backtesting engine. Calculates returns, sharpe, and other metrics.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class BacktestResult:
    strategy_name: str
    params: dict
    mean_excess_return_bps: float
    sharpe_ratio: float
    sharpe_ratio_diff: float  # vs benchmark
    n_trades: int
    total_return: float
    benchmark_return: float
    betc: float  # break-even transaction cost
    returns_series: np.ndarray


def calculate_positions_with_delay(signals, d):
    """
    Apply delay filter: signal must persist for d periods before position changes.
    """
    n = len(signals)
    positions = np.zeros(n)
    
    if d == 0:
        return signals.copy()
    
    last_signal = 0
    signal_change_idx = None
    prev_position = 0
    
    for i in range(n):
        if signals[i] != last_signal:
            if (i + d) < n and np.all(signals[i+1:i+d+1] == signals[i]):
                signal_change_idx = i
                last_signal = signals[i]
        
        if signal_change_idx is not None and i >= signal_change_idx + d:
            if np.all(signals[signal_change_idx:i+1] == last_signal):
                positions[i] = last_signal
                prev_position = last_signal
                signal_change_idx = None
            else:
                positions[i] = prev_position
        else:
            positions[i] = prev_position
    
    return positions


def apply_holding_period(positions, c):
    """Min holding period: position held for at least c periods after change."""
    if c == 0:
        return positions.copy()
    
    modified = positions.copy()
    n = len(modified)
    
    for i in range(1, n):
        if modified[i] == 1 and modified[i-1] == -1:
            end_idx = min(i + c + 1, n)
            modified[i:end_idx] = 1
        elif modified[i] == -1 and modified[i-1] == 1:
            end_idx = min(i + c + 1, n)
            modified[i:end_idx] = -1
    
    return modified


def backtest_strategy(prices, positions, strategy_name, params, transaction_cost=0.001):
    """
    Run backtest. Returns BacktestResult with all metrics.
    TC default = 10bps one-way.
    """
    log_returns = np.log(prices[1:] / prices[:-1])
    
    strat_ret = positions[:-1] * log_returns
    bench_ret = log_returns.copy()
    
    pos_changes = np.diff(positions)
    n_trades = np.sum(pos_changes != 0)
    
    # amortized TC
    tc_per_trade = transaction_cost * 2
    total_tc = n_trades * tc_per_trade / len(positions)
    
    mean_ret = np.mean(strat_ret)
    mean_bench = np.mean(bench_ret)
    excess_ret = mean_ret - mean_bench
    excess_ret_bps = excess_ret * 10000
    
    # annualized sharpe (5-min data: 365 * 288 periods/year)
    periods_per_year = 365 * 288
    
    std_ret = np.std(strat_ret)
    std_bench = np.std(bench_ret)
    
    sharpe = (mean_ret / std_ret) * np.sqrt(periods_per_year) if std_ret > 0 else 0.0
    sharpe_bench = (mean_bench / std_bench) * np.sqrt(periods_per_year) if std_bench > 0 else 0.0
    
    total_return = np.sum(strat_ret)
    benchmark_return = np.sum(bench_ret)
    
    # break-even TC
    betc = total_return / (n_trades * 2) * 10000 if n_trades > 0 else np.inf
    
    return BacktestResult(
        strategy_name=strategy_name,
        params=params,
        mean_excess_return_bps=excess_ret_bps,
        sharpe_ratio=sharpe,
        sharpe_ratio_diff=sharpe - sharpe_bench,
        n_trades=n_trades,
        total_return=total_return,
        benchmark_return=benchmark_return,
        betc=betc,
        returns_series=strat_ret
    )


def results_to_dataframe(results):
    """Convert list of BacktestResult to DataFrame."""
    rows = []
    for r in results:
        rows.append({
            "strategy": r.strategy_name,
            **r.params,
            "mean_excess_return_bps": r.mean_excess_return_bps,
            "sharpe_ratio": r.sharpe_ratio,
            "sharpe_ratio_diff": r.sharpe_ratio_diff,
            "n_trades": r.n_trades,
            "total_return": r.total_return,
            "benchmark_return": r.benchmark_return,
            "betc": r.betc
        })
    return pd.DataFrame(rows)
