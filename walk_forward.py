"""
Walk-Forward Analysis (WFA) and Walk-Forward Optimization (WFO)

WFA: test fixed strategy on sequential OOS periods
WFO: optimize params in-sample, test OOS

    python walk_forward.py path/to/.bitstampUSD.csv
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import time


@dataclass
class WalkForwardResult:
    period_name: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    strategy_name: str
    strategy_params: dict
    
    # in-sample
    is_return_bps: float
    is_sharpe: float
    is_n_trades: int
    
    # out-of-sample
    oos_return_bps: float
    oos_sharpe: float
    oos_n_trades: int
    oos_total_return: float
    oos_buy_hold_return: float


def generate_ma_signals(prices: np.ndarray, s: int, l: int, b: float) -> np.ndarray:
    """
    MA Contrarian signals.
    
    Parameters
    ----------
    s, l : short/long MA periods
    b : band threshold
    
    Returns: +1 (long), -1 (short), 0 (neutral)
    """
    n = len(prices)
    signals = np.zeros(n)
    
    if l >= n:
        return signals
    
    ma_short = np.zeros(n)
    ma_long = np.zeros(n)
    
    for i in range(s - 1, n):
        ma_short[i] = np.mean(prices[i - s + 1:i + 1])
    
    for i in range(l - 1, n):
        ma_long[i] = np.mean(prices[i - l + 1:i + 1])
    
    # contrarian: opposite of trend-following
    for i in range(l - 1, n):
        if ma_long[i] > 0:
            ratio = ma_short[i] / ma_long[i] - 1
            
            if ratio > b:
                signals[i] = -1  # sell when short > long
            elif ratio < -b:
                signals[i] = 1   # buy when short < long
    
    # forward fill
    last_signal = 0
    for i in range(n):
        if signals[i] == 0:
            signals[i] = last_signal
        else:
            last_signal = signals[i]
    
    return signals


def apply_delay_filter(signals, d):
    if d == 0:
        return signals.copy()
    
    n = len(signals)
    positions = np.zeros(n)
    last_signal = 0
    prev_position = 0
    
    for i in range(n):
        if signals[i] != last_signal:
            if (i + d) < n and np.all(signals[i+1:i+d] == signals[i]):
                last_signal = signals[i]
                if i + d < n:
                    positions[i + d] = last_signal
                    prev_position = last_signal
            else:
                positions[i] = prev_position
        else:
            positions[i] = prev_position
    
    return positions


def apply_holding_period(positions, c):
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


def backtest_single_strategy(prices, s, l, b, d, c, transaction_cost=0.001):
    """Backtest a single MAc strategy, returns dict with metrics."""
    n = len(prices)
    
    if n < l + 10:
        return {
            'mean_return_bps': 0, 'sharpe': 0, 'n_trades': 0,
            'total_return': 0, 'buy_hold_return': 0
        }
    
    signals = generate_ma_signals(prices, s, l, b)
    positions = apply_delay_filter(signals, d)
    positions = apply_holding_period(positions, c)
    
    log_returns = np.log(prices[1:] / prices[:-1])
    strategy_returns = positions[:-1] * log_returns
    
    n_trades = np.sum(np.diff(positions) != 0)
    
    n_periods = len(strategy_returns)
    tc_adj = (n_trades * 2 * transaction_cost) / n_periods if n_periods > 0 else 0
    net_returns = strategy_returns - tc_adj
    
    mean_ret = np.mean(net_returns) if len(net_returns) > 0 else 0
    std_ret = np.std(net_returns) if len(net_returns) > 0 else 1
    
    periods_per_year = 365 * 288
    sharpe = (mean_ret / std_ret) * np.sqrt(periods_per_year) if std_ret > 0 else 0
    
    total_log = np.sum(net_returns)
    bh_log = np.sum(log_returns)
    
    return {
        'mean_return_bps': mean_ret * 10000,
        'sharpe': sharpe,
        'n_trades': int(n_trades),
        'total_return': (np.exp(total_log) - 1) * 100,
        'buy_hold_return': (np.exp(bh_log) - 1) * 100
    }


def run_walk_forward_analysis(
    prices, dates, strategy_params, periods,
    transaction_cost=0.001, verbose=True
) -> List[WalkForwardResult]:
    """
    Walk-Forward Analysis: test FIXED strategy on OOS periods.
    """
    results = []
    
    s = strategy_params['s']
    l = strategy_params['l']
    b = strategy_params['b']
    d = strategy_params['d']
    c = strategy_params['c']
    
    strat_name = f"MAc({s},{l},{b},{d},{c})"
    
    if verbose:
        print("\n" + "=" * 70)
        print("WALK-FORWARD ANALYSIS (Fixed Strategy)")
        print("=" * 70)
        print(f"Strategy: {strat_name}")
        print(f"Transaction cost: {transaction_cost * 10000:.0f} bps")
    
    for i, (train_start, train_end, test_start, test_end) in enumerate(periods):
        period_name = f"Period {i+1}"
        
        test_mask = (dates >= test_start) & (dates <= test_end)
        test_prices = prices[test_mask]
        
        train_mask = (dates >= train_start) & (dates <= train_end)
        train_prices = prices[train_mask]
        
        is_metrics = backtest_single_strategy(train_prices, s, l, b, d, c, transaction_cost)
        oos_metrics = backtest_single_strategy(test_prices, s, l, b, d, c, transaction_cost)
        
        result = WalkForwardResult(
            period_name=period_name,
            train_start=train_start, train_end=train_end,
            test_start=test_start, test_end=test_end,
            strategy_name=strat_name,
            strategy_params=strategy_params,
            is_return_bps=is_metrics['mean_return_bps'],
            is_sharpe=is_metrics['sharpe'],
            is_n_trades=is_metrics['n_trades'],
            oos_return_bps=oos_metrics['mean_return_bps'],
            oos_sharpe=oos_metrics['sharpe'],
            oos_n_trades=oos_metrics['n_trades'],
            oos_total_return=oos_metrics['total_return'],
            oos_buy_hold_return=oos_metrics['buy_hold_return']
        )
        results.append(result)
        
        if verbose:
            profitable = "✓" if oos_metrics['mean_return_bps'] > 0 else "✗"
            print(f"\n{period_name}: Test {test_start[:7]} to {test_end[:7]}")
            print(f"  OOS Return: {oos_metrics['mean_return_bps']:.4f} bps/period {profitable}")
            print(f"  OOS Sharpe: {oos_metrics['sharpe']:.4f}")
            print(f"  OOS Trades: {oos_metrics['n_trades']}")
            print(f"  OOS Total:  {oos_metrics['total_return']:.2f}% vs B&H {oos_metrics['buy_hold_return']:.2f}%")
    
    if verbose:
        n_profitable = sum(1 for r in results if r.oos_return_bps > 0)
        
        print("\n" + "=" * 70)
        print("WFA SUMMARY")
        print("=" * 70)
        print(f"Profitable OOS periods: {n_profitable} / {len(results)}")
        
        avg_oos_ret = np.mean([r.oos_return_bps for r in results])
        avg_oos_sharpe = np.mean([r.oos_sharpe for r in results])
        
        print(f"Average OOS return: {avg_oos_ret:.4f} bps/period")
        print(f"Average OOS Sharpe: {avg_oos_sharpe:.4f}")
        
        if n_profitable >= 3:
            print("\n→ Strategy shows consistent OOS performance")
        elif n_profitable >= 2:
            print("\n→ Mixed results - strategy may be partially valid")
        else:
            print("\n→ Strategy fails OOS validation - confirms WRC result")
    
    return results


def run_walk_forward_optimization(
    prices, dates, periods, param_grid,
    transaction_cost=0.001, verbose=True, debug=False
) -> List[WalkForwardResult]:
    """WFO: optimize in-sample, test best strategy OOS."""
    results = []
    
    if verbose:
        print("\n" + "=" * 70)
        print("WALK-FORWARD OPTIMIZATION")
        print("=" * 70)
        print(f"Transaction cost: {transaction_cost * 10000:.0f} bps")
        
        total_strats = 0
        for s in param_grid['s']:
            for l in param_grid['l']:
                if l > s:
                    total_strats += len(param_grid['b']) * len(param_grid['d']) * len(param_grid['c'])
        print(f"Strategies per period: {total_strats}")
    
    for period_idx, (train_start, train_end, test_start, test_end) in enumerate(periods):
        period_name = f"Period {period_idx + 1}"
        
        if verbose:
            print(f"\n{period_name}: Train {train_start[:7]} to {train_end[:7]}, Test {test_start[:7]} to {test_end[:7]}")
        
        train_mask = (dates >= train_start) & (dates <= train_end)
        test_mask = (dates >= test_start) & (dates <= test_end)
        train_prices = prices[train_mask]
        test_prices = prices[test_mask]
        
        best_is_sharpe = -np.inf
        best_params = None
        best_is_metrics = None
        all_is_results = []
        
        for s in param_grid['s']:
            for l in param_grid['l']:
                if l <= s:
                    continue
                for b in param_grid['b']:
                    for d in param_grid['d']:
                        for c in param_grid['c']:
                            metrics = backtest_single_strategy(
                                train_prices, s, l, b, d, c, transaction_cost
                            )
                            
                            if debug:
                                all_is_results.append({
                                    's': s, 'l': l, 'b': b, 'd': d, 'c': c,
                                    'sharpe': metrics['sharpe'],
                                    'return_bps': metrics['mean_return_bps'],
                                    'n_trades': metrics['n_trades']
                                })
                            
                            if metrics['sharpe'] > best_is_sharpe:
                                best_is_sharpe = metrics['sharpe']
                                best_params = {'s': s, 'l': l, 'b': b, 'd': d, 'c': c}
                                best_is_metrics = metrics
        
        if best_params is None:
            if verbose:
                print("  No valid strategy found")
            continue
        
        strat_name = f"MAc({best_params['s']},{best_params['l']},{best_params['b']},{best_params['d']},{best_params['c']})"
        
        if verbose:
            print(f"  Best IS strategy: {strat_name}")
            print(f"  IS Sharpe: {best_is_sharpe:.4f}, IS Return: {best_is_metrics['mean_return_bps']:.4f} bps, IS Trades: {best_is_metrics['n_trades']}")
        
        if debug and all_is_results:
            print(f"\n  DEBUG: Top 5 IS strategies by Sharpe:")
            sorted_results = sorted(all_is_results, key=lambda x: x['sharpe'], reverse=True)
            for idx, r in enumerate(sorted_results[:5]):
                print(f"    {idx+1}. MAc({r['s']},{r['l']},{r['b']},{r['d']},{r['c']}): Sharpe={r['sharpe']:.4f}, Return={r['return_bps']:.4f} bps, Trades={r['n_trades']}")
            
            mac_710 = [r for r in all_is_results if r['s']==7 and r['l']==10 and r['b']==0.01 and r['d']==0 and r['c']==0]
            if mac_710:
                r = mac_710[0]
                print(f"\n  DEBUG: MAc(7,10,0.01,0,0): Sharpe={r['sharpe']:.4f}, Return={r['return_bps']:.4f} bps, Trades={r['n_trades']}")
        
        oos_metrics = backtest_single_strategy(
            test_prices,
            best_params['s'], best_params['l'], best_params['b'],
            best_params['d'], best_params['c'],
            transaction_cost
        )
        
        result = WalkForwardResult(
            period_name=period_name,
            train_start=train_start, train_end=train_end,
            test_start=test_start, test_end=test_end,
            strategy_name=strat_name,
            strategy_params=best_params,
            is_return_bps=best_is_metrics['mean_return_bps'],
            is_sharpe=best_is_metrics['sharpe'],
            is_n_trades=best_is_metrics['n_trades'],
            oos_return_bps=oos_metrics['mean_return_bps'],
            oos_sharpe=oos_metrics['sharpe'],
            oos_n_trades=oos_metrics['n_trades'],
            oos_total_return=oos_metrics['total_return'],
            oos_buy_hold_return=oos_metrics['buy_hold_return']
        )
        results.append(result)
        
        profitable = "✓" if oos_metrics['mean_return_bps'] > 0 else "✗"
        if verbose:
            print(f"  OOS Return: {oos_metrics['mean_return_bps']:.4f} bps/period {profitable}")
            print(f"  OOS Sharpe: {oos_metrics['sharpe']:.4f}")
            print(f"  OOS Total:  {oos_metrics['total_return']:.2f}% vs B&H {oos_metrics['buy_hold_return']:.2f}%")
    
    if verbose:
        n_profitable = sum(1 for r in results if r.oos_return_bps > 0)
        
        print("\n" + "=" * 70)
        print("WFO SUMMARY")
        print("=" * 70)
        print(f"Profitable OOS periods: {n_profitable} / {len(results)}")
        
        print("\nSelected strategies per period:")
        for r in results:
            profitable = "✓" if r.oos_return_bps > 0 else "✗"
            print(f"  {r.period_name}: {r.strategy_name} → OOS {r.oos_return_bps:.4f} bps {profitable}")
        
        avg_oos_ret = np.mean([r.oos_return_bps for r in results])
        avg_oos_sharpe = np.mean([r.oos_sharpe for r in results])
        
        print(f"\nAverage OOS return: {avg_oos_ret:.4f} bps/period")
        print(f"Average OOS Sharpe: {avg_oos_sharpe:.4f}")
        
        avg_is_sharpe = np.mean([r.is_sharpe for r in results])
        degradation = (avg_is_sharpe - avg_oos_sharpe) / avg_is_sharpe * 100 if avg_is_sharpe > 0 else 0
        
        print(f"\nSharpe degradation IS → OOS: {degradation:.1f}%")
        
        if degradation > 50:
            print("→ High degradation suggests overfitting")
        elif degradation > 25:
            print("→ Moderate degradation - some overfitting likely")
        else:
            print("→ Low degradation - strategy may be robust")
        
        if n_profitable >= 3:
            print("\n→ WFO shows strategy concept may be valid")
        elif n_profitable >= 2:
            print("\n→ Mixed results - needs more investigation")
        else:
            print("\n→ WFO fails - adaptive optimization doesn't help")
    
    return results


def run_walk_forward_optimization_fixed_sl(
    prices, dates, periods,
    s=7, l=10, b=0.01,
    d_values=[0, 1, 2], c_values=[0, 1, 2],
    transaction_cost=0.001, verbose=True
) -> List[WalkForwardResult]:
    """
    WFO with fixed s,l,b - only optimizes d (delay) and c (holding).
    Tests whether adaptive filter optimization helps MAc(7,10,0.01).
    """
    results = []
    
    if verbose:
        print("\n" + "=" * 70)
        print("WALK-FORWARD OPTIMIZATION (Fixed s,l,b)")
        print("=" * 70)
        print(f"Fixed parameters: s={s}, l={l}, b={b}")
        print(f"Optimizing: d ∈ {d_values}, c ∈ {c_values}")
        print(f"Strategies per period: {len(d_values) * len(c_values)}")
    
    for period_idx, (train_start, train_end, test_start, test_end) in enumerate(periods):
        period_name = f"Period {period_idx + 1}"
        
        if verbose:
            print(f"\n{period_name}: Train {train_start[:7]} to {train_end[:7]}, Test {test_start[:7]} to {test_end[:7]}")
        
        train_mask = (dates >= train_start) & (dates <= train_end)
        test_mask = (dates >= test_start) & (dates <= test_end)
        train_prices = prices[train_mask]
        test_prices = prices[test_mask]
        
        best_is_sharpe = -np.inf
        best_d = 0
        best_c = 0
        best_is_metrics = None
        
        for d in d_values:
            for c in c_values:
                metrics = backtest_single_strategy(
                    train_prices, s, l, b, d, c, transaction_cost
                )
                if metrics['sharpe'] > best_is_sharpe:
                    best_is_sharpe = metrics['sharpe']
                    best_d = d
                    best_c = c
                    best_is_metrics = metrics
        
        strat_name = f"MAc({s},{l},{b},{best_d},{best_c})"
        
        if verbose:
            print(f"  Best IS: d={best_d}, c={best_c}")
            print(f"  IS Sharpe: {best_is_sharpe:.4f}, IS Return: {best_is_metrics['mean_return_bps']:.4f} bps")
        
        oos_metrics = backtest_single_strategy(
            test_prices, s, l, b, best_d, best_c, transaction_cost
        )
        
        result = WalkForwardResult(
            period_name=period_name,
            train_start=train_start, train_end=train_end,
            test_start=test_start, test_end=test_end,
            strategy_name=strat_name,
            strategy_params={'s': s, 'l': l, 'b': b, 'd': best_d, 'c': best_c},
            is_return_bps=best_is_metrics['mean_return_bps'],
            is_sharpe=best_is_metrics['sharpe'],
            is_n_trades=best_is_metrics['n_trades'],
            oos_return_bps=oos_metrics['mean_return_bps'],
            oos_sharpe=oos_metrics['sharpe'],
            oos_n_trades=oos_metrics['n_trades'],
            oos_total_return=oos_metrics['total_return'],
            oos_buy_hold_return=oos_metrics['buy_hold_return']
        )
        results.append(result)
        
        profitable = "✓" if oos_metrics['mean_return_bps'] > 0 else "✗"
        if verbose:
            print(f"  OOS Return: {oos_metrics['mean_return_bps']:.4f} bps/period {profitable}")
            print(f"  OOS Sharpe: {oos_metrics['sharpe']:.4f}")
            print(f"  OOS Total:  {oos_metrics['total_return']:.2f}% vs B&H {oos_metrics['buy_hold_return']:.2f}%")
    
    if verbose:
        n_profitable = sum(1 for r in results if r.oos_return_bps > 0)
        total_periods = len(results)
        
        print("\n" + "=" * 70)
        print("WFO (Fixed s,l,b) SUMMARY")
        print("=" * 70)
        print(f"Profitable OOS periods: {n_profitable} / {total_periods}")
        
        print("\nSelected d,c per period:")
        for r in results:
            profitable = "✓" if r.oos_return_bps > 0 else "✗"
            print(f"  {r.period_name}: {r.strategy_name} → OOS {r.oos_return_bps:.4f} bps {profitable}")
        
        avg_oos_ret = np.mean([r.oos_return_bps for r in results])
        avg_oos_sharpe = np.mean([r.oos_sharpe for r in results])
        print(f"\nAverage OOS return: {avg_oos_ret:.4f} bps/period")
        print(f"Average OOS Sharpe: {avg_oos_sharpe:.4f}")
        
        avg_is_sharpe = np.mean([r.is_sharpe for r in results])
        degradation = (avg_is_sharpe - avg_oos_sharpe) / avg_is_sharpe * 100 if avg_is_sharpe > 0 else 0
        print(f"\nSharpe degradation IS → OOS: {degradation:.1f}%")
    
    return results


def results_to_dataframe(results):
    """Convert WalkForwardResult list to DataFrame."""
    rows = []
    for r in results:
        rows.append({
            'period': r.period_name,
            'train': f"{r.train_start[:7]} to {r.train_end[:7]}",
            'test': f"{r.test_start[:7]} to {r.test_end[:7]}",
            'strategy': r.strategy_name,
            'is_return_bps': r.is_return_bps,
            'is_sharpe': r.is_sharpe,
            'oos_return_bps': r.oos_return_bps,
            'oos_sharpe': r.oos_sharpe,
            'oos_n_trades': r.oos_n_trades,
            'oos_total_return_pct': r.oos_total_return,
            'oos_bh_return_pct': r.oos_buy_hold_return,
            'oos_profitable': r.oos_return_bps > 0
        })
    return pd.DataFrame(rows)


def main():
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Walk-Forward Analysis & Optimization")
    parser.add_argument("data_path", nargs="?", default=None, help="Path to .bitstampUSD.csv")
    parser.add_argument("--tc", type=float, default=0.001, help="Transaction cost (default: 0.001)")
    parser.add_argument("--skip-full-wfo", action="store_true", help="Skip slow full WFO (only run WFA and WFO fixed s,l)")
    parser.add_argument("--debug", action="store_true", help="Show debug output for WFO optimization")
    parser.add_argument("--rolling", action="store_true", 
                        help="Use rolling window (1yr train, 1yr test) instead of expanding")
    args = parser.parse_args()
    
    # auto-detect data file
    if args.data_path is None:
        for filename in [".bitstampUSD.csv", "bitstampUSD.csv", "data.csv"]:
            if Path(filename).exists():
                args.data_path = filename
                break
    
    if args.data_path is None:
        print("ERROR: Data file not found!")
        return
    
    print(f"Loading data from: {args.data_path}")
    
    df = pd.read_csv(args.data_path, names=["Date", "Price", "Volume"])
    df['Date'] = pd.to_datetime(df['Date'], unit='s')
    df.set_index('Date', inplace=True)
    df = df.resample('5min').last()
    df = df.interpolate()
    df = df[df.index >= '2017-01-01']
    
    prices = df["Price"].values
    dates = df.index
    
    print(f"Data: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    print(f"Observations: {len(prices):,}")
    
    if args.rolling:
        periods = [
            ("2017-01-01", "2017-12-31", "2018-01-01", "2018-12-31"),
            ("2018-01-01", "2018-12-31", "2019-01-01", "2019-12-31"),
            ("2019-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
            ("2020-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
            ("2021-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
        ]
        print("Window type: Rolling (1yr train, 1yr test)")
    else:
        periods = [
            ("2017-01-01", "2018-12-31", "2019-01-01", "2019-12-31"),
            ("2017-01-01", "2019-12-31", "2020-01-01", "2020-12-31"),
            ("2017-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
            ("2017-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
        ]
        print("Window type: Expanding (growing train, 1yr test)")

    # WFA with fixed MAc(7,10,0.01,0,0)
    strategy_params = {'s': 7, 'l': 10, 'b': 0.01, 'd': 0, 'c': 0}
    
    start_time = time.time()
    wfa_results = run_walk_forward_analysis(
        prices, dates, strategy_params, periods,
        transaction_cost=args.tc, verbose=True
    )
    wfa_time = time.time() - start_time
    print(f"\nWFA completed in {wfa_time:.1f} seconds")
    
    # WFO: optimize all params
    wfo_results = None
    wfo_time = 0
    
    if not args.skip_full_wfo:
        # only optimize MA params (s,l,b), fix d=0 c=0
        # optimizing d,c leads to "don't trade" artifact
        param_grid = {
            's': [2, 3, 5, 7, 10],
            'l': [3, 5, 7, 10, 15],
            'b': [0, 0.00025, 0.0005, 0.001, 0.01],
            'd': [0],
            'c': [0]
        }
        
        start_time = time.time()
        wfo_results = run_walk_forward_optimization(
            prices, dates, periods, param_grid,
            transaction_cost=args.tc, verbose=True, debug=args.debug
        )
        wfo_time = time.time() - start_time
        print(f"\nWFO completed in {wfo_time:.1f} seconds")
    else:
        print("\n[Skipping full WFO - use without --skip-full-wfo to run]")
    
    # save results
    wfa_df = results_to_dataframe(wfa_results)
    
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    wfa_df.to_csv(output_dir / "wfa_results.csv", index=False)
    
    if wfo_results:
        wfo_df = results_to_dataframe(wfo_results)
        wfo_df.to_csv(output_dir / "wfo_results.csv", index=False)
    
    print(f"\nResults saved to {output_dir}/")
    print("  - wfa_results.csv")
    if wfo_results:
        print("  - wfo_results.csv")
    
    # WFO with fixed s,l,b - only optimize d,c
    start_time = time.time()
    wfo_fixed_results = run_walk_forward_optimization_fixed_sl(
        prices, dates, periods,
        s=7, l=10, b=0.01,
        d_values=[0, 1, 2],
        c_values=[0, 1, 2],
        transaction_cost=args.tc, verbose=True
    )
    wfo_fixed_time = time.time() - start_time
    print(f"\nWFO (fixed s,l) completed in {wfo_fixed_time:.1f} seconds")
    
    wfo_fixed_df = results_to_dataframe(wfo_fixed_results)
    wfo_fixed_df.to_csv(output_dir / "wfo_fixed_sl_results.csv", index=False)
    print(f"  - wfo_fixed_sl_results.csv")
    
    # final comparison
    print("\n" + "=" * 70)
    if wfo_results:
        print("FINAL COMPARISON: WFA vs WFO vs WFO (Fixed s,l)")
    else:
        print("FINAL COMPARISON: WFA vs WFO (Fixed s,l)")
    print("=" * 70)
    
    wfa_profitable = sum(1 for r in wfa_results if r.oos_return_bps > 0)
    wfo_profitable = sum(1 for r in wfo_results if r.oos_return_bps > 0) if wfo_results else 0
    wfo_fixed_profitable = sum(1 for r in wfo_fixed_results if r.oos_return_bps > 0)
    
    print(f"{'Method':<25} {'Profitable':<15} {'Avg OOS (bps)':<15} {'Avg Sharpe':<15}")
    print("-" * 70)
    print(f"{'WFA (all fixed)':<25} {wfa_profitable}/4{'':<12} {np.mean([r.oos_return_bps for r in wfa_results]):>10.4f} {np.mean([r.oos_sharpe for r in wfa_results]):>12.4f}")
    if wfo_results:
        print(f"{'WFO (all optimized)':<25} {wfo_profitable}/4{'':<12} {np.mean([r.oos_return_bps for r in wfo_results]):>10.4f} {np.mean([r.oos_sharpe for r in wfo_results]):>12.4f}")
    print(f"{'WFO (fixed s,l,b)':<25} {wfo_fixed_profitable}/4{'':<12} {np.mean([r.oos_return_bps for r in wfo_fixed_results]):>10.4f} {np.mean([r.oos_sharpe for r in wfo_fixed_results]):>12.4f}")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    methods = [
        ("WFA", wfa_profitable, np.mean([r.oos_return_bps for r in wfa_results])),
        ("WFO Fixed", wfo_fixed_profitable, np.mean([r.oos_return_bps for r in wfo_fixed_results]))
    ]
    if wfo_results:
        methods.insert(1, ("WFO", wfo_profitable, np.mean([r.oos_return_bps for r in wfo_results])))
    
    best_method = max(methods, key=lambda x: (x[1], x[2]))
    print(f"Best method: {best_method[0]} ({best_method[1]}/4 profitable, {best_method[2]:.4f} bps avg)")
    
    max_profitable = max(wfa_profitable, wfo_fixed_profitable)
    if wfo_results:
        max_profitable = max(max_profitable, wfo_profitable)
    
    if max_profitable >= 3:
        print("→ Strategy shows promise - consider further validation")
    elif max_profitable >= 2:
        print("→ Mixed results - no consistent OOS performance")
    else:
        print("→ All methods fail - confirms WRC null result")


if __name__ == "__main__":
    main()
