"""
Signal combination via LightGBM - Phase 3

Combines all TA strategy signals using LightGBM multiclass classifier.
Based on Bakker (2017) bandwidth-adjusted targets.

    python signal_combination.py .bitstampUSD.csv
    python signal_combination.py .bitstampUSD.csv --save-signals
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import time
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not installed. Run: pip install lightgbm")


# Strategy parameter grids (from thesis - matching Bakker 2017)
PARAMS = {
    'MA': {
        's': [2, 3, 5, 7, 10],
        'l': [3, 5, 7, 10, 15],
        'b': [0, 0.00025, 0.0005, 0.001, 0.01],
        'c': [0, 1, 2],
        'd': [0, 1, 2],
        'contrarian': True
    },
    'SR': {
        'n': [3, 5, 10, 20, 30],
        'b': [0, 0.00025, 0.0005, 0.001, 0.0025, 0.005],
        'c': [0, 1, 2],
        'd': [0, 1, 2],
        'contrarian': True  # 5×6×3×3 = 270 × 2 = 540
    },
    'CB': {
        'n': [5, 10, 15, 20, 30],
        'x': [0.01, 0.02, 0.04, 0.075],
        'b': [0, 0.00025, 0.0005, 0.001, 0.0015],
        'c': [0, 1, 2],
        'contrarian': True
    },
    'RSI': {
        'n': [3, 5, 7, 12, 18],
        'v': [10, 20, 30, 40],
        'c': [0, 1, 2],
        'd': [0, 1, 2],
        'contrarian': False
    },
    'BB': {
        'n': [3, 5, 7, 12, 18],
        'k': [0.25, 0.5, 1, 2],
        'c': [0, 1, 2],
        'd': [0, 1, 2],
        'contrarian': False
    },
    'OBV': {
        's': [2, 3, 5, 7, 10],
        'l': [3, 5, 7, 10, 15],
        'b': [0, 0.025, 0.05, 0.1, 0.5],
        'c': [0, 1, 2],
        'd': [0, 1, 2],
        'contrarian': False
    },
    'Filter': {
        'n': [3, 5, 10, 18],
        'x': [0.0005, 0.001, 0.0025, 0.005],
        'c': [0, 1, 2],
        'd': [0, 1, 2],
        'contrarian': False
    }
}


# --- signal generators ---

def generate_ma_signals(prices, s, l, b, d, c, contrarian):
    """MA crossover signals. Contrarian flips direction."""
    n = len(prices)
    if l >= n:
        return np.zeros(n)
    
    ma_short = pd.Series(prices).rolling(window=s, min_periods=s).mean().values
    ma_long = pd.Series(prices).rolling(window=l, min_periods=l).mean().values
    
    signals = np.zeros(n)
    for i in range(l - 1, n):
        if ma_long[i] > 0:
            ratio = ma_short[i] / ma_long[i] - 1
            if ratio > b:
                signals[i] = -1 if contrarian else 1
            elif ratio < -b:
                signals[i] = 1 if contrarian else -1
    
    return apply_filters(signals, d, c)


def generate_sr_signals(prices, n_periods, b, d, c, contrarian):
    """Support & Resistance breakout signals."""
    n = len(prices)
    if n_periods >= n:
        return np.zeros(n)
    
    signals = np.zeros(n)
    for i in range(n_periods, n):
        window = prices[i - n_periods:i]
        resistance = np.max(window)
        support = np.min(window)
        
        if prices[i] > resistance * (1 + b):
            signals[i] = -1 if contrarian else 1
        elif prices[i] < support * (1 - b):
            signals[i] = 1 if contrarian else -1
    
    return apply_filters(signals, d, c)


def generate_cb_signals(prices, n_periods, x, b, c, contrarian):
    """Channel breakout. No delay filter for CB."""
    n = len(prices)
    if n_periods >= n:
        return np.zeros(n)
    
    signals = np.zeros(n)
    for i in range(n_periods, n):
        window = prices[i - n_periods:i]
        high = np.max(window)
        low = np.min(window)
        
        if low > 0 and (high / low - 1) < x:
            if prices[i] > high * (1 + b):
                signals[i] = -1 if contrarian else 1
            elif prices[i] < low * (1 - b):
                signals[i] = 1 if contrarian else -1
    
    return apply_filters(signals, d=0, c=c)


def generate_rsi_signals(prices, n_periods, v, d, c):
    """RSI mean-reversion signals."""
    n = len(prices)
    if n_periods >= n:
        return np.zeros(n)
    
    changes = np.diff(prices, prepend=prices[0])
    
    signals = np.zeros(n)
    for i in range(n_periods, n):
        window = changes[i - n_periods + 1:i + 1]
        gains = np.sum(window[window > 0])
        losses = -np.sum(window[window < 0])
        
        if gains + losses > 0:
            rsi = 100 * gains / (gains + losses)
            if rsi < 50 - v:
                signals[i] = 1  # oversold
            elif rsi > 50 + v:
                signals[i] = -1  # overbought
    
    return apply_filters(signals, d, c)


def generate_bb_signals(prices, n_periods, k, d, c):
    """Bollinger band mean-reversion."""
    n = len(prices)
    if n_periods >= n:
        return np.zeros(n)
    
    ma = pd.Series(prices).rolling(window=n_periods, min_periods=n_periods).mean().values
    std = pd.Series(prices).rolling(window=n_periods, min_periods=n_periods).std().values
    
    signals = np.zeros(n)
    for i in range(n_periods, n):
        if std[i] > 0:
            upper = ma[i] + k * std[i]
            lower = ma[i] - k * std[i]
            
            if prices[i] < lower:
                signals[i] = 1
            elif prices[i] > upper:
                signals[i] = -1
    
    return apply_filters(signals, d, c)


def generate_obv_signals(prices, volume, s, l, b, d, c):
    """OBV moving average crossover."""
    n = len(prices)
    if l >= n:
        return np.zeros(n)
    
    obv = np.zeros(n)
    for i in range(1, n):
        if prices[i] > prices[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif prices[i] < prices[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]
    
    obv_ma_short = pd.Series(obv).rolling(window=s, min_periods=s).mean().values
    obv_ma_long = pd.Series(obv).rolling(window=l, min_periods=l).mean().values
    
    signals = np.zeros(n)
    for i in range(l - 1, n):
        if obv_ma_long[i] != 0:
            ratio = obv_ma_short[i] / obv_ma_long[i] - 1
            if ratio > b:
                signals[i] = 1
            elif ratio < -b:
                signals[i] = -1
    
    return apply_filters(signals, d, c)


def generate_filter_signals(prices, n_periods, x, d, c):
    """Filter rule signals."""
    n_obs = len(prices)
    if n_periods >= n_obs:
        return np.zeros(n_obs)
    
    signals = np.zeros(n_obs)
    for i in range(n_periods, n_obs):
        window = prices[i - n_periods:i]
        local_max = np.max(window)
        local_min = np.min(window)
        
        if local_min > 0 and (prices[i] - local_min) / local_min > x:
            signals[i] = 1
        elif local_max > 0 and (local_max - prices[i]) / local_max > x:
            signals[i] = -1
    
    return apply_filters(signals, d, c)


def apply_filters(signals, d, c):
    """
    Apply delay and holding period filters.
    Returns raw signals (with zeros = no opinion), not forward-filled.
    
    Matches Bakker (2017): signals are +1/-1/0 where 0 means "no opinion"
    """
    n = len(signals)
    result = signals.copy()
    
    # delay: signal must persist d periods before firing
    if d > 0:
        filtered = np.zeros(n)
        for i in range(d, n):
            if signals[i] != 0:
                if np.all(signals[i-d:i] == signals[i]):
                    filtered[i] = signals[i]
        result = filtered
    
    # holding period: suppress new signals for c periods after a signal fires
    if c > 0:
        filtered = np.zeros(n)
        last_signal_idx = -c - 1
        for i in range(n):
            if result[i] != 0:
                if i - last_signal_idx > c:
                    filtered[i] = result[i]
                    last_signal_idx = i
        result = filtered
    
    return result


def generate_all_signals(prices, volume=None, verbose=True):
    """
    Generate raw signal arrays for all ~3669 strategies.
    Signals: +1 (long), -1 (short), 0 (no opinion).
    """
    import time as time_module
    
    if volume is None:
        volume = np.ones(len(prices))
    
    all_signals = []
    all_names = []
    total_start = time_module.time()
    
    # MA (with contrarian)
    t0 = time_module.time()
    if verbose:
        print("Generating MA signals...", end=" ", flush=True)
    count = 0
    for s in PARAMS['MA']['s']:
        for l in PARAMS['MA']['l']:
            if s >= l:
                continue
            for b in PARAMS['MA']['b']:
                for d in PARAMS['MA']['d']:
                    for c in PARAMS['MA']['c']:
                        pos = generate_ma_signals(prices, s, l, b, d, c, contrarian=False)
                        all_signals.append(pos)
                        all_names.append(f"MA({s},{l},{b},{d},{c})")
                        
                        pos_c = generate_ma_signals(prices, s, l, b, d, c, contrarian=True)
                        all_signals.append(pos_c)
                        all_names.append(f"MAc({s},{l},{b},{d},{c})")
                        count += 2
    if verbose:
        print(f"{count} strategies [{time_module.time() - t0:.1f}s]")
    
    # SR (with contrarian)
    t0 = time_module.time()
    if verbose:
        print("Generating SR signals...", end=" ", flush=True)
    count = 0
    for n in PARAMS['SR']['n']:
        for b in PARAMS['SR']['b']:
            for d in PARAMS['SR']['d']:
                for c in PARAMS['SR']['c']:
                    pos = generate_sr_signals(prices, n, b, d, c, contrarian=False)
                    all_signals.append(pos)
                    all_names.append(f"SR({n},{b},{d},{c})")
                    
                    pos_c = generate_sr_signals(prices, n, b, d, c, contrarian=True)
                    all_signals.append(pos_c)
                    all_names.append(f"SRc({n},{b},{d},{c})")
                    count += 2
    if verbose:
        print(f"{count} strategies [{time_module.time() - t0:.1f}s]")
    
    # CB (with contrarian, no delay)
    t0 = time_module.time()
    if verbose:
        print("Generating CB signals...", end=" ", flush=True)
    count = 0
    for n in PARAMS['CB']['n']:
        for x in PARAMS['CB']['x']:
            for b in PARAMS['CB']['b']:
                for c in PARAMS['CB']['c']:
                    pos = generate_cb_signals(prices, n, x, b, c, contrarian=False)
                    all_signals.append(pos)
                    all_names.append(f"CB({n},{x},{b},{c})")
                    
                    pos_c = generate_cb_signals(prices, n, x, b, c, contrarian=True)
                    all_signals.append(pos_c)
                    all_names.append(f"CBc({n},{x},{b},{c})")
                    count += 2
    if verbose:
        print(f"{count} strategies [{time_module.time() - t0:.1f}s]")
    
    # RSI
    t0 = time_module.time()
    if verbose:
        print("Generating RSI signals...", end=" ", flush=True)
    count = 0
    for n in PARAMS['RSI']['n']:
        for v in PARAMS['RSI']['v']:
            for d in PARAMS['RSI']['d']:
                for c in PARAMS['RSI']['c']:
                    pos = generate_rsi_signals(prices, n, v, d, c)
                    all_signals.append(pos)
                    all_names.append(f"RSI({n},{v},{d},{c})")
                    count += 1
    if verbose:
        print(f"{count} strategies [{time_module.time() - t0:.1f}s]")
    
    # BB
    t0 = time_module.time()
    if verbose:
        print("Generating BB signals...", end=" ", flush=True)
    count = 0
    for n in PARAMS['BB']['n']:
        for k in PARAMS['BB']['k']:
            for d in PARAMS['BB']['d']:
                for c in PARAMS['BB']['c']:
                    pos = generate_bb_signals(prices, n, k, d, c)
                    all_signals.append(pos)
                    all_names.append(f"BB({n},{k},{d},{c})")
                    count += 1
    if verbose:
        print(f"{count} strategies [{time_module.time() - t0:.1f}s]")
    
    # OBV
    t0 = time_module.time()
    if verbose:
        print("Generating OBV signals...", end=" ", flush=True)
    count = 0
    for s in PARAMS['OBV']['s']:
        for l in PARAMS['OBV']['l']:
            if s >= l:
                continue
            for b in PARAMS['OBV']['b']:
                for d in PARAMS['OBV']['d']:
                    for c in PARAMS['OBV']['c']:
                        pos = generate_obv_signals(prices, volume, s, l, b, d, c)
                        all_signals.append(pos)
                        all_names.append(f"OBV({s},{l},{b},{d},{c})")
                        count += 1
    if verbose:
        print(f"{count} strategies [{time_module.time() - t0:.1f}s]")
    
    # Filter
    t0 = time_module.time()
    if verbose:
        print("Generating Filter signals...", end=" ", flush=True)
    count = 0
    for n in PARAMS['Filter']['n']:
        for x in PARAMS['Filter']['x']:
            for d in PARAMS['Filter']['d']:
                for c in PARAMS['Filter']['c']:
                    pos = generate_filter_signals(prices, n, x, d, c)
                    all_signals.append(pos)
                    all_names.append(f"F({n},{x},{d},{c})")
                    count += 1
    if verbose:
        print(f"{count} strategies [{time_module.time() - t0:.1f}s]")
    
    # Stack into matrix (int8 to save memory, signals are only -1/0/1)
    t0 = time_module.time()
    if verbose:
        print("Stacking signals...", end=" ", flush=True)
    
    signals = np.column_stack(all_signals).astype(np.int8)
    
    if verbose:
        print(f"[{time_module.time() - t0:.1f}s]")
        print(f"\nTotal: {len(all_names)} strategies")
        print(f"Memory: {signals.nbytes / 1e9:.2f} GB")
        
        n_zeros = np.sum(signals == 0)
        pct_zeros = n_zeros / signals.size * 100
        print(f"Signal sparsity: {pct_zeros:.1f}% zeros (no opinion)")
        print(f"Signal generation total: {time_module.time() - total_start:.1f}s")
    
    return signals, all_names


def deduplicate_signals(signals, names, verbose=True):
    """Remove duplicate signal columns via hashing."""
    import time as time_module
    t0 = time_module.time()
    
    if verbose:
        print(f"\nDeduplicating {signals.shape[1]} signals...")
    
    seen = {}
    unique_idx = []
    mapping = {}
    
    for i in range(signals.shape[1]):
        key = signals[:, i].tobytes()
        if key not in seen:
            seen[key] = i
            unique_idx.append(i)
        else:
            mapping[names[i]] = names[seen[key]]
    
    unique_signals = signals[:, unique_idx]
    unique_names = [names[i] for i in unique_idx]
    
    if verbose:
        print(f"  Unique: {len(unique_names)}")
        print(f"  Duplicates removed: {signals.shape[1] - len(unique_names)}")
        print(f"  Memory: {unique_signals.nbytes / 1e9:.2f} GB")
        print(f"  Time: {time_module.time() - t0:.1f}s")
    
    return unique_signals, unique_names, mapping


def create_target(prices, bandwidth_bps=10.0):
    """
    Classification target with bandwidth filter.
    y[t] = +1 if ret(t->t+1) > bandwidth, -1 if < -bandwidth, else 0.
    """
    returns = np.diff(prices) / prices[:-1]
    returns = np.append(returns, 0)
    
    bw = bandwidth_bps / 10000
    
    target = np.zeros(len(prices))
    target[returns > bw] = 1
    target[returns < -bw] = -1
    target[-1] = 0  # last target unknown
    
    return target


def get_quarterly_dates(start_year, end_year):
    """Generate quarterly (start, end) date tuples."""
    quarters = []
    for year in range(start_year, end_year + 1):
        quarters.append((f"{year}-01-01", f"{year}-03-31"))
        quarters.append((f"{year}-04-01", f"{year}-06-30"))
        quarters.append((f"{year}-07-01", f"{year}-09-30"))
        quarters.append((f"{year}-10-01", f"{year}-12-31"))
    return quarters


def walk_forward_lightgbm(signals, target, prices, dates, tc_bps=10.0, verbose=True):
    """
    Walk-forward with quarterly retraining, expanding window.
    Train: 2017-01-01 to end of quarter before test
    Test: Q1-2019 through Q4-2022
    """
    import time as time_module
    
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM not installed")
    
    results = {
        'periods': [], 'return_pct': [], 'sharpe': [],
        'n_trades': [], 'accuracy': []
    }
    
    lgb_params = {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'verbose': -1,
        'force_col_wise': True,
        'seed': 42
    }
    
    test_quarters = get_quarterly_dates(2019, 2022)
    train_start = "2017-01-01"
    
    if verbose:
        print(f"\n{'='*70}")
        print("WALK-FORWARD LIGHTGBM")
        print(f"{'='*70}")
        print(f"Features: {signals.shape[1]} unique signals (sparse)")
        print(f"TC: {tc_bps} bps")
        print(f"Test periods: {len(test_quarters)} quarters")
        print(f"{'='*70}\n")
    
    all_predictions = np.zeros(len(prices))
    wf_start = time_module.time()
    
    for i, (test_start, test_end) in enumerate(test_quarters):
        q_start = time_module.time()
        
        train_end = (pd.Timestamp(test_start) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        
        train_mask = (dates >= train_start) & (dates <= train_end)
        test_mask = (dates >= test_start) & (dates <= test_end)
        
        if test_mask.sum() == 0:
            continue
        
        X_train = signals[train_mask].astype(np.float32)
        y_train = target[train_mask]
        X_test = signals[test_mask].astype(np.float32)
        y_test = target[test_mask]
        
        valid_train = ~np.isnan(y_train)
        X_train = X_train[valid_train]
        y_train = y_train[valid_train]
        
        y_train_lgb = (y_train + 1).astype(int)  # map -1,0,1 -> 0,1,2
        
        if verbose:
            print(f"Q{(i%4)+1}-{2019 + i//4}: train {train_start[:7]}→{train_end[:7]} ({len(X_train):,}), test ({len(X_test):,})", end=" ")
        
        train_data = lgb.Dataset(X_train, label=y_train_lgb)
        model = lgb.train(lgb_params, train_data, num_boost_round=100)
        
        pred_proba = model.predict(X_test)
        pred_class = np.argmax(pred_proba, axis=1) - 1
        
        test_indices = np.where(test_mask)[0]
        all_predictions[test_indices] = pred_class
        
        # forward fill predictions -> positions
        test_positions = np.zeros(len(pred_class))
        last_pos = 0
        for j in range(len(pred_class)):
            if pred_class[j] != 0:
                last_pos = pred_class[j]
            test_positions[j] = last_pos
        
        test_prices = prices[test_mask]
        log_ret = np.log(test_prices[1:] / test_prices[:-1])
        strat_ret = test_positions[:-1] * log_ret
        
        # TC: amortized over period (Bakker's method)
        n_trades = np.sum(np.diff(test_positions) != 0)
        tc_per_period = (n_trades * 2 * tc_bps / 10000) / len(strat_ret)
        net_ret = strat_ret - tc_per_period
        
        total_return = np.exp(np.sum(net_ret)) - 1
        
        if len(net_ret) > 1 and np.std(net_ret) > 0:
            sharpe = np.mean(net_ret) / np.std(net_ret) * np.sqrt(252 * 288)
        else:
            sharpe = 0
        
        accuracy = np.mean(pred_class == y_test)
        
        results['periods'].append(f"Q{(i%4)+1}-{2019 + i//4}")
        results['return_pct'].append(total_return * 100)
        results['sharpe'].append(sharpe)
        results['n_trades'].append(n_trades)
        results['accuracy'].append(accuracy)
        
        profitable = "✓" if total_return > 0 else "✗"
        q_time = time_module.time() - q_start
        if verbose:
            print(f"→ {total_return*100:+.2f}% | trades: {n_trades} | {q_time:.1f}s {profitable}")
    
    results['all_predictions'] = all_predictions
    results['profitable_periods'] = sum(1 for r in results['return_pct'] if r > 0)
    results['total_periods'] = len(results['return_pct'])
    
    # compounded total return
    compounded = 1.0
    for r in results['return_pct']:
        compounded *= (1 + r / 100)
    results['total_return'] = (compounded - 1) * 100
    results['sum_return'] = sum(results['return_pct'])
    
    results['avg_sharpe'] = np.mean(results['sharpe'])
    
    wf_total_time = time_module.time() - wf_start
    
    if verbose:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"Profitable: {results['profitable_periods']}/{results['total_periods']}")
        print(f"Total return: {results['total_return']:.2f}%")
        print(f"Avg Sharpe: {results['avg_sharpe']:.3f}")
        print(f"Avg accuracy: {np.mean(results['accuracy'])*100:.1f}%")
        print(f"Walk-forward time: {wf_total_time:.1f}s")
    
    return results


def load_data(filepath):
    """Load and preprocess BTC tick data."""
    import time as time_module
    t0 = time_module.time()
    print(f"Loading data from {filepath}...")
    
    df = pd.read_csv(filepath, names=["Date", "Price", "Volume"])
    df['Date'] = pd.to_datetime(df['Date'], unit='s')
    df.set_index('Date', inplace=True)
    df = df.resample('5min').last()
    df = df.interpolate()
    df = df[df.index >= '2017-01-01']
    
    print(f"  Period: {df.index[0]} to {df.index[-1]}")
    print(f"  Observations: {len(df):,}")
    print(f"  Time: {time_module.time() - t0:.1f}s")
    
    return df, df['Price'].values, df['Volume'].values, df.index


def main():
    import argparse
    import time as time_module
    
    total_start = time_module.time()
    
    parser = argparse.ArgumentParser(description='Signal Combination with LightGBM')
    parser.add_argument('data_path', nargs='?', default='.bitstampUSD.csv',
                       help='Path to .bitstampUSD.csv')
    parser.add_argument('--bandwidth', type=float, default=10.0,
                       help='Bandwidth in bps (default: 10)')
    parser.add_argument('--tc', type=float, default=10.0,
                       help='Transaction costs in bps (default: 10)')
    parser.add_argument('--save-signals', action='store_true',
                       help='Save signals to parquet')
    parser.add_argument('--load-signals', action='store_true',
                       help='Load signals from parquet')
    
    args = parser.parse_args()
    
    print("="*70)
    print("SIGNAL COMBINATION - PHASE 3")
    print("="*70)
    print(f"Bandwidth: {args.bandwidth} bps")
    print(f"Transaction costs: {args.tc} bps")
    print("="*70 + "\n")
    
    df, prices, volume, dates = load_data(args.data_path)
    
    if args.load_signals and Path('signals_cache.npz').exists():
        print("\nLoading cached signals...")
        data = np.load('signals_cache.npz', allow_pickle=True)
        if 'signals' in data:
            signals = data['signals']
        else:
            signals = data['positions']  # old cache format
        names = data['names'].tolist()
    else:
        signals, names = generate_all_signals(prices, volume, verbose=True)
        signals, names, mapping = deduplicate_signals(signals, names, verbose=True)
        
        if args.save_signals:
            print("\nSaving signals to cache...")
            np.savez_compressed('signals_cache.npz', 
                               signals=signals,
                               names=np.array(names))
    
    # target
    t0 = time_module.time()
    print(f"\nCreating target (bandwidth={args.bandwidth} bps)...")
    target = create_target(prices, args.bandwidth)
    
    unique, counts = np.unique(target, return_counts=True)
    print("  Class distribution:")
    for u, c in zip(unique, counts):
        label = {-1: 'Short', 0: 'Hold', 1: 'Long'}[int(u)]
        print(f"    {label}: {c:,} ({c/len(target)*100:.1f}%)")
    print(f"  Time: {time_module.time() - t0:.1f}s")
    
    if LIGHTGBM_AVAILABLE:
        results = walk_forward_lightgbm(
            signals, target, prices, dates,
            tc_bps=args.tc, verbose=True
        )
        
        # compare with buy & hold
        bh_mask = (dates >= '2019-01-01') & (dates <= '2022-12-31')
        bh_prices = prices[bh_mask]
        bh_return = (bh_prices[-1] / bh_prices[0] - 1) * 100
        
        print(f"\nBuy & Hold (2019-2022): {bh_return:.2f}%")
        print(f"LightGBM total: {results['total_return']:.2f}%")
        print(f"Outperformance: {results['total_return'] - bh_return:.2f}%")
    else:
        print("\nInstall LightGBM: pip install lightgbm")
    
    print("\n" + "="*70)
    print("COMPLETE")
    print("="*70)
    print(f"Total execution time: {time_module.time() - total_start:.1f}s")


if __name__ == "__main__":
    main()
