"""
Bootstrap significance test for the combined strategy.
Politis-Romano stationary bootstrap, testing H0: Sharpe <= 0.

Requires: signals_cache.npz from signal_combination.py

    python bootstrap_significance.py
"""

import numpy as np
import pandas as pd
import time

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: numba not installed, bootstrap will be slow")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Error: lightgbm required")


# --- stationary bootstrap ---

if NUMBA_AVAILABLE:
    @njit
    def _bootstrap_indices(n, q, seed):
        np.random.seed(seed)
        indices = np.empty(n, dtype=np.int64)
        idx = np.random.randint(0, n)
        indices[0] = idx
        for t in range(1, n):
            if np.random.random() < q:
                idx = np.random.randint(0, n)
            else:
                idx = (idx + 1) % n
            indices[t] = idx
        return indices
else:
    def _bootstrap_indices(n, q, seed):
        """Politis-Romano bootstrap indices (pure python fallback)."""
        np.random.seed(seed)
        indices = np.empty(n, dtype=np.int64)
        idx = np.random.randint(0, n)
        indices[0] = idx
        for t in range(1, n):
            if np.random.random() < q:
                idx = np.random.randint(0, n)
            else:
                idx = (idx + 1) % n
            indices[t] = idx
        return indices


def calc_sharpe(returns, periods_per_year=252 * 288):
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year)


def bootstrap_sharpe_test(returns, B=1000, q=0.1, seed=42):
    """
    Bootstrap test: H0: Sharpe <= 0, H1: Sharpe > 0
    
    Returns (observed_sharpe, p_value, bootstrap_sharpes)
    """
    n = len(returns)
    observed = calc_sharpe(returns)
    
    # center returns under null (mean=0 -> sharpe=0)
    centered = returns - np.mean(returns)
    
    boot_sharpes = np.zeros(B)
    for b in range(B):
        idx = _bootstrap_indices(n, q, seed + b)
        boot_sharpes[b] = calc_sharpe(centered[idx])
    
    # p-value: fraction of bootstrap sharpes >= observed
    p_val = np.mean(boot_sharpes >= observed)
    
    return observed, p_val, boot_sharpes


# --- strategy returns ---

def get_combined_returns(bandwidth_bps=25.0, tc_bps=10.0):
    """Run LightGBM walk-forward, return concatenated net returns."""
    
    print("Loading data...")
    df = pd.read_csv('.bitstampUSD.csv', names=["Date", "Price", "Volume"])
    df['Date'] = pd.to_datetime(df['Date'], unit='s')
    df.set_index('Date', inplace=True)
    df = df.resample('5min').last().interpolate()
    df = df[df.index >= '2017-01-01']
    
    prices = df['Price'].values
    dates = df.index
    
    print("Loading cached signals...")
    cache = np.load('signals_cache.npz', allow_pickle=True)
    if 'signals' in cache:
        signals = cache['signals']
    else:
        signals = cache['positions']
    print(f"  Signals: {signals.shape}")
    
    ret_arr = np.diff(prices) / prices[:-1]
    ret_arr = np.append(ret_arr, 0)
    bw = bandwidth_bps / 10000
    
    target = np.zeros(len(prices))
    target[ret_arr > bw] = 1
    target[ret_arr < -bw] = -1
    target[-1] = 0
    
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
    
    all_net_returns = []
    
    test_quarters = []
    for year in range(2019, 2023):
        for q in range(1, 5):
            if q == 1:
                start, end = f"{year}-01-01", f"{year}-03-31"
            elif q == 2:
                start, end = f"{year}-04-01", f"{year}-06-30"
            elif q == 3:
                start, end = f"{year}-07-01", f"{year}-09-30"
            else:
                start, end = f"{year}-10-01", f"{year}-12-31"
            test_quarters.append((start, end))
    
    train_start = "2017-01-01"
    
    print(f"\nRunning walk-forward ({len(test_quarters)} quarters)...")
    
    for i, (test_start, test_end) in enumerate(test_quarters):
        train_end_dt = pd.to_datetime(test_start) - pd.Timedelta(days=1)
        train_end = train_end_dt.strftime('%Y-%m-%d')
        
        train_mask = (dates >= train_start) & (dates <= train_end)
        test_mask = (dates >= test_start) & (dates <= test_end)
        
        X_train = signals[train_mask].astype(np.float32)
        y_train = target[train_mask]
        X_test = signals[test_mask].astype(np.float32)
        
        valid = ~np.isnan(y_train)
        X_train = X_train[valid]
        y_train = y_train[valid]
        y_train_lgb = (y_train + 1).astype(int)
        
        train_data = lgb.Dataset(X_train, label=y_train_lgb)
        model = lgb.train(lgb_params, train_data, num_boost_round=100)
        
        pred_proba = model.predict(X_test)
        pred_class = np.argmax(pred_proba, axis=1) - 1
        
        test_positions = np.zeros(len(pred_class))
        last_pos = 0
        for j in range(len(pred_class)):
            if pred_class[j] != 0:
                last_pos = pred_class[j]
            test_positions[j] = last_pos
        
        test_prices = prices[test_mask]
        log_ret = np.log(test_prices[1:] / test_prices[:-1])
        strat_ret = test_positions[:-1] * log_ret
        
        n_trades = np.sum(np.diff(test_positions) != 0)
        tc_per_period = (n_trades * 2 * tc_bps / 10000) / len(strat_ret)
        net_ret = strat_ret - tc_per_period
        
        all_net_returns.extend(net_ret.tolist())
        print(f"  Q{(i%4)+1}-{2019 + i//4}: {len(net_ret)} returns, {n_trades} trades")
    
    all_net_returns = np.array(all_net_returns)
    
    info = {
        'n_returns': len(all_net_returns),
        'mean_return': np.mean(all_net_returns),
        'std_return': np.std(all_net_returns),
        'total_return': (np.exp(np.sum(all_net_returns)) - 1) * 100,
        'sharpe': calc_sharpe(all_net_returns)
    }
    
    return all_net_returns, info


def main():
    print("="*70)
    print("BOOTSTRAP SIGNIFICANCE TEST FOR COMBINED STRATEGY")
    print("="*70)
    print("\nParameters:")
    print("  Bandwidth: 25 bps")
    print("  TC: 10 bps")
    print("  Bootstrap iterations: 1000")
    print("  Politis-Romano q: 0.1 (mean block ~10)")
    print("="*70)
    
    returns, info = get_combined_returns(bandwidth_bps=25.0, tc_bps=10.0)
    
    print(f"\nSTRATEGY PERFORMANCE")
    print(f"Total returns: {info['n_returns']:,}")
    print(f"Total return: {info['total_return']:.2f}%")
    print(f"Observed Sharpe: {info['sharpe']:.3f}")
    
    # bootstrap
    print(f"\nBOOTSTRAP TEST")
    print("H0: True Sharpe <= 0")
    print("H1: True Sharpe > 0")
    print("\nRunning bootstrap...")
    
    t0 = time.time()
    observed, p_val, boot_sharpes = bootstrap_sharpe_test(
        returns, B=1000, q=0.1, seed=42
    )
    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.1f}s")
    
    print(f"\nRESULTS")
    print(f"Observed Sharpe: {observed:.3f}")
    print(f"Bootstrap Sharpe (mean): {np.mean(boot_sharpes):.3f}")
    print(f"Bootstrap Sharpe (std): {np.std(boot_sharpes):.3f}")
    print(f"P-value: {p_val:.4f}")
    
    if p_val < 0.01:
        print(f"\n*** SIGNIFICANT at 1% level (p < 0.01) ***")
    elif p_val < 0.05:
        print(f"\n** SIGNIFICANT at 5% level (p < 0.05) **")
    elif p_val < 0.10:
        print(f"\n* MARGINALLY SIGNIFICANT at 10% level (p < 0.10) *")
    else:
        print(f"\nNot statistically significant (p >= 0.10)")
    
    # robustness: different block lengths
    # TODO: could also vary B here, but 500 is probably fine for a sanity check
    print(f"\nROBUSTNESS (different q values)")
    for q in [0.05, 0.1, 0.2]:
        _, p, _ = bootstrap_sharpe_test(returns, B=500, q=q, seed=42)
        print(f"  q={q} (mean block ~{int(1/q)}): p-value = {p:.4f}")
    
    print(f"\n{'='*70}")
    print("COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
