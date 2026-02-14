"""
White's Reality Check for data snooping.

Based on White (2000), Politis-Romano (1994) stationary bootstrap,
and Sullivan, Timmermann & White (1999).

    from utils.white_reality_check import white_reality_check
    p_value, V_l, V_l_star = white_reality_check(returns_matrix, q=0.1, B=1000)
"""

import numpy as np
from numba import njit, prange
from typing import Tuple


@njit
def _stationary_bootstrap_indices(n, q, seed):
    """Politis-Romano (1994) stationary bootstrap indices."""
    np.random.seed(seed)
    indices = np.empty(n, dtype=np.int64)
    
    idx = np.random.randint(0, n)
    indices[0] = idx
    
    for t in range(1, n):
        u = np.random.random()
        if u < q:
            idx = np.random.randint(0, n)
        else:
            idx = (idx + 1) % n
        indices[t] = idx
    
    return indices


@njit(parallel=True)
def _bootstrap_loop(f, f_bar, sqrt_n, q, B, base_seed):
    """Main bootstrap loop, parallelized."""
    n, l = f.shape
    V_l_star = np.empty(B)
    
    for b in prange(B):
        indices = _stationary_bootstrap_indices(n, q, base_seed + b)
        
        # bootstrap mean per strategy
        f_bar_star = np.zeros(l)
        for k in range(l):
            total = 0.0
            for t in range(n):
                total += f[indices[t], k]
            f_bar_star[k] = total / n
        
        # centered bootstrap stat
        max_val = -np.inf
        for k in range(l):
            val = sqrt_n * (f_bar_star[k] - f_bar[k])
            if val > max_val:
                max_val = val
        
        V_l_star[b] = max_val
    
    return V_l_star


def white_reality_check(returns_matrix, q=0.1, B=1000, seed=42, verbose=True):
    """
    White's Reality Check.
    
    Tests H0: max_k E(f_k) <= 0  (no strategy beats benchmark)
    vs    H1: max_k E(f_k) > 0
    
    Parameters
    ----------
    returns_matrix : (n_periods, n_strategies) array of net log returns
    q : stationary bootstrap param, mean block = 1/q
    B : bootstrap iterations
    
    Returns (p_value, V_l, V_l_star)
    """
    if returns_matrix.ndim != 2:
        raise ValueError(f"returns_matrix must be 2D, got {returns_matrix.ndim}D")
    
    n, l = returns_matrix.shape
    
    if verbose:
        print(f"\nWhite's Reality Check")
        print(f"=" * 50)
        print(f"  Strategies: {l:,}")
        print(f"  Observations: {n:,}")
        print(f"  Bootstrap iterations: {B:,}")
        print(f"  Block parameter q: {q} (mean block length: {1/q:.1f})")
    
    f = returns_matrix.astype(np.float64)
    f_bar = np.mean(f, axis=0)
    
    sqrt_n = np.sqrt(n)
    V_l = sqrt_n * np.max(f_bar)
    
    best_idx = np.argmax(f_bar)
    best_mean_bps = f_bar[best_idx] * 10000
    
    if verbose:
        print(f"\nBest strategy:")
        print(f"  Index: {best_idx}")
        print(f"  Mean return: {best_mean_bps:.4f} bps/period")
        print(f"  Test statistic V_l: {V_l:.4f}")
        print(f"\nRunning bootstrap", end="", flush=True)
    
    V_l_star = _bootstrap_loop(f, f_bar, sqrt_n, q, B, seed)
    
    if verbose:
        print(" done.")
    
    p_value = np.mean(V_l_star > V_l)
    
    if verbose:
        print(f"\nResults:")
        print(f"  p-value: {p_value:.4f}")
        
        if p_value < 0.01:
            print(f"  Interpretation: Strong rejection of H0 (p < 0.01)")
        elif p_value < 0.05:
            print(f"  Interpretation: Reject H0 at 5% level")
        elif p_value < 0.10:
            print(f"  Interpretation: Marginal rejection at 10% level")
        else:
            print(f"  Interpretation: Cannot reject H0 - best strategy may be due to chance")
        print(f"=" * 50)
    
    return p_value, V_l, V_l_star


def compute_net_returns(gross_returns, positions, transaction_cost):
    """Net returns after TC, using position changes."""
    n = len(gross_returns)
    
    pos_changes = np.abs(np.diff(positions[:-1]))
    pos_changes = np.concatenate([[0], pos_changes])
    
    return gross_returns - transaction_cost * pos_changes


def run_wrc_on_results(results, transaction_cost=0.001, q=0.1, B=1000, seed=42, verbose=True):
    """
    Convenience wrapper: extract returns from BacktestResult list and run WRC.
    Uses gross returns with amortized TC adjustment.
    """
    if verbose:
        print(f"\nPreparing data for WRC...")
        print(f"  Total strategies: {len(results)}")
    
    returns_list = [r.returns_series for r in results]
    
    lengths = [len(r) for r in returns_list]
    if len(set(lengths)) != 1:
        raise ValueError(f"All strategies must have same number of returns. Got: {set(lengths)}")
    
    returns_matrix = np.column_stack(returns_list)
    
    if verbose:
        print(f"  Returns matrix shape: {returns_matrix.shape}")
    
    p_value, V_l, V_l_star = white_reality_check(
        returns_matrix, q=q, B=B, seed=seed, verbose=verbose
    )
    
    mean_returns = returns_matrix.mean(axis=0)
    best_idx = np.argmax(mean_returns)
    
    return p_value, V_l, V_l_star, best_idx


def wrc_robustness_check(returns_matrix, q_values=[0.05, 0.1, 0.2],
                         B_values=[500, 1000, 2000], seed=42):
    """Run WRC with different params to check sensitivity."""
    results = {}
    
    print("\nWRC Robustness Check")
    print("=" * 60)
    print(f"{'q':<8} {'B':<8} {'p-value':<12} {'V_l':<12}")
    print("-" * 60)
    
    for q in q_values:
        for B in B_values:
            p_value, V_l, _ = white_reality_check(
                returns_matrix, q=q, B=B, seed=seed, verbose=False
            )
            results[(q, B)] = {'p_value': p_value, 'V_l': V_l}
            print(f"{q:<8.2f} {B:<8} {p_value:<12.4f} {V_l:<12.4f}")
    
    print("=" * 60)
    
    p_vals = [r['p_value'] for r in results.values()]
    p_range = max(p_vals) - min(p_vals)
    
    print(f"\nP-value range: {min(p_vals):.4f} - {max(p_vals):.4f} (spread: {p_range:.4f})")
    
    if p_range < 0.02:
        print("Results are stable across parameters.")
    elif p_range < 0.05:
        print("Results show moderate sensitivity to parameters.")
    else:
        print("Warning: Results are sensitive to parameter choice. Consider increasing B.")
    
    return results


if __name__ == "__main__":
    print("Testing WRC with synthetic data...")
    
    np.random.seed(42)
    n_periods = 10000
    n_strategies = 100
    
    # all random, one with small positive drift
    returns = np.random.randn(n_periods, n_strategies) * 0.001
    returns[:, 0] += 0.00005
    
    print("\nScenario 1: One slightly profitable strategy among 100 random")
    p_value, V_l, V_l_star = white_reality_check(returns, B=500, verbose=True)
    
    print("\n" + "=" * 60)
    returns_null = np.random.randn(n_periods, n_strategies) * 0.001
    print("\nScenario 2: All random (H0 should hold)")
    p_null, _, _ = white_reality_check(returns_null, B=500, verbose=True)
