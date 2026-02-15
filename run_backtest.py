"""
Run all trading strategies on BTC data and generate results.
Includes White's Reality Check for data snooping.

    python run_backtest.py path/to/.bitstampUSD.csv
"""

import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils.data_loader import load_bitstamp_data, get_data_summary
from utils.backtester import results_to_dataframe, BacktestResult
from utils.white_reality_check import white_reality_check, wrc_robustness_check

from strategies.filter_strategy import backtest_all_filter_strategies
from strategies.ma_strategy import backtest_all_ma_strategies
from strategies.sr_strategy import backtest_all_sr_strategies
from strategies.cb_strategy import backtest_all_cb_strategies
from strategies.rsi_strategy import backtest_all_rsi_strategies
from strategies.bb_strategy import backtest_all_bb_strategies
from strategies.obv_strategy import backtest_all_obv_strategies


def compute_net_returns_matrix(results, n_observations, transaction_cost):
    """Net returns matrix (after TC) for all strategies."""
    n_periods = n_observations - 1
    n_strats = len(results)
    
    net_ret = np.zeros((n_periods, n_strats))
    
    for i, res in enumerate(results):
        tc_adj = (res.n_trades * 2 * transaction_cost) / n_periods
        net_ret[:, i] = res.returns_series - tc_adj
    
    return net_ret


def run_all_strategies(
    data_path, start_date="2017-01-01", end_date=None,
    transaction_cost=0.001, save_results=True, output_dir="results",
    verbose=True, run_wrc=True, wrc_B=1000, wrc_q=0.1, wrc_robustness=False
):
    """
    Run all 3669 strategy backtests.
    
    Contrarian variants only for MA, SR, CB (matching thesis methodology).
    Filter, RSI, BB, OBV have no contrarian versions.
    """
    start_time = time.time()
    
    if verbose:
        print("=" * 60)
        print("BITCOIN TECHNICAL TRADING STRATEGIES BACKTEST")
        print("=" * 60)
        print(f"\nLoading data from: {data_path}")
    
    df = load_bitstamp_data(data_path, start_date=start_date, end_date=end_date)
    prices = df["Price"].values
    
    if verbose:
        summary = get_data_summary(df)
        print(f"  Period: {summary['start_date']} to {summary['end_date']}")
        print(f"  Observations: {summary['n_observations']:,}")
        print(f"  Price range: ${summary['price_min']:.2f} - ${summary['price_max']:.2f}")
        print(f"  Buy & hold return: {summary['total_return']:.1f}%")
        print(f"\nRunning 3,669 strategies...")
    
    all_results = []
    
    # strategies WITH contrarian (MA, SR, CB only)
    with_contrarian = [
        ("MA", backtest_all_ma_strategies),
        ("SR", backtest_all_sr_strategies),
        ("CB", backtest_all_cb_strategies),
    ]
    
    # strategies WITHOUT contrarian
    no_contrarian = [
        ("Filter", backtest_all_filter_strategies),
        ("RSI", backtest_all_rsi_strategies),
        ("BB", backtest_all_bb_strategies),
    ]
    
    for name, runner in with_contrarian:
        if verbose:
            print(f"\n{name} strategies...")
        results = runner(
            prices=prices, include_contrarian=True,
            transaction_cost=transaction_cost, verbose=verbose
        )
        all_results.extend(results)
        if verbose:
            print(f"  {name}: {len(results)} strategies completed")
    
    for name, runner in no_contrarian:
        if verbose:
            print(f"\n{name} strategies...")
        results = runner(
            prices=prices, include_contrarian=False,
            transaction_cost=transaction_cost, verbose=verbose
        )
        all_results.extend(results)
        if verbose:
            print(f"  {name}: {len(results)} strategies completed")
    
    # OBV needs volume data
    if verbose:
        print(f"\nOBV strategies...")
    obv_results = backtest_all_obv_strategies(
        prices=prices, volumes=df["Volume"].values,
        include_contrarian=False,
        transaction_cost=transaction_cost, verbose=verbose
    )
    all_results.extend(obv_results)
    if verbose:
        print(f"  OBV: {len(obv_results)} strategies completed")
    
    results_df = results_to_dataframe(all_results)
    results_df = results_df.sort_values("mean_excess_return_bps", ascending=False)
    
    # White's Reality Check
    wrc_p_value = None
    
    if run_wrc:
        if verbose:
            print("\n" + "=" * 70)
            print("WHITE'S REALITY CHECK FOR DATA SNOOPING")
            print("=" * 70)
        
        net_ret_matrix = compute_net_returns_matrix(all_results, len(prices), transaction_cost)
        
        wrc_p_value, V_l, V_l_star = white_reality_check(
            net_ret_matrix, q=wrc_q, B=wrc_B, seed=42, verbose=verbose
        )
        
        mean_net = net_ret_matrix.mean(axis=0)
        best_idx = np.argmax(mean_net)
        best_strategy = all_results[best_idx]
        
        if verbose:
            print(f"\nBest strategy (net returns): {best_strategy.strategy_name}")
            if wrc_p_value < 0.01:
                print(f"✓ SURVIVES at 1% significance level (p = {wrc_p_value:.4f})")
            elif wrc_p_value < 0.05:
                print(f"✓ SURVIVES at 5% significance level (p = {wrc_p_value:.4f})")
            elif wrc_p_value < 0.10:
                print(f"~ Marginal at 10% level (p = {wrc_p_value:.4f})")
            else:
                print(f"✗ Does NOT survive data snooping correction (p = {wrc_p_value:.4f})")
                print("  The best strategy's performance may be due to chance.")
        
        if wrc_robustness:
            if verbose:
                print("\n")
            wrc_robustness_check(
                net_ret_matrix,
                q_values=[0.05, 0.1, 0.2],
                B_values=[500, 1000, 2000],
                seed=42
            )
    
    # save
    if save_results:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        results_df.to_csv(output_path / "all_results.csv", index=False)
        
        top_n = min(100, len(results_df))
        results_df.head(top_n).to_csv(output_path / "top_strategies_before_tc.csv", index=False)
        
        results_df_tc = results_df[results_df["betc"] > 10].copy()
        results_df_tc.to_csv(output_path / "profitable_after_tc.csv", index=False)
        
        if wrc_p_value is not None:
            wrc_summary = pd.DataFrame([{
                'wrc_p_value': wrc_p_value,
                'wrc_B': wrc_B,
                'wrc_q': wrc_q,
                'best_strategy': best_strategy.strategy_name,
                'n_strategies': len(all_results),
                'n_observations': len(prices),
                'transaction_cost_bps': transaction_cost * 10000
            }])
            wrc_summary.to_csv(output_path / "wrc_results.csv", index=False)
        
        if verbose:
            print(f"\nResults saved to {output_path}/")
            print(f"  - all_results.csv ({len(results_df)} strategies)")
            print(f"  - top_strategies_before_tc.csv (top {top_n})")
            print(f"  - profitable_after_tc.csv ({len(results_df_tc)} with BETC > 10 bps)")
            if wrc_p_value is not None:
                print(f"  - wrc_results.csv (White's Reality Check)")
    
    # summary tables
    if verbose:
        tc_bps = transaction_cost * 10000
        n_obs = len(prices)
        
        results_df["net_excess_return_bps"] = (
            results_df["mean_excess_return_bps"] - 
            (results_df["n_trades"] * 2 * transaction_cost / n_obs) * 10000
        )
        
        def parse_strategy(name):
            is_contrarian = False
            for prefix in ["Fc", "MAc", "SRc", "CBc", "RSIc", "BBc", "OBVc"]:
                if name.startswith(prefix + "("):
                    is_contrarian = True
                    break
            
            if name.startswith("F(") or name.startswith("Fc("):
                return "Filter", is_contrarian
            elif name.startswith("MA(") or name.startswith("MAc("):
                return "Double moving average", is_contrarian
            elif name.startswith("SR(") or name.startswith("SRc("):
                return "Support & resistance", is_contrarian
            elif name.startswith("CB(") or name.startswith("CBc("):
                return "Channel breakout", is_contrarian
            elif name.startswith("RSI(") or name.startswith("RSIc("):
                return "Relative strength indicator", is_contrarian
            elif name.startswith("BB(") or name.startswith("BBc("):
                return "Bollinger bands", is_contrarian
            elif name.startswith("OBV(") or name.startswith("OBVc("):
                return "On-balance volume average", is_contrarian
            return "Unknown", is_contrarian
        
        results_df["class"] = results_df["strategy"].apply(lambda x: parse_strategy(x)[0])
        results_df["is_contrarian"] = results_df["strategy"].apply(lambda x: parse_strategy(x)[1])
        
        class_order = [
            "Filter", "Double moving average", "Support & resistance",
            "Channel breakout", "Relative strength indicator", 
            "Bollinger bands", "On-balance volume average"
        ]
        
        # profitability tables (without TC)
        print("\n" + "=" * 70)
        print("NUMBER OF PROFITABLE STRATEGIES (WITHOUT TRANSACTION COSTS)")
        print("=" * 70)
        
        print(f"{'Class':<30} {'Tested':>8} {'MER>0':>8} {'SR>0':>8}")
        print("-" * 70)
        
        print("Standard strategies")
        total_std_tested = 0
        total_std_mer = 0
        total_std_sr = 0
        
        for cls in class_order:
            subset = results_df[(results_df["class"] == cls) & (~results_df["is_contrarian"])]
            tested = len(subset)
            mer_pos = len(subset[subset["mean_excess_return_bps"] > 0])
            sr_pos = len(subset[subset["sharpe_ratio_diff"] > 0])
            print(f"  {cls:<28} {tested:>8} {mer_pos:>8} {sr_pos:>8}")
            total_std_tested += tested
            total_std_mer += mer_pos
            total_std_sr += sr_pos
        
        print(f"  {'Subtotal':<28} {total_std_tested:>8} {total_std_mer:>8} {total_std_sr:>8}")
        
        print("Contrarian strategies")
        total_con_tested = 0
        total_con_mer = 0
        total_con_sr = 0
        
        for cls in class_order:
            subset = results_df[(results_df["class"] == cls) & (results_df["is_contrarian"])]
            tested = len(subset)
            mer_pos = len(subset[subset["mean_excess_return_bps"] > 0])
            sr_pos = len(subset[subset["sharpe_ratio_diff"] > 0])
            print(f"  {cls:<28} {tested:>8} {mer_pos:>8} {sr_pos:>8}")
            total_con_tested += tested
            total_con_mer += mer_pos
            total_con_sr += sr_pos
        
        print(f"  {'Subtotal':<28} {total_con_tested:>8} {total_con_mer:>8} {total_con_sr:>8}")
        
        print("-" * 70)
        total_tested = total_std_tested + total_con_tested
        total_mer = total_std_mer + total_con_mer
        total_sr = total_std_sr + total_con_sr
        print(f"{'Total':<30} {total_tested:>8} {total_mer:>8} {total_sr:>8}")
        
        # with TC
        print("\n" + "=" * 70)
        print(f"NUMBER OF PROFITABLE STRATEGIES (WITH TC = {tc_bps:.0f} bps)")
        print("=" * 70)
        
        print(f"{'Class':<30} {'Tested':>8} {'MER>0':>8} {'SR>0':>8}")
        print("-" * 70)
        
        print("Standard strategies")
        total_std_tested = 0
        total_std_mer = 0
        total_std_sr = 0
        
        for cls in class_order:
            subset = results_df[(results_df["class"] == cls) & (~results_df["is_contrarian"])]
            tested = len(subset)
            mer_pos = len(subset[subset["net_excess_return_bps"] > 0])
            sr_pos = len(subset[subset["sharpe_ratio_diff"] > 0])
            print(f"  {cls:<28} {tested:>8} {mer_pos:>8} {sr_pos:>8}")
            total_std_tested += tested
            total_std_mer += mer_pos
            total_std_sr += sr_pos
        
        print(f"  {'Subtotal':<28} {total_std_tested:>8} {total_std_mer:>8} {total_std_sr:>8}")
        
        print("Contrarian strategies")
        total_con_tested = 0
        total_con_mer = 0
        total_con_sr = 0
        
        for cls in class_order:
            subset = results_df[(results_df["class"] == cls) & (results_df["is_contrarian"])]
            tested = len(subset)
            mer_pos = len(subset[subset["net_excess_return_bps"] > 0])
            sr_pos = len(subset[subset["sharpe_ratio_diff"] > 0])
            print(f"  {cls:<28} {tested:>8} {mer_pos:>8} {sr_pos:>8}")
            total_con_tested += tested
            total_con_mer += mer_pos
            total_con_sr += sr_pos
        
        print(f"  {'Subtotal':<28} {total_con_tested:>8} {total_con_mer:>8} {total_con_sr:>8}")
        
        print("-" * 70)
        total_tested = total_std_tested + total_con_tested
        total_mer = total_std_mer + total_con_mer
        total_sr = total_std_sr + total_con_sr
        print(f"{'Total':<30} {total_tested:>8} {total_mer:>8} {total_sr:>8}")
        
        # top 10
        print("\n" + "=" * 70)
        print("TOP 10 STRATEGIES BY MEAN EXCESS RETURN (BEFORE TC)")
        print("=" * 70)
        top_cols = ["strategy", "mean_excess_return_bps", "sharpe_ratio", "n_trades", "betc"]
        print(results_df.nlargest(10, "mean_excess_return_bps")[top_cols].to_string(index=False))
        
        print("\n" + "=" * 70)
        print(f"TOP 10 STRATEGIES BY MEAN EXCESS RETURN (AFTER TC, h = {tc_bps:.0f} bps)")
        print("=" * 70)
        top_cols_tc = ["strategy", "net_excess_return_bps", "sharpe_ratio", "n_trades"]
        print(results_df.nlargest(10, "net_excess_return_bps")[top_cols_tc].to_string(index=False))
        
        # WRC summary
        if wrc_p_value is not None:
            print("\n" + "=" * 70)
            print("WHITE'S REALITY CHECK SUMMARY")
            print("=" * 70)
            print(f"  Test: H0: max E(f_k) <= 0 vs H1: max E(f_k) > 0")
            print(f"  Bootstrap iterations: {wrc_B}")
            print(f"  Block parameter q: {wrc_q} (mean block: {1/wrc_q:.0f} periods)")
            print(f"  p-value: {wrc_p_value:.4f}")
            
            if wrc_p_value < 0.05:
                print(f"\n  ✓ Best strategy survives data snooping correction")
                print(f"    {best_strategy.strategy_name}")
            else:
                print(f"\n  ✗ No strategy survives data snooping correction")
                print(f"    Best apparent strategy may be due to chance")
        
        profitable_before = len(results_df[results_df["mean_excess_return_bps"] > 0])
        profitable_after = len(results_df[results_df["net_excess_return_bps"] > 0])
        
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)
        print(f"  Profitable before TC: {profitable_before} / {len(results_df)} ({100*profitable_before/len(results_df):.1f}%)")
        print(f"  Profitable after TC:  {profitable_after} / {len(results_df)} ({100*profitable_after/len(results_df):.1f}%)")
        if wrc_p_value is not None:
            survives = "Yes" if wrc_p_value < 0.05 else "No"
            print(f"  Survives WRC (5%):    {survives} (p = {wrc_p_value:.4f})")
        
        elapsed = time.time() - start_time
        print(f"\n  Total time: {elapsed:.1f} seconds")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Run Bitcoin technical trading strategies backtest"
    )
    parser.add_argument("data_path", nargs="?", default=None,
                        help="Path to .bitstampUSD.csv file")
    parser.add_argument("--start-date", default="2017-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--tc", type=float, default=0.001,
                        help="Transaction cost (default: 0.001 = 10 bps)")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--no-wrc", action="store_true", help="Skip White's Reality Check")
    parser.add_argument("--wrc-B", type=int, default=1000)
    parser.add_argument("--wrc-q", type=float, default=0.1)
    parser.add_argument("--wrc-robustness", action="store_true",
                        help="Run WRC robustness check with multiple parameters")
    
    args = parser.parse_args()
    
    # auto-detect data file
    if args.data_path is None:
        for filename in [".bitstampUSD.csv", "bitstampUSD.csv", "data.csv"]:
            if Path(filename).exists():
                args.data_path = filename
                break
    
    if args.data_path is None:
        print("ERROR: Data file not found!")
        print("Please make sure .bitstampUSD.csv is in the current folder")
        print("Or specify path: python run_backtest.py path/to/data.csv")
        return
    
    print(f"Using data file: {args.data_path}")
    
    run_all_strategies(
        data_path=args.data_path,
        start_date=args.start_date,
        end_date=args.end_date,
        transaction_cost=args.tc,
        output_dir=args.output_dir,
        run_wrc=not args.no_wrc,
        wrc_B=args.wrc_B,
        wrc_q=args.wrc_q,
        wrc_robustness=args.wrc_robustness
    )


if __name__ == "__main__":
    main()
