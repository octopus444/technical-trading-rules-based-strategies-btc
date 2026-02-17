"""
Run walk-forward once, cache everything for visualizations.
Output: strategy_data.pkl

    python collect_data.py
"""

import numpy as np
import pandas as pd
import pickle

try:
    import lightgbm as lgb
except ImportError:
    print("Error: lightgbm required")
    exit(1)


def run_strategy_and_collect(bandwidth_bps=25.0, tc_bps=10.0):
    """Run LightGBM signal combination, collect data for viz."""
    
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
    signals = cache['signals'] if 'signals' in cache else cache['positions']
    
    # target based on bandwidth
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
    
    all_dates = []
    all_returns = []
    all_positions = []
    all_prices = []
    quarterly_data = []
    trade_dates = []
    
    # build test quarters
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
            test_quarters.append((f"Q{q}-{year}", start, end))
    
    train_start = "2017-01-01"
    
    print(f"Running walk-forward ({len(test_quarters)} quarters)...")
    
    for i, (q_name, test_start, test_end) in enumerate(test_quarters):
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
        
        # forward fill predictions
        test_positions = np.zeros(len(pred_class))
        last_pos = 0
        for j in range(len(pred_class)):
            if pred_class[j] != 0:
                last_pos = pred_class[j]
            test_positions[j] = last_pos
        
        test_prices = prices[test_mask]
        test_dates = dates[test_mask]
        log_ret = np.log(test_prices[1:] / test_prices[:-1])
        strat_ret = test_positions[:-1] * log_ret
        
        n_trades = np.sum(np.diff(test_positions) != 0)
        tc_per_period = (n_trades * 2 * tc_bps / 10000) / len(strat_ret)
        net_ret = strat_ret - tc_per_period
        
        total_return = (np.exp(np.sum(net_ret)) - 1) * 100
        quarterly_data.append({
            'quarter': q_name,
            'return': total_return,
            'trades': n_trades,
            'sharpe': np.mean(net_ret) / np.std(net_ret) * np.sqrt(252*288) if np.std(net_ret) > 0 else 0
        })
        
        all_dates.extend(test_dates[:-1].tolist())
        all_returns.extend(net_ret.tolist())
        all_positions.extend(test_positions[:-1].tolist())
        all_prices.extend(test_prices[:-1].tolist())
        
        pos_changes = np.where(np.diff(test_positions) != 0)[0]
        for idx in pos_changes:
            trade_dates.append(test_dates[idx])
        
        print(f"  {q_name}: {total_return:+.1f}%")
    
    # assemble results
    df_results = pd.DataFrame({
        'date': all_dates,
        'return': all_returns,
        'position': all_positions,
        'price': all_prices
    })
    df_results['date'] = pd.to_datetime(df_results['date'])
    df_results.set_index('date', inplace=True)
    
    df_results['cumulative'] = np.exp(np.cumsum(df_results['return']))
    df_results['bh_return'] = np.log(df_results['price'] / df_results['price'].iloc[0])
    df_results['bh_cumulative'] = np.exp(df_results['bh_return'])
    df_results['peak'] = df_results['cumulative'].cummax()
    df_results['drawdown'] = (df_results['cumulative'] - df_results['peak']) / df_results['peak'] * 100
    
    df_daily = df_results['return'].resample('D').sum()
    df_daily = df_daily[df_daily != 0]
    
    trade_dates_s = pd.Series(1, index=pd.DatetimeIndex(trade_dates))
    monthly_trades = trade_dates_s.resample('M').sum().fillna(0)
    
    return {
        'df_results': df_results,
        'df_daily': df_daily,
        'quarterly_data': quarterly_data,
        'monthly_trades': monthly_trades,
        'trade_dates': trade_dates,
        'bandwidth_bps': bandwidth_bps,
        'tc_bps': tc_bps
    }


if __name__ == "__main__":
    print("="*70)
    print("COLLECTING STRATEGY DATA")
    print("="*70)
    
    data = run_strategy_and_collect(bandwidth_bps=25.0, tc_bps=10.0)
    
    output_file = 'strategy_data.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"\nData saved to: {output_file}")
    print("Now run: python visualizations.py")
