"""
Visualizations for the signal combination strategy.
Reads from strategy_data.pkl (run collect_data.py first).

Outputs PNG files to ./figures/

    python visualizations.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-white')
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

COLORS = {
    'strategy': '#2E86AB',
    'benchmark': '#A23B72',
    'positive': '#28A745',
    'negative': '#DC3545',
    'neutral': '#6C757D',
    'drawdown': '#E63946',
}


def load_cached_data(filepath='strategy_data.pkl'):
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found!")
        print("Run 'python collect_data.py' first to generate data.")
        exit(1)
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded data from {filepath}")
    return data


def calculate_metrics(df_results, quarterly_data):
    """Calculate key performance metrics for summary table."""
    
    returns = df_results['return'].values
    cumulative = df_results['cumulative'].values
    positions = df_results['position'].values
    
    total_return = (cumulative[-1] - 1) * 100
    
    # CAGR over 4 years (2019-2022)
    years = 4
    cagr = (cumulative[-1] ** (1/years) - 1) * 100
    
    ann_vol = np.std(returns) * np.sqrt(252 * 288) * 100
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 288) if np.std(returns) > 0 else 0
    
    # sortino
    neg_ret = returns[returns < 0]
    downside_std = np.std(neg_ret) if len(neg_ret) > 0 else np.std(returns)
    sortino = np.mean(returns) / downside_std * np.sqrt(252 * 288) if downside_std > 0 else 0
    
    peak = np.maximum.accumulate(cumulative)
    dd = (cumulative - peak) / peak
    max_dd = np.min(dd) * 100
    
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    
    # trade-level stuff
    pos_changes = np.where(np.diff(positions) != 0)[0]
    n_trades = len(pos_changes)
    
    trade_pnls = []
    for i in range(len(pos_changes)):
        start = pos_changes[i]
        end = pos_changes[i+1] if i+1 < len(pos_changes) else len(returns)
        trade_pnls.append(np.sum(returns[start:end]))
    trade_pnls = np.array(trade_pnls)
    
    win_rate = np.mean(trade_pnls > 0) * 100 if len(trade_pnls) > 0 else 0
    
    gross_profit = np.sum(trade_pnls[trade_pnls > 0])
    gross_loss = np.abs(np.sum(trade_pnls[trade_pnls < 0]))
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    return {
        'Total Return': f"{total_return:.1f}%",
        'CAGR': f"{cagr:.1f}%",
        'Ann. Volatility': f"{ann_vol:.1f}%",
        'Sharpe Ratio': f"{sharpe:.3f}",
        'Sortino Ratio': f"{sortino:.3f}",
        'Max Drawdown': f"{max_dd:.1f}%",
        'Calmar Ratio': f"{calmar:.3f}",
        'Win Rate': f"{win_rate:.1f}%",
        'Profit Factor': f"{pf:.2f}",
        'Total Trades': f"{n_trades}",
    }


def get_trade_pnls(df_results):
    returns = df_results['return'].values
    positions = df_results['position'].values
    pos_changes = np.where(np.diff(positions) != 0)[0]
    
    pnls = []
    for i in range(len(pos_changes)):
        start = pos_changes[i]
        end = pos_changes[i+1] if i+1 < len(pos_changes) else len(returns)
        pnls.append((np.exp(np.sum(returns[start:end])) - 1) * 100)
    return np.array(pnls)


# --- plots ---

def plot_equity_curve(df_results, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df_results.index, df_results['cumulative'], 
            color=COLORS['strategy'], linewidth=1.5)
    
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Cumulative P&L (normalized)', fontsize=11)
    ax.set_title('Cumulative P&L', fontsize=14, fontweight='bold')
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.set_xlim(df_results.index[0], df_results.index[-1])
    ax.tick_params(axis='both', which='major', length=5)
    plt.xticks(rotation=45)
    
    final_val = df_results['cumulative'].iloc[-1]
    ax.annotate(f'{final_val:.2f}', 
                xy=(df_results.index[-1], final_val),
                xytext=(10, 0), textcoords='offset points',
                fontsize=10, fontweight='bold', color=COLORS['strategy'])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_strategy_vs_benchmark(df_results, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df_results.index, df_results['cumulative'], 
            color=COLORS['strategy'], linewidth=1.5, label='Strategy')
    ax.plot(df_results.index, df_results['bh_cumulative'], 
            color=COLORS['benchmark'], linewidth=1.5, alpha=0.7, label='Buy & Hold')
    
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Cumulative P&L (normalized)', fontsize=11)
    ax.set_title('Strategy vs Buy & Hold', fontsize=14, fontweight='bold')
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.set_xlim(df_results.index[0], df_results.index[-1])
    ax.tick_params(axis='both', which='major', length=5)
    plt.xticks(rotation=45)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_drawdown(df_results, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 4))
    
    ax.fill_between(df_results.index, df_results['drawdown'], 0,
                    color=COLORS['drawdown'], alpha=0.5)
    ax.plot(df_results.index, df_results['drawdown'], 
            color=COLORS['drawdown'], linewidth=0.8)
    
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Drawdown (%)', fontsize=11)
    ax.set_title('Drawdown', fontsize=14, fontweight='bold')
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.set_xlim(df_results.index[0], df_results.index[-1])
    ax.tick_params(axis='both', which='major', length=5)
    plt.xticks(rotation=45)
    
    max_dd = df_results['drawdown'].min()
    max_dd_date = df_results['drawdown'].idxmin()
    ax.annotate(f'Max: {max_dd:.1f}%', 
                xy=(max_dd_date, max_dd),
                xytext=(10, -20), textcoords='offset points',
                fontsize=10, fontweight='bold', color=COLORS['drawdown'],
                arrowprops=dict(arrowstyle='->', color=COLORS['drawdown']))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_trade_pnl_distribution(trade_pnls, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pos = trade_pnls[trade_pnls >= 0]
    neg = trade_pnls[trade_pnls < 0]
    
    bins = np.linspace(np.percentile(trade_pnls, 1), np.percentile(trade_pnls, 99), 40)
    
    ax.hist(pos, bins=bins, color=COLORS['positive'], alpha=0.7, label=f'Winners ({len(pos)})')
    ax.hist(neg, bins=bins, color=COLORS['negative'], alpha=0.7, label=f'Losers ({len(neg)})')
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.axvline(x=np.mean(trade_pnls), color=COLORS['strategy'], linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(trade_pnls):.2f}%')
    
    ax.set_xlabel('Trade Return (%)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Trade P&L Distribution', fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', which='major', length=5)
    ax.legend(loc='upper right')
    
    stats_text = f'Mean: {np.mean(trade_pnls):.2f}%\nMedian: {np.median(trade_pnls):.2f}%\nStd: {np.std(trade_pnls):.2f}%'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_quarterly_returns(quarterly_data, save_path=None):
    fig, ax = plt.subplots(figsize=(14, 6))
    
    quarters = [q['quarter'] for q in quarterly_data]
    rets = [q['return'] for q in quarterly_data]
    colors = [COLORS['positive'] if r >= 0 else COLORS['negative'] for r in rets]
    
    bars = ax.bar(quarters, rets, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    ax.set_xlabel('Quarter', fontsize=11)
    ax.set_ylabel('Return (%)', fontsize=11)
    ax.set_title('Quarterly Returns', fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', which='major', length=5)
    plt.xticks(rotation=45)
    
    for bar, ret in zip(bars, rets):
        h = bar.get_height()
        ax.annotate(f'{ret:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3 if h >= 0 else -12),
                    textcoords="offset points",
                    ha='center', va='bottom' if h >= 0 else 'top',
                    fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_monthly_trades(monthly_trades, save_path=None):
    fig, ax = plt.subplots(figsize=(14, 5))
    
    ax.bar(monthly_trades.index, monthly_trades.values, 
           color=COLORS['strategy'], alpha=0.7, width=20)
    
    ax.set_xlabel('Month', fontsize=11)
    ax.set_ylabel('Trades', fontsize=11)
    ax.set_title('Monthly Trading Frequency', fontsize=14, fontweight='bold')
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.set_xlim(monthly_trades.index[0] - pd.Timedelta(days=15), 
                monthly_trades.index[-1] + pd.Timedelta(days=15))
    ax.tick_params(axis='both', which='major', length=5)
    plt.xticks(rotation=45)
    
    avg = monthly_trades.mean()
    ax.axhline(y=avg, color=COLORS['negative'], linestyle='--', 
               linewidth=2, label=f'Avg: {avg:.0f}')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_rolling_sharpe(df_results, window_days=180, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 5))
    
    daily_ret = df_results['return'].resample('D').sum()
    roll_mean = daily_ret.rolling(window=window_days).mean()
    roll_std = daily_ret.rolling(window=window_days).std()
    roll_sharpe = (roll_mean / roll_std) * np.sqrt(252)
    
    ax.plot(roll_sharpe.index, roll_sharpe.values, 
            color=COLORS['strategy'], linewidth=1.5)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.axhline(y=roll_sharpe.mean(), color=COLORS['benchmark'], linestyle='--', 
               linewidth=1.5, alpha=0.7, label=f'Avg: {roll_sharpe.mean():.2f}')
    
    ax.fill_between(roll_sharpe.index, roll_sharpe.values, 0,
                    where=roll_sharpe.values >= 0, color=COLORS['positive'], alpha=0.3)
    ax.fill_between(roll_sharpe.index, roll_sharpe.values, 0,
                    where=roll_sharpe.values < 0, color=COLORS['negative'], alpha=0.3)
    
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Sharpe', fontsize=11)
    ax.set_title(f'Rolling Sharpe ({window_days}-day)', fontsize=14, fontweight='bold')
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.set_xlim(roll_sharpe.dropna().index[0], roll_sharpe.index[-1])
    ax.tick_params(axis='both', which='major', length=5)
    plt.xticks(rotation=45)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def create_metrics_table(metrics, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off')
    
    rows = list(metrics.keys())
    values = list(metrics.values())
    
    table = ax.table(
        cellText=[[v] for v in values],
        rowLabels=rows,
        colLabels=['Value'],
        cellLoc='center',
        rowLoc='right',
        loc='center',
        colWidths=[0.4]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.5, 1.8)
    
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor(COLORS['strategy'])
            cell.set_text_props(color='white')
        elif key[1] == -1:
            cell.set_text_props(fontweight='bold')
            cell.set_facecolor('#f0f0f0')
    
    ax.set_title('Key Performance Metrics', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_monthly_returns_heatmap(df_results, save_path=None):
    """Year x month heatmap of returns."""
    monthly_ret = df_results['return'].resample('M').sum()
    monthly_pct = (np.exp(monthly_ret) - 1) * 100
    
    df_m = pd.DataFrame({
        'year': monthly_pct.index.year,
        'month': monthly_pct.index.month,
        'return': monthly_pct.values
    })
    
    pivot = df_m.pivot(index='year', columns='month', values='return')
    pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-30, vmax=30)
    
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                color = 'white' if abs(val) > 15 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', 
                       fontsize=9, color=color, fontweight='bold')
    
    ax.set_title('Monthly Returns (%)', fontsize=14, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Return (%)', fontsize=10)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def main():
    print("="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    fig_dir = './figures'
    os.makedirs(fig_dir, exist_ok=True)
    
    print("\nLoading cached data...")
    data = load_cached_data('strategy_data.pkl')
    
    df_results = data['df_results']
    quarterly_data = data['quarterly_data']
    monthly_trades = data['monthly_trades']
    
    trade_pnls = get_trade_pnls(df_results)
    
    print("\nCalculating metrics...")
    metrics = calculate_metrics(df_results, quarterly_data)
    
    print("\nKEY METRICS")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    print("\nGenerating charts...")
    
    plot_equity_curve(df_results, f'{fig_dir}/equity_curve.png')
    plot_strategy_vs_benchmark(df_results, f'{fig_dir}/strategy_vs_benchmark.png')
    plot_drawdown(df_results, f'{fig_dir}/drawdown.png')
    plot_trade_pnl_distribution(trade_pnls, f'{fig_dir}/trade_pnl_distribution.png')
    plot_quarterly_returns(quarterly_data, f'{fig_dir}/quarterly_returns.png')
    plot_monthly_trades(monthly_trades, f'{fig_dir}/monthly_trades.png')
    plot_rolling_sharpe(df_results, 180, f'{fig_dir}/rolling_sharpe.png')
    create_metrics_table(metrics, f'{fig_dir}/metrics_table.png')
    plot_monthly_returns_heatmap(df_results, f'{fig_dir}/monthly_returns_heatmap.png')
    
    print(f"\nAll figures saved to: {fig_dir}/")
    for f in sorted(os.listdir(fig_dir)):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
