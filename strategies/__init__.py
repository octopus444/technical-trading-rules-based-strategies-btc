"""Trading strategies module."""

from .filter_strategy import backtest_all_filter_strategies, count_strategies as count_filter
from .ma_strategy import backtest_all_ma_strategies, count_strategies as count_ma
from .sr_strategy import backtest_all_sr_strategies, count_strategies as count_sr
from .cb_strategy import backtest_all_cb_strategies, count_strategies as count_cb
from .rsi_strategy import backtest_all_rsi_strategies, count_strategies as count_rsi
from .bb_strategy import backtest_all_bb_strategies, count_strategies as count_bb
from .obv_strategy import backtest_all_obv_strategies, count_strategies as count_obv

__all__ = [
    "backtest_all_filter_strategies",
    "backtest_all_ma_strategies",
    "backtest_all_sr_strategies",
    "backtest_all_cb_strategies",
    "backtest_all_rsi_strategies",
    "backtest_all_bb_strategies",
    "backtest_all_obv_strategies",
]


def count_all_strategies(include_contrarian: bool = True) -> dict:
    """Count strategies by type."""
    return {
        "Filter": count_filter(include_contrarian),
        "MA": count_ma(include_contrarian),
        "SR": count_sr(include_contrarian),
        "CB": count_cb(include_contrarian),
        "RSI": count_rsi(include_contrarian),
        "BB": count_bb(include_contrarian),
        "OBV": count_obv(include_contrarian),
    }
