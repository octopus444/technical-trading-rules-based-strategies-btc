"""
Data loading for Bitstamp BTC data.
Resamples tick data to 5-minute bars.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_bitstamp_data(filepath, start_date="2017-01-01", end_date=None, resample_freq="5T"):
    """Load and preprocess Bitstamp CSV. Returns DataFrame with Price, Volume, Returns."""
    
    df = pd.read_csv(filepath, names=["Date", "Price", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"], unit="s")
    df["Date"] = df["Date"].dt.tz_localize(None)
    df.set_index("Date", inplace=True)
    
    df = df.resample(resample_freq).last()
    df = df.interpolate(method="linear")
    
    df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]
    
    df["Returns"] = np.log(df["Price"] / df["Price"].shift(1))
    df = df.dropna()
    
    return df


def get_data_summary(df):
    """Quick summary stats for loaded data."""
    return {
        "start_date": df.index.min(),
        "end_date": df.index.max(),
        "n_observations": len(df),
        "price_min": df["Price"].min(),
        "price_max": df["Price"].max(),
        "price_mean": df["Price"].mean(),
        "total_return": (df["Price"].iloc[-1] / df["Price"].iloc[0] - 1) * 100,
        "annualized_volatility": df["Returns"].std() * np.sqrt(252 * 288) * 100
    }


if __name__ == "__main__":
    import sys
    
    filepath = sys.argv[1] if len(sys.argv) > 1 else ".bitstampUSD.csv"
    
    print(f"Loading data from {filepath}...")
    df = load_bitstamp_data(filepath)
    
    summary = get_data_summary(df)
    print("\nData Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
