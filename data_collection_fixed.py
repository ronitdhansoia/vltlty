"""
Data Collection for Rough Volatility Transfer Learning Research (FIXED)

This script downloads DAILY data for both S&P 500 and Bitcoin to enable
proper rough volatility analysis and transfer learning research.

Key Changes:
- S&P 500: Daily data 2018-2024 (matches paper claims)
- Bitcoin: Daily data 2014-2024 (matches paper claims)
- Both use daily frequency for proper Hurst estimation

Author: Ronit Dhansoia
Date: December 2025
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = "data"

# Date ranges matching paper claims
SPX_START = "2018-01-01"
SPX_END = "2024-12-01"
BTC_START = "2014-09-17"  # Bitcoin started trading on Yahoo Finance
BTC_END = "2024-12-01"

# Volatility computation parameters (daily)
RV_WINDOW = 22  # ~1 month of trading days

# Split ratios
TRAIN_RATIO = 0.50
VAL_RATIO = 0.25
TEST_RATIO = 0.25


def ensure_data_dir():
    """Create data directory if it doesn't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)


def download_spx_data() -> pd.DataFrame:
    """Download S&P 500 daily data from Yahoo Finance."""
    print("\n" + "=" * 70)
    print("DOWNLOADING S&P 500 DATA")
    print("=" * 70)

    print(f"Ticker: ^GSPC (S&P 500 Index)")
    print(f"Period: {SPX_START} to {SPX_END}")
    print(f"Interval: Daily")

    df = yf.download('^GSPC', start=SPX_START, end=SPX_END, interval='1d', progress=False)

    if df.empty:
        raise ValueError("Failed to download S&P 500 data")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print(f"✓ Downloaded {len(df):,} daily observations")
    print(f"  Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")

    return df


def download_btc_data() -> pd.DataFrame:
    """Download Bitcoin DAILY data from Yahoo Finance."""
    print("\n" + "=" * 70)
    print("DOWNLOADING BITCOIN DATA (DAILY)")
    print("=" * 70)

    print(f"Ticker: BTC-USD")
    print(f"Period: {BTC_START} to {BTC_END}")
    print(f"Interval: Daily")

    df = yf.download('BTC-USD', start=BTC_START, end=BTC_END, interval='1d', progress=False)

    if df.empty:
        raise ValueError("Failed to download Bitcoin data")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print(f"✓ Downloaded {len(df):,} daily observations")
    print(f"  Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")

    return df


def compute_features(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Compute log returns and realized volatility.

    Uses:
    - Log returns: ln(Close_t / Close_{t-1})
    - Realized volatility: rolling std of log returns × √252 (annualized)
    - Log realized volatility: ln(RV)
    """
    print(f"\nComputing features for {name}...")

    data = df.copy()

    # Log returns
    data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))

    # Realized volatility (annualized)
    data['realized_vol'] = data['log_return'].rolling(window=RV_WINDOW).std() * np.sqrt(252)

    # Log realized volatility
    data['log_rv'] = np.log(data['realized_vol'])

    # Drop NaN values
    data = data.dropna()
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    print(f"✓ Cleaned data: {len(data):,} observations remaining")

    return data


def split_data(df: pd.DataFrame, name: str) -> Dict[str, pd.DataFrame]:
    """Split data into train/val/test sets with temporal ordering."""
    print(f"\nSplitting {name} data...")

    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    splits = {
        'train': df.iloc[:train_end].copy(),
        'val': df.iloc[train_end:val_end].copy(),
        'test': df.iloc[val_end:].copy()
    }

    print(f"✓ Train: {len(splits['train']):,} ({TRAIN_RATIO*100:.0f}%)")
    print(f"✓ Val:   {len(splits['val']):,} ({VAL_RATIO*100:.0f}%)")
    print(f"✓ Test:  {len(splits['test']):,} ({TEST_RATIO*100:.0f}%)")

    return splits


def save_dataset(df: pd.DataFrame, filename: str):
    """Save DataFrame to CSV."""
    filepath = os.path.join(DATA_DIR, filename)
    df.to_csv(filepath)
    size_kb = os.path.getsize(filepath) / 1024
    print(f"  Saved: {filepath} ({size_kb:.1f} KB)")


def print_stats(df: pd.DataFrame, name: str):
    """Print summary statistics."""
    print(f"\n{name} Log RV Statistics:")
    print(f"  Count:    {len(df):,}")
    print(f"  Mean:     {df['log_rv'].mean():.4f}")
    print(f"  Std:      {df['log_rv'].std():.4f}")
    print(f"  Skewness: {df['log_rv'].skew():.4f}")
    print(f"  Kurtosis: {df['log_rv'].kurtosis():.4f}")
    print(f"  Min:      {df['log_rv'].min():.4f}")
    print(f"  Max:      {df['log_rv'].max():.4f}")


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("ROUGH VOLATILITY DATA COLLECTION (FIXED - DAILY DATA)")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    ensure_data_dir()

    # Download and process S&P 500
    spx_raw = download_spx_data()
    spx_data = compute_features(spx_raw, "S&P 500")
    print_stats(spx_data, "S&P 500")
    spx_splits = split_data(spx_data, "S&P 500")

    # Download and process Bitcoin
    btc_raw = download_btc_data()
    btc_data = compute_features(btc_raw, "Bitcoin")
    print_stats(btc_data, "Bitcoin")
    btc_splits = split_data(btc_data, "Bitcoin")

    # Save all data
    print("\n" + "=" * 70)
    print("SAVING DATA")
    print("=" * 70)

    print("\nS&P 500 data:")
    save_dataset(spx_data, 'spx_data.csv')
    save_dataset(spx_splits['train'], 'spx_train.csv')
    save_dataset(spx_splits['val'], 'spx_val.csv')
    save_dataset(spx_splits['test'], 'spx_test.csv')

    print("\nBitcoin data:")
    save_dataset(btc_data, 'btc_data.csv')
    save_dataset(btc_splits['train'], 'btc_train.csv')
    save_dataset(btc_splits['val'], 'btc_val.csv')
    save_dataset(btc_splits['test'], 'btc_test.csv')

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'S&P 500':<15} {'Bitcoin':<15}")
    print("-" * 55)
    print(f"{'Observations':<25} {len(spx_data):<15,} {len(btc_data):<15,}")
    print(f"{'Date range':<25} {'2018-2024':<15} {'2014-2024':<15}")
    print(f"{'Train samples':<25} {len(spx_splits['train']):<15,} {len(btc_splits['train']):<15,}")
    print(f"{'Test samples':<25} {len(spx_splits['test']):<15,} {len(btc_splits['test']):<15,}")
    print(f"{'Mean log RV':<25} {spx_data['log_rv'].mean():<15.2f} {btc_data['log_rv'].mean():<15.2f}")
    print(f"{'Std log RV':<25} {spx_data['log_rv'].std():<15.2f} {btc_data['log_rv'].std():<15.2f}")

    print("\n✅ Data collection complete!")


if __name__ == "__main__":
    main()
