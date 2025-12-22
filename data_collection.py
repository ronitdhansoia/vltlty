"""
Data Collection for Rough Volatility Transfer Learning Research

This script downloads and preprocesses S&P 500 and Bitcoin data for studying
cross-asset transfer learning in rough volatility forecasting.

Key Features:
- Downloads S&P 500 daily data (2019-2024) from Yahoo Finance
- Downloads Bitcoin hourly data (2020-2024) from Yahoo Finance
- Computes log returns and realized volatility for both assets
- Creates train/val/test splits with proper temporal ordering
- Includes comprehensive data quality checks

Usage:
    python data_collection.py

Author: Research Project
Date: December 2024
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data directory
DATA_DIR = "data"

# Date ranges
SPX_START = "2019-01-01"
SPX_END = "2024-12-01"
BTC_START = "2020-01-01"
BTC_END = "2024-12-01"

# Volatility computation parameters
SPX_RV_WINDOW = 22  # ~1 month of trading days for S&P 500
BTC_RV_WINDOW = 24  # 24 hours for Bitcoin (hourly data)

# Split ratios
SPX_TRAIN_RATIO = 0.70
SPX_VAL_RATIO = 0.15
SPX_TEST_RATIO = 0.15

# For Bitcoin: simulate data scarcity - use only last 30% of history
BTC_USE_RATIO = 0.30  # Only use last 30% of available data
BTC_TRAIN_RATIO = 0.50  # Then split: 50% train
BTC_VAL_RATIO = 0.25   # 25% val
BTC_TEST_RATIO = 0.25  # 25% test


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_data_dir():
    """Create data directory if it doesn't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"✓ Data directory: {os.path.abspath(DATA_DIR)}")


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_stats(df: pd.DataFrame, name: str, column: str = 'log_rv'):
    """Print summary statistics for a dataset."""
    print(f"\n{name} Summary Statistics ({column}):")
    print(f"  Count:    {len(df):,}")
    print(f"  Mean:     {df[column].mean():.4f}")
    print(f"  Std:      {df[column].std():.4f}")
    print(f"  Min:      {df[column].min():.4f}")
    print(f"  Max:      {df[column].max():.4f}")
    print(f"  Median:   {df[column].median():.4f}")
    print(f"  Skewness: {df[column].skew():.4f}")


def check_data_quality(df: pd.DataFrame, name: str) -> bool:
    """
    Check data quality and report issues.

    Args:
        df: DataFrame to check
        name: Name of the dataset for reporting

    Returns:
        True if data passes all quality checks
    """
    print(f"\n{name} Data Quality Checks:")

    issues = []

    # Check for NaN values
    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        issues.append(f"  ⚠ Found {nan_count} NaN values")
    else:
        print("  ✓ No NaN values")

    # Check for infinite values
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    if inf_count > 0:
        issues.append(f"  ⚠ Found {inf_count} infinite values")
    else:
        print("  ✓ No infinite values")

    # Check for duplicate indices
    dup_count = df.index.duplicated().sum()
    if dup_count > 0:
        issues.append(f"  ⚠ Found {dup_count} duplicate timestamps")
    else:
        print("  ✓ No duplicate timestamps")

    # Check for extreme outliers in log returns (> 5 std)
    if 'log_return' in df.columns:
        mean_lr = df['log_return'].mean()
        std_lr = df['log_return'].std()
        outliers = ((df['log_return'] - mean_lr).abs() > 5 * std_lr).sum()
        if outliers > 0:
            print(f"  ⚠ Found {outliers} extreme log return outliers (>5σ)")
        else:
            print("  ✓ No extreme log return outliers")

    # Check temporal ordering
    if not df.index.is_monotonic_increasing:
        issues.append("  ⚠ Index is not monotonically increasing")
    else:
        print("  ✓ Temporal ordering is correct")

    if issues:
        for issue in issues:
            print(issue)
        return False

    return True


# ============================================================================
# S&P 500 DATA COLLECTION
# ============================================================================

def download_spx_data() -> pd.DataFrame:
    """
    Download S&P 500 daily data from Yahoo Finance.

    Returns:
        DataFrame with OHLCV data and computed features
    """
    print_header("DOWNLOADING S&P 500 DATA")

    print(f"Ticker: ^GSPC (S&P 500 Index)")
    print(f"Period: {SPX_START} to {SPX_END}")
    print(f"Interval: Daily")

    # Download data
    print("\nFetching data from Yahoo Finance...")
    df = yf.download('^GSPC', start=SPX_START, end=SPX_END, progress=False)

    if df.empty:
        raise ValueError("Failed to download S&P 500 data")

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print(f"✓ Downloaded {len(df):,} daily observations")
    print(f"  Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")

    return df


def compute_spx_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns and realized volatility for S&P 500.

    Uses:
    - Log returns: ln(Close_t / Close_{t-1})
    - Realized volatility: rolling std of log returns × √252 (annualized)
    - Log realized volatility: ln(RV)

    Args:
        df: Raw OHLCV DataFrame

    Returns:
        DataFrame with computed features
    """
    print("\nComputing features...")

    # Make a copy
    data = df.copy()

    # Log returns
    data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))

    # Realized volatility (annualized)
    # Rolling std of log returns × sqrt(252 trading days)
    data['realized_vol'] = data['log_return'].rolling(window=SPX_RV_WINDOW).std() * np.sqrt(252)

    # Log realized volatility
    data['log_rv'] = np.log(data['realized_vol'])

    # Drop NaN values from rolling window
    data = data.dropna()

    # Remove any remaining infinite values
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    print(f"✓ Computed log returns")
    print(f"✓ Computed realized volatility (window={SPX_RV_WINDOW} days, annualized)")
    print(f"✓ Computed log realized volatility")
    print(f"✓ Cleaned data: {len(data):,} observations remaining")

    return data


def split_spx_data(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Split S&P 500 data into train/val/test sets.

    Uses temporal ordering (no shuffling) to prevent data leakage.

    Args:
        df: Processed DataFrame

    Returns:
        Dictionary with 'train', 'val', 'test' DataFrames
    """
    print("\nSplitting data...")

    n = len(df)
    train_end = int(n * SPX_TRAIN_RATIO)
    val_end = int(n * (SPX_TRAIN_RATIO + SPX_VAL_RATIO))

    splits = {
        'train': df.iloc[:train_end].copy(),
        'val': df.iloc[train_end:val_end].copy(),
        'test': df.iloc[val_end:].copy()
    }

    print(f"✓ Train: {len(splits['train']):,} observations ({SPX_TRAIN_RATIO*100:.0f}%)")
    print(f"    {splits['train'].index[0].strftime('%Y-%m-%d')} to {splits['train'].index[-1].strftime('%Y-%m-%d')}")
    print(f"✓ Val:   {len(splits['val']):,} observations ({SPX_VAL_RATIO*100:.0f}%)")
    print(f"    {splits['val'].index[0].strftime('%Y-%m-%d')} to {splits['val'].index[-1].strftime('%Y-%m-%d')}")
    print(f"✓ Test:  {len(splits['test']):,} observations ({SPX_TEST_RATIO*100:.0f}%)")
    print(f"    {splits['test'].index[0].strftime('%Y-%m-%d')} to {splits['test'].index[-1].strftime('%Y-%m-%d')}")

    return splits


# ============================================================================
# BITCOIN DATA COLLECTION
# ============================================================================

def download_btc_data() -> pd.DataFrame:
    """
    Download Bitcoin hourly data from Yahoo Finance.

    Note: Yahoo Finance has limited hourly history (~730 days max).
    For longer history, would need to use ccxt/Binance API.

    Returns:
        DataFrame with OHLCV data
    """
    print_header("DOWNLOADING BITCOIN DATA")

    print(f"Ticker: BTC-USD")
    print(f"Period: {BTC_START} to {BTC_END}")
    print(f"Interval: Hourly")

    # Download data - try maximum period for hourly data
    print("\nFetching data from Yahoo Finance...")
    print("Note: Yahoo Finance limits hourly data to ~730 days")

    # Download in chunks to get more data
    all_data = []

    # Try to get as much historical data as possible
    df = yf.download('BTC-USD', period='730d', interval='1h', progress=False)

    if df.empty:
        raise ValueError("Failed to download Bitcoin data")

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print(f"✓ Downloaded {len(df):,} hourly observations")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")

    return df


def compute_btc_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute log returns and realized volatility for Bitcoin.

    Uses:
    - Log returns: ln(Close_t / Close_{t-1})
    - Parkinson volatility estimator: sqrt(1/(4*ln(2))) * ln(High/Low)
      (more efficient than close-to-close with OHLC data)
    - Log realized volatility: ln(RV)

    Args:
        df: Raw OHLCV DataFrame

    Returns:
        DataFrame with computed features
    """
    print("\nComputing features...")

    # Make a copy
    data = df.copy()

    # Log returns (close-to-close)
    data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))

    # Parkinson volatility estimator (using High-Low)
    # This is more efficient than close-to-close for intraday data
    parkinson_const = 1.0 / (4.0 * np.log(2))
    data['parkinson_vol'] = np.sqrt(parkinson_const) * np.log(data['High'] / data['Low'])

    # Rolling realized volatility (24-hour window for hourly data)
    # Annualized: sqrt(24 * 365) for hourly data
    annualization_factor = np.sqrt(24 * 365)
    data['realized_vol'] = data['log_return'].rolling(window=BTC_RV_WINDOW).std() * annualization_factor

    # Log realized volatility
    data['log_rv'] = np.log(data['realized_vol'])

    # Drop NaN values
    data = data.dropna()

    # Remove any remaining infinite values
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    print(f"✓ Computed log returns")
    print(f"✓ Computed Parkinson volatility (high-low estimator)")
    print(f"✓ Computed realized volatility (window={BTC_RV_WINDOW} hours, annualized)")
    print(f"✓ Computed log realized volatility")
    print(f"✓ Cleaned data: {len(data):,} observations remaining")

    return data


def split_btc_data(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Split Bitcoin data into train/val/test sets.

    Simulates data scarcity by only using the last 30% of history,
    then splitting that into 50/25/25 train/val/test.

    Args:
        df: Processed DataFrame

    Returns:
        Dictionary with 'train', 'val', 'test' DataFrames
    """
    print("\nSplitting data (simulating data scarcity)...")

    # Only use last 30% of data to simulate limited crypto history
    n_total = len(df)
    use_start = int(n_total * (1 - BTC_USE_RATIO))
    df_use = df.iloc[use_start:].copy()

    print(f"✓ Using last {BTC_USE_RATIO*100:.0f}% of data: {len(df_use):,} observations")
    print(f"  (Simulating limited historical data for crypto)")

    # Split the usable portion
    n = len(df_use)
    train_end = int(n * BTC_TRAIN_RATIO)
    val_end = int(n * (BTC_TRAIN_RATIO + BTC_VAL_RATIO))

    splits = {
        'train': df_use.iloc[:train_end].copy(),
        'val': df_use.iloc[train_end:val_end].copy(),
        'test': df_use.iloc[val_end:].copy()
    }

    print(f"✓ Train: {len(splits['train']):,} observations ({BTC_TRAIN_RATIO*100:.0f}%)")
    print(f"    {splits['train'].index[0]} to {splits['train'].index[-1]}")
    print(f"✓ Val:   {len(splits['val']):,} observations ({BTC_VAL_RATIO*100:.0f}%)")
    print(f"    {splits['val'].index[0]} to {splits['val'].index[-1]}")
    print(f"✓ Test:  {len(splits['test']):,} observations ({BTC_TEST_RATIO*100:.0f}%)")
    print(f"    {splits['test'].index[0]} to {splits['test'].index[-1]}")

    return splits


# ============================================================================
# SAVE DATA
# ============================================================================

def save_dataset(df: pd.DataFrame, filename: str):
    """Save DataFrame to CSV with proper formatting."""
    filepath = os.path.join(DATA_DIR, filename)
    df.to_csv(filepath)
    size_kb = os.path.getsize(filepath) / 1024
    print(f"  Saved: {filepath} ({size_kb:.1f} KB)")


def save_all_data(spx_full: pd.DataFrame, spx_splits: Dict[str, pd.DataFrame],
                  btc_full: pd.DataFrame, btc_splits: Dict[str, pd.DataFrame]):
    """Save all datasets to CSV files."""
    print_header("SAVING DATA TO CSV")

    # S&P 500
    print("\nS&P 500 data:")
    save_dataset(spx_full, 'spx_data.csv')
    save_dataset(spx_splits['train'], 'spx_train.csv')
    save_dataset(spx_splits['val'], 'spx_val.csv')
    save_dataset(spx_splits['test'], 'spx_test.csv')

    # Bitcoin
    print("\nBitcoin data:")
    save_dataset(btc_full, 'btc_data.csv')
    save_dataset(btc_splits['train'], 'btc_train.csv')
    save_dataset(btc_splits['val'], 'btc_val.csv')
    save_dataset(btc_splits['test'], 'btc_test.csv')


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("ROUGH VOLATILITY TRANSFER LEARNING - DATA COLLECTION")
    print("=" * 70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create data directory
    ensure_data_dir()

    # -------------------------------------------------------------------------
    # S&P 500 Data
    # -------------------------------------------------------------------------

    # Download
    spx_raw = download_spx_data()

    # Compute features
    spx_data = compute_spx_features(spx_raw)

    # Quality checks
    check_data_quality(spx_data, "S&P 500")

    # Statistics
    print_stats(spx_data, "S&P 500")

    # Split
    spx_splits = split_spx_data(spx_data)

    # -------------------------------------------------------------------------
    # Bitcoin Data
    # -------------------------------------------------------------------------

    # Download
    btc_raw = download_btc_data()

    # Compute features
    btc_data = compute_btc_features(btc_raw)

    # Quality checks
    check_data_quality(btc_data, "Bitcoin")

    # Statistics
    print_stats(btc_data, "Bitcoin")

    # Split
    btc_splits = split_btc_data(btc_data)

    # -------------------------------------------------------------------------
    # Save All Data
    # -------------------------------------------------------------------------

    save_all_data(spx_data, spx_splits, btc_data, btc_splits)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print_header("SUMMARY")

    print("\n📊 Dataset Comparison:")
    print("-" * 50)
    print(f"{'Metric':<25} {'S&P 500':<15} {'Bitcoin':<15}")
    print("-" * 50)
    print(f"{'Total observations':<25} {len(spx_data):<15,} {len(btc_data):<15,}")
    print(f"{'Frequency':<25} {'Daily':<15} {'Hourly':<15}")
    print(f"{'Train samples':<25} {len(spx_splits['train']):<15,} {len(btc_splits['train']):<15,}")
    print(f"{'Val samples':<25} {len(spx_splits['val']):<15,} {len(btc_splits['val']):<15,}")
    print(f"{'Test samples':<25} {len(spx_splits['test']):<15,} {len(btc_splits['test']):<15,}")
    print(f"{'Mean log RV':<25} {spx_data['log_rv'].mean():<15.4f} {btc_data['log_rv'].mean():<15.4f}")
    print(f"{'Std log RV':<25} {spx_data['log_rv'].std():<15.4f} {btc_data['log_rv'].std():<15.4f}")
    print("-" * 50)

    print("\n✅ Data collection complete!")
    print(f"\nFiles saved to: {os.path.abspath(DATA_DIR)}/")
    print("  - spx_data.csv, spx_train.csv, spx_val.csv, spx_test.csv")
    print("  - btc_data.csv, btc_train.csv, btc_val.csv, btc_test.csv")

    print("\n📋 Next Steps:")
    print("  1. Run hurst_estimation.py to estimate Hurst parameters")
    print("  2. Create exploratory analysis notebook")
    print("  3. Build model architecture")

    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
