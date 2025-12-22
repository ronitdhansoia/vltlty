"""
PyTorch Dataset Classes for Rough Volatility Transfer Learning

This module provides dataset classes for loading and preprocessing
volatility time series data for LSTM-based forecasting models.

Classes:
    VolatilityDataset: PyTorch Dataset for volatility sequences

Usage:
    from datasets import VolatilityDataset
    dataset = VolatilityDataset(log_rv_series, lookback=20, horizon=1)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

Author: Ronit Dhansoia
Date: 22nd December 2025
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Union
import os


class VolatilityDataset(Dataset):
    """
    PyTorch Dataset for volatility time series forecasting.

    Creates sequences of log realized volatility for training LSTM models.
    Each sample consists of:
        - X: lookback window of past log RV values [t-lookback : t]
        - y: future log RV values [t+1 : t+1+horizon]

    Attributes:
        data: Normalized log RV time series
        lookback: Number of past observations to use as input
        horizon: Number of future steps to predict
        mean: Mean used for normalization (stored for inverse transform)
        std: Std used for normalization (stored for inverse transform)
    """

    def __init__(self,
                 log_rv: Union[np.ndarray, pd.Series],
                 lookback: int = 20,
                 horizon: int = 1,
                 normalize: bool = True,
                 mean: Optional[float] = None,
                 std: Optional[float] = None):
        """
        Initialize the VolatilityDataset.

        Args:
            log_rv: Log realized volatility time series (1D array or Series)
            lookback: Number of past observations for input sequence (default: 20)
            horizon: Number of future steps to predict (default: 1)
            normalize: Whether to z-score normalize the data (default: True)
            mean: Pre-computed mean for normalization (use training set mean for val/test)
            std: Pre-computed std for normalization (use training set std for val/test)
        """
        # Convert to numpy array if needed
        if isinstance(log_rv, pd.Series):
            log_rv = log_rv.values

        self.raw_data = log_rv.astype(np.float32)
        self.lookback = lookback
        self.horizon = horizon
        self.normalize = normalize

        # Compute or use provided normalization parameters
        if normalize:
            if mean is None:
                self.mean = np.mean(self.raw_data)
            else:
                self.mean = mean

            if std is None:
                self.std = np.std(self.raw_data)
            else:
                self.std = std

            # Normalize data
            self.data = (self.raw_data - self.mean) / (self.std + 1e-8)
        else:
            self.mean = 0.0
            self.std = 1.0
            self.data = self.raw_data

        # Compute valid indices (where we have enough data for both input and target)
        self.valid_length = len(self.data) - lookback - horizon + 1

        if self.valid_length <= 0:
            raise ValueError(
                f"Data length ({len(self.data)}) is too short for "
                f"lookback ({lookback}) + horizon ({horizon})"
            )

    def __len__(self) -> int:
        """Return the number of valid samples."""
        return self.valid_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            X: Input sequence of shape (lookback, 1)
            y: Target value(s) of shape (horizon,)
        """
        # Input sequence: [idx : idx + lookback]
        X = self.data[idx : idx + self.lookback]

        # Target: [idx + lookback : idx + lookback + horizon]
        y = self.data[idx + self.lookback : idx + self.lookback + self.horizon]

        # Convert to tensors
        # X shape: (lookback, 1) for LSTM input
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)

        # y shape: (horizon,) or scalar if horizon=1
        y = torch.tensor(y, dtype=torch.float32)
        if self.horizon == 1:
            y = y.squeeze()

        return X, y

    def inverse_transform(self, normalized_data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Convert normalized data back to original scale.

        Args:
            normalized_data: Z-score normalized values

        Returns:
            Data in original scale
        """
        if isinstance(normalized_data, torch.Tensor):
            normalized_data = normalized_data.numpy()

        return normalized_data * self.std + self.mean

    def get_normalization_params(self) -> Tuple[float, float]:
        """Return normalization parameters (mean, std)."""
        return self.mean, self.std


def create_dataloaders(
    train_log_rv: np.ndarray,
    val_log_rv: np.ndarray,
    test_log_rv: np.ndarray,
    lookback: int = 20,
    horizon: int = 1,
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Create DataLoaders for train, validation, and test sets.

    Normalization is computed on training data only and applied to all sets
    to prevent data leakage.

    Args:
        train_log_rv: Training log RV series
        val_log_rv: Validation log RV series
        test_log_rv: Test log RV series
        lookback: Lookback window size
        horizon: Forecast horizon
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes for DataLoader

    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
        norm_params: Dictionary with mean and std
    """
    # Create training dataset (computes normalization params)
    train_dataset = VolatilityDataset(
        train_log_rv, lookback=lookback, horizon=horizon, normalize=True
    )

    # Get normalization parameters from training set
    mean, std = train_dataset.get_normalization_params()

    # Create val/test datasets using training normalization
    val_dataset = VolatilityDataset(
        val_log_rv, lookback=lookback, horizon=horizon,
        normalize=True, mean=mean, std=std
    )

    test_dataset = VolatilityDataset(
        test_log_rv, lookback=lookback, horizon=horizon,
        normalize=True, mean=mean, std=std
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    norm_params = {'mean': mean, 'std': std}

    return train_loader, val_loader, test_loader, norm_params


def load_data_splits(data_dir: str = 'data', asset: str = 'spx') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load pre-split data from CSV files.

    Args:
        data_dir: Directory containing data files
        asset: Asset identifier ('spx' or 'btc')

    Returns:
        train_log_rv: Training log RV array
        val_log_rv: Validation log RV array
        test_log_rv: Test log RV array
    """
    train_df = pd.read_csv(os.path.join(data_dir, f'{asset}_train.csv'), index_col=0)
    val_df = pd.read_csv(os.path.join(data_dir, f'{asset}_val.csv'), index_col=0)
    test_df = pd.read_csv(os.path.join(data_dir, f'{asset}_test.csv'), index_col=0)

    return (
        train_df['log_rv'].values,
        val_df['log_rv'].values,
        test_df['log_rv'].values
    )


# ============================================================================
# TESTING / EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING VOLATILITY DATASET")
    print("=" * 60)

    # Load S&P 500 data
    print("\n📂 Loading S&P 500 data...")
    train_rv, val_rv, test_rv = load_data_splits('data', 'spx')

    print(f"  Train: {len(train_rv)} samples")
    print(f"  Val:   {len(val_rv)} samples")
    print(f"  Test:  {len(test_rv)} samples")

    # Create dataloaders
    print("\n📦 Creating DataLoaders...")
    train_loader, val_loader, test_loader, norm_params = create_dataloaders(
        train_rv, val_rv, test_rv,
        lookback=20, horizon=1, batch_size=32
    )

    print(f"  Normalization: mean={norm_params['mean']:.4f}, std={norm_params['std']:.4f}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")

    # Test a batch
    print("\n🔍 Sample batch:")
    X, y = next(iter(train_loader))
    print(f"  X shape: {X.shape}  (batch, lookback, features)")
    print(f"  y shape: {y.shape}  (batch,)")
    print(f"  X sample [0,:5,0]: {X[0,:5,0].numpy()}")
    print(f"  y sample [0]: {y[0].item():.4f}")

    # Test Bitcoin data
    print("\n" + "-" * 60)
    print("📂 Loading Bitcoin data...")
    btc_train, btc_val, btc_test = load_data_splits('data', 'btc')

    print(f"  Train: {len(btc_train)} samples")
    print(f"  Val:   {len(btc_val)} samples")
    print(f"  Test:  {len(btc_test)} samples")

    btc_train_loader, btc_val_loader, btc_test_loader, btc_norm = create_dataloaders(
        btc_train, btc_val, btc_test,
        lookback=20, horizon=1, batch_size=16
    )

    print(f"  Normalization: mean={btc_norm['mean']:.4f}, std={btc_norm['std']:.4f}")

    print("\n✅ Dataset tests passed!")
