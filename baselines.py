"""
Baseline Models for Rough Volatility Forecasting

This module implements baseline models for comparison:
    1. LSTM from scratch (no transfer learning)
    2. HAR (Heterogeneous Autoregressive) model
    3. Simple moving average baseline

These baselines help quantify the benefit of transfer learning.

Author: Research Project
Date: December 2024
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from typing import Tuple, Dict, Optional
import os

from models import TransferRoughVolModel
from datasets import VolatilityDataset, create_dataloaders, load_data_splits


# ============================================================================
# LSTM FROM SCRATCH (NO TRANSFER LEARNING)
# ============================================================================

def train_lstm_from_scratch(
    train_loader,
    val_loader,
    epochs: int = 45,
    learning_rate: float = 0.001,
    device: torch.device = None,
    verbose: bool = True
) -> Tuple[TransferRoughVolModel, Dict]:
    """
    Train LSTM model from scratch on Bitcoin data only.

    Same architecture as transfer model but without pre-training.

    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to use
        verbose: Print progress

    Returns:
        model: Trained model
        history: Training history
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if verbose:
        print("\n" + "=" * 60)
        print("BASELINE: LSTM FROM SCRATCH (No Transfer Learning)")
        print("=" * 60)

    # Create new model (same architecture)
    model = TransferRoughVolModel(
        input_dim=1,
        hidden_dim=64,
        encoding_dim=32,
        num_layers=3,
        num_heads=4,
        forecast_horizon=1,
        dropout=0.1
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    history = {'train_loss': [], 'val_loss': [], 'val_rmse': []}
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    patience = 10

    if verbose:
        print(f"\nTraining for {epochs} epochs...")
        print(f"Trainable parameters: {model.get_trainable_params():,}")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                val_loss += criterion(pred, y).item()
                val_preds.append(pred.cpu())
                val_targets.append(y.cpu())

        val_loss /= len(val_loader)
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_rmse = torch.sqrt(torch.mean((val_preds - val_targets) ** 2)).item()

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}")
                break

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_rmse={val_rmse:.4f}")

    model.load_state_dict(best_state)

    if verbose:
        print(f"\n✅ Training complete!")
        print(f"   Best validation RMSE: {min(history['val_rmse']):.4f}")

    return model, history


# ============================================================================
# HAR MODEL (Heterogeneous Autoregressive)
# ============================================================================

class HARModel:
    """
    Heterogeneous Autoregressive (HAR) Model for Volatility.

    Uses three features:
        - RV_{t-1}: Previous day volatility
        - RV_weekly: Average of last 5 days
        - RV_monthly: Average of last 22 days

    Reference: Corsi (2009)
    """

    def __init__(self):
        self.model = LinearRegression()
        self.fitted = False

    def create_features(self, log_rv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create HAR features from log RV series.

        Args:
            log_rv: Log realized volatility array

        Returns:
            X: Feature matrix (n_samples, 3)
            y: Target values
        """
        n = len(log_rv)
        start_idx = 22  # Need 22 days of history for monthly average

        X = []
        y = []

        for i in range(start_idx, n):
            rv_daily = log_rv[i - 1]  # Previous day
            rv_weekly = np.mean(log_rv[i - 5:i])  # 5-day average
            rv_monthly = np.mean(log_rv[i - 22:i])  # 22-day average

            X.append([rv_daily, rv_weekly, rv_monthly])
            y.append(log_rv[i])

        return np.array(X), np.array(y)

    def fit(self, log_rv: np.ndarray):
        """Fit HAR model on training data."""
        X, y = self.create_features(log_rv)
        self.model.fit(X, y)
        self.fitted = True
        return self

    def predict(self, log_rv: np.ndarray) -> np.ndarray:
        """Generate predictions for test data."""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        X, _ = self.create_features(log_rv)
        return self.model.predict(X)

    def get_targets(self, log_rv: np.ndarray) -> np.ndarray:
        """Get target values (for computing metrics)."""
        _, y = self.create_features(log_rv)
        return y


# ============================================================================
# SIMPLE MOVING AVERAGE BASELINE
# ============================================================================

class MovingAverageBaseline:
    """
    Simple Moving Average baseline.

    Predicts next value as average of last n observations.
    """

    def __init__(self, window: int = 5):
        self.window = window

    def predict(self, log_rv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions.

        Returns:
            predictions: Predicted values
            targets: Actual values (for computing metrics)
        """
        predictions = []
        targets = []

        for i in range(self.window, len(log_rv)):
            pred = np.mean(log_rv[i - self.window:i])
            predictions.append(pred)
            targets.append(log_rv[i])

        return np.array(predictions), np.array(targets)


# ============================================================================
# RUN ALL BASELINES
# ============================================================================

def run_all_baselines(
    train_log_rv: np.ndarray,
    val_log_rv: np.ndarray,
    test_log_rv: np.ndarray,
    lookback: int = 20,
    verbose: bool = True
) -> Dict:
    """
    Run all baseline models and return predictions.

    Args:
        train_log_rv: Training data
        val_log_rv: Validation data
        test_log_rv: Test data
        lookback: Lookback window for LSTM
        verbose: Print progress

    Returns:
        Dictionary with predictions and metrics for each baseline
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}

    # -------------------------------------------------------------------------
    # 1. LSTM from Scratch
    # -------------------------------------------------------------------------
    train_loader, val_loader, test_loader, norm_params = create_dataloaders(
        train_log_rv, val_log_rv, test_log_rv,
        lookback=lookback, horizon=1, batch_size=16
    )

    lstm_model, lstm_history = train_lstm_from_scratch(
        train_loader, val_loader,
        epochs=45, learning_rate=0.001,
        device=device, verbose=verbose
    )

    # Get test predictions
    lstm_model.eval()
    lstm_preds, lstm_targets = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            pred = lstm_model(X)
            lstm_preds.append(pred.cpu())
            lstm_targets.append(y)

    lstm_preds = torch.cat(lstm_preds).numpy()
    lstm_targets = torch.cat(lstm_targets).numpy()

    # Inverse transform
    lstm_preds = lstm_preds * norm_params['std'] + norm_params['mean']
    lstm_targets = lstm_targets * norm_params['std'] + norm_params['mean']

    results['lstm_scratch'] = {
        'predictions': lstm_preds,
        'targets': lstm_targets,
        'history': lstm_history
    }

    # -------------------------------------------------------------------------
    # 2. HAR Model
    # -------------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 60)
        print("BASELINE: HAR Model")
        print("=" * 60)

    # Combine train + val for fitting
    full_train = np.concatenate([train_log_rv, val_log_rv])

    har_model = HARModel()
    har_model.fit(full_train)

    har_preds = har_model.predict(test_log_rv)
    har_targets = har_model.get_targets(test_log_rv)

    results['har'] = {
        'predictions': har_preds,
        'targets': har_targets
    }

    if verbose:
        har_rmse = np.sqrt(np.mean((har_preds - har_targets) ** 2))
        print(f"HAR Test RMSE: {har_rmse:.4f}")

    # -------------------------------------------------------------------------
    # 3. Moving Average
    # -------------------------------------------------------------------------
    if verbose:
        print("\n" + "=" * 60)
        print("BASELINE: Moving Average (window=5)")
        print("=" * 60)

    ma_model = MovingAverageBaseline(window=5)
    ma_preds, ma_targets = ma_model.predict(test_log_rv)

    results['moving_avg'] = {
        'predictions': ma_preds,
        'targets': ma_targets
    }

    if verbose:
        ma_rmse = np.sqrt(np.mean((ma_preds - ma_targets) ** 2))
        print(f"Moving Average Test RMSE: {ma_rmse:.4f}")

    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING BASELINE MODELS")
    print("=" * 60)

    # Load Bitcoin data
    btc_train, btc_val, btc_test = load_data_splits('data', 'btc')

    print(f"\nBitcoin data:")
    print(f"  Train: {len(btc_train)} samples")
    print(f"  Val:   {len(btc_val)} samples")
    print(f"  Test:  {len(btc_test)} samples")

    # Run baselines
    results = run_all_baselines(btc_train, btc_val, btc_test, verbose=True)

    # Summary
    print("\n" + "=" * 60)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 60)

    for name, data in results.items():
        rmse = np.sqrt(np.mean((data['predictions'] - data['targets']) ** 2))
        mae = np.mean(np.abs(data['predictions'] - data['targets']))
        print(f"{name:15s}: RMSE={rmse:.4f}, MAE={mae:.4f}")
