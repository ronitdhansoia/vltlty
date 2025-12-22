"""
Residual Boosting: Neural Network Error Correction for HAR

This approach trains a neural network to predict HAR's errors rather than
volatility directly. The final prediction is: HAR + NN(error_prediction)

The intuition:
- HAR captures the dominant linear patterns well
- NN learns to correct HAR's systematic errors
- No competition between models - they cooperate

Architecture:
1. Compute HAR predictions
2. Compute HAR residuals (actual - HAR)
3. Train NN to predict these residuals
4. Final: HAR + NN_correction

Author: Ronit Dhansoia
Date: 23rd December 2025
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

from datasets import load_data_splits


# ============================================================================
# NORMALIZER
# ============================================================================

class Normalizer:
    """Simple z-score normalizer."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.mean = np.mean(data)
        self.std = np.std(data)
        return (data - self.mean) / (self.std + 1e-8)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / (self.std + 1e-8)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean


# ============================================================================
# HAR MODEL (for computing residuals)
# ============================================================================

def compute_har_features(rv_series: np.ndarray, lookback: int = 22) -> np.ndarray:
    """
    Compute HAR features: daily, weekly (5-day), monthly (22-day) averages.
    """
    n = len(rv_series)
    features = []

    for i in range(lookback, n):
        # Daily RV (yesterday)
        rv_d = rv_series[i-1]

        # Weekly RV (5-day average)
        rv_w = np.mean(rv_series[i-5:i])

        # Monthly RV (22-day average)
        rv_m = np.mean(rv_series[i-22:i])

        features.append([rv_d, rv_w, rv_m])

    return np.array(features)


class HARModel:
    """Simple HAR model using OLS."""

    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, features: np.ndarray, targets: np.ndarray):
        """Fit HAR model using OLS."""
        X = np.column_stack([np.ones(len(features)), features])
        y = targets

        # OLS: (X'X)^{-1} X'y
        XtX = X.T @ X
        Xty = X.T @ y
        params = np.linalg.solve(XtX, Xty)

        self.intercept = params[0]
        self.coefficients = params[1:]

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict using HAR model."""
        return self.intercept + features @ self.coefficients


# ============================================================================
# RESIDUAL PREDICTION NETWORK
# ============================================================================

class ResidualPredictorLSTM(nn.Module):
    """
    LSTM network that predicts HAR's errors.

    Takes the full volatility sequence and HAR features as input,
    outputs a correction term to add to HAR's prediction.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        har_feature_dim: int = 3
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Process raw volatility sequence
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Process HAR features
        self.har_embed = nn.Sequential(
            nn.Linear(har_feature_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Combine and predict correction
        self.correction_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Initialize with small weights (start with small corrections)
        self._init_small()

    def _init_small(self):
        """Initialize final layer with small weights for small initial corrections."""
        for module in self.correction_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        vol_sequence: torch.Tensor,
        har_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            vol_sequence: [batch, seq_len, 1] - raw volatility sequence
            har_features: [batch, 3] - HAR features (daily, weekly, monthly)

        Returns:
            correction: [batch, 1] - correction to add to HAR prediction
        """
        # Process sequence
        lstm_out, (h_n, c_n) = self.lstm(vol_sequence)
        lstm_encoding = h_n[-1]  # [batch, hidden_dim]

        # Process HAR features
        har_embed = self.har_embed(har_features)  # [batch, hidden_dim//2]

        # Combine and predict correction
        combined = torch.cat([lstm_encoding, har_embed], dim=-1)
        correction = self.correction_head(combined)

        return correction


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_residual_data(
    train_rv: np.ndarray,
    val_rv: np.ndarray,
    test_rv: np.ndarray,
    lookback: int = 22
) -> Tuple[Dict, Dict, Dict, HARModel, Normalizer]:
    """
    Prepare data for residual learning.

    Args:
        train_rv: Training log RV array (already log-transformed)
        val_rv: Validation log RV array
        test_rv: Test log RV array
        lookback: Lookback window (default 22 for monthly)

    Returns dictionaries with sequences, HAR features, HAR predictions, and targets.
    """
    # Data is already log-transformed, just normalize
    normalizer = Normalizer()
    train_rv_norm = normalizer.fit_transform(train_rv)
    val_rv_norm = normalizer.transform(val_rv)
    test_rv_norm = normalizer.transform(test_rv)

    # Compute HAR features (on normalized data)
    train_har_feat = compute_har_features(train_rv_norm, lookback)
    val_har_feat = compute_har_features(val_rv_norm, lookback)
    test_har_feat = compute_har_features(test_rv_norm, lookback)

    # Get targets (next day RV)
    train_targets = train_rv_norm[lookback:]
    val_targets = val_rv_norm[lookback:]
    test_targets = test_rv_norm[lookback:]

    # Fit HAR model on training data
    har_model = HARModel()
    har_model.fit(train_har_feat, train_targets)

    # Get HAR predictions
    train_har_pred = har_model.predict(train_har_feat)
    val_har_pred = har_model.predict(val_har_feat)
    test_har_pred = har_model.predict(test_har_feat)

    # Compute HAR residuals (what we want NN to predict)
    train_residuals = train_targets - train_har_pred
    val_residuals = val_targets - val_har_pred
    test_residuals = test_targets - test_har_pred

    # Create sequences for LSTM
    def create_sequences(rv_norm, start_idx, n_samples):
        sequences = []
        for i in range(n_samples):
            seq = rv_norm[start_idx + i - lookback : start_idx + i]
            sequences.append(seq.reshape(-1, 1))
        return np.array(sequences)

    train_sequences = create_sequences(train_rv_norm, lookback, len(train_targets))
    val_sequences = create_sequences(val_rv_norm, lookback, len(val_targets))
    test_sequences = create_sequences(test_rv_norm, lookback, len(test_targets))

    # Package into dictionaries
    train_data = {
        'sequences': train_sequences,
        'har_features': train_har_feat,
        'har_predictions': train_har_pred,
        'targets': train_targets,
        'residuals': train_residuals
    }

    val_data = {
        'sequences': val_sequences,
        'har_features': val_har_feat,
        'har_predictions': val_har_pred,
        'targets': val_targets,
        'residuals': val_residuals
    }

    test_data = {
        'sequences': test_sequences,
        'har_features': test_har_feat,
        'har_predictions': test_har_pred,
        'targets': test_targets,
        'residuals': test_residuals
    }

    return train_data, val_data, test_data, har_model, normalizer


def create_residual_dataloaders(
    data: Dict,
    batch_size: int = 32,
    shuffle: bool = True
) -> DataLoader:
    """Create DataLoader for residual learning."""
    sequences = torch.FloatTensor(data['sequences'])
    har_features = torch.FloatTensor(data['har_features'])
    residuals = torch.FloatTensor(data['residuals']).unsqueeze(-1)

    dataset = TensorDataset(sequences, har_features, residuals)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ============================================================================
# TRAINING
# ============================================================================

def train_residual_predictor(
    model: ResidualPredictorLSTM,
    train_loader: DataLoader,
    val_data: Dict,
    epochs: int = 100,
    learning_rate: float = 0.001,
    patience: int = 15,
    device: torch.device = None
) -> Tuple[ResidualPredictorLSTM, Dict]:
    """
    Train the residual predictor.

    We train on residuals but evaluate on final predictions (HAR + correction).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    history = {
        'train_loss': [],
        'val_residual_mse': [],
        'val_final_rmse': [],
        'har_rmse': []
    }

    # Compute HAR baseline RMSE on validation
    har_rmse = np.sqrt(np.mean((val_data['har_predictions'] - val_data['targets']) ** 2))

    best_val_rmse = float('inf')
    best_state = None
    patience_counter = 0

    print(f"\nTraining Residual Predictor")
    print(f"HAR Validation RMSE: {har_rmse:.6f}")
    print("-" * 60)

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0.0

        for sequences, har_feat, residuals in train_loader:
            sequences = sequences.to(device)
            har_feat = har_feat.to(device)
            residuals = residuals.to(device)

            optimizer.zero_grad()
            corrections = model(sequences, har_feat)
            loss = criterion(corrections, residuals)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        with torch.no_grad():
            val_sequences = torch.FloatTensor(val_data['sequences']).to(device)
            val_har_feat = torch.FloatTensor(val_data['har_features']).to(device)

            val_corrections = model(val_sequences, val_har_feat).cpu().numpy().flatten()

        # Final predictions = HAR + NN correction
        val_final_pred = val_data['har_predictions'] + val_corrections
        val_final_rmse = np.sqrt(np.mean((val_final_pred - val_data['targets']) ** 2))

        # Residual MSE
        val_residual_mse = np.mean((val_corrections - val_data['residuals']) ** 2)

        scheduler.step(val_final_rmse)

        history['train_loss'].append(avg_train_loss)
        history['val_residual_mse'].append(val_residual_mse)
        history['val_final_rmse'].append(val_final_rmse)
        history['har_rmse'].append(har_rmse)

        # Early stopping on final RMSE
        if val_final_rmse < best_val_rmse:
            best_val_rmse = val_final_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            improvement = (har_rmse - val_final_rmse) / har_rmse * 100
            print(f"Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.6f} | "
                  f"Val RMSE: {val_final_rmse:.6f} | vs HAR: {improvement:+.2f}%")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_state)
    return model, history


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_residual_model(
    model: ResidualPredictorLSTM,
    test_data: Dict,
    device: torch.device = None
) -> Dict:
    """Evaluate the residual boosting model on test set."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        test_sequences = torch.FloatTensor(test_data['sequences']).to(device)
        test_har_feat = torch.FloatTensor(test_data['har_features']).to(device)

        corrections = model(test_sequences, test_har_feat).cpu().numpy().flatten()

    # Final predictions
    har_predictions = test_data['har_predictions']
    boosted_predictions = har_predictions + corrections
    targets = test_data['targets']

    # Metrics for HAR
    har_rmse = np.sqrt(np.mean((har_predictions - targets) ** 2))
    har_mae = np.mean(np.abs(har_predictions - targets))

    # Metrics for boosted model
    boost_rmse = np.sqrt(np.mean((boosted_predictions - targets) ** 2))
    boost_mae = np.mean(np.abs(boosted_predictions - targets))

    # Correlation
    har_corr = np.corrcoef(har_predictions, targets)[0, 1]
    boost_corr = np.corrcoef(boosted_predictions, targets)[0, 1]

    # R-squared
    ss_res_har = np.sum((targets - har_predictions) ** 2)
    ss_res_boost = np.sum((targets - boosted_predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    har_r2 = 1 - ss_res_har / ss_tot
    boost_r2 = 1 - ss_res_boost / ss_tot

    # Correction statistics
    correction_mean = np.mean(corrections)
    correction_std = np.std(corrections)
    correction_abs_mean = np.mean(np.abs(corrections))

    results = {
        'har': {
            'rmse': har_rmse,
            'mae': har_mae,
            'r_squared': har_r2,
            'correlation': har_corr
        },
        'boosted': {
            'rmse': boost_rmse,
            'mae': boost_mae,
            'r_squared': boost_r2,
            'correlation': boost_corr
        },
        'corrections': {
            'mean': correction_mean,
            'std': correction_std,
            'abs_mean': correction_abs_mean,
            'min': np.min(corrections),
            'max': np.max(corrections)
        },
        'improvement': {
            'rmse_pct': (har_rmse - boost_rmse) / har_rmse * 100,
            'mae_pct': (har_mae - boost_mae) / har_mae * 100,
            'r2_pct': (boost_r2 - har_r2) / har_r2 * 100
        },
        'predictions': {
            'har': har_predictions,
            'boosted': boosted_predictions,
            'corrections': corrections,
            'targets': targets
        }
    }

    return results


def plot_residual_results(results: Dict, history: Dict, save_path: str = None):
    """Plot residual boosting results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    predictions = results['predictions']

    # 1. Training history
    ax = axes[0, 0]
    epochs = range(1, len(history['val_final_rmse']) + 1)
    ax.plot(epochs, history['val_final_rmse'], 'b-', label='Boosted Model')
    ax.axhline(y=history['har_rmse'][0], color='r', linestyle='--', label='HAR Baseline')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation RMSE')
    ax.set_title('Training Progress: RMSE vs HAR')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Predictions comparison
    ax = axes[0, 1]
    n_points = min(200, len(predictions['targets']))
    x = range(n_points)
    ax.plot(x, predictions['targets'][:n_points], 'k-', label='Actual', alpha=0.7)
    ax.plot(x, predictions['har'][:n_points], 'r--', label='HAR', alpha=0.7)
    ax.plot(x, predictions['boosted'][:n_points], 'b-', label='Boosted', alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('Normalized RV')
    ax.set_title('Predictions Comparison (first 200 points)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Correction distribution
    ax = axes[0, 2]
    ax.hist(predictions['corrections'], bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='r', linestyle='--', label='Zero correction')
    ax.axvline(x=np.mean(predictions['corrections']), color='g', linestyle='-',
               label=f'Mean: {np.mean(predictions["corrections"]):.4f}')
    ax.set_xlabel('Correction Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of NN Corrections')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Scatter plot: HAR vs Actual
    ax = axes[1, 0]
    ax.scatter(predictions['targets'], predictions['har'], alpha=0.3, s=10, label='HAR')
    lims = [min(predictions['targets'].min(), predictions['har'].min()),
            max(predictions['targets'].max(), predictions['har'].max())]
    ax.plot(lims, lims, 'r--', label='Perfect')
    ax.set_xlabel('Actual RV')
    ax.set_ylabel('HAR Prediction')
    ax.set_title(f'HAR: R² = {results["har"]["r_squared"]:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Scatter plot: Boosted vs Actual
    ax = axes[1, 1]
    ax.scatter(predictions['targets'], predictions['boosted'], alpha=0.3, s=10,
               color='blue', label='Boosted')
    ax.plot(lims, lims, 'r--', label='Perfect')
    ax.set_xlabel('Actual RV')
    ax.set_ylabel('Boosted Prediction')
    ax.set_title(f'Boosted: R² = {results["boosted"]["r_squared"]:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Results summary
    ax = axes[1, 2]
    ax.axis('off')

    summary_text = f"""
    RESIDUAL BOOSTING RESULTS
    ========================

    HAR Baseline:
      RMSE:  {results['har']['rmse']:.6f}
      MAE:   {results['har']['mae']:.6f}
      R²:    {results['har']['r_squared']:.6f}
      Corr:  {results['har']['correlation']:.6f}

    Boosted Model (HAR + NN):
      RMSE:  {results['boosted']['rmse']:.6f}
      MAE:   {results['boosted']['mae']:.6f}
      R²:    {results['boosted']['r_squared']:.6f}
      Corr:  {results['boosted']['correlation']:.6f}

    Improvement:
      RMSE:  {results['improvement']['rmse_pct']:+.2f}%
      MAE:   {results['improvement']['mae_pct']:+.2f}%
      R²:    {results['improvement']['r2_pct']:+.2f}%

    Correction Stats:
      Mean:  {results['corrections']['mean']:.6f}
      Std:   {results['corrections']['std']:.6f}
      Range: [{results['corrections']['min']:.4f}, {results['corrections']['max']:.4f}]
    """

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved plot to: {save_path}")

    plt.close()


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_residual_boost_experiment():
    """Run the complete residual boosting experiment."""
    print("=" * 70)
    print("RESIDUAL BOOSTING: NEURAL NETWORK ERROR CORRECTION FOR HAR")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load data
    print("\n1. Loading Bitcoin data...")
    btc_train, btc_val, btc_test = load_data_splits('data', 'btc')
    print(f"   Train: {len(btc_train)}, Val: {len(btc_val)}, Test: {len(btc_test)}")

    # Prepare data
    print("\n2. Preparing data with HAR residuals...")
    train_data, val_data, test_data, har_model, normalizer = prepare_residual_data(
        btc_train, btc_val, btc_test, lookback=22
    )

    print(f"   Train residuals - Mean: {np.mean(train_data['residuals']):.6f}, "
          f"Std: {np.std(train_data['residuals']):.6f}")

    # Create dataloaders
    train_loader = create_residual_dataloaders(train_data, batch_size=32, shuffle=True)

    # Create model
    print("\n3. Creating Residual Predictor model...")
    model = ResidualPredictorLSTM(
        input_dim=1,
        hidden_dim=64,
        num_layers=2,
        dropout=0.2,
        har_feature_dim=3
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    # Train
    print("\n4. Training residual predictor...")
    model, history = train_residual_predictor(
        model, train_loader, val_data,
        epochs=100,
        learning_rate=0.001,
        patience=15,
        device=device
    )

    # Evaluate
    print("\n5. Evaluating on test set...")
    results = evaluate_residual_model(model, test_data, device)

    # Print results
    print("\n" + "=" * 70)
    print("FINAL RESULTS ON TEST SET")
    print("=" * 70)

    print(f"\nHAR Baseline:")
    print(f"  RMSE:        {results['har']['rmse']:.6f}")
    print(f"  MAE:         {results['har']['mae']:.6f}")
    print(f"  R²:          {results['har']['r_squared']:.6f}")
    print(f"  Correlation: {results['har']['correlation']:.6f}")

    print(f"\nBoosted Model (HAR + NN Correction):")
    print(f"  RMSE:        {results['boosted']['rmse']:.6f}")
    print(f"  MAE:         {results['boosted']['mae']:.6f}")
    print(f"  R²:          {results['boosted']['r_squared']:.6f}")
    print(f"  Correlation: {results['boosted']['correlation']:.6f}")

    print(f"\nImprovement over HAR:")
    print(f"  RMSE:  {results['improvement']['rmse_pct']:+.2f}%")
    print(f"  MAE:   {results['improvement']['mae_pct']:+.2f}%")
    print(f"  R²:    {results['improvement']['r2_pct']:+.2f}%")

    print(f"\nNN Correction Statistics:")
    print(f"  Mean correction:     {results['corrections']['mean']:.6f}")
    print(f"  Std correction:      {results['corrections']['std']:.6f}")
    print(f"  Mean |correction|:   {results['corrections']['abs_mean']:.6f}")
    print(f"  Range: [{results['corrections']['min']:.4f}, {results['corrections']['max']:.4f}]")

    # Plot results
    os.makedirs('results', exist_ok=True)
    plot_residual_results(results, history, 'results/residual_boost_results.png')

    # Save results - convert numpy floats to Python floats for JSON
    def convert_to_python_types(obj):
        if isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_save = convert_to_python_types({
        'har': results['har'],
        'boosted': results['boosted'],
        'corrections': results['corrections'],
        'improvement': results['improvement']
    })

    with open('results/residual_boost_results.json', 'w') as f:
        json.dump(results_save, f, indent=2)

    print("\n" + "=" * 70)
    if results['improvement']['rmse_pct'] > 0:
        print(f"SUCCESS! Boosted model beats HAR by {results['improvement']['rmse_pct']:.2f}%")
    else:
        print(f"HAR still leads by {-results['improvement']['rmse_pct']:.2f}%")
    print("=" * 70)

    return results, history


if __name__ == "__main__":
    results, history = run_residual_boost_experiment()
