"""
Regime-Aware Ensemble: Context-Dependent Model Combination

Key insight: HAR may perform differently in different market regimes.
This approach:
1. Identifies volatility regimes (low, medium, high)
2. Learns optimal model weights for each regime
3. Uses soft regime assignment for smooth transitions

The idea is that neural networks might capture non-linear patterns
better in certain market conditions (e.g., regime transitions).

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
from typing import Tuple, Dict, List
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

from datasets import load_data_splits


# ============================================================================
# DATA UTILITIES
# ============================================================================

class Normalizer:
    """Z-score normalizer."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        self.mean = np.mean(data)
        self.std = np.std(data)
        return (data - self.mean) / (self.std + 1e-8)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / (self.std + 1e-8)


def compute_extended_features(rv_series: np.ndarray, lookback: int = 22) -> np.ndarray:
    """
    Compute extended features including regime indicators.

    Features:
    - HAR features (daily, weekly, monthly)
    - Volatility of volatility (rolling std)
    - Trend (difference from weekly mean)
    - Regime indicator (percentile rank)
    - Rate of change
    """
    n = len(rv_series)
    features = []

    for i in range(lookback, n):
        # HAR features
        rv_d = rv_series[i-1]
        rv_w = np.mean(rv_series[i-5:i])
        rv_m = np.mean(rv_series[i-22:i])

        # Volatility of volatility (10-day rolling std)
        vol_of_vol = np.std(rv_series[i-10:i])

        # Trend: deviation from weekly mean
        trend = rv_d - rv_w

        # Regime indicator: percentile rank in recent history
        recent_rvs = rv_series[max(0, i-66):i]  # ~3 month window
        regime_pct = np.mean(recent_rvs < rv_d)  # percentile rank

        # Rate of change
        roc = rv_d - rv_series[i-2] if i >= 2 else 0

        # Acceleration (2nd derivative)
        if i >= 3:
            accel = (rv_series[i-1] - rv_series[i-2]) - (rv_series[i-2] - rv_series[i-3])
        else:
            accel = 0

        features.append([rv_d, rv_w, rv_m, vol_of_vol, trend, regime_pct, roc, accel])

    return np.array(features)


# ============================================================================
# HAR MODEL
# ============================================================================

class HARModel:
    """HAR model using OLS."""

    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, features: np.ndarray, targets: np.ndarray):
        """Fit using only HAR features (first 3 columns)."""
        X = np.column_stack([np.ones(len(features)), features[:, :3]])
        y = targets
        params = np.linalg.solve(X.T @ X, X.T @ y)
        self.intercept = params[0]
        self.coefficients = params[1:]

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict using HAR features."""
        return self.intercept + features[:, :3] @ self.coefficients


# ============================================================================
# REGIME-AWARE ENSEMBLE MODEL
# ============================================================================

class RegimeClassifier(nn.Module):
    """
    Soft regime classifier that outputs probabilities for each regime.
    Uses volatility context to determine regime membership.
    """

    def __init__(self, input_dim: int = 8, num_regimes: int = 3, hidden_dim: int = 32):
        super().__init__()
        self.num_regimes = num_regimes

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_regimes),
            nn.Softmax(dim=-1)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch, num_features]
        Returns:
            regime_probs: [batch, num_regimes] - soft regime assignments
        """
        return self.classifier(features)


class RegimeSpecificPredictor(nn.Module):
    """
    LSTM-based predictor for a specific regime.
    Each regime has its own predictor to capture regime-specific patterns.
    """

    def __init__(
        self,
        seq_len: int = 22,
        input_dim: int = 1,
        hidden_dim: int = 32,
        feature_dim: int = 8
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim + feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, sequence: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequence: [batch, seq_len, 1]
            features: [batch, feature_dim]
        Returns:
            prediction: [batch, 1]
        """
        _, (h_n, _) = self.lstm(sequence)
        encoding = h_n[-1]  # [batch, hidden_dim]
        combined = torch.cat([encoding, features], dim=-1)
        return self.predictor(combined)


class RegimeAwareEnsemble(nn.Module):
    """
    Ensemble that combines HAR with regime-specific neural predictors.

    For each sample:
    1. Soft-classify into regimes
    2. Each regime predictor outputs a correction to HAR
    3. Final = HAR + weighted sum of regime corrections
    """

    def __init__(
        self,
        seq_len: int = 22,
        feature_dim: int = 8,
        num_regimes: int = 3,
        hidden_dim: int = 32
    ):
        super().__init__()

        self.num_regimes = num_regimes

        # Regime classifier
        self.regime_classifier = RegimeClassifier(
            input_dim=feature_dim,
            num_regimes=num_regimes,
            hidden_dim=hidden_dim
        )

        # Regime-specific predictors (predict corrections)
        self.regime_predictors = nn.ModuleList([
            RegimeSpecificPredictor(
                seq_len=seq_len,
                input_dim=1,
                hidden_dim=hidden_dim,
                feature_dim=feature_dim
            )
            for _ in range(num_regimes)
        ])

        # Learnable base weight for HAR
        self.har_weight = nn.Parameter(torch.tensor(0.95))

    def forward(
        self,
        sequence: torch.Tensor,
        features: torch.Tensor,
        har_pred: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sequence: [batch, seq_len, 1] - raw volatility sequence
            features: [batch, feature_dim] - extended features
            har_pred: [batch, 1] - HAR predictions

        Returns:
            final_pred: [batch, 1] - ensemble prediction
            regime_probs: [batch, num_regimes] - soft regime assignments
        """
        # Get regime probabilities
        regime_probs = self.regime_classifier(features)  # [batch, num_regimes]

        # Get predictions from each regime-specific model
        regime_corrections = []
        for predictor in self.regime_predictors:
            correction = predictor(sequence, features)  # [batch, 1]
            regime_corrections.append(correction)

        regime_corrections = torch.stack(regime_corrections, dim=1)  # [batch, num_regimes, 1]

        # Weighted combination of corrections
        weighted_correction = torch.sum(
            regime_probs.unsqueeze(-1) * regime_corrections, dim=1
        )  # [batch, 1]

        # Final prediction: HAR + correction (with learnable weight)
        har_weight = torch.sigmoid(self.har_weight)  # Constrain to (0, 1)
        final_pred = har_weight * har_pred + (1 - har_weight) * (har_pred + weighted_correction)

        return final_pred, regime_probs


# ============================================================================
# TRAINING
# ============================================================================

def prepare_regime_data(
    train_rv: np.ndarray,
    val_rv: np.ndarray,
    test_rv: np.ndarray,
    lookback: int = 22
) -> Tuple[Dict, Dict, Dict, HARModel, Normalizer]:
    """Prepare data for regime-aware ensemble."""

    normalizer = Normalizer()
    train_norm = normalizer.fit_transform(train_rv)
    val_norm = normalizer.transform(val_rv)
    test_norm = normalizer.transform(test_rv)

    # Compute extended features
    train_feat = compute_extended_features(train_norm, lookback)
    val_feat = compute_extended_features(val_norm, lookback)
    test_feat = compute_extended_features(test_norm, lookback)

    # Get targets
    train_targets = train_norm[lookback:]
    val_targets = val_norm[lookback:]
    test_targets = test_norm[lookback:]

    # Fit HAR model
    har_model = HARModel()
    har_model.fit(train_feat, train_targets)

    # Get HAR predictions
    train_har = har_model.predict(train_feat)
    val_har = har_model.predict(val_feat)
    test_har = har_model.predict(test_feat)

    # Create sequences
    def create_sequences(rv_norm, start_idx, n_samples, lookback):
        sequences = []
        for i in range(n_samples):
            seq = rv_norm[start_idx + i - lookback : start_idx + i]
            sequences.append(seq.reshape(-1, 1))
        return np.array(sequences)

    train_seq = create_sequences(train_norm, lookback, len(train_targets), lookback)
    val_seq = create_sequences(val_norm, lookback, len(val_targets), lookback)
    test_seq = create_sequences(test_norm, lookback, len(test_targets), lookback)

    train_data = {
        'sequences': train_seq,
        'features': train_feat,
        'har_pred': train_har,
        'targets': train_targets
    }
    val_data = {
        'sequences': val_seq,
        'features': val_feat,
        'har_pred': val_har,
        'targets': val_targets
    }
    test_data = {
        'sequences': test_seq,
        'features': test_feat,
        'har_pred': test_har,
        'targets': test_targets
    }

    return train_data, val_data, test_data, har_model, normalizer


def train_regime_ensemble(
    model: RegimeAwareEnsemble,
    train_data: Dict,
    val_data: Dict,
    epochs: int = 100,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    patience: int = 15,
    device: torch.device = None
) -> Tuple[RegimeAwareEnsemble, Dict]:
    """Train the regime-aware ensemble."""

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)

    # Create DataLoader
    train_dataset = TensorDataset(
        torch.FloatTensor(train_data['sequences']),
        torch.FloatTensor(train_data['features']),
        torch.FloatTensor(train_data['har_pred']).unsqueeze(-1),
        torch.FloatTensor(train_data['targets']).unsqueeze(-1)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    history = {
        'train_loss': [],
        'val_rmse': [],
        'har_rmse': [],
        'regime_entropy': []
    }

    # HAR baseline
    har_rmse = np.sqrt(np.mean((val_data['har_pred'] - val_data['targets']) ** 2))

    best_val_rmse = float('inf')
    best_state = None
    patience_counter = 0

    print(f"\nTraining Regime-Aware Ensemble")
    print(f"HAR Validation RMSE: {har_rmse:.6f}")
    print("-" * 60)

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0.0
        total_entropy = 0.0

        for seq, feat, har, target in train_loader:
            seq, feat, har, target = seq.to(device), feat.to(device), har.to(device), target.to(device)

            optimizer.zero_grad()
            pred, regime_probs = model(seq, feat, har)
            loss = criterion(pred, target)

            # Add entropy regularization to encourage regime specialization
            entropy = -torch.mean(torch.sum(regime_probs * torch.log(regime_probs + 1e-8), dim=-1))
            loss_total = loss - 0.01 * entropy  # Minimize entropy = more distinct regimes

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            total_entropy += entropy.item()

        avg_loss = total_loss / len(train_loader)
        avg_entropy = total_entropy / len(train_loader)

        # Validation
        model.eval()
        with torch.no_grad():
            val_seq = torch.FloatTensor(val_data['sequences']).to(device)
            val_feat = torch.FloatTensor(val_data['features']).to(device)
            val_har = torch.FloatTensor(val_data['har_pred']).unsqueeze(-1).to(device)

            val_pred, val_regimes = model(val_seq, val_feat, val_har)
            val_pred = val_pred.cpu().numpy().flatten()

        val_rmse = np.sqrt(np.mean((val_pred - val_data['targets']) ** 2))
        scheduler.step(val_rmse)

        history['train_loss'].append(avg_loss)
        history['val_rmse'].append(val_rmse)
        history['har_rmse'].append(har_rmse)
        history['regime_entropy'].append(avg_entropy)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            improvement = (har_rmse - val_rmse) / har_rmse * 100
            har_w = torch.sigmoid(model.har_weight).item()
            print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.6f} | Val RMSE: {val_rmse:.6f} | "
                  f"vs HAR: {improvement:+.2f}% | HAR_w: {har_w:.3f}")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_state)
    return model, history


def evaluate_regime_ensemble(
    model: RegimeAwareEnsemble,
    test_data: Dict,
    device: torch.device = None
) -> Dict:
    """Evaluate the regime-aware ensemble."""

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        test_seq = torch.FloatTensor(test_data['sequences']).to(device)
        test_feat = torch.FloatTensor(test_data['features']).to(device)
        test_har = torch.FloatTensor(test_data['har_pred']).unsqueeze(-1).to(device)

        predictions, regime_probs = model(test_seq, test_feat, test_har)
        predictions = predictions.cpu().numpy().flatten()
        regime_probs = regime_probs.cpu().numpy()

    targets = test_data['targets']
    har_pred = test_data['har_pred']

    # Metrics
    har_rmse = np.sqrt(np.mean((har_pred - targets) ** 2))
    ens_rmse = np.sqrt(np.mean((predictions - targets) ** 2))

    har_mae = np.mean(np.abs(har_pred - targets))
    ens_mae = np.mean(np.abs(predictions - targets))

    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    har_r2 = 1 - np.sum((targets - har_pred) ** 2) / ss_tot
    ens_r2 = 1 - np.sum((targets - predictions) ** 2) / ss_tot

    har_corr = np.corrcoef(har_pred, targets)[0, 1]
    ens_corr = np.corrcoef(predictions, targets)[0, 1]

    # Regime statistics
    regime_assignments = np.argmax(regime_probs, axis=1)
    regime_counts = np.bincount(regime_assignments, minlength=model.num_regimes)

    results = {
        'har': {'rmse': har_rmse, 'mae': har_mae, 'r2': har_r2, 'corr': har_corr},
        'ensemble': {'rmse': ens_rmse, 'mae': ens_mae, 'r2': ens_r2, 'corr': ens_corr},
        'improvement': {
            'rmse_pct': (har_rmse - ens_rmse) / har_rmse * 100,
            'mae_pct': (har_mae - ens_mae) / har_mae * 100,
            'r2_pct': (ens_r2 - har_r2) / har_r2 * 100
        },
        'regime_stats': {
            'counts': regime_counts.tolist(),
            'mean_probs': np.mean(regime_probs, axis=0).tolist(),
            'entropy_mean': -np.mean(np.sum(regime_probs * np.log(regime_probs + 1e-8), axis=-1))
        },
        'predictions': {
            'har': har_pred,
            'ensemble': predictions,
            'targets': targets,
            'regime_probs': regime_probs
        }
    }

    return results


def plot_regime_results(results: Dict, history: Dict, save_path: str = None):
    """Plot regime ensemble results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    predictions = results['predictions']

    # 1. Training progress
    ax = axes[0, 0]
    epochs = range(1, len(history['val_rmse']) + 1)
    ax.plot(epochs, history['val_rmse'], 'b-', label='Ensemble')
    ax.axhline(y=history['har_rmse'][0], color='r', linestyle='--', label='HAR')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation RMSE')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Predictions comparison
    ax = axes[0, 1]
    n_pts = min(200, len(predictions['targets']))
    ax.plot(range(n_pts), predictions['targets'][:n_pts], 'k-', label='Actual', alpha=0.7)
    ax.plot(range(n_pts), predictions['har'][:n_pts], 'r--', label='HAR', alpha=0.7)
    ax.plot(range(n_pts), predictions['ensemble'][:n_pts], 'b-', label='Ensemble', alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('Normalized RV')
    ax.set_title('Predictions (first 200 points)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Regime distribution
    ax = axes[0, 2]
    regime_probs = predictions['regime_probs']
    for i in range(regime_probs.shape[1]):
        ax.hist(regime_probs[:, i], bins=30, alpha=0.5, label=f'Regime {i+1}')
    ax.set_xlabel('Probability')
    ax.set_ylabel('Frequency')
    ax.set_title('Regime Probability Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Scatter: HAR vs Actual
    ax = axes[1, 0]
    ax.scatter(predictions['targets'], predictions['har'], alpha=0.3, s=10)
    lims = [min(predictions['targets'].min(), predictions['har'].min()),
            max(predictions['targets'].max(), predictions['har'].max())]
    ax.plot(lims, lims, 'r--')
    ax.set_xlabel('Actual')
    ax.set_ylabel('HAR Prediction')
    ax.set_title(f'HAR: R² = {results["har"]["r2"]:.4f}')
    ax.grid(True, alpha=0.3)

    # 5. Scatter: Ensemble vs Actual
    ax = axes[1, 1]
    ax.scatter(predictions['targets'], predictions['ensemble'], alpha=0.3, s=10, c='blue')
    ax.plot(lims, lims, 'r--')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Ensemble Prediction')
    ax.set_title(f'Ensemble: R² = {results["ensemble"]["r2"]:.4f}')
    ax.grid(True, alpha=0.3)

    # 6. Results summary
    ax = axes[1, 2]
    ax.axis('off')

    summary = f"""
    REGIME-AWARE ENSEMBLE RESULTS
    ============================

    HAR Baseline:
      RMSE:  {results['har']['rmse']:.6f}
      MAE:   {results['har']['mae']:.6f}
      R²:    {results['har']['r2']:.6f}

    Ensemble Model:
      RMSE:  {results['ensemble']['rmse']:.6f}
      MAE:   {results['ensemble']['mae']:.6f}
      R²:    {results['ensemble']['r2']:.6f}

    Improvement:
      RMSE:  {results['improvement']['rmse_pct']:+.2f}%
      MAE:   {results['improvement']['mae_pct']:+.2f}%
      R²:    {results['improvement']['r2_pct']:+.2f}%

    Regime Statistics:
      Counts: {results['regime_stats']['counts']}
      Mean probs: {[f'{p:.2f}' for p in results['regime_stats']['mean_probs']]}
    """

    ax.text(0.1, 0.9, summary, transform=ax.transAxes,
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

def run_regime_ensemble_experiment():
    """Run the regime-aware ensemble experiment."""
    print("=" * 70)
    print("REGIME-AWARE ENSEMBLE FOR VOLATILITY FORECASTING")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # Load data
    print("\n1. Loading Bitcoin data...")
    btc_train, btc_val, btc_test = load_data_splits('data', 'btc')
    print(f"   Train: {len(btc_train)}, Val: {len(btc_val)}, Test: {len(btc_test)}")

    # Prepare data
    print("\n2. Preparing data with extended features...")
    lookback = 22
    train_data, val_data, test_data, har_model, normalizer = prepare_regime_data(
        btc_train, btc_val, btc_test, lookback=lookback
    )

    print(f"   Features shape: {train_data['features'].shape}")

    # Create model
    print("\n3. Creating Regime-Aware Ensemble...")
    model = RegimeAwareEnsemble(
        seq_len=lookback,
        feature_dim=8,
        num_regimes=3,
        hidden_dim=32
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    # Train
    print("\n4. Training...")
    model, history = train_regime_ensemble(
        model, train_data, val_data,
        epochs=100,
        learning_rate=0.001,
        batch_size=32,
        patience=15,
        device=device
    )

    # Evaluate
    print("\n5. Evaluating on test set...")
    results = evaluate_regime_ensemble(model, test_data, device)

    # Print results
    print("\n" + "=" * 70)
    print("FINAL RESULTS ON TEST SET")
    print("=" * 70)

    print(f"\nHAR Baseline:")
    print(f"  RMSE:        {results['har']['rmse']:.6f}")
    print(f"  MAE:         {results['har']['mae']:.6f}")
    print(f"  R²:          {results['har']['r2']:.6f}")
    print(f"  Correlation: {results['har']['corr']:.6f}")

    print(f"\nRegime-Aware Ensemble:")
    print(f"  RMSE:        {results['ensemble']['rmse']:.6f}")
    print(f"  MAE:         {results['ensemble']['mae']:.6f}")
    print(f"  R²:          {results['ensemble']['r2']:.6f}")
    print(f"  Correlation: {results['ensemble']['corr']:.6f}")

    print(f"\nImprovement over HAR:")
    print(f"  RMSE:  {results['improvement']['rmse_pct']:+.2f}%")
    print(f"  MAE:   {results['improvement']['mae_pct']:+.2f}%")
    print(f"  R²:    {results['improvement']['r2_pct']:+.2f}%")

    print(f"\nRegime Statistics:")
    print(f"  Regime counts: {results['regime_stats']['counts']}")
    print(f"  Mean probabilities: {[f'{p:.3f}' for p in results['regime_stats']['mean_probs']]}")

    # Plot
    os.makedirs('results', exist_ok=True)
    plot_regime_results(results, history, 'results/regime_ensemble_results.png')

    # Save results (without numpy arrays)
    def to_python(obj):
        if isinstance(obj, dict):
            return {k: to_python(v) for k, v in obj.items()}
        elif isinstance(obj, np.ndarray):
            return None  # Skip arrays
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, list):
            return [to_python(x) for x in obj]
        return obj

    results_save = {
        'har': to_python(results['har']),
        'ensemble': to_python(results['ensemble']),
        'improvement': to_python(results['improvement']),
        'regime_stats': to_python(results['regime_stats'])
    }

    with open('results/regime_ensemble_results.json', 'w') as f:
        json.dump(results_save, f, indent=2)

    print("\n" + "=" * 70)
    if results['improvement']['rmse_pct'] > 0:
        print(f"SUCCESS! Ensemble beats HAR by {results['improvement']['rmse_pct']:.2f}%")
    else:
        print(f"HAR still leads by {-results['improvement']['rmse_pct']:.2f}%")
    print("=" * 70)

    return results, history


if __name__ == "__main__":
    results, history = run_regime_ensemble_experiment()
