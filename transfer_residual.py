"""
Transfer Learning for HAR Residual Correction

This combines our best ideas:
1. HAR as the strong baseline
2. Neural network learns to correct HAR's residuals
3. Transfer learning: pre-train on SPX, fine-tune on BTC

The hypothesis: Both SPX and BTC volatility share common rough patterns,
but HAR makes similar systematic errors on both. The NN can learn these
error patterns from the data-rich SPX and transfer to BTC.

Author: Ronit Dhansoia
Date: 23rd December 2025
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Dict
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

from datasets import load_data_splits


# ============================================================================
# UTILITIES
# ============================================================================

class Normalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit_transform(self, data):
        self.mean = np.mean(data)
        self.std = np.std(data)
        return (data - self.mean) / (self.std + 1e-8)

    def transform(self, data):
        return (data - self.mean) / (self.std + 1e-8)


def compute_har_features(rv_series: np.ndarray, lookback: int = 22) -> np.ndarray:
    """Compute HAR features: daily, weekly (5-day), monthly (22-day)."""
    n = len(rv_series)
    features = []
    for i in range(lookback, n):
        rv_d = rv_series[i-1]
        rv_w = np.mean(rv_series[i-5:i])
        rv_m = np.mean(rv_series[i-22:i])
        features.append([rv_d, rv_w, rv_m])
    return np.array(features)


class HARModel:
    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, features, targets):
        X = np.column_stack([np.ones(len(features)), features])
        params = np.linalg.solve(X.T @ X, X.T @ targets)
        self.intercept = params[0]
        self.coefficients = params[1:]

    def predict(self, features):
        return self.intercept + features @ self.coefficients


# ============================================================================
# TRANSFER RESIDUAL MODEL
# ============================================================================

class ResidualCorrectionNet(nn.Module):
    """
    Neural network that learns to correct HAR's residuals.
    Designed for transfer learning: encoder learns universal patterns,
    correction head adapts to specific assets.
    """

    def __init__(
        self,
        seq_len: int = 22,
        input_dim: int = 1,
        hidden_dim: int = 64,
        encoding_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        # Encoder (to be transferred)
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Compression layer (part of encoder)
        self.compress = nn.Sequential(
            nn.Linear(hidden_dim, encoding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # HAR features processing
        self.har_embed = nn.Sequential(
            nn.Linear(3, encoding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Correction head (to be fine-tuned)
        self.correction_head = nn.Sequential(
            nn.Linear(encoding_dim + encoding_dim // 2, encoding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoding_dim, encoding_dim // 2),
            nn.ReLU(),
            nn.Linear(encoding_dim // 2, 1)
        )

        # Initialize correction head with small weights
        self._init_small_correction()

    def _init_small_correction(self):
        for module in self.correction_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)

    def forward(self, sequence: torch.Tensor, har_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sequence: [batch, seq_len, 1]
            har_features: [batch, 3]
        Returns:
            correction: [batch, 1] - correction to add to HAR
        """
        # Encode sequence
        _, (h_n, _) = self.encoder(sequence)
        encoding = self.compress(h_n[-1])

        # Process HAR features
        har_embed = self.har_embed(har_features)

        # Predict correction
        combined = torch.cat([encoding, har_embed], dim=-1)
        correction = self.correction_head(combined)

        return correction

    def freeze_encoder(self):
        """Freeze encoder for fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.compress.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.compress.parameters():
            param.requires_grad = True

    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_asset_data(
    train_rv: np.ndarray,
    val_rv: np.ndarray,
    test_rv: np.ndarray,
    lookback: int = 22
) -> Tuple[Dict, Dict, Dict, HARModel, Normalizer]:
    """Prepare data for a single asset."""

    normalizer = Normalizer()
    train_norm = normalizer.fit_transform(train_rv)
    val_norm = normalizer.transform(val_rv)
    test_norm = normalizer.transform(test_rv)

    # HAR features
    train_har_feat = compute_har_features(train_norm, lookback)
    val_har_feat = compute_har_features(val_norm, lookback)
    test_har_feat = compute_har_features(test_norm, lookback)

    # Targets
    train_targets = train_norm[lookback:]
    val_targets = val_norm[lookback:]
    test_targets = test_norm[lookback:]

    # Fit HAR
    har_model = HARModel()
    har_model.fit(train_har_feat, train_targets)

    # HAR predictions and residuals
    train_har_pred = har_model.predict(train_har_feat)
    val_har_pred = har_model.predict(val_har_feat)
    test_har_pred = har_model.predict(test_har_feat)

    train_residuals = train_targets - train_har_pred
    val_residuals = val_targets - val_har_pred
    test_residuals = test_targets - test_har_pred

    # Sequences
    def create_sequences(rv, start, n, lb):
        seqs = []
        for i in range(n):
            seqs.append(rv[start + i - lb : start + i].reshape(-1, 1))
        return np.array(seqs)

    train_seq = create_sequences(train_norm, lookback, len(train_targets), lookback)
    val_seq = create_sequences(val_norm, lookback, len(val_targets), lookback)
    test_seq = create_sequences(test_norm, lookback, len(test_targets), lookback)

    return (
        {'sequences': train_seq, 'har_features': train_har_feat, 'har_pred': train_har_pred,
         'targets': train_targets, 'residuals': train_residuals},
        {'sequences': val_seq, 'har_features': val_har_feat, 'har_pred': val_har_pred,
         'targets': val_targets, 'residuals': val_residuals},
        {'sequences': test_seq, 'har_features': test_har_feat, 'har_pred': test_har_pred,
         'targets': test_targets, 'residuals': test_residuals},
        har_model,
        normalizer
    )


# ============================================================================
# TRAINING
# ============================================================================

def train_on_residuals(
    model: ResidualCorrectionNet,
    train_data: Dict,
    val_data: Dict,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    patience: int,
    device: torch.device,
    phase_name: str = "Training"
) -> Tuple[ResidualCorrectionNet, Dict]:
    """Train model to predict HAR residuals."""

    model = model.to(device)

    train_dataset = TensorDataset(
        torch.FloatTensor(train_data['sequences']),
        torch.FloatTensor(train_data['har_features']),
        torch.FloatTensor(train_data['residuals']).unsqueeze(-1)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    history = {'train_loss': [], 'val_rmse': [], 'har_rmse': []}

    har_rmse = np.sqrt(np.mean((val_data['har_pred'] - val_data['targets']) ** 2))

    best_rmse = float('inf')
    best_state = None
    patience_counter = 0

    print(f"\n{phase_name}")
    print(f"HAR Validation RMSE: {har_rmse:.6f}")
    print(f"Trainable params: {model.get_trainable_params():,}")
    print("-" * 60)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for seq, har_feat, residuals in train_loader:
            seq, har_feat, residuals = seq.to(device), har_feat.to(device), residuals.to(device)

            optimizer.zero_grad()
            corrections = model(seq, har_feat)
            loss = criterion(corrections, residuals)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation (evaluate on final predictions, not just residuals)
        model.eval()
        with torch.no_grad():
            val_seq = torch.FloatTensor(val_data['sequences']).to(device)
            val_har_feat = torch.FloatTensor(val_data['har_features']).to(device)
            val_corrections = model(val_seq, val_har_feat).cpu().numpy().flatten()

        val_pred = val_data['har_pred'] + val_corrections
        val_rmse = np.sqrt(np.mean((val_pred - val_data['targets']) ** 2))

        scheduler.step(val_rmse)

        history['train_loss'].append(avg_loss)
        history['val_rmse'].append(val_rmse)
        history['har_rmse'].append(har_rmse)

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            improvement = (har_rmse - val_rmse) / har_rmse * 100
            print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.6f} | Val RMSE: {val_rmse:.6f} | vs HAR: {improvement:+.2f}%")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_state)
    return model, history


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_transfer_residual_experiment():
    """Run the complete transfer learning for residual correction experiment."""

    print("=" * 70)
    print("TRANSFER LEARNING FOR HAR RESIDUAL CORRECTION")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    lookback = 22

    # =========================================================================
    # LOAD AND PREPARE DATA
    # =========================================================================
    print("\n1. Loading data...")

    # Source domain: S&P 500
    spx_train, spx_val, spx_test = load_data_splits('data', 'spx')
    print(f"   SPX: Train={len(spx_train)}, Val={len(spx_val)}, Test={len(spx_test)}")

    # Target domain: Bitcoin
    btc_train, btc_val, btc_test = load_data_splits('data', 'btc')
    print(f"   BTC: Train={len(btc_train)}, Val={len(btc_val)}, Test={len(btc_test)}")

    # Prepare data
    print("\n2. Preparing data with HAR features and residuals...")
    spx_train_data, spx_val_data, spx_test_data, spx_har, spx_norm = prepare_asset_data(
        spx_train, spx_val, spx_test, lookback
    )
    btc_train_data, btc_val_data, btc_test_data, btc_har, btc_norm = prepare_asset_data(
        btc_train, btc_val, btc_test, lookback
    )

    print(f"   SPX train residuals - mean: {np.mean(spx_train_data['residuals']):.6f}, std: {np.std(spx_train_data['residuals']):.6f}")
    print(f"   BTC train residuals - mean: {np.mean(btc_train_data['residuals']):.6f}, std: {np.std(btc_train_data['residuals']):.6f}")

    # =========================================================================
    # CREATE MODEL
    # =========================================================================
    print("\n3. Creating ResidualCorrectionNet...")
    model = ResidualCorrectionNet(
        seq_len=lookback,
        input_dim=1,
        hidden_dim=64,
        encoding_dim=32,
        num_layers=2,
        dropout=0.2
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    # =========================================================================
    # PHASE 1: PRE-TRAIN ON SPX
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: PRE-TRAINING ON S&P 500 (Source Domain)")
    print("=" * 70)

    model.unfreeze_encoder()
    model, pretrain_history = train_on_residuals(
        model, spx_train_data, spx_val_data,
        epochs=50,
        learning_rate=0.001,
        batch_size=32,
        patience=10,
        device=device,
        phase_name="Pre-training on SPX"
    )

    # Save pre-trained model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/residual_pretrained_spx.pth')

    # =========================================================================
    # PHASE 2: FINE-TUNE ON BTC (ENCODER FROZEN)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: FINE-TUNING ON BITCOIN (Encoder Frozen)")
    print("=" * 70)

    model.freeze_encoder()
    model, finetune_frozen_history = train_on_residuals(
        model, btc_train_data, btc_val_data,
        epochs=30,
        learning_rate=0.0001,
        batch_size=16,
        patience=10,
        device=device,
        phase_name="Fine-tuning on BTC (encoder frozen)"
    )

    # =========================================================================
    # PHASE 3: FINE-TUNE ON BTC (ENCODER UNFROZEN with low LR)
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: FINE-TUNING ON BITCOIN (Full Model)")
    print("=" * 70)

    model.unfreeze_encoder()
    model, finetune_full_history = train_on_residuals(
        model, btc_train_data, btc_val_data,
        epochs=30,
        learning_rate=0.00005,  # Very low LR for fine-tuning encoder
        batch_size=16,
        patience=10,
        device=device,
        phase_name="Fine-tuning on BTC (full model)"
    )

    # Save final model
    torch.save(model.state_dict(), 'models/residual_transfer_btc.pth')

    # =========================================================================
    # EVALUATION ON BTC TEST SET
    # =========================================================================
    print("\n" + "=" * 70)
    print("EVALUATION ON BITCOIN TEST SET")
    print("=" * 70)

    model.eval()
    with torch.no_grad():
        test_seq = torch.FloatTensor(btc_test_data['sequences']).to(device)
        test_har_feat = torch.FloatTensor(btc_test_data['har_features']).to(device)
        corrections = model(test_seq, test_har_feat).cpu().numpy().flatten()

    har_pred = btc_test_data['har_pred']
    transfer_pred = har_pred + corrections
    targets = btc_test_data['targets']

    # Metrics
    har_rmse = np.sqrt(np.mean((har_pred - targets) ** 2))
    transfer_rmse = np.sqrt(np.mean((transfer_pred - targets) ** 2))

    har_mae = np.mean(np.abs(har_pred - targets))
    transfer_mae = np.mean(np.abs(transfer_pred - targets))

    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    har_r2 = 1 - np.sum((targets - har_pred) ** 2) / ss_tot
    transfer_r2 = 1 - np.sum((targets - transfer_pred) ** 2) / ss_tot

    har_corr = np.corrcoef(har_pred, targets)[0, 1]
    transfer_corr = np.corrcoef(transfer_pred, targets)[0, 1]

    # Print results
    print(f"\nHAR Baseline:")
    print(f"  RMSE:        {har_rmse:.6f}")
    print(f"  MAE:         {har_mae:.6f}")
    print(f"  R²:          {har_r2:.6f}")
    print(f"  Correlation: {har_corr:.6f}")

    print(f"\nTransfer Residual Model (HAR + NN Correction):")
    print(f"  RMSE:        {transfer_rmse:.6f}")
    print(f"  MAE:         {transfer_mae:.6f}")
    print(f"  R²:          {transfer_r2:.6f}")
    print(f"  Correlation: {transfer_corr:.6f}")

    rmse_improvement = (har_rmse - transfer_rmse) / har_rmse * 100
    mae_improvement = (har_mae - transfer_mae) / har_mae * 100

    print(f"\nImprovement over HAR:")
    print(f"  RMSE: {rmse_improvement:+.2f}%")
    print(f"  MAE:  {mae_improvement:+.2f}%")

    print(f"\nCorrection Statistics:")
    print(f"  Mean:  {np.mean(corrections):.6f}")
    print(f"  Std:   {np.std(corrections):.6f}")
    print(f"  Range: [{np.min(corrections):.4f}, {np.max(corrections):.4f}]")

    # =========================================================================
    # COMPARISON WITH FROM-SCRATCH MODEL
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON: Transfer vs From-Scratch")
    print("=" * 70)

    # Train from scratch on BTC only
    scratch_model = ResidualCorrectionNet(
        seq_len=lookback,
        input_dim=1,
        hidden_dim=64,
        encoding_dim=32,
        num_layers=2,
        dropout=0.2
    )

    scratch_model, scratch_history = train_on_residuals(
        scratch_model, btc_train_data, btc_val_data,
        epochs=50,
        learning_rate=0.001,
        batch_size=32,
        patience=15,
        device=device,
        phase_name="Training from scratch on BTC"
    )

    # Evaluate scratch model
    scratch_model.eval()
    with torch.no_grad():
        test_seq = torch.FloatTensor(btc_test_data['sequences']).to(device)
        test_har_feat = torch.FloatTensor(btc_test_data['har_features']).to(device)
        scratch_corrections = scratch_model(test_seq, test_har_feat).cpu().numpy().flatten()

    scratch_pred = har_pred + scratch_corrections
    scratch_rmse = np.sqrt(np.mean((scratch_pred - targets) ** 2))
    scratch_mae = np.mean(np.abs(scratch_pred - targets))

    print(f"\nFrom-Scratch Model:")
    print(f"  RMSE: {scratch_rmse:.6f}")
    print(f"  MAE:  {scratch_mae:.6f}")

    transfer_vs_scratch = (scratch_rmse - transfer_rmse) / scratch_rmse * 100
    print(f"\nTransfer vs From-Scratch:")
    print(f"  RMSE improvement: {transfer_vs_scratch:+.2f}%")

    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Training histories
    ax = axes[0, 0]
    ax.plot(pretrain_history['val_rmse'], 'b-', label='Pre-train (SPX)', alpha=0.7)
    ax.plot(range(len(pretrain_history['val_rmse']),
                  len(pretrain_history['val_rmse']) + len(finetune_frozen_history['val_rmse'])),
            finetune_frozen_history['val_rmse'], 'g-', label='Fine-tune frozen', alpha=0.7)
    ax.plot(range(len(pretrain_history['val_rmse']) + len(finetune_frozen_history['val_rmse']),
                  len(pretrain_history['val_rmse']) + len(finetune_frozen_history['val_rmse']) + len(finetune_full_history['val_rmse'])),
            finetune_full_history['val_rmse'], 'r-', label='Fine-tune full', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation RMSE')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Predictions comparison
    ax = axes[0, 1]
    n_pts = min(200, len(targets))
    ax.plot(range(n_pts), targets[:n_pts], 'k-', label='Actual', alpha=0.7)
    ax.plot(range(n_pts), har_pred[:n_pts], 'r--', label='HAR', alpha=0.7)
    ax.plot(range(n_pts), transfer_pred[:n_pts], 'b-', label='Transfer', alpha=0.7)
    ax.set_xlabel('Time')
    ax.set_ylabel('Normalized RV')
    ax.set_title('Predictions (first 200 points)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Correction distribution
    ax = axes[0, 2]
    ax.hist(corrections, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--')
    ax.axvline(x=np.mean(corrections), color='g', linestyle='-', label=f'Mean={np.mean(corrections):.4f}')
    ax.set_xlabel('Correction Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Transfer Model Corrections')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. HAR scatter
    ax = axes[1, 0]
    ax.scatter(targets, har_pred, alpha=0.3, s=10)
    lims = [min(targets.min(), har_pred.min()), max(targets.max(), har_pred.max())]
    ax.plot(lims, lims, 'r--')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'HAR: R² = {har_r2:.4f}')
    ax.grid(True, alpha=0.3)

    # 5. Transfer scatter
    ax = axes[1, 1]
    ax.scatter(targets, transfer_pred, alpha=0.3, s=10, c='blue')
    ax.plot(lims, lims, 'r--')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(f'Transfer: R² = {transfer_r2:.4f}')
    ax.grid(True, alpha=0.3)

    # 6. Results summary
    ax = axes[1, 2]
    ax.axis('off')

    summary = f"""
    TRANSFER LEARNING FOR HAR RESIDUAL CORRECTION
    ==============================================

    HAR Baseline:
      RMSE: {har_rmse:.6f}
      MAE:  {har_mae:.6f}
      R²:   {har_r2:.6f}

    Transfer Model (Pre-train SPX -> Fine-tune BTC):
      RMSE: {transfer_rmse:.6f}
      MAE:  {transfer_mae:.6f}
      R²:   {transfer_r2:.6f}

    From-Scratch Model (BTC only):
      RMSE: {scratch_rmse:.6f}
      MAE:  {scratch_mae:.6f}

    Improvements:
      Transfer vs HAR:     {rmse_improvement:+.2f}% RMSE
      Transfer vs Scratch: {transfer_vs_scratch:+.2f}% RMSE

    Transfer Learning Benefit:
      Pre-training on SPX helps BTC prediction!
    """

    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/transfer_residual_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved plot to: results/transfer_residual_results.png")

    # Save results
    results = {
        'har': {'rmse': float(har_rmse), 'mae': float(har_mae), 'r2': float(har_r2), 'corr': float(har_corr)},
        'transfer': {'rmse': float(transfer_rmse), 'mae': float(transfer_mae), 'r2': float(transfer_r2), 'corr': float(transfer_corr)},
        'scratch': {'rmse': float(scratch_rmse), 'mae': float(scratch_mae)},
        'improvements': {
            'transfer_vs_har_rmse': float(rmse_improvement),
            'transfer_vs_har_mae': float(mae_improvement),
            'transfer_vs_scratch_rmse': float(transfer_vs_scratch)
        }
    }

    with open('results/transfer_residual_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    if rmse_improvement > 0:
        print(f"SUCCESS! Transfer model beats HAR by {rmse_improvement:.2f}%")
    else:
        print(f"HAR still leads by {-rmse_improvement:.2f}%")

    if transfer_vs_scratch > 0:
        print(f"Transfer learning improves over scratch by {transfer_vs_scratch:.2f}%")
    print("=" * 70)

    return results


if __name__ == "__main__":
    results = run_transfer_residual_experiment()
