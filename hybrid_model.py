"""
HAR-Augmented Neural Network for Rough Volatility Forecasting

This module implements a hybrid approach that combines:
    1. HAR features (daily, weekly, monthly averages) - proven effective
    2. LSTM encoder for learning additional patterns from sequence
    3. Transfer learning from S&P 500

The idea: Let the neural network learn to IMPROVE upon HAR, not replace it.

Author: Ronit Dhansoia
Date: 23rd December 2025
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from models import RoughVolatilityEncoder
from datasets import load_data_splits
from train import EarlyStopping
from evaluation import rmse

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# HAR-AUGMENTED DATASET
# ============================================================================

class HARDataset(Dataset):
    """
    Dataset that provides both sequence data AND HAR features.

    For each sample:
        - X_seq: Sequence of past log RV values (for LSTM)
        - X_har: HAR features [RV_daily, RV_weekly, RV_monthly]
        - y: Target (next log RV)
    """

    def __init__(self,
                 log_rv: np.ndarray,
                 lookback: int = 20,
                 normalize: bool = True,
                 mean: Optional[float] = None,
                 std: Optional[float] = None):
        """
        Initialize HAR Dataset.

        Args:
            log_rv: Log realized volatility series
            lookback: Lookback window for sequence
            normalize: Whether to normalize
            mean: Pre-computed mean (use training set for val/test)
            std: Pre-computed std
        """
        self.raw_data = log_rv.astype(np.float32)
        self.lookback = lookback

        # Normalization
        if normalize:
            self.mean = mean if mean is not None else np.mean(self.raw_data)
            self.std = std if std is not None else np.std(self.raw_data)
            self.data = (self.raw_data - self.mean) / (self.std + 1e-8)
        else:
            self.mean = 0.0
            self.std = 1.0
            self.data = self.raw_data

        # Need at least 22 days for monthly HAR feature + lookback
        self.start_idx = max(lookback, 22)
        self.valid_length = len(self.data) - self.start_idx - 1

        if self.valid_length <= 0:
            raise ValueError("Data too short for HAR features + lookback")

    def __len__(self):
        return self.valid_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get sample with sequence and HAR features.

        Returns:
            X_seq: Sequence (lookback, 1)
            X_har: HAR features (3,) - [daily, weekly, monthly]
            y: Target scalar
        """
        actual_idx = self.start_idx + idx

        # Sequence for LSTM
        X_seq = self.data[actual_idx - self.lookback : actual_idx]
        X_seq = torch.tensor(X_seq, dtype=torch.float32).unsqueeze(-1)

        # HAR features (using normalized data)
        rv_daily = self.data[actual_idx - 1]  # Previous day
        rv_weekly = np.mean(self.data[actual_idx - 5 : actual_idx])  # 5-day avg
        rv_monthly = np.mean(self.data[actual_idx - 22 : actual_idx])  # 22-day avg

        X_har = torch.tensor([rv_daily, rv_weekly, rv_monthly], dtype=torch.float32)

        # Target
        y = torch.tensor(self.data[actual_idx], dtype=torch.float32)

        return X_seq, X_har, y

    def get_norm_params(self):
        return self.mean, self.std


def create_har_dataloaders(
    train_rv: np.ndarray,
    val_rv: np.ndarray,
    test_rv: np.ndarray,
    lookback: int = 20,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """Create DataLoaders for HAR-augmented dataset."""

    train_dataset = HARDataset(train_rv, lookback=lookback, normalize=True)
    mean, std = train_dataset.get_norm_params()

    val_dataset = HARDataset(val_rv, lookback=lookback, normalize=True, mean=mean, std=std)
    test_dataset = HARDataset(test_rv, lookback=lookback, normalize=True, mean=mean, std=std)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, {'mean': mean, 'std': std}


# ============================================================================
# HYBRID MODEL: HAR + LSTM
# ============================================================================

class HARLSTMHybrid(nn.Module):
    """
    Hybrid model combining HAR features with LSTM encoder.

    Architecture:
        - LSTM encoder processes sequence → encoding (32 dim)
        - HAR features (3 dim) concatenated with encoding
        - MLP predictor: (32 + 3) → 64 → 32 → 1

    The model learns to use LSTM features to REFINE HAR predictions.
    """

    def __init__(self,
                 input_dim: int = 1,
                 hidden_dim: int = 64,
                 encoding_dim: int = 32,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        # LSTM Encoder (same as before)
        self.encoder = RoughVolatilityEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            encoding_dim=encoding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )

        # HAR feature processing
        self.har_fc = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Combined predictor: encoding (32) + processed HAR (16) = 48
        self.predictor = nn.Sequential(
            nn.Linear(encoding_dim + 16, 64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(32, 1)
        )

        # Residual connection weight (learnable)
        # Output = HAR_prediction + alpha * neural_refinement
        self.har_weight = nn.Parameter(torch.tensor(0.8))  # Start with strong HAR bias

        # Simple HAR linear predictor (for residual)
        self.har_linear = nn.Linear(3, 1)

    def forward(self, X_seq: torch.Tensor, X_har: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            X_seq: Sequence input (batch, lookback, 1)
            X_har: HAR features (batch, 3)

        Returns:
            Prediction (batch,)
        """
        # LSTM encoding
        encoding = self.encoder(X_seq)  # (batch, encoding_dim)

        # Process HAR features
        har_processed = self.har_fc(X_har)  # (batch, 16)

        # Combine
        combined = torch.cat([encoding, har_processed], dim=-1)  # (batch, 48)

        # Neural network prediction
        nn_pred = self.predictor(combined).squeeze(-1)  # (batch,)

        # HAR baseline prediction
        har_pred = self.har_linear(X_har).squeeze(-1)  # (batch,)

        # Weighted combination (residual learning)
        # The NN learns to correct HAR's mistakes
        output = self.har_weight * har_pred + (1 - self.har_weight) * nn_pred

        return output

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("✓ Encoder frozen")

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("✓ Encoder unfrozen")

    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch_har(model, train_loader, criterion, optimizer, device):
    """Train one epoch for HAR hybrid model."""
    model.train()
    total_loss = 0

    for X_seq, X_har, y in train_loader:
        X_seq = X_seq.to(device)
        X_har = X_har.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(X_seq, X_har)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate_har(model, val_loader, criterion, device):
    """Validate HAR hybrid model."""
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_seq, X_har, y in val_loader:
            X_seq = X_seq.to(device)
            X_har = X_har.to(device)
            y = y.to(device)

            pred = model(X_seq, X_har)
            loss = criterion(pred, y)

            total_loss += loss.item()
            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    val_rmse = torch.sqrt(torch.mean((all_preds - all_targets) ** 2)).item()

    return total_loss / len(val_loader), val_rmse


def pretrain_hybrid_on_source(
    model: HARLSTMHybrid,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 0.001,
    patience: int = 10,
    verbose: bool = True
) -> Tuple[HARLSTMHybrid, Dict]:
    """Pre-train hybrid model on S&P 500."""

    if verbose:
        print("\n" + "=" * 60)
        print("PRE-TRAINING HYBRID MODEL ON S&P 500")
        print("=" * 60)

    model = model.to(DEVICE)
    model.unfreeze_encoder()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=patience)

    history = {'train_loss': [], 'val_loss': [], 'val_rmse': []}
    best_val_loss = float('inf')
    best_state = None

    if verbose:
        print(f"Trainable params: {model.get_trainable_params():,}")

    pbar = tqdm(range(epochs), desc="Pre-training", disable=not verbose)
    for epoch in pbar:
        train_loss = train_epoch_har(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_rmse = validate_har(model, val_loader, criterion, DEVICE)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        pbar.set_postfix({'train': f'{train_loss:.4f}', 'val': f'{val_loss:.4f}', 'rmse': f'{val_rmse:.4f}'})

        if early_stopping(val_loss):
            if verbose:
                print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_state)

    if verbose:
        print(f"✅ Best validation RMSE: {min(history['val_rmse']):.4f}")

    return model, history


def finetune_hybrid_on_target(
    model: HARLSTMHybrid,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    encoder_lr: float = 0.00001,
    other_lr: float = 0.0005,
    patience: int = 15,
    verbose: bool = True
) -> Tuple[HARLSTMHybrid, Dict]:
    """Fine-tune hybrid model on Bitcoin with differential learning rates."""

    if verbose:
        print("\n" + "=" * 60)
        print("FINE-TUNING HYBRID MODEL ON BITCOIN")
        print("=" * 60)

    model = model.to(DEVICE)
    model.unfreeze_encoder()

    # Differential learning rates
    optimizer = optim.Adam([
        {'params': model.encoder.parameters(), 'lr': encoder_lr},
        {'params': model.har_fc.parameters(), 'lr': other_lr},
        {'params': model.predictor.parameters(), 'lr': other_lr},
        {'params': model.har_linear.parameters(), 'lr': other_lr},
        {'params': [model.har_weight], 'lr': other_lr}
    ])

    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)
    early_stopping = EarlyStopping(patience=patience)

    history = {'train_loss': [], 'val_loss': [], 'val_rmse': [], 'har_weight': []}
    best_val_loss = float('inf')
    best_state = None

    if verbose:
        print(f"Encoder LR: {encoder_lr}, Other LR: {other_lr}")
        print(f"Trainable params: {model.get_trainable_params():,}")

    pbar = tqdm(range(epochs), desc="Fine-tuning", disable=not verbose)
    for epoch in pbar:
        train_loss = train_epoch_har(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_rmse = validate_har(model, val_loader, criterion, DEVICE)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        history['har_weight'].append(model.har_weight.item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        pbar.set_postfix({
            'train': f'{train_loss:.4f}',
            'val': f'{val_loss:.4f}',
            'rmse': f'{val_rmse:.4f}',
            'har_w': f'{model.har_weight.item():.2f}'
        })

        if early_stopping(val_loss):
            if verbose:
                print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_state)

    if verbose:
        print(f"✅ Best validation RMSE: {min(history['val_rmse']):.4f}")
        print(f"   Final HAR weight: {model.har_weight.item():.3f}")

    return model, history


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_hybrid_experiment():
    """Run the HAR-augmented hybrid model experiment."""

    print("\n" + "=" * 80)
    print("HAR-AUGMENTED HYBRID MODEL EXPERIMENT")
    print("=" * 80)

    # Load data
    print("\n📂 Loading data...")
    spx_train, spx_val, spx_test = load_data_splits('data', 'spx')
    btc_train, btc_val, btc_test = load_data_splits('data', 'btc')

    # Create HAR-augmented dataloaders
    print("\n📦 Creating HAR-augmented dataloaders...")

    spx_train_loader, spx_val_loader, _, spx_norm = create_har_dataloaders(
        spx_train, spx_val, spx_test, lookback=20, batch_size=32
    )

    btc_train_loader, btc_val_loader, btc_test_loader, btc_norm = create_har_dataloaders(
        btc_train, btc_val, btc_test, lookback=20, batch_size=16
    )

    # Compute HAR baseline
    print("\n📊 Computing HAR baseline...")
    from baselines import HARModel
    full_train = np.concatenate([btc_train, btc_val])
    har_model = HARModel()
    har_model.fit(full_train)
    har_preds = har_model.predict(btc_test)
    har_targets = har_model.get_targets(btc_test)
    har_rmse_val = rmse(har_targets, har_preds)
    print(f"   HAR RMSE: {har_rmse_val:.4f} (TARGET TO BEAT)")

    # Create hybrid model
    print("\n🏗️ Creating hybrid model...")
    model = HARLSTMHybrid(
        input_dim=1,
        hidden_dim=64,
        encoding_dim=32,
        num_layers=3,
        num_heads=4,
        dropout=0.1
    )
    print(f"   Total params: {model.get_total_params():,}")

    # Pre-train on S&P 500
    model, pretrain_history = pretrain_hybrid_on_source(
        model, spx_train_loader, spx_val_loader,
        epochs=50, lr=0.001, patience=10
    )

    # Fine-tune on Bitcoin
    model, finetune_history = finetune_hybrid_on_target(
        model, btc_train_loader, btc_val_loader,
        epochs=60, encoder_lr=0.00001, other_lr=0.0005, patience=15
    )

    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_seq, X_har, y in btc_test_loader:
            X_seq = X_seq.to(DEVICE)
            X_har = X_har.to(DEVICE)
            pred = model(X_seq, X_har)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.numpy())

    hybrid_preds = np.concatenate(all_preds) * btc_norm['std'] + btc_norm['mean']
    hybrid_targets = np.concatenate(all_targets) * btc_norm['std'] + btc_norm['mean']

    # Align lengths
    min_len = min(len(hybrid_targets), len(har_targets))
    hybrid_rmse_val = rmse(har_targets[:min_len], hybrid_preds[:min_len])

    # Results
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    print(f"\n{'Model':<30} {'RMSE':<12} {'vs HAR':<15}")
    print("-" * 60)
    print(f"{'HAR Baseline':<30} {har_rmse_val:<12.4f} {'---':<15}")
    print(f"{'Hybrid (HAR + LSTM)':<30} {hybrid_rmse_val:<12.4f} {((har_rmse_val - hybrid_rmse_val) / har_rmse_val * 100):+.2f}%")
    print("-" * 60)

    if hybrid_rmse_val < har_rmse_val:
        print(f"\n🎉 SUCCESS! Hybrid beats HAR by {((har_rmse_val - hybrid_rmse_val) / har_rmse_val * 100):.2f}%")
    else:
        print(f"\n⚠️ HAR still wins by {((hybrid_rmse_val - har_rmse_val) / har_rmse_val * 100):.2f}%")

    print(f"\n📊 Final HAR weight in model: {model.har_weight.item():.3f}")
    print(f"   (1.0 = pure HAR, 0.0 = pure neural network)")

    # Save comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    n_plot = min(200, len(har_targets))
    x = np.arange(n_plot)

    # HAR
    ax = axes[0]
    ax.plot(x, har_targets[:n_plot], 'k-', label='Actual', linewidth=1)
    ax.plot(x, har_preds[:n_plot], 'r-', label='HAR', linewidth=1, alpha=0.8)
    ax.fill_between(x, har_targets[:n_plot], har_preds[:n_plot], alpha=0.3, color='red')
    ax.set_title(f'HAR Model (RMSE: {har_rmse_val:.4f})', fontweight='bold')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Log RV')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Hybrid
    ax = axes[1]
    ax.plot(x, har_targets[:n_plot], 'k-', label='Actual', linewidth=1)
    ax.plot(x, hybrid_preds[:n_plot], 'g-', label='Hybrid', linewidth=1, alpha=0.8)
    ax.fill_between(x, har_targets[:n_plot], hybrid_preds[:n_plot], alpha=0.3, color='green')
    ax.set_title(f'Hybrid HAR+LSTM (RMSE: {hybrid_rmse_val:.4f})', fontweight='bold')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Log RV')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/hybrid_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✓ Saved comparison to: results/hybrid_comparison.png")

    return {
        'har_rmse': har_rmse_val,
        'hybrid_rmse': hybrid_rmse_val,
        'har_weight': model.har_weight.item(),
        'har_preds': har_preds,
        'hybrid_preds': hybrid_preds,
        'targets': har_targets
    }


if __name__ == "__main__":
    results = run_hybrid_experiment()
