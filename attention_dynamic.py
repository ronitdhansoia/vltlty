"""
Attention-Based Dynamic Weighting for HAR + Neural Network

This model dynamically decides when to trust HAR vs the neural network
based on market context using an attention mechanism.

Key Innovation:
    - Instead of fixed weight: Final = 0.88*HAR + 0.12*NN
    - Dynamic weight: Final = α(context)*HAR + (1-α(context))*NN
    - α is computed by attention over the sequence and features

The model can learn to:
    - Trust HAR during stable/normal periods
    - Trust NN during volatile/unusual periods where nonlinearities matter

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
# EXTENDED FEATURES DATASET
# ============================================================================

class ExtendedHARDataset(Dataset):
    """
    Dataset with extended HAR features for dynamic weighting.

    Features include:
        - Standard HAR: daily, weekly, monthly averages
        - Volatility-of-volatility: rolling std of RV
        - Recent trend: slope of recent RV
        - Extreme indicator: how far from mean
    """

    def __init__(self,
                 log_rv: np.ndarray,
                 lookback: int = 20,
                 normalize: bool = True,
                 mean: Optional[float] = None,
                 std: Optional[float] = None):

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

        # Need 22 days for monthly HAR + some buffer
        self.start_idx = max(lookback, 22)
        self.valid_length = len(self.data) - self.start_idx - 1

    def __len__(self):
        return self.valid_length

    def __getitem__(self, idx: int):
        actual_idx = self.start_idx + idx

        # Sequence for LSTM
        X_seq = self.data[actual_idx - self.lookback : actual_idx]
        X_seq = torch.tensor(X_seq, dtype=torch.float32).unsqueeze(-1)

        # Extended HAR features (8 features)
        features = self._compute_features(actual_idx)
        X_features = torch.tensor(features, dtype=torch.float32)

        # Target
        y = torch.tensor(self.data[actual_idx], dtype=torch.float32)

        return X_seq, X_features, y

    def _compute_features(self, idx: int) -> np.ndarray:
        """Compute extended feature set."""
        data = self.data

        # Standard HAR features
        rv_daily = data[idx - 1]
        rv_weekly = np.mean(data[idx - 5 : idx])
        rv_monthly = np.mean(data[idx - 22 : idx])

        # Extended features
        # 1. Volatility of volatility (5-day rolling std)
        vol_of_vol = np.std(data[idx - 5 : idx])

        # 2. Recent trend (5-day slope approximation)
        recent = data[idx - 5 : idx]
        trend = (recent[-1] - recent[0]) / 5 if len(recent) >= 5 else 0

        # 3. Distance from mean (how unusual is current level)
        distance_from_mean = rv_daily - rv_monthly

        # 4. 10-day average (intermediate scale)
        rv_10day = np.mean(data[idx - 10 : idx])

        # 5. Acceleration (change in trend)
        if idx >= 10:
            prev_trend = (data[idx - 5] - data[idx - 10]) / 5
            acceleration = trend - prev_trend
        else:
            acceleration = 0

        return np.array([
            rv_daily,           # 0: Previous day
            rv_weekly,          # 1: 5-day average
            rv_10day,           # 2: 10-day average
            rv_monthly,         # 3: 22-day average
            vol_of_vol,         # 4: Volatility of volatility
            trend,              # 5: Recent trend
            distance_from_mean, # 6: Distance from mean
            acceleration        # 7: Acceleration
        ], dtype=np.float32)

    def get_norm_params(self):
        return self.mean, self.std


def create_extended_dataloaders(
    train_rv, val_rv, test_rv, lookback=20, batch_size=32
):
    train_ds = ExtendedHARDataset(train_rv, lookback=lookback, normalize=True)
    mean, std = train_ds.get_norm_params()

    val_ds = ExtendedHARDataset(val_rv, lookback=lookback, normalize=True, mean=mean, std=std)
    test_ds = ExtendedHARDataset(test_rv, lookback=lookback, normalize=True, mean=mean, std=std)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, {'mean': mean, 'std': std}


# ============================================================================
# ATTENTION-BASED DYNAMIC WEIGHTING MODEL
# ============================================================================

class DynamicAttentionGate(nn.Module):
    """
    Attention mechanism that computes dynamic weight between HAR and NN.

    Takes context (encoding + features) and outputs α ∈ (0, 1).
    """

    def __init__(self, encoding_dim: int, feature_dim: int, hidden_dim: int = 32):
        super().__init__()

        # Context processor
        self.context_net = nn.Sequential(
            nn.Linear(encoding_dim + feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Attention weights for HAR vs NN
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Output α ∈ (0, 1)
        )

    def forward(self, encoding: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Compute dynamic weight α.

        Args:
            encoding: LSTM encoding (batch, encoding_dim)
            features: Extended features (batch, feature_dim)

        Returns:
            α: Weight for HAR prediction (batch, 1)
        """
        context = torch.cat([encoding, features], dim=-1)
        hidden = self.context_net(context)
        alpha = self.gate(hidden)
        return alpha


class AttentionDynamicHybrid(nn.Module):
    """
    Hybrid model with attention-based dynamic weighting.

    Architecture:
        1. LSTM encoder processes sequence → encoding
        2. Extended HAR features processed
        3. Attention gate computes α(context)
        4. HAR predictor: features → HAR prediction
        5. NN predictor: encoding + features → NN prediction
        6. Final = α * HAR + (1-α) * NN
    """

    def __init__(self,
                 input_dim: int = 1,
                 hidden_dim: int = 64,
                 encoding_dim: int = 32,
                 feature_dim: int = 8,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()

        self.encoding_dim = encoding_dim
        self.feature_dim = feature_dim

        # LSTM Encoder
        self.encoder = RoughVolatilityEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            encoding_dim=encoding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )

        # Feature processor
        self.feature_net = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        # Dynamic attention gate
        self.attention_gate = DynamicAttentionGate(
            encoding_dim=encoding_dim,
            feature_dim=feature_dim,
            hidden_dim=32
        )

        # HAR predictor (linear, like classic HAR)
        # Uses first 4 features: daily, weekly, 10-day, monthly
        self.har_predictor = nn.Linear(4, 1)

        # NN predictor (nonlinear)
        self.nn_predictor = nn.Sequential(
            nn.Linear(encoding_dim + 16, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(32, 1)
        )

        # Initialize HAR weights close to optimal values
        self._init_har_weights()

    def _init_har_weights(self):
        """Initialize HAR weights based on typical optimal values."""
        # Typical HAR coefficients: daily ~0.3, weekly ~0.4, monthly ~0.3
        with torch.no_grad():
            self.har_predictor.weight.data = torch.tensor([[0.35, 0.35, 0.15, 0.15]])
            self.har_predictor.bias.data = torch.tensor([0.0])

    def forward(self, X_seq: torch.Tensor, X_features: torch.Tensor):
        """
        Forward pass with dynamic weighting.

        Args:
            X_seq: Sequence (batch, lookback, 1)
            X_features: Extended features (batch, 8)

        Returns:
            prediction: Final prediction (batch,)
            alpha: Dynamic weight used (batch, 1) - for analysis
        """
        # Encode sequence
        encoding = self.encoder(X_seq)  # (batch, encoding_dim)

        # Process features
        features_processed = self.feature_net(X_features)  # (batch, 16)

        # Compute dynamic weight
        alpha = self.attention_gate(encoding, X_features)  # (batch, 1)

        # HAR prediction (using first 4 features: daily, weekly, 10day, monthly)
        har_features = X_features[:, :4]  # (batch, 4)
        har_pred = self.har_predictor(har_features)  # (batch, 1)

        # NN prediction
        nn_input = torch.cat([encoding, features_processed], dim=-1)
        nn_pred = self.nn_predictor(nn_input)  # (batch, 1)

        # Dynamic combination
        final_pred = alpha * har_pred + (1 - alpha) * nn_pred

        return final_pred.squeeze(-1), alpha.squeeze(-1)

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

def train_epoch_dynamic(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_alpha = 0
    n_batches = 0

    for X_seq, X_features, y in train_loader:
        X_seq = X_seq.to(device)
        X_features = X_features.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred, alpha = model(X_seq, X_features)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_alpha += alpha.mean().item()
        n_batches += 1

    return total_loss / n_batches, total_alpha / n_batches


def validate_dynamic(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets, all_alphas = [], [], []

    with torch.no_grad():
        for X_seq, X_features, y in val_loader:
            X_seq = X_seq.to(device)
            X_features = X_features.to(device)
            y = y.to(device)

            pred, alpha = model(X_seq, X_features)
            loss = criterion(pred, y)

            total_loss += loss.item()
            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())
            all_alphas.append(alpha.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_alphas = torch.cat(all_alphas)

    val_rmse = torch.sqrt(torch.mean((all_preds - all_targets) ** 2)).item()
    avg_alpha = all_alphas.mean().item()

    return total_loss / len(val_loader), val_rmse, avg_alpha


def pretrain_dynamic(model, train_loader, val_loader, epochs=50, lr=0.001, patience=10, verbose=True):
    """Pre-train on S&P 500."""

    if verbose:
        print("\n" + "=" * 60)
        print("PRE-TRAINING DYNAMIC ATTENTION MODEL ON S&P 500")
        print("=" * 60)

    model = model.to(DEVICE)
    model.unfreeze_encoder()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=patience)

    history = {'train_loss': [], 'val_loss': [], 'val_rmse': [], 'avg_alpha': []}
    best_val_loss = float('inf')
    best_state = None

    if verbose:
        print(f"Trainable params: {model.get_trainable_params():,}")

    pbar = tqdm(range(epochs), desc="Pre-training", disable=not verbose)
    for epoch in pbar:
        train_loss, train_alpha = train_epoch_dynamic(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_rmse, avg_alpha = validate_dynamic(model, val_loader, criterion, DEVICE)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        history['avg_alpha'].append(avg_alpha)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        pbar.set_postfix({
            'train': f'{train_loss:.4f}',
            'val': f'{val_loss:.4f}',
            'rmse': f'{val_rmse:.4f}',
            'α': f'{avg_alpha:.2f}'
        })

        if early_stopping(val_loss):
            if verbose:
                print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_state)

    if verbose:
        print(f"✅ Best validation RMSE: {min(history['val_rmse']):.4f}")

    return model, history


def finetune_dynamic(model, train_loader, val_loader, epochs=60, encoder_lr=0.00001,
                     other_lr=0.0005, patience=15, verbose=True):
    """Fine-tune on Bitcoin with differential learning rates."""

    if verbose:
        print("\n" + "=" * 60)
        print("FINE-TUNING DYNAMIC ATTENTION MODEL ON BITCOIN")
        print("=" * 60)

    model = model.to(DEVICE)
    model.unfreeze_encoder()

    # Differential learning rates
    optimizer = optim.Adam([
        {'params': model.encoder.parameters(), 'lr': encoder_lr},
        {'params': model.feature_net.parameters(), 'lr': other_lr},
        {'params': model.attention_gate.parameters(), 'lr': other_lr},
        {'params': model.har_predictor.parameters(), 'lr': other_lr * 0.5},  # Lower for HAR
        {'params': model.nn_predictor.parameters(), 'lr': other_lr}
    ])

    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)
    early_stopping = EarlyStopping(patience=patience)

    history = {'train_loss': [], 'val_loss': [], 'val_rmse': [], 'avg_alpha': []}
    best_val_loss = float('inf')
    best_state = None

    if verbose:
        print(f"Encoder LR: {encoder_lr}, Other LR: {other_lr}")
        print(f"Trainable params: {model.get_trainable_params():,}")

    pbar = tqdm(range(epochs), desc="Fine-tuning", disable=not verbose)
    for epoch in pbar:
        train_loss, train_alpha = train_epoch_dynamic(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_rmse, avg_alpha = validate_dynamic(model, val_loader, criterion, DEVICE)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        history['avg_alpha'].append(avg_alpha)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        pbar.set_postfix({
            'train': f'{train_loss:.4f}',
            'val': f'{val_loss:.4f}',
            'rmse': f'{val_rmse:.4f}',
            'α': f'{avg_alpha:.2f}'
        })

        if early_stopping(val_loss):
            if verbose:
                print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_state)

    if verbose:
        print(f"✅ Best validation RMSE: {min(history['val_rmse']):.4f}")
        print(f"   Final avg α (HAR weight): {history['avg_alpha'][-1]:.3f}")

    return model, history


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_dynamic_attention_experiment():
    """Run the attention-based dynamic weighting experiment."""

    print("\n" + "=" * 80)
    print("ATTENTION-BASED DYNAMIC WEIGHTING EXPERIMENT")
    print("=" * 80)

    # Load data
    print("\n📂 Loading data...")
    spx_train, spx_val, spx_test = load_data_splits('data', 'spx')
    btc_train, btc_val, btc_test = load_data_splits('data', 'btc')

    # Create extended dataloaders
    print("\n📦 Creating extended feature dataloaders...")

    spx_train_loader, spx_val_loader, _, spx_norm = create_extended_dataloaders(
        spx_train, spx_val, spx_test, lookback=20, batch_size=32
    )

    btc_train_loader, btc_val_loader, btc_test_loader, btc_norm = create_extended_dataloaders(
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

    # Create dynamic attention model
    print("\n🏗️ Creating dynamic attention model...")
    model = AttentionDynamicHybrid(
        input_dim=1,
        hidden_dim=64,
        encoding_dim=32,
        feature_dim=8,
        num_layers=3,
        num_heads=4,
        dropout=0.1
    )
    print(f"   Total params: {model.get_total_params():,}")

    # Pre-train on S&P 500
    model, pretrain_history = pretrain_dynamic(
        model, spx_train_loader, spx_val_loader,
        epochs=50, lr=0.001, patience=10
    )

    # Fine-tune on Bitcoin
    model, finetune_history = finetune_dynamic(
        model, btc_train_loader, btc_val_loader,
        epochs=70, encoder_lr=0.00001, other_lr=0.0005, patience=15
    )

    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    model.eval()
    all_preds, all_targets, all_alphas = [], [], []

    with torch.no_grad():
        for X_seq, X_features, y in btc_test_loader:
            X_seq = X_seq.to(DEVICE)
            X_features = X_features.to(DEVICE)
            pred, alpha = model(X_seq, X_features)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.numpy())
            all_alphas.append(alpha.cpu().numpy())

    dynamic_preds = np.concatenate(all_preds) * btc_norm['std'] + btc_norm['mean']
    dynamic_targets = np.concatenate(all_targets) * btc_norm['std'] + btc_norm['mean']
    dynamic_alphas = np.concatenate(all_alphas)

    # Align lengths
    min_len = min(len(dynamic_targets), len(har_targets))
    dynamic_rmse_val = rmse(har_targets[:min_len], dynamic_preds[:min_len])

    # Results
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    print(f"\n{'Model':<35} {'RMSE':<12} {'vs HAR':<15}")
    print("-" * 65)
    print(f"{'HAR Baseline':<35} {har_rmse_val:<12.4f} {'---':<15}")
    print(f"{'Dynamic Attention (HAR + LSTM)':<35} {dynamic_rmse_val:<12.4f} {((har_rmse_val - dynamic_rmse_val) / har_rmse_val * 100):+.2f}%")
    print("-" * 65)

    if dynamic_rmse_val < har_rmse_val:
        improvement = (har_rmse_val - dynamic_rmse_val) / har_rmse_val * 100
        print(f"\n🎉 SUCCESS! Dynamic model beats HAR by {improvement:.2f}%")
    else:
        gap = (dynamic_rmse_val - har_rmse_val) / har_rmse_val * 100
        print(f"\n⚠️ HAR still wins by {gap:.2f}%")

    # Analyze dynamic weights
    print(f"\n📊 Dynamic Weight Analysis (α = HAR weight):")
    print(f"   Mean α: {dynamic_alphas.mean():.3f}")
    print(f"   Std α:  {dynamic_alphas.std():.3f}")
    print(f"   Min α:  {dynamic_alphas.min():.3f}")
    print(f"   Max α:  {dynamic_alphas.max():.3f}")

    # Save comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    n_plot = min(200, len(har_targets))
    x = np.arange(n_plot)

    # HAR predictions
    ax = axes[0, 0]
    ax.plot(x, har_targets[:n_plot], 'k-', label='Actual', linewidth=1)
    ax.plot(x, har_preds[:n_plot], 'r-', label='HAR', linewidth=1, alpha=0.8)
    ax.fill_between(x, har_targets[:n_plot], har_preds[:n_plot], alpha=0.3, color='red')
    ax.set_title(f'HAR Model (RMSE: {har_rmse_val:.4f})', fontweight='bold')
    ax.set_ylabel('Log RV')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Dynamic model predictions
    ax = axes[0, 1]
    ax.plot(x, har_targets[:n_plot], 'k-', label='Actual', linewidth=1)
    ax.plot(x, dynamic_preds[:n_plot], 'g-', label='Dynamic', linewidth=1, alpha=0.8)
    ax.fill_between(x, har_targets[:n_plot], dynamic_preds[:n_plot], alpha=0.3, color='green')
    ax.set_title(f'Dynamic Attention (RMSE: {dynamic_rmse_val:.4f})', fontweight='bold')
    ax.set_ylabel('Log RV')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Dynamic weights over time
    ax = axes[1, 0]
    ax.plot(x, dynamic_alphas[:n_plot], 'b-', linewidth=1)
    ax.axhline(y=dynamic_alphas.mean(), color='r', linestyle='--', label=f'Mean: {dynamic_alphas.mean():.3f}')
    ax.fill_between(x, 0, dynamic_alphas[:n_plot], alpha=0.3, color='blue')
    ax.set_title('Dynamic α (HAR Weight) Over Time', fontweight='bold')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('α (HAR weight)')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Alpha distribution
    ax = axes[1, 1]
    ax.hist(dynamic_alphas, bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax.axvline(x=dynamic_alphas.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {dynamic_alphas.mean():.3f}')
    ax.set_title('Distribution of α (HAR Weight)', fontweight='bold')
    ax.set_xlabel('α')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/dynamic_attention_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✓ Saved comparison to: results/dynamic_attention_comparison.png")

    return {
        'har_rmse': har_rmse_val,
        'dynamic_rmse': dynamic_rmse_val,
        'alphas': dynamic_alphas,
        'har_preds': har_preds,
        'dynamic_preds': dynamic_preds,
        'targets': har_targets
    }


if __name__ == "__main__":
    results = run_dynamic_attention_experiment()
