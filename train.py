"""
Training Pipeline for Rough Volatility Transfer Learning

This module implements the two-phase training protocol:
    1. Pre-training on source domain (S&P 500)
    2. Fine-tuning on target domain (Bitcoin)

Features:
    - Early stopping with patience
    - Learning rate scheduling (ReduceLROnPlateau)
    - Progress tracking with tqdm
    - Model checkpointing
    - Training history logging

Usage:
    from train import pretrain_on_source, finetune_on_target

    # Pre-train on S&P 500
    model, history = pretrain_on_source(model, spx_train_loader, spx_val_loader)

    # Fine-tune on Bitcoin
    model, history = finetune_on_target(model, btc_train_loader, btc_val_loader)

Author: Ronit Dhansoia
Date: 22nd December 2025
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime

# Import our modules
from models import TransferRoughVolModel
from datasets import create_dataloaders, load_data_splits


# ============================================================================
# CONFIGURATION
# ============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELS_DIR = 'models'
RESULTS_DIR = 'results'


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Stops training when validation loss stops improving.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-6, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for metrics like accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current validation metric

        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """
    Train for one epoch.

    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use

    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y)
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Validate the model.

    Args:
        model: PyTorch model
        val_loader: Validation DataLoader
        criterion: Loss function
        device: Device to use

    Returns:
        Average validation loss, RMSE
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            predictions = model(X)
            loss = criterion(predictions, y)

            total_loss += loss.item()
            all_predictions.append(predictions.cpu())
            all_targets.append(y.cpu())

    avg_loss = total_loss / len(val_loader)

    # Compute RMSE
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    rmse = torch.sqrt(torch.mean((all_predictions - all_targets) ** 2)).item()

    return avg_loss, rmse


# ============================================================================
# PRE-TRAINING ON SOURCE DOMAIN
# ============================================================================

def pretrain_on_source(
    model: TransferRoughVolModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    learning_rate: float = 0.001,
    patience: int = 10,
    save_path: Optional[str] = None,
    verbose: bool = True
) -> Tuple[TransferRoughVolModel, Dict]:
    """
    Pre-train model on source domain (S&P 500).

    This phase trains the entire model (encoder + predictor) on
    the data-rich source domain to learn universal volatility patterns.

    Args:
        model: TransferRoughVolModel instance
        train_loader: Source domain training DataLoader
        val_loader: Source domain validation DataLoader
        epochs: Maximum number of epochs (default: 50)
        learning_rate: Initial learning rate (default: 0.001)
        patience: Early stopping patience (default: 10)
        save_path: Path to save best model (optional)
        verbose: Print progress (default: True)

    Returns:
        model: Trained model
        history: Dictionary with training history
    """
    if verbose:
        print("\n" + "=" * 60)
        print("PHASE 1: PRE-TRAINING ON SOURCE DOMAIN (S&P 500)")
        print("=" * 60)

    model = model.to(DEVICE)

    # Ensure all parameters are trainable
    model.unfreeze_encoder()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    early_stopping = EarlyStopping(patience=patience)

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_rmse': [],
        'lr': []
    }

    best_val_loss = float('inf')
    best_state = None

    if verbose:
        print(f"\nTraining Configuration:")
        print(f"  Device: {DEVICE}")
        print(f"  Epochs: {epochs}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Batch size: {train_loader.batch_size}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Trainable params: {model.get_trainable_params():,}")
        print()

    # Training loop
    pbar = tqdm(range(epochs), desc="Pre-training", disable=not verbose)
    for epoch in pbar:
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)

        # Validate
        val_loss, val_rmse = validate(model, val_loader, criterion, DEVICE)

        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        history['lr'].append(current_lr)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()

        # Update progress bar
        pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'val_rmse': f'{val_rmse:.4f}',
            'lr': f'{current_lr:.6f}'
        })

        # Early stopping check
        if early_stopping(val_loss):
            if verbose:
                print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    # Load best model
    model.load_state_dict(best_state)

    # Save model
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'model_state_dict': best_state,
            'history': history,
            'best_val_loss': best_val_loss,
            'phase': 'pretrain'
        }, save_path)
        if verbose:
            print(f"\n✓ Saved pre-trained model to: {save_path}")

    if verbose:
        print(f"\n✅ Pre-training complete!")
        print(f"   Best validation loss: {best_val_loss:.4f}")
        print(f"   Best validation RMSE: {min(history['val_rmse']):.4f}")

    return model, history


# ============================================================================
# FINE-TUNING ON TARGET DOMAIN
# ============================================================================

def finetune_on_target(
    model: TransferRoughVolModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    learning_rate: float = 0.0001,
    patience: int = 10,
    freeze_encoder: bool = True,
    save_path: Optional[str] = None,
    verbose: bool = True
) -> Tuple[TransferRoughVolModel, Dict]:
    """
    Fine-tune model on target domain (Bitcoin).

    This phase adapts the model to the target domain, typically with:
    - Frozen encoder (transfer learned features)
    - Lower learning rate
    - Fewer epochs

    Args:
        model: Pre-trained TransferRoughVolModel
        train_loader: Target domain training DataLoader
        val_loader: Target domain validation DataLoader
        epochs: Maximum number of epochs (default: 30)
        learning_rate: Initial learning rate (default: 0.0001, lower than pre-train)
        patience: Early stopping patience (default: 10)
        freeze_encoder: Whether to freeze encoder (default: True)
        save_path: Path to save best model (optional)
        verbose: Print progress (default: True)

    Returns:
        model: Fine-tuned model
        history: Dictionary with training history
    """
    if verbose:
        print("\n" + "=" * 60)
        print("PHASE 2: FINE-TUNING ON TARGET DOMAIN (Bitcoin)")
        print("=" * 60)

    model = model.to(DEVICE)

    # Freeze or unfreeze encoder
    if freeze_encoder:
        model.freeze_encoder()
    else:
        model.unfreeze_encoder()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    early_stopping = EarlyStopping(patience=patience)

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_rmse': [],
        'lr': []
    }

    best_val_loss = float('inf')
    best_state = None

    if verbose:
        print(f"\nFine-tuning Configuration:")
        print(f"  Device: {DEVICE}")
        print(f"  Epochs: {epochs}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Encoder frozen: {freeze_encoder}")
        print(f"  Batch size: {train_loader.batch_size}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Trainable params: {model.get_trainable_params():,}")
        print()

    # Training loop
    pbar = tqdm(range(epochs), desc="Fine-tuning", disable=not verbose)
    for epoch in pbar:
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)

        # Validate
        val_loss, val_rmse = validate(model, val_loader, criterion, DEVICE)

        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        history['lr'].append(current_lr)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()

        # Update progress bar
        pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'val_rmse': f'{val_rmse:.4f}',
            'lr': f'{current_lr:.6f}'
        })

        # Early stopping check
        if early_stopping(val_loss):
            if verbose:
                print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    # Load best model
    model.load_state_dict(best_state)

    # Save model
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'model_state_dict': best_state,
            'history': history,
            'best_val_loss': best_val_loss,
            'phase': 'finetune',
            'encoder_frozen': freeze_encoder
        }, save_path)
        if verbose:
            print(f"\n✓ Saved fine-tuned model to: {save_path}")

    if verbose:
        print(f"\n✅ Fine-tuning complete!")
        print(f"   Best validation loss: {best_val_loss:.4f}")
        print(f"   Best validation RMSE: {min(history['val_rmse']):.4f}")

    return model, history


# ============================================================================
# MAIN EXECUTION (FOR TESTING)
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING TRAINING PIPELINE")
    print("=" * 60)

    # Create model directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load data
    print("\n📂 Loading data...")

    spx_train, spx_val, spx_test = load_data_splits('data', 'spx')
    btc_train, btc_val, btc_test = load_data_splits('data', 'btc')

    # Create dataloaders
    spx_train_loader, spx_val_loader, spx_test_loader, spx_norm = create_dataloaders(
        spx_train, spx_val, spx_test,
        lookback=20, horizon=1, batch_size=32
    )

    btc_train_loader, btc_val_loader, btc_test_loader, btc_norm = create_dataloaders(
        btc_train, btc_val, btc_test,
        lookback=20, horizon=1, batch_size=16
    )

    print(f"  S&P 500: {len(spx_train_loader)} train batches, {len(spx_val_loader)} val batches")
    print(f"  Bitcoin: {len(btc_train_loader)} train batches, {len(btc_val_loader)} val batches")

    # Create model
    print("\n🏗️ Creating model...")
    model = TransferRoughVolModel(
        input_dim=1,
        hidden_dim=64,
        encoding_dim=32,
        num_layers=3,
        num_heads=4,
        forecast_horizon=1,
        dropout=0.1
    )
    print(f"  Total parameters: {model.get_total_params():,}")

    # Phase 1: Pre-train on S&P 500
    model, pretrain_history = pretrain_on_source(
        model, spx_train_loader, spx_val_loader,
        epochs=50,
        learning_rate=0.001,
        patience=10,
        save_path=os.path.join(MODELS_DIR, 'pretrained_spx.pth')
    )

    # Phase 2: Fine-tune on Bitcoin
    model, finetune_history = finetune_on_target(
        model, btc_train_loader, btc_val_loader,
        epochs=30,
        learning_rate=0.0001,
        patience=10,
        freeze_encoder=True,
        save_path=os.path.join(MODELS_DIR, 'finetuned_btc.pth')
    )

    # Save training histories
    histories = {
        'pretrain': pretrain_history,
        'finetune': finetune_history
    }
    with open(os.path.join(RESULTS_DIR, 'training_history.json'), 'w') as f:
        json.dump(histories, f, indent=2)

    print("\n" + "=" * 60)
    print("✅ Training pipeline test complete!")
    print("=" * 60)
