"""
Improved Experiment: Unfrozen Encoder with Differential Learning Rates

This script attempts to beat HAR by:
    1. Unfreezing encoder during fine-tuning
    2. Using differential learning rates (lower for encoder, higher for predictor)
    3. Training for more epochs with patience
    4. Testing multiple configurations

Author: Ronit Dhansoia
Date: 22nd December 2025
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from models import TransferRoughVolModel
from datasets import create_dataloaders, load_data_splits
from train import pretrain_on_source, EarlyStopping, train_epoch, validate
from baselines import HARModel
from evaluation import compute_all_metrics, rmse, mae

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def finetune_unfrozen(
    model: TransferRoughVolModel,
    train_loader,
    val_loader,
    epochs: int = 50,
    encoder_lr: float = 0.00001,  # Very low LR for encoder
    predictor_lr: float = 0.0005,  # Higher LR for predictor
    patience: int = 15,
    verbose: bool = True
):
    """
    Fine-tune with unfrozen encoder using differential learning rates.

    Key insight: Use much lower learning rate for encoder to preserve
    learned representations while allowing adaptation.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("FINE-TUNING WITH UNFROZEN ENCODER")
        print("=" * 60)

    model = model.to(DEVICE)
    model.unfreeze_encoder()

    # Differential learning rates
    optimizer = optim.Adam([
        {'params': model.encoder.parameters(), 'lr': encoder_lr},
        {'params': model.predictor.parameters(), 'lr': predictor_lr}
    ])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7
    )

    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=patience)

    history = {'train_loss': [], 'val_loss': [], 'val_rmse': []}
    best_val_loss = float('inf')
    best_state = None

    if verbose:
        print(f"\nConfiguration:")
        print(f"  Encoder LR: {encoder_lr}")
        print(f"  Predictor LR: {predictor_lr}")
        print(f"  Epochs: {epochs}")
        print(f"  Trainable params: {model.get_trainable_params():,}")

    pbar = tqdm(range(epochs), desc="Fine-tuning (unfrozen)", disable=not verbose)
    for epoch in pbar:
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_rmse = validate(model, val_loader, criterion, DEVICE)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        pbar.set_postfix({
            'train': f'{train_loss:.4f}',
            'val': f'{val_loss:.4f}',
            'rmse': f'{val_rmse:.4f}'
        })

        if early_stopping(val_loss):
            if verbose:
                print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(best_state)

    if verbose:
        print(f"\n✅ Best validation RMSE: {min(history['val_rmse']):.4f}")

    return model, history


def finetune_gradual_unfreeze(
    model: TransferRoughVolModel,
    train_loader,
    val_loader,
    verbose: bool = True
):
    """
    Gradual unfreezing strategy:
    1. First train only predictor (10 epochs)
    2. Then unfreeze last LSTM layer (10 epochs)
    3. Finally unfreeze all (20 epochs)
    """
    if verbose:
        print("\n" + "=" * 60)
        print("GRADUAL UNFREEZING STRATEGY")
        print("=" * 60)

    model = model.to(DEVICE)
    criterion = nn.MSELoss()

    history = {'train_loss': [], 'val_loss': [], 'val_rmse': []}
    best_val_loss = float('inf')
    best_state = None

    # Phase 1: Only predictor
    if verbose:
        print("\n📍 Phase 1: Training predictor only...")
    model.freeze_encoder()
    optimizer = optim.Adam(model.predictor.parameters(), lr=0.001)

    for epoch in tqdm(range(15), desc="Phase 1", disable=not verbose):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_rmse = validate(model, val_loader, criterion, DEVICE)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Phase 2: Unfreeze attention + bottleneck
    if verbose:
        print("\n📍 Phase 2: Unfreezing attention layer...")
    for param in model.encoder.attention.parameters():
        param.requires_grad = True
    for param in model.encoder.bottleneck.parameters():
        param.requires_grad = True

    optimizer = optim.Adam([
        {'params': model.encoder.attention.parameters(), 'lr': 0.0001},
        {'params': model.encoder.bottleneck.parameters(), 'lr': 0.0001},
        {'params': model.predictor.parameters(), 'lr': 0.0005}
    ])

    for epoch in tqdm(range(15), desc="Phase 2", disable=not verbose):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_rmse = validate(model, val_loader, criterion, DEVICE)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Phase 3: Unfreeze everything with very low LR
    if verbose:
        print("\n📍 Phase 3: Full fine-tuning...")
    model.unfreeze_encoder()
    optimizer = optim.Adam([
        {'params': model.encoder.lstm.parameters(), 'lr': 0.00001},
        {'params': model.encoder.layer_norm.parameters(), 'lr': 0.00005},
        {'params': model.encoder.attention.parameters(), 'lr': 0.00005},
        {'params': model.encoder.bottleneck.parameters(), 'lr': 0.0001},
        {'params': model.predictor.parameters(), 'lr': 0.0003}
    ])

    early_stopping = EarlyStopping(patience=15)

    for epoch in tqdm(range(30), desc="Phase 3", disable=not verbose):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_rmse = validate(model, val_loader, criterion, DEVICE)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if early_stopping(val_loss):
            if verbose:
                print(f"\nEarly stopping at phase 3 epoch {epoch + 1}")
            break

    model.load_state_dict(best_state)

    if verbose:
        print(f"\n✅ Best validation RMSE: {min(history['val_rmse']):.4f}")

    return model, history


def run_improved_experiment():
    """Run experiment with multiple unfreezing strategies."""

    print("\n" + "=" * 80)
    print("IMPROVED EXPERIMENT: BEATING HAR WITH UNFROZEN ENCODER")
    print("=" * 80)

    # Load data
    print("\n📂 Loading data...")
    spx_train, spx_val, spx_test = load_data_splits('data', 'spx')
    btc_train, btc_val, btc_test = load_data_splits('data', 'btc')

    spx_train_loader, spx_val_loader, _, spx_norm = create_dataloaders(
        spx_train, spx_val, spx_test, lookback=20, horizon=1, batch_size=32
    )

    btc_train_loader, btc_val_loader, btc_test_loader, btc_norm = create_dataloaders(
        btc_train, btc_val, btc_test, lookback=20, horizon=1, batch_size=16
    )

    # HAR baseline
    print("\n📊 Computing HAR baseline...")
    full_train = np.concatenate([btc_train, btc_val])
    har_model = HARModel()
    har_model.fit(full_train)
    har_preds = har_model.predict(btc_test)
    har_targets = har_model.get_targets(btc_test)
    har_rmse = rmse(har_targets, har_preds)
    print(f"   HAR RMSE: {har_rmse:.4f} (TARGET TO BEAT)")

    results = {'har': {'rmse': har_rmse, 'preds': har_preds, 'targets': har_targets}}

    # =========================================================================
    # Strategy 1: Simple Unfrozen with Differential LR
    # =========================================================================
    print("\n" + "=" * 80)
    print("STRATEGY 1: Differential Learning Rates")
    print("=" * 80)

    model1 = TransferRoughVolModel(
        input_dim=1, hidden_dim=64, encoding_dim=32,
        num_layers=3, num_heads=4, forecast_horizon=1, dropout=0.1
    )

    # Pre-train
    model1, _ = pretrain_on_source(
        model1, spx_train_loader, spx_val_loader,
        epochs=50, learning_rate=0.001, patience=10, verbose=True
    )

    # Fine-tune with unfrozen encoder
    model1, history1 = finetune_unfrozen(
        model1, btc_train_loader, btc_val_loader,
        epochs=60, encoder_lr=0.00001, predictor_lr=0.0005, patience=15
    )

    # Evaluate
    model1.eval()
    preds1, targets1 = [], []
    with torch.no_grad():
        for X, y in btc_test_loader:
            X = X.to(DEVICE)
            pred = model1(X)
            preds1.append(pred.cpu().numpy())
            targets1.append(y.numpy())

    preds1 = np.concatenate(preds1) * btc_norm['std'] + btc_norm['mean']
    targets1 = np.concatenate(targets1) * btc_norm['std'] + btc_norm['mean']

    rmse1 = rmse(targets1[:len(har_targets)], preds1[:len(har_targets)])
    print(f"\n📊 Strategy 1 RMSE: {rmse1:.4f}")
    print(f"   vs HAR: {((har_rmse - rmse1) / har_rmse * 100):+.2f}%")

    results['strategy1'] = {'rmse': rmse1, 'preds': preds1, 'history': history1}

    # =========================================================================
    # Strategy 2: Gradual Unfreezing
    # =========================================================================
    print("\n" + "=" * 80)
    print("STRATEGY 2: Gradual Unfreezing")
    print("=" * 80)

    model2 = TransferRoughVolModel(
        input_dim=1, hidden_dim=64, encoding_dim=32,
        num_layers=3, num_heads=4, forecast_horizon=1, dropout=0.1
    )

    # Pre-train
    model2, _ = pretrain_on_source(
        model2, spx_train_loader, spx_val_loader,
        epochs=50, learning_rate=0.001, patience=10, verbose=True
    )

    # Gradual unfreeze
    model2, history2 = finetune_gradual_unfreeze(
        model2, btc_train_loader, btc_val_loader, verbose=True
    )

    # Evaluate
    model2.eval()
    preds2 = []
    with torch.no_grad():
        for X, y in btc_test_loader:
            X = X.to(DEVICE)
            pred = model2(X)
            preds2.append(pred.cpu().numpy())

    preds2 = np.concatenate(preds2) * btc_norm['std'] + btc_norm['mean']

    rmse2 = rmse(targets1[:len(har_targets)], preds2[:len(har_targets)])
    print(f"\n📊 Strategy 2 RMSE: {rmse2:.4f}")
    print(f"   vs HAR: {((har_rmse - rmse2) / har_rmse * 100):+.2f}%")

    results['strategy2'] = {'rmse': rmse2, 'preds': preds2, 'history': history2}

    # =========================================================================
    # Strategy 3: More Aggressive Fine-tuning
    # =========================================================================
    print("\n" + "=" * 80)
    print("STRATEGY 3: Aggressive Full Fine-tuning")
    print("=" * 80)

    model3 = TransferRoughVolModel(
        input_dim=1, hidden_dim=64, encoding_dim=32,
        num_layers=3, num_heads=4, forecast_horizon=1, dropout=0.15  # More dropout
    )

    # Pre-train
    model3, _ = pretrain_on_source(
        model3, spx_train_loader, spx_val_loader,
        epochs=50, learning_rate=0.001, patience=10, verbose=True
    )

    # Aggressive fine-tuning
    model3, history3 = finetune_unfrozen(
        model3, btc_train_loader, btc_val_loader,
        epochs=80, encoder_lr=0.00005, predictor_lr=0.001, patience=20
    )

    # Evaluate
    model3.eval()
    preds3 = []
    with torch.no_grad():
        for X, y in btc_test_loader:
            X = X.to(DEVICE)
            pred = model3(X)
            preds3.append(pred.cpu().numpy())

    preds3 = np.concatenate(preds3) * btc_norm['std'] + btc_norm['mean']

    rmse3 = rmse(targets1[:len(har_targets)], preds3[:len(har_targets)])
    print(f"\n📊 Strategy 3 RMSE: {rmse3:.4f}")
    print(f"   vs HAR: {((har_rmse - rmse3) / har_rmse * 100):+.2f}%")

    results['strategy3'] = {'rmse': rmse3, 'preds': preds3, 'history': history3}

    # =========================================================================
    # FINAL COMPARISON
    # =========================================================================
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    print(f"\n{'Model':<30} {'RMSE':<12} {'vs HAR':<15}")
    print("-" * 60)
    print(f"{'HAR (Baseline)':<30} {har_rmse:<12.4f} {'---':<15}")
    print(f"{'Strategy 1 (Diff. LR)':<30} {rmse1:<12.4f} {((har_rmse - rmse1) / har_rmse * 100):+.2f}%")
    print(f"{'Strategy 2 (Gradual)':<30} {rmse2:<12.4f} {((har_rmse - rmse2) / har_rmse * 100):+.2f}%")
    print(f"{'Strategy 3 (Aggressive)':<30} {rmse3:<12.4f} {((har_rmse - rmse3) / har_rmse * 100):+.2f}%")
    print("-" * 60)

    # Find best
    best_rmse = min(rmse1, rmse2, rmse3)
    if best_rmse < har_rmse:
        print(f"\n🎉 SUCCESS! Best transfer model beats HAR by {((har_rmse - best_rmse) / har_rmse * 100):.2f}%")
    else:
        print(f"\n⚠️ HAR still wins. Best transfer RMSE: {best_rmse:.4f}")
        print(f"   Gap to close: {((best_rmse - har_rmse) / har_rmse * 100):.2f}%")

    # Save comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    n_plot = min(200, len(har_targets))
    x = np.arange(n_plot)

    # HAR
    ax = axes[0, 0]
    ax.plot(x, har_targets[:n_plot], 'k-', label='Actual', linewidth=1)
    ax.plot(x, har_preds[:n_plot], 'r-', label='HAR', linewidth=1, alpha=0.8)
    ax.fill_between(x, har_targets[:n_plot], har_preds[:n_plot], alpha=0.3, color='red')
    ax.set_title(f'HAR Model (RMSE: {har_rmse:.4f})', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Strategy 1
    ax = axes[0, 1]
    ax.plot(x, har_targets[:n_plot], 'k-', label='Actual', linewidth=1)
    ax.plot(x, preds1[:n_plot], 'g-', label='Transfer', linewidth=1, alpha=0.8)
    ax.fill_between(x, har_targets[:n_plot], preds1[:n_plot], alpha=0.3, color='green')
    ax.set_title(f'Strategy 1: Diff. LR (RMSE: {rmse1:.4f})', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Strategy 2
    ax = axes[1, 0]
    ax.plot(x, har_targets[:n_plot], 'k-', label='Actual', linewidth=1)
    ax.plot(x, preds2[:n_plot], 'b-', label='Transfer', linewidth=1, alpha=0.8)
    ax.fill_between(x, har_targets[:n_plot], preds2[:n_plot], alpha=0.3, color='blue')
    ax.set_title(f'Strategy 2: Gradual (RMSE: {rmse2:.4f})', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Strategy 3
    ax = axes[1, 1]
    ax.plot(x, har_targets[:n_plot], 'k-', label='Actual', linewidth=1)
    ax.plot(x, preds3[:n_plot], 'm-', label='Transfer', linewidth=1, alpha=0.8)
    ax.fill_between(x, har_targets[:n_plot], preds3[:n_plot], alpha=0.3, color='purple')
    ax.set_title(f'Strategy 3: Aggressive (RMSE: {rmse3:.4f})', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/unfrozen_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✓ Saved comparison to: results/unfrozen_comparison.png")

    return results


if __name__ == "__main__":
    results = run_improved_experiment()
