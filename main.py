"""
Main Experiment Script: Cross-Asset Transfer Learning for Rough Volatility Forecasting

This script orchestrates the complete experiment pipeline:
    1. Load and prepare data (S&P 500 and Bitcoin)
    2. Pre-train model on S&P 500 (source domain)
    3. Fine-tune on Bitcoin (target domain)
    4. Train baseline models for comparison
    5. Evaluate all models and generate results
    6. Save visualizations and metrics

Usage:
    python main.py

Author: Ronit Dhansoia
Date: 22nd December 2025
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from models import TransferRoughVolModel
from datasets import create_dataloaders, load_data_splits
from train import pretrain_on_source, finetune_on_target
from baselines import train_lstm_from_scratch, HARModel, MovingAverageBaseline
from evaluation import (
    compute_all_metrics, compare_models, print_comparison_table,
    compute_improvement, diebold_mariano_test
)


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Data
    'data_dir': 'data',
    'lookback': 20,
    'horizon': 1,

    # Model architecture
    'input_dim': 1,
    'hidden_dim': 64,
    'encoding_dim': 32,
    'num_layers': 3,
    'num_heads': 4,
    'dropout': 0.1,

    # Pre-training (S&P 500)
    'pretrain_epochs': 50,
    'pretrain_lr': 0.001,
    'pretrain_patience': 10,
    'pretrain_batch_size': 32,

    # Fine-tuning (Bitcoin)
    'finetune_epochs': 30,
    'finetune_lr': 0.0001,
    'finetune_patience': 10,
    'finetune_batch_size': 16,
    'freeze_encoder': True,

    # Output
    'models_dir': 'models',
    'results_dir': 'results',
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_training_history(pretrain_history: dict, finetune_history: dict, save_path: str):
    """Plot training curves for pre-training and fine-tuning phases."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Pre-training losses
    ax1 = axes[0, 0]
    ax1.plot(pretrain_history['train_loss'], label='Train Loss', color='blue')
    ax1.plot(pretrain_history['val_loss'], label='Val Loss', color='orange')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Phase 1: Pre-training on S&P 500', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Pre-training RMSE
    ax2 = axes[0, 1]
    ax2.plot(pretrain_history['val_rmse'], label='Val RMSE', color='green')
    ax2.axhline(y=min(pretrain_history['val_rmse']), color='red', linestyle='--',
                label=f"Best: {min(pretrain_history['val_rmse']):.4f}")
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('RMSE')
    ax2.set_title('Pre-training Validation RMSE', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Fine-tuning losses
    ax3 = axes[1, 0]
    ax3.plot(finetune_history['train_loss'], label='Train Loss', color='blue')
    ax3.plot(finetune_history['val_loss'], label='Val Loss', color='orange')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss (MSE)')
    ax3.set_title('Phase 2: Fine-tuning on Bitcoin', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Fine-tuning RMSE
    ax4 = axes[1, 1]
    ax4.plot(finetune_history['val_rmse'], label='Val RMSE', color='green')
    ax4.axhline(y=min(finetune_history['val_rmse']), color='red', linestyle='--',
                label=f"Best: {min(finetune_history['val_rmse']):.4f}")
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('RMSE')
    ax4.set_title('Fine-tuning Validation RMSE', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved training history to: {save_path}")


def plot_predictions_comparison(
    test_targets: np.ndarray,
    predictions: dict,
    save_path: str,
    n_samples: int = 200
):
    """Plot actual vs predicted values for all models."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Limit samples for clarity
    n = min(n_samples, len(test_targets))
    x = np.arange(n)
    targets = test_targets[:n]

    models = list(predictions.keys())
    colors = {'transfer': '#2ecc71', 'lstm_scratch': '#3498db', 'har': '#e74c3c', 'moving_avg': '#9b59b6'}

    for idx, model_name in enumerate(models[:4]):
        ax = axes[idx // 2, idx % 2]
        preds = predictions[model_name][:n]

        ax.plot(x, targets, 'k-', label='Actual', linewidth=1, alpha=0.8)
        ax.plot(x, preds, '-', color=colors.get(model_name, 'gray'),
                label=f'{model_name} predictions', linewidth=1, alpha=0.8)
        ax.fill_between(x, targets, preds, alpha=0.3, color=colors.get(model_name, 'gray'))

        ax.set_xlabel('Time Step')
        ax.set_ylabel('Log Realized Volatility')
        ax.set_title(f'{model_name.replace("_", " ").title()}', fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved predictions comparison to: {save_path}")


def plot_results_summary(results: dict, improvements: dict, save_path: str):
    """Create a summary visualization of model performance."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    models = [k for k in results.keys() if k != 'dm_tests']
    metrics_to_plot = ['rmse', 'mae', 'r_squared']
    titles = ['RMSE (Lower is Better)', 'MAE (Lower is Better)', 'R² (Higher is Better)']
    colors = {'transfer': '#2ecc71', 'lstm_scratch': '#3498db', 'har': '#e74c3c', 'moving_avg': '#9b59b6'}

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        values = [results[m][metric] for m in models]
        bars = ax.bar(models, values, color=[colors.get(m, 'gray') for m in models])

        # Highlight best model
        if metric in ['rmse', 'mae']:
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)

        ax.set_ylabel(metric.upper())
        ax.set_title(titles[idx], fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved results summary to: {save_path}")


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment():
    """Run the complete transfer learning experiment."""

    print("\n" + "=" * 80)
    print("CROSS-ASSET TRANSFER LEARNING FOR ROUGH VOLATILITY FORECASTING")
    print("=" * 80)
    print(f"\nExperiment started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")

    # Create directories
    os.makedirs(CONFIG['models_dir'], exist_ok=True)
    os.makedirs(CONFIG['results_dir'], exist_ok=True)

    # =========================================================================
    # STEP 1: LOAD DATA
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: LOADING DATA")
    print("=" * 80)

    # Load S&P 500 data (source domain)
    spx_train, spx_val, spx_test = load_data_splits(CONFIG['data_dir'], 'spx')
    print(f"\nS&P 500 (Source Domain):")
    print(f"  Train: {len(spx_train):,} samples")
    print(f"  Val:   {len(spx_val):,} samples")
    print(f"  Test:  {len(spx_test):,} samples")

    # Load Bitcoin data (target domain)
    btc_train, btc_val, btc_test = load_data_splits(CONFIG['data_dir'], 'btc')
    print(f"\nBitcoin (Target Domain):")
    print(f"  Train: {len(btc_train):,} samples")
    print(f"  Val:   {len(btc_val):,} samples")
    print(f"  Test:  {len(btc_test):,} samples")

    # Create DataLoaders
    spx_train_loader, spx_val_loader, spx_test_loader, spx_norm = create_dataloaders(
        spx_train, spx_val, spx_test,
        lookback=CONFIG['lookback'],
        horizon=CONFIG['horizon'],
        batch_size=CONFIG['pretrain_batch_size']
    )

    btc_train_loader, btc_val_loader, btc_test_loader, btc_norm = create_dataloaders(
        btc_train, btc_val, btc_test,
        lookback=CONFIG['lookback'],
        horizon=CONFIG['horizon'],
        batch_size=CONFIG['finetune_batch_size']
    )

    print(f"\nNormalization Parameters:")
    print(f"  S&P 500: mean={spx_norm['mean']:.4f}, std={spx_norm['std']:.4f}")
    print(f"  Bitcoin: mean={btc_norm['mean']:.4f}, std={btc_norm['std']:.4f}")

    # =========================================================================
    # STEP 2: CREATE AND PRE-TRAIN MODEL
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: PRE-TRAINING ON S&P 500")
    print("=" * 80)

    # Create model
    model = TransferRoughVolModel(
        input_dim=CONFIG['input_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        encoding_dim=CONFIG['encoding_dim'],
        num_layers=CONFIG['num_layers'],
        num_heads=CONFIG['num_heads'],
        forecast_horizon=CONFIG['horizon'],
        dropout=CONFIG['dropout']
    )

    print(f"\nModel Architecture:")
    print(f"  Total parameters: {model.get_total_params():,}")

    # Pre-train on S&P 500
    model, pretrain_history = pretrain_on_source(
        model, spx_train_loader, spx_val_loader,
        epochs=CONFIG['pretrain_epochs'],
        learning_rate=CONFIG['pretrain_lr'],
        patience=CONFIG['pretrain_patience'],
        save_path=os.path.join(CONFIG['models_dir'], 'pretrained_spx.pth'),
        verbose=True
    )

    # =========================================================================
    # STEP 3: FINE-TUNE ON BITCOIN
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: FINE-TUNING ON BITCOIN")
    print("=" * 80)

    model, finetune_history = finetune_on_target(
        model, btc_train_loader, btc_val_loader,
        epochs=CONFIG['finetune_epochs'],
        learning_rate=CONFIG['finetune_lr'],
        patience=CONFIG['finetune_patience'],
        freeze_encoder=CONFIG['freeze_encoder'],
        save_path=os.path.join(CONFIG['models_dir'], 'finetuned_btc.pth'),
        verbose=True
    )

    # =========================================================================
    # STEP 4: TRAIN BASELINES
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: TRAINING BASELINE MODELS")
    print("=" * 80)

    # 4.1 LSTM from scratch (no transfer learning)
    lstm_scratch, lstm_history = train_lstm_from_scratch(
        btc_train_loader, btc_val_loader,
        epochs=45,
        learning_rate=0.001,
        device=DEVICE,
        verbose=True
    )

    # 4.2 HAR Model
    print("\n" + "-" * 60)
    print("BASELINE: HAR Model")
    print("-" * 60)
    full_train = np.concatenate([btc_train, btc_val])
    har_model = HARModel()
    har_model.fit(full_train)
    print("✓ HAR model fitted")

    # 4.3 Moving Average
    print("\n" + "-" * 60)
    print("BASELINE: Moving Average")
    print("-" * 60)
    ma_model = MovingAverageBaseline(window=5)
    print("✓ Moving Average model ready")

    # =========================================================================
    # STEP 5: GENERATE TEST PREDICTIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: GENERATING TEST PREDICTIONS")
    print("=" * 80)

    # Transfer model predictions
    model.eval()
    transfer_preds, transfer_targets = [], []
    with torch.no_grad():
        for X, y in btc_test_loader:
            X = X.to(DEVICE)
            pred = model(X)
            transfer_preds.append(pred.cpu().numpy())
            transfer_targets.append(y.numpy())

    transfer_preds = np.concatenate(transfer_preds)
    transfer_targets = np.concatenate(transfer_targets)

    # Inverse transform
    transfer_preds = transfer_preds * btc_norm['std'] + btc_norm['mean']
    transfer_targets = transfer_targets * btc_norm['std'] + btc_norm['mean']

    # LSTM scratch predictions
    lstm_scratch.eval()
    lstm_preds = []
    with torch.no_grad():
        for X, y in btc_test_loader:
            X = X.to(DEVICE)
            pred = lstm_scratch(X)
            lstm_preds.append(pred.cpu().numpy())

    lstm_preds = np.concatenate(lstm_preds)
    lstm_preds = lstm_preds * btc_norm['std'] + btc_norm['mean']

    # HAR predictions
    har_preds = har_model.predict(btc_test)
    har_targets = har_model.get_targets(btc_test)

    # Moving Average predictions
    ma_preds, ma_targets = ma_model.predict(btc_test)

    print(f"\nPrediction shapes:")
    print(f"  Transfer: {transfer_preds.shape}")
    print(f"  LSTM Scratch: {lstm_preds.shape}")
    print(f"  HAR: {har_preds.shape}")
    print(f"  Moving Avg: {ma_preds.shape}")

    # =========================================================================
    # STEP 6: EVALUATE ALL MODELS
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: EVALUATING MODELS")
    print("=" * 80)

    # Align predictions to common length (due to different lookback requirements)
    min_len = min(len(transfer_targets), len(har_targets), len(ma_targets))

    predictions = {
        'transfer': transfer_preds[:min_len],
        'lstm_scratch': lstm_preds[:min_len],
        'har': har_preds[:min_len],
        'moving_avg': ma_preds[:min_len]
    }

    # Use HAR targets as ground truth (they're aligned with HAR predictions)
    test_targets = har_targets[:min_len]

    # Compute metrics
    results = compare_models(test_targets, predictions, baseline_name='har')
    print_comparison_table(results, "FINAL MODEL COMPARISON")

    # Compute improvement over baselines
    print("\n" + "-" * 80)
    print("IMPROVEMENT ANALYSIS")
    print("-" * 80)

    for baseline in ['lstm_scratch', 'har', 'moving_avg']:
        print(f"\n📈 Transfer vs {baseline.replace('_', ' ').title()}:")
        improvements = compute_improvement(results[baseline], results['transfer'])
        for metric, value in improvements.items():
            if 'rmse' in metric or 'mae' in metric or 'mape' in metric:
                direction = "reduction" if value > 0 else "increase"
            else:
                direction = "improvement" if value > 0 else "decrease"
            print(f"   {metric.replace('_', ' ')}: {value:+.2f}% ({direction})")

    # =========================================================================
    # STEP 7: SAVE RESULTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 7: SAVING RESULTS")
    print("=" * 80)

    # Save training history plot
    plot_training_history(
        pretrain_history, finetune_history,
        os.path.join(CONFIG['results_dir'], 'training_history.png')
    )

    # Save predictions comparison
    plot_predictions_comparison(
        test_targets, predictions,
        os.path.join(CONFIG['results_dir'], 'predictions_comparison.png')
    )

    # Save results summary
    plot_results_summary(
        results,
        compute_improvement(results['har'], results['transfer']),
        os.path.join(CONFIG['results_dir'], 'results_summary.png')
    )

    # Save metrics to JSON
    metrics_json = {
        'timestamp': datetime.now().isoformat(),
        'config': CONFIG,
        'results': {
            name: {k: float(v) if isinstance(v, (np.floating, float)) else v
                   for k, v in metrics.items()}
            for name, metrics in results.items() if name != 'dm_tests'
        },
        'dm_tests': {
            name: {k: float(v) if isinstance(v, (np.floating, float)) else v
                   for k, v in test.items()}
            for name, test in results.get('dm_tests', {}).items()
        },
        'improvements': {
            f"vs_{baseline}": compute_improvement(results[baseline], results['transfer'])
            for baseline in ['lstm_scratch', 'har', 'moving_avg']
        }
    }

    with open(os.path.join(CONFIG['results_dir'], 'experiment_results.json'), 'w') as f:
        json.dump(metrics_json, f, indent=2, default=str)

    print(f"✓ Saved experiment results to: {CONFIG['results_dir']}/experiment_results.json")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE - FINAL SUMMARY")
    print("=" * 80)

    transfer_rmse = results['transfer']['rmse']
    lstm_rmse = results['lstm_scratch']['rmse']
    har_rmse = results['har']['rmse']

    print(f"\n📊 Key Results:")
    print(f"   Transfer Model RMSE: {transfer_rmse:.4f}")
    print(f"   LSTM (no transfer) RMSE: {lstm_rmse:.4f}")
    print(f"   HAR Baseline RMSE: {har_rmse:.4f}")

    improvement_vs_lstm = (lstm_rmse - transfer_rmse) / lstm_rmse * 100
    improvement_vs_har = (har_rmse - transfer_rmse) / har_rmse * 100

    print(f"\n📈 Transfer Learning Benefits:")
    print(f"   vs LSTM from scratch: {improvement_vs_lstm:+.2f}% RMSE reduction")
    print(f"   vs HAR baseline: {improvement_vs_har:+.2f}% RMSE reduction")

    # Statistical significance
    if 'dm_tests' in results and 'transfer_vs_har' in results['dm_tests']:
        dm = results['dm_tests']['transfer_vs_har']
        print(f"\n📉 Statistical Significance (Diebold-Mariano test):")
        print(f"   DM statistic: {dm['dm_statistic']:.3f}")
        print(f"   p-value: {dm['p_value_one_sided']:.4f}")
        print(f"   Result: {dm['interpretation']}")

    print(f"\n📁 Output files saved to: {CONFIG['results_dir']}/")
    print(f"   - training_history.png")
    print(f"   - predictions_comparison.png")
    print(f"   - results_summary.png")
    print(f"   - experiment_results.json")

    print(f"\n🏁 Experiment completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return results


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    results = run_experiment()
