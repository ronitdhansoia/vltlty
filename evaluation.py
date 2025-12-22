"""
Evaluation Metrics for Rough Volatility Forecasting

This module implements evaluation metrics for comparing model performance:
    1. Standard metrics: RMSE, MAE, MAPE
    2. Directional accuracy (for trading applications)
    3. Diebold-Mariano test for statistical significance
    4. R-squared and correlation metrics

These metrics quantify the benefit of transfer learning over baselines.

Author: Ronit Dhansoia
Date: 22nd December 2025
"""

import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# STANDARD FORECASTING METRICS
# ============================================================================

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        RMSE value
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        MAE value
    """
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        epsilon: Small value to prevent division by zero

    Returns:
        MAPE value (as percentage)
    """
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R-squared (coefficient of determination).

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        R² value
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / (ss_tot + 1e-8))


def correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Pearson correlation coefficient.

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        Correlation coefficient
    """
    return np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]


# ============================================================================
# DIRECTIONAL ACCURACY (FOR TRADING)
# ============================================================================

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Directional Accuracy - proportion of correct direction predictions.

    Useful for trading: predicting whether volatility will increase or decrease.

    Args:
        y_true: Actual values (differences or levels)
        y_pred: Predicted values (differences or levels)

    Returns:
        Directional accuracy (0 to 1)
    """
    # Compute directions (sign of change)
    # For volatility, we compare against previous value
    if len(y_true) < 2:
        return np.nan

    # If inputs are level predictions, compute directions
    true_direction = np.sign(np.diff(y_true.flatten()))
    pred_direction = np.sign(np.diff(y_pred.flatten()))

    # Handle zero changes
    true_direction[true_direction == 0] = 1
    pred_direction[pred_direction == 0] = 1

    return np.mean(true_direction == pred_direction)


def directional_accuracy_from_levels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prev_values: np.ndarray
) -> float:
    """
    Directional accuracy comparing predicted change to actual change.

    Args:
        y_true: Actual next values
        y_pred: Predicted next values
        prev_values: Previous values (used to compute direction)

    Returns:
        Directional accuracy
    """
    true_direction = np.sign(y_true - prev_values)
    pred_direction = np.sign(y_pred - prev_values)

    # Handle zero changes
    true_direction[true_direction == 0] = 1
    pred_direction[pred_direction == 0] = 1

    return np.mean(true_direction == pred_direction)


# ============================================================================
# DIEBOLD-MARIANO TEST
# ============================================================================

def diebold_mariano_test(
    y_true: np.ndarray,
    y_pred_1: np.ndarray,
    y_pred_2: np.ndarray,
    loss_type: str = 'squared'
) -> Dict:
    """
    Diebold-Mariano test for comparing forecast accuracy.

    Tests H0: Two forecasts have equal predictive accuracy
    Tests H1: Forecast 1 is more accurate than Forecast 2

    Reference: Diebold & Mariano (1995)

    Args:
        y_true: Actual values
        y_pred_1: Predictions from model 1 (should be better model)
        y_pred_2: Predictions from model 2 (baseline)
        loss_type: 'squared' for MSE, 'absolute' for MAE

    Returns:
        Dictionary with test statistic, p-value, and interpretation
    """
    # Compute loss differentials
    if loss_type == 'squared':
        e1 = (y_true - y_pred_1) ** 2
        e2 = (y_true - y_pred_2) ** 2
    else:  # absolute
        e1 = np.abs(y_true - y_pred_1)
        e2 = np.abs(y_true - y_pred_2)

    d = e2 - e1  # Positive if model 1 is better

    n = len(d)
    mean_d = np.mean(d)

    # Compute variance using Newey-West HAC estimator (simplified)
    # Using autocorrelation at lag 0 only for simplicity
    var_d = np.var(d, ddof=1) / n

    # DM statistic
    if var_d > 0:
        dm_stat = mean_d / np.sqrt(var_d)
    else:
        dm_stat = 0.0

    # One-sided p-value (testing if model 1 is better)
    p_value = 1 - stats.norm.cdf(dm_stat)

    # Two-sided p-value
    p_value_two_sided = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

    # Interpretation
    if p_value < 0.01:
        interpretation = "Model 1 significantly better (p < 0.01)"
    elif p_value < 0.05:
        interpretation = "Model 1 significantly better (p < 0.05)"
    elif p_value < 0.10:
        interpretation = "Model 1 marginally better (p < 0.10)"
    else:
        interpretation = "No significant difference"

    return {
        'dm_statistic': dm_stat,
        'p_value_one_sided': p_value,
        'p_value_two_sided': p_value_two_sided,
        'mean_loss_diff': mean_d,
        'model_1_better': mean_d > 0,
        'interpretation': interpretation
    }


# ============================================================================
# COMPREHENSIVE EVALUATION
# ============================================================================

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str = "Model"
) -> Dict:
    """
    Compute all evaluation metrics for a model.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        name: Model name for labeling

    Returns:
        Dictionary with all metrics
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Ensure same length
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    return {
        'name': name,
        'rmse': rmse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'mape': mape(y_true, y_pred),
        'r_squared': r_squared(y_true, y_pred),
        'correlation': correlation(y_true, y_pred),
        'directional_accuracy': directional_accuracy(y_true, y_pred),
        'n_samples': min_len
    }


def compare_models(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    baseline_name: str = 'har'
) -> Dict:
    """
    Compare multiple models against a baseline.

    Args:
        y_true: Actual values
        predictions: Dict mapping model name to predictions
        baseline_name: Name of baseline model for DM test

    Returns:
        Dictionary with metrics for all models and DM test results
    """
    results = {}

    # Compute metrics for each model
    for name, y_pred in predictions.items():
        results[name] = compute_all_metrics(y_true, y_pred, name)

    # Diebold-Mariano tests against baseline
    if baseline_name in predictions:
        baseline_pred = predictions[baseline_name]
        results['dm_tests'] = {}

        for name, y_pred in predictions.items():
            if name != baseline_name:
                # Align lengths
                min_len = min(len(y_true), len(y_pred), len(baseline_pred))
                dm_result = diebold_mariano_test(
                    y_true[:min_len],
                    y_pred[:min_len],
                    baseline_pred[:min_len]
                )
                results['dm_tests'][f"{name}_vs_{baseline_name}"] = dm_result

    return results


def print_comparison_table(results: Dict, title: str = "Model Comparison"):
    """
    Print a formatted comparison table.

    Args:
        results: Results from compare_models()
        title: Table title
    """
    print("\n" + "=" * 80)
    print(f"{title}")
    print("=" * 80)

    # Header
    print(f"\n{'Model':<20} {'RMSE':<10} {'MAE':<10} {'MAPE(%)':<10} {'R²':<10} {'Corr':<10} {'Dir.Acc':<10}")
    print("-" * 80)

    # Results for each model
    for name, metrics in results.items():
        if name == 'dm_tests':
            continue
        print(f"{metrics['name']:<20} {metrics['rmse']:<10.4f} {metrics['mae']:<10.4f} "
              f"{metrics['mape']:<10.2f} {metrics['r_squared']:<10.4f} "
              f"{metrics['correlation']:<10.4f} {metrics['directional_accuracy']:<10.4f}")

    print("-" * 80)

    # DM test results
    if 'dm_tests' in results:
        print("\nDiebold-Mariano Test Results:")
        print("-" * 80)
        for comparison, dm in results['dm_tests'].items():
            print(f"  {comparison}: DM={dm['dm_statistic']:.3f}, p={dm['p_value_one_sided']:.4f} - {dm['interpretation']}")


def compute_improvement(
    baseline_metrics: Dict,
    transfer_metrics: Dict
) -> Dict:
    """
    Compute percentage improvement of transfer model over baseline.

    Args:
        baseline_metrics: Metrics for baseline model
        transfer_metrics: Metrics for transfer learning model

    Returns:
        Dictionary with improvement percentages
    """
    improvements = {}

    # Lower is better for RMSE, MAE, MAPE
    for metric in ['rmse', 'mae', 'mape']:
        baseline_val = baseline_metrics[metric]
        transfer_val = transfer_metrics[metric]
        improvement = (baseline_val - transfer_val) / baseline_val * 100
        improvements[f'{metric}_improvement_%'] = improvement

    # Higher is better for R², correlation, directional accuracy
    for metric in ['r_squared', 'correlation', 'directional_accuracy']:
        baseline_val = baseline_metrics[metric]
        transfer_val = transfer_metrics[metric]
        if baseline_val != 0:
            improvement = (transfer_val - baseline_val) / abs(baseline_val) * 100
        else:
            improvement = float('inf') if transfer_val > baseline_val else 0
        improvements[f'{metric}_improvement_%'] = improvement

    return improvements


# ============================================================================
# MAIN (FOR TESTING)
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING EVALUATION METRICS")
    print("=" * 60)

    # Create synthetic data for testing
    np.random.seed(42)
    n = 100

    y_true = np.sin(np.linspace(0, 4*np.pi, n)) + np.random.normal(0, 0.1, n)

    # Good predictions (transfer model)
    y_pred_good = y_true + np.random.normal(0, 0.15, n)

    # Okay predictions (baseline)
    y_pred_okay = y_true + np.random.normal(0, 0.25, n)

    # Bad predictions (naive)
    y_pred_bad = y_true + np.random.normal(0, 0.5, n)

    # Test individual metrics
    print("\n📊 Individual Metrics (Good Model):")
    print(f"  RMSE: {rmse(y_true, y_pred_good):.4f}")
    print(f"  MAE:  {mae(y_true, y_pred_good):.4f}")
    print(f"  MAPE: {mape(y_true, y_pred_good):.2f}%")
    print(f"  R²:   {r_squared(y_true, y_pred_good):.4f}")
    print(f"  Corr: {correlation(y_true, y_pred_good):.4f}")
    print(f"  Dir.Acc: {directional_accuracy(y_true, y_pred_good):.4f}")

    # Test comparison
    print("\n📊 Model Comparison:")
    predictions = {
        'transfer': y_pred_good,
        'baseline': y_pred_okay,
        'naive': y_pred_bad
    }

    results = compare_models(y_true, predictions, baseline_name='baseline')
    print_comparison_table(results, "Model Comparison Results")

    # Test improvement calculation
    print("\n📈 Improvement (Transfer vs Baseline):")
    improvements = compute_improvement(
        results['baseline'],
        results['transfer']
    )
    for metric, value in improvements.items():
        print(f"  {metric}: {value:+.2f}%")

    print("\n✅ Evaluation metrics test complete!")
