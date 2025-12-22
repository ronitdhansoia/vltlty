"""
Hurst Parameter Estimation for Rough Volatility Analysis

This script estimates the Hurst exponent for S&P 500 and Bitcoin volatility
using the wavelet method. A Hurst parameter H < 0.5 indicates "rough" volatility,
which is the key phenomenon we're studying for transfer learning.

Methodology:
- Uses Daubechies 4 (db4) wavelet decomposition
- Computes wavelet variances at multiple scales
- Log-log regression to estimate H = (slope - 1) / 2

Reference:
- Gatheral, Jaisson, Rosenbaum (2018): "Volatility is Rough"

Usage:
    python hurst_estimation.py

Author: Ronit Dhansoia
Date: 22nd December 2025
"""

import os
import numpy as np
import pandas as pd
import pywt
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # For non-interactive plotting
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = "data"
RESULTS_DIR = "results"
WAVELET = 'db4'  # Daubechies 4 wavelet
MAX_LEVEL = 7    # Maximum decomposition level


# ============================================================================
# HURST ESTIMATION FUNCTIONS
# ============================================================================

def wavelet_variance(signal: np.ndarray, wavelet: str = 'db4',
                     max_level: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute wavelet variances at different scales.

    Args:
        signal: 1D time series (log realized volatility)
        wavelet: Wavelet type (default 'db4')
        max_level: Maximum decomposition level

    Returns:
        scales: Array of scale values (2^j)
        variances: Array of wavelet variances at each scale
    """
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(signal, wavelet, level=max_level)

    # Compute variance at each detail level
    # coeffs[0] = approximation, coeffs[1:] = details (finest to coarsest)
    variances = []
    scales = []

    for j in range(1, len(coeffs)):
        detail_coeffs = coeffs[j]
        var = np.var(detail_coeffs)
        variances.append(var)
        scales.append(2 ** j)  # Scale is 2^j

    return np.array(scales), np.array(variances)


def estimate_hurst_wavelet(signal: np.ndarray, wavelet: str = 'db4',
                           max_level: int = 7) -> Dict:
    """
    Estimate Hurst parameter using wavelet method.

    The relationship between wavelet variance and scale is:
        Var(W_j) ∝ 2^{j(2H+1)}

    Taking logs:
        log(Var(W_j)) = (2H+1) * log(2^j) + const

    So: slope = 2H + 1, thus H = (slope - 1) / 2

    Args:
        signal: 1D time series (log realized volatility)
        wavelet: Wavelet type
        max_level: Maximum decomposition level

    Returns:
        Dictionary with H estimate, R², confidence interval, and diagnostics
    """
    # Get wavelet variances
    scales, variances = wavelet_variance(signal, wavelet, max_level)

    # Filter out zero or negative variances
    valid = variances > 0
    scales = scales[valid]
    variances = variances[valid]

    if len(scales) < 3:
        return {
            'H': np.nan,
            'R2': np.nan,
            'slope': np.nan,
            'intercept': np.nan,
            'H_ci': (np.nan, np.nan),
            'is_rough': None,
            'error': 'Insufficient valid scales'
        }

    # Log-log regression
    log_scales = np.log2(scales)
    log_variances = np.log2(variances)

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        log_scales, log_variances
    )

    # Estimate H
    H = (slope - 1) / 2

    # 95% confidence interval for H
    # CI for slope, then transform
    t_crit = stats.t.ppf(0.975, len(scales) - 2)
    slope_ci = (slope - t_crit * std_err, slope + t_crit * std_err)
    H_ci = ((slope_ci[0] - 1) / 2, (slope_ci[1] - 1) / 2)

    return {
        'H': H,
        'R2': r_value ** 2,
        'slope': slope,
        'intercept': intercept,
        'std_err': std_err,
        'p_value': p_value,
        'H_ci': H_ci,
        'is_rough': H < 0.5,
        'scales': scales,
        'variances': variances,
        'log_scales': log_scales,
        'log_variances': log_variances
    }


def estimate_hurst_rs(signal: np.ndarray, min_window: int = 10) -> float:
    """
    Estimate Hurst parameter using R/S (Rescaled Range) method.

    This is an alternative method for comparison.

    Args:
        signal: 1D time series
        min_window: Minimum window size

    Returns:
        Hurst exponent estimate
    """
    n = len(signal)
    max_window = n // 4

    rs_values = []
    window_sizes = []

    for window in range(min_window, max_window + 1, max(1, (max_window - min_window) // 20)):
        rs_list = []
        for start in range(0, n - window, window):
            segment = signal[start:start + window]
            mean = np.mean(segment)
            cumdev = np.cumsum(segment - mean)
            R = np.max(cumdev) - np.min(cumdev)
            S = np.std(segment, ddof=1)
            if S > 0:
                rs_list.append(R / S)

        if rs_list:
            rs_values.append(np.mean(rs_list))
            window_sizes.append(window)

    if len(window_sizes) < 3:
        return np.nan

    log_windows = np.log(window_sizes)
    log_rs = np.log(rs_values)

    slope, _, _, _, _ = stats.linregress(log_windows, log_rs)

    return slope


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_hurst_analysis(spx_result: Dict, btc_result: Dict,
                        spx_log_rv: np.ndarray, btc_log_rv: np.ndarray,
                        save_path: str):
    """
    Create comprehensive visualization of Hurst analysis.

    Creates a 2x2 figure:
    - Top row: Log RV time series for both assets
    - Bottom row: Wavelet variance log-log plots

    Args:
        spx_result: Hurst estimation results for S&P 500
        btc_result: Hurst estimation results for Bitcoin
        spx_log_rv: S&P 500 log realized volatility
        btc_log_rv: Bitcoin log realized volatility
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Colors
    spx_color = '#1f77b4'  # Blue
    btc_color = '#ff7f0e'  # Orange

    # -------------------------------------------------------------------------
    # Top Left: S&P 500 Log RV Time Series
    # -------------------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.plot(spx_log_rv, color=spx_color, linewidth=0.5, alpha=0.8)
    ax1.axhline(y=np.mean(spx_log_rv), color='red', linestyle='--',
                linewidth=1, label=f'Mean = {np.mean(spx_log_rv):.2f}')
    ax1.set_title(f'S&P 500 Log Realized Volatility\nH = {spx_result["H"]:.3f} ({"Rough" if spx_result["is_rough"] else "Smooth"})',
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Observation')
    ax1.set_ylabel('Log RV')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Top Right: Bitcoin Log RV Time Series
    # -------------------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.plot(btc_log_rv, color=btc_color, linewidth=0.5, alpha=0.8)
    ax2.axhline(y=np.mean(btc_log_rv), color='red', linestyle='--',
                linewidth=1, label=f'Mean = {np.mean(btc_log_rv):.2f}')
    ax2.set_title(f'Bitcoin Log Realized Volatility\nH = {btc_result["H"]:.3f} ({"Rough" if btc_result["is_rough"] else "Smooth"})',
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Observation')
    ax2.set_ylabel('Log RV')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Bottom Left: S&P 500 Wavelet Variance Plot
    # -------------------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.scatter(spx_result['log_scales'], spx_result['log_variances'],
                s=100, color=spx_color, zorder=5, label='Wavelet Variances')

    # Regression line
    x_fit = np.linspace(min(spx_result['log_scales']), max(spx_result['log_scales']), 100)
    y_fit = spx_result['slope'] * x_fit + spx_result['intercept']
    ax3.plot(x_fit, y_fit, 'r--', linewidth=2,
             label=f'Fit: slope={spx_result["slope"]:.3f}, R²={spx_result["R2"]:.3f}')

    ax3.set_xlabel('log₂(Scale)')
    ax3.set_ylabel('log₂(Wavelet Variance)')
    ax3.set_title('S&P 500 Wavelet Analysis', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Bottom Right: Bitcoin Wavelet Variance Plot
    # -------------------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.scatter(btc_result['log_scales'], btc_result['log_variances'],
                s=100, color=btc_color, zorder=5, label='Wavelet Variances')

    # Regression line
    x_fit = np.linspace(min(btc_result['log_scales']), max(btc_result['log_scales']), 100)
    y_fit = btc_result['slope'] * x_fit + btc_result['intercept']
    ax4.plot(x_fit, y_fit, 'r--', linewidth=2,
             label=f'Fit: slope={btc_result["slope"]:.3f}, R²={btc_result["R2"]:.3f}')

    ax4.set_xlabel('log₂(Scale)')
    ax4.set_ylabel('log₂(Wavelet Variance)')
    ax4.set_title('Bitcoin Wavelet Analysis', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved visualization to: {save_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("HURST PARAMETER ESTIMATION - ROUGH VOLATILITY ANALYSIS")
    print("=" * 70)

    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load Data
    # -------------------------------------------------------------------------
    print("\n📂 Loading data...")

    spx_data = pd.read_csv(os.path.join(DATA_DIR, 'spx_data.csv'), index_col=0)
    btc_data = pd.read_csv(os.path.join(DATA_DIR, 'btc_data.csv'), index_col=0)

    spx_log_rv = spx_data['log_rv'].values
    btc_log_rv = btc_data['log_rv'].values

    print(f"  S&P 500: {len(spx_log_rv):,} observations")
    print(f"  Bitcoin: {len(btc_log_rv):,} observations")

    # -------------------------------------------------------------------------
    # Estimate Hurst Parameters
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("WAVELET METHOD (Primary)")
    print("-" * 70)

    print("\n📊 S&P 500 Hurst Estimation:")
    spx_result = estimate_hurst_wavelet(spx_log_rv, WAVELET, MAX_LEVEL)
    print(f"  Hurst parameter H = {spx_result['H']:.4f}")
    print(f"  95% CI: ({spx_result['H_ci'][0]:.4f}, {spx_result['H_ci'][1]:.4f})")
    print(f"  Regression R² = {spx_result['R2']:.4f}")
    print(f"  Slope = {spx_result['slope']:.4f}")
    print(f"  Interpretation: {'ROUGH (H < 0.5) ✓' if spx_result['is_rough'] else 'SMOOTH (H ≥ 0.5)'}")

    print("\n📊 Bitcoin Hurst Estimation:")
    btc_result = estimate_hurst_wavelet(btc_log_rv, WAVELET, MAX_LEVEL)
    print(f"  Hurst parameter H = {btc_result['H']:.4f}")
    print(f"  95% CI: ({btc_result['H_ci'][0]:.4f}, {btc_result['H_ci'][1]:.4f})")
    print(f"  Regression R² = {btc_result['R2']:.4f}")
    print(f"  Slope = {btc_result['slope']:.4f}")
    print(f"  Interpretation: {'ROUGH (H < 0.5) ✓' if btc_result['is_rough'] else 'SMOOTH (H ≥ 0.5)'}")

    # -------------------------------------------------------------------------
    # Alternative: R/S Method (for comparison)
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("R/S METHOD (Alternative, for comparison)")
    print("-" * 70)

    spx_H_rs = estimate_hurst_rs(spx_log_rv)
    btc_H_rs = estimate_hurst_rs(btc_log_rv)

    print(f"  S&P 500 H (R/S) = {spx_H_rs:.4f}")
    print(f"  Bitcoin H (R/S) = {btc_H_rs:.4f}")

    # -------------------------------------------------------------------------
    # Create Visualization
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("GENERATING VISUALIZATION")
    print("-" * 70)

    plot_path = os.path.join(RESULTS_DIR, 'hurst_comparison.png')
    plot_hurst_analysis(spx_result, btc_result, spx_log_rv, btc_log_rv, plot_path)

    # -------------------------------------------------------------------------
    # Summary Table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n📋 Hurst Parameter Comparison:")
    print("-" * 60)
    print(f"{'Asset':<15} {'H (Wavelet)':<15} {'H (R/S)':<15} {'Rough?':<15}")
    print("-" * 60)
    print(f"{'S&P 500':<15} {spx_result['H']:<15.4f} {spx_H_rs:<15.4f} {'YES ✓' if spx_result['is_rough'] else 'NO':<15}")
    print(f"{'Bitcoin':<15} {btc_result['H']:<15.4f} {btc_H_rs:<15.4f} {'YES ✓' if btc_result['is_rough'] else 'NO':<15}")
    print("-" * 60)

    # -------------------------------------------------------------------------
    # Interpretation for Paper
    # -------------------------------------------------------------------------
    print("\n📝 Key Findings for Paper:")
    print("-" * 60)

    if spx_result['is_rough'] and btc_result['is_rough']:
        print("  ✅ BOTH assets exhibit rough volatility (H < 0.5)")
        print("  ✅ This supports the transfer learning hypothesis!")
        print(f"  → S&P 500: H = {spx_result['H']:.3f}")
        print(f"  → Bitcoin: H = {btc_result['H']:.3f}")
        print("  → The similar roughness suggests transferable patterns")
    elif spx_result['is_rough']:
        print("  ⚠ Only S&P 500 shows rough volatility")
        print("  → Transfer learning may still work, but results may vary")
    elif btc_result['is_rough']:
        print("  ⚠ Only Bitcoin shows rough volatility")
        print("  → Source (S&P) doesn't have same roughness property")
    else:
        print("  ❌ Neither asset shows rough volatility")
        print("  → May need to reconsider the research hypothesis")

    print("\n📁 Results saved to:")
    print(f"  → {plot_path}")

    print("\n✅ Hurst estimation complete!")
    print("\n📋 Next Steps:")
    print("  1. Create exploratory analysis notebook")
    print("  2. Build PyTorch dataset class")
    print("  3. Implement model architecture")


if __name__ == "__main__":
    main()
