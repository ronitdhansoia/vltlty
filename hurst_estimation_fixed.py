"""
Hurst Parameter Estimation for Rough Volatility Analysis (FIXED)

Estimates Hurst exponent using multiple methods on daily log-RV data.
Both S&P 500 and Bitcoin exhibit H < 0.5, confirming rough volatility.

Author: Ronit Dhansoia
Date: December 2025
"""

import os
import numpy as np
import pandas as pd
import pywt
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "data"
RESULTS_DIR = "results"


def variogram_hurst(signal, max_lag=100):
    """
    Estimate Hurst parameter using variogram method.

    For a process with Hurst exponent H:
    E[(X(t+τ) - X(t))²] ∝ τ^{2H}

    Returns H, R², and the slope.
    """
    n = len(signal)
    lags = np.arange(1, min(max_lag, n // 4))
    gamma = []

    for lag in lags:
        diff = signal[lag:] - signal[:-lag]
        gamma.append(np.mean(diff**2))

    log_lags = np.log(lags)
    log_gamma = np.log(gamma)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_lags, log_gamma)

    H = slope / 2

    return {
        'H': H,
        'R2': r_value**2,
        'slope': slope,
        'lags': lags,
        'gamma': np.array(gamma),
        'is_rough': H < 0.5
    }


def dfa_hurst(signal, min_scale=10, max_scale=None):
    """
    Estimate Hurst parameter using Detrended Fluctuation Analysis.
    """
    n = len(signal)
    if max_scale is None:
        max_scale = n // 4

    # Integrate signal (cumulative sum of deviations from mean)
    y = np.cumsum(signal - np.mean(signal))

    scales = np.unique(np.logspace(np.log10(min_scale), np.log10(max_scale), 25).astype(int))
    fluctuations = []
    valid_scales = []

    for s in scales:
        num_segments = n // s
        if num_segments < 2:
            continue

        F2 = 0
        for i in range(num_segments):
            segment = y[i*s:(i+1)*s]
            x = np.arange(s)
            # Linear detrend
            p = np.polyfit(x, segment, 1)
            trend = np.polyval(p, x)
            F2 += np.sum((segment - trend)**2)

        F2 /= (num_segments * s)
        fluctuations.append(np.sqrt(F2))
        valid_scales.append(s)

    log_scales = np.log(valid_scales)
    log_fluct = np.log(fluctuations)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_scales, log_fluct)

    return {
        'H': slope,
        'R2': r_value**2,
        'scales': np.array(valid_scales),
        'fluctuations': np.array(fluctuations),
        'is_rough': slope < 0.5
    }


def rs_hurst(signal, min_window=10):
    """
    Estimate Hurst parameter using Rescaled Range (R/S) analysis.
    """
    n = len(signal)
    max_window = n // 4

    windows = np.unique(np.logspace(np.log10(min_window), np.log10(max_window), 20).astype(int))
    rs_values = []
    valid_windows = []

    for window in windows:
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
            valid_windows.append(window)

    log_windows = np.log(valid_windows)
    log_rs = np.log(rs_values)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_windows, log_rs)

    return {
        'H': slope,
        'R2': r_value**2,
        'windows': np.array(valid_windows),
        'rs_values': np.array(rs_values),
        'is_rough': slope < 0.5
    }


def wavelet_analysis(signal, wavelet='db4', max_level=7):
    """
    Perform wavelet analysis for roughness characterization.
    Returns the slope of log-variance vs log-scale.
    """
    coeffs = pywt.wavedec(signal, wavelet, level=max_level)

    variances = []
    scales = []

    for j in range(1, len(coeffs)):
        var = np.var(coeffs[j])
        if var > 0:
            variances.append(var)
            scales.append(2 ** j)

    log_scales = np.log2(scales)
    log_variances = np.log2(variances)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_scales, log_variances)

    return {
        'slope': slope,
        'R2': r_value**2,
        'scales': np.array(scales),
        'variances': np.array(variances),
        'log_scales': log_scales,
        'log_variances': log_variances
    }


def plot_hurst_analysis(spx_var, btc_var, spx_wav, btc_wav,
                        spx_log_rv, btc_log_rv, save_path):
    """Create visualization of Hurst analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    spx_color = '#1f77b4'
    btc_color = '#ff7f0e'

    # Top Left: S&P 500 Log RV
    ax1 = axes[0, 0]
    ax1.plot(spx_log_rv, color=spx_color, linewidth=0.5, alpha=0.8)
    ax1.axhline(y=np.mean(spx_log_rv), color='red', linestyle='--', linewidth=1)
    ax1.set_title(f'S&P 500 Log Realized Volatility\n'
                  f'H = {spx_var["H"]:.3f} (Variogram), {"Rough" if spx_var["is_rough"] else "Smooth"}',
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Trading Day')
    ax1.set_ylabel('Log RV')
    ax1.grid(True, alpha=0.3)

    # Top Right: Bitcoin Log RV
    ax2 = axes[0, 1]
    ax2.plot(btc_log_rv, color=btc_color, linewidth=0.5, alpha=0.8)
    ax2.axhline(y=np.mean(btc_log_rv), color='red', linestyle='--', linewidth=1)
    ax2.set_title(f'Bitcoin Log Realized Volatility\n'
                  f'H = {btc_var["H"]:.3f} (Variogram), {"Rough" if btc_var["is_rough"] else "Smooth"}',
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Trading Day')
    ax2.set_ylabel('Log RV')
    ax2.grid(True, alpha=0.3)

    # Bottom Left: S&P 500 Wavelet Analysis
    ax3 = axes[1, 0]
    ax3.scatter(spx_wav['log_scales'], spx_wav['log_variances'],
                s=100, color=spx_color, zorder=5, label='Wavelet Variances')
    x_fit = np.linspace(min(spx_wav['log_scales']), max(spx_wav['log_scales']), 100)
    y_fit = spx_wav['slope'] * x_fit + (spx_wav['log_variances'][0] - spx_wav['slope'] * spx_wav['log_scales'][0])
    ax3.plot(x_fit, y_fit, 'r--', linewidth=2,
             label=f'Slope = {spx_wav["slope"]:.3f}, R² = {spx_wav["R2"]:.3f}')
    ax3.set_xlabel('log₂(Scale)')
    ax3.set_ylabel('log₂(Wavelet Variance)')
    ax3.set_title('S&P 500 Wavelet Analysis', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Bottom Right: Bitcoin Wavelet Analysis
    ax4 = axes[1, 1]
    ax4.scatter(btc_wav['log_scales'], btc_wav['log_variances'],
                s=100, color=btc_color, zorder=5, label='Wavelet Variances')
    x_fit = np.linspace(min(btc_wav['log_scales']), max(btc_wav['log_scales']), 100)
    y_fit = btc_wav['slope'] * x_fit + (btc_wav['log_variances'][0] - btc_wav['slope'] * btc_wav['log_scales'][0])
    ax4.plot(x_fit, y_fit, 'r--', linewidth=2,
             label=f'Slope = {btc_wav["slope"]:.3f}, R² = {btc_wav["R2"]:.3f}')
    ax4.set_xlabel('log₂(Scale)')
    ax4.set_ylabel('log₂(Wavelet Variance)')
    ax4.set_title('Bitcoin Wavelet Analysis', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved visualization to: {save_path}")


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("HURST PARAMETER ESTIMATION - ROUGH VOLATILITY ANALYSIS")
    print("=" * 70)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load data
    print("\n📂 Loading data...")
    spx_data = pd.read_csv(os.path.join(DATA_DIR, 'spx_data.csv'), index_col=0)
    btc_data = pd.read_csv(os.path.join(DATA_DIR, 'btc_data.csv'), index_col=0)

    spx_log_rv = spx_data['log_rv'].values
    btc_log_rv = btc_data['log_rv'].values

    print(f"  S&P 500: {len(spx_log_rv):,} daily observations")
    print(f"  Bitcoin: {len(btc_log_rv):,} daily observations")

    # Variogram method
    print("\n" + "-" * 70)
    print("VARIOGRAM METHOD (Primary)")
    print("-" * 70)

    spx_var = variogram_hurst(spx_log_rv)
    btc_var = variogram_hurst(btc_log_rv)

    print(f"\n📊 S&P 500:")
    print(f"  H = {spx_var['H']:.4f} (R² = {spx_var['R2']:.4f})")
    print(f"  {'ROUGH (H < 0.5) ✓' if spx_var['is_rough'] else 'SMOOTH (H ≥ 0.5)'}")

    print(f"\n📊 Bitcoin:")
    print(f"  H = {btc_var['H']:.4f} (R² = {btc_var['R2']:.4f})")
    print(f"  {'ROUGH (H < 0.5) ✓' if btc_var['is_rough'] else 'SMOOTH (H ≥ 0.5)'}")

    # DFA method
    print("\n" + "-" * 70)
    print("DFA METHOD")
    print("-" * 70)

    spx_dfa = dfa_hurst(spx_log_rv)
    btc_dfa = dfa_hurst(btc_log_rv)

    print(f"  S&P 500: H = {spx_dfa['H']:.4f}")
    print(f"  Bitcoin: H = {btc_dfa['H']:.4f}")

    # R/S method
    print("\n" + "-" * 70)
    print("R/S METHOD")
    print("-" * 70)

    spx_rs = rs_hurst(spx_log_rv)
    btc_rs = rs_hurst(btc_log_rv)

    print(f"  S&P 500: H = {spx_rs['H']:.4f}")
    print(f"  Bitcoin: H = {btc_rs['H']:.4f}")

    # Wavelet analysis
    print("\n" + "-" * 70)
    print("WAVELET ANALYSIS")
    print("-" * 70)

    spx_wav = wavelet_analysis(spx_log_rv)
    btc_wav = wavelet_analysis(btc_log_rv)

    print(f"  S&P 500: slope = {spx_wav['slope']:.4f} (R² = {spx_wav['R2']:.4f})")
    print(f"  Bitcoin: slope = {btc_wav['slope']:.4f} (R² = {btc_wav['R2']:.4f})")

    # Generate visualization
    print("\n" + "-" * 70)
    print("GENERATING VISUALIZATION")
    print("-" * 70)

    plot_path = os.path.join(RESULTS_DIR, 'hurst_comparison.png')
    plot_hurst_analysis(spx_var, btc_var, spx_wav, btc_wav,
                       spx_log_rv, btc_log_rv, plot_path)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY TABLE (For Paper)")
    print("=" * 70)
    print(f"\n{'Asset':<10} {'Variogram':<12} {'DFA':<12} {'R/S':<12} {'Wavelet Slope':<15}")
    print("-" * 60)
    print(f"{'S&P 500':<10} {spx_var['H']:<12.3f} {spx_dfa['H']:<12.3f} {spx_rs['H']:<12.3f} {spx_wav['slope']:<15.3f}")
    print(f"{'Bitcoin':<10} {btc_var['H']:<12.3f} {btc_dfa['H']:<12.3f} {btc_rs['H']:<12.3f} {btc_wav['slope']:<15.3f}")

    print("\n📝 Key Finding:")
    if spx_var['is_rough'] and btc_var['is_rough']:
        print("  ✅ BOTH assets exhibit rough volatility (Variogram H < 0.5)")
        print(f"  → S&P 500: H = {spx_var['H']:.3f}")
        print(f"  → Bitcoin: H = {btc_var['H']:.3f}")

    print("\n✅ Hurst estimation complete!")


if __name__ == "__main__":
    main()
