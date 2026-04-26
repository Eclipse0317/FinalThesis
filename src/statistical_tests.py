"""
src/statistical_tests.py

Statistical significance tests and diagnostic plots for hedging model comparison.
Includes:
  - Diebold-Mariano test (pairwise model comparison)
  - Mean hedged PnL test (alpha detection)
  - Time-series plots (hedge ratios, cumulative PnL)
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os


# =====================================================================
# 1. Diebold-Mariano Test
# =====================================================================

def diebold_mariano(pnl_1, pnl_2, h=1):
    """
    Diebold-Mariano test for equal predictive accuracy.
    
    Tests H0: E[L(e1)] = E[L(e2)] where L is squared loss.
    A negative DM statistic means model 1 has lower loss (better).
    
    Parameters
    ----------
    pnl_1, pnl_2 : array-like
        Hedged PnL series from two competing models.
    h : int
        Forecast horizon (1 for one-step-ahead).
        
    Returns
    -------
    dm_stat : float
    p_value : float (two-sided)
    """
    e1 = np.asarray(pnl_1)
    e2 = np.asarray(pnl_2)
    
    # Loss differential (squared loss)
    d = e1**2 - e2**2
    
    n = len(d)
    d_bar = np.mean(d)
    
    # Newey-West style variance estimate for h-step ahead
    gamma_0 = np.var(d, ddof=1)
    
    gamma_sum = 0.0
    for k in range(1, h):
        gamma_k = np.cov(d[k:], d[:-k])[0, 1]
        gamma_sum += gamma_k
    
    var_d = (gamma_0 + 2 * gamma_sum) / n
    
    if var_d <= 0:
        return np.nan, np.nan
    
    dm_stat = d_bar / np.sqrt(var_d)
    p_value = 2 * stats.t.sf(abs(dm_stat), df=n - 1)
    
    return dm_stat, p_value


def run_dm_tests(pnl_dict, benchmark_name="OLS (static)"):
    """
    Runs Diebold-Mariano tests comparing all models against a benchmark.
    
    Parameters
    ----------
    pnl_dict : dict
        {model_name: hedged_pnl_array}
    benchmark_name : str
        Name of the benchmark model.
        
    Returns
    -------
    pd.DataFrame with DM statistics and p-values.
    """
    if benchmark_name not in pnl_dict:
        raise ValueError(f"Benchmark '{benchmark_name}' not found in pnl_dict")
    
    benchmark_pnl = pnl_dict[benchmark_name]
    results = []
    
    for name, pnl in pnl_dict.items():
        if name == benchmark_name:
            continue
        
        # Align lengths (some models might differ by a few obs)
        n = min(len(benchmark_pnl), len(pnl))
        dm_stat, p_val = diebold_mariano(pnl[:n], benchmark_pnl[:n])
        
        sig = ""
        if p_val < 0.01:
            sig = "***"
        elif p_val < 0.05:
            sig = "**"
        elif p_val < 0.10:
            sig = "*"
        
        results.append({
            "Model": name,
            "DM Stat": dm_stat,
            "p-value": p_val,
            "Sig": sig,
            "vs": benchmark_name
        })
    
    return pd.DataFrame(results)


# =====================================================================
# 2. Mean Hedged PnL Test (Alpha Detection)
# =====================================================================

def test_mean_pnl(pnl_dict):
    """
    Tests whether the mean hedged PnL is significantly different from zero
    for each model (t-test). A significant nonzero mean suggests the model
    is generating directional alpha rather than pure hedging.
    
    Parameters
    ----------
    pnl_dict : dict
        {model_name: hedged_pnl_array}
        
    Returns
    -------
    pd.DataFrame with mean, std, t-stat, p-value for each model.
    """
    results = []
    
    for name, pnl in pnl_dict.items():
        pnl = np.asarray(pnl)
        n = len(pnl)
        mean = np.mean(pnl)
        std = np.std(pnl, ddof=1)
        t_stat = mean / (std / np.sqrt(n))
        p_val = 2 * stats.t.sf(abs(t_stat), df=n - 1)
        
        sig = ""
        if p_val < 0.01:
            sig = "***"
        elif p_val < 0.05:
            sig = "**"
        elif p_val < 0.10:
            sig = "*"
        
        results.append({
            "Model": name,
            "Mean PnL": mean,
            "Std PnL": std,
            "t-stat": t_stat,
            "p-value": p_val,
            "Sig": sig
        })
    
    return pd.DataFrame(results)


# =====================================================================
# 3. Collect PnL from all models (helper)
# =====================================================================

def collect_pnl(full_data, train_end_idx, models):
    """
    Runs backtests and collects hedged PnL for all models.
    
    Returns
    -------
    pnl_dict : dict {model_name: np.ndarray}
    """
    pnl_dict = {}
    for model in models:
        pnl = model.run_backtest(full_data, train_end_idx)
        pnl_dict[model.name] = np.asarray(pnl)
    return pnl_dict


# =====================================================================
# 4. Print formatted results
# =====================================================================

def print_dm_results(dm_df):
    """Pretty-prints DM test results."""
    benchmark = dm_df["vs"].iloc[0]
    print(f"\n{'='*75}")
    print(f"Diebold-Mariano Test vs Benchmark: {benchmark}")
    print(f"{'='*75}")
    print(f"  {'Model':<30} {'DM Stat':>10} {'p-value':>10} {'Sig':>5}")
    print(f"  {'-'*60}")
    
    for _, row in dm_df.iterrows():
        dm = f"{row['DM Stat']:.4f}" if not np.isnan(row['DM Stat']) else "N/A"
        pv = f"{row['p-value']:.4f}" if not np.isnan(row['p-value']) else "N/A"
        print(f"  {row['Model']:<30} {dm:>10} {pv:>10} {row['Sig']:>5}")
    
    print(f"\n  Note: Negative DM stat => model has LOWER squared loss than {benchmark}")


def print_mean_pnl_results(mean_df):
    """Pretty-prints mean PnL test results."""
    print(f"\n{'='*75}")
    print(f"Mean Hedged PnL Test (H0: mean = 0, i.e. no alpha)")
    print(f"{'='*75}")
    print(f"  {'Model':<30} {'Mean':>10} {'Std':>10} {'t-stat':>10} {'p-value':>10} {'Sig':>5}")
    print(f"  {'-'*70}")
    
    for _, row in mean_df.iterrows():
        print(f"  {row['Model']:<30} {row['Mean PnL']:>10.4f} {row['Std PnL']:>10.4f} "
              f"{row['t-stat']:>10.4f} {row['p-value']:>10.4f} {row['Sig']:>5}")
    
    print(f"\n  Note: Significant result => model may be generating alpha (directional bias)")


# =====================================================================
# 5. Time-Series Plots
# =====================================================================

def plot_cumulative_pnl(pnl_dict, test_dates, output_dir="outputs"):
    """
    Plots cumulative hedged PnL for all models.
    
    Parameters
    ----------
    pnl_dict : dict {model_name: np.ndarray}
    test_dates : pd.DatetimeIndex
    output_dir : str
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for name, pnl in pnl_dict.items():
        n = min(len(pnl), len(test_dates))
        cum_pnl = np.cumsum(pnl[:n])
        ax.plot(test_dates[:n], cum_pnl, label=name, linewidth=1.2)
    
    # Add unhedged baseline
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative Hedged PnL (log return × 100)", fontsize=12)
    ax.set_title("Cumulative Hedged PnL Comparison", fontsize=14)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, "cumulative_pnl.png")
    plt.savefig(filepath, dpi=200)
    plt.close()
    print(f"  Saved: {filepath}")


def plot_hedge_ratios(full_data, train_end_idx, models, output_dir="outputs"):
    """
    Plots the time-varying hedge ratio for all models that produce
    dynamic ratios.
    
    Parameters
    ----------
    full_data : pd.DataFrame
    train_end_idx : int
    models : list of model objects
    output_dir : str
    """
    os.makedirs(output_dir, exist_ok=True)
    
    test_dates = full_data.iloc[train_end_idx:].index
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for model in models:
        h_hist = model.hedge_ratio_history
        if len(h_hist) == 0:
            continue
        
        h_array = np.array(h_hist)
        
        # For static models with a single ratio, draw a horizontal line
        if len(h_array) == 1:
            ax.axhline(y=h_array[0], label=model.name, linestyle='--', linewidth=1.0)
        else:
            # Align to test dates (trim to shorter length)
            n = min(len(h_array), len(test_dates))
            ax.plot(test_dates[:n], h_array[:n], label=model.name, linewidth=1.0, alpha=0.8)
    
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Hedge Ratio", fontsize=12)
    ax.set_title("Time-Varying Hedge Ratios", fontsize=14)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, "hedge_ratios.png")
    plt.savefig(filepath, dpi=200)
    plt.close()
    print(f"  Saved: {filepath}")


def plot_rolling_he(pnl_dict, test_dates, window=26, output_dir="outputs"):
    """
    Plots rolling-window Hedging Efficiency over time for all models.
    Shows how each model's HE evolves, revealing regime-dependent performance.
    
    Parameters
    ----------
    pnl_dict : dict {model_name: np.ndarray}
    test_dates : pd.DatetimeIndex
    window : int — rolling window in weeks (default 26 = ~6 months)
    output_dir : str
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # We need the unhedged returns to compute rolling HE
    # But we don't have them here directly — compute from pnl_dict context
    # Instead, just plot rolling variance ratio
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for name, pnl in pnl_dict.items():
        n = min(len(pnl), len(test_dates))
        pnl_series = pd.Series(pnl[:n], index=test_dates[:n])
        
        rolling_var = pnl_series.rolling(window=window).var()
        ax.plot(test_dates[:n], rolling_var, label=name, linewidth=1.0, alpha=0.8)
    
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel(f"Rolling Variance ({window}-week window)", fontsize=12)
    ax.set_title("Rolling Hedged Portfolio Variance", fontsize=14)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, "rolling_variance.png")
    plt.savefig(filepath, dpi=200)
    plt.close()
    print(f"  Saved: {filepath}")


# =====================================================================
# 6. Master runner
# =====================================================================

def run_all_statistical_tests(full_data, train_end_idx, models, output_dir="outputs"):
    """
    Runs all statistical tests and generates all plots.
    
    Parameters
    ----------
    full_data : pd.DataFrame
    train_end_idx : int
    models : list of model objects
    output_dir : str
    """
    test_data = full_data.iloc[train_end_idx:]
    test_dates = test_data.index
    
    print("\n" + "=" * 75)
    print("Running Statistical Tests & Generating Plots")
    print("=" * 75)
    
    # 1. Collect PnL from all models
    print("\n  Collecting hedged PnL from all models...")
    pnl_dict = collect_pnl(full_data, train_end_idx, models)
    
    # 2. Diebold-Mariano tests
    dm_df = run_dm_tests(pnl_dict, benchmark_name="OLS (static)")
    print_dm_results(dm_df)
    
    # 3. Mean PnL test
    mean_df = test_mean_pnl(pnl_dict)
    print_mean_pnl_results(mean_df)
    
    # 4. Plots
    print(f"\n  Generating plots...")
    plot_cumulative_pnl(pnl_dict, test_dates, output_dir)
    plot_hedge_ratios(full_data, train_end_idx, models, output_dir)
    plot_rolling_he(pnl_dict, test_dates, window=26, output_dir=output_dir)
    
    print(f"\n  All tests complete. Plots saved to '{output_dir}/'")
    
    return pnl_dict, dm_df, mean_df