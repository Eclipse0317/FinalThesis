"""
run_eda.py

Exploratory Data Analysis for the CNY/CNH Hedging Project.
Produces all descriptive statistics, stationarity tests, cointegration
analysis, and plots needed for the thesis data chapter.

Run from project root: python run_eda.py
Outputs saved to outputs/eda/
"""

import warnings
from statsmodels.tools.sm_exceptions import ValueWarning
warnings.filterwarnings("ignore", category=ValueWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os

from src.data_loader import load_data
from src.config import VECM_MAX_LAGS
from src.cointegration_tests import run_cointegration_tests


OUTPUT_DIR = "outputs/eda"


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================================
# 1. Descriptive Statistics
# =====================================================================

def descriptive_statistics(data):
    """Summary statistics for price levels and log returns."""
    print("\n" + "=" * 70)
    print("Descriptive Statistics")
    print("=" * 70)

    # Price levels
    print("\n  Price Levels (USD/CNY and USD/CNH):")
    prices = data[["CNY", "CNH"]]
    desc = prices.describe().T
    desc["skew"] = prices.skew()
    desc["kurt"] = prices.kurtosis()
    print(desc.to_string(float_format=lambda x: f"{x:.4f}"))

    # Log returns
    print("\n  Weekly Log Returns (× 100):")
    returns = data[["r_CNY", "r_CNH"]]
    desc_r = returns.describe().T
    desc_r["skew"] = returns.skew()
    desc_r["kurt"] = returns.kurtosis()
    print(desc_r.to_string(float_format=lambda x: f"{x:.4f}"))

    # Correlation
    corr = returns.corr().iloc[0, 1]
    print(f"\n  Correlation (r_CNY, r_CNH): {corr:.4f}")

    # Basis (CNH - CNY spread)
    basis = data["CNH"] - data["CNY"]
    print(f"\n  CNH-CNY Basis (price spread):")
    print(f"    Mean: {basis.mean():.4f}")
    print(f"    Std:  {basis.std():.4f}")
    print(f"    Min:  {basis.min():.4f}")
    print(f"    Max:  {basis.max():.4f}")

    # Jarque-Bera normality test on returns
    print(f"\n  Jarque-Bera Normality Test:")
    for col in ["r_CNY", "r_CNH"]:
        jb_stat, jb_p = stats.jarque_bera(data[col].dropna())
        sig = "***" if jb_p < 0.01 else "**" if jb_p < 0.05 else ""
        print(f"    {col}: JB={jb_stat:.4f}, p={jb_p:.6f} {sig}")


# =====================================================================
# 2. Stationarity Tests
# =====================================================================

def stationarity_tests(data):
    """ADF tests on price levels and log returns."""
    print("\n" + "=" * 70)
    print("Stationarity Tests (Augmented Dickey-Fuller)")
    print("=" * 70)

    series_to_test = {
        "CNY (price level)": data["CNY"],
        "CNH (price level)": data["CNH"],
        "log(CNY)": np.log(data["CNY"]),
        "log(CNH)": np.log(data["CNH"]),
        "r_CNY (log return)": data["r_CNY"],
        "r_CNH (log return)": data["r_CNH"],
        "CNH-CNY basis": data["CNH"] - data["CNY"],
        "log(CNY)-log(CNH)": np.log(data["CNY"]) - np.log(data["CNH"]),
    }

    print(f"\n  {'Series':<25} {'ADF Stat':>10} {'p-value':>10} {'Lags':>6} {'Result':>20}")
    print(f"  {'-'*75}")

    for name, series in series_to_test.items():
        s = series.dropna()
        adf_stat, adf_p, adf_lags, _, _, _ = adfuller(s, autolag="AIC")
        
        if adf_p < 0.01:
            result = "Stationary ***"
        elif adf_p < 0.05:
            result = "Stationary **"
        elif adf_p < 0.10:
            result = "Stationary *"
        else:
            result = "Unit root (I(1))"
        
        print(f"  {name:<25} {adf_stat:>10.4f} {adf_p:>10.4f} {adf_lags:>6} {result:>20}")

    print(f"\n  Expected: Price levels are I(1), log returns are I(0), spread is I(0).")


# =====================================================================
# 3. Plots
# =====================================================================

def plot_price_levels(data):
    """Plot CNY and CNH price levels over time."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(data.index, data["CNY"], label="USD/CNY", linewidth=1.0)
    axes[0].plot(data.index, data["CNH"], label="USD/CNH", linewidth=1.0)
    axes[0].set_ylabel("Exchange Rate", fontsize=12)
    axes[0].set_title("USD/CNY and USD/CNH Weekly Spot Rates", fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Basis
    basis = data["CNH"] - data["CNY"]
    axes[1].plot(data.index, basis, color="darkred", linewidth=0.8)
    axes[1].axhline(y=0, color="black", linestyle="--", linewidth=0.5)
    axes[1].set_ylabel("CNH - CNY Spread", fontsize=12)
    axes[1].set_title("Onshore-Offshore Basis", fontsize=14)
    axes[1].set_xlabel("Date", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "price_levels.png")
    plt.savefig(filepath, dpi=200)
    plt.close()
    print(f"  Saved: {filepath}")


def plot_log_returns(data):
    """Plot log return series for CNY and CNH."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(data.index, data["r_CNY"], color="steelblue", linewidth=0.7)
    axes[0].set_ylabel("r_CNY (× 100)", fontsize=12)
    axes[0].set_title("USD/CNY Weekly Log Returns", fontsize=14)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(data.index, data["r_CNH"], color="coral", linewidth=0.7)
    axes[1].set_ylabel("r_CNH (× 100)", fontsize=12)
    axes[1].set_title("USD/CNH Weekly Log Returns", fontsize=14)
    axes[1].set_xlabel("Date", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "log_returns.png")
    plt.savefig(filepath, dpi=200)
    plt.close()
    print(f"  Saved: {filepath}")


def plot_return_scatter(data):
    """Scatter plot of CNY vs CNH returns with OLS fit line."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(data["r_CNH"], data["r_CNY"], alpha=0.4, s=15, color="steelblue")

    # OLS fit line
    slope = np.cov(data["r_CNY"], data["r_CNH"])[0, 1] / np.var(data["r_CNH"], ddof=1)
    x_range = np.linspace(data["r_CNH"].min(), data["r_CNH"].max(), 100)
    ax.plot(x_range, slope * x_range, color="red", linewidth=1.5,
            label=f"OLS: h = {slope:.4f}")

    ax.set_xlabel("r_CNH (× 100)", fontsize=12)
    ax.set_ylabel("r_CNY (× 100)", fontsize=12)
    ax.set_title("CNY vs CNH Weekly Log Returns", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "return_scatter.png")
    plt.savefig(filepath, dpi=200)
    plt.close()
    print(f"  Saved: {filepath}")


def plot_return_distributions(data):
    """Histograms of return distributions with normal overlay."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, col, color in zip(axes, ["r_CNY", "r_CNH"], ["steelblue", "coral"]):
        returns = data[col].dropna()
        ax.hist(returns, bins=40, density=True, alpha=0.7, color=color, edgecolor="white")

        # Normal overlay
        x = np.linspace(returns.min(), returns.max(), 200)
        ax.plot(x, stats.norm.pdf(x, returns.mean(), returns.std()),
                color="black", linewidth=1.5, label="Normal")

        ax.set_xlabel(f"{col} (× 100)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"{col} Distribution", fontsize=14)
        ax.legend(fontsize=10)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "return_distributions.png")
    plt.savefig(filepath, dpi=200)
    plt.close()
    print(f"  Saved: {filepath}")


def plot_acf_returns(data):
    """ACF and PACF plots for return series and squared returns."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    plot_acf(data["r_CNY"].dropna(), ax=axes[0, 0], lags=20, title="ACF: r_CNY")
    plot_acf(data["r_CNH"].dropna(), ax=axes[0, 1], lags=20, title="ACF: r_CNH")
    plot_acf(data["r_CNY"].dropna()**2, ax=axes[1, 0], lags=20, title="ACF: r_CNY² (volatility clustering)")
    plot_acf(data["r_CNH"].dropna()**2, ax=axes[1, 1], lags=20, title="ACF: r_CNH² (volatility clustering)")

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "acf_plots.png")
    plt.savefig(filepath, dpi=200)
    plt.close()
    print(f"  Saved: {filepath}")


def plot_rolling_correlation(data, window=26):
    """Rolling correlation between CNY and CNH returns."""
    fig, ax = plt.subplots(figsize=(14, 5))

    rolling_corr = data["r_CNY"].rolling(window=window).corr(data["r_CNH"])
    ax.plot(data.index, rolling_corr, color="darkblue", linewidth=1.0)
    ax.axhline(y=data["r_CNY"].corr(data["r_CNH"]), color="red", linestyle="--",
               linewidth=1.0, label=f"Full-sample: {data['r_CNY'].corr(data['r_CNH']):.4f}")

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Correlation", fontsize=12)
    ax.set_title(f"Rolling {window}-Week Correlation (r_CNY, r_CNH)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "rolling_correlation.png")
    plt.savefig(filepath, dpi=200)
    plt.close()
    print(f"  Saved: {filepath}")


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    weekly_data = load_data()

    ensure_output_dir()

    # 1. Descriptive Statistics
    descriptive_statistics(weekly_data)

    # 2. Stationarity Tests
    stationarity_tests(weekly_data)

    # 3. Cointegration Tests (on full sample, log prices)
    run_cointegration_tests(weekly_data, max_lags=VECM_MAX_LAGS)

    # 4. Plots
    print("\n  Generating EDA plots...")
    plot_price_levels(weekly_data)
    plot_log_returns(weekly_data)
    plot_return_scatter(weekly_data)
    plot_return_distributions(weekly_data)
    plot_acf_returns(weekly_data)
    plot_rolling_correlation(weekly_data)

    print("\nEDA Complete! All outputs saved to outputs/eda/")