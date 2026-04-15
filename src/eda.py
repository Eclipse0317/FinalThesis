import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy import stats as sp_stats
from statsmodels.tsa.stattools import adfuller, kpss, coint

def run_eda(weekly):
    """EDA：描述性统计、正态性、平稳性、协整检验"""

    weekly["r_CNY"] = np.log(weekly["CNY"]).diff()
    weekly["r_CNH"] = np.log(weekly["CNH"]).diff()
    weekly["Spread"] = weekly["CNY"] - weekly["CNH"]

    print("\n" + "="*60)
    print("一、描述性统计 (价格水平)")
    print("="*60)
    print(weekly[["CNY", "CNH"]].describe().round(4))

    print(f"\n价格相关系数: {weekly['CNY'].corr(weekly['CNH']):.6f}")
    print(f"收益率相关系数: {weekly['r_CNY'].corr(weekly['r_CNH']):.6f}")

    # 价差
    print("\n" + "="*60)
    print("二、价差统计 (CNY - CNH)")
    print("="*60)
    spread = weekly["Spread"].dropna()
    print(f"  均值: {spread.mean():.6f}")
    print(f"  标准差: {spread.std():.6f}")
    print(f"  偏度: {spread.skew():.4f}")
    print(f"  峰度: {spread.kurtosis():.4f}")

    # 收益率
    print("\n" + "="*60)
    print("三、对数收益率统计")
    print("="*60)
    for col in ["r_CNY", "r_CNH"]:
        s = weekly[col].dropna()
        print(f"\n  {col}:")
        print(f"    均值: {s.mean():.6f}, 标准差: {s.std():.6f}")
        print(f"    偏度: {s.skew():.4f}, 峰度: {s.kurtosis():.4f}")

    # Jarque-Bera
    print("\n" + "="*60)
    print("四、正态性检验 (Jarque-Bera)")
    print("="*60)
    for col in ["r_CNY", "r_CNH", "Spread"]:
        s = weekly[col].dropna()
        jb, p = sp_stats.jarque_bera(s)
        sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        print(f"  {col}: JB={jb:.2f}, p={p:.6f} {sig}")

    # ADF & KPSS
    print("\n" + "="*60)
    print("五、平稳性检验 (ADF & KPSS)")
    print("="*60)
    for col in ["CNY", "CNH", "Spread", "r_CNY", "r_CNH"]:
        s = weekly[col].dropna()
        adf_stat, adf_p, _, _, _, _ = adfuller(s, autolag="AIC")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat, kpss_p, _, _ = kpss(s, regression="c", nlags="auto")
        print(f"  {col:8s}: ADF={adf_stat:7.3f}(p={adf_p:.4f}) | KPSS={kpss_stat:.4f}(p={kpss_p:.4f})")

    # 协整
    print("\n" + "="*60)
    print("六、协整检验 (Engle-Granger)")
    print("="*60)
    coint_stat, coint_p, crit = coint(weekly["CNY"].dropna(), weekly["CNH"].dropna())
    print(f"  统计量: {coint_stat:.4f}, p值: {coint_p:.6f}")
    print(f"  临界值: 1%={crit[0]:.4f}, 5%={crit[1]:.4f}, 10%={crit[2]:.4f}")

    # EDA图表
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    w = weekly.dropna()

    axes[0].plot(w.index, w["CNY"], label="USD/CNY", linewidth=0.8)
    axes[0].plot(w.index, w["CNH"], label="USD/CNH", linewidth=0.8, alpha=0.85)
    axes[0].set_title("USD/CNY vs USD/CNH (Weekly)")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(w.index, w["Spread"], color="darkred", linewidth=0.7)
    axes[1].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[1].set_title("CNY - CNH Spread"); axes[1].grid(True, alpha=0.3)

    axes[2].plot(w.index, w["r_CNY"], label="CNY", linewidth=0.6, alpha=0.8)
    axes[2].plot(w.index, w["r_CNH"], label="CNH", linewidth=0.6, alpha=0.8)
    axes[2].set_title("Weekly Log Returns"); axes[2].legend(); axes[2].grid(True, alpha=0.3)

    axes[3].scatter(w["r_CNY"], w["r_CNH"], s=8, alpha=0.4)
    axes[3].set_title(f"CNY vs CNH Returns (corr={w['r_CNY'].corr(w['r_CNH']):.3f})")
    axes[3].set_xlabel("CNY"); axes[3].set_ylabel("CNH"); axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("eda_overview.png", dpi=150)
    plt.close()
    print("\n图表已保存: eda_overview.png")

    return weekly
