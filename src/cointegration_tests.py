"""
src/cointegration_tests.py

Formal cointegration testing for the thesis.
Runs Johansen trace and eigenvalue tests on log prices,
plus ADF tests on the spread for additional confirmation.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import select_order, select_coint_rank
from statsmodels.tsa.stattools import adfuller


def run_cointegration_tests(data, max_lags=12):
    """
    Runs and reports cointegration tests on log price levels.
    
    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'CNY' and 'CNH' price columns.
    max_lags : int
        Maximum lags for lag order selection.
    """
    log_prices = np.log(data[["CNY", "CNH"]]).dropna()
    
    print("\n" + "=" * 70)
    print("Cointegration Analysis (Log Prices)")
    print("=" * 70)
    
    # 1. Lag Order Selection
    lag_order = select_order(log_prices, maxlags=max_lags, deterministic="ci")
    print(f"\nLag Order Selection (max={max_lags}):")
    print(f"  AIC: {lag_order.aic}  |  BIC: {lag_order.bic}  |  HQIC: {lag_order.hqic}")
    chosen_lag = max(lag_order.bic, 1)
    print(f"  Selected lag (BIC): {chosen_lag}")
    
    # 2. Johansen Cointegration Test
    print(f"\nJohansen Cointegration Test (lag={chosen_lag}):")
    
    # Trace test
    print(f"\n  Trace Test:")
    print(f"  {'H0':<20} {'Test Stat':>12} {'5% CV':>12} {'Reject?':>10}")
    print(f"  {'-'*55}")
    
    rank_trace = select_coint_rank(log_prices, det_order=0, k_ar_diff=chosen_lag, 
                                    method="trace", signif=0.05)
    
    # Access test statistics from the CointRankResults object
    trace_stats = rank_trace.test_stats
    trace_cvs = rank_trace.crit_vals
    
    for i in range(len(trace_stats)):
        h0 = f"r ≤ {i}"
        stat = trace_stats[i]
        cv = trace_cvs[i]
        reject = "Yes ***" if stat > cv else "No"
        print(f"  {h0:<20} {stat:>12.4f} {cv:>12.4f} {reject:>10}")
    
    print(f"  Selected rank: {rank_trace.rank}")
    
    # Max eigenvalue test
    print(f"\n  Maximum Eigenvalue Test:")
    print(f"  {'H0':<20} {'Test Stat':>12} {'5% CV':>12} {'Reject?':>10}")
    print(f"  {'-'*55}")
    
    rank_maxeig = select_coint_rank(log_prices, det_order=0, k_ar_diff=chosen_lag, 
                                     method="maxeig", signif=0.05)
    
    maxeig_stats = rank_maxeig.test_stats
    maxeig_cvs = rank_maxeig.crit_vals
    
    for i in range(len(maxeig_stats)):
        h0 = f"r = {i}"
        stat = maxeig_stats[i]
        cv = maxeig_cvs[i]
        reject = "Yes ***" if stat > cv else "No"
        print(f"  {h0:<20} {stat:>12.4f} {cv:>12.4f} {reject:>10}")
    
    print(f"  Selected rank: {rank_maxeig.rank}")
    
    # 3. ADF test on the log spread (supplementary evidence)
    log_spread = log_prices["CNY"] - log_prices["CNH"]
    adf_stat, adf_p, adf_lags, _, adf_cvs, _ = adfuller(log_spread, autolag="AIC")
    
    print(f"\n  Supplementary: ADF Test on log(CNY) - log(CNH) spread:")
    print(f"    ADF Statistic: {adf_stat:.4f}")
    print(f"    p-value:       {adf_p:.6f}")
    print(f"    Lags used:     {adf_lags}")
    print(f"    Critical values: 1%={adf_cvs['1%']:.4f}, 5%={adf_cvs['5%']:.4f}, 10%={adf_cvs['10%']:.4f}")
    
    sig = "***" if adf_p < 0.01 else "**" if adf_p < 0.05 else "*" if adf_p < 0.10 else ""
    print(f"    Conclusion: {'Stationary (cointegrated)' if adf_p < 0.05 else 'Non-stationary'} {sig}")
    
    # 4. Also test individual series for unit roots (should NOT reject)
    print(f"\n  Unit Root Tests on Individual Log Price Series:")
    for col in ["CNY", "CNH"]:
        adf_s, adf_pv, _, _, _, _ = adfuller(log_prices[col], autolag="AIC")
        sig = "***" if adf_pv < 0.01 else "**" if adf_pv < 0.05 else ""
        result = "Stationary" if adf_pv < 0.05 else "Unit root (I(1))"
        print(f"    log({col}): ADF={adf_s:.4f}, p={adf_pv:.4f} => {result} {sig}")
    
    print(f"\n{'='*70}")
    
    return chosen_lag, rank_trace.rank