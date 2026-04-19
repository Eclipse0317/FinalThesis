"""
src/evaluation.py

This module evaluates the out-of-sample performance of fitted hedge models.
It leverages the internal backtest engine of BaseHedgeModel.
"""

import numpy as np
import pandas as pd

def evaluate_out_of_sample(full_data, train_end_idx, models_to_test):
    """
    Takes a list of initialized models, triggers their internal backtests,
    and calculates Hedging Efficiency (HE).
    
    Parameters:
    -----------
    full_data : pd.DataFrame
        The complete dataset containing 'r_CNY' and 'r_CNH'.
    train_end_idx : int
        The index where the training period ends and the out-of-sample period begins.
    models_to_test : list
        A list of initialized model objects.
        
    Returns:
    --------
    pd.DataFrame : A summary table of the results.
    """
    test_data = full_data.iloc[train_end_idx:]
    r_test_cny = test_data["r_CNY"].values
    var_unhedged = np.var(r_test_cny)

    print(f"\n{'='*80}")
    print(f"Out-of-Sample Hedging Efficiency Comparison (N={len(test_data)})")
    print(f"{'='*80}")
    
    summary = {}

    for model in models_to_test:
        # 1. Trigger the master backtest loop (this handles fitting and predicting)
        pnl = model.run_backtest(full_data, train_end_idx)
        
        # 2. Get Hedge Ratio Info (Dynamic or Static)
        h_info = model.get_hedge_info()
        
        # 3. Calculate Hedging Efficiency
        var_hedged = np.var(pnl)
        he = 1 - (var_hedged / var_unhedged)
        
        # 4. Store the results
        summary[model.name] = {
            "Hedge Ratio / Setup": h_info,
            "Hedging Efficiency": he,
            "Variance": var_hedged 
        }

    # Format and print the table
    summary_df = pd.DataFrame.from_dict(summary, orient="index")
    
    print(f"{'Strategy':<30} {'Hedge Ratio / Setup':>30} {'HE':>10}")
    print("-" * 80)
    for index, row in summary_df.iterrows():
        print(f"  {index:<28} {str(row['Hedge Ratio / Setup']):>30} {row['Hedging Efficiency']:>10.4f}")
    
    # Print the unhedged baseline for reference
    print("-" * 80)
    print(f"  {'Unhedged Baseline':<28} {'None':>30} {0.0:>10.4f}")

    return summary_df


def run_robustness_checks(full_data, models_to_test, splits=[0.6, 0.7, 0.8]):
    """
    Orchestrates the evaluation across multiple train/test splits to verify stability.
    """
    print("\n" + "="*80)
    print("Robustness Checks Across Different Splits")
    print("="*80)
    
    robustness_results = {}
    
    for split in splits:
        n_total = len(full_data)
        train_end_idx = int(n_total * split)
        
        print(f"\n--- Running Split: {split*100:.0f}/{100-split*100:.0f} ---")
        
        # Notice we NO LONGER call model.fit() here! 
        # We just pass the data and split index straight to the evaluator.
        split_summary = evaluate_out_of_sample(full_data, train_end_idx, models_to_test)
        robustness_results[f"Split_{split}"] = split_summary

    return robustness_results