"""
src/evaluation.py

This module evaluates the out-of-sample performance of fitted hedge models.
It leverages the internal backtest engine of BaseHedgeModel.
"""

import numpy as np
import pandas as pd

def evaluate_out_of_sample(full_data, train_end_idx, models_to_test, print_attributes=False):
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
    print_attributes : bool
        If True, fetches and prints model-specific attributes (e.g., DCC a/b, VECM lags).
        
    Returns:
    --------
    pd.DataFrame : A summary table of the results.
    """
    test_data = full_data.iloc[train_end_idx:]
    r_test_cny = test_data["r_CNY"].values
    var_unhedged = np.var(r_test_cny)

    # Dynamically size the terminal separator line based on the columns
    sep_len = 105 if print_attributes else 80

    print(f"\n{'='*sep_len}")
    print(f"Out-of-Sample Hedging Efficiency Comparison (N={len(test_data)})")
    print(f"{'='*80}")
    
    summary = {}

    for model in models_to_test:
        pnl = model.run_backtest(full_data, train_end_idx)
        h_info = model.get_hedge_info()
        
        var_hedged = np.var(pnl)
        he = 1 - (var_hedged / var_unhedged)
        
        if print_attributes:
            model_attrs = model.get_model_attributes()
            summary[model.name] = {
                "Hedge Ratio": h_info,
                "Model Attributes": model_attrs,
                "HE": he
            }
        else:
            summary[model.name] = {
                "Hedge Ratio": h_info,
                "HE": he
            }

    summary_df = pd.DataFrame.from_dict(summary, orient="index")
    
    if print_attributes:
        print(f"{'Strategy':<30} {'Hedge Ratio':>25} {'Model Attributes':>30} {'HE':>10}")
        print("-" * sep_len)
        for index, row in summary_df.iterrows():
            print(f"  {index:<28} {str(row['Hedge Ratio']):>25} {str(row['Model Attributes']):>30} {row['HE']:>10.4f}")
        print("-" * sep_len)
        print(f"  {'Unhedged Baseline':<28} {'None':>25} {'-':>30} {0.0:>10.4f}")
    else:
        print(f"{'Strategy':<30} {'Hedge Ratio':>30} {'HE':>10}")
        print("-" * sep_len)
        for index, row in summary_df.iterrows():
            print(f"  {index:<28} {str(row['Hedge Ratio']):>30} {row['HE']:>10.4f}")
        print("-" * sep_len)
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