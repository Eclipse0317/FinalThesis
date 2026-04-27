"""
run_scaler_comparison.py

Head-to-head comparison of PathSig with and without StandardScaler.
Uses expanded lambda grid to give Ridge full freedom.
"""

import warnings
from statsmodels.tools.sm_exceptions import ValueWarning
warnings.filterwarnings("ignore", category=ValueWarning)

import numpy as np
from src.data_loader import load_data
from src.models.path_sig import PathSigHedgeModel

# Expanded lambda grid
ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

if __name__ == "__main__":
    weekly_data = load_data()

    configs = [
        ("W=3, D=2", 3, 2),
    ]

    for split in [0.6, 0.7, 0.8]:
        n_train = int(len(weekly_data) * split)
        test_data = weekly_data.iloc[n_train:]
        var_unhedged = np.var(test_data["r_CNY"].values)

        print(f"\n{'='*95}")
        print(f"Scaler Comparison (Split={split:.0%}, N_test={len(test_data)})")
        print(f"{'='*95}")
        print(f"  {'Config':<22} {'Variant':<12} {'Window':<10} {'HE':>8} {'Avg h':>8} {'λ':>8} {'β₀':>8}")
        print(f"  {'-'*85}")

        for label, W, D in configs:
            for use_scaler in [False, True]:
                scaler_label = "Scaled" if use_scaler else "Raw"

                for wtype, wsize in [('static', None), ('rolling', 208), ('expanding', None)]:
                    refit = 4 if wtype != 'static' else 1

                    # Override RIDGE_ALPHAS temporarily via model
                    import src.config as cfg
                    old_alphas = cfg.RIDGE_ALPHAS
                    cfg.RIDGE_ALPHAS = ALPHAS

                    m = PathSigHedgeModel(
                        window=W, depth=D,
                        window_type=wtype,
                        window_size=wsize,
                        refit_step=refit,
                        use_scaler=use_scaler
                    )

                    pnl = m.run_backtest(weekly_data, n_train)
                    he = 1 - np.var(pnl) / var_unhedged
                    lam = m.ridge_model.alpha_
                    b0 = m.ridge_model.coef_[0]
                    avg_h = np.mean(m.hedge_ratio_history) if m.hedge_ratio_history else float('nan')

                    wlabel = f"{wtype}" + (f"({wsize})" if wsize else "")
                    print(f"  {label:<22} {scaler_label:<12} {wlabel:<10} {he:>8.4f} {avg_h:>8.4f} {lam:>8.3f} {b0:>8.4f}")

                    cfg.RIDGE_ALPHAS = old_alphas

                print()  # blank line between scaled/raw

    print("\nComparison Complete!")