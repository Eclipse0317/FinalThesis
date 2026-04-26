"""
src/sensitivity.py

Sensitivity analysis for PathSig hyperparameters (sig_window, sig_depth).
Tests all combinations and reports HE in a compact table.
"""

import numpy as np
import itertools
from src.models.path_sig import PathSigHedgeModel
from src.config import REFIT_STEP


def run_pathsig_sensitivity(full_data, train_split=0.7,
                            windows=[3, 4, 6, 8],
                            depths=[2, 3],
                            rolling_sizes=[104, 208]):
    """
    Tests PathSig across a grid of (sig_window, sig_depth) for
    static, rolling, and expanding variants.

    Parameters
    ----------
    full_data : pd.DataFrame
    train_split : float
    windows : list of int — signature trailing window lengths
    depths : list of int — signature truncation depths
    rolling_sizes : list of int — rolling window sizes to test
    """
    n_train = int(len(full_data) * train_split)
    test_data = full_data.iloc[n_train:]
    r_test_cny = test_data["r_CNY"].values
    var_unhedged = np.var(r_test_cny)

    print("\n" + "=" * 90)
    print(f"PathSig Sensitivity Analysis (N_test={len(test_data)}, train={train_split:.0%})")
    print("=" * 90)

    # ---- Static & Expanding ----
    print(f"\n{'Config':<20} {'Sig Dim':>8} {'Static HE':>12} {'Expanding HE':>14} {'λ_s':>8} {'λ_e':>8} {'β₀_s':>8} {'β₀_e':>8}")
    print("-" * 90)

    for W, D in itertools.product(windows, depths):
        sig_dim = _sig_dim_str(D)

        # Static
        m_static = PathSigHedgeModel(window=W, depth=D, window_type='static')
        pnl_s = m_static.run_backtest(full_data, n_train)
        he_s = 1 - np.var(pnl_s) / var_unhedged
        lam_s = m_static.ridge_model.alpha_
        b0_s = m_static.ridge_model.coef_[0]

        # Expanding
        m_exp = PathSigHedgeModel(window=W, depth=D, window_type='expanding', refit_step=REFIT_STEP)
        pnl_e = m_exp.run_backtest(full_data, n_train)
        he_e = 1 - np.var(pnl_e) / var_unhedged
        lam_e = m_exp.ridge_model.alpha_
        b0_e = m_exp.ridge_model.coef_[0]

        print(f"  W={W}, D={D:<10} {sig_dim:>8} {he_s:>12.4f} {he_e:>14.4f} {lam_s:>8.2f} {lam_e:>8.2f} {b0_s:>8.4f} {b0_e:>8.4f}")

    # ---- Rolling (separate table, tests different window sizes) ----
    print(f"\n{'Rolling Config':<28} {'Sig Dim':>8} {'HE':>10} {'Avg h':>10} {'λ':>8} {'β₀':>8}")
    print("-" * 80)

    for W, D in itertools.product(windows, depths):
        sig_dim = _sig_dim_str(D)
        for rs in rolling_sizes:
            m_roll = PathSigHedgeModel(window=W, depth=D, window_type='rolling',
                                       window_size=rs, refit_step=REFIT_STEP)
            pnl_r = m_roll.run_backtest(full_data, n_train)
            he_r = 1 - np.var(pnl_r) / var_unhedged
            lam_r = m_roll.ridge_model.alpha_
            b0_r = m_roll.ridge_model.coef_[0]
            avg_h = np.mean(m_roll.hedge_ratio_history) if m_roll.hedge_ratio_history else float('nan')

            print(f"  W={W}, D={D}, roll={rs:<8} {sig_dim:>8} {he_r:>10.4f} {avg_h:>10.4f} {lam_r:>8.2f} {b0_r:>8.4f}")


def _sig_dim_str(depth):
    """Returns the total number of signature features (incl. depth-0) for a 2D path."""
    import esig
    return str(esig.sigdim(2, depth))