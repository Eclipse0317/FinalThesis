from statsmodels.tsa.stattools import bds
from statsmodels.stats.diagnostic import het_arch

def run_residual_diagnostics(model_name, residuals_dict):
    """
    Runs BDS and ARCH-LM tests on a dictionary of residuals.
    Expects format: {'CNY': array_like, 'CNH': array_like}
    """
    print(f"\n{'='*60}")
    print(f"Residual Diagnostics for: {model_name}")
    print(f"{'='*60}")

    # 1. BDS Test
    print("\nBDS Non-Linearity Test (epsilon=1.0 std):")
    for name, resid in residuals_dict.items():
        # Standardize residuals for BDS
        s = (resid - resid.mean()) / resid.std()
        bds_stat, bds_pval = bds(s, max_dim=6, epsilon=1.0)
        
        print(f"  {name}:")
        for j, dim in enumerate(range(2, 7)):
            sig = "***" if bds_pval[j] < 0.01 else "**" if bds_pval[j] < 0.05 else ""
            print(f"    dim={dim}: stat={bds_stat[j]:.4f}, p={bds_pval[j]:.6f} {sig}")

    # 2. ARCH-LM Test
    print("\nARCH-LM Heteroskedasticity Test:")
    for name, resid in residuals_dict.items():
        for nlags in [1, 5, 10]:
            lm_stat, lm_p, _, _ = het_arch(resid, nlags=nlags)
            sig = "***" if lm_p < 0.01 else "**" if lm_p < 0.05 else ""
            print(f"  {name} lag={nlags}: LM={lm_stat:.4f}, p={lm_p:.6f} {sig}")