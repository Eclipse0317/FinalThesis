from statsmodels.tsa.vector_ar.vecm import VECM, select_order, select_coint_rank
from statsmodels.stats.diagnostic import bds, het_arch

def run_vecm(weekly):
    """估计VECM并对残差做非线性检验"""

    data = weekly[["CNY", "CNH"]].dropna()

    print("\n" + "="*60)
    print("VECM模型估计")
    print("="*60)

    lag_order = select_order(data, maxlags=12, deterministic="ci")
    chosen_lag = max(lag_order.bic, 1)
    print(f"AIC选择: {lag_order.aic}阶, BIC选择: {lag_order.bic}阶, 采用: {chosen_lag}阶")

    rank_test = select_coint_rank(data, det_order=0, k_ar_diff=chosen_lag)
    print(rank_test.summary())

    model = VECM(data, k_ar_diff=chosen_lag, coint_rank=1, deterministic="ci")
    result = model.fit()

    alpha = result.alpha
    beta = result.beta
    print(f"\n协整向量 beta: {beta.flatten()}")
    print(f"调整系数 alpha_CNY: {alpha[0,0]:.6f}")
    print(f"调整系数 alpha_CNH: {alpha[1,0]:.6f}")

    resid = result.resid

    # BDS检验
    print("\nBDS非线性检验 (VECM残差, epsilon=1.0 std):")
    for i, name in enumerate(["CNY", "CNH"]):
        s = (resid[:,i] - resid[:,i].mean()) / resid[:,i].std()
        bds_stat, bds_pval = bds(s, max_dim=6, epsilon=1.0)
        print(f"  {name}:")
        for j, dim in enumerate(range(2, 7)):
            sig = "***" if bds_pval[j] < 0.01 else "**" if bds_pval[j] < 0.05 else ""
            print(f"    dim={dim}: stat={bds_stat[j]:.4f}, p={bds_pval[j]:.6f} {sig}")

    # ARCH-LM检验
    print("\nARCH-LM检验 (VECM残差):")
    for i, name in enumerate(["CNY", "CNH"]):
        for nlags in [5, 10]:
            lm_stat, lm_p, _, _ = het_arch(resid[:,i], nlags=nlags)
            sig = "***" if lm_p < 0.01 else "**" if lm_p < 0.05 else ""
            print(f"  {name} lag={nlags}: LM={lm_stat:.4f}, p={lm_p:.6f} {sig}")

    return result