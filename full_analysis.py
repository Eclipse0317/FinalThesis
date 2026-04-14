"""
CNY/CNH 双轨汇率套期保值策略实证分析
完整代码：从数据加载到最终结果

依赖: pip install pandas numpy matplotlib statsmodels arch esig scikit-learn scipy
数据: 需要以下CSV文件放在同目录下
  - USD_CNY_Historical_Weekly_Data.csv
  - USD_CNH_Historical_Weekly_Data.csv
  (从investing.com下载)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from scipy import stats as sp_stats
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller, kpss, coint, bds
from statsmodels.tsa.vector_ar.vecm import VECM, select_order, select_coint_rank
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from arch import arch_model
import esig
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


# ============================================================
# 第一部分：数据加载与清洗
# ============================================================

def load_data():
    """加载并清洗investing.com的周度数据"""
    cny = pd.read_csv("USD_CNY_Historical_Weekly_Data.csv")
    cnh = pd.read_csv("USD_CNH_Historical_Weekly_Data.csv")
    
    for df in [cny, cnh]:
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", inplace=True)
        df.set_index("Date", inplace=True)
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    
    weekly = cny[["Price"]].rename(columns={"Price": "CNY"}).join(
        cnh[["Price"]].rename(columns={"Price": "CNH"}), how="inner"
    )
    
    print(f"数据范围: {weekly.index[0].date()} 到 {weekly.index[-1].date()}")
    print(f"观测数: {len(weekly)}")
    print(f"缺失值: CNY={weekly['CNY'].isna().sum()}, CNH={weekly['CNH'].isna().sum()}")
    
    return weekly


# ============================================================
# 第二部分：探索性数据分析 (EDA)
# ============================================================

def run_eda(weekly):
    """完整的EDA：描述性统计、正态性、平稳性、协整检验"""
    
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


# ============================================================
# 第三部分：VECM基准模型
# ============================================================

def run_vecm(weekly):
    """估计VECM并对残差做非线性检验"""
    
    data = weekly[["CNY", "CNH"]].dropna()
    
    print("\n" + "="*60)
    print("VECM模型估计")
    print("="*60)
    
    # 滞后阶数
    lag_order = select_order(data, maxlags=12, deterministic="ci")
    print(f"AIC选择: {lag_order.aic}阶, BIC选择: {lag_order.bic}阶")
    
    # Johansen检验
    rank_test = select_coint_rank(data, det_order=0, k_ar_diff=1)
    print(rank_test.summary())
    
    # 估计VECM(1)
    model = VECM(data, k_ar_diff=1, coint_rank=1, deterministic="ci")
    result = model.fit()
    
    alpha = result.alpha
    beta = result.beta
    print(f"\n协整向量 beta: {beta.flatten()}")
    print(f"调整系数 alpha_CNY: {alpha[0,0]:.6f}")
    print(f"调整系数 alpha_CNH: {alpha[1,0]:.6f}")
    
    # 残差
    resid = result.resid
    
    # BDS检验
    print("\n" + "="*60)
    print("BDS非线性检验 (VECM残差, epsilon=1.0 std)")
    print("="*60)
    for i, name in enumerate(["CNY", "CNH"]):
        s = (resid[:,i] - resid[:,i].mean()) / resid[:,i].std()
        bds_stat, bds_pval = bds(s, max_dim=6, epsilon=1.0)
        print(f"\n  {name}残差:")
        for j, dim in enumerate(range(2, 7)):
            sig = "***" if bds_pval[j] < 0.01 else "**" if bds_pval[j] < 0.05 else ""
            print(f"    dim={dim}: stat={bds_stat[j]:.4f}, p={bds_pval[j]:.6f} {sig}")
    
    # ARCH-LM检验
    print("\n" + "="*60)
    print("ARCH-LM检验 (VECM残差)")
    print("="*60)
    for i, name in enumerate(["CNY", "CNH"]):
        for nlags in [5, 10]:
            lm_stat, lm_p, _, _ = het_arch(resid[:,i], nlags=nlags)
            sig = "***" if lm_p < 0.01 else "**" if lm_p < 0.05 else ""
            print(f"  {name} lag={nlags}: LM={lm_stat:.4f}, p={lm_p:.6f} {sig}")
    
    return result


# ============================================================
# 第四部分：DCC-GARCH
# ============================================================

def run_dcc_garch(weekly):
    """估计DCC-GARCH模型"""
    
    r_cny = np.log(weekly["CNY"]).diff() * 100
    r_cnh = np.log(weekly["CNH"]).diff() * 100
    df = pd.DataFrame({"r_CNY": r_cny, "r_CNH": r_cnh}).dropna()
    
    print("\n" + "="*60)
    print("DCC-GARCH模型")
    print("="*60)
    
    # 单变量GARCH
    res = {}
    for col in ["r_CNY", "r_CNH"]:
        am = arch_model(df[col], vol="Garch", p=1, q=1, mean="ARX", lags=1)
        res[col] = am.fit(disp="off")
        print(f"\n{col} GARCH(1,1):")
        print(f"  条件波动率均值: {res[col].conditional_volatility.mean():.4f}%")
    
    # 标准化残差
    e1 = res["r_CNY"].std_resid.values
    e2 = res["r_CNH"].std_resid.values
    h1 = res["r_CNY"].conditional_volatility.values
    h2 = res["r_CNH"].conditional_volatility.values
    
    valid = ~(np.isnan(e1) | np.isnan(e2))
    e1, e2, h1, h2 = e1[valid], e2[valid], h1[valid], h2[valid]
    dates = df.index[valid]
    
    Qbar = np.corrcoef(e1, e2)
    
    # DCC估计
    def dcc_loglik(params):
        a, b = params
        if a < 1e-6 or b < 1e-6 or a + b >= 0.999:
            return 1e10
        Q = Qbar.copy()
        ll = 0
        for t in range(1, len(e1)):
            et = np.array([e1[t-1], e2[t-1]])
            Q = (1-a-b)*Qbar + a*np.outer(et, et) + b*Q
            d = np.sqrt(np.diag(Q))
            if d[0] < 1e-8 or d[1] < 1e-8: return 1e10
            R01 = np.clip(Q[0,1]/(d[0]*d[1]), -0.9999, 0.9999)
            det_R = 1 - R01**2
            if det_R <= 1e-10: return 1e10
            e_now = np.array([e1[t], e2[t]])
            quad = (e_now[0]**2 + e_now[1]**2 - 2*R01*e_now[0]*e_now[1]) / det_R
            ll += -0.5 * (np.log(det_R) + quad - e_now[0]**2 - e_now[1]**2)
        return -ll
    
    opt = minimize(dcc_loglik, [0.03, 0.93], method="Nelder-Mead",
                   options={"maxiter": 10000})
    a_dcc, b_dcc = opt.x
    print(f"\nDCC参数: a={a_dcc:.6f}, b={b_dcc:.6f}, a+b={a_dcc+b_dcc:.6f}")
    
    # 提取时变相关和对冲比率
    Q = Qbar.copy()
    rho_t = np.zeros(len(e1))
    hedge_dcc = np.zeros(len(e1))
    
    for t in range(len(e1)):
        if t > 0:
            et = np.array([e1[t-1], e2[t-1]])
            Q = (1-a_dcc-b_dcc)*Qbar + a_dcc*np.outer(et, et) + b_dcc*Q
        d = np.sqrt(np.diag(Q))
        rho_t[t] = np.clip(Q[0,1]/(d[0]*d[1]), -0.9999, 0.9999)
        hedge_dcc[t] = rho_t[t] * h1[t] / h2[t] if h2[t] > 0 else 0.75
    
    print(f"时变相关系数: 均值={rho_t.mean():.4f}, 范围=[{rho_t.min():.4f}, {rho_t.max():.4f}]")
    print(f"对冲比率: 均值={hedge_dcc.mean():.4f}, 范围=[{hedge_dcc.min():.4f}, {hedge_dcc.max():.4f}]")
    
    # BDS on GARCH residuals
    print("\nBDS检验 (GARCH标准化残差, epsilon=1.0 std):")
    for name, e in [("CNY", e1), ("CNH", e2)]:
        bds_stat, bds_pval = bds(e, max_dim=6, epsilon=1.0)
        print(f"  {name}: dim2 p={bds_pval[0]:.4f}, dim4 p={bds_pval[2]:.4f}, dim6 p={bds_pval[4]:.4f}")
    
    # 图表
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    axes[0].plot(dates, h1, label="CNY vol", linewidth=0.7)
    axes[0].plot(dates, h2, label="CNH vol", linewidth=0.7)
    axes[0].set_title("GARCH(1,1) Conditional Volatility (Weekly %)"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(dates, rho_t, color="purple", linewidth=0.7)
    axes[1].set_title("DCC Time-Varying Correlation"); axes[1].grid(True, alpha=0.3)
    axes[2].plot(dates, hedge_dcc, color="darkgreen", linewidth=0.7)
    axes[2].set_title("DCC-GARCH Dynamic Hedge Ratio"); axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("dcc_garch_plots.png", dpi=150)
    plt.close()
    print("图表已保存: dcc_garch_plots.png")
    
    return hedge_dcc, dates


# ============================================================
# 第五部分：完整对冲效率比较（含路径签名和择优结售汇）
# ============================================================

def run_comparison(weekly, split_ratio=0.7):
    """运行所有模型的样本外对冲效率比较"""
    
    weekly = weekly.copy()
    weekly["r_CNY"] = np.log(weekly["CNY"]).diff() * 100
    weekly["r_CNH"] = np.log(weekly["CNH"]).diff() * 100
    weekly = weekly.dropna()
    
    cny_prices = weekly["CNY"].values
    cnh_prices = weekly["CNH"].values
    r_cny = weekly["r_CNY"].values
    r_cnh = weekly["r_CNH"].values
    n_total = len(weekly)
    n_train = int(n_total * split_ratio)
    
    r_train_cny, r_test_cny = r_cny[:n_train], r_cny[n_train:]
    r_train_cnh, r_test_cnh = r_cnh[:n_train], r_cnh[n_train:]
    n_test = len(r_test_cny)
    var_unhedged = np.var(r_test_cny)
    
    print(f"\n{'='*60}")
    print(f"对冲效率比较 (split={split_ratio}, 分割点={weekly.index[n_train].date()})")
    print(f"训练集={n_train}, 测试集={n_test}")
    print(f"{'='*60}")
    
    results = {}
    
    # --- 1. OLS ---
    h_ols = np.cov(r_train_cny, r_train_cnh)[0,1] / np.var(r_train_cnh)
    pnl_ols = r_test_cny - h_ols * r_test_cnh
    results["OLS"] = {"h": h_ols, "var": np.var(pnl_ols), "he": 1 - np.var(pnl_ols)/var_unhedged}
    
    # --- 2. VECM ---
    try:
        train_data = weekly[["CNY", "CNH"]].iloc[:n_train]
        vecm = VECM(train_data, k_ar_diff=1, coint_rank=1, deterministic="ci")
        vecm_res = vecm.fit()
        h_vecm = np.cov(vecm_res.resid[:,0], vecm_res.resid[:,1])[0,1] / np.var(vecm_res.resid[:,1])
        pnl_vecm = r_test_cny - h_vecm * r_test_cnh
        results["VECM"] = {"h": h_vecm, "var": np.var(pnl_vecm), "he": 1 - np.var(pnl_vecm)/var_unhedged}
    except:
        results["VECM"] = {"h": np.nan, "var": np.nan, "he": np.nan}
    
    # --- 3. DCC-GARCH ---
    try:
        am1 = arch_model(weekly["r_CNY"], vol="Garch", p=1, q=1, mean="ARX", lags=1)
        res1 = am1.fit(disp="off")
        am2 = arch_model(weekly["r_CNH"], vol="Garch", p=1, q=1, mean="ARX", lags=1)
        res2 = am2.fit(disp="off")
        
        e1 = res1.std_resid.values; e2 = res2.std_resid.values
        h1 = res1.conditional_volatility.values; h2 = res2.conditional_volatility.values
        valid = ~(np.isnan(e1) | np.isnan(e2))
        
        # Train DCC on training set only
        e1_tr = e1[:n_train][valid[:n_train]]
        e2_tr = e2[:n_train][valid[:n_train]]
        Qbar = np.corrcoef(e1_tr, e2_tr)
        
        def dcc_ll(params):
            a, b = params
            if a < 1e-6 or b < 1e-6 or a + b >= 0.999: return 1e10
            Q = Qbar.copy(); ll = 0
            for t in range(1, len(e1_tr)):
                et = np.array([e1_tr[t-1], e2_tr[t-1]])
                Q = (1-a-b)*Qbar + a*np.outer(et, et) + b*Q
                d = np.sqrt(np.diag(Q))
                if d[0]<1e-8 or d[1]<1e-8: return 1e10
                R01 = np.clip(Q[0,1]/(d[0]*d[1]), -0.9999, 0.9999)
                det_R = 1 - R01**2
                if det_R <= 1e-10: return 1e10
                en = np.array([e1_tr[t], e2_tr[t]])
                ll += -0.5*(np.log(det_R) + (en[0]**2+en[1]**2-2*R01*en[0]*en[1])/det_R - en[0]**2 - en[1]**2)
            return -ll
        
        opt = minimize(dcc_ll, [0.03, 0.93], method="Nelder-Mead", options={"maxiter": 10000})
        a_dcc, b_dcc = opt.x
        
        Q = Qbar.copy()
        h_dcc_all = np.zeros(n_total)
        for t in range(n_total):
            if not valid[t]: h_dcc_all[t] = h_ols; continue
            if t > 0 and valid[t-1]:
                et = np.array([e1[t-1], e2[t-1]])
                Q = (1-a_dcc-b_dcc)*Qbar + a_dcc*np.outer(et, et) + b_dcc*Q
            d = np.sqrt(np.diag(Q))
            rho = np.clip(Q[0,1]/(d[0]*d[1]), -0.9999, 0.9999)
            h_dcc_all[t] = rho * h1[t] / h2[t] if h2[t] > 0 else h_ols
        
        h_dcc_test = h_dcc_all[n_train:]
        pnl_dcc = r_test_cny - h_dcc_test * r_test_cnh
        results["DCC-GARCH"] = {"h": np.mean(h_dcc_test), "var": np.var(pnl_dcc), 
                                 "he": 1 - np.var(pnl_dcc)/var_unhedged}
    except:
        results["DCC-GARCH"] = {"h": np.nan, "var": np.nan, "he": np.nan}
    
    # --- 4. 路径签名 ---
    try:
        window = 4; depth = 3
        
        sig_features = []
        for t in range(window, n_total):
            path = np.column_stack([
                cny_prices[t-window:t+1] - cny_prices[t-window],
                cnh_prices[t-window:t+1] - cnh_prices[t-window]
            ])
            sig = esig.stream2sig(path, depth)
            sig_features.append(sig)
        
        sig_features = np.array(sig_features)
        y_sig = r_cny[window:]
        r_h_sig = r_cnh[window:]
        
        n_tr_sig = n_train - window
        X_all = np.column_stack([sig_features, r_h_sig.reshape(-1,1),
                                  sig_features * r_h_sig.reshape(-1,1)])
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_all[:n_tr_sig])
        X_te = scaler.transform(X_all[n_tr_sig:])
        y_tr = y_sig[:n_tr_sig]
        y_te = y_sig[n_tr_sig:]
        
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_tr, y_tr)
        y_pred = ridge.predict(X_te)
        
        pnl_sig = y_te - y_pred
        results["Path Signature"] = {"h": "implicit", "var": np.var(pnl_sig),
                                      "he": 1 - np.var(pnl_sig)/np.var(y_te)}
    except:
        results["Path Signature"] = {"h": np.nan, "var": np.nan, "he": np.nan}
    
    # --- 5. 择优结售汇 ---
    cny_test = cny_prices[n_train:]
    cnh_test = cnh_prices[n_train:]
    
    best_settle = np.maximum(cny_test, cnh_test)
    r_best_settle = np.diff(np.log(best_settle)) * 100
    var_cny_only = np.var(np.diff(np.log(cny_test)) * 100)
    results["Best-Rate Settle"] = {"h": "择优", "var": np.var(r_best_settle),
                                    "he": 1 - np.var(r_best_settle)/var_cny_only}
    
    best_buy = np.minimum(cny_test, cnh_test)
    r_best_buy = np.diff(np.log(best_buy)) * 100
    results["Best-Rate Buy"] = {"h": "择优", "var": np.var(r_best_buy),
                                 "he": 1 - np.var(r_best_buy)/var_cny_only}
    
    # 打印结果
    print(f"\n{'策略':<20} {'对冲比率':>10} {'组合方差':>12} {'对冲效率':>10}")
    print("-" * 56)
    for name, r in results.items():
        h_str = f"{r['h']:.4f}" if isinstance(r['h'], float) else str(r['h'])
        print(f"  {name:<18} {h_str:>10} {r['var']:>12.6f} {r['he']:>10.4f}")
    
    return results


# ============================================================
# 第六部分：稳健性检验
# ============================================================

def run_robustness(weekly):
    """三种分割方案的稳健性检验"""
    
    print("\n" + "="*60)
    print("稳健性检验")
    print("="*60)
    
    all_results = {}
    for ratio, label in [(0.7, "70/30"), (0.6, "60/40"), (0.8, "80/20")]:
        print(f"\n--- {label} ---")
        all_results[label] = run_comparison(weekly, split_ratio=ratio)
    
    # 汇总表
    print(f"\n{'='*60}")
    print("稳健性汇总")
    print(f"{'='*60}")
    print(f"\n{'分割':<8} {'OLS':>8} {'VECM':>8} {'DCC':>8} {'签名':>8}")
    print("-" * 40)
    for label, res in all_results.items():
        print(f"  {label:<6} {res['OLS']['he']:>8.4f} {res['VECM']['he']:>8.4f} "
              f"{res['DCC-GARCH']['he']:>8.4f} {res['Path Signature']['he']:>8.4f}")
    
    return all_results


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    # 1. 加载数据
    weekly = load_data()
    
    # 2. EDA
    weekly = run_eda(weekly)
    
    # 3. VECM
    vecm_result = run_vecm(weekly)
    
    # 4. DCC-GARCH
    hedge_dcc, dcc_dates = run_dcc_garch(weekly)
    
    # 5. 完整对比
    results = run_comparison(weekly, split_ratio=0.7)
    
    # 6. 稳健性检验
    robustness = run_robustness(weekly)
    
    print("\n" + "="*60)
    print("全部完成!")
    print("="*60)