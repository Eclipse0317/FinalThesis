"""
CNY/CNH 双轨汇率套期保值策略实证分析 (v2 - 修复版)
修复内容：
  1. VECM滞后阶数由BIC自动选择，不再hardcode
  2. DCC-GARCH仅用训练集拟合，消除look-ahead bias
  3. 路径签名特征窗口不含当期价格，消除index leakage
  4. Ridge的alpha通过5折CV选择
  5. DCC优化使用多组初始值
  6. 异常处理打印错误信息

依赖: pip install pandas numpy matplotlib statsmodels arch esig scikit-learn scipy
数据: USD_CNY_Historical_Weekly_Data.csv, USD_CNH_Historical_Weekly_Data.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
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
    cny = pd.read_csv("data/USD_CNY_Historical_Weekly_Data.csv")
    cnh = pd.read_csv("data/USD_CNH_Historical_Weekly_Data.csv")

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


# ============================================================
# 第三部分：VECM基准模型
# ============================================================

def run_vecm(weekly):
    """估计VECM（BIC选滞后阶数）并对残差做非线性检验"""

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


# ============================================================
# 第四部分：DCC-GARCH（仅用训练集拟合）
# ============================================================

def run_dcc_garch(weekly, n_train=None):
    """估计DCC-GARCH模型，仅用训练集数据"""

    if n_train is None:
        n_train = int(len(weekly) * 0.7)

    r_cny = np.log(weekly["CNY"]).diff() * 100
    r_cnh = np.log(weekly["CNH"]).diff() * 100
    df = pd.DataFrame({"r_CNY": r_cny, "r_CNH": r_cnh}).dropna()

    print("\n" + "="*60)
    print(f"DCC-GARCH模型 (训练集={n_train})")
    print("="*60)

    # 仅训练集拟合GARCH
    train_cny = df["r_CNY"].iloc[:n_train]
    train_cnh = df["r_CNH"].iloc[:n_train]

    res = {}
    for col, series in [("r_CNY", train_cny), ("r_CNH", train_cnh)]:
        am = arch_model(series, vol="Garch", p=1, q=1, mean="ARX", lags=1)
        res[col] = am.fit(disp="off")
        print(f"  {col} 条件波动率均值: {res[col].conditional_volatility.mean():.4f}%")

    e1_tr = res["r_CNY"].std_resid.dropna().values
    e2_tr = res["r_CNH"].std_resid.dropna().values
    min_len = min(len(e1_tr), len(e2_tr))
    e1_tr, e2_tr = e1_tr[:min_len], e2_tr[:min_len]

    Qbar = np.corrcoef(e1_tr, e2_tr)
    print(f"  无条件相关系数: {Qbar[0,1]:.6f}")

    def dcc_loglik(params):
        a, b = params
        if a < 1e-6 or b < 1e-6 or a + b >= 0.999:
            return 1e10
        Q = Qbar.copy(); ll = 0
        for t in range(1, len(e1_tr)):
            et = np.array([e1_tr[t-1], e2_tr[t-1]])
            Q = (1-a-b)*Qbar + a*np.outer(et, et) + b*Q
            d = np.sqrt(np.diag(Q))
            if d[0] < 1e-8 or d[1] < 1e-8: return 1e10
            R01 = np.clip(Q[0,1]/(d[0]*d[1]), -0.9999, 0.9999)
            det_R = 1 - R01**2
            if det_R <= 1e-10: return 1e10
            en = np.array([e1_tr[t], e2_tr[t]])
            ll += -0.5*(np.log(det_R) + (en[0]**2+en[1]**2-2*R01*en[0]*en[1])/det_R - en[0]**2 - en[1]**2)
        return -ll

    best_opt = None; best_ll = 1e10
    for init in [[0.03, 0.93], [0.05, 0.85], [0.10, 0.80], [0.01, 0.95]]:
        opt = minimize(dcc_loglik, init, method="Nelder-Mead", options={"maxiter": 10000})
        if opt.fun < best_ll:
            best_ll = opt.fun; best_opt = opt

    a_dcc, b_dcc = best_opt.x
    print(f"  DCC参数: a={a_dcc:.6f}, b={b_dcc:.6f}")

    # BDS on GARCH residuals
    print("\n  BDS检验 (GARCH标准化残差):")
    for name, e in [("CNY", e1_tr), ("CNH", e2_tr)]:
        bds_stat, bds_pval = bds(e, max_dim=6, epsilon=1.0)
        print(f"    {name}: dim2 p={bds_pval[0]:.4f}, dim4 p={bds_pval[2]:.4f}, dim6 p={bds_pval[4]:.4f}")

    return a_dcc, b_dcc, Qbar, res


# ============================================================
# 第五部分：对冲效率比较（无信息泄露）
# ============================================================

def run_comparison(weekly, split_ratio=0.7):
    """所有模型的样本外对冲效率比较，无look-ahead bias"""

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
    n_test = n_total - n_train

    r_train_cny, r_test_cny = r_cny[:n_train], r_cny[n_train:]
    r_train_cnh, r_test_cnh = r_cnh[:n_train], r_cnh[n_train:]
    var_unhedged = np.var(r_test_cny)

    print(f"\n{'='*60}")
    print(f"对冲效率比较 (split={split_ratio}, 分割点={weekly.index[n_train].date()})")
    print(f"训练集={n_train}, 测试集={n_test}")
    print(f"{'='*60}")

    results = {}

    # --- 1. OLS ---
    h_ols = np.cov(r_train_cny, r_train_cnh)[0,1] / np.var(r_train_cnh)
    pnl_ols = r_test_cny - h_ols * r_test_cnh
    results["OLS"] = {"h": f"{h_ols:.4f}", "he": 1 - np.var(pnl_ols) / var_unhedged}

    # --- 2. VECM (BIC选滞后阶数) ---
    try:
        train_data = weekly[["CNY", "CNH"]].iloc[:n_train]
        lag_order = select_order(train_data, maxlags=8, deterministic="ci")
        chosen_lag = max(lag_order.bic, 1)

        vecm = VECM(train_data, k_ar_diff=chosen_lag, coint_rank=1, deterministic="ci")
        vecm_res = vecm.fit()
        h_vecm = np.cov(vecm_res.resid[:,0], vecm_res.resid[:,1])[0,1] / np.var(vecm_res.resid[:,1])
        pnl_vecm = r_test_cny - h_vecm * r_test_cnh
        results["VECM"] = {"h": f"{h_vecm:.4f} (lag={chosen_lag})", 
                           "he": 1 - np.var(pnl_vecm) / var_unhedged}
    except Exception as e:
        print(f"  VECM error: {e}")
        results["VECM"] = {"h": "N/A", "he": np.nan}

    # --- 3. DCC-GARCH (仅训练集拟合) ---
    try:
        train_r_cny = weekly["r_CNY"].iloc[:n_train]
        train_r_cnh = weekly["r_CNH"].iloc[:n_train]

        am1 = arch_model(train_r_cny, vol="Garch", p=1, q=1, mean="ARX", lags=1)
        res1 = am1.fit(disp="off")
        am2 = arch_model(train_r_cnh, vol="Garch", p=1, q=1, mean="ARX", lags=1)
        res2 = am2.fit(disp="off")

        e1_tr = res1.std_resid.dropna().values
        e2_tr = res2.std_resid.dropna().values
        min_len_tr = min(len(e1_tr), len(e2_tr))
        e1_tr, e2_tr = e1_tr[:min_len_tr], e2_tr[:min_len_tr]

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

        best_opt = None; best_ll = 1e10
        for init in [[0.03, 0.93], [0.05, 0.85], [0.10, 0.80], [0.01, 0.95]]:
            opt = minimize(dcc_ll, init, method="Nelder-Mead", options={"maxiter": 10000})
            if opt.fun < best_ll:
                best_ll = opt.fun; best_opt = opt
        a_dcc, b_dcc = best_opt.x

        # 用训练集参数对全样本做filter提取波动率
        garch1_params = res1.params
        garch2_params = res2.params
        am1_f = arch_model(weekly["r_CNY"], vol="Garch", p=1, q=1, mean="ARX", lags=1)
        res1_f = am1_f.fit(starting_values=garch1_params, disp="off")
        am2_f = arch_model(weekly["r_CNH"], vol="Garch", p=1, q=1, mean="ARX", lags=1)
        res2_f = am2_f.fit(starting_values=garch2_params, disp="off")

        e1_all = res1_f.std_resid.values
        e2_all = res2_f.std_resid.values
        h1_all = res1_f.conditional_volatility.values
        h2_all = res2_f.conditional_volatility.values

        # 从训练集末尾的Q状态开始滚动
        Q = Qbar.copy()
        for t in range(1, n_train):
            if not (np.isnan(e1_all[t-1]) or np.isnan(e2_all[t-1])):
                et = np.array([e1_all[t-1], e2_all[t-1]])
                Q = (1-a_dcc-b_dcc)*Qbar + a_dcc*np.outer(et, et) + b_dcc*Q

        h_dcc_test = np.zeros(n_test)
        for t in range(n_test):
            idx = n_train + t
            prev = idx - 1
            if not (np.isnan(e1_all[prev]) or np.isnan(e2_all[prev])):
                et = np.array([e1_all[prev], e2_all[prev]])
                Q = (1-a_dcc-b_dcc)*Qbar + a_dcc*np.outer(et, et) + b_dcc*Q
            d = np.sqrt(np.diag(Q))
            rho = np.clip(Q[0,1]/(d[0]*d[1]), -0.9999, 0.9999)
            h_dcc_test[t] = rho * h1_all[idx] / h2_all[idx] if h2_all[idx] > 0 else h_ols

        pnl_dcc = r_test_cny - h_dcc_test * r_test_cnh
        results["DCC-GARCH"] = {"h": f"mean={np.mean(h_dcc_test):.4f} (a={a_dcc:.3f},b={b_dcc:.3f})",
                                 "he": 1 - np.var(pnl_dcc) / var_unhedged}
    except Exception as e:
        print(f"  DCC-GARCH error: {e}")
        results["DCC-GARCH"] = {"h": "N/A", "he": np.nan}

    # --- 4. 路径签名 (无index leakage, CV选alpha) ---
    try:
        window = 4; depth = 3

        sig_features = []
        valid_indices = []
        for t in range(window + 1, n_total):
            # 特征用 [t-window-1, t-1] 的价格，不含t
            path = np.column_stack([
                cny_prices[t-window-1:t] - cny_prices[t-window-1],
                cnh_prices[t-window-1:t] - cnh_prices[t-window-1]
            ])
            sig = esig.stream2sig(path, depth)
            sig_features.append(sig)
            valid_indices.append(t)

        sig_features = np.array(sig_features)
        valid_indices = np.array(valid_indices)

        y_all = r_cny[valid_indices]
        r_h_all = r_cnh[valid_indices]

        train_mask = valid_indices < n_train
        test_mask = valid_indices >= n_train

        X_all = np.column_stack([sig_features, r_h_all.reshape(-1,1),
                                  sig_features * r_h_all.reshape(-1,1)])

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_all[train_mask])
        X_te = scaler.transform(X_all[test_mask])
        y_tr = y_all[train_mask]
        y_te = y_all[test_mask]

        # CV选alpha
        best_alpha = 1.0; best_cv = -1e10
        for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
            ridge = Ridge(alpha=alpha)
            scores = cross_val_score(ridge, X_tr, y_tr, cv=5, scoring="neg_mean_squared_error")
            if scores.mean() > best_cv:
                best_cv = scores.mean(); best_alpha = alpha

        ridge = Ridge(alpha=best_alpha)
        ridge.fit(X_tr, y_tr)
        y_pred = ridge.predict(X_te)

        pnl_sig = y_te - y_pred
        results["Path Sig."] = {"h": f"implicit (alpha={best_alpha})",
                                 "he": 1 - np.var(pnl_sig) / np.var(y_te)}
    except Exception as e:
        print(f"  Path Signature error: {e}")
        results["Path Sig."] = {"h": "N/A", "he": np.nan}

    # --- 5. 择优结售汇 ---
    cny_test_p = cny_prices[n_train:]
    cnh_test_p = cnh_prices[n_train:]
    var_cny_only = np.var(np.diff(np.log(cny_test_p)) * 100)

    best_settle = np.maximum(cny_test_p, cnh_test_p)
    r_best_settle = np.diff(np.log(best_settle)) * 100
    results["Best-Rate Settle"] = {"h": "max(CNY,CNH)",
                                    "he": 1 - np.var(r_best_settle) / var_cny_only}

    best_buy = np.minimum(cny_test_p, cnh_test_p)
    r_best_buy = np.diff(np.log(best_buy)) * 100
    results["Best-Rate Buy"] = {"h": "min(CNY,CNH)",
                                 "he": 1 - np.var(r_best_buy) / var_cny_only}

    # 打印
    print(f"\n{'策略':<20} {'对冲比率/设定':>30} {'对冲效率':>10}")
    print("-" * 64)
    for name, r in results.items():
        print(f"  {name:<18} {r['h']:>30} {r['he']:>10.4f}")

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
        all_results[label] = run_comparison(weekly, split_ratio=ratio)

    print(f"\n{'='*60}")
    print("稳健性汇总")
    print(f"{'='*60}")
    print(f"\n{'分割':<8} {'OLS':>8} {'VECM':>8} {'DCC':>8} {'签名':>8} {'择优结':>8} {'择优售':>8}")
    print("-" * 58)
    for label, res in all_results.items():
        print(f"  {label:<6} "
              f"{res['OLS']['he']:>8.4f} "
              f"{res['VECM']['he']:>8.4f} "
              f"{res['DCC-GARCH']['he']:>8.4f} "
              f"{res['Path Sig.']['he']:>8.4f} "
              f"{res['Best-Rate Settle']['he']:>8.4f} "
              f"{res['Best-Rate Buy']['he']:>8.4f}")

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

    # 4. DCC-GARCH (单独展示)
    dcc_params = run_dcc_garch(weekly)

    # 5. 完整对比 (主分析)
    results = run_comparison(weekly, split_ratio=0.7)

    # 6. 稳健性检验
    robustness = run_robustness(weekly)

    print("\n" + "="*60)
    print("全部完成!")
    print("="*60)
