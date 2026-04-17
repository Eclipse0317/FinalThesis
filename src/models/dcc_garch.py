import numpy as np
import pandas as pd
import warnings

# Suppress rpy2 warnings that clutter the terminal
warnings.filterwarnings("ignore", category=UserWarning, module="rpy2")

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.packages import importr

# Activate automatic conversions between Pandas/Numpy and R DataFrames/Matrices
pandas2ri.activate()
numpy2ri.activate()

# Import necessary R packages
try:
    rmgarch = importr("rmgarch")
    rugarch = importr("rugarch")
    base = importr("base")
except Exception as e:
    raise ImportError("Failed to import R packages. Ensure 'rugarch' and 'rmgarch' are installed in your R environment.") from e


def run_dcc_garch_rmgarch(weekly, n_train=None):
    """
    Estimate DCC(1,1)-GARCH(1,1) on CNY and CNH log returns using R's rmgarch.
    """
    r = np.log(weekly[["CNY", "CNH"]]).diff().dropna() * 100
    r.columns = ["r_CNY", "r_CNH"]
    
    if n_train is None:
        n_train = len(r)
    r_train = r.iloc[:n_train]
    
    # Push data to R global environment
    ro.globalenv["returns"] = pandas2ri.py2rpy(r_train)
    
    ro.r("""
        uspec <- ugarchspec(
            variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
            mean.model     = list(armaOrder = c(0, 0), include.mean = TRUE),
            distribution.model = "norm"
        )
        mspec <- multispec(replicate(2, uspec))
        
        dcc_spec <- dccspec(
            uspec    = mspec,
            dccOrder = c(1, 1),
            distribution = "mvnorm"
        )
        
        fit <- dccfit(dcc_spec, data = returns, fit.control = list(eval.se = TRUE))
    """)
    
    fit = ro.r("fit")
    
    coefs = dict(zip(list(ro.r("names(coef(fit))")), list(ro.r("coef(fit)"))))
    se_matrix = np.array(ro.r("fit@mfit$matcoef"))
    se_names  = list(ro.r("rownames(fit@mfit$matcoef)"))
    ses = dict(zip(se_names, se_matrix[:, 1]))
    
    R_array = np.array(ro.r("rcor(fit)"))
    rho_series = pd.Series(R_array[0, 1, :], index=r_train.index, name="rho_CNY_CNH")
    
    sigma = np.array(ro.r("sigma(fit)"))
    sigma_df = pd.DataFrame(sigma, index=r_train.index, columns=["sigma_CNY", "sigma_CNH"])
    
    loglik = float(ro.r("likelihood(fit)")[0])
    ic_vals = list(ro.r("infocriteria(fit)"))
    ic = dict(zip(["Akaike", "Bayes", "Shibata", "Hannan-Quinn"], 
                  [float(np.asarray(v).item()) for v in ic_vals]))
    
    out = {
        "coefficients": coefs,
        "std_errors": ses,
        "conditional_correlation": rho_series,
        "conditional_volatility": sigma_df,
        "loglik": loglik,
        "info_criteria": ic,
    }
    
    # Clean up R environment to prevent memory leaks
    ro.r("rm(returns, uspec, mspec, dcc_spec, fit)")
    return out


def dcc_hedging_efficiency(weekly, n_train, refit_every=20):
    """
    滚动计算样本外对冲效率 (using rmgarch dccroll)
    """
    r = np.log(weekly[["CNY", "CNH"]]).diff().dropna() * 100
    r.columns = ["r_CNY", "r_CNH"]

    ro.globalenv["returns"] = pandas2ri.py2rpy(r)
    ro.globalenv["n_train"] = n_train
    ro.globalenv["refit_every"] = refit_every

    ro.r("""
        uspec <- ugarchspec(
            variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
            mean.model     = list(armaOrder = c(0, 0), include.mean = TRUE),
            distribution.model = "norm"
        )
        mspec <- multispec(replicate(2, uspec))

        dcc_spec <- dccspec(
            uspec = mspec,
            dccOrder = c(1, 1),
            distribution = "mvnorm"
        )

        roll <- dccroll(
            dcc_spec,
            data = returns,
            n.ahead = 1,
            forecast.length = nrow(returns) - n_train,
            refit.every = refit_every,
            refit.window = "moving",
            fit.control = list(eval.se = FALSE)
        )
        
        fcst_cov <- rcov(roll)
        n_fcst <- dim(fcst_cov)[3]
        
        h_dcc <- rep(0, n_fcst)
        for (i in 1:n_fcst) {
            h_dcc[i] <- fcst_cov[1, 2, i] / fcst_cov[2, 2, i]
        }
    """)

    h_dcc = np.array(ro.r("h_dcc"))

    r_test_cny = r["r_CNY"].iloc[n_train:].values
    r_test_cnh = r["r_CNH"].iloc[n_train:].values

    min_len = min(len(h_dcc), len(r_test_cny))
    h_dcc = h_dcc[:min_len]
    r_test_cny = r_test_cny[:min_len]
    r_test_cnh = r_test_cnh[:min_len]

    pnl_unhedged = r_test_cny
    pnl_dcc = r_test_cny - h_dcc * r_test_cnh
    
    var_unhedged = np.var(pnl_unhedged)
    he_dcc = 1 - np.var(pnl_dcc) / var_unhedged
    
    # OLS Baseline
    r_train_cny = r["r_CNY"].iloc[:n_train].values
    r_train_cnh = r["r_CNH"].iloc[:n_train].values
    h_ols = np.cov(r_train_cny, r_train_cnh)[0, 1] / np.var(r_train_cnh)
    he_ols = 1 - np.var(r_test_cny - h_ols * r_test_cnh) / var_unhedged

    # Clean up R environment
    ro.r("rm(returns, uspec, mspec, dcc_spec, roll, fcst_cov, h_dcc)")

    return {
        "he_ols": he_ols,
        "he_dcc": he_dcc,
        "h_ols": h_ols,
        "h_dcc": h_dcc,
        "pnl_dcc": pnl_dcc, 
    }