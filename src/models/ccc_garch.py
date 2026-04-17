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

# Import necessary R packages safely
try:
    rmgarch = importr("rmgarch")
    rugarch = importr("rugarch")
    base = importr("base")
except Exception as e:
    raise ImportError("Failed to import R packages. Ensure 'rugarch' and 'rmgarch' are installed in your R environment.") from e


def run_ccc_garch(weekly, n_train=None):
    """
    Estimate CCC-GARCH (Constant Conditional Correlation) on CNY and CNH log returns.
    """
    r = np.log(weekly[["CNY", "CNH"]]).diff().dropna() * 100
    r.columns = ["r_CNY", "r_CNH"]

    if n_train is None:
        n_train = len(r)
    r_train = r.iloc[:n_train]

    ro.globalenv["returns"] = pandas2ri.py2rpy(r_train)

    ro.r("""
        uspec <- ugarchspec(
            variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
            mean.model     = list(armaOrder = c(0, 0), include.mean = TRUE),
            distribution.model = "norm"
        )
        mspec <- multispec(replicate(2, uspec))

        ccc_spec <- dccspec(
            uspec = mspec,
            dccOrder = c(1, 1),
            model = "CCC",
            distribution = "mvnorm"
        )

        ccc_fit <- dccfit(ccc_spec, data = returns, fit.control = list(eval.se = TRUE))
    """)

    coefs = dict(zip(
        list(ro.r("names(coef(ccc_fit))")),
        list(ro.r("coef(ccc_fit)"))
    ))

    R_array = np.array(ro.r("rcor(ccc_fit)"))
    rho_constant = R_array[0, 1, 0]  # Constant across all t in CCC

    sigma = np.array(ro.r("sigma(ccc_fit)"))
    sigma_df = pd.DataFrame(sigma, index=r_train.index, columns=["sigma_CNY", "sigma_CNH"])

    loglik = float(ro.r("likelihood(ccc_fit)")[0])
    ic_vals = list(ro.r("infocriteria(ccc_fit)"))
    ic = dict(zip(
        ["Akaike", "Bayes", "Shibata", "Hannan-Quinn"],
        [float(np.asarray(v).item()) for v in ic_vals]
    ))

    # Clean up R environment
    ro.r("rm(returns, uspec, mspec, ccc_spec, ccc_fit)")

    return {
        "coefficients": coefs,
        "rho": rho_constant,
        "conditional_volatility": sigma_df,
        "loglik": loglik,
        "info_criteria": ic,
    }


def ccc_hedging_efficiency(weekly, n_train, refit_every=20):
    """
    Calculate out-of-sample hedging efficiency using rolling CCC-GARCH.
    """
    r = np.log(weekly[["CNY", "CNH"]]).diff().dropna() * 100
    r.columns = ["r_CNY", "r_CNH"]
    
    ro.globalenv["returns"] = pandas2ri.py2rpy(r)
    ro.globalenv["n_train"] = n_train
    ro.globalenv["refit_every"] = refit_every

    # CCC roll estimation
    ro.r("""
        uspec <- ugarchspec(
            variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
            mean.model     = list(armaOrder = c(0, 0), include.mean = TRUE),
            distribution.model = "norm"
        )
        mspec <- multispec(replicate(2, uspec))

        ccc_spec <- dccspec(
            uspec = mspec,
            dccOrder = c(1, 1),
            model = "CCC",
            distribution = "mvnorm"
        )

        ccc_roll <- dccroll(
            ccc_spec,
            data = returns,
            n.ahead = 1,
            forecast.length = nrow(returns) - n_train,
            refit.every = refit_every,
            refit.window = "moving",
            fit.control = list(eval.se = FALSE)
        )

        ccc_cov <- rcov(ccc_roll)
        n_fcst <- dim(ccc_cov)[3]
        h_ccc <- rep(0, n_fcst)
        for (i in 1:n_fcst) {
            h_ccc[i] <- ccc_cov[1, 2, i] / ccc_cov[2, 2, i]
        }
    """)

    h_ccc = np.array(ro.r("h_ccc"))

    # Test set returns
    r_test_cny = r["r_CNY"].iloc[n_train:].values
    r_test_cnh = r["r_CNH"].iloc[n_train:].values

    min_len = min(len(h_ccc), len(r_test_cny))
    h_ccc = h_ccc[:min_len]
    r_test_cny = r_test_cny[:min_len]
    r_test_cnh = r_test_cnh[:min_len]

    var_unhedged = np.var(r_test_cny)

    # OLS Baseline
    r_train_cny = r["r_CNY"].iloc[:n_train].values
    r_train_cnh = r["r_CNH"].iloc[:n_train].values
    h_ols = np.cov(r_train_cny, r_train_cnh)[0, 1] / np.var(r_train_cnh)
    pnl_ols = r_test_cny - h_ols * r_test_cnh
    he_ols = 1 - np.var(pnl_ols) / var_unhedged

    # CCC Hedging Efficiency
    pnl_ccc = r_test_cny - h_ccc * r_test_cnh
    he_ccc = 1 - np.var(pnl_ccc) / var_unhedged

    # Clean up R environment
    ro.r("rm(returns, uspec, mspec, ccc_spec, ccc_roll, ccc_cov, h_ccc)")

    return {
        "he_ols": he_ols, 
        "he_ccc": he_ccc,
        "h_ols": h_ols, 
        "h_ccc": h_ccc,
        "pnl_ccc": pnl_ccc
    }