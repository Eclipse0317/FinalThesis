import numpy as np
import pandas as pd
import warnings

# Suppress rpy2 warnings that clutter the terminal
warnings.filterwarnings("ignore", category=UserWarning, module="rpy2")

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.packages import importr
from .base import BaseHedgeModel  

# Activate automatic conversions
pandas2ri.activate()
numpy2ri.activate()

try:
    rmgarch = importr("rmgarch")
    rugarch = importr("rugarch")
    base = importr("base")
except Exception as e:
    raise ImportError("Failed to import R packages. Ensure 'rugarch' and 'rmgarch' are installed in R.") from e


class BaseMGARCHModel(BaseHedgeModel):
    """
    A parent class for R-based MGARCH models. 
    It leverages the Python-side BaseHedgeModel for rolling windows, 
    using R purely for fitting (dccfit) and forecasting (dccforecast).
    """
    def __init__(self, name, mgarch_type, window_type='static', window_size=None, refit_step=1):
        # Pass the rolling config up to the Python master loop
        super().__init__(name=name, window_type=window_type, window_size=window_size, refit_step=refit_step)
        self.mgarch_type = mgarch_type.upper()  # "CCC" or "DCC"
        
        # Create a unique variable name in the R environment to avoid collisions 
        # if CCC and DCC are running simultaneously in the same pipeline.
        self.r_fit_name = f"fit_{id(self)}"

    def fit(self, train_data):
        r_train = train_data[["r_CNY", "r_CNH"]].dropna()
        ro.globalenv["returns"] = pandas2ri.py2rpy(r_train)
        
        ro.r(f"""
            uspec <- ugarchspec(
                variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                mean.model     = list(armaOrder = c(0, 0), include.mean = TRUE),
                distribution.model = "norm"
            )
            mspec <- multispec(replicate(2, uspec))
            
            spec <- dccspec(uspec = mspec, dccOrder = c(1, 1), 
                            model = "{self.mgarch_type}", distribution = "mvnorm")
            
            {self.r_fit_name} <- dccfit(spec, data = returns, fit.control = list(eval.se = FALSE))
            
            cfs <- coef({self.r_fit_name})
            cf_names <- names(cfs)
            cf_vals <- as.numeric(cfs)

            # rcor returns an array of shape (2, 2, T). We just grab the [1, 2] element from the first time step.
            corrs <- rcor({self.r_fit_name})
            static_corr <- corrs[1, 2, 1]
        """)
        
        # Save the latest parameters to a Python dictionary
        names = np.array(ro.globalenv["cf_names"])
        vals = np.array(ro.globalenv["cf_vals"])
        self.latest_params = dict(zip(names, vals))
        self.static_corr = ro.globalenv["static_corr"][0]

    def get_model_attributes(self):
        if not hasattr(self, 'latest_params'):
            return "Not fitted yet"
            
        if self.mgarch_type == "DCC":
            a = self.latest_params.get('[Joint]dcca1', np.nan)
            b = self.latest_params.get('[Joint]dccb1', np.nan)
            return f"a={a:.4f}, b={b:.4f}"
        else:
            return f"Static Corr={self.static_corr:.4f}"

    def run_backtest(self, full_data, train_end_idx):
        """
        Override the master loop entirely for MGARCH models.
        Uses R's native n.roll mechanism for 1-step-ahead conditional forecasts.
        """
        self.reset()

        test_data = full_data.iloc[train_end_idx:]
        n_test = len(test_data)

        if self.window_type == 'rolling':
            # For rolling: use a window of size window_size ending at the end of full_data
            start_idx = train_end_idx - self.window_size
            r_data = full_data[["r_CNY", "r_CNH"]].iloc[start_idx:train_end_idx + n_test].dropna()
            n_oos = n_test  # out.sample count relative to this slice
        else:
            # Static and expanding both use all data from the start
            r_data = full_data[["r_CNY", "r_CNH"]].iloc[:train_end_idx + n_test].dropna()
            n_oos = n_test

        ro.globalenv["returns_data"] = pandas2ri.py2rpy(r_data)
        ro.globalenv["n_oos"] = n_oos

        ro.r(f"""
            uspec <- ugarchspec(
                variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                mean.model     = list(armaOrder = c(0, 0), include.mean = TRUE),
                distribution.model = "norm"
            )
            mspec <- multispec(replicate(2, uspec))
            spec <- dccspec(uspec = mspec, dccOrder = c(1, 1),
                            model = "{self.mgarch_type}", distribution = "mvnorm")

            {self.r_fit_name} <- dccfit(spec, data = returns_data,
                                        out.sample = n_oos,
                                        fit.control = list(eval.se = FALSE))

            fcst <- dccforecast({self.r_fit_name}, n.ahead = 1, n.roll = n_oos - 1)

            # Extract 1-step-ahead hedge ratios from each rolling forecast
            h_mgarch <- rep(0, n_oos)
            for (i in 1:n_oos) {{
                cov_i <- rcov(fcst)[[i]]
                h_mgarch[i] <- cov_i[1, 2, 1] / cov_i[2, 2, 1]
            }}

            # Extract model parameters
            cfs <- coef({self.r_fit_name})
            cf_names <- names(cfs)
            cf_vals <- as.numeric(cfs)
            corrs <- rcor({self.r_fit_name})
            static_corr <- corrs[1, 2, 1]
        """)

        h_array = np.array(ro.r("h_mgarch"))
        self.hedge_ratio_history.extend(h_array.tolist())

        names = np.array(ro.globalenv["cf_names"])
        vals = np.array(ro.globalenv["cf_vals"])
        self.latest_params = dict(zip(names, vals))
        self.static_corr = ro.globalenv["static_corr"][0]

        r_test_cny = test_data["r_CNY"].values
        r_test_cnh = test_data["r_CNH"].values
        return r_test_cny - h_array * r_test_cnh

    def get_hedge_info(self):
        if len(self.hedge_ratio_history) > 0:
            avg_h = np.mean(self.hedge_ratio_history)
            return f"Dynamic (Avg: {avg_h:.4f})"
        return "N/A"


# --- The Actual Models Used in main.py ---

class CCCHedgeModel(BaseMGARCHModel):
    def __init__(self, window_type='static', window_size=None, refit_step=1):
        super().__init__(name="CCC-GARCH", mgarch_type="CCC", 
                         window_type=window_type, window_size=window_size, refit_step=refit_step)


class DCCHedgeModel(BaseMGARCHModel):
    def __init__(self, window_type='static', window_size=None, refit_step=1):
        super().__init__(name="DCC-GARCH", mgarch_type="DCC", 
                         window_type=window_type, window_size=window_size, refit_step=refit_step)