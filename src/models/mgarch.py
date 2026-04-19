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
        # Use the pre-calculated returns directly from data loader
        r_train = train_data[["r_CNY", "r_CNH"]].dropna()
        
        ro.globalenv["returns"] = pandas2ri.py2rpy(r_train)
        
        # Fit the model and store it in R's global environment under our unique name
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
        """)

    def predict_step(self, test_step_data):
        # Tell R exactly how many steps ahead we need to forecast for this chunk
        n_ahead = len(test_step_data)
        ro.globalenv["n_ahead"] = n_ahead
        
        # Generate the N-step ahead forecast using the model we just fitted
        ro.r(f"""
            fcst <- dccforecast({self.r_fit_name}, n.ahead = n_ahead)
            
            # rcov(fcst) returns a list. [[1]] grabs the array of covariance matrices
            fcst_cov <- rcov(fcst)[[1]]
            
            h_mgarch <- rep(0, n_ahead)
            for (i in 1:n_ahead) {{
                h_mgarch[i] <- fcst_cov[1, 2, i] / fcst_cov[2, 2, i]
            }}
        """)

        # Pull the array of dynamic hedge ratios back into Python
        h_array = np.array(ro.r("h_mgarch"))
        
        # Because MGARCH generates an array of ratios (one for each step), we use extend()
        self.hedge_ratio_history.extend(h_array.tolist())

        # Calculate Hedged PnL
        r_test_cny = test_step_data["r_CNY"].values
        r_test_cnh = test_step_data["r_CNH"].values

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