import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM, select_order, select_coint_rank
from .base import BaseHedgeModel

class VECMHedgeModel(BaseHedgeModel):
    def __init__(self, window_type='static', window_size=None, refit_step=1, max_lags=12):
        # Pass the rolling config up to the parent
        super().__init__(name="VECM", window_type=window_type, window_size=window_size, refit_step=refit_step)
        self.max_lags = max_lags
        self.chosen_lag = None
        self.result = None
        self.h_vecm = None

    def fit(self, train_data):
        # VECM needs raw price levels to find cointegration, not log returns
        data = train_data[["CNY", "CNH"]].dropna()
        
        # 1. Select Lags
        lag_order = select_order(data, maxlags=self.max_lags, deterministic="ci")
        self.chosen_lag = max(lag_order.bic, 1)
        
        # 2. Fit Model
        model = VECM(data, k_ar_diff=self.chosen_lag, coint_rank=1, deterministic="ci")
        self.result = model.fit()
        
        # Calculate hedge ratio from residuals
        cov_matrix = np.cov(self.result.resid[:, 0], self.result.resid[:, 1])
        self.h_vecm = cov_matrix[0, 1] / cov_matrix[1, 1]

        # Track the ratio over time
        self.hedge_ratio_history.append(self.h_vecm)

        # Print model-specific stats ONLY if static, to avoid terminal spam during rolling windows
        if self.window_type == 'static':
            print(f"\n[{self.name}] Fitted with lag={self.chosen_lag}")
            print(f"[{self.name}] Cointegration vector beta: {self.result.beta.flatten()}")

    def get_model_attributes(self):
        if self.chosen_lag is not None:
            # The alpha attribute is a matrix where row 0 is CNY and row 1 is CNH
            alpha_cny = self.result.alpha[0, 0] if self.result is not None else np.nan
            alpha_cnh = self.result.alpha[1, 0] if self.result is not None else np.nan
            
            return f"Lag={self.chosen_lag}, α_cny={alpha_cny:.4f}, α_cnh={alpha_cnh:.4f}"
        return "Not fitted yet"

    def predict_step(self, test_step_data):
        r_test_cny = test_step_data["r_CNY"].values
        r_test_cnh = test_step_data["r_CNH"].values
        return r_test_cny - self.h_vecm * r_test_cnh

    def get_residuals(self):
        """Returns residuals safely packaged as a dictionary for the diagnostics module."""
        if self.result is None:
            raise ValueError("Model must be fitted before getting residuals.")
        
        return {
            "CNY": self.result.resid[:, 0],
            "CNH": self.result.resid[:, 1]
        }

    def get_hedge_info(self):
        if self.window_type == 'static':
            return f"{self.h_vecm:.4f} (lag={self.chosen_lag})"
        else:
            avg_h = np.mean(self.hedge_ratio_history)
            return f"Dynamic (Avg: {avg_h:.4f})"