import numpy as np
from .base import BaseHedgeModel

class OLSHedgeModel(BaseHedgeModel):
    def __init__(self, window_type='static', window_size=None, refit_step=1):
        # Initialize the parent class with the model's name
        super().__init__(name="OLS", window_type=window_type, window_size=window_size, refit_step=refit_step)
        self.h_ols = None

    def fit(self, train_data):
        r_train_cny = train_data["r_CNY"].values
        r_train_cnh = train_data["r_CNH"].values
        
        # Calculate static hedge ratio
        cov_matrix = np.cov(r_train_cny, r_train_cnh)
        self.h_ols = cov_matrix[0, 1] / np.var(r_train_cnh)

        # Track the ratio over time
        self.hedge_ratio_history.append(self.h_ols)

    def predict_step(self, test_step_data):
        r_test_cny = test_step_data["r_CNY"].values
        r_test_cnh = test_step_data["r_CNH"].values
        
        # Calculate hedged returns
        return r_test_cny - self.h_ols * r_test_cnh

    def get_hedge_info(self):
        if self.window_type == 'static':
            return f"{self.h_ols:.4f}"
        else:
            avg_h = np.mean(self.hedge_ratio_history)
            return f"Dynamic (Avg: {avg_h:.4f})"