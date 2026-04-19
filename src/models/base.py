import numpy as np
import pandas as pd

class BaseHedgeModel:
    def __init__(self, name, window_type='static', window_size=None, refit_step=1):
        """
        window_type: 'static', 'rolling', or 'expanding'
        window_size: Integer number of weeks for the lookback window (if 'rolling')
        refit_step: How often to recalculate the hedge ratio (e.g., 4 = every 4 weeks)
        """
        self.name = f"{name} ({window_type})"
        self.window_type = window_type
        self.window_size = window_size
        self.refit_step = refit_step
        self.hedge_ratio_history = []

    def fit(self, train_data):
        """Trains the model on a specific slice of data."""
        raise NotImplementedError("Subclasses must implement fit()")

    def predict_step(self, test_step_data):
        """Applies the current model state to generate PnL for a single step/chunk."""
        raise NotImplementedError("Subclasses must implement predict_step()")
    
    def get_residuals(self):
        """
        Returns the in-sample training residuals.
        Should return a dictionary or a pandas DataFrame (e.g., {'CNY': array, 'CNH': array})
        """
        raise NotImplementedError("Subclasses must implement get_residuals()")

    def get_hedge_info(self):
        """
        Returns a string describing the hedge ratio or model setup 
        (e.g., "0.8421" or "implicit (alpha=1.0)").
        """
        return "N/A"
    
    def run_backtest(self, full_data, train_end_idx):
        """
        The Master Loop: Written ONCE, inherited by all models.
        Handles data slicing and routing for out-of-sample evaluation.
        """
        test_data = full_data.iloc[train_end_idx:]
        n_test = len(test_data)
        out_of_sample_pnl = []

        # Scenario 1: Static (The old way)
        if self.window_type == 'static':
            train_slice = full_data.iloc[:train_end_idx]
            self.fit(train_slice)
            return self.predict_step(test_data)

        # Scenario 2: Dynamic (Rolling or Expanding)
        for i in range(0, n_test, self.refit_step):
            # Define the current test chunk
            step_end = min(i + self.refit_step, n_test)
            test_chunk = test_data.iloc[i:step_end]

            # Define the training slice based on window type
            current_time_idx = train_end_idx + i
            if self.window_type == 'rolling':
                start_idx = current_time_idx - self.window_size
            elif self.window_type == 'expanding':
                start_idx = 0  # Always start from the beginning
            
            train_slice = full_data.iloc[start_idx:current_time_idx]

            # Fit on the new historical slice, predict the future chunk
            self.fit(train_slice)
            chunk_pnl = self.predict_step(test_chunk)
            out_of_sample_pnl.append(chunk_pnl)

        # Concatenate all chunks back into a single pandas Series or array
        return pd.concat(out_of_sample_pnl) if isinstance(out_of_sample_pnl[0], pd.Series) else np.concatenate(out_of_sample_pnl)
        
    def calculate_he(self, full_data, train_end_idx):
        """Calculates Hedging Efficiency."""
        test_data = full_data.iloc[train_end_idx:]
        r_test_cny = test_data["r_CNY"].values
        var_unhedged = np.var(r_test_cny)
        
        pnl = self.run_backtest(full_data, train_end_idx)
        var_hedged = np.var(pnl)
        
        return 1 - (var_hedged / var_unhedged)


