"""
src/models/path_sig.py

Path Signature Hedge Model
--------------------------
Uses iterated-integral signatures of the (CNY, CNH) price path as features
to predict r_CNY via Ridge regression.  The hedged PnL is  y - ŷ , so the
model implicitly learns a state-dependent, nonlinear hedge ratio.

Feature vector for each time step t:
    [ sig(path_{t-W-1 : t}),  r_CNH_t,  sig(path) * r_CNH_t ]

where the path is mean-shifted (subtract initial point) so the signature
is translation-invariant.  Interaction terms (sig * r_CNH) let the Ridge
learn a hedge ratio that varies with the recent path geometry.
"""

import numpy as np
import pandas as pd
import esig
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from .base import BaseHedgeModel
from src.config import SIG_WINDOW, SIG_DEPTH, RIDGE_ALPHAS, RIDGE_CV_FOLDS


class PathSigHedgeModel(BaseHedgeModel):
    def __init__(self, window_type='static', window_size=None, refit_step=1,
                 sig_window=SIG_WINDOW, sig_depth=SIG_DEPTH):
        super().__init__(name="Path Sig", window_type=window_type,
                         window_size=window_size, refit_step=refit_step)
        self.sig_window = sig_window
        self.sig_depth = sig_depth

        # Fitted state
        self.ridge = None
        self.scaler = None
        self.best_alpha = None
        self._price_tail = None  # last (sig_window + 1) prices from training set

    def reset(self):
        super().reset()
        self.ridge = None
        self.scaler = None
        self.best_alpha = None
        self._price_tail = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_signature(self, cny_prices, cnh_prices):
        """
        Build a single signature vector from two aligned price arrays of
        length (sig_window + 1).  The path is mean-shifted so the first
        point is the origin (translation invariance).
        """
        path = np.column_stack([
            cny_prices - cny_prices[0],
            cnh_prices - cnh_prices[0]
        ])
        return esig.stream2sig(path, self.sig_depth)

    def _build_features(self, prices_cny, prices_cnh, r_cnh):
        """
        Given full price arrays and the corresponding r_CNH returns,
        build the feature matrix for all valid time steps.

        Parameters
        ----------
        prices_cny, prices_cnh : 1-D arrays
            Raw price levels, length N.
        r_cnh : 1-D array
            Log returns of CNH, length N.  The first (sig_window + 1)
            entries are consumed by the lookback and cannot be predicted.

        Returns
        -------
        X : ndarray of shape (n_valid, n_features)
        valid_idx : ndarray of integer indices into the input arrays
        """
        W = self.sig_window
        n = len(prices_cny)
        sig_list = []
        valid_idx = []

        for t in range(W + 1, n):
            # Use prices [t-W-1 .. t-1] inclusive  →  W+1 points, W intervals
            # This avoids look-ahead: price at t is NOT included
            sig = self._build_signature(
                prices_cny[t - W - 1: t],
                prices_cnh[t - W - 1: t]
            )
            sig_list.append(sig)
            valid_idx.append(t)

        sigs = np.array(sig_list)
        valid_idx = np.array(valid_idx)
        r_h = r_cnh[valid_idx].reshape(-1, 1)

        # Feature vector: [ sig,  r_CNH,  sig * r_CNH ]
        X = np.column_stack([sigs, r_h, sigs * r_h])
        return X, valid_idx

    # ------------------------------------------------------------------
    # Framework interface
    # ------------------------------------------------------------------
    def fit(self, train_data):
        """
        Fit Ridge regression on in-sample signature features.
        Also stores the price tail needed by predict_step.
        """
        prices_cny = train_data["CNY"].values
        prices_cnh = train_data["CNH"].values
        r_cny = train_data["r_CNY"].values
        r_cnh = train_data["r_CNH"].values

        # Build training features
        X_train, valid_idx = self._build_features(prices_cny, prices_cnh, r_cnh)
        y_train = r_cny[valid_idx]

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Cross-validate to pick best Ridge alpha
        best_alpha = RIDGE_ALPHAS[0]
        best_cv = -np.inf
        for alpha in RIDGE_ALPHAS:
            ridge = Ridge(alpha=alpha)
            scores = cross_val_score(
                ridge, X_train_scaled, y_train,
                cv=RIDGE_CV_FOLDS, scoring="neg_mean_squared_error"
            )
            if scores.mean() > best_cv:
                best_cv = scores.mean()
                best_alpha = alpha

        self.best_alpha = best_alpha
        self.ridge = Ridge(alpha=best_alpha)
        self.ridge.fit(X_train_scaled, y_train)

        # Store the tail of prices so predict_step can form signatures
        # for the first few out-of-sample points that need lookback into
        # the training period.
        tail_len = self.sig_window + 1
        self._price_tail = {
            "CNY": prices_cny[-tail_len:],
            "CNH": prices_cnh[-tail_len:]
        }

        # Track hedge ratio history (implicit — store alpha for reporting)
        self.hedge_ratio_history.append(best_alpha)

    def predict_step(self, test_step_data):
        """
        Predict r_CNY for each row in test_step_data using stored Ridge
        model and scaler.  Returns hedged PnL = actual − predicted.
        """
        # Prepend the price tail from training so we can form signatures
        # for the earliest test observations
        prices_cny = np.concatenate([self._price_tail["CNY"], test_step_data["CNY"].values])
        prices_cnh = np.concatenate([self._price_tail["CNH"], test_step_data["CNH"].values])

        # r_CNH needs the same prepend treatment — but we only need it
        # for alignment, not as a target.  Compute returns on the joined
        # price array so indices stay consistent.
        r_cnh_full = np.diff(np.log(prices_cnh)) * 100
        r_cny_full = np.diff(np.log(prices_cny)) * 100

        # The tail has (sig_window + 1) prices, producing (sig_window)
        # returns when differenced.  The test returns start at index
        # sig_window in r_*_full.
        tail_returns = self.sig_window  # number of return observations from tail

        X_full, valid_idx = self._build_features(prices_cny, prices_cnh, r_cnh_full)

        # Map valid_idx back: indices >= tail_returns correspond to
        # actual test observations.  The offset is because r_*_full[0]
        # is the return between tail[-1] and test[0], and the first
        # (sig_window) returns come from within the tail + overlap.
        # valid_idx is relative to r_*_full, and test returns start at
        # index = sig_window in r_*_full (since tail had sig_window+1 prices
        # → sig_window returns, then test returns follow).
        test_mask = valid_idx >= tail_returns
        X_test = X_full[test_mask]
        y_test = r_cny_full[valid_idx[test_mask]]

        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.ridge.predict(X_test_scaled)

        return y_test - y_pred

    def get_hedge_info(self):
        if self.window_type == 'static':
            return f"implicit (α={self.best_alpha})"
        else:
            return f"implicit (dynamic)"

    def get_model_attributes(self):
        if self.ridge is not None:
            sig_dim = esig.sigdim(2, self.sig_depth)
            n_features = sig_dim + 1 + sig_dim  # sig + r_cnh + interactions
            return f"W={self.sig_window}, d={self.sig_depth}, α={self.best_alpha}, feat={n_features}"
        return "Not fitted yet"