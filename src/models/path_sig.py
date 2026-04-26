import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
import esig

from .base import BaseHedgeModel
from src.config import SIG_WINDOW, SIG_DEPTH, RIDGE_ALPHAS, RIDGE_CV_FOLDS


class PathSigHedgeModel(BaseHedgeModel):
    """
    Path Signature Hedge Model (Varying-Coefficient with OLS anchor).

    Uses the full truncated signature INCLUDING the depth-0 constant
    term (always 1.0) so that the interaction feature vector:

        z_t = Sig(X_t) * r_t^{CNH}

    naturally contains r_t^{CNH} as its first element. This means:

        r_t^{CNY} = beta' * z_t + eps_t
                  = beta_0 * r_t^{CNH}                    [OLS core]
                    + beta_1 * S^(1) * r_t^{CNH}          [depth-1 corrections]
                    + beta_2 * S^(1,1) * r_t^{CNH}        [depth-2 corrections]
                    + ...                                  [higher-order]
                    + eps_t

    Ridge regularisation shrinks the higher-order correction terms
    toward zero, so the model degrades gracefully to OLS when
    signatures are uninformative. The first coefficient beta_0
    carries the bulk of the signal and lands near the OLS hedge
    ratio (~0.78).

    No intercept is used — the model can only act through its
    CNH position, preventing directional alpha.
    """

    def __init__(self, window=SIG_WINDOW, depth=SIG_DEPTH,
                 window_type='static', window_size=None, refit_step=1):
        super().__init__(
            name="PathSig",
            window_type=window_type,
            window_size=window_size,
            refit_step=refit_step
        )
        self.sig_window = window
        self.sig_depth = depth
        self.ridge_model = None
        self.sig_dim = esig.sigdim(2, self.sig_depth)

    # ------------------------------------------------------------------
    # Path & Signature construction
    # ------------------------------------------------------------------

    def _build_path(self, r_cny_window, r_cnh_window):
        """
        Builds a 2D cumulative-return path from the origin.
        Shape: (W+1, 2).
        """
        cum_cny = np.concatenate([[0.0], np.cumsum(r_cny_window)])
        cum_cnh = np.concatenate([[0.0], np.cumsum(r_cnh_window)])
        return np.column_stack([cum_cny, cum_cnh]).astype(np.float64)

    def _compute_signature(self, path):
        """
        Computes the truncated signature of a path.
        """
        return esig.stream2sig(path, self.sig_depth)

    def _build_features(self, r_cny, r_cnh):
        """
        Constructs the interaction feature matrix for the varying-
        coefficient regression.

        For each valid index t (t >= sig_window):
          z_t = Sig(X_{t-W:t}) * r_t^{CNH}

        The first element of z_t is 1.0 * r_t^{CNH} = r_t^{CNH},
        which acts as the OLS regressor.

        Returns
        -------
        Z : np.ndarray, shape (n_samples, sig_dim)
        y : np.ndarray, shape (n_samples,)
        """
        n = len(r_cny)
        W = self.sig_window

        sigs = []
        for t in range(W, n):
            path = self._build_path(r_cny[t - W:t], r_cnh[t - W:t])
            sig = self._compute_signature(path)
            sigs.append(sig)

        sigs = np.array(sigs)
        r_cnh_aligned = r_cnh[W:]
        y = r_cny[W:]

        # z_t = Sig(X_t) * r_t^{CNH}  (element-wise broadcast)
        Z = sigs * r_cnh_aligned[:, np.newaxis]

        return Z, y

    # ------------------------------------------------------------------
    # BaseHedgeModel interface
    # ------------------------------------------------------------------

    def fit(self, train_data):
        r_cny = train_data["r_CNY"].values
        r_cnh = train_data["r_CNH"].values

        # Store training tail for predict_step boundary handling
        self._tail_cny = r_cny[-self.sig_window:]
        self._tail_cnh = r_cnh[-self.sig_window:]

        Z, y = self._build_features(r_cny, r_cnh)

        # Ridge CV — no intercept, no standardisation
        # The first column of Z is r_CNH itself (from depth-0 = 1),
        # so beta_0 naturally estimates the OLS hedge ratio.
        # Higher-order terms are shrunk toward zero by Ridge.
        self.ridge_model = RidgeCV(
            alphas=RIDGE_ALPHAS,
            fit_intercept=False,
            cv=RIDGE_CV_FOLDS
        )
        self.ridge_model.fit(Z, y)

    def predict_step(self, test_step_data):
        """
        Generates hedged PnL for a test chunk.

        Prepends training tail for signature window continuity.
        """
        r_cny_test = test_step_data["r_CNY"].values
        r_cnh_test = test_step_data["r_CNH"].values

        # Prepend training tail for signature window history
        r_cny = np.concatenate([self._tail_cny, r_cny_test])
        r_cnh = np.concatenate([self._tail_cnh, r_cnh_test])

        W = self.sig_window
        n_test = len(r_cny_test)
        pnl = []

        for i in range(n_test):
            t = W + i

            path = self._build_path(r_cny[t - W:t], r_cnh[t - W:t])
            sig = self._compute_signature(path)

            # z_t = Sig(X_t) * r_t^{CNH}  — first element is just r_CNH
            z_t = sig * r_cnh[t]
            predicted_cny = self.ridge_model.predict(z_t.reshape(1, -1))[0]
            pnl.append(r_cny[t] - predicted_cny)

            # Track effective hedge ratio
            if abs(r_cnh[t]) > 1e-10:
                h_t = predicted_cny / r_cnh[t]
                self.hedge_ratio_history.append(h_t)

        return np.array(pnl)

    def get_hedge_info(self):
        if len(self.hedge_ratio_history) > 0:
            avg_h = np.mean(self.hedge_ratio_history)
            return f"Dynamic (Avg: {avg_h:.4f})"
        return "N/A"

    def get_model_attributes(self):
        if self.ridge_model is not None:
            alpha = self.ridge_model.alpha_
            beta_0 = self.ridge_model.coef_[0]
            return f"W={self.sig_window}, D={self.sig_depth}, λ={alpha:.2f}, β₀={beta_0:.4f}"
        return "Not fitted yet"

    def get_residuals(self):
        """Not directly applicable — returns empty dict."""
        return {}