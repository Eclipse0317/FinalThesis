"""
src/config.py

Central configuration file for the CNY/CNH Hedging Project.
All hyperparameters, file paths, and project-wide constants live here.
"""
from pathlib import Path

# =========================================================
# 1. Directory & File Paths
# =========================================================
# This automatically finds the root of your project, no matter where you run the script from
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

CNY_DATA_FILE = DATA_DIR / "USD_CNY_Historical_Weekly_Data.csv"
CNH_DATA_FILE = DATA_DIR / "USD_CNH_Historical_Weekly_Data.csv"


# =========================================================
# 2. Global Data & Evaluation Parameters
# =========================================================
TRAIN_SPLIT = 0.7  # 70% training, 30% out-of-sample testing
ROBUSTNESS_SPLITS = [0.6, 0.8]  # Splits for robustness checks


# =========================================================
# 3. Model Hyperparameters
# =========================================================

# --- General Settings ---
REFIT_STEP = 4

# --- VECM Settings ---
VECM_MAX_LAGS = 12

# --- MGARCH (CCC & DCC) Settings ---
# Frequency of refitting the rolling window in out-of-sample forecasts
GARCH_REFIT_EVERY = 20  

# --- Path Signature Settings ---
SIG_WINDOW = 3
SIG_DEPTH = 2
RIDGE_ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
RIDGE_CV_FOLDS = 5