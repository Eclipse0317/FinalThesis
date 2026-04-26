"""
run_sensitivity.py

Standalone script for PathSig hyperparameter sensitivity analysis.
Run from project root: python run_sensitivity.py
"""

import warnings
from statsmodels.tools.sm_exceptions import ValueWarning
warnings.filterwarnings("ignore", category=ValueWarning)

from src.data_loader import load_data
from src.sensitivity import run_pathsig_sensitivity

if __name__ == "__main__":
    weekly_data = load_data()

    # Main split
    run_pathsig_sensitivity(weekly_data, train_split=0.7)

    # Robustness splits
    run_pathsig_sensitivity(weekly_data, train_split=0.6)
    run_pathsig_sensitivity(weekly_data, train_split=0.8)

    print("\nSensitivity Analysis Complete!")