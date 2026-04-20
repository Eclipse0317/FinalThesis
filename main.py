import warnings
from statsmodels.tools.sm_exceptions import ValueWarning
warnings.filterwarnings("ignore", category=ValueWarning)

from src.config import TRAIN_SPLIT, ROBUSTNESS_SPLITS, VECM_MAX_LAGS
from src.data_loader import load_data
from src.models.ols import OLSHedgeModel
from src.models.vecm import VECMHedgeModel
from src.models.mgarch import CCCHedgeModel, DCCHedgeModel
from src.diagnostics import run_residual_diagnostics
from src.evaluation import evaluate_out_of_sample, run_robustness_checks

if __name__ == "__main__":
    # 1. Load Data
    weekly_data = load_data()
    
    # 2. Split Data
    n_train = int(len(weekly_data) * TRAIN_SPLIT)
    train_data = weekly_data.iloc[:n_train]
    test_data = weekly_data.iloc[n_train:]

    # 3. Initialize Models
    models = [
        OLSHedgeModel(window_type='static'),
        OLSHedgeModel(window_type='rolling', window_size=104, refit_step=4),
        OLSHedgeModel(window_type='expanding', refit_step=4), 
        VECMHedgeModel(window_type='static', max_lags=VECM_MAX_LAGS),
        VECMHedgeModel(window_type='rolling', window_size=104, refit_step=4, max_lags=VECM_MAX_LAGS),
        CCCHedgeModel(window_type='static'),
        CCCHedgeModel(window_type='rolling', window_size=104, refit_step=4),
        CCCHedgeModel(window_type='expanding', refit_step=4),
        DCCHedgeModel(window_type='static'),
        DCCHedgeModel(window_type='rolling', window_size=104, refit_step=4),
        # PathSigHedgeModel(window=4, depth=3)
    ]

    # 4. Train Models
    for model in models:
        model.fit(train_data)

    # 5. Run Diagnostics (Example: VECM residuals)
    vecm_model = models[3] 
    run_residual_diagnostics(vecm_model.name, vecm_model.get_residuals())

    # 6. Evaluate Out-of-Sample Performance
    final_results = evaluate_out_of_sample(weekly_data, n_train, models, True)

    # 7. Run Robustness Checks
    # run_robustness_checks(weekly_data, models, splits=ROBUSTNESS_SPLITS)
    
    print("\nAnalysis Complete!")