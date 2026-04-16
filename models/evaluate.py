"""
FairLend | models/evaluate.py | Complete model comparison — accuracy and fairness
"""

import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from fairlearn.metrics import (
    demographic_parity_ratio,
    equalized_odds_difference,
    MetricFrame
)
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.baseline import prepare_features, get_or_create_split_indices
from database.db import get_ml_features

SAVED_DIR = Path(__file__).parent / "saved"


def run_full_comparison() -> pd.DataFrame:
    """
    Load all 4 saved models. Run on same test set.
    Return DataFrame with accuracy + fairness metrics side by side.
    """

    print("Loading data and models...")
    df = get_ml_features()
    X, y, feature_cols, encoders = prepare_features(df)
    protected_features = {"race_simplified", "sex_simplified"}
    assert not protected_features.intersection(feature_cols), "Protected attributes present in evaluation feature matrix"
    train_idx, test_idx = get_or_create_split_indices(y)

    X_test = X[test_idx]
    y_test = y[test_idx]

    # Sensitive feature is used only for fairness measurement, not model input.
    sensitive_test = df.iloc[test_idx]["race_simplified"].to_numpy()

    scaler = joblib.load(SAVED_DIR / "scaler.joblib")
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": joblib.load(SAVED_DIR / "logistic_regression.joblib"),
        "Decision Tree": joblib.load(SAVED_DIR / "decision_tree.joblib"),
        "LightGBM": joblib.load(SAVED_DIR / "lgbm_unconstrained.joblib"),
    }

    results = []

    for name, model in models.items():
        print(f"Evaluating {name}...")

        X_eval = X_test_scaled if name == "Logistic Regression" else X_test
        y_pred = model.predict(X_eval)
        y_prob = model.predict_proba(X_eval)[:, 1]

        ks, _ = ks_2samp(y_prob[y_test == 1], y_prob[y_test == 0])
        dpr = demographic_parity_ratio(
            y_test, y_pred,
            sensitive_features=sensitive_test
        )
        eod = equalized_odds_difference(
            y_test, y_pred,
            sensitive_features=sensitive_test
        )

        results.append({
            "Model": name,
            "AUC-ROC": round(roc_auc_score(y_test, y_prob), 4),
            "F1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "KS Stat": round(float(ks), 4),
            "DPR": round(float(dpr), 4),
            "EOD": round(float(eod), 4),
            "Passes DPR": "YES" if dpr >= 0.8 else "NO",
        })

    # Fair model
    print("Evaluating LightGBM + Fairlearn...")
    fair_model = joblib.load(SAVED_DIR / "lgbm_fair.joblib")
    X_fair = X[test_idx]
    y_fair = y[test_idx]
    sensitive_fair = df.iloc[test_idx]["race_simplified"].to_numpy()

    y_pred_fair = fair_model.predict(X_fair)
    y_prob_fair = y_pred_fair.astype(float)

    dpr_fair = demographic_parity_ratio(
        y_fair, y_pred_fair,
        sensitive_features=sensitive_fair
    )
    eod_fair = equalized_odds_difference(
        y_fair, y_pred_fair,
        sensitive_features=sensitive_fair
    )

    results.append({
        "Model": "LightGBM + Fairlearn",
        "AUC-ROC": round(roc_auc_score(y_fair, y_prob_fair), 4),
        "F1": round(f1_score(y_fair, y_pred_fair, zero_division=0), 4),
        "Precision": round(precision_score(y_fair, y_pred_fair, zero_division=0), 4),
        "Recall": round(recall_score(y_fair, y_pred_fair, zero_division=0), 4),
        "KS Stat": 0.0,
        "DPR": round(float(dpr_fair), 4),
        "EOD": round(float(eod_fair), 4),
        "Passes DPR": "YES" if dpr_fair >= 0.8 else "NO",
    })

    comparison_df = pd.DataFrame(results)

    print()
    print("=" * 80)
    print("FAIRLEND — MODEL COMPARISON")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print()
    print("DPR = Demographic Parity Ratio  (must be >= 0.8 to pass legal threshold)")
    print("EOD = Equalized Odds Difference (lower is fairer)")

    comparison_df.to_csv(SAVED_DIR / "model_comparison.csv", index=False)
    print()
    print("Saved to models/saved/model_comparison.csv")

    return comparison_df


if __name__ == "__main__":
    run_full_comparison()
