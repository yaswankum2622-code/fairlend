"""
FairLend | models/fair_model.py | Fairlearn constrained LightGBM
"""

import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.baseline import prepare_features, compute_metrics, get_or_create_split_indices
from database.db import get_ml_features

SAVED_DIR = Path(__file__).parent / "saved"
SAVED_DIR.mkdir(parents=True, exist_ok=True)


def train_fair_model():
    """
    Train LightGBM with Fairlearn demographic parity constraint.
    Optimises accuracy subject to: approval rate equal across race groups.
    """

    print("Loading features from DB...")
    df = get_ml_features()
    X, y, feature_cols, encoders = prepare_features(df)
    protected_features = {"race_simplified", "sex_simplified"}
    assert not protected_features.intersection(feature_cols), "Protected attributes present in fair-model inputs"
    train_idx, test_idx = get_or_create_split_indices(y)

    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    # Protected attributes remain available only as sensitive features for fairness constraints.
    sensitive_train = df.iloc[train_idx]["race_simplified"].to_numpy()
    sensitive_test = df.iloc[test_idx]["race_simplified"].to_numpy()

    print("Training LightGBM + Fairlearn (demographic parity)...")
    print("This takes 3-5 minutes — running 50 reduction iterations...")

    base_estimator = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=5,
        random_state=42,
        class_weight="balanced",
        verbose=-1
    )

    constraint = DemographicParity(difference_bound=0.05)

    fair_model = ExponentiatedGradient(
        estimator=base_estimator,
        constraints=constraint,
        eps=0.05,
        max_iter=50,
        nu=1e-6
    )

    fair_model.fit(
        X_tr, y_tr,
        sensitive_features=sensitive_train
    )

    y_pred = fair_model.predict(X_te)

    # ExponentiatedGradient does not expose calibrated probabilities.
    y_prob = y_pred.astype(float)
    metrics = compute_metrics(y_te, y_pred, y_prob)

    joblib.dump(fair_model, SAVED_DIR / "lgbm_fair.joblib")
    joblib.dump(train_idx, SAVED_DIR / "train_idx.joblib")
    joblib.dump(test_idx, SAVED_DIR / "test_idx.joblib")

    print()
    print("LightGBM + Fairlearn Results:")
    for k, v in metrics.items():
        print(f"  {k:<12} {v}")

    return metrics, fair_model, X_te, y_te, feature_cols


if __name__ == "__main__":
    train_fair_model()
