"""
FairLend | models/lgbm_model.py | LightGBM unconstrained model
"""

import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.baseline import prepare_features, compute_metrics, get_or_create_split_indices
from database.db import get_ml_features

SAVED_DIR = Path(__file__).parent / "saved"
SAVED_DIR.mkdir(parents=True, exist_ok=True)


def train_lgbm():
    """Train LightGBM without fairness constraints. Save model."""

    print("Loading features from DB...")
    df = get_ml_features()
    X, y, feature_cols, encoders = prepare_features(df)
    protected_features = {"race_simplified", "sex_simplified"}
    assert not protected_features.intersection(feature_cols), "Protected attributes present in LightGBM inputs"
    train_idx, test_idx = get_or_create_split_indices(y)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print("Training LightGBM (unconstrained)...")
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=6,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        class_weight="balanced",
        verbose=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=-1)]
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_prob)

    joblib.dump(model, SAVED_DIR / "lgbm_unconstrained.joblib")

    print()
    print("LightGBM (unconstrained) Results:")
    for k, v in metrics.items():
        print(f"  {k:<12} {v}")

    return metrics, model, X_test, y_test, feature_cols


if __name__ == "__main__":
    train_lgbm()
