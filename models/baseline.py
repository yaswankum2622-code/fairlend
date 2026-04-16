"""
FairLend | models/baseline.py | Logistic regression and decision tree baselines
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, brier_score_loss, classification_report
)
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).parent.parent))
from database.db import get_ml_features

SAVED_DIR = Path(__file__).parent / "saved"
SAVED_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_IDX_PATH = SAVED_DIR / "train_idx.joblib"
TEST_IDX_PATH = SAVED_DIR / "test_idx.joblib"


def prepare_features(df: pd.DataFrame):
    """
    Encode categorical features for sklearn models.
    Returns X (features), y (target), feature_names, encoders dict.
    """
    df = df.copy()

    num_cols = [
        "loan_amount", "income", "dti_ratio",
        "loan_to_income_ratio", "is_joint_application", "is_conventional"
    ]
    cat_cols = [
        "loan_type", "lien_status", "state", "age"
    ]
    # race_simplified and sex_simplified are excluded from model inputs.
    # They are protected attributes under ECOA.
    # They are used only for fairness evaluation in evaluate.py.

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = df[col].fillna("Unknown").astype(str)
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    for col in num_cols:
        numeric = pd.to_numeric(df[col], errors="coerce")
        df[col] = numeric.fillna(numeric.median())

    feature_cols = num_cols + cat_cols
    protected_features = {"race_simplified", "sex_simplified"}
    assert not protected_features.intersection(feature_cols), "Protected attributes present in model features"
    X = df[feature_cols].to_numpy()
    y = pd.to_numeric(df["approved"], errors="coerce").fillna(0).astype(int).to_numpy()

    return X, y, feature_cols, encoders


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Compute full accuracy metric set."""
    ks_stat, _ = ks_2samp(
        y_prob[y_true == 1],
        y_prob[y_true == 0]
    )
    return {
        "auc_roc": round(roc_auc_score(y_true, y_prob), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "brier": round(brier_score_loss(y_true, y_prob), 4),
        "ks_stat": round(float(ks_stat), 4),
    }


def get_split_indices(y: np.ndarray):
    """Return a consistent stratified train/test split index pair."""
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y
    )
    return np.sort(train_idx), np.sort(test_idx)


def get_or_create_split_indices(y: np.ndarray):
    """Load split indices if present, otherwise create and persist them."""
    if TRAIN_IDX_PATH.exists() and TEST_IDX_PATH.exists():
        train_idx = np.asarray(joblib.load(TRAIN_IDX_PATH))
        test_idx = np.asarray(joblib.load(TEST_IDX_PATH))
        if len(train_idx) + len(test_idx) == len(y):
            return train_idx, test_idx

    train_idx, test_idx = get_split_indices(y)
    joblib.dump(train_idx, TRAIN_IDX_PATH)
    joblib.dump(test_idx, TEST_IDX_PATH)
    return train_idx, test_idx


def train_baselines():
    """Train logistic regression and decision tree. Save both."""

    print("Loading features from DB...")
    df = get_ml_features()
    X, y, feature_cols, encoders = prepare_features(df)
    train_idx, test_idx = get_or_create_split_indices(y)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Scale for logistic regression only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
        C=1.0
    )
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
    results["logistic_regression"] = compute_metrics(y_test, y_pred_lr, y_prob_lr)

    # Decision Tree
    print("Training Decision Tree...")
    dt = DecisionTreeClassifier(
        max_depth=6,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=100
    )
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    y_prob_dt = dt.predict_proba(X_test)[:, 1]
    results["decision_tree"] = compute_metrics(y_test, y_pred_dt, y_prob_dt)

    # Save everything
    joblib.dump(lr, SAVED_DIR / "logistic_regression.joblib")
    joblib.dump(dt, SAVED_DIR / "decision_tree.joblib")
    joblib.dump(scaler, SAVED_DIR / "scaler.joblib")
    joblib.dump(encoders, SAVED_DIR / "encoders.joblib")
    joblib.dump(feature_cols, SAVED_DIR / "feature_cols.joblib")

    # Save test split indices for consistent evaluation
    np.save(SAVED_DIR / "X_test.npy", X_test)
    np.save(SAVED_DIR / "y_test.npy", y_test)

    print()
    print("Baseline Results:")
    for model, metrics in results.items():
        print(f"  {model}:")
        for k, v in metrics.items():
            print(f"    {k:<12} {v}")

    return results, X_test, y_test, feature_cols, encoders


if __name__ == "__main__":
    train_baselines()
