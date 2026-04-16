"""
FairLend | explainability/shap_analysis.py | SHAP values for credit decisions
"""

import sys
import joblib
import shap
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.baseline import prepare_features
from database.db import get_ml_features

SAVED_DIR = Path(__file__).parent.parent / "models" / "saved"

warnings.filterwarnings(
    "ignore",
    message="LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray",
)


def _normalize_shap_output(shap_values):
    """Return SHAP values for the positive class with a stable shape."""
    if isinstance(shap_values, list):
        return np.asarray(shap_values[1])

    values = np.asarray(shap_values)
    if values.ndim == 3:
        return values[:, :, 1]
    return values


def _get_expected_value(explainer):
    """Return the baseline SHAP expected value for approval class."""
    expected = explainer.expected_value
    if isinstance(expected, list):
        return float(expected[1])

    expected_array = np.asarray(expected)
    if expected_array.ndim == 0:
        return float(expected_array)
    return float(expected_array[-1])


def _encode_single_row(applicant_row: pd.Series, feature_cols, encoders: dict) -> np.ndarray:
    """Encode one applicant row using the fitted training encoders."""
    row = applicant_row.copy()
    cat_cols = [
        "loan_type", "lien_status", "state", "age"
    ]
    for col in cat_cols:
        le = encoders[col]
        value = str(row.get(col, "Unknown"))
        if value in le.classes_:
            row[col] = le.transform([value])[0]
        else:
            row[col] = 0

    num_cols = [
        "loan_amount", "income", "dti_ratio",
        "loan_to_income_ratio", "is_joint_application", "is_conventional"
    ]
    for col in num_cols:
        row[col] = float(pd.to_numeric(row.get(col, 0), errors="coerce") or 0)

    return np.array([[row[col] for col in feature_cols]])


def get_shap_explainer():
    """Load LightGBM model and return SHAP TreeExplainer."""
    model = joblib.load(SAVED_DIR / "lgbm_unconstrained.joblib")
    explainer = shap.TreeExplainer(model)
    return explainer


def explain_applicant(applicant_row: pd.Series) -> dict:
    """
    Compute SHAP values for a single applicant.

    Input:  pd.Series with same columns as get_ml_features() output
    Output: dict with:
              shap_values   list of (feature_name, shap_value) tuples
              base_value    float — model baseline prediction
              prediction    float — final approval probability
              top_factors   list of top 5 features driving decision
    """
    df = get_ml_features()
    X, y, feature_cols, encoders = prepare_features(df)
    protected_features = {"race_simplified", "sex_simplified"}
    assert not protected_features.intersection(feature_cols), "Protected attributes present in SHAP feature list"

    X_single = _encode_single_row(applicant_row, feature_cols, encoders)

    explainer = get_shap_explainer()
    shap_values = explainer.shap_values(X_single)
    sv = _normalize_shap_output(shap_values)[0]

    base_value = _get_expected_value(explainer)

    model = joblib.load(SAVED_DIR / "lgbm_unconstrained.joblib")
    prediction = float(model.predict_proba(X_single)[0][1])

    feature_shap = list(zip(feature_cols, sv.tolist()))
    feature_shap_sorted = sorted(
        feature_shap, key=lambda x: abs(x[1]), reverse=True
    )

    return {
        "shap_values": feature_shap_sorted,
        "base_value": base_value,
        "prediction": prediction,
        "top_factors": feature_shap_sorted[:5],
        "feature_cols": feature_cols,
    }


def plot_waterfall(explanation: dict, applicant_id: str = "Applicant") -> go.Figure:
    """
    Plotly waterfall chart showing SHAP values for one applicant.
    Shows top 10 features contributing to or against approval.
    """
    top_n = 10
    features = explanation["shap_values"][:top_n]
    names = [f[0].replace("_", " ").title() for f in features]
    values = [f[1] for f in features]

    colors = ["#1D9E75" if v > 0 else "#E24B4A" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=names,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in values],
        textposition="outside",
        textfont=dict(size=11, family="Inter, sans-serif"),
    ))

    pred_pct = explanation["prediction"] * 100
    fig.update_layout(
        title=dict(
            text=(f"SHAP Explanation — {applicant_id}<br>"
                  f"<sup>Approval probability: {pred_pct:.1f}%</sup>"),
            font=dict(size=14)
        ),
        xaxis_title="SHAP Value (impact on approval probability)",
        yaxis=dict(autorange="reversed"),
        height=450,
        showlegend=False,
        xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor="#888888"),
    )

    return fig


def get_global_feature_importance() -> go.Figure:
    """
    Global SHAP feature importance across 500 random applicants.
    Shows which features matter most across the entire dataset.
    """
    print("Computing global SHAP importance (500 samples)...")
    df = get_ml_features()
    X, y, feature_cols, encoders = prepare_features(df)

    rng = np.random.default_rng(42)
    sample_size = min(500, len(X))
    sample_idx = rng.choice(len(X), sample_size, replace=False)
    X_sample = X[sample_idx]

    explainer = get_shap_explainer()
    shap_values = explainer.shap_values(X_sample)
    sv = _normalize_shap_output(shap_values)

    mean_abs_shap = np.abs(sv).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": [f.replace("_", " ").title() for f in feature_cols],
        "importance": mean_abs_shap
    }).sort_values("importance", ascending=True)

    fig = go.Figure(go.Bar(
        x=importance_df["importance"],
        y=importance_df["feature"],
        orientation="h",
        marker_color="#534AB7",
        opacity=0.85,
        text=[f"{v:.4f}" for v in importance_df["importance"]],
        textposition="outside",
        textfont=dict(size=10),
    ))

    fig.update_layout(
        title="Global Feature Importance (mean |SHAP| across 500 applicants)",
        xaxis_title="Mean |SHAP value|",
        height=500,
        showlegend=False,
    )

    return fig
