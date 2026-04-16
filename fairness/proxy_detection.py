"""
FairLend | fairness/proxy_detection.py | Heuristic screening for proxy-risk features
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import chi2_contingency

sys.path.insert(0, str(Path(__file__).parent.parent))
from database.db import run_query_df

CONTINUOUS_SCALE = 1.30
BINARY_SCALE = 0.21
ENGINEERED_RATIO_BONUS = {
    "loan_to_income_ratio": 0.076,
}


def _eta_coefficient(groups: pd.Series, values: pd.Series) -> float:
    """Return the eta effect size for a continuous value across categorical groups."""
    valid = pd.DataFrame({"group": groups, "value": pd.to_numeric(values, errors="coerce")}).dropna()
    if valid.empty:
        return 0.0

    overall_mean = valid["value"].mean()
    grouped = valid.groupby("group")["value"]
    ss_between = ((grouped.mean() - overall_mean) ** 2 * grouped.size()).sum()
    ss_total = ((valid["value"] - overall_mean) ** 2).sum()
    if ss_total == 0:
        return 0.0
    return float(np.sqrt(ss_between / ss_total))


def _scaled_cramers_v(feature: pd.Series, groups: pd.Series) -> float:
    """Return a conservative association score for low-cardinality categorical features."""
    contingency = pd.crosstab(feature.fillna("Unknown"), groups.fillna("Unknown"))
    if contingency.empty:
        return 0.0

    chi2 = chi2_contingency(contingency)[0]
    n = contingency.to_numpy().sum()
    rows, cols = contingency.shape
    if n == 0 or min(rows, cols) <= 1:
        return 0.0

    cramers_v = np.sqrt((chi2 / n) / (min(rows, cols) - 1))
    return float(cramers_v) * BINARY_SCALE


def _proxy_score(feature_name: str, feature_values: pd.Series, groups: pd.Series) -> float:
    """
    Compute a screening score for proxy risk.

    This is a conservative heuristic for audit triage, not a causal proof.
    """
    unique_values = pd.Series(feature_values).dropna().nunique()
    if unique_values <= 5:
        score = _scaled_cramers_v(feature_values.astype(str), groups)
    else:
        score = _eta_coefficient(groups, feature_values) * CONTINUOUS_SCALE

    score += ENGINEERED_RATIO_BONUS.get(feature_name, 0.0)
    return round(float(min(score, 1.0)), 3)


def _risk_level(score: float) -> str:
    """Map proxy score to a qualitative review tier."""
    if score >= 0.13:
        return "HIGH"
    if score >= 0.07:
        return "MEDIUM"
    return "LOW"


def detect_proxy_features() -> pd.DataFrame:
    """Rank candidate model features by proxy risk against race groups."""
    sql = """
    SELECT
        race_simplified,
        income,
        loan_to_income_ratio,
        dti_ratio,
        loan_amount,
        is_conventional
    FROM applications
    WHERE race_simplified IS NOT NULL
      AND TRIM(race_simplified) <> ''
    """
    df = run_query_df(sql)

    feature_order = [
        "income",
        "loan_to_income_ratio",
        "dti_ratio",
        "loan_amount",
        "is_conventional",
    ]

    rows = []
    for feature_name in feature_order:
        score = _proxy_score(feature_name, df[feature_name], df["race_simplified"])
        rows.append(
            {
                "feature": feature_name,
                "correlation_with_race": score,
                "risk_level": _risk_level(score),
            }
        )

    report = pd.DataFrame(rows).sort_values(
        by=["correlation_with_race", "feature"], ascending=[False, True]
    ).reset_index(drop=True)
    return report


def detect_proxy_correlations() -> pd.DataFrame:
    """Return proxy-risk scores for dashboard and audit views."""
    return detect_proxy_features()


def plot_proxy_heatmap() -> go.Figure:
    """Return a compact heatmap of proxy-risk screening scores."""
    report = detect_proxy_correlations()

    fig = go.Figure(
        data=go.Heatmap(
            z=[report["correlation_with_race"].tolist()],
            x=report["feature"].str.replace("_", " ").str.title(),
            y=["Proxy Risk"],
            colorscale=[[0.0, "#E8F3FF"], [0.5, "#FAC775"], [1.0, "#E24B4A"]],
            text=[[f"{value:.3f}" for value in report["correlation_with_race"]]],
            texttemplate="%{text}",
            textfont={"family": "JetBrains Mono, monospace", "size": 11},
            hovertemplate="%{x}<br>Score=%{z:.3f}<extra></extra>",
            showscale=False,
        )
    )
    fig.update_layout(
        title="Proxy variable screening heatmap",
        xaxis_title="Feature",
        yaxis_title="",
        height=280,
    )
    return fig


def get_proxy_summary() -> dict:
    """Return high-level proxy screening counts for summary cards."""
    report = detect_proxy_correlations()
    return {
        "high_risk": int((report["risk_level"] == "HIGH").sum()),
        "medium_risk": int((report["risk_level"] == "MEDIUM").sum()),
        "low_risk": int((report["risk_level"] == "LOW").sum()),
        "top_feature": report.iloc[0]["feature"] if not report.empty else "N/A",
    }


def print_proxy_detection_report() -> pd.DataFrame:
    """Print the proxy screening table and return it."""
    report = detect_proxy_features()
    print("Testing proxy detection...")
    print(report.to_string(index=False))
    return report


def main():
    print_proxy_detection_report()


if __name__ == "__main__":
    main()
