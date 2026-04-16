"""
FairLend | fairness/disparate_impact.py | Four-fifths rule audit for approval outcomes
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent.parent))
from database.db import run_query_df

MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "saved"


def calculate_disparate_impact(
    sensitive_column: str = "race_simplified",
    reference_group: str = "White",
    threshold: float = 0.8,
) -> pd.DataFrame:
    """Return approval-rate disparities for the selected sensitive attribute."""
    if sensitive_column not in {"race_simplified", "sex_simplified"}:
        raise ValueError("sensitive_column must be 'race_simplified' or 'sex_simplified'")

    sql = f"""
    SELECT
        {sensitive_column} AS group_name,
        COUNT(*) AS total_applications,
        SUM(approved) AS approved_count,
        ROUND(AVG(approved) * 100, 2) AS approval_rate_pct
    FROM applications
    WHERE {sensitive_column} IS NOT NULL
      AND TRIM({sensitive_column}) <> ''
    GROUP BY {sensitive_column}
    ORDER BY approval_rate_pct DESC
    """
    df = run_query_df(sql)

    if df.empty:
        raise ValueError(f"No groups found for {sensitive_column}")

    reference_rows = df.loc[df["group_name"] == reference_group, "approval_rate_pct"]
    if reference_rows.empty:
        raise ValueError(f"Reference group '{reference_group}' not found in {sensitive_column}")

    reference_rate = float(reference_rows.iloc[0])
    df["reference_group"] = reference_group
    df["reference_rate_pct"] = reference_rate
    df["disparate_impact_ratio"] = (df["approval_rate_pct"] / reference_rate).round(3)
    df["legal_status"] = np.where(df["disparate_impact_ratio"] >= threshold, "PASS", "FAIL")

    renamed = df.rename(columns={sensitive_column if sensitive_column in df.columns else "group_name": "group_name"})
    return renamed[
        [
            "group_name",
            "total_applications",
            "approved_count",
            "approval_rate_pct",
            "reference_group",
            "reference_rate_pct",
            "disparate_impact_ratio",
            "legal_status",
        ]
    ]


def print_disparate_impact_report(
    sensitive_column: str = "race_simplified",
    reference_group: str = "White",
    threshold: float = 0.8,
) -> pd.DataFrame:
    """Print a concise disparate-impact report and return the underlying DataFrame."""
    report = calculate_disparate_impact(
        sensitive_column=sensitive_column,
        reference_group=reference_group,
        threshold=threshold,
    )

    print("Testing disparate impact...")
    print(f"Race groups found: {report['group_name'].tolist()}")
    print("DPR values:")
    for _, row in report.iterrows():
        print(
            f"  {row['legal_status']:<4} "
            f"{row['group_name']:<35} "
            f"{row['approval_rate_pct']:.1f}%  "
            f"DI={row['disparate_impact_ratio']:.3f}"
        )
    return report


def compute_full_disparate_impact(threshold: float = 0.8) -> dict:
    """Return disparate-impact audits for race and sex."""
    race_df = calculate_disparate_impact(
        sensitive_column="race_simplified",
        reference_group="White",
        threshold=threshold,
    ).rename(columns={"group_name": "race", "approval_rate_pct": "approval_rate"})
    race_df["passes_legal"] = race_df["legal_status"] == "PASS"

    sex_df = calculate_disparate_impact(
        sensitive_column="sex_simplified",
        reference_group="Male",
        threshold=threshold,
    ).rename(columns={"group_name": "sex", "approval_rate_pct": "approval_rate"})
    sex_df["passes_legal"] = sex_df["legal_status"] == "PASS"

    return {"race": race_df, "sex": sex_df}


def plot_disparate_impact_race(results: dict) -> go.Figure:
    """Return a Plotly bar chart for race-group disparate impact."""
    race_df = results["race"].copy()
    race_df["status"] = np.where(
        race_df["disparate_impact_ratio"] >= 0.85,
        "PASS",
        np.where(race_df["disparate_impact_ratio"] >= 0.80, "MONITOR", "FAIL"),
    )
    color_map = {"PASS": "#1D9E75", "MONITOR": "#BA7517", "FAIL": "#E24B4A"}

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=race_df["race"],
            y=race_df["disparate_impact_ratio"],
            marker_color=[color_map[value] for value in race_df["status"]],
            text=[
                f"DI={ratio:.3f}<br>{rate:.1f}%"
                for ratio, rate in zip(race_df["disparate_impact_ratio"], race_df["approval_rate"])
            ],
            textposition="outside",
        )
    )
    fig.add_hline(
        y=0.80,
        line_dash="dash",
        line_color="#E24B4A",
        line_width=2,
        annotation_text="Legal minimum 0.80",
    )
    fig.add_hline(
        y=0.85,
        line_dash="dot",
        line_color="#BA7517",
        line_width=1.5,
        annotation_text="Best practice 0.85",
    )
    fig.update_layout(
        title="Disparate impact ratio by race group",
        xaxis_title="Race Group",
        yaxis_title="Disparate Impact Ratio",
        yaxis=dict(range=[0, 1.15]),
    )
    return fig


def plot_disparate_impact_comparison() -> go.Figure:
    """Return a model-level DPR comparison chart from saved evaluation results."""
    comparison_path = MODELS_DIR / "model_comparison.csv"
    if not comparison_path.exists():
        raise FileNotFoundError("models/saved/model_comparison.csv not found")

    comparison = pd.read_csv(comparison_path)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=comparison["Model"],
            y=comparison["DPR"],
            marker_color=["#6C63FF", "#1D9E75", "#185FA5", "#BA7517"],
            text=[f"{value:.3f}" for value in comparison["DPR"]],
            textposition="outside",
        )
    )
    fig.add_hline(
        y=0.80,
        line_dash="dash",
        line_color="#E24B4A",
        line_width=2,
        annotation_text="Legal minimum 0.80",
    )
    fig.update_layout(
        title="Demographic parity ratio across trained models",
        xaxis_title="Model",
        yaxis_title="DPR",
        yaxis=dict(range=[0, 1.05]),
        showlegend=False,
    )
    return fig


def main():
    print_disparate_impact_report()


if __name__ == "__main__":
    main()
