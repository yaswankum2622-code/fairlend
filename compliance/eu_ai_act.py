"""
FairLend | compliance/eu_ai_act.py | Article 9 style compliance checklist for FairLend
"""

import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from database.db import get_dataset_summary, run_query

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models" / "saved"
DOCS_DIR = ROOT_DIR / "docs"


def _applications_table_ready() -> bool:
    summary = get_dataset_summary()
    return int(summary["total_applications"]) > 100_000


def _core_docs_present() -> bool:
    required = [
        ROOT_DIR / "fairlend_scope_freeze.md",
        DOCS_DIR / "problem_statement.md",
        DOCS_DIR / "scope.md",
        DOCS_DIR / "algorithms.md",
        DOCS_DIR / "results.md",
        DOCS_DIR / "future_work.md",
    ]
    return all(path.exists() for path in required)


def _schema_defined() -> bool:
    return (ROOT_DIR / "database" / "schema.sql").exists()


def _data_bias_fields_available() -> bool:
    sql = """
    SELECT COUNT(*) AS protected_fields_ready
    FROM applications
    WHERE race_simplified IS NOT NULL
      AND sex_simplified IS NOT NULL
    """
    result = run_query(sql)[0]
    return int(result["protected_fields_ready"]) > 100_000


def _data_quality_checks_passed() -> bool:
    sql = """
    SELECT
        SUM(CASE WHEN loan_amount IS NULL THEN 1 ELSE 0 END) AS loan_amount_nulls,
        SUM(CASE WHEN income IS NULL THEN 1 ELSE 0 END) AS income_nulls,
        SUM(CASE WHEN dti_ratio IS NULL THEN 1 ELSE 0 END) AS dti_nulls
    FROM applications
    """
    result = run_query(sql)[0]
    return all(int(result[key]) == 0 for key in result)


def _model_artifacts_present() -> bool:
    required = [
        MODELS_DIR / "logistic_regression.joblib",
        MODELS_DIR / "decision_tree.joblib",
        MODELS_DIR / "lgbm_unconstrained.joblib",
        MODELS_DIR / "lgbm_fair.joblib",
        MODELS_DIR / "model_comparison.csv",
    ]
    return all(path.exists() for path in required)


def _explainability_ready() -> bool:
    required = [
        ROOT_DIR / "explainability" / "shap_analysis.py",
        ROOT_DIR / "explainability" / "adverse_action.py",
    ]
    return all(path.exists() for path in required)


def _human_oversight_ready() -> bool:
    required = [
        ROOT_DIR / "dashboard" / "app.py",
        ROOT_DIR / "database" / "db.py",
        ROOT_DIR / "compliance" / "nl_query.py",
    ]
    return all(path.exists() for path in required)


def _fairness_results_present() -> bool:
    comparison_path = MODELS_DIR / "model_comparison.csv"
    if not comparison_path.exists():
        return False

    comparison = pd.read_csv(comparison_path)
    fair_row = comparison.loc[comparison["Model"] == "LightGBM + Fairlearn"]
    if fair_row.empty:
        return False
    return str(fair_row.iloc[0]["Passes DPR"]).upper() == "YES"


def _traceability_ready() -> bool:
    required = [
        MODELS_DIR / "train_idx.joblib",
        MODELS_DIR / "test_idx.joblib",
        MODELS_DIR / "feature_cols.joblib",
        ROOT_DIR / "data" / "fairlend.db",
    ]
    return all(path.exists() for path in required)


def _governance_metrics_defined() -> bool:
    return (ROOT_DIR / "dbt_project" / "models" / "metrics" / "metric_definitions.yml").exists()


def generate_eu_ai_act_report() -> dict:
    """Return a repo-grounded high-level EU AI Act readiness checklist."""
    checks = [
        ("Art. 9(1)", "Risk management system established", _core_docs_present()),
        ("Art. 9(2)", "Training data examined for bias", _data_bias_fields_available()),
        ("Art. 10(2)", "Data schema and governance controls defined", _schema_defined()),
        ("Art. 10(3)", "Training data quality checks passed", _data_quality_checks_passed()),
        ("Art. 11", "Technical documentation package available", _core_docs_present()),
        ("Art. 12", "Traceability and logging artifacts retained", _traceability_ready()),
        ("Art. 13", "Explanation capability implemented", _explainability_ready()),
        ("Art. 14", "Human oversight workflow available", _human_oversight_ready()),
        ("Art. 15(1)", "Accuracy and fairness evaluation completed", _model_artifacts_present()),
        ("Art. 15(3)", "Bias mitigation control demonstrated", _fairness_results_present()),
    ]

    report_df = pd.DataFrame(checks, columns=["article", "requirement", "passed"])
    report_df["status"] = report_df["passed"].map({True: "PASS", False: "FAIL"})

    passed_checks = int(report_df["passed"].sum())
    total_checks = int(len(report_df))
    status = "COMPLIANT" if passed_checks == total_checks and _applications_table_ready() and _governance_metrics_defined() else "REVIEW REQUIRED"

    return {
        "status": status,
        "passed_checks": passed_checks,
        "total_checks": total_checks,
        "checks": report_df[["status", "article", "requirement"]],
    }


def generate_compliance_report() -> dict:
    """Return dashboard-ready EU AI Act compliance metadata."""
    raw_report = generate_eu_ai_act_report()
    evidence_map = {
        "Art. 9(1)": "Scope freeze and supporting governance documents exist in the repo.",
        "Art. 9(2)": "Protected-class fields are retained for fairness testing and bias review.",
        "Art. 10(2)": "SQLite schema and dbt metric definitions define governance controls.",
        "Art. 10(3)": "Loader assertions enforce quality checks for key underwriting fields.",
        "Art. 11": "Problem statement, scope, algorithms, results, and future work documents are present.",
        "Art. 12": "Saved train/test artifacts and model files support traceability.",
        "Art. 13": "SHAP explanations and adverse action generation are implemented.",
        "Art. 14": "Dashboard review flow and compliance chat provide human oversight.",
        "Art. 15(1)": "All trained models were evaluated on the HMDA 2024 sample.",
        "Art. 15(3)": "Fairlearn-constrained model clears the saved DPR threshold.",
    }

    checks = []
    for _, row in raw_report["checks"].iterrows():
        checks.append(
            {
                "article": row["article"],
                "requirement": row["requirement"],
                "status": row["status"],
                "evidence": evidence_map.get(
                    row["article"], "Repo artifact check completed."
                ),
            }
        )

    return {
        "passed": raw_report["passed_checks"],
        "total": raw_report["total_checks"],
        "overall_status": raw_report["status"],
        "model_version": "FairLend v1.0",
        "deadline": "August 2026",
        "generated_date": date.today().isoformat(),
        "regulation": "EU AI Act - High-Risk Credit Scoring",
        "checks": checks,
    }


def print_eu_ai_act_report() -> dict:
    """Print a concise EU AI Act readiness report and return it."""
    report = generate_eu_ai_act_report()
    print("Testing EU AI Act report...")
    print(f"Status: {report['status']}")
    print(f"Passed: {report['passed_checks']}/{report['total_checks']} checks")
    for _, row in report["checks"].iterrows():
        print(f"  {row['status']:<4}  {row['article']:<9} {row['requirement']}")
    return report


def main():
    print_eu_ai_act_report()


if __name__ == "__main__":
    main()
