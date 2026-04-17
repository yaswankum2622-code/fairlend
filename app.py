"""
FairLend | app.py | Hugging Face Spaces entry point.

Bootstraps a database and trained artifacts if they are missing, then
hands off to the Streamlit dashboard script.
"""

import os
import runpy
import subprocess
import sys

DB_PATH = "data/fairlend.db"
RAW_DATA_PATH = "data/2024_public_lar_csv.csv"
MODELS_NEEDED = [
    "models/saved/lgbm_unconstrained.joblib",
    "models/saved/lgbm_fair.joblib",
    "models/saved/logistic_regression.joblib",
    "models/saved/decision_tree.joblib",
    "models/saved/model_comparison.csv",
]


def run_script(script_path: str, message: str) -> None:
    """Run a bootstrap script with a clear startup log message."""
    print(message)
    subprocess.run([sys.executable, script_path], check=True)


def ensure_database() -> None:
    """Create the SQLite database from real data or a synthetic fallback."""
    if os.path.exists(DB_PATH):
        return

    if os.path.exists(RAW_DATA_PATH):
        run_script(
            "data/loader.py",
            "First run - generating database from HMDA 2024 data...",
        )
        return

    run_script(
        "ci_setup.py",
        "Raw HMDA CSV not found - generating synthetic demo database for app startup...",
    )


def ensure_models() -> None:
    """Train saved model artifacts when they are not already available."""
    if all(os.path.exists(path) for path in MODELS_NEEDED):
        return

    print("First run - training all 4 models (takes several minutes)...")
    for script in [
        "models/baseline.py",
        "models/lgbm_model.py",
        "models/fair_model.py",
        "models/evaluate.py",
    ]:
        subprocess.run([sys.executable, script], check=True)
    print("Models trained and saved.")


def main() -> None:
    ensure_database()
    ensure_models()
    runpy.run_path("dashboard/app.py", run_name="__main__")


if __name__ == "__main__":
    main()
