"""
FairLend | app.py | Hugging Face Spaces entry point
Auto-generates database and trains models on first run.
"""

import subprocess
import sys
import os

# Auto-generate database if missing
if not os.path.exists("data/fairlend.db"):
    print("First run — generating database from HMDA 2024 data...")
    subprocess.run([sys.executable, "data/loader.py"], check=True)

# Auto-train models if missing
models_needed = [
    "models/saved/lgbm_unconstrained.joblib",
    "models/saved/lgbm_fair.joblib",
    "models/saved/logistic_regression.joblib",
    "models/saved/decision_tree.joblib",
    "models/saved/model_comparison.csv",
]
if not all(os.path.exists(f) for f in models_needed):
    print("First run — training all 4 models (takes 5-8 minutes)...")
    for script in [
        "models/baseline.py",
        "models/lgbm_model.py",
        "models/fair_model.py",
        "models/evaluate.py",
    ]:
        subprocess.run([sys.executable, script], check=True)
    print("Models trained and saved.")

# Launch Streamlit dashboard
import streamlit.web.cli as stcli

if __name__ == "__main__":
    sys.argv = [
        "streamlit", "run", "dashboard/app.py",
        "--server.port=7860",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--browser.gatherUsageStats=false"
    ]
    sys.exit(stcli.main())
