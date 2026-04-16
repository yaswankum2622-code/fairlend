"""
FairLend | verify.py | End-to-end local verification script.
Run this after setup to confirm everything works before pushing.
"""

import subprocess
import sys
import os

checks = []


def ok(msg):
    print(f"  ✓  {msg}")
    checks.append(True)


def fail(msg):
    print(f"  ✗  {msg}")
    checks.append(False)


print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print(" FairLend — End-to-End Verification")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
print()

# 1. Check data file
print("[1/6] Checking data file...")
if os.path.exists("data/2024_public_lar_csv.csv"):
    ok("2024_public_lar_csv.csv found")
else:
    fail("2024_public_lar_csv.csv NOT found")
    print("      Download from:")
    print("      https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/2023")

# 2. Generate database
print("\n[2/6] Generating database...")
result = subprocess.run(
    [sys.executable, "data/loader.py"],
    capture_output=True,
    text=True,
)
print("     ", result.stdout.strip().split("\n")[-1])
if os.path.exists("data/fairlend.db"):
    ok("fairlend.db created")
else:
    fail("fairlend.db NOT created")

# 3. Test imports
print("\n[3/6] Testing imports...")
modules = [
    ("database.db", "get_connection"),
    ("models.baseline", "prepare_features"),
    ("explainability.shap_analysis", "explain_applicant"),
    ("explainability.adverse_action", "generate_adverse_action_letter"),
    ("fairness.disparate_impact", "compute_full_disparate_impact"),
    ("fairness.proxy_detection", "detect_proxy_correlations"),
    ("compliance.eu_ai_act", "generate_compliance_report"),
    ("compliance.nl_query", "answer_question"),
]
for module, obj in modules:
    try:
        mod = __import__(module, fromlist=[obj])
        getattr(mod, obj)
        ok(f"{module}.{obj}")
    except Exception as e:
        fail(f"{module} — {e}")

# 4. Check trained models
print("\n[4/6] Checking trained models...")
model_files = [
    "models/saved/logistic_regression.joblib",
    "models/saved/decision_tree.joblib",
    "models/saved/lgbm_unconstrained.joblib",
    "models/saved/lgbm_fair.joblib",
    "models/saved/model_comparison.csv",
]
for f in model_files:
    if os.path.exists(f):
        ok(f)
    else:
        fail(f"{f} — run training pipeline first")

# 5. Run test suite
print("\n[5/6] Running test suite...")
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"],
    capture_output=True,
    text=True,
)
last_line = result.stdout.strip().split("\n")[-1]
print(f"      {last_line}")
if result.returncode == 0:
    ok("All tests passed")
else:
    fail("Tests failed — check output above")

# 6. Syntax check dashboard
print("\n[6/6] Checking dashboard syntax...")
result = subprocess.run(
    [sys.executable, "-m", "py_compile", "dashboard/app.py"],
    capture_output=True,
)
if result.returncode == 0:
    ok("dashboard/app.py — no syntax errors")
else:
    fail(f"Syntax error: {result.stderr.decode()}")

# Summary
print()
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
passed = sum(checks)
total = len(checks)
if all(checks):
    print(f" ALL {total} CHECKS PASSED")
    print(" Ready to push to GitHub and Hugging Face")
else:
    print(f" {passed}/{total} checks passed")
    print(" Fix the failing items above")
print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

if __name__ == "__main__":
    pass
