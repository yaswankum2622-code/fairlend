"""
FairLend | ci_setup.py | Generate synthetic database for CI testing.
The real HMDA 2024 CSV is 1.8GB and cannot run in GitHub Actions.
This script generates a small realistic synthetic dataset
that is sufficient for running the test suite.
"""

import sqlite3
import pandas as pd
import numpy as np
import os
from pathlib import Path

np.random.seed(42)
N = 10_000

print("Generating synthetic HMDA-like dataset for CI...")

races = [
    "White", "Black or African American",
    "Asian", "Other or Not Provided"
]
race_weights = [0.63, 0.08, 0.08, 0.21]
race_approvals = [0.86, 0.71, 0.88, 0.82]

sexes = ["Male", "Female", "Not Provided"]
sex_weights = [0.32, 0.23, 0.45]
ages = ["25-34", "35-44", "45-54", "55-64", "65-74", "<25", ">74"]
states = [
    "CA", "TX", "FL", "NY", "IL", "PA", "OH", "GA",
    "NC", "MI", "NJ", "VA", "WA", "AZ", "MA"
]

race_col = np.random.choice(races, N, p=race_weights)
sex_col = np.random.choice(sexes, N, p=sex_weights)
age_col = np.random.choice(ages, N)
state_col = np.random.choice(states, N)

income = np.random.lognormal(4.2, 0.6, N).clip(20, 500)
loan_amount = (income * np.random.uniform(2.5, 5.5, N)).clip(50, 2000)
dti = np.random.beta(2, 5, N).clip(0.05, 0.65)
lti = (loan_amount / income).clip(0.5, 15)

approved = np.array([
    np.random.binomial(
        1,
        race_approvals[races.index(r)] * (
            0.95 if dti[i] < 0.35 else
            0.85 if dti[i] < 0.50 else 0.70
        )
    )
    for i, r in enumerate(race_col)
])

df = pd.DataFrame({
    "year": ["2024"] * N,
    "lender_id": [f"LEI{np.random.randint(1000, 9999)}" for _ in range(N)],
    "state": state_col,
    "county": [f"{np.random.randint(1, 999):03d}" for _ in range(N)],
    "msa_code": [str(np.random.randint(10000, 99999)) for _ in range(N)],
    "race": race_col,
    "race_simplified": race_col,
    "sex": sex_col,
    "sex_simplified": sex_col,
    "age": age_col,
    "action_taken": np.where(approved, "1", "3"),
    "loan_type": np.random.choice(["1", "2"], N, p=[0.75, 0.25]),
    "lien_status": np.random.choice(["1", "2"], N, p=[0.95, 0.05]),
    "loan_amount": loan_amount.round(2),
    "income": income.round(2),
    "dti_ratio": dti.round(4),
    "loan_to_income_ratio": lti.round(4),
    "is_joint_application": np.random.binomial(1, 0.3, N),
    "is_conventional": np.random.binomial(1, 0.75, N),
    "approved": approved,
})

os.makedirs("data", exist_ok=True)
os.makedirs("models/saved", exist_ok=True)

conn = sqlite3.connect("data/fairlend.db")
df.to_sql("applications", conn, if_exists="replace", index=False)
conn.close()

total = len(df)
approved_n = int(df["approved"].sum())
print(f"Generated {total:,} synthetic applications")
print(f"Approved: {approved_n:,} ({approved_n / total * 100:.1f}%)")
print("Saved to: data/fairlend.db")

# Quick sanity checks
assert len(df) == N
assert df["approved"].nunique() == 2
assert df["race_simplified"].nunique() == 4
assert df["approved"].mean() > 0.5
assert df["approved"].mean() < 0.95
print("ALL CI ASSERTIONS PASSED")
