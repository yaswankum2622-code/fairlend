import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np


def pytest_configure(config):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(" FairLend — Test Suite")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


@pytest.fixture(scope="session")
def db_path():
    return "data/fairlend.db"


@pytest.fixture(scope="session")
def saved_dir():
    return "models/saved"


@pytest.fixture(scope="session")
def sample_applicant():
    return pd.Series({
        "loan_amount": 350.0,
        "income": 95.0,
        "dti_ratio": 0.32,
        "loan_to_income_ratio": 3.68,
        "is_joint_application": 0,
        "is_conventional": 1,
        "loan_type": "1",
        "lien_status": "1",
        "state": "CA",
        "age": "35-44",
        "race_simplified": "White",
        "sex_simplified": "Male",
        "approved": 0,
    })
