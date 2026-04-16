"""
FairLend | database/db.py | SQLite connection and query helpers
"""

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent.parent / "data" / "fairlend.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def run_query(sql: str, params: tuple = ()) -> list[dict]:
    """Execute SELECT query, return list of dicts."""
    with get_connection() as conn:
        cursor = conn.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]


def run_query_df(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Execute SELECT query, return DataFrame."""
    with get_connection() as conn:
        return pd.read_sql_query(sql, conn, params=params)


def get_approval_stats() -> pd.DataFrame:
    """Approval rates by race and sex for dashboard summary."""
    sql = """
    SELECT
        race_simplified,
        sex_simplified,
        COUNT(*)                            AS total,
        SUM(approved)                       AS approved_count,
        ROUND(AVG(approved) * 100, 2)      AS approval_rate,
        ROUND(AVG(loan_amount), 0)          AS avg_loan_amount,
        ROUND(AVG(income), 0)               AS avg_income
    FROM applications
    GROUP BY race_simplified, sex_simplified
    ORDER BY race_simplified, sex_simplified
    """
    return run_query_df(sql)


def get_disparate_impact() -> pd.DataFrame:
    """
    Disparate impact ratio per race group.
    Reference group: White applicants.
    Legal threshold: ratio must be >= 0.8
    """
    sql = """
    WITH rates AS (
        SELECT
            race_simplified,
            ROUND(AVG(approved) * 100, 2) AS approval_rate
        FROM applications
        GROUP BY race_simplified
    ),
    white_rate AS (
        SELECT approval_rate AS white_approval
        FROM rates
        WHERE race_simplified = 'White'
    )
    SELECT
        r.race_simplified,
        r.approval_rate,
        ROUND(r.approval_rate / w.white_approval, 3) AS disparate_impact_ratio,
        CASE
            WHEN r.approval_rate / w.white_approval >= 0.8
            THEN 'PASS'
            ELSE 'FAIL'
        END AS legal_status
    FROM rates r, white_rate w
    ORDER BY r.approval_rate DESC
    """
    return run_query_df(sql)


def get_dataset_summary() -> dict:
    """High-level stats for dashboard sidebar."""
    sql = """
    SELECT
        COUNT(*)                            AS total_applications,
        SUM(approved)                       AS total_approved,
        ROUND(AVG(approved) * 100, 1)      AS overall_approval_rate,
        COUNT(DISTINCT state)               AS states,
        COUNT(DISTINCT lender_id)           AS lenders,
        COUNT(DISTINCT race_simplified)     AS race_groups,
        MIN(year)                           AS data_year
    FROM applications
    """
    return run_query(sql)[0]


def get_ml_features() -> pd.DataFrame:
    """
    Load features and target for model training.
    Returns clean DataFrame ready for sklearn.
    """
    sql = """
    SELECT
        loan_amount,
        income,
        dti_ratio,
        loan_to_income_ratio,
        is_joint_application,
        is_conventional,
        loan_type,
        lien_status,
        state,
        race_simplified,
        sex_simplified,
        age,
        approved
    FROM applications
    """
    return run_query_df(sql)
