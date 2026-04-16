"""
FairLend | compliance/nl_query.py | Natural language to SQL compliance chat
"""

import os
import re
import sys
import sqlite3
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))

DB_PATH = Path(__file__).parent.parent / "data" / "fairlend.db"

# Full schema injected into every Gemini call
# Gemini must know the exact table and column names
DB_SCHEMA = """
Table: applications
Columns:
  year                    TEXT    -- activity year (e.g. '2024')
  lender_id               TEXT    -- lender identifier (LEI code)
  state                   TEXT    -- US state code (e.g. 'CA', 'TX')
  county                  TEXT    -- county code
  msa_code                TEXT    -- metropolitan statistical area code
  race                    TEXT    -- full race description from HMDA
  race_simplified         TEXT    -- one of: 'White', 'Black or African American',
                                  --   'Asian', 'Hispanic or Latino',
                                  --   'Other or Not Provided'
  sex                     TEXT    -- full sex description from HMDA
  sex_simplified          TEXT    -- one of: 'Male', 'Female', 'Not Provided'
  age                     TEXT    -- applicant age range (e.g. '25-34', '45-54')
  action_taken            TEXT    -- '1'=originated(approved), '3'=denied
  loan_type               TEXT    -- '1'=conventional, '2'=FHA
  lien_status             TEXT    -- lien position
  loan_amount             REAL    -- loan amount in thousands of dollars
  income                  REAL    -- annual income in thousands of dollars
  dti_ratio               REAL    -- debt-to-income ratio as decimal
  loan_to_income_ratio    REAL    -- loan amount / income
  is_joint_application    INTEGER -- 1 if joint application, 0 if individual
  is_conventional         INTEGER -- 1 if conventional loan, 0 otherwise
  approved                INTEGER -- 1 if approved, 0 if denied (TARGET VARIABLE)

Row count: ~500,000
"""

SYSTEM_PROMPT = f"""
You are a SQL expert helping a mortgage lending compliance officer
audit loan application data for fair lending compliance.

DATABASE SCHEMA:
{DB_SCHEMA}

RULES FOR SQL GENERATION:
1. Only generate SELECT statements — never INSERT, UPDATE, DELETE, DROP
2. Always use approved = 1 for approved loans, approved = 0 for denied
3. When computing approval rates use: ROUND(AVG(approved) * 100, 2)
4. Always include a LIMIT clause — maximum 100 rows
5. Use race_simplified not race (cleaner categories)
6. Use sex_simplified not sex
7. For percentage comparisons always show both count and percentage
8. If the question mentions "worst" or "highest gap" use ORDER BY DESC
9. If the question is ambiguous ask for clarification — do not guess

Return ONLY the SQL query. No explanation. No markdown. No backticks.
Just the raw SQL that can be executed directly.
"""

BANNED_SQL_WORDS = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "ATTACH", "DETACH", "PRAGMA"]


def _sanitize_sql(sql: str) -> str:
    """Normalize and validate generated SQL before execution."""
    clean_sql = sql.strip().replace("```sql", "").replace("```", "").strip()
    clean_sql = clean_sql.rstrip(";").strip()
    upper_sql = clean_sql.upper()

    if not upper_sql.startswith(("SELECT", "WITH")):
        raise ValueError("Only SELECT queries are permitted.")

    for word in BANNED_SQL_WORDS:
        if re.search(rf"\b{word}\b", upper_sql):
            raise ValueError(f"Query contains banned keyword: {word}")

    if ";" in clean_sql:
        raise ValueError("Only single-statement queries are permitted.")

    limit_match = re.search(r"\bLIMIT\s+(\d+)\b", upper_sql)
    if limit_match:
        limit_value = int(limit_match.group(1))
        if limit_value > 100:
            clean_sql = re.sub(r"(?i)\bLIMIT\s+\d+\b", "LIMIT 100", clean_sql)
    else:
        clean_sql = f"{clean_sql}\nLIMIT 100"

    return clean_sql


def generate_sql(question: str) -> str:
    """
    Convert plain English question to SQL using Gemini 2.5 Flash.
    Returns raw SQL string ready to execute.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return _fallback_sql(question)

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            "gemini-2.5-flash-preview-04-17",
            system_instruction=SYSTEM_PROMPT
        )

        response = model.generate_content(
            question,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=500,
            )
        )

        return _sanitize_sql(response.text)

    except Exception:
        return _fallback_sql(question)


def run_query(sql: str) -> pd.DataFrame:
    """
    Execute SQL on fairlend.db.
    Returns DataFrame. Raises clear error if SQL is invalid.
    """
    safe_sql = _sanitize_sql(sql)

    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(safe_sql, conn)
        return df
    except Exception as e:
        raise ValueError(f"SQL execution failed: {e}")
    finally:
        conn.close()


def interpret_results(
    question: str,
    sql: str,
    results: pd.DataFrame
) -> str:
    """
    Write a plain English interpretation of query results.
    Uses Gemini 2.5 Flash.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or results.empty:
        return _fallback_interpretation(results)

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")

        results_text = results.to_string(index=False, max_rows=20)

        prompt = f"""
A mortgage lending compliance officer asked this question:
"{question}"

The SQL query was:
{sql}

The results are:
{results_text}

Write a 2-3 sentence plain English interpretation of these results
for a compliance officer who needs to understand the fair lending
implications. Focus on any disparities, patterns, or compliance
risks visible in the data. Be specific — use the actual numbers.
Do not repeat the question. Do not explain what SQL is.
"""

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.4,
                max_output_tokens=300,
            )
        )

        return response.text.strip()

    except Exception:
        return _fallback_interpretation(results)


def answer_question(question: str) -> dict:
    """
    Full pipeline: question → SQL → results → interpretation.

    Returns dict:
      sql:            the generated SQL
      results:        pd.DataFrame of query results
      interpretation: plain English summary
      error:          None or error message string
    """
    try:
        sql = generate_sql(question)
        results = run_query(sql)
        interpretation = interpret_results(question, sql, results)

        return {
            "sql": sql,
            "results": results,
            "interpretation": interpretation,
            "error": None
        }

    except ValueError as e:
        return {
            "sql": "",
            "results": pd.DataFrame(),
            "interpretation": "",
            "error": str(e)
        }


def _fallback_sql(question: str) -> str:
    """Fallback SQL when Gemini is unavailable."""
    q = question.lower()

    if "approval rate" in q and "race" in q:
        return """
SELECT
    race_simplified,
    COUNT(*)                            AS total_applications,
    SUM(approved)                       AS approved_count,
    ROUND(AVG(approved) * 100, 2)      AS approval_rate_pct
FROM applications
GROUP BY race_simplified
ORDER BY approval_rate_pct DESC
LIMIT 20
""".strip()

    if "state" in q and ("lowest approval" in q or "lowest approval rates" in q):
        return """
SELECT
    state,
    COUNT(*)                            AS total_applications,
    SUM(approved)                       AS approved_count,
    ROUND(AVG(approved) * 100, 2)      AS approval_rate_pct
FROM applications
GROUP BY state
ORDER BY approval_rate_pct ASC
LIMIT 20
""".strip()

    if "average loan amount" in q and ("approved" in q or "denied" in q):
        return """
SELECT
    CASE
        WHEN approved = 1 THEN 'Approved'
        ELSE 'Denied'
    END AS application_status,
    COUNT(*)                           AS total_applications,
    ROUND(AVG(loan_amount), 2)         AS avg_loan_amount_k
FROM applications
GROUP BY approved
ORDER BY approved DESC
LIMIT 20
""".strip()

    return """
SELECT
    race_simplified,
    COUNT(*)                            AS total_applications,
    SUM(approved)                       AS approved_count,
    ROUND(AVG(approved) * 100, 2)      AS approval_rate_pct
FROM applications
GROUP BY race_simplified
ORDER BY approval_rate_pct DESC
LIMIT 20
""".strip()


def _fallback_interpretation(results: pd.DataFrame) -> str:
    """Fallback interpretation when Gemini is unavailable."""
    if results.empty:
        return "No results returned for this query."

    if "approval_rate_pct" in results.columns:
        top_row = results.iloc[0]
        group_col = next((col for col in ["race_simplified", "state", "application_status"] if col in results.columns), None)
        if group_col:
            return (
                f"Query returned {len(results)} rows. "
                f"The first listed result is {top_row[group_col]} at {top_row['approval_rate_pct']:.2f}% approval. "
                f"Review the spread across groups for potential fair lending disparities."
            )

    if "avg_loan_amount_k" in results.columns:
        return (
            f"Query returned {len(results)} rows. "
            f"The table compares average loan sizes across decision outcomes. "
            f"Use the difference in average loan amount to assess whether denied applications cluster at larger requested balances."
        )

    return (
        f"Query returned {len(results)} rows. "
        f"Review the table above for fair lending patterns. "
        f"Pay attention to differences across demographic or geographic groups."
    )
