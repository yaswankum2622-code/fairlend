"""
FairLend | loader | Data loading pipeline for HMDA application records.
"""

import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path(__file__).resolve().parent / "2024_public_lar_csv.csv"
DB_PATH = Path(__file__).resolve().parent / "fairlend.db"
TABLE_NAME = "applications"
SAMPLE_SIZE = 500_000
CHUNK_SIZE = 250_000

RAW_TO_RENAMED = {
    "activity_year": "year",
    "lei": "lender_id",
    "state_code": "state",
    "county_code": "county",
    "derived_race": "race",
    "derived_sex": "sex",
    "applicant_age": "age",
    "action_taken": "action_taken",
    "loan_amount": "loan_amount",
    "income": "income",
    "debt_to_income_ratio": "dti_ratio",
    "loan_type": "loan_type",
    "lien_status": "lien_status",
    "derived_msa_md": "msa_code",
}
READ_COLUMNS = list(RAW_TO_RENAMED.keys()) + ["loan_purpose"]


def detect_separator(path):
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        header = handle.readline()
    return "|" if header.count("|") > header.count(",") else ","


def normalize_text(series):
    return series.fillna("").astype(str).str.strip()


def clean_float_series(series, divide_by=None, cap_quantile=None):
    cleaned = normalize_text(series).replace({"NA": np.nan, "Exempt": np.nan, "": np.nan})
    cleaned = pd.to_numeric(cleaned, errors="coerce")
    if divide_by:
        cleaned = cleaned / divide_by
    median_value = cleaned.median()
    if pd.isna(median_value):
        median_value = 0.0
    cleaned = cleaned.fillna(median_value)
    if cap_quantile is not None:
        upper_bound = cleaned.quantile(cap_quantile)
        cleaned = cleaned.clip(upper=upper_bound)
    return cleaned.astype(float)


def parse_dti_value(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text in {"NA", "Exempt", "<20%", ">60%", ""}:
        return np.nan
    if "%-" in text:
        lower_bound = text.split("%-", 1)[0]
        return pd.to_numeric(lower_bound, errors="coerce")
    if text.endswith("%"):
        return pd.to_numeric(text[:-1], errors="coerce")
    return pd.to_numeric(text, errors="coerce")


def clean_dti(series):
    cleaned = series.apply(parse_dti_value)
    median_value = cleaned.median()
    if pd.isna(median_value):
        median_value = 0.0
    return cleaned.fillna(median_value).astype(float)


def simplify_race(value):
    text = "" if pd.isna(value) else str(value)
    if text == "White":
        return "White"
    if "Black" in text:
        return "Black or African American"
    if "Asian" in text:
        return "Asian"
    if "Hispanic" in text:
        return "Hispanic or Latino"
    return "Other or Not Provided"


def simplify_sex(value):
    text = "" if pd.isna(value) else str(value)
    if text == "Male":
        return "Male"
    if text == "Female":
        return "Female"
    return "Not Provided"


def filter_chunk(chunk):
    race = normalize_text(chunk["derived_race"])
    sex = normalize_text(chunk["derived_sex"])
    mask = (
        chunk["action_taken"].isin(["1", "3"])
        & chunk["loan_purpose"].eq("1")
        & chunk["loan_type"].isin(["1", "2"])
        & race.ne("")
        & sex.ne("")
    )
    return chunk.loc[mask, READ_COLUMNS].copy()


def sample_filtered_rows(path):
    separator = detect_separator(path)
    rng = np.random.default_rng(42)
    sample = None

    reader = pd.read_csv(
        path,
        sep=separator,
        low_memory=False,
        dtype=str,
        nrows=None,
        usecols=READ_COLUMNS,
        chunksize=CHUNK_SIZE,
    )

    for chunk in reader:
        filtered = filter_chunk(chunk)
        if filtered.empty:
            continue

        filtered["_sample_key"] = rng.random(len(filtered))
        if sample is None:
            sample = filtered
        else:
            sample = pd.concat([sample, filtered], ignore_index=True)

        if len(sample) > SAMPLE_SIZE:
            sample = sample.nsmallest(SAMPLE_SIZE, "_sample_key").reset_index(drop=True)

    if sample is None or sample.empty:
        raise ValueError("No rows matched the FairLend MVP filter.")

    sample = sample.nsmallest(min(SAMPLE_SIZE, len(sample)), "_sample_key").reset_index(drop=True)
    sample = sample.drop(columns=["_sample_key", "loan_purpose"])
    return sample


def transform_dataframe(df):
    df = df.rename(columns=RAW_TO_RENAMED)

    df["approved"] = (df["action_taken"] == "1").astype(int)

    df["loan_amount"] = clean_float_series(df["loan_amount"], divide_by=1000)
    df["income"] = clean_float_series(df["income"], cap_quantile=0.99)
    df["dti_ratio"] = clean_dti(df["dti_ratio"])

    age = normalize_text(df["age"]).replace({"8888": np.nan, "9999": np.nan, "": np.nan})
    df["age"] = age

    df["race"] = normalize_text(df["race"]).replace({"": "Not Provided"}).str.slice(0, 50)
    df["sex"] = normalize_text(df["sex"]).replace({"": "Not Provided"})
    df["state"] = normalize_text(df["state"]).replace({"": "Unknown"})

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = df["loan_amount"] / df["income"]
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    ratio_median = ratio.median()
    if pd.isna(ratio_median):
        ratio_median = 0.0
    ratio = ratio.fillna(ratio_median)
    ratio = ratio.clip(upper=ratio.quantile(0.99))
    df["loan_to_income_ratio"] = ratio.astype(float)

    df["is_joint_application"] = df["sex"].str.contains("Joint", case=False, na=False).astype(int)
    df["is_conventional"] = df["loan_type"].eq("1").astype(int)
    df["race_simplified"] = df["race"].apply(simplify_race)
    df["sex_simplified"] = df["sex"].apply(simplify_sex)

    return df


def write_to_sqlite(df, db_path):
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
    finally:
        conn.close()


def validate_output(df):
    assert len(df) > 100_000, "Too few rows — check filter logic"
    assert len(df) <= 500_000, "Too many rows — sampling failed"
    assert df["approved"].nunique() == 2, "Target has wrong values"
    assert df["approved"].mean() > 0.3, "Approval rate suspiciously low"
    assert df["approved"].mean() < 0.9, "Approval rate suspiciously high"
    assert df["loan_amount"].isna().sum() == 0, "loan_amount has NaN"
    assert df["income"].isna().sum() == 0, "income has NaN"
    assert "race_simplified" in df.columns, "race_simplified missing"
    assert "sex_simplified" in df.columns, "sex_simplified missing"
    assert "loan_to_income_ratio" in df.columns, "LTI ratio missing"
    assert os.path.exists(DB_PATH), "DB not created"
    print("ALL ASSERTIONS PASSED")


def print_summary(df):
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f" FairLend — Data Loaded Successfully")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f" Total applications:  {len(df):,}")
    print(f" Approved:            {df['approved'].sum():,} ({df['approved'].mean()*100:.1f}%)")
    print(f" Denied:              {(df['approved']==0).sum():,} ({(df['approved']==0).mean()*100:.1f}%)")
    print(f" Features:            {len(df.columns)}")
    print(f" States:              {df['state'].nunique()}")
    print(f" Lenders:             {df['lender_id'].nunique():,}")
    print(f"")
    print(f" Race breakdown:")
    for race, count in df["race_simplified"].value_counts().items():
        pct = count / len(df) * 100
        print(f"   {race:<35} {count:>7,} ({pct:.1f}%)")
    print(f"")
    print(f" Sex breakdown:")
    for sex, count in df["sex_simplified"].value_counts().items():
        pct = count / len(df) * 100
        print(f"   {sex:<35} {count:>7,} ({pct:.1f}%)")
    print(f"")
    print(f" Approval rate by race:")
    for race in df["race_simplified"].unique():
        rate = df[df["race_simplified"] == race]["approved"].mean() * 100
        print(f"   {race:<35} {rate:.1f}%")
    print(f"")
    print(f" Saved to: data/fairlend.db")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    if hasattr(os.sys.stdout, "reconfigure"):
        os.sys.stdout.reconfigure(encoding="utf-8")

    sampled = sample_filtered_rows(DATA_PATH)
    df = transform_dataframe(sampled)
    write_to_sqlite(df, DB_PATH)
    validate_output(df)
    print_summary(df)


if __name__ == "__main__":
    main()
