import pytest
import os
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.baseline import prepare_features, compute_metrics
from database.db import get_ml_features

DB_EXISTS = os.path.exists("data/fairlend.db")
MODELS_EXIST = os.path.exists("models/saved/lgbm_unconstrained.joblib")


@pytest.mark.skipif(not DB_EXISTS, reason="Run loader.py first")
class TestDataPipeline:

    def test_db_has_correct_row_count(self, db_path):
        from database.db import get_dataset_summary
        s = get_dataset_summary()
        assert s["total_applications"] >= 100_000
        assert s["total_applications"] <= 500_000

    def test_approval_rate_is_realistic(self, db_path):
        from database.db import get_dataset_summary
        s = get_dataset_summary()
        rate = float(s["overall_approval_rate"])
        assert 40.0 <= rate <= 95.0

    def test_all_required_columns_present(self, db_path):
        from database.db import get_ml_features
        df = get_ml_features()
        required = [
            "loan_amount", "income", "dti_ratio",
            "loan_to_income_ratio", "is_joint_application",
            "is_conventional", "loan_type", "lien_status",
            "state", "age", "approved",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_nulls_in_numeric_columns(self, db_path):
        from database.db import get_ml_features
        df = get_ml_features()
        for col in ["loan_amount", "income", "dti_ratio"]:
            assert df[col].isna().sum() == 0, f"Nulls in {col}"

    def test_target_is_binary(self, db_path):
        from database.db import get_ml_features
        df = get_ml_features()
        assert set(df["approved"].unique()).issubset({0, 1})

    def test_race_simplified_has_correct_categories(self, db_path):
        from database.db import get_ml_features
        df = get_ml_features()
        allowed = {
            "White", "Black or African American",
            "Asian", "Hispanic or Latino",
            "Other or Not Provided",
        }
        found = set(df["race_simplified"].unique())
        assert found.issubset(allowed), f"Unexpected races: {found - allowed}"


@pytest.mark.skipif(
    not DB_EXISTS or not MODELS_EXIST,
    reason="Run full training pipeline first",
)
class TestModels:

    def test_logistic_regression_loads(self, saved_dir):
        model = joblib.load(f"{saved_dir}/logistic_regression.joblib")
        assert model is not None

    def test_decision_tree_loads(self, saved_dir):
        model = joblib.load(f"{saved_dir}/decision_tree.joblib")
        assert model is not None

    def test_lgbm_loads(self, saved_dir):
        model = joblib.load(f"{saved_dir}/lgbm_unconstrained.joblib")
        assert model is not None

    def test_fair_model_loads(self, saved_dir):
        model = joblib.load(f"{saved_dir}/lgbm_fair.joblib")
        assert model is not None

    def test_lgbm_auc_above_threshold(self, saved_dir):
        comparison = __import__("pandas").read_csv(
            f"{saved_dir}/model_comparison.csv"
        )
        lgbm_row = comparison[comparison["Model"] == "LightGBM"].iloc[0]
        assert float(lgbm_row["AUC-ROC"]) >= 0.75, (
            f"LightGBM AUC too low: {lgbm_row['AUC-ROC']}"
        )

    def test_fair_model_passes_dpr(self, saved_dir):
        comparison = __import__("pandas").read_csv(
            f"{saved_dir}/model_comparison.csv"
        )
        fair_row = comparison[
            comparison["Model"] == "LightGBM + Fairlearn"
        ].iloc[0]
        assert str(fair_row["Passes DPR"]).upper() == "YES", (
            f"Fair model does not pass DPR: {fair_row['DPR']}"
        )

    def test_fair_model_dpr_above_080(self, saved_dir):
        comparison = __import__("pandas").read_csv(
            f"{saved_dir}/model_comparison.csv"
        )
        fair_row = comparison[
            comparison["Model"] == "LightGBM + Fairlearn"
        ].iloc[0]
        assert float(fair_row["DPR"]) >= 0.80, (
            f"DPR {fair_row['DPR']} below legal threshold 0.80"
        )

    def test_fair_model_eod_lower_than_unconstrained(self, saved_dir):
        comparison = __import__("pandas").read_csv(
            f"{saved_dir}/model_comparison.csv"
        )
        lgbm_eod = float(
            comparison[comparison["Model"] == "LightGBM"]["EOD"].iloc[0]
        )
        fair_eod = float(
            comparison[
                comparison["Model"] == "LightGBM + Fairlearn"
            ]["EOD"].iloc[0]
        )
        assert fair_eod < lgbm_eod, (
            f"Fair EOD {fair_eod} not lower than LightGBM EOD {lgbm_eod}"
        )

    def test_no_protected_attributes_in_features(self, saved_dir):
        feature_cols = joblib.load(f"{saved_dir}/feature_cols.joblib")
        assert "race_simplified" not in feature_cols, (
            "race_simplified must not be a model feature"
        )
        assert "sex_simplified" not in feature_cols, (
            "sex_simplified must not be a model feature"
        )

    def test_prepare_features_returns_correct_shape(self, db_path):
        df = get_ml_features()
        X, y, feature_cols, encoders = prepare_features(df)
        assert X.shape[0] == len(df)
        assert X.shape[1] == len(feature_cols)
        assert len(y) == len(df)
