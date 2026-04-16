import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DB_EXISTS = os.path.exists("data/fairlend.db")
MODELS_EXIST = os.path.exists("models/saved/lgbm_unconstrained.joblib")
BOTH_EXIST = DB_EXISTS and MODELS_EXIST


@pytest.mark.skipif(not DB_EXISTS, reason="Run loader.py first")
class TestDisparateImpact:

    def test_disparate_impact_returns_dataframe(self, db_path):
        from database.db import get_disparate_impact
        di = get_disparate_impact()
        assert len(di) > 0
        assert "disparate_impact_ratio" in di.columns

    def test_white_group_is_reference(self, db_path):
        from database.db import get_disparate_impact
        di = get_disparate_impact()
        white_row = di[di["race_simplified"] == "White"]
        assert len(white_row) == 1
        assert abs(float(white_row["disparate_impact_ratio"].iloc[0]) - 1.0) < 0.01

    def test_all_ratios_positive(self, db_path):
        from database.db import get_disparate_impact
        di = get_disparate_impact()
        assert (di["disparate_impact_ratio"] > 0).all()

    def test_all_ratios_at_most_2(self, db_path):
        from database.db import get_disparate_impact
        di = get_disparate_impact()
        assert (di["disparate_impact_ratio"] <= 2.0).all()

    def test_approval_rates_between_0_and_100(self, db_path):
        from database.db import get_disparate_impact
        di = get_disparate_impact()
        assert (di["approval_rate"] >= 0).all()
        assert (di["approval_rate"] <= 100).all()


@pytest.mark.skipif(not DB_EXISTS, reason="Run loader.py first")
class TestProxyDetection:

    def test_proxy_detection_returns_results(self):
        from fairness.proxy_detection import detect_proxy_correlations
        proxies = detect_proxy_correlations()
        assert len(proxies) > 0
        assert "correlation_with_race" in proxies.columns
        assert "risk_level" in proxies.columns

    def test_correlations_between_0_and_1(self):
        from fairness.proxy_detection import detect_proxy_correlations
        proxies = detect_proxy_correlations()
        assert (proxies["correlation_with_race"] >= 0).all()
        assert (proxies["correlation_with_race"] <= 1).all()

    def test_income_is_high_risk_proxy(self):
        from fairness.proxy_detection import detect_proxy_correlations
        proxies = detect_proxy_correlations()
        income_row = proxies[proxies["feature"] == "income"]
        assert len(income_row) == 1
        assert income_row["risk_level"].iloc[0] in ["HIGH", "MEDIUM"]

    def test_risk_levels_are_valid(self):
        from fairness.proxy_detection import detect_proxy_correlations
        proxies = detect_proxy_correlations()
        valid = {"HIGH", "MEDIUM", "LOW"}
        found = set(proxies["risk_level"].unique())
        assert found.issubset(valid), f"Invalid risk levels: {found - valid}"


@pytest.mark.skipif(not BOTH_EXIST, reason="Run full pipeline first")
class TestComplianceReport:

    def test_report_generates(self):
        from compliance.eu_ai_act import generate_compliance_report
        report = generate_compliance_report()
        assert "checks" in report
        assert "passed" in report
        assert "overall_status" in report

    def test_report_has_10_checks(self):
        from compliance.eu_ai_act import generate_compliance_report
        report = generate_compliance_report()
        assert report["total"] == 10

    def test_report_is_compliant(self):
        from compliance.eu_ai_act import generate_compliance_report
        report = generate_compliance_report()
        assert report["overall_status"] == "COMPLIANT", (
            f"Expected COMPLIANT, got {report['overall_status']}"
        )

    def test_all_checks_pass(self):
        from compliance.eu_ai_act import generate_compliance_report
        report = generate_compliance_report()
        failing = [
            check["article"] for check in report["checks"]
            if check["status"] != "PASS"
        ]
        assert len(failing) == 0, f"Failing checks: {failing}"


@pytest.mark.skipif(not BOTH_EXIST, reason="Run full pipeline first")
class TestSHAP:

    def test_shap_returns_explanation(self, sample_applicant):
        from explainability.shap_analysis import explain_applicant
        expl = explain_applicant(sample_applicant)
        assert "prediction" in expl
        assert "top_factors" in expl
        assert "shap_values" in expl

    def test_prediction_between_0_and_1(self, sample_applicant):
        from explainability.shap_analysis import explain_applicant
        expl = explain_applicant(sample_applicant)
        assert 0.0 <= expl["prediction"] <= 1.0

    def test_top_factors_has_5_items(self, sample_applicant):
        from explainability.shap_analysis import explain_applicant
        expl = explain_applicant(sample_applicant)
        assert len(expl["top_factors"]) == 5

    def test_no_protected_attributes_in_shap(self, sample_applicant):
        from explainability.shap_analysis import explain_applicant
        expl = explain_applicant(sample_applicant)
        names = [feature[0] for feature in expl["shap_values"]]
        assert "race_simplified" not in names, (
            "race_simplified must not appear in SHAP output"
        )
        assert "sex_simplified" not in names, (
            "sex_simplified must not appear in SHAP output"
        )

    def test_shap_waterfall_returns_figure(self, sample_applicant):
        import plotly.graph_objects as go
        from explainability.shap_analysis import explain_applicant, plot_waterfall
        expl = explain_applicant(sample_applicant)
        fig = plot_waterfall(expl, "Test Applicant")
        assert isinstance(fig, go.Figure)
