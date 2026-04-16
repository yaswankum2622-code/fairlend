import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import sqlite3
import joblib
import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from database.db import (
    get_approval_stats,
    get_disparate_impact,
    get_dataset_summary,
    get_ml_features,
)
from models.baseline import prepare_features
from explainability.shap_analysis import explain_applicant, plot_waterfall
from explainability.adverse_action import generate_adverse_action_letter
from fairness.disparate_impact import (
    compute_full_disparate_impact,
    plot_disparate_impact_race,
    plot_disparate_impact_comparison,
)
from fairness.proxy_detection import (
    detect_proxy_correlations,
    plot_proxy_heatmap,
    get_proxy_summary,
)
from compliance.eu_ai_act import generate_compliance_report
from compliance.nl_query import answer_question

DB_PATH = "data/fairlend.db"
SAVED_DIR = "models/saved"

CHART_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#FFFFFF",
    font=dict(
        family="Inter, sans-serif",
        color="#1a1a18",
        size=12,
    ),
    margin=dict(l=40, r=30, t=50, b=40),
    hoverlabel=dict(
        bgcolor="#1a1a18",
        bordercolor="#1a1a18",
        font=dict(
            family="JetBrains Mono, monospace",
            color="#FFFFFF",
            size=11,
        ),
    ),
)

AXIS_STYLE = dict(
    gridcolor="#F0F0F0",
    linecolor="#E8E8E8",
    tickfont=dict(color="#888888", size=11),
)

EXTERNAL_FIG_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#FFFFFF",
    font=dict(
        family="Inter, sans-serif",
        color="#1a1a18",
        size=12,
    ),
    margin=dict(l=40, r=30, t=50, b=40),
)

DEFAULT_STATES = [
    "CA",
    "TX",
    "FL",
    "NY",
    "IL",
    "PA",
    "OH",
    "GA",
    "NC",
    "MI",
    "NJ",
    "VA",
    "WA",
    "AZ",
    "MA",
    "Other",
]

st.set_page_config(
    page_title="FairLend",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
.stApp { background: #F7F8FA; }
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
.block-container {
    padding: 2rem 2.5rem 4rem !important;
    max-width: 1400px !important;
}
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #EBEBEB !important;
}
[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 0.5px solid #E8E8E8;
    border-radius: 10px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    font-family: 'JetBrains Mono', monospace !important;
    color: #1a1a18 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
    color: #888888 !important;
}
.stButton > button {
    background: #534AB7 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.stButton > button:hover {
    background: #443DA0 !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #534AB7 !important;
    border-bottom: 2px solid #534AB7 !important;
}
.page-title {
    font-size: 1.7rem;
    font-weight: 700;
    color: #1a1a18;
    letter-spacing: -0.03em;
    margin-bottom: 0.25rem;
}
.page-subtitle {
    font-size: 0.82rem;
    color: #888888;
    margin-bottom: 1.5rem;
}
.chat-user {
    background: #EEEDFE;
    border: 0.5px solid #C5C0F0;
    border-radius: 12px 12px 2px 12px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
}
.chat-ai {
    background: #FFFFFF;
    border: 0.5px solid #E8E8E8;
    border-left: 3px solid #534AB7;
    border-radius: 2px 12px 12px 12px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.9rem;
}
hr { border-color: #EBEBEB !important; }
</style>
""",
    unsafe_allow_html=True,
)


def header(title: str, subtitle: str):
    st.markdown(
        f'<div class="page-title">{title}</div>'
        f'<div class="page-subtitle">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def db_check():
    if not os.path.exists(DB_PATH):
        st.error("Database not found. Run: python data/loader.py")
        st.stop()


def models_check():
    files = [
        f"{SAVED_DIR}/lgbm_unconstrained.joblib",
        f"{SAVED_DIR}/lgbm_fair.joblib",
        f"{SAVED_DIR}/logistic_regression.joblib",
        f"{SAVED_DIR}/decision_tree.joblib",
        f"{SAVED_DIR}/model_comparison.csv",
    ]
    missing = [path for path in files if not os.path.exists(path)]
    if missing:
        st.error("Models not trained yet. Run the training pipeline.")
        st.code(
            "python models/baseline.py\n"
            "python models/lgbm_model.py\n"
            "python models/fair_model.py\n"
            "python models/evaluate.py"
        )
        st.stop()


def safe_fig(fig):
    fig.update_layout(**EXTERNAL_FIG_THEME)
    return fig


@st.cache_data(show_spinner=False)
def load_state_options():
    if not os.path.exists(DB_PATH):
        return DEFAULT_STATES

    conn = sqlite3.connect(DB_PATH)
    try:
        query = """
        SELECT state, COUNT(*) AS n
        FROM applications
        WHERE state IS NOT NULL
          AND TRIM(state) <> ''
        GROUP BY state
        ORDER BY n DESC
        LIMIT 15
        """
        states = pd.read_sql_query(query, conn)
    finally:
        conn.close()

    values = states["state"].astype(str).tolist()
    if "Other" not in values:
        values.append("Other")
    return values


@st.cache_data(show_spinner=False)
def load_model_comparison():
    return pd.read_csv(f"{SAVED_DIR}/model_comparison.csv")


def initialise_session():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_input" not in st.session_state:
        st.session_state.chat_input = ""


def set_chat_input(question: str):
    st.session_state.chat_input = question


def clear_chat_history():
    st.session_state.chat_history = []


def render_model_comparison_page():
    try:
        db_check()
        models_check()
        header(
            "Model Comparison",
            "4 models × accuracy + fairness metrics · HMDA 2024 · 500K applications",
        )

        comparison = load_model_comparison()
        tab_acc, tab_fair, tab_roc = st.tabs(
            ["Accuracy Metrics", "Fairness Metrics", "ROC Curves"]
        )

        with tab_acc:
            st.markdown("#### Accuracy metrics — all 4 models")
            st.dataframe(
                comparison[
                    ["Model", "AUC-ROC", "F1", "Precision", "Recall", "KS Stat"]
                ],
                use_container_width=True,
                hide_index=True,
            )

            colors = ["#6C63FF", "#1D9E75", "#185FA5", "#BA7517"]
            fig_acc = go.Figure()
            for i, (_, row) in enumerate(comparison.iterrows()):
                fig_acc.add_trace(
                    go.Bar(
                        name=row["Model"],
                        x=["AUC-ROC", "F1", "Precision", "Recall"],
                        y=[
                            row["AUC-ROC"],
                            row["F1"],
                            row["Precision"],
                            row["Recall"],
                        ],
                        marker_color=colors[i % len(colors)],
                        opacity=0.85,
                        text=[
                            f"{row['AUC-ROC']:.3f}",
                            f"{row['F1']:.3f}",
                            f"{row['Precision']:.3f}",
                            f"{row['Recall']:.3f}",
                        ],
                        textposition="outside",
                        textfont=dict(size=9),
                    )
                )

            fig_acc.update_layout(
                **CHART_THEME,
                barmode="group",
                height=400,
                title="Accuracy metrics — 4 models compared",
                yaxis_title="Score",
                xaxis=dict(**AXIS_STYLE),
                yaxis=dict(range=[0, 1.15], **AXIS_STYLE),
                legend=dict(
                    bgcolor="#FFFFFF",
                    bordercolor="#E8E8E8",
                    borderwidth=1,
                ),
            )
            st.plotly_chart(fig_acc, use_container_width=True)

        with tab_fair:
            st.markdown("#### Fairness metrics — legal threshold = 0.80 DPR")
            st.dataframe(
                comparison[["Model", "DPR", "EOD", "Passes DPR"]],
                use_container_width=True,
                hide_index=True,
            )

            bar_colors = [
                "#1D9E75" if str(row["Passes DPR"]).upper() == "YES" else "#E24B4A"
                for _, row in comparison.iterrows()
            ]

            fig_dpr = go.Figure()
            fig_dpr.add_trace(
                go.Bar(
                    x=comparison["Model"],
                    y=comparison["DPR"],
                    marker_color=bar_colors,
                    text=[f"DPR={value:.3f}" for value in comparison["DPR"]],
                    textposition="outside",
                    textfont=dict(size=11),
                )
            )
            fig_dpr.add_hline(
                y=0.80,
                line_dash="dash",
                line_color="#E24B4A",
                line_width=2,
                annotation_text="Legal minimum 0.80",
                annotation_font=dict(color="#E24B4A", size=11),
            )
            fig_dpr.add_hline(
                y=0.85,
                line_dash="dot",
                line_color="#BA7517",
                line_width=1.5,
                annotation_text="Best practice 0.85",
                annotation_font=dict(color="#BA7517", size=10),
            )
            fig_dpr.update_layout(
                **CHART_THEME,
                height=400,
                title="Demographic Parity Ratio — legal threshold 0.80",
                yaxis_title="DPR",
                xaxis=dict(**AXIS_STYLE),
                yaxis=dict(range=[0, 1.15], **AXIS_STYLE),
                showlegend=False,
            )
            st.plotly_chart(fig_dpr, use_container_width=True)

            st.info(
                "**DPR ≥ 0.80** is the minimum legal threshold. "
                "The constrained Fairlearn model is the only saved model that clears it.",
                icon="ℹ️",
            )

        with tab_roc:
            st.markdown("#### AUC-ROC summary")
            st.dataframe(
                comparison[["Model", "AUC-ROC", "KS Stat"]],
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Full ROC curves are not persisted. AUC-ROC and KS Stat provide the saved discrimination summary."
            )
    except Exception as exc:
        st.error(f"Model comparison failed: {exc}")


def render_applicant_explorer_page():
    try:
        db_check()
        models_check()
        header(
            "Applicant Explorer",
            "Enter applicant details → approval probability + SHAP explanation",
        )

        st.markdown("#### Applicant details")
        states = load_state_options()
        c1, c2, c3 = st.columns(3)

        with c1:
            loan_amount = st.number_input("Loan amount ($K)", 50, 2000, 350, 10)
            income = st.number_input("Annual income ($K)", 20, 500, 95, 5)
            dti = st.slider("Debt-to-income ratio", 0.0, 0.65, 0.32, 0.01)

        with c2:
            age = st.selectbox(
                "Age range",
                ["25-34", "35-44", "45-54", "55-64", "65-74", "<25", ">74"],
            )
            loan_type_label = st.selectbox("Loan type", ["Conventional", "FHA"])
            lien_status_label = st.selectbox(
                "Lien status", ["First lien", "Subordinate"]
            )

        with c3:
            state = st.selectbox("State", states)
            application_type = st.selectbox(
                "Application type", ["Individual", "Joint"]
            )

        lti = round(loan_amount / income, 4) if income > 0 else 0.0
        loan_type = "1" if loan_type_label == "Conventional" else "2"
        lien_status = "1" if lien_status_label == "First lien" else "2"
        is_joint = 1 if application_type == "Joint" else 0

        if st.button("Analyse Application", type="primary"):
            with st.spinner("Running SHAP analysis..."):
                applicant = pd.Series(
                    {
                        "loan_amount": float(loan_amount),
                        "income": float(income),
                        "dti_ratio": float(dti),
                        "loan_to_income_ratio": float(lti),
                        "is_joint_application": int(is_joint),
                        "is_conventional": 1 if loan_type == "1" else 0,
                        "loan_type": str(loan_type),
                        "lien_status": str(lien_status),
                        "state": str(state),
                        "age": str(age),
                        "race_simplified": "White",
                        "sex_simplified": "Male",
                        "approved": 0,
                    }
                )

                explanation = explain_applicant(applicant)
                probability = explanation["prediction"]

            color = (
                "#1D9E75"
                if probability > 0.75
                else "#BA7517"
                if probability > 0.50
                else "#E24B4A"
            )
            label = (
                "Likely Approved"
                if probability > 0.75
                else "Borderline"
                if probability > 0.50
                else "Likely Denied"
            )

            st.divider()
            r1, r2, r3 = st.columns(3)
            r1.metric("Approval Probability", f"{probability * 100:.1f}%")
            r2.metric("Loan-to-Income Ratio", f"{lti:.2f}x")
            r3.markdown(
                f"""
                <div style='padding:0.9rem 0'>
                  <span style='background:{color}22;color:{color};
                               border:0.5px solid {color}55;
                               border-radius:20px;padding:5px 14px;
                               font-weight:600;font-size:0.88rem;
                               font-family:JetBrains Mono,monospace'>
                    {label}
                  </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            fig_shap = safe_fig(plot_waterfall(explanation, "This Applicant"))
            st.plotly_chart(fig_shap, use_container_width=True)

            st.markdown("#### Top factors driving this decision")
            for feature, value in explanation["top_factors"]:
                direction = "FOR approval" if value > 0 else "AGAINST approval"
                factor_color = "#1D9E75" if value > 0 else "#E24B4A"
                st.markdown(
                    f"<div style='padding:6px 0;font-size:0.88rem'>"
                    f"<span style='font-family:JetBrains Mono,monospace;"
                    f"color:{factor_color};font-weight:600'>{value:+.4f}</span>"
                    f"&nbsp;·&nbsp;<b>{feature.replace('_', ' ').title()}</b>"
                    f"&nbsp;<span style='color:#888;font-size:0.8rem'>"
                    f"({direction})</span></div>",
                    unsafe_allow_html=True,
                )
    except Exception as exc:
        st.error(f"Applicant explorer failed: {exc}")


def render_fairness_audit_page():
    try:
        db_check()
        models_check()
        header(
            "Fairness Audit",
            "Disparate impact · Proxy variable detection · HMDA 2024",
        )

        with st.spinner("Computing fairness metrics..."):
            di_results = compute_full_disparate_impact()
            proxy_df = detect_proxy_correlations()
            proxy_summary = get_proxy_summary()
            comparison = load_model_comparison()

        race_df = di_results["race"]
        min_dpr = float(race_df["disparate_impact_ratio"].min())
        n_fail = int((race_df["disparate_impact_ratio"] < 0.80).sum())
        n_watch = int(
            (
                (race_df["disparate_impact_ratio"] >= 0.80)
                & (race_df["disparate_impact_ratio"] < 0.85)
            ).sum()
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Minimum DPR", f"{min_dpr:.3f}")
        m2.metric("Groups Failing (< 0.80)", str(n_fail))
        m3.metric("Groups on Watch (< 0.85)", str(n_watch))
        m4.metric("High-Risk Proxies", str(proxy_summary["high_risk"]))

        st.divider()
        tab_race, tab_proxy, tab_compare = st.tabs(
            ["Race Analysis", "Proxy Variables", "Before vs After"]
        )

        with tab_race:
            st.plotly_chart(
                safe_fig(plot_disparate_impact_race(di_results)),
                use_container_width=True,
            )

            display_race = race_df[
                ["race", "approval_rate", "disparate_impact_ratio", "passes_legal"]
            ].copy()
            display_race.columns = [
                "Race Group",
                "Approval Rate (%)",
                "DI Ratio",
                "Passes Legal",
            ]
            display_race["Status"] = display_race["DI Ratio"].apply(
                lambda value: "PASS"
                if value >= 0.85
                else "MONITOR"
                if value >= 0.80
                else "FAIL"
            )
            st.dataframe(display_race, use_container_width=True, hide_index=True)

            black_rows = race_df[race_df["race"] == "Black or African American"]
            if not black_rows.empty:
                black_di = float(black_rows.iloc[0]["disparate_impact_ratio"])
                st.warning(
                    f"**Black or African American applicants** show a DI ratio of {black_di:.3f}. "
                    "That is legally passing, but still close enough to the boundary to monitor.",
                    icon="⚠️",
                )

        with tab_proxy:
            st.markdown("#### Features correlated with race")
            st.info(
                "These variables are economically legitimate, but they still act as proxy pathways for race."
                " That is why the unconstrained model can drift into discriminatory outcomes.",
                icon="🔍",
            )
            st.plotly_chart(safe_fig(plot_proxy_heatmap()), use_container_width=True)

            display_proxy = proxy_df[
                ["feature", "correlation_with_race", "risk_level"]
            ].copy()
            display_proxy.columns = [
                "Feature",
                "Correlation with Race",
                "Risk Level",
            ]
            st.dataframe(display_proxy, use_container_width=True, hide_index=True)

        with tab_compare:
            st.plotly_chart(
                safe_fig(plot_disparate_impact_comparison()),
                use_container_width=True,
            )
            unconstrained = comparison.loc[comparison["Model"] == "LightGBM", "DPR"]
            constrained = comparison.loc[
                comparison["Model"] == "LightGBM + Fairlearn", "DPR"
            ]
            if not unconstrained.empty and not constrained.empty:
                st.success(
                    f"Fairlearn improved DPR from **{float(unconstrained.iloc[0]):.3f}** "
                    f"to **{float(constrained.iloc[0]):.3f}**. "
                    "That is the measurable tradeoff between raw accuracy and legal defensibility.",
                    icon="✅",
                )
    except Exception as exc:
        st.error(f"Fairness audit failed: {exc}")


def render_adverse_action_page():
    try:
        db_check()
        models_check()
        header(
            "Adverse Action Letter",
            "ECOA-compliant denial letter · Gemini 2.5 Flash",
        )

        st.markdown("#### Applicant details")
        c1, c2, c3 = st.columns(3)

        with c1:
            loan_amount = st.number_input("Loan amount ($K)", 50, 2000, 280, 10)
            income = st.number_input("Annual income ($K)", 20, 500, 72, 5)
        with c2:
            dti = st.slider("DTI ratio", 0.0, 0.65, 0.48, 0.01)
            state = st.selectbox("State", ["CA", "TX", "FL", "NY", "GA", "NC", "Other"])
        with c3:
            age = st.selectbox("Age range", ["25-34", "35-44", "45-54", "55-64"])
            loan_type_label = st.selectbox("Loan type", ["Conventional", "FHA"])

        if st.button("Generate ECOA Adverse Action Letter", type="primary"):
            with st.spinner("Gemini is writing the letter..."):
                lti = round(loan_amount / income, 4) if income > 0 else 0.0
                loan_type = "1" if loan_type_label == "Conventional" else "2"
                applicant = pd.Series(
                    {
                        "loan_amount": float(loan_amount),
                        "income": float(income),
                        "dti_ratio": float(dti),
                        "loan_to_income_ratio": float(lti),
                        "is_joint_application": 0,
                        "is_conventional": 1 if loan_type == "1" else 0,
                        "loan_type": str(loan_type),
                        "lien_status": "1",
                        "state": state,
                        "age": age,
                        "race_simplified": "Other or Not Provided",
                        "sex_simplified": "Not Provided",
                        "approved": 0,
                    }
                )

                explanation = explain_applicant(applicant)
                letter = generate_adverse_action_letter(
                    {
                        "loan_amount": loan_amount,
                        "income": income,
                        "dti_ratio": dti,
                        "loan_to_income_ratio": lti,
                        "state": state,
                    },
                    explanation,
                    explanation["prediction"],
                )

            st.success("Letter generated", icon="✅")
            st.markdown(
                f"""<div style='background:#FFFFFF;border:0.5px solid #E8E8E8;
                    border-left:4px solid #534AB7;border-radius:10px;
                    padding:1.5rem 1.8rem;line-height:1.9;font-size:0.88rem;
                    color:#1a1a18;white-space:pre-wrap'>{letter}</div>""",
                unsafe_allow_html=True,
            )
            st.download_button(
                "⬇  Download Letter (.txt)",
                letter,
                file_name="adverse_action_letter.txt",
                mime="text/plain",
            )
            st.caption(
                "Gemini 2.5 Flash · ECOA Regulation B · No protected attributes · FairLend v1.0"
            )
    except Exception as exc:
        st.error(f"Adverse action page failed: {exc}")


def render_compliance_chat_page():
    try:
        db_check()
        initialise_session()
        header(
            "Compliance Chat",
            "Plain English questions about HMDA 2024 · Gemini 2.5 Flash → SQL",
        )

        st.markdown("#### Try one of these")
        suggested = [
            "What is the approval rate by race?",
            "Which states have the lowest approval rates?",
            "Show denial rates for Female vs Male applicants",
            "Which lenders have the most applications?",
            "What is average loan amount for approved vs denied?",
            "Show approval rates by age group",
        ]

        cols = st.columns(3)
        for i, question in enumerate(suggested):
            cols[i % 3].button(
                question,
                key=f"sq_{i}",
                on_click=set_chat_input,
                args=(question,),
            )

        st.divider()
        question = st.text_input(
            "Ask a question",
            placeholder="e.g. Which states have the highest denial rates for Black applicants?",
            key="chat_input",
        )

        if st.button("Ask", type="primary", key="ask_btn"):
            cleaned = question.strip()
            if not cleaned:
                st.warning("Enter a question before asking.", icon="ℹ️")
            else:
                with st.spinner("Querying HMDA data..."):
                    result = answer_question(cleaned)
                st.session_state.chat_history.append(
                    {"question": cleaned, "result": result}
                )

        for entry in reversed(st.session_state.chat_history):
            question_text = entry["question"]
            result = entry["result"]

            st.markdown(
                f'<div class="chat-user">🧑 {question_text}</div>',
                unsafe_allow_html=True,
            )

            if result.get("error"):
                st.error(result["error"])
            else:
                results_df = result.get("results", pd.DataFrame())
                if results_df is not None and not results_df.empty:
                    st.dataframe(
                        results_df,
                        use_container_width=True,
                        hide_index=True,
                    )
                if result.get("interpretation"):
                    st.markdown(
                        f'<div class="chat-ai">⚖️ {result["interpretation"]}</div>',
                        unsafe_allow_html=True,
                    )
                with st.expander("View generated SQL", expanded=False):
                    st.code(result.get("sql", ""), language="sql")

        if st.session_state.chat_history:
            st.button("Clear history", key="clear_btn", on_click=clear_chat_history)
    except Exception as exc:
        st.error(f"Compliance chat failed: {exc}")


def render_eu_ai_act_report_page():
    try:
        db_check()
        header(
            "EU AI Act Compliance Report",
            "High-Risk AI · Credit Scoring · Article 9 · Deadline August 2026",
        )

        with st.spinner("Generating compliance report..."):
            report = generate_compliance_report()

        passed = report["passed"]
        total = report["total"]
        status = report["overall_status"]
        status_color = "#1D9E75" if status == "COMPLIANT" else "#E24B4A"

        s1, s2, s3, s4 = st.columns(4)
        s1.markdown(
            f"""
            <div style='background:#FFFFFF;border:0.5px solid #E8E8E8;
                        border-top:3px solid {status_color};border-radius:10px;
                        padding:1rem;text-align:center'>
              <div style='font-size:1.5rem;font-weight:700;
                          font-family:JetBrains Mono,monospace;
                          color:{status_color}'>{status}</div>
              <div style='font-size:0.65rem;font-weight:600;text-transform:uppercase;
                          letter-spacing:.07em;color:#888;margin-top:4px'>
                Overall Status
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        s2.metric("Checks Passed", f"{passed}/{total}")
        s3.metric("Model Version", report["model_version"])
        s4.metric("Compliance Deadline", report["deadline"])

        st.divider()
        st.markdown("#### Article-by-article checklist")

        for check in report["checks"]:
            icon = "✅" if check["status"] == "PASS" else "❌"
            background = "#E1F5EE" if check["status"] == "PASS" else "#FCEBEB"
            border = "#9FE1CB" if check["status"] == "PASS" else "#F09595"
            st.markdown(
                f"""
                <div style='background:{background};border:0.5px solid {border};
                            border-radius:10px;padding:0.9rem 1.2rem;
                            margin-bottom:0.6rem'>
                  <div style='display:flex;justify-content:space-between;align-items:center'>
                    <div>
                      <code style='font-family:JetBrains Mono,monospace;
                                   font-size:0.75rem;color:#534AB7'>
                        {check['article']}
                      </code>
                      <span style='font-weight:600;font-size:0.88rem;
                                   color:#1a1a18;margin-left:10px'>
                        {check['requirement']}
                      </span>
                    </div>
                    <span>{icon}</span>
                  </div>
                  <div style='color:#555;font-size:0.78rem;margin-top:6px'>
                    {check['evidence']}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        report_text = (
            f"EU AI Act Compliance Report — FairLend\n"
            f"Generated: {report['generated_date']}\n"
            f"Model: {report['model_version']}\n"
            f"Regulation: {report['regulation']}\n"
            f"Deadline: {report['deadline']}\n"
            f"Status: {report['overall_status']}\n"
            f"Checks: {report['passed']}/{report['total']}\n\n"
        )
        for check in report["checks"]:
            report_text += (
                f"{check['status']}  {check['article']}  {check['requirement']}\n"
                f"Evidence: {check['evidence']}\n\n"
            )

        st.download_button(
            "⬇  Download Compliance Report (.txt)",
            report_text,
            file_name="eu_ai_act_report.txt",
            mime="text/plain",
        )
    except Exception as exc:
        st.error(f"EU AI Act report failed: {exc}")


with st.sidebar:
    st.markdown(
        """
        <div style='padding:0.2rem 0 1.4rem'>
          <div style='font-size:1.4rem;margin-bottom:6px'>⚖️</div>
          <div style='font-size:1.1rem;font-weight:700;
                      color:#1a1a18;letter-spacing:-0.02em'>
            FairLend
          </div>
          <div style='font-size:0.68rem;color:#888;
                      text-transform:uppercase;
                      letter-spacing:0.07em;margin-top:3px'>
            Fair Credit Scoring Platform
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "nav",
        [
            "📊  Model Comparison",
            "🔍  Applicant Explorer",
            "⚖️  Fairness Audit",
            "📄  Adverse Action",
            "💬  Compliance Chat",
            "✅  EU AI Act Report",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    if os.path.exists(DB_PATH):
        try:
            summary = get_dataset_summary()
            sidebar_items = [
                ("Total Applications", f"{summary['total_applications']:,}"),
                ("Approval Rate", f"{summary['overall_approval_rate']}%"),
                ("States", str(summary["states"])),
                ("Lenders", f"{summary['lenders']:,}"),
                ("Dataset", "HMDA 2024"),
            ]
            for label, value in sidebar_items:
                st.markdown(
                    f"""
                    <div style='background:#F7F8FA;border-radius:8px;
                                padding:8px 12px;margin-bottom:8px'>
                      <div style='font-size:0.63rem;font-weight:600;
                                  text-transform:uppercase;
                                  letter-spacing:.07em;color:#888;
                                  margin-bottom:3px'>{label}</div>
                      <div style='font-size:0.88rem;font-weight:600;
                                  color:#1a1a18;
                                  font-family:JetBrains Mono,monospace'>
                        {value}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        except Exception:
            pass

    st.divider()
    st.markdown(
        """
        <div style='font-size:0.68rem;color:#aaa;line-height:1.9'>
          Built by <b style='color:#534AB7'>Yashwanth</b><br>
          HMDA 2024 · 500K applications<br>
          LightGBM · Fairlearn · SHAP · DoWhy
        </div>
        """,
        unsafe_allow_html=True,
    )


if "Model Comparison" in page:
    render_model_comparison_page()
elif "Applicant Explorer" in page:
    render_applicant_explorer_page()
elif "Fairness Audit" in page:
    render_fairness_audit_page()
elif "Adverse Action" in page:
    render_adverse_action_page()
elif "Compliance Chat" in page:
    render_compliance_chat_page()
elif "EU AI Act Report" in page:
    render_eu_ai_act_report_page()
