# Scope

## What Is Built

| Component | Status |
|---|---|
| HMDA 2024 data pipeline | ✅ |
| Logistic regression baseline | ✅ |
| Decision tree | ✅ |
| LightGBM unconstrained | ✅ |
| LightGBM + Fairlearn constrained | ✅ |
| Model comparison — accuracy + fairness | ✅ |
| SHAP per-applicant explainability | ✅ |
| ECOA adverse action letter | ✅ |
| Disparate impact report | ✅ |
| Proxy variable detection | ✅ |
| Compliance chat (NL → SQL) | ✅ |
| EU AI Act Article 9 checklist | ✅ |
| Streamlit dashboard — 6 pages | ✅ |
| 34 pytest tests | ✅ |
| GitHub Actions CI | ✅ |
| Hugging Face Spaces deployment | ✅ |

## What Is Out of Scope

| Feature | Reason |
|---|---|
| IBM AIF360 adversarial debiasing | Future work |
| Counterfactual fairness | Future work |
| PostgreSQL | SQLite sufficient for MVP |
| Docker | Add after MVP |
| Real-time monitoring | Future work |
| Multi-jurisdiction compliance | Future work |

## Design Decisions

**Why LightGBM over neural networks**
LightGBM outperforms neural networks on tabular structured data.
SHAP TreeExplainer gives exact Shapley values for tree models.
Neural networks would break ECOA explainability requirements.

**Why Fairlearn ExponentiatedGradient**
Mathematically proven to find Pareto-optimal accuracy-fairness
tradeoff. Published by Microsoft Research. Used in Azure ML.
Compatible with any sklearn estimator.

**Why SQLite over PostgreSQL**
Zero setup. Runs identically on every machine and on HF Spaces.
All SQL is portable to PostgreSQL with no changes.

**Why HMDA over synthetic data**
Real data has real distribution shapes, real seasonal patterns,
real proxy correlations, and real class imbalances.
Synthetic data cannot replicate the Goldman Sachs problem.
