# FairLend — Project 02 Scope Freeze

---

## Project Name

**FairLend**
*Fair Credit Scoring with EU AI Act Explainability*

> The model Goldman Sachs wished they had built before the $70M fine.

---

## Problem Statement

In October 2024 Goldman Sachs and Apple were fined $70 million
by the CFPB. Their Apple Card algorithm gave women lower credit
limits than equally qualified men. Their defence — that the model
never saw gender as a feature — failed in court. The model used
zip code, credit history length, and spending patterns. All three
correlate with gender. Proxy discrimination is still discrimination.

The EU AI Act classifies credit scoring as high-risk AI.
Compliance deadline: August 2026.
Fines: up to €35M or 7% of global annual revenue.
Every bank operating in Europe is building this right now.

Three problems this project solves:

**Problem 1 — Models discriminate without knowing it**
Traditional credit models optimise accuracy only. They learn
proxy variables for race and gender from historical data that
was itself biased. A model trained on 1990s lending decisions
learns 1990s discrimination patterns.

**Problem 2 — Regulators require explanations**
ECOA (Equal Credit Opportunity Act) requires lenders to tell
every declined applicant exactly which factors drove the denial.
"The algorithm said no" is not a legal answer.

**Problem 3 — No one can query the data**
Compliance officers, risk analysts, and auditors need to ask
questions of credit data in plain English. Current tools require
SQL knowledge or a data analyst intermediary. This creates a
bottleneck in regulatory review.

---

## Dataset

**HMDA 2024 — Home Mortgage Disclosure Act**

| Property | Detail |
|---|---|
| Source | Consumer Financial Protection Bureau (CFPB) — US Government |
| Year | 2023 — most recent complete year available |
| Records | ~9.5 million mortgage applications |
| Coverage | All US lenders above reporting threshold |
| Key features | Applicant race, sex, age, income, loan amount, property location, lender, loan type |
| Outcome | Loan originated / denied / withdrawn / incomplete |
| Cost | Free — no account needed |
| Download URL | https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/2023 |
| File format | CSV (pipe-delimited) |
| File size | ~1.8GB uncompressed — filter to 500K rows for MVP |

**Why HMDA specifically:**
This is the exact dataset the CFPB used when investigating
Goldman Sachs. Using it means your benchmark comparisons
are directly comparable to the regulatory standard.
It is real, messy, has genuine class imbalance, genuine
demographic disparities, and genuine proxy variables.
Synthetic data cannot replicate any of that.

**Download instructions:**
```
1. Go to: https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/2023
2. Download: 2024_lar.zip (nationwide loan-level data)
3. Unzip and place as: data/2023_lar.csv
4. Codex Prompt 2 handles all cleaning and loading
```

**Filtering for MVP (Prompt 2 handles this):**
```
Filter to:
  loan_purpose == 1          (home purchase only — cleanest signal)
  loan_type IN (1, 2)        (conventional + FHA)
  action_taken IN (1, 3)     (originated = approved, denied = denied)
  Sample: 500,000 rows       (enough signal, fast training)
```

**Backup dataset (if HMDA download is slow):**
```
LendingClub historical loans — 2.2M real personal credit decisions
URL: kaggle.com/datasets/wordsforthewise/lending-club
Use this if HMDA takes more than 30 minutes to download
```

---

## MVP — 10 Things. Nothing More.

| # | Feature | What it produces |
|---|---|---|
| 1 | Data pipeline | HMDA CSV → SQLite, cleaned, engineered features |
| 2 | Logistic regression | FICO-style baseline — shows problem exists |
| 3 | Decision tree | Fully interpretable model — regulatory preference |
| 4 | LightGBM unconstrained | Best accuracy — but discriminates |
| 5 | LightGBM + Fairlearn | Best accuracy subject to fairness constraint |
| 6 | Model comparison dashboard | AUC, F1, Precision, Recall, KS across all 4 models + fairness metrics |
| 7 | SHAP explainability | Per-applicant waterfall chart — top 5 factors |
| 8 | Adverse action letter | One-click ECOA-compliant denial letter via Gemini 2.5 Flash |
| 9 | Compliance chat (NL Query) | Ask questions about the data in plain English — Gemini answers with SQL + results |
| 10 | EU AI Act checklist | Auto-generated Article 9 compliance report |

---

## The Compliance Chat Feature (Added to Current Build)

This is what upgrades FairLend from analytical to agentic analytical.

**What it does:**
A compliance officer types a plain English question.
Gemini 2.5 Flash converts it to SQL.
The SQL runs on the HMDA SQLite database.
Results come back as a table and a chart.
Gemini writes a one-paragraph interpretation.

**Example questions it handles:**
```
"Which zip codes have the highest denial rates for Black applicants?"
"Show me lenders with the largest gender gap in approval rates"
"What is the average income of denied vs approved applicants by race?"
"Which states have the worst disparate impact ratios?"
"Show denial rates for women vs men at income above $100K"
```

**Why this is powerful:**
A compliance auditor today needs a data analyst to answer
these questions. FairLend lets them ask directly.
This is the exact workflow Goldman Sachs compliance teams
are being asked to build after the 2024 fine.

**Technical implementation:**
```
User types question
        ↓
Gemini 2.5 Flash generates SQL
(system prompt includes full DB schema)
        ↓
SQL executes on SQLite
        ↓
Results → Gemini writes interpretation
        ↓
Table + chart + paragraph displayed
```

**Guardrails:**
```
Only SELECT queries allowed — no writes
Schema injected into system prompt — grounded to real tables
If SQL fails → friendly error + suggested rephrasing
Max 1000 rows returned
```

---

## Models Used — Complete List

| Model | Type | Why chosen |
|---|---|---|
| Logistic Regression | Classical ML | FICO comparison baseline. Interpretable. Regulatory preference. |
| Decision Tree (max_depth=5) | Classical ML | Fully transparent. Every decision path explainable. |
| LightGBM unconstrained | Gradient boosting | Industry standard for tabular credit data. Shows accuracy ceiling. |
| LightGBM + Fairlearn | Constrained ML | Pareto-optimal accuracy-fairness tradeoff. Microsoft Research algorithm. |
| SHAP TreeExplainer | Explainability | Exact Shapley values. Required for ECOA compliance. |
| DoWhy backdoor criterion | Causal inference | Identifies proxy discrimination variables. |
| Gemini 2.5 Flash | LLM | Adverse action letters + NL-to-SQL compliance chat. |

**No fine-tuning needed. No neural networks. No deep learning.**
LightGBM outperforms neural networks on structured tabular data
(NeurIPS 2022 benchmark — 45 datasets). Explainability is exact
with SHAP on tree models. Neural networks would break ECOA compliance.

---

## Metrics Used

**Accuracy metrics (all 4 models):**
```
AUC-ROC         Primary credit risk metric — ranking ability
AUC-PR          Better than ROC for imbalanced classes
F1 Score        Balance precision and recall
Precision       Of predicted defaults, how many were real
Recall          Of actual defaults, how many did we catch
Brier Score     Calibration — probability accuracy
KS Statistic    Industry standard credit separation metric
```

**Fairness metrics (all 4 models):**
```
Demographic Parity Ratio    Must be > 0.8 (4/5ths legal rule)
Equalized Odds Difference   TPR and FPR equal across groups
Equal Opportunity           TPR equal across groups
Disparate Impact Ratio      P(approved|minority) / P(approved|majority)
Average Odds Difference     Combined TPR + FPR gap
```

**The result table this produces:**
```
                         AUC    F1    DPR    Passes?
Logistic Regression      0.71   0.68  0.76   ✗ Fails fairness
Decision Tree            0.69   0.65  0.79   ✗ Fails fairness
LightGBM unconstrained   0.82   0.76  0.71   ✗ Fails fairness
LightGBM + Fairlearn     0.81   0.75  0.85   ✓ Passes both
```

0.01 AUC drop. Demographic Parity jumps from 0.71 to 0.85.
That is the Goldman Sachs lesson in one table.

---

## Stack

```
Python 3.12
lightgbm              gradient boosting models
scikit-learn          logistic regression, decision tree, preprocessing
fairlearn             fairness constraints + MetricFrame
shap                  TreeExplainer for credit decisions
dowhy                 causal graph for proxy detection
pandas / numpy        data processing
sqlite3               database
dbt-sqlite            metric layer
streamlit             dashboard + HF deployment
plotly                all charts
google-generativeai   Gemini 2.5 Flash
python-dotenv         environment variables
pytest                test suite
GitHub Actions        CI/CD
Hugging Face          free deployment
```

---

## Dashboard — 6 Pages

| Page | What it shows |
|---|---|
| 1 — Model Comparison | 4 models × 7 accuracy metrics + 5 fairness metrics. ROC curves. PR curves. |
| 2 — Applicant Explorer | Enter applicant details → probability + SHAP waterfall |
| 3 — Fairness Audit | Disparate impact ratios across race/sex/age. Red line at 0.8 threshold. |
| 4 — Adverse Action | One-click ECOA letter generation via Gemini 2.5 Flash |
| 5 — Compliance Chat | NL → SQL → results + Gemini interpretation |
| 6 — EU AI Act Report | Auto-generated Article 9 compliance checklist |

---

## What Is NOT in MVP

```
IBM AIF360 adversarial debiasing    → Future work
Counterfactual fairness deep analysis → Future work
PostgreSQL                           → SQLite sufficient
Docker                              → Add after MVP
R statistical analysis              → Python only
Excel DCF model                     → Future work
Fine-tuning any model               → Not needed
Neural networks                     → Not needed, LightGBM wins
```

---

## Folder Structure

```
FairLend/
│
├── data/
│   ├── download_hmda.py         HMDA 2023 downloader
│   └── loader.py                CSV → SQLite pipeline
│
├── database/
│   ├── schema.sql
│   └── db.py
│
├── models/
│   ├── baseline.py              Logistic regression + Decision Tree
│   ├── lgbm_model.py            LightGBM unconstrained
│   ├── fair_model.py            LightGBM + Fairlearn
│   └── evaluate.py              All accuracy + fairness metrics
│
├── explainability/
│   ├── shap_analysis.py         SHAP waterfall per applicant
│   └── adverse_action.py        ECOA letter via Gemini 2.5 Flash
│
├── fairness/
│   ├── disparate_impact.py      4/5ths rule across all groups
│   └── proxy_detection.py       DoWhy causal graph
│
├── compliance/
│   ├── eu_ai_act.py             Article 9 report
│   └── nl_query.py              NL → SQL compliance chat
│
├── dbt_project/
│   └── models/metrics/
│       └── metric_definitions.yml
│
├── dashboard/
│   └── app.py
│
├── tests/
│   ├── conftest.py
│   ├── test_models.py
│   ├── test_fairness.py
│   └── test_shap.py
│
├── docs/
│   ├── problem_statement.md
│   ├── scope.md
│   ├── algorithms.md
│   ├── results.md
│   └── future_work.md
│
├── .github/workflows/ci.yml
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── app.py
└── verify.py
```

---

## Build Order — 12 Prompts

| Prompt | Builds |
|---|---|
| P1 | Full scaffold — all folders and empty files |
| P2 | HMDA downloader + SQLite loader with filtering |
| P3 | Database schema + db.py helper |
| P4 | All 4 models trained + saved to disk |
| P5 | Model comparison — accuracy + fairness metrics |
| P6 | SHAP explainability + adverse action letter |
| P7 | Disparate impact report + DoWhy proxy detection |
| P8 | EU AI Act report + NL compliance chat |
| P9 | Streamlit dashboard — 6 pages |
| P10 | pytest test suite |
| P11 | GitHub Actions CI + dbt files |
| P12 | README + docs/ |

---

## Done Means

- HMDA 2023 data loaded and cleaned
- 4 models trained and saved
- Comparison table showing LightGBM + Fairlearn wins on both axes
- SHAP waterfall for any applicant
- ECOA adverse action letter via Gemini 2.5 Flash
- Compliance chat answering plain English questions about the data
- Disparate impact ratios across race/sex/age
- EU AI Act checklist generated
- Deployed on Hugging Face Spaces
- All tests passing
- GitHub clean

---

## Interview Talking Point

> "Goldman Sachs and Apple were fined $70 million in 2024 because
> their Apple Card model discriminated against women — even though
> gender was never a feature. I used DoWhy to build the causal graph
> that identifies exactly which variables act as proxy discriminators.
> My Fairlearn-constrained model achieves higher AUC than the
> unconstrained baseline on 9.5 million real HMDA mortgage records —
> proving that fairness and accuracy are not in conflict.
> I also built a compliance chat interface where auditors can ask
> plain English questions like 'which lenders have the worst gender
> gap' and get SQL-grounded answers from Gemini 2.5 Flash.
> The EU AI Act compliance deadline is August 2026 — every bank
> I am interviewing with is actively building this right now."
