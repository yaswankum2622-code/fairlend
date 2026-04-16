---
title: FairLend
emoji: ⚖️
colorFrom: purple
colorTo: green
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: true
license: mit
short_description: Fair Credit Scoring with EU AI Act Explainability
---

<div align="center">

<br>

# ⚖️ FairLend

**Fair Credit Scoring with EU AI Act Explainability**

<br>

[![Live Demo](https://img.shields.io/badge/🚀%20Open%20Live%20App-534AB7?style=for-the-badge&logoColor=white)](https://yaswtutu-fairlend.hf.space)
[![HF Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-FFD21E?style=for-the-badge&logoColor=black)](https://huggingface.co/spaces/yaswtutu/fairlend)
[![CI](https://img.shields.io/github/actions/workflow/status/yaswankum2622-code/FairLend/ci.yml?style=for-the-badge&label=34%20Tests&logo=github&logoColor=white)](https://github.com/yaswankum2622-code/FairLend/actions)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-1D9E75?style=for-the-badge)](LICENSE)

<br>

![LightGBM](https://img.shields.io/badge/Model-LightGBM-FF6B35?style=flat-square)
![Fairlearn](https://img.shields.io/badge/Fairness-Fairlearn-534AB7?style=flat-square)
![SHAP](https://img.shields.io/badge/XAI-SHAP-1D9E75?style=flat-square)
![Gemini](https://img.shields.io/badge/AI-Gemini%202.5%20Flash-4285F4?style=flat-square&logo=google&logoColor=white)
![HMDA](https://img.shields.io/badge/Data-HMDA%202024-orange?style=flat-square)
![EU AI Act](https://img.shields.io/badge/Compliance-EU%20AI%20Act-003399?style=flat-square)

</div>

---

## Why this exists

Goldman Sachs and Apple were fined $70 million in 2024 for
algorithmic credit discrimination. Their defence — that the
model never saw gender as a feature — failed in court.

The model had learned to discriminate through income patterns,
loan-to-income ratios, and debt obligations. All legitimate
financial variables. All correlated with race and gender due
to decades of historical lending bias. Proxy discrimination
is still discrimination.

The EU AI Act classifies credit scoring as high-risk AI.
Compliance deadline is August 2026. Fines run up to €35 million
or 7% of global annual revenue.

FairLend is what their model should have been.

---

## See it live

<div align="center">

[![Open in Hugging Face Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-xl.svg)](https://yaswtutu-fairlend.hf.space)

**No login · No install · Opens in your browser**

</div>

---

## What the data shows

500,000 real US mortgage applications from HMDA 2024.

```text
Approval rate by race (raw data):
  Asian                           88.6%
  White                           86.2%
  Other or Not Provided           82.0%
  Black or African American       71.1%   ← 15pp gap
```

The 15-point gap exists before any model is trained.
The unconstrained model makes it worse.
The Fairlearn-constrained model closes it.

---

## The model comparison

| Model | AUC-ROC | DPR | Passes legal threshold? |
|---|---|---|---|
| Logistic Regression | 0.68 | 0.67 | ❌ NO |
| Decision Tree | 0.81 | 0.79 | ❌ NO |
| LightGBM | 0.84 | 0.78 | ❌ NO |
| **LightGBM + Fairlearn** | **0.71** | **0.90** | **✅ YES** |

DPR = Demographic Parity Ratio. Legal threshold = 0.80 (CFPB 4/5ths rule).

Three of four models fail the legal standard despite never
seeing race as a direct input. The Fairlearn constraint costs
13 percentage points of AUC. Goldman Sachs paid $70M instead.

---

## Six things it does

**Model Comparison**
Four models head to head. Accuracy metrics and fairness metrics
on the same screen. Red threshold line at 0.80. Only one bar
is green.

**Applicant Explorer**
Enter any applicant profile. Get approval probability plus a
SHAP waterfall showing exactly which financial factors drove
the decision — and by how much. No protected attributes in
the explanation.

**Fairness Audit**
Disparate impact ratios across all race and sex groups.
Proxy variable heatmap showing which financial features carry
the highest correlation with race. Income and
loan-to-income ratio are the top two.

**Adverse Action Letter**
One click generates an ECOA-compliant denial letter via
Gemini 2.5 Flash. Explains the denial in plain English.
References only legitimate financial factors.
Includes CFPB contact information and applicant rights.

**Compliance Chat**
Type a plain English question about the HMDA data.
Gemini converts it to SQL. The query runs on the database.
Results come back as a table with an interpretation.

**EU AI Act Report**
Auto-generated Article 9 compliance checklist.
10 checks. All green. Downloadable as a text file.

---

## The proxy discrimination story

```text
Race is not a model feature.
The model still discriminates.

Why:
  income              correlation with race = 0.175  HIGH RISK
  loan_to_income_ratio                     = 0.143  HIGH RISK
  dti_ratio                                = 0.115  MEDIUM
  loan_amount                              = 0.100  MEDIUM

These are real financial metrics.
Their correlation with race is a historical artefact.
The unconstrained model learns and amplifies it.
The Fairlearn constraint corrects for it.
```

---

## Quick start

```bash
git clone https://github.com/yaswankum2622-code/FairLend.git
cd FairLend

pip install -r requirements.txt

# Download HMDA 2024 data from CFPB and place as:
# data/2024_public_lar_csv.csv

python data/loader.py
python models/baseline.py
python models/lgbm_model.py
python models/fair_model.py
python models/evaluate.py

streamlit run dashboard/app.py
```

---

## Dataset

**HMDA 2024 — Home Mortgage Disclosure Act**
Official US government mortgage application data from the CFPB.
500,000 applications filtered from ~9.5 million nationwide records.

[Download from CFPB](https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/2023)
→ Save as `data/2024_public_lar_csv.csv`

---

## Run the tests

```bash
pytest tests/ -v
```

```text
tests/test_models.py    ·  12 passed
tests/test_fairness.py  ·  22 passed
─────────────────────────────────────
34 passed in 128s
```

---

## Stack

```text
Language      Python 3.11
ML            LightGBM · scikit-learn
Fairness      Fairlearn (ExponentiatedGradient)
Explainability SHAP TreeExplainer
Causal        DoWhy proxy detection
Dashboard     Streamlit · Plotly
AI            Google Gemini 2.5 Flash
Data          pandas · SQLite · dbt-sqlite
CI/CD         GitHub Actions
Hosting       Hugging Face Spaces (free)
Dataset       HMDA 2024 — CFPB official data
```

---

## Project structure

```text
FairLend/
│
├── data/
│   └── loader.py                HMDA CSV → SQLite pipeline
│
├── models/
│   ├── baseline.py              Logistic regression + Decision Tree
│   ├── lgbm_model.py            LightGBM unconstrained
│   ├── fair_model.py            LightGBM + Fairlearn
│   └── evaluate.py              Full comparison table
│
├── explainability/
│   ├── shap_analysis.py         SHAP per-applicant waterfall
│   └── adverse_action.py        ECOA letter via Gemini 2.5 Flash
│
├── fairness/
│   ├── disparate_impact.py      4/5ths rule analysis
│   └── proxy_detection.py       Correlation with protected attributes
│
├── compliance/
│   ├── eu_ai_act.py             Article 9 compliance report
│   └── nl_query.py              NL → SQL compliance chat
│
├── dbt_project/                 Governed metric definitions
├── dashboard/app.py             Streamlit 6-page dashboard
├── tests/                       34 pytest tests
├── docs/                        Technical documentation
└── .github/workflows/           CI on every push
```

---

## Documentation

| File | Contents |
|---|---|
| [`docs/problem_statement.md`](docs/problem_statement.md) | Why credit AI discriminates and what it costs |
| [`docs/scope.md`](docs/scope.md) | What is in MVP, what is out |
| [`docs/algorithms.md`](docs/algorithms.md) | Fairlearn, SHAP, DoWhy — the math |
| [`docs/results.md`](docs/results.md) | Full model comparison and findings |
| [`docs/future_work.md`](docs/future_work.md) | What comes next |

---

## What is next

- Counterfactual fairness analysis
- IBM AIF360 adversarial debiasing as fourth approach
- PostgreSQL + Docker production stack
- Real-time monitoring with metric drift detection
- Multi-jurisdiction compliance — UK FCA, US CFPB, EU AI Act

---

<div align="center">

**Built by Yashwanth**
M.Tech CSE · Business Analytics · VIT Chennai · Bengaluru

*Star the repo if this was useful.*  ⭐

</div>
