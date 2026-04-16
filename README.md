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
<br>

<img src="https://img.shields.io/badge/⚖️%20%20FairLend-v1.0-534AB7?style=for-the-badge&labelColor=0F1117" />

<br>
<br>

[![Open Live App](https://img.shields.io/badge/🚀%20Open%20Live%20App-534AB7?style=for-the-badge&logoColor=white)](https://yaswtutu-fairlend.hf.space)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-FFD21E?style=for-the-badge&logoColor=black)](https://huggingface.co/spaces/yaswtutu/fairlend)
[![CI](https://img.shields.io/github/actions/workflow/status/yaswankum2622-code/FairLend/ci.yml?style=for-the-badge&label=34%20Tests&logo=github&logoColor=white)](https://github.com/yaswankum2622-code/FairLend/actions)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-1D9E75?style=for-the-badge)](LICENSE)

<br>

![LightGBM](https://img.shields.io/badge/LightGBM-FF6B35?style=flat-square)
![Fairlearn](https://img.shields.io/badge/Fairlearn-534AB7?style=flat-square)
![SHAP](https://img.shields.io/badge/SHAP-1D9E75?style=flat-square)
![Gemini 2.5 Flash](https://img.shields.io/badge/Gemini%202.5%20Flash-4285F4?style=flat-square&logo=google&logoColor=white)
![HMDA 2024](https://img.shields.io/badge/HMDA%202024-E8640A?style=flat-square)
![EU AI Act](https://img.shields.io/badge/EU%20AI%20Act%20Compliant-003399?style=flat-square)

<br>
<br>

### Goldman Sachs paid $70 million.
### This is the model they should have built.

<br>

</div>

---

In October 2024 the CFPB fined Goldman Sachs and Apple $70 million for
discriminatory credit decisions. Their defence was that the model never
saw gender as a feature. The regulator rejected it.

The model had learned through income, debt ratios, and loan-to-income
patterns — all legitimate financial variables, all correlated with race
and gender through decades of historical lending bias. The algorithm did
not know about gender. It did not need to.

The EU AI Act classifies credit scoring as high-risk AI. Compliance
deadline: **August 2026**. Fines: up to **€35 million or 7% of global revenue**.

FairLend trains four models on 500,000 real US mortgage applications,
proves three of them fail the legal fairness threshold, and shows that the
one that passes costs 13 percentage points of AUC — not $70 million.

---

<div align="center">

[![Open in Hugging Face Spaces](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-xl.svg)](https://yaswtutu-fairlend.hf.space)

*No login. No install. Opens in your browser.*

</div>

---

## The data

500,000 real US mortgage applications. HMDA 2024. Official CFPB data.

```
Approval rate by race — before any model is trained:

  Asian                       88.6%
  White                       86.2%
  Other or Not Provided       82.0%
  Black or African American   71.1%   ← 15 percentage point gap
```

That gap exists in the raw data. The unconstrained model makes it wider.
The constrained model closes it. The gap is not a modelling choice —
it is a reflection of 70 years of US lending history in a single CSV file.

---

## What four models tell you about $70 million

```
Model                    AUC-ROC    DPR     Passes legal threshold?
──────────────────────────────────────────────────────────────────
Logistic Regression       0.68      0.67    ✗  NO
Decision Tree             0.81      0.79    ✗  NO
LightGBM                  0.84      0.78    ✗  NO
LightGBM + Fairlearn      0.71      0.90    ✓  YES
```

DPR = Demographic Parity Ratio. Legal minimum = 0.80 under CFPB 4/5ths rule.

Race was not a model input in any of these. Three still fail.
The model with the highest accuracy (0.84 AUC) has the worst fairness (0.78 DPR).
That is the Apple Card. The Fairlearn constraint costs 13pp of AUC.
Goldman Sachs chose the 0.84 model and paid $70 million for it.

---

## Why the accurate model discriminates anyway

```
income               correlation with race = 0.175   HIGH RISK
loan_to_income_ratio                       = 0.143   HIGH RISK
dti_ratio                                  = 0.115   MEDIUM
loan_amount                                = 0.100   MEDIUM
```

Every one of these is a legitimate financial variable.
Every one carries a signal about race because of how wealth
has been distributed in the United States over the last century.
A model that optimises for accuracy learns these correlations
and amplifies them. That is proxy discrimination. It is still illegal.

The Fairlearn constraint does not remove these variables. It imposes
a mathematical bound that forces the model to find an accuracy-fairness
tradeoff rather than ignoring the tradeoff entirely.

---

## Six pages

**Model Comparison** — four models, seven accuracy metrics, five fairness
metrics, one red threshold line at 0.80. Only one bar is green.

**Applicant Explorer** — enter any financial profile. Get an approval
probability and a SHAP waterfall showing which specific factors drove
the decision and by exactly how much. No protected attributes anywhere.

**Fairness Audit** — disparate impact ratios across every race and sex
group. A proxy variable heatmap showing which financial features carry
the most race correlation. A before-and-after comparison of DPR across
all four models.

**Adverse Action** — one click generates a full ECOA-compliant denial
letter via Gemini 2.5 Flash. Plain English. Legally formatted. References
only financial factors. Includes CFPB contact information.

**Compliance Chat** — type a plain English question about the HMDA data.
Gemini converts it to SQL, runs it against 500,000 real applications,
returns a table and an interpretation. No SQL knowledge required.

**EU AI Act Report** — auto-generated Article 9 compliance checklist.
10 checks. All green. Downloadable as a text file. Deadline August 2026.

---

## Run it yourself

```bash
git clone https://github.com/yaswankum2622-code/FairLend.git
cd FairLend
pip install -r requirements.txt

# Download HMDA 2024 from CFPB → save as data/2024_public_lar_csv.csv
# https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/2023

python data/loader.py
python models/baseline.py
python models/lgbm_model.py
python models/fair_model.py
python models/evaluate.py
streamlit run dashboard/app.py
```

---

## Tests

```bash
pytest tests/ -v
# 34 passed in 128s
```

CI uses synthetic data so the 1.8GB HMDA file is not needed in GitHub Actions.
Real data runs locally and on Hugging Face Spaces.

---

## Stack

```
Python 3.11          LightGBM          scikit-learn
Fairlearn            SHAP              DoWhy
Streamlit            Plotly            SQLite
dbt-sqlite           Gemini 2.5 Flash  GitHub Actions
HMDA 2024 (CFPB)    Hugging Face Spaces
```

---

## Structure

```
FairLend/
├── data/loader.py                  HMDA CSV → SQLite
├── models/
│   ├── baseline.py                 Logistic regression + Decision Tree
│   ├── lgbm_model.py               LightGBM unconstrained
│   ├── fair_model.py               LightGBM + Fairlearn constraint
│   └── evaluate.py                 Full accuracy + fairness comparison
├── explainability/
│   ├── shap_analysis.py            Per-applicant SHAP waterfall
│   └── adverse_action.py           ECOA letter via Gemini 2.5 Flash
├── fairness/
│   ├── disparate_impact.py         4/5ths rule across all groups
│   └── proxy_detection.py          Feature correlation with race
├── compliance/
│   ├── eu_ai_act.py                Article 9 compliance report
│   └── nl_query.py                 NL → SQL → interpretation
├── dashboard/app.py                Streamlit 6-page dashboard
├── tests/                          34 pytest tests
└── docs/                           Problem, scope, algorithms, results
```

---

## Docs

[`docs/problem_statement.md`](docs/problem_statement.md) — the $70M problem in detail

[`docs/algorithms.md`](docs/algorithms.md) — Fairlearn, SHAP, DoWhy: the math

[`docs/results.md`](docs/results.md) — full comparison table and proxy analysis

[`docs/future_work.md`](docs/future_work.md) — counterfactual fairness, AIF360, production stack

---

<div align="center">

<br>

*Built by Yashwanth · M.Tech CSE Business Analytics · VIT Chennai · Bengaluru*

<br>

**⭐ Star this repo**

<br>

</div>
