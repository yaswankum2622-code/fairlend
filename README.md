# <div align="center">𝗙𝗔𝗜𝗥𝗟𝗘𝗡𝗗</div>

<div align="center">

### Fair Credit Scoring with Explainability, Fairness Controls, and EU AI Act Readiness

<br>

[![Open Live App](https://img.shields.io/badge/Open%20Live%20App-534AB7?style=for-the-badge&logo=streamlit&logoColor=white)](https://yaswtutu-fairlend.hf.space)
[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Space-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/yaswtutu/fairlend)
[![GitHub Actions](https://img.shields.io/github/actions/workflow/status/yaswankum2622-code/fairlend/ci.yml?style=for-the-badge&label=CI&logo=github)](https://github.com/yaswankum2622-code/fairlend/actions)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License MIT](https://img.shields.io/badge/License-MIT-1D9E75?style=for-the-badge)](LICENSE)

<br>
<br>

<img src="https://raw.githubusercontent.com/yaswankum2622-code/fairlend/main/VISUAL%27S/Screenshot%202026-04-16%20220513.png" alt="FairLend dashboard model comparison" width="92%">

</div>

---

## Why FairLend Exists

Goldman Sachs and Apple were fined **$70 million** after their credit model produced discriminatory outcomes.  
The defense was familiar: *the model did not use gender directly*. That was not enough.

FairLend is built around the real problem:

| Problem | What Usually Happens | What FairLend Does |
|---|---|---|
| Proxy discrimination | Models learn bias through income, debt ratios, and other correlated signals | Measures proxy risk and constrains the model with Fairlearn |
| Black-box decisions | Teams cannot explain why an applicant was denied | Generates SHAP-based explanations and ECOA-ready adverse action letters |
| Compliance blind spots | Fairness checks happen late or not at all | Ships fairness audit, compliance chat, and EU AI Act reporting in one workflow |

---

## At a Glance

| Item | Value |
|---|---|
| Dataset | HMDA 2024 mortgage applications |
| Sample size | 500,000 real US applications |
| Best raw model | LightGBM, AUC-ROC `0.8402` |
| Legally safest model | LightGBM + Fairlearn, DPR `0.9025` |
| Explainability | SHAP per-applicant waterfall |
| Compliance tooling | ECOA letters, NL to SQL chat, EU AI Act checklist |
| Deployment | Streamlit on Hugging Face Spaces |
| Test suite | 34 passing pytest checks |

---

## The Core Result

Three out of four trained models fail the fairness threshold even without race or sex being used as direct inputs.

| Model | AUC-ROC | F1 | DPR | EOD | Passes 0.80 DPR? |
|---|---:|---:|---:|---:|---|
| Logistic Regression | 0.6756 | 0.6862 | 0.6683 | 0.1697 | No |
| Decision Tree | 0.8082 | 0.8905 | 0.7937 | 0.1752 | No |
| LightGBM | 0.8402 | 0.8888 | 0.7819 | 0.1622 | No |
| **LightGBM + Fairlearn** | **0.7111** | **0.9056** | **0.9025** | **0.0748** | **Yes** |

**Takeaway:** the highest-accuracy model is not the safest model. FairLend makes that tradeoff explicit.

---

## Dashboard Preview

| Model Comparison | Fairness Audit |
|---|---|
| <img src="https://raw.githubusercontent.com/yaswankum2622-code/fairlend/main/VISUAL%27S/Screenshot%202026-04-16%20220607.png" alt="Model fairness view" width="100%"> | <img src="https://raw.githubusercontent.com/yaswankum2622-code/fairlend/main/VISUAL%27S/Screenshot%202026-04-16%20220904.png" alt="Fairness audit view" width="100%"> |

| Applicant Explorer | Compliance Chat |
|---|---|
| <img src="https://raw.githubusercontent.com/yaswankum2622-code/fairlend/main/VISUAL%27S/Screenshot%202026-04-16%20220755.png" alt="Applicant explorer view" width="100%"> | <img src="https://raw.githubusercontent.com/yaswankum2622-code/fairlend/main/VISUAL%27S/Screenshot%202026-04-16%20221036.png" alt="Compliance chat view" width="100%"> |

| EU AI Act Report |
|---|
| <img src="https://raw.githubusercontent.com/yaswankum2622-code/fairlend/main/VISUAL%27S/Screenshot%202026-04-16%20221112.png" alt="EU AI Act report view" width="100%"> |

---

## What the App Does

| Page | Purpose | Why It Matters |
|---|---|---|
| Model Comparison | Compares 4 models on accuracy and fairness | Shows the legal cost of optimizing only for AUC |
| Applicant Explorer | Scores one applicant and explains the decision with SHAP | Makes individual decisions transparent |
| Fairness Audit | Computes disparate impact and proxy correlations | Finds discrimination even when protected fields are excluded |
| Adverse Action | Drafts ECOA-style denial letters with Gemini | Turns model output into regulator-friendly communication |
| Compliance Chat | Converts plain English questions into SQL over HMDA data | Gives compliance teams direct access to evidence |
| EU AI Act Report | Produces an article-by-article checklist | Connects ML outputs to high-risk AI governance |

---

## The Proxy Discrimination Story

Protected attributes are excluded from the model inputs. Bias still shows up through correlated financial features.

| Feature | Correlation with Race | Risk Level |
|---|---:|---|
| income | 0.175 | High |
| loan_to_income_ratio | 0.143 | High |
| dti_ratio | 0.115 | Medium |
| loan_amount | 0.100 | Medium |
| is_conventional | 0.034 | Low |

That is the point of FairLend: **removing race from the feature list is not the same as removing racial bias from the model**.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data | pandas, SQLite, dbt-sqlite |
| Classical ML | scikit-learn |
| Boosted Trees | LightGBM |
| Fairness | Fairlearn |
| Explainability | SHAP |
| Compliance AI | Gemini 2.5 Flash |
| Dashboard | Streamlit, Plotly |
| Testing | pytest |
| Deployment | GitHub Actions, Hugging Face Spaces |

---

## Quick Start

```bash
git clone https://github.com/yaswankum2622-code/fairlend.git
cd fairlend
pip install -r requirements.txt

# Place the HMDA 2024 CSV at:
# data/2024_public_lar_csv.csv

python data/loader.py
python models/baseline.py
python models/lgbm_model.py
python models/fair_model.py
python models/evaluate.py

streamlit run dashboard/app.py
```

---

## Project Structure

```text
FairLend/
├── data/                     HMDA CSV loader and SQLite database pipeline
├── database/                 Schema and query helpers
├── models/                   Baselines, LightGBM, Fairlearn, evaluation
├── explainability/           SHAP analysis and adverse action generation
├── fairness/                 Disparate impact and proxy detection
├── compliance/               EU AI Act report and NL-to-SQL chat
├── dashboard/                Streamlit 6-page interface
├── tests/                    Pytest suite
├── docs/                     Problem, scope, algorithms, results, future work
└── .github/workflows/        CI
```

---

## Documentation

| File | Description |
|---|---|
| [`docs/problem_statement.md`](docs/problem_statement.md) | The business and regulatory problem |
| [`docs/scope.md`](docs/scope.md) | Built features and MVP boundaries |
| [`docs/algorithms.md`](docs/algorithms.md) | Model and fairness methods |
| [`docs/results.md`](docs/results.md) | Evaluation results and fairness findings |
| [`docs/future_work.md`](docs/future_work.md) | Next steps for production and research |

---

## Built For

- recruiters looking for a serious applied ML portfolio project
- ML engineers working on tabular decision systems
- compliance and risk teams evaluating AI in lending
- anyone who wants to see the accuracy versus fairness tradeoff on real credit data

---

<div align="center">

### Built by Yashwanth

**M.Tech CSE | Business Analytics | VIT Chennai | Bengaluru**

If this project was useful, star the repo.

</div>
