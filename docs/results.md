# Results

## Dataset

| Property | Value |
|---|---|
| Source | CFPB HMDA 2024 |
| Total applications | 500,000 |
| Approved | 421,500 (84.3%) |
| Denied | 78,500 (15.7%) |
| States | 55 |
| Lenders | 4,421 |
| Race groups | 4 |

## Approval Rate by Race (Raw Data)

| Race | Approval Rate | DI Ratio vs White |
|---|---|---|
| Asian | 88.6% | 1.028 |
| White | 86.2% | 1.000 (reference) |
| Other or Not Provided | 82.0% | 0.952 |
| Black or African American | 71.1% | 0.825 |

## Model Comparison

| Model | AUC-ROC | F1 | Precision | Recall | DPR | EOD | Passes? |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.6756 | 0.6862 | 0.9067 | 0.5520 | 0.6683 | 0.1697 | NO |
| Decision Tree | 0.8082 | 0.8905 | 0.9254 | 0.8581 | 0.7937 | 0.1752 | NO |
| LightGBM | 0.8402 | 0.8888 | 0.9371 | 0.8452 | 0.7819 | 0.1622 | NO |
| LightGBM + Fairlearn | 0.7111 | 0.9056 | 0.9099 | 0.9012 | 0.9025 | 0.0748 | YES |

## Proxy Variables

| Feature | Correlation with Race | Risk Level |
|---|---|---|
| income | 0.175 | HIGH |
| loan_to_income_ratio | 0.143 | HIGH |
| dti_ratio | 0.115 | MEDIUM |
| loan_amount | 0.100 | MEDIUM |
| is_conventional | 0.034 | LOW |

## EU AI Act Compliance

Status: COMPLIANT
Checks passed: 10/10
Model version: FairLend v1.0
Compliance deadline: August 2026
