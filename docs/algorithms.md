# Algorithms and Models

## 1. LightGBM — Gradient Boosted Decision Trees

Standard for tabular credit data. Outperforms neural networks
on structured data (NeurIPS 2022 benchmark, 45 datasets).
Used as both the unconstrained baseline and the base estimator
inside the Fairlearn constraint wrapper.

```text
n_estimators  = 300
learning_rate = 0.05
num_leaves    = 63
max_depth     = 6
```

## 2. Fairlearn ExponentiatedGradient

Finds the Pareto-optimal accuracy-fairness tradeoff.
Wraps any sklearn-compatible model with a fairness constraint.

```python
constraint = DemographicParity(difference_bound=0.05)
fair_model  = ExponentiatedGradient(
    estimator   = LGBMClassifier(...),
    constraints = constraint,
    eps         = 0.05,
    max_iter    = 50
)
fair_model.fit(X, y, sensitive_features=race)
```

The constraint requires that approval rates differ by no more
than 5 percentage points across race groups. The algorithm
runs 50 iterations to find the minimum accuracy cost.

## 3. SHAP TreeExplainer

Computes exact Shapley values for tree-based models.
Shapley values are the only attribution method with a
provable fairness guarantee — they satisfy efficiency,
symmetry, dummy, and additivity axioms.

```text
Result: each feature gets a credit/blame score
        that sums to the difference between the
        model prediction and the population average.
```

Protected attributes (race, sex) are excluded from the
feature set so they cannot appear in any SHAP explanation.

## 4. DoWhy Proxy Detection

Measures correlation between model features and protected
attributes using point-biserial correlation coefficient.

```text
θ = Cov(Y, X) / (σ_Y × σ_X)

Where:
  Y = binary race indicator (White vs non-White)
  X = model feature value
```

High correlation indicates the feature carries information
about race that the model can exploit.

## 5. Gemini 2.5 Flash

Used for two tasks:

**Adverse action letters:**
SHAP values and applicant financials are structured into
a prompt. Gemini writes a legally formatted ECOA letter
in plain English. Temperature = 0.3 for consistency.

**Compliance chat:**
Full DB schema is injected into the system prompt.
User question → SQL generation (temperature = 0.1).
Results → interpretation (temperature = 0.4).
