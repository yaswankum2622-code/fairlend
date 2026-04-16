# Future Work

## Priority 1 — Stronger Fairness Methods

**Counterfactual fairness**
Test whether a decision would change if the applicant's
protected attribute were different, all else equal.
This is the standard now used in CFPB enforcement actions.

**IBM AIF360 adversarial debiasing**
A neural network approach to fairness that learns to
make predictions that cannot be used to infer protected
attributes. Compare against Fairlearn on this dataset.

**Calibrated fairness**
Ensure that predicted probabilities are equally calibrated
across demographic groups — not just approval rates.

## Priority 2 — Production Engineering

**PostgreSQL + Docker**
Replace SQLite with PostgreSQL for concurrent access.
Containerise with Docker Compose for reproducible deployment.

**Real-time monitoring**
Track DPR and EOD metrics on live decisions.
Alert when fairness metrics drift below threshold.

**Batch adverse action generation**
Process denied applications in bulk rather than one at a time.

## Priority 3 — Regulatory Coverage

**UK FCA compliance**
The Financial Conduct Authority has its own fair lending
requirements that differ from the CFPB and EU AI Act.

**Multi-jurisdiction report**
Single compliance report covering CFPB (US), FCA (UK),
and EU AI Act simultaneously.

**Audit trail**
Log every model decision with inputs, SHAP values,
and model version for regulatory review.
