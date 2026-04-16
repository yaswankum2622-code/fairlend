# Problem Statement

## The $70 Million Lesson

In October 2024 the CFPB fined Goldman Sachs and Apple $70 million
for discriminatory credit decisions in the Apple Card programme.
Women were systematically offered lower credit limits than men
with equivalent financial profiles.

Goldman's defence was straightforward: the model never saw
gender as a feature. The CFPB rejected this defence.

The model had learned to use zip code, credit history length,
and spending patterns as proxies. All three are correlated with
gender due to historical patterns in employment, property
ownership, and consumer behaviour. The model had not been
told about gender. It had inferred it.

## What the Law Requires

**ECOA (Equal Credit Opportunity Act)**
Requires lenders to provide specific written reasons for
any credit denial. "The algorithm decided" is not a valid reason.

**CFPB 4/5ths Rule**
The approval rate for any protected group must be at least
80% of the approval rate for the most-favoured group.
This is the Disparate Impact Ratio threshold of 0.80.

**EU AI Act — Article 9**
Credit scoring is classified as high-risk AI.
Compliance deadline: August 2026.
Required: bias testing, explainability, human oversight,
technical documentation, and ongoing monitoring.
Fines: up to €35 million or 7% of global annual revenue.

## The Three Problems FairLend Solves

**Problem 1 — Models discriminate without intent**
A model trained on historical lending data learns historical
discrimination patterns. Removing protected attributes from
the feature set is not sufficient — proxy variables remain.

**Problem 2 — Decisions cannot be explained**
ECOA requires specific reasons for denial. SHAP provides
exact per-applicant attribution to legitimate financial
factors. No protected attributes appear in any explanation.

**Problem 3 — Compliance cannot be audited**
Compliance officers cannot query credit model behaviour
without writing SQL. The compliance chat interface converts
plain English questions to SQL and returns interpreted results.
