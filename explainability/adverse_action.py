"""
FairLend | explainability/adverse_action.py | ECOA adverse action letter generator
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent))

PROTECTED_FEATURES = {"race", "race_simplified", "sex", "sex_simplified", "age"}
FEATURE_LABELS = {
    "loan_to_income_ratio": "Your requested loan amount relative to your income",
    "dti_ratio": "Your current monthly debt obligations",
    "income": "Your documented annual income",
    "loan_amount": "The size of the loan you requested",
    "lien_status": "The lien structure of the loan request",
    "loan_type": "The loan program selected for this application",
    "state": "State-level underwriting and property context",
    "is_joint_application": "Whether the application includes a co-applicant",
    "is_conventional": "Whether the request is for a conventional loan product",
}


def _format_number(value, suffix=""):
    """Format numeric values safely for prompts and fallback letters."""
    numeric = None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A"

    if suffix:
        return f"{numeric:,.0f}{suffix}"
    return f"{numeric:,.2f}"


def _feature_label(feature_name: str) -> str:
    """Map internal feature names to plain-English descriptions."""
    return FEATURE_LABELS.get(feature_name, feature_name.replace("_", " ").title())


def generate_adverse_action_letter(
    applicant_data: dict,
    shap_explanation: dict,
    approval_probability: float,
) -> str:
    """
    Generate an ECOA-compliant adverse action letter using Gemini 2.5 Flash.

    Input:
      applicant_data:      dict of applicant features (loan_amount, income, etc.)
      shap_explanation:    output from explain_applicant()
      approval_probability: float 0-1

    Output:
      str — full formatted adverse action letter
    """

    # Build top factors text from SHAP values
    top_factors = shap_explanation["top_factors"]
    denial_factors = []
    support_factors = []

    for feat, val in top_factors:
        if feat in PROTECTED_FEATURES:
            continue
        feat_readable = _feature_label(feat)
        if val < 0:
            denial_factors.append(
                f"  - {feat_readable}: contributed {val:.4f} against approval"
            )
        else:
            support_factors.append(
                f"  - {feat_readable}: contributed +{val:.4f} toward approval"
            )

    factors_text = (
        "Factors working AGAINST approval:\n" +
        ("\n".join(denial_factors) if denial_factors
         else "  None significant") +
        "\n\nFactors working FOR approval:\n" +
        ("\n".join(support_factors) if support_factors
         else "  None significant")
    )

    loan_amount = applicant_data.get("loan_amount", "N/A")
    income = applicant_data.get("income", "N/A")
    dti = applicant_data.get("dti_ratio", "N/A")
    lti = applicant_data.get("loan_to_income_ratio", "N/A")
    state = applicant_data.get("state", "N/A")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return _fallback_letter(
            applicant_data, denial_factors, approval_probability
        )

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")

        prompt = f"""
You are a compliance officer at a US mortgage lender writing a
legally required adverse action notice under the Equal Credit
Opportunity Act (ECOA) and the Fair Housing Act.

APPLICANT INFORMATION:
- Loan amount requested: ${_format_number(loan_amount, 'K')}
- Annual income: ${_format_number(income, 'K')}
- Debt-to-income ratio: {dti}
- Loan-to-income ratio: {_format_number(lti)}
- State: {state}
- Model approval probability: {approval_probability:.1%}

MODEL EXPLANATION (SHAP analysis):
{factors_text}

LEGAL REQUIREMENTS FOR THIS LETTER:
1. Must follow ECOA Regulation B format
2. Must state specific reasons for denial — cannot say "algorithm"
3. Must inform applicant of right to statement of reasons
4. Must include CFPB contact information
5. Must not reference race, sex, age, or any protected class
6. Must be written in plain English — reading level grade 8
7. Must include specific actionable steps the applicant can take

Write the complete adverse action letter.
Use today's date. Use formal letter format.
Address the applicant as "Dear Applicant".
Do not invent or assume any information not provided above.
Translate SHAP factor names into plain English explanations.
For example:
  loan_to_income_ratio → "Your requested loan amount relative to your income"
  dti_ratio → "Your current monthly debt obligations"
  income → "Your documented annual income"
  loan_amount → "The size of the loan you requested"

The letter must be professional, compassionate, and legally compliant.
Minimum length: 350 words.
"""

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1500,
            )
        )

        return response.text.strip()

    except Exception:
        return _fallback_letter(
            applicant_data, denial_factors, approval_probability
        )


def _fallback_letter(
    applicant_data: dict,
    denial_factors: list,
    approval_probability: float
) -> str:
    """
    Fallback letter when Gemini API is unavailable.
    Uses real applicant numbers — never generic.
    """
    import datetime
    today = datetime.date.today().strftime("%B %d, %Y")

    factors_str = "\n".join(denial_factors) if denial_factors else (
        "  - Insufficient income relative to requested loan amount\n"
        "  - Debt-to-income ratio exceeds guidelines"
    )

    loan_amount = applicant_data.get("loan_amount", 0)
    income = applicant_data.get("income", 0)
    probability_text = f"{approval_probability:.1%}"

    return f"""
{today}

Dear Applicant,

RE: Notice of Adverse Action — Mortgage Loan Application

We have carefully reviewed your application for a mortgage loan
in the amount of ${_format_number(loan_amount)},000. After thorough evaluation,
we are unable to approve your application at this time. Based on our review,
the application did not meet the approval standards applied to this request,
and the estimated likelihood of approval under those standards was {probability_text}.

SPECIFIC REASONS FOR THIS DECISION:

The following factors were the primary reasons for this decision:

{factors_str}

YOUR RIGHTS UNDER FEDERAL LAW:

The Equal Credit Opportunity Act (ECOA) and the Fair Housing Act
prohibit creditors from discriminating against credit applicants
on the basis of race, color, religion, national origin, sex,
marital status, age, or because you receive public assistance.

You have the right to:
1. Learn the specific reasons for this decision within 60 days
   of receiving this notice by contacting us in writing.
2. Have the information in your credit report reviewed if credit
   was a factor in this decision.
3. Obtain a free copy of your credit report from the reporting
   agency within 60 days.

STEPS YOU CAN TAKE:

To strengthen a future application, we suggest:
- Reducing your existing monthly debt obligations
- Increasing your documented income or adding a co-applicant
- Considering a smaller loan amount relative to your income
- Maintaining on-time payments on all existing obligations

CONTACT INFORMATION:

Consumer Financial Protection Bureau (CFPB):
www.consumerfinance.gov | 1-855-411-2372

We encourage you to reapply when your financial situation changes.

Sincerely,
FairLend Credit Review Team
"""
