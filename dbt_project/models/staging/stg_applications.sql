SELECT
    activity_year,
    lei,
    derived_msa_md,
    state_code,
    county_code,
    census_tract,
    derived_ethnicity,
    derived_race,
    derived_sex,
    action_taken,
    purchaser_type,
    loan_type,
    loan_purpose,
    lien_status,
    loan_amount,
    income,
    debt_to_income_ratio,
    applicant_age,
    co_applicant_age,
    CASE WHEN action_taken = 1 THEN 1 ELSE 0 END AS approved
FROM applications
WHERE action_taken IN (1, 3)
  AND loan_purpose = 1
  AND loan_type IN (1, 2)
