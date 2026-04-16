SELECT
    derived_race,
    derived_sex,
    COUNT(*)                                    AS total_applications,
    SUM(approved)                               AS approved_count,
    ROUND(AVG(approved) * 100, 2)              AS approval_rate_pct,
    ROUND(AVG(CAST(loan_amount AS FLOAT)), 0)  AS avg_loan_amount,
    ROUND(AVG(CAST(income AS FLOAT)), 0)       AS avg_income
FROM {{ ref('stg_applications') }}
GROUP BY derived_race, derived_sex
ORDER BY derived_race, derived_sex
