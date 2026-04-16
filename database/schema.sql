CREATE TABLE IF NOT EXISTS applications (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    year                  TEXT,
    lender_id             TEXT,
    state                 TEXT,
    county                TEXT,
    msa_code              TEXT,
    race                  TEXT,
    race_simplified       TEXT,
    sex                   TEXT,
    sex_simplified        TEXT,
    age                   TEXT,
    action_taken          TEXT,
    loan_type             TEXT,
    lien_status           TEXT,
    loan_amount           REAL,
    income                REAL,
    dti_ratio             REAL,
    loan_to_income_ratio  REAL,
    is_joint_application  INTEGER,
    is_conventional       INTEGER,
    approved              INTEGER NOT NULL CHECK(approved IN (0,1))
);

CREATE INDEX IF NOT EXISTS idx_race     ON applications(race_simplified);
CREATE INDEX IF NOT EXISTS idx_sex      ON applications(sex_simplified);
CREATE INDEX IF NOT EXISTS idx_state    ON applications(state);
CREATE INDEX IF NOT EXISTS idx_approved ON applications(approved);
CREATE INDEX IF NOT EXISTS idx_lender   ON applications(lender_id);
