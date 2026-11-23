"""
Step 1: Data Ingestion & Portfolio Filtering (DuckDB).
Reads raw Lending Club loan data directly from CSV using DuckDB's out-of-core
engine, isolating defaulted ('Charged Off') loans for LGD modeling and active
('Current') loans for IFRS 9 ECL portfolio simulation.
"""

import duckdb

DB_PATH = "../data/lgd_ifrs9.db"
RAW_CSV = "../data/lending_club.csv"

with duckdb.connect(DB_PATH) as con:
    # 1. Ingest defaulted loans for historical LGD calibration
    con.execute(f"""
        CREATE OR REPLACE TABLE defaulted_loans AS
        SELECT
            loan_amnt,
            funded_amnt,
            term,
            int_rate,
            installment,
            grade,
            sub_grade,
            emp_length,
            home_ownership,
            annual_inc,
            verification_status,
            purpose,
            dti,
            delinq_2yrs,
            fico_range_low,
            fico_range_high,
            open_acc,
            pub_rec,
            revol_bal,
            revol_util,
            total_acc,
            total_rec_prncp,
            recoveries,
            collection_recovery_fee
        FROM read_csv_auto('{RAW_CSV}', ignore_errors=true)
        WHERE loan_status = 'Charged Off'
          AND funded_amnt > 0
    """)

    # 2. Sample 25,000 active loans to simulate a live bank portfolio under IFRS 9
    con.execute(f"""
        CREATE OR REPLACE TABLE active_portfolio AS
        SELECT
            loan_amnt,
            funded_amnt,
            out_prncp AS ead,
            term,
            int_rate,
            grade,
            sub_grade,
            home_ownership,
            annual_inc,
            dti,
            delinq_2yrs,
            fico_range_low,
            fico_range_high,
            revol_util
        FROM read_csv_auto('{RAW_CSV}', ignore_errors=true)
        WHERE loan_status = 'Current'
          AND out_prncp > 0
        USING SAMPLE 25000 ROWS (reservoir, 42)
    """)

    n_default = con.execute("SELECT COUNT(*) FROM defaulted_loans").fetchone()[0]
    n_active = con.execute("SELECT COUNT(*) FROM active_portfolio").fetchone()[0]

    print(f"Loaded defaulted_loans (LGD training base): {n_default:,} rows")
    print(f"Loaded active_portfolio (IFRS 9 ECL base):  {n_active:,} rows")