"""
Step 2: SQL Data Cleaning, LGD Target Engineering & Exploratory Analysis.
Computes Exposure at Default (EAD), Net Recoveries, and realized Loss Given
Default (LGD) in DuckDB, engineers clean numeric features, and saves a bimodal
LGD distribution chart to reports/.
"""

import duckdb
import matplotlib.pyplot as plt

DB_PATH = "../data/lgd_ifrs9.db"

with duckdb.connect(DB_PATH) as con:
    # 1. Engineer LGD target and clean features in SQL
    con.execute("""
        CREATE OR REPLACE TABLE lgd_clean AS
        WITH base AS (
            SELECT
                funded_amnt,
                total_rec_prncp,
                GREATEST(funded_amnt - COALESCE(total_rec_prncp, 0), 0) AS ead_at_default,
                GREATEST(COALESCE(recoveries, 0) - COALESCE(collection_recovery_fee, 0), 0) AS net_recovery,
                CAST(REGEXP_EXTRACT(term, '([0-9]+)', 1) AS INTEGER) AS term_months,
                CAST(int_rate AS DOUBLE) AS int_rate,
                grade,
                sub_grade,
                CASE
                    WHEN emp_length IS NULL THEN -1
                    WHEN emp_length LIKE '<%' THEN 0
                    WHEN emp_length LIKE '10+%' THEN 10
                    ELSE CAST(REGEXP_EXTRACT(emp_length, '([0-9]+)', 1) AS INTEGER)
                END AS emp_years,
                CASE
                    WHEN home_ownership IN ('MORTGAGE', 'OWN', 'RENT') THEN home_ownership
                    ELSE 'OTHER'
                END AS home_ownership,
                COALESCE(annual_inc, 50000.0) AS annual_inc,
                COALESCE(dti, 18.0) AS dti,
                COALESCE(delinq_2yrs, 0) AS delinq_2yrs,
                (COALESCE(fico_range_low, 680) + COALESCE(fico_range_high, 684)) / 2.0 AS fico_score,
                COALESCE(open_acc, 10) AS open_acc,
                COALESCE(pub_rec, 0) AS pub_rec,
                COALESCE(revol_util, 50.0) AS revol_util,
                COALESCE(total_acc, 20) AS total_acc
            FROM defaulted_loans
            WHERE funded_amnt - COALESCE(total_rec_prncp, 0) > 100
        )
        SELECT
            *,
            LEAST(GREATEST(1.0 - (net_recovery / ead_at_default), 0.0), 1.0) AS lgd,
            CASE WHEN net_recovery = 0 THEN 1 ELSE 0 END AS is_zero_recovery
        FROM base
    """)

    # 2. Clean active_portfolio with matching feature transformations
    con.execute("""
        CREATE OR REPLACE TABLE active_clean AS
        SELECT
            funded_amnt,
            ead,
            CAST(REGEXP_EXTRACT(term, '([0-9]+)', 1) AS INTEGER) AS term_months,
            CAST(int_rate AS DOUBLE) AS int_rate,
            grade,
            sub_grade,
            CASE
                WHEN home_ownership IN ('MORTGAGE', 'OWN', 'RENT') THEN home_ownership
                ELSE 'OTHER'
            END AS home_ownership,
            COALESCE(annual_inc, 50000.0) AS annual_inc,
            COALESCE(dti, 18.0) AS dti,
            COALESCE(delinq_2yrs, 0) AS delinq_2yrs,
            (COALESCE(fico_range_low, 680) + COALESCE(fico_range_high, 684)) / 2.0 AS fico_score,
            COALESCE(revol_util, 50.0) AS revol_util
        FROM active_portfolio
    """)

    summary = con.execute("""
        SELECT
            COUNT(*) AS total_defaults,
            ROUND(AVG(lgd), 4) AS mean_lgd,
            ROUND(MEDIAN(lgd), 4) AS median_lgd,
            ROUND(AVG(is_zero_recovery) * 100, 2) AS zero_recovery_pct,
            ROUND(AVG(ead_at_default), 2) AS avg_ead
        FROM lgd_clean
    """).df()

    by_grade = con.execute("""
        SELECT
            grade,
            COUNT(*) AS loans,
            ROUND(AVG(lgd), 4) AS avg_lgd,
            ROUND(AVG(is_zero_recovery) * 100, 2) AS zero_rec_pct
        FROM lgd_clean
        GROUP BY grade
        ORDER BY grade
    """).df()

    lgd_vals = con.execute("SELECT lgd FROM lgd_clean USING SAMPLE 50000 ROWS").df()["lgd"]

print("=== LGD Portfolio Summary ===")
print(summary.to_string(index=False))
print("\n=== Realized LGD by Credit Grade ===")
print(by_grade.to_string(index=False))

# 3. Plot bimodal LGD distribution
plt.figure(figsize=(8, 5))
plt.hist(lgd_vals, bins=40, color="#1f77b4", edgecolor="black", alpha=0.85)
plt.axvline(summary["mean_lgd"].iloc[0], color="#d62728", linestyle="--", linewidth=2,
            label=f"Mean LGD = {summary['mean_lgd'].iloc[0]:.2%}")
plt.title("Realized Loss Given Default (LGD) Distribution")
plt.xlabel("Realized LGD (0 = Full Recovery, 1 = Total Loss)")
plt.ylabel("Number of Defaulted Loans (Sample)")
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("../reports/lgd_distribution.png", dpi=150)
plt.close()
print("\nSaved reports/lgd_distribution.png!")