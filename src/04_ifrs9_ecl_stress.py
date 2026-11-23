"""
Step 4: IFRS 9 Expected Credit Loss (ECL) Engine & Macro Stress Testing.
Assigns active portfolio exposures to IFRS 9 Stages (Stage 1: 12-month ECL,
Stage 2: Lifetime ECL under SICR triggers, Stage 3: Impaired), computes
expected credit losses (ECL = PD * LGD * EAD), runs macroeconomic stress
scenarios (Baseline, Mild Downturn, Severe Recession), and exports summary
tables and charts to reports/.
"""

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DB_PATH = "../data/lgd_ifrs9.db"

# 1. Load scored active portfolio from DuckDB
with duckdb.connect(DB_PATH) as con:
    df = con.execute("SELECT * FROM active_scored_lgd").df()

# 2. Assign 12-month PD benchmark by Lending Club Credit Grade (empirically calibrated)
GRADE_PD_12M = {
    "A": 0.015,
    "B": 0.032,
    "C": 0.060,
    "D": 0.105,
    "E": 0.165,
    "F": 0.240,
    "G": 0.320,
}
df["pd_12m"] = df["grade"].map(GRADE_PD_12M).fillna(0.08)

# 3. IFRS 9 Staging Logic (SICR = Significant Increase in Credit Risk)
# Stage 3: severe delinquency / impaired proxy (delinq_2yrs >= 3)
# Stage 2: SICR triggered (delinq_2yrs >= 1 OR fico_score < 670 OR dti > 32)
# Stage 1: performing loans without SICR triggers
conditions = [
    df["delinq_2yrs"] >= 3,
    (df["delinq_2yrs"] >= 1) | (df["fico_score"] < 670) | (df["dti"] > 32.0),
]
df["ifrs9_stage"] = np.select(conditions, ["Stage 3", "Stage 2"], default="Stage 1")

# 4. Effective PD under IFRS 9:
#    - Stage 1 uses 12-month PD
#    - Stage 2 uses Cumulative Lifetime PD: 1 - (1 - PD_12m)^remaining_years
#    - Stage 3 uses PD = 100%
remaining_years = np.where(df["term_months"] == 60, 3.5, 2.0)
pd_lifetime = 1.0 - np.power(1.0 - df["pd_12m"], remaining_years)

df["pd_ifrs9"] = np.where(
    df["ifrs9_stage"] == "Stage 3",
    1.0,
    np.where(df["ifrs9_stage"] == "Stage 2", pd_lifetime, df["pd_12m"]),
)

# 5. Baseline ECL = PD * LGD * EAD
df["ecl_baseline"] = df["pd_ifrs9"] * df["pred_lgd"] * df["ead"]

# 6. Macroeconomic Stress Scenarios (PD multipliers & LGD add-ons)
scenarios = {
    "Baseline": {"pd_mult": 1.00, "lgd_addon": 0.00},
    "Mild Downturn": {"pd_mult": 1.35, "lgd_addon": 0.04},
    "Severe Recession": {"pd_mult": 1.85, "lgd_addon": 0.08},
}

scenario_rows = []
for name, params in scenarios.items():
    stressed_pd = np.minimum(df["pd_ifrs9"] * params["pd_mult"], 1.0)
    stressed_lgd = np.minimum(df["pred_lgd"] + params["lgd_addon"], 1.0)
    ecl_val = (stressed_pd * stressed_lgd * df["ead"]).sum()
    total_ead = df["ead"].sum()
    scenario_rows.append({
        "Scenario": name,
        "Total_EAD_USD": round(total_ead, 2),
        "Total_ECL_USD": round(ecl_val, 2),
        "ECL_Coverage_Pct": round((ecl_val / total_ead) * 100, 2),
        "Uplift_vs_Baseline_Pct": round(((ecl_val / df["ecl_baseline"].sum()) - 1.0) * 100, 2),
    })

stress_df = pd.DataFrame(scenario_rows)

# Stage breakdown table (Baseline)
stage_summary = (
    df.groupby("ifrs9_stage", as_index=False)
    .agg(
        Loans=("loan_id", "count"),
        Total_EAD_USD=("ead", "sum"),
        Avg_PD_Pct=("pd_ifrs9", lambda s: s.mean() * 100),
        Avg_LGD_Pct=("pred_lgd", lambda s: s.mean() * 100),
        Baseline_ECL_USD=("ecl_baseline", "sum"),
    )
)
stage_summary["ECL_Share_Pct"] = (
    stage_summary["Baseline_ECL_USD"] / stage_summary["Baseline_ECL_USD"].sum() * 100
)
for col in ["Total_EAD_USD", "Avg_PD_Pct", "Avg_LGD_Pct", "Baseline_ECL_USD", "ECL_Share_Pct"]:
    stage_summary[col] = stage_summary[col].round(2)

print("=== IFRS 9 Portfolio Breakdown by Stage (Baseline) ===")
print(stage_summary.to_string(index=False))

print("\n=== Macroeconomic Stress Testing Results ===")
print(stress_df.to_string(index=False))

# Save tables to reports/
stage_summary.to_csv("../reports/ifrs9_stage_summary.csv", index=False)
stress_df.to_csv("../reports/macro_stress_results.csv", index=False)

# 7. Plot IFRS 9 Stage ECL & Stress Test Impact
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

# Left panel: Baseline ECL by Stage
colors_stage = ["#2ca02c", "#ff7f0e", "#d62728"]
axes[0].bar(stage_summary["ifrs9_stage"], stage_summary["Baseline_ECL_USD"] / 1e6,
            color=colors_stage, edgecolor="black", alpha=0.88)
axes[0].set_title("Baseline IFRS 9 ECL by Stage")
axes[0].set_ylabel("Expected Credit Loss ($ Millions)")
axes[0].grid(axis="y", alpha=0.3)
for idx, val in enumerate(stage_summary["Baseline_ECL_USD"] / 1e6):
    axes[0].text(idx, val + 0.05, f"${val:.2f}M", ha="center", fontweight="bold")

# Right panel: Total ECL under Macro Stress Scenarios
colors_stress = ["#1f77b4", "#ff7f0e", "#d62728"]
axes[1].bar(stress_df["Scenario"], stress_df["Total_ECL_USD"] / 1e6,
            color=colors_stress, edgecolor="black", alpha=0.88)
axes[1].set_title("Macroeconomic Stress Test — Total Portfolio ECL")
axes[1].set_ylabel("Expected Credit Loss ($ Millions)")
axes[1].grid(axis="y", alpha=0.3)
for idx, row in stress_df.iterrows():
    val = row["Total_ECL_USD"] / 1e6
    uplift = row["Uplift_vs_Baseline_Pct"]
    label = f"${val:.2f}M" if uplift == 0 else f"${val:.2f}M\n(+{uplift:.1f}%)"
    axes[1].text(idx, val + 0.15, label, ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("../reports/ifrs9_ecl_stress.png", dpi=150)
plt.close()
print("\nSaved reports/ifrs9_ecl_stress.png and CSV summaries!")