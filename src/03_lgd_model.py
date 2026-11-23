"""
Step 3: Two-Stage Loss Given Default (LGD) Modeling.
Implements a banking-standard Hurdle LGD architecture:
  - Stage 1: Logistic Regression predicting P(Zero Recovery | X)
  - Stage 2: Gradient Boosted Regressor predicting conditional LGD given partial recovery
Evaluates combined out-of-sample LGD on a stratified test split, saves
calibration diagnostics to reports/, and scores the active IFRS 9 portfolio.
"""

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, mean_absolute_error, root_mean_squared_error

DB_PATH = "../data/lgd_ifrs9.db"

NUM_FEATURES = [
    "funded_amnt",
    "term_months",
    "int_rate",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "fico_score",
    "revol_util",
]
CAT_FEATURES = ["grade", "home_ownership"]
ALL_FEATURES = NUM_FEATURES + CAT_FEATURES

# 1. Load cleaned datasets from DuckDB
with duckdb.connect(DB_PATH) as con:
    df = con.execute(f"""
        SELECT {', '.join(ALL_FEATURES)}, lgd, is_zero_recovery
        FROM lgd_clean
    """).df()

    active_df = con.execute(f"""
        SELECT rowid AS loan_id, ead, sub_grade, {', '.join(ALL_FEATURES)}
        FROM active_clean
    """).df()

X = df[ALL_FEATURES]
y_lgd = df["lgd"].values
y_zero = df["is_zero_recovery"].values

# Stratified train/test split on zero-recovery indicator
X_train, X_test, y_lgd_train, y_lgd_test, y_zero_train, y_zero_test = train_test_split(
    X, y_lgd, y_zero, test_size=0.20, random_state=42, stratify=y_zero
)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CAT_FEATURES),
    ]
)

# 2. Stage 1: Classification model for P(LGD = 1.0)
stage1_clf = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
])
stage1_clf.fit(X_train, y_zero_train)

p_zero_test = stage1_clf.predict_proba(X_test)[:, 1]
stage1_auc = roc_auc_score(y_zero_test, p_zero_test)

# 3. Stage 2: Conditional Severity Regressor on partial-recovery loans (LGD < 1)
partial_mask_train = y_zero_train == 0
X_train_partial = X_train[partial_mask_train]
y_lgd_train_partial = y_lgd_train[partial_mask_train]

stage2_reg = Pipeline([
    ("prep", preprocessor),
    ("reg", HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=100,
        random_state=42,
    )),
])
stage2_reg.fit(X_train_partial, y_lgd_train_partial)

cond_lgd_test = np.clip(stage2_reg.predict(X_test), 0.0, 1.0)

# 4. Combine Stage 1 and Stage 2 into expected LGD
pred_lgd_test = p_zero_test * 1.0 + (1.0 - p_zero_test) * cond_lgd_test

rmse = root_mean_squared_error(y_lgd_test, pred_lgd_test)
mae = mean_absolute_error(y_lgd_test, pred_lgd_test)

print("=== Two-Stage LGD Out-of-Sample Performance ===")
print(f"Stage 1 (Zero-Recovery Classifier) ROC AUC: {stage1_auc:.4f}")
print(f"Combined Two-Stage LGD RMSE:                {rmse:.4f}")
print(f"Combined Two-Stage LGD MAE:                 {mae:.4f}")
print(f"Actual Mean LGD (Test Set):                 {y_lgd_test.mean():.4f}")
print(f"Predicted Mean LGD (Test Set):              {pred_lgd_test.mean():.4f}")

# 5. Calibration plot by decile
cal_df = pd.DataFrame({"actual": y_lgd_test, "pred": pred_lgd_test})
cal_df["decile"] = pd.qcut(cal_df["pred"], q=10, duplicates="drop")
decile_summary = cal_df.groupby("decile", observed=False).mean().reset_index()

plt.figure(figsize=(7.5, 5))
plt.plot(range(1, len(decile_summary) + 1), decile_summary["actual"], "o-", label="Actual Mean LGD", linewidth=2)
plt.plot(range(1, len(decile_summary) + 1), decile_summary["pred"], "s--", label="Predicted Mean LGD", linewidth=2)
plt.title("Two-Stage LGD Model Calibration by Risk Decile (Test Set)")
plt.xlabel("Predicted LGD Risk Decile (1 = Lowest Loss, 10 = Highest Loss)")
plt.ylabel("Mean LGD")
plt.ylim(0.80, 1.00)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("../reports/lgd_calibration.png", dpi=150)
plt.close()
print("Saved reports/lgd_calibration.png!")

# 6. Score active portfolio with predicted LGD and save to DuckDB
p_zero_active = stage1_clf.predict_proba(active_df[ALL_FEATURES])[:, 1]
cond_lgd_active = np.clip(stage2_reg.predict(active_df[ALL_FEATURES]), 0.0, 1.0)
active_df["pred_lgd"] = np.clip(p_zero_active * 1.0 + (1.0 - p_zero_active) * cond_lgd_active, 0.05, 1.0)

with duckdb.connect(DB_PATH) as con:
    con.register("scored_active_view", active_df)
    con.execute("CREATE OR REPLACE TABLE active_scored_lgd AS SELECT * FROM scored_active_view")
    print(f"Scored {len(active_df):,} active loans and saved to DuckDB table 'active_scored_lgd'.")