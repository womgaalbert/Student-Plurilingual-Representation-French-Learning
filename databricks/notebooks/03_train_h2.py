# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Train H2: French Representations → Motivation & Difficulties
# MAGIC
# MAGIC **Databricks ML Pipeline — H2: Multi-output Classification**
# MAGIC
# MAGIC - **Target A**: motivation (0=Faible, 1=Moyen, 2=Élevé)
# MAGIC - **Target B**: difficulties multi-label (grammaire, vocabulaire, etc.)
# MAGIC - **Thresholds**: F1-weighted_A ≥ 0.65, F1-micro_B ≥ 0.72

# COMMAND ----------

# MAGIC %pip install xgboost imbalanced-learn

# COMMAND ----------

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb

CATALOG = "flp_catalog"
EXPERIMENT = "/Shared/FLP_H2_French_Representations"
mlflow.set_experiment(EXPERIMENT)
print("✅ MLflow experiment: " + EXPERIMENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Features from Delta Lake

# COMMAND ----------

df_h2 = spark.read.table(f"{CATALOG}.processed.h2_features").toPandas()

FEATURES = [
    "perc_utile", "perc_belle", "perc_difficile", "perc_importante",
    "mots_assoc_sent", "importance_bin", "importance_sent",
    "hierarchie_fr", "sexe_bin", "age",
]
diff_cols = [c for c in df_h2.columns if c.startswith("diff_")]
region_cols = [c for c in df_h2.columns if c.startswith("region_")]
feat_cols = [c for c in FEATURES if c in df_h2.columns] + region_cols

X = df_h2[feat_cols].fillna(0)
y_A = df_h2["h2_target_motivation"]
y_B = df_h2[diff_cols].fillna(0).astype(int)

print(f"H2 — {len(X)} rows, {len(feat_cols)} features, {len(diff_cols)} difficulty labels")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Split & Train

# COMMAND ----------

RS = 42
TEST_SIZE = 0.15

X_tr, X_te, yA_tr, yA_te, yB_tr, yB_te = train_test_split(
    X, y_A, y_B, test_size=TEST_SIZE, stratify=y_A, random_state=RS)

# SMOTE on target A (k_neighbors adapted to minority class size)
_min_count = int(yA_tr.value_counts().min())
sm = SMOTE(random_state=RS, k_neighbors=max(1, _min_count - 1))
X_tr_res, yA_tr_res = sm.fit_resample(X_tr, yA_tr)

# Model A: Motivation (3-class)
model_A = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.08,
        eval_metric="mlogloss", random_state=RS,
    )),
])
model_A.fit(X_tr_res, yA_tr_res)
print("✅ H2-A (motivation) trained")

# Model B: Difficulties (multi-label)
model_B = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", MultiOutputClassifier(
        xgb.XGBClassifier(n_estimators=200,
                          eval_metric="logloss", random_state=RS))),
])
model_B.fit(X_tr, yB_tr)
print("✅ H2-B (difficulties) trained")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Evaluate

# COMMAND ----------

yA_pred = model_A.predict(X_te)
yB_pred = model_B.predict(X_te)

f1_A = f1_score(yA_te, yA_pred, average="weighted")
f1_B = f1_score(yB_te, yB_pred, average="micro", zero_division=0)

print(f"Target A (motivation) F1-weighted: {f1_A:.4f} (≥0.65)")
print(f"Target B (difficulties) F1-micro:  {f1_B:.4f} (≥0.72)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Log to MLflow & Register

# COMMAND ----------

thresholds_met = {
    "f1_weighted_A": f1_A >= 0.65,
    "f1_micro_B": f1_B >= 0.72,
}

with mlflow.start_run(run_name=f"h2_xgb_{datetime.now().strftime('%Y%m%d_%H%M')}"):
    mlflow.log_params({
        "n_estimators": 200, "max_depth": 4, "learning_rate": 0.08,
        "model_A": "XGBClassifier (3-class)", "model_B": "MultiOutputClassifier(XGB)",
        "smote": True,
    })
    mlflow.log_metrics({
        "f1_weighted_A": round(f1_A, 4), "f1_micro_B": round(f1_B, 4),
        "n_train": len(X_tr), "n_test": len(X_te),
    })
    mlflow.set_tags({
        "model_type": "multioutput_xgboost", "hypothesis": "H2",
        "status": "OK" if all(thresholds_met.values()) else "WARNING",
        "platform": "databricks",
    })
    _sigA = infer_signature(X_tr_res[:200], model_A.predict(X_tr_res[:200]))
    _sigB = infer_signature(X_tr[:200], model_B.predict(X_tr[:200]))
    mlflow.sklearn.log_model(model_A, "h2_model_A", signature=_sigA,
                             input_example=X_tr_res[:5],
                             registered_model_name=f"{CATALOG}.models.FLP_H2_motivation")
    mlflow.sklearn.log_model(model_B, "h2_model_B", signature=_sigB,
                             input_example=X_tr[:5],
                             registered_model_name=f"{CATALOG}.models.FLP_H2_difficultes")

    run_id = mlflow.active_run().info.run_id

print(f"✅ Run: {run_id} | Status: {'✅ VALIDATED' if all(thresholds_met.values()) else '⚠️ WARNING'}")

# COMMAND ----------

print("=" * 55)
print("  H2 — French Representations → Motivation & Difficulties")
print("=" * 55)
print(f"  F1-weighted A : {f1_A:.4f}  (≥0.65) {'✅' if thresholds_met['f1_weighted_A'] else '⚠️'}")
print(f"  F1-micro B   : {f1_B:.4f}  (≥0.72) {'✅' if thresholds_met['f1_micro_B'] else '⚠️'}")
print("=" * 55)
print("  Next: Run notebook 04_train_h3.py")
