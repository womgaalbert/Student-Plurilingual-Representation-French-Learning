# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Train H4: Local Language Integration → Engagement
# MAGIC
# MAGIC **Databricks ML Pipeline — H4: Multi-label + Ordinal**
# MAGIC
# MAGIC - **Target A**: motivation (binary: Motivé/Peu motivé)
# MAGIC - **Target B**: engagement score (ordinal: 1-4)
# MAGIC - **Target C**: discipline preference (multi-label)
# MAGIC - **Thresholds**: F1_A ≥ 0.70, Spearman ρ ≥ 0.55, Subset accuracy ≥ 0.45

# COMMAND ----------

# MAGIC %pip install xgboost imbalanced-learn

# COMMAND ----------

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from datetime import datetime
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb

CATALOG = "flp_catalog"
EXPERIMENT = "/Shared/FLP_H4_Local_Language_Integration"
mlflow.set_experiment(EXPERIMENT)
print("✅ MLflow experiment: " + EXPERIMENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Features

# COMMAND ----------

df_h4 = spark.read.table(f"{CATALOG}.processed.h4_features").toPandas()

FEATURES = [
    "interet_camarades_bin", "interet_camarades_sent",
    "souhait_freq", "sexe_bin", "age",
]
disc_cols = [c for c in df_h4.columns if c.startswith("vd_disc_")]
vi_cols = [c for c in df_h4.columns if c.startswith("vi_disc_")]
region_cols = [c for c in df_h4.columns if c.startswith("region_")]
feat_cols = [c for c in FEATURES if c in df_h4.columns] + vi_cols + region_cols

X = df_h4[feat_cols].fillna(0)
y_A = df_h4["h4_target_motivation"].fillna(0).astype(int)
y_B = df_h4["h4_engagement_score"].fillna(1).astype(int)
y_C = df_h4[disc_cols].fillna(0).astype(int)

print(f"H4 — {len(X)} rows, {len(feat_cols)} features, {len(disc_cols)} disciplines")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Split & Train

# COMMAND ----------

RS = 42
TEST_SIZE = 0.15

X_tr, X_te, yA_tr, yA_te, yB_tr, yB_te, yC_tr, yC_te = train_test_split(
    X, y_A, y_B, y_C, test_size=TEST_SIZE, stratify=y_A, random_state=RS)

# SMOTE on target A (k_neighbors adapted to minority class size)
_min_count = int(yA_tr.value_counts().min())
sm = SMOTE(random_state=RS, k_neighbors=max(1, _min_count - 1))
X_tr_res, yA_tr_res = sm.fit_resample(X_tr, yA_tr)

# Model A: Motivation (binary) — VotingClassifier
model_A = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", VotingClassifier([
        ("xgb", xgb.XGBClassifier(eval_metric="logloss", random_state=RS)),
        ("lr", LogisticRegression(max_iter=500, class_weight="balanced")),
    ], voting="soft")),
])
model_A.fit(X_tr_res, yA_tr_res)
print("✅ H4-A (motivation) trained")

# Model B: Engagement (ordinal, 4 classes)
model_B = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", xgb.XGBClassifier(
        n_estimators=200,
        eval_metric="mlogloss", random_state=RS)),
])
model_B.fit(X_tr, yB_tr - 1)
print("✅ H4-B (engagement) trained")

# Model C: Discipline (multi-label)
model_C = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", MultiOutputClassifier(
        xgb.XGBClassifier(eval_metric="logloss", random_state=RS))),
])
model_C.fit(X_tr, yC_tr)
print("✅ H4-C (discipline) trained")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Evaluate

# COMMAND ----------

yA_pred = model_A.predict(X_te)
yB_pred = model_B.predict(X_te) + 1
yC_pred = model_C.predict(X_te)

f1_A = f1_score(yA_te, yA_pred)
rho, _ = spearmanr(yB_te, yB_pred)
sub_acc = accuracy_score(yC_te, yC_pred)

print(f"F1-A (motivation)   : {f1_A:.4f} (≥0.70)")
print(f"Spearman ρ (engage) : {rho:.4f} (≥0.55)")
print(f"Subset acc (discip) : {sub_acc:.4f} (≥0.45)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Log to MLflow & Register

# COMMAND ----------

thresholds_met = {
    "f1_A": f1_A >= 0.70,
    "spearman_B": rho >= 0.55,
    "subset_C": sub_acc >= 0.45,
}

with mlflow.start_run(run_name=f"h4_xgb_{datetime.now().strftime('%Y%m%d_%H%M')}"):
    mlflow.log_params({
        "model_A": "VotingClassifier(XGB+LR)",
        "model_B": "XGBClassifier (4-class ordinal)",
        "model_C": "MultiOutputClassifier(XGB)",
        "smote": True,
    })
    mlflow.log_metrics({
        "f1_A": round(f1_A, 4), "spearman_B": round(rho, 4),
        "subset_C": round(sub_acc, 4),
        "n_train": len(X_tr), "n_test": len(X_te),
    })
    mlflow.set_tags({
        "model_type": "multioutput_ordinal", "hypothesis": "H4",
        "status": "OK" if all(thresholds_met.values()) else "WARNING",
        "platform": "databricks",
    })
    _sigA = infer_signature(X_tr_res[:200], model_A.predict(X_tr_res[:200]))
    _sigB = infer_signature(X_tr[:200], model_B.predict(X_tr[:200]))
    _sigC = infer_signature(X_tr[:200], model_C.predict(X_tr[:200]))
    mlflow.sklearn.log_model(model_A, "h4_model_A", signature=_sigA,
                             input_example=X_tr_res[:5],
                             registered_model_name=f"{CATALOG}.models.FLP_H4_motivation")
    mlflow.sklearn.log_model(model_B, "h4_model_B", signature=_sigB,
                             input_example=X_tr[:5],
                             registered_model_name=f"{CATALOG}.models.FLP_H4_engagement")
    mlflow.sklearn.log_model(model_C, "h4_model_C", signature=_sigC,
                             input_example=X_tr[:5],
                             registered_model_name=f"{CATALOG}.models.FLP_H4_discipline")

    run_id = mlflow.active_run().info.run_id

print(f"✅ Run: {run_id} | Status: {'✅ VALIDATED' if all(thresholds_met.values()) else '⚠️ WARNING'}")

# COMMAND ----------

print("=" * 55)
print("  H4 — Local Language Integration → Engagement")
print("=" * 55)
print(f"  F1-A       : {f1_A:.4f}  (≥0.70) {'✅' if thresholds_met['f1_A'] else '⚠️'}")
print(f"  Spearman ρ : {rho:.4f}  (≥0.55) {'✅' if thresholds_met['spearman_B'] else '⚠️'}")
print(f"  Subset acc : {sub_acc:.4f}  (≥0.45) {'✅' if thresholds_met['subset_C'] else '⚠️'}")
print("=" * 55)
print("  Next: Run notebook 06_evaluate.py")
