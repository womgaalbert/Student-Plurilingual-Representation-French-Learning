# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Train H3: Plurilingual Exposure → Attitudes toward French
# MAGIC
# MAGIC **Databricks ML Pipeline — H3: Regression + Classification + Causal Analysis**
# MAGIC
# MAGIC - **Regression**: h3_score_attitude [1.0-5.0]
# MAGIC - **Classification**: attitude class (Positive/Neutre/Négative)
# MAGIC - **Causal**: Pearson correlation between exposure and attitude
# MAGIC - **Thresholds**: MAE ≤ 0.50, F1-weighted ≥ 0.68, Pearson p < 0.05

# COMMAND ----------

# MAGIC %pip install xgboost

# COMMAND ----------

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from datetime import datetime
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score, mean_absolute_error
import xgboost as xgb

CATALOG = "flp_catalog"
EXPERIMENT = "/Shared/FLP_H3_Plurilingual_Exposure"
mlflow.set_experiment(EXPERIMENT)
print("✅ MLflow experiment: " + EXPERIMENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Features

# COMMAND ----------

df_h3 = spark.read.table(f"{CATALOG}.processed.h3_features").toPandas()

FEATURES = [
    "exposition_freq", "interet_bin", "interet_sent",
    "perception_multi_sent", "perception_multi_ord",
    "nb_langues", "sexe_bin", "age",
]
region_cols = [c for c in df_h3.columns if c.startswith("region_")]
feat_cols = [c for c in FEATURES if c in df_h3.columns] + region_cols

X = df_h3[feat_cols].fillna(0)
y_reg = df_h3["h3_score_attitude"]
le = LabelEncoder().fit(["Négative", "Neutre", "Positive"])
y_clf = le.transform(df_h3["h3_attitude_class"].fillna("Neutre"))

print(f"H3 — {len(X)} rows, {len(feat_cols)} features")
print(f"Score attitude: mean={y_reg.mean():.2f}, std={y_reg.std():.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Split & Train

# COMMAND ----------

RS = 42
TEST_SIZE = 0.15

X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(
    X, y_reg, y_clf, test_size=TEST_SIZE, random_state=RS)

# Regression
model_reg = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", RandomForestRegressor(
        n_estimators=400, max_depth=8, min_samples_leaf=2, random_state=RS)),
])
model_reg.fit(X_tr, yr_tr)
print("✅ H3 Regression trained")

# Classification
model_clf = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", xgb.XGBClassifier(
        n_estimators=200,
        eval_metric="mlogloss", random_state=RS)),
])
model_clf.fit(X_tr, yc_tr)
print("✅ H3 Classification trained")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Evaluate

# COMMAND ----------

yr_pred = model_reg.predict(X_te)
yc_pred = model_clf.predict(X_te)

mae = mean_absolute_error(yr_te, yr_pred)
f1_clf = f1_score(yc_te, yc_pred, average="weighted")

# Pearson correlation (causal signal)
expo = df_h3["exposition_freq"].fillna(0)
score = df_h3["h3_score_attitude"].fillna(3)
r, p = pearsonr(expo, score)

print(f"MAE regression : {mae:.4f} (≤0.50)")
print(f"F1-weighted clf: {f1_clf:.4f} (≥0.68)")
print(f"Pearson r      : {r:.4f}  p={p:.4f} (p<0.05 for significance)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Log to MLflow & Register

# COMMAND ----------

thresholds_met = {
    "mae": mae <= 0.50,
    "f1_weighted": f1_clf >= 0.68,
    "pearson_significant": p < 0.05 and r > 0,
}

with mlflow.start_run(run_name=f"h3_models_{datetime.now().strftime('%Y%m%d_%H%M')}"):
    mlflow.log_params({
        "n_estimators_reg": 400, "max_depth_reg": 8, "min_samples_leaf": 2,
        "n_estimators_clf": 200, "causal": True,
    })
    mlflow.log_metrics({
        "mae": round(mae, 4), "f1_weighted": round(f1_clf, 4),
        "pearson_r": round(r, 4), "pearson_p": round(p, 4),
        "n_train": len(X_tr), "n_test": len(X_te),
    })
    mlflow.set_tags({
        "model_type": "regression+classification", "hypothesis": "H3",
        "status": "OK" if all(thresholds_met.values()) else "WARNING",
        "platform": "databricks",
        "causal_significant": str(thresholds_met["pearson_significant"]),
    })
    _sigR = infer_signature(X_tr[:200], model_reg.predict(X_tr[:200]))
    _sigC = infer_signature(X_tr[:200], model_clf.predict(X_tr[:200]))
    mlflow.sklearn.log_model(model_reg, "h3_reg", signature=_sigR,
                             input_example=X_tr[:5],
                             registered_model_name=f"{CATALOG}.models.FLP_H3_attitude_reg")
    mlflow.sklearn.log_model(model_clf, "h3_clf", signature=_sigC,
                             input_example=X_tr[:5],
                             registered_model_name=f"{CATALOG}.models.FLP_H3_attitude_clf")

    run_id = mlflow.active_run().info.run_id

print(f"✅ Run: {run_id} | Status: {'✅ VALIDATED' if all(thresholds_met.values()) else '⚠️ WARNING'}")

# COMMAND ----------

print("=" * 55)
print("  H3 — Plurilingual Exposure → Attitudes toward French")
print("=" * 55)
print(f"  MAE          : {mae:.4f}  (≤0.50) {'✅' if thresholds_met['mae'] else '⚠️'}")
print(f"  F1-weighted  : {f1_clf:.4f}  (≥0.68) {'✅' if thresholds_met['f1_weighted'] else '⚠️'}")
print(f"  Pearson r    : {r:.4f}  p={p:.4f} {'✅ Significant' if thresholds_met['pearson_significant'] else '⚠️ Not significant'}")
print("=" * 55)
print("  Next: Run notebook 05_train_h4.py")
