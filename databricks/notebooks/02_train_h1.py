# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Train H1: Multilingual Repertoire → Daily Mobilization
# MAGIC
# MAGIC **Databricks ML Pipeline — H1: Binary Classification**
# MAGIC
# MAGIC Replaces `src/models/train_h1.py` with Databricks-native MLflow tracking.
# MAGIC
# MAGIC - **Features**: nb_langues, apprent_anterieur, relation_lm, domaine_usage, valorisation_sent, age, sexe, region_*, lm_*
# MAGIC - **Target**: h1_target (0=NON, 1=OUI)
# MAGIC - **Model**: XGBoost with SMOTE
# MAGIC - **Thresholds**: F1-macro ≥ 0.70, ROC-AUC ≥ 0.75

# COMMAND ----------

# MAGIC %pip install xgboost imbalanced-learn shap

# COMMAND ----------

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap

CATALOG = "flp_catalog"
EXPERIMENT = "/Shared/FLP_H1_Multilingual_Repertoire"

mlflow.set_experiment(EXPERIMENT)
print("✅ MLflow experiment: " + EXPERIMENT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Features from Delta Lake

# COMMAND ----------

df_h1 = spark.read.table(f"{CATALOG}.processed.h1_features").toPandas()

FEATURES = [
    "nb_langues", "apprent_anterieur_bin", "relation_lm_ord",
    "domaine_usage_freq", "valorisation_sent", "sexe_bin", "age",
]
LM_PREFIX = "lm_"
REGION_PREFIX = "region_"
TARGET = "h1_target"

feat_cols = (
    [c for c in FEATURES if c in df_h1.columns]
    + [c for c in df_h1.columns if c.startswith(LM_PREFIX)]
    + [c for c in df_h1.columns if c.startswith(REGION_PREFIX)]
)
X = df_h1[feat_cols].fillna(0)
y = df_h1[TARGET]

print(f"H1 — {len(X)} rows, {len(feat_cols)} features")
print(f"Balance: {y.value_counts().to_dict()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Train / Validation / Test Split

# COMMAND ----------

TEST_SIZE = 0.15
VAL_SIZE = 0.15
RS = 42

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RS)
X_tr, X_va, y_tr, y_va = train_test_split(
    X_tr, y_tr, test_size=VAL_SIZE / (1 - TEST_SIZE), stratify=y_tr, random_state=RS)

print(f"Train: {len(X_tr)} | Val: {len(X_va)} | Test: {len(X_te)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. SMOTE + Training

# COMMAND ----------

# SMOTE on train only (k_neighbors adapted to minority class size)
_min_count = int(y_tr.value_counts().min())
sm = SMOTE(random_state=RS, k_neighbors=max(1, _min_count - 1))
X_tr_res, y_tr_res = sm.fit_resample(X_tr, y_tr)
print(f"After SMOTE: {len(X_tr_res)} rows (balanced)")

# XGBoost pipeline
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
    ("model",   xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=RS,
    )),
])

model.fit(X_tr_res, y_tr_res)
print("✅ XGBoost H1 trained")

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
cv_res = cross_validate(
    Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=RS,
        )),
    ]),
    X_tr_res, y_tr_res, cv=cv, scoring=["f1_macro", "roc_auc"]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Evaluate on Test Set

# COMMAND ----------

y_pred = model.predict(X_te)
y_proba = model.predict_proba(X_te)[:, 1]

f1 = f1_score(y_te, y_pred, average="macro")
auc = roc_auc_score(y_te, y_proba)

print(f"TEST — F1-macro: {f1:.4f} | ROC-AUC: {auc:.4f}")
print(classification_report(y_te, y_pred, target_names=["NON", "OUI"]))

# SHAP
explainer = shap.TreeExplainer(model.named_steps["model"])
X_te_scaled = model[:-1].transform(X_te)
shap_vals = explainer.shap_values(X_te_scaled)
mean_shap = np.abs(shap_vals).mean(axis=0)
top5 = sorted(zip(feat_cols, mean_shap), key=lambda x: x[1], reverse=True)[:5]
print(f"SHAP top-5: {[(f, round(v, 4)) for f, v in top5]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Log to MLflow & Register Model

# COMMAND ----------

thresholds_met = {
    "f1_macro": f1 >= 0.70,
    "roc_auc": auc >= 0.75,
}

with mlflow.start_run(run_name=f"h1_xgb_{datetime.now().strftime('%Y%m%d_%H%M')}"):
    # Params
    mlflow.log_params({
        "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "smote": True, "cv_folds": 5,
    })
    # Metrics
    mlflow.log_metrics({
        "f1_macro": round(f1, 4),
        "roc_auc": round(auc, 4),
        "cv_f1_mean": round(cv_res["test_f1_macro"].mean(), 4),
        "cv_f1_std": round(cv_res["test_f1_macro"].std(), 4),
        "cv_auc_mean": round(cv_res["test_roc_auc"].mean(), 4),
        "n_train": len(X_tr_res), "n_test": len(X_te),
    })
    # Tags
    mlflow.set_tags({
        "model_type": "xgboost", "hypothesis": "H1",
        "status": "OK" if all(thresholds_met.values()) else "WARNING",
        "shap_top1": top5[0][0] if top5 else "",
        "platform": "databricks",
    })
    # Model artifact + register in Unity Catalog (signature required by UC)
    _sig = infer_signature(X_tr_res[:200], model.predict(X_tr_res[:200]))
    mlflow.sklearn.log_model(
        model, "h1_model",
        signature=_sig,
        input_example=X_tr_res[:5],
        registered_model_name=f"{CATALOG}.models.FLP_H1_usage_quotidien",
    )

    run_id = mlflow.active_run().info.run_id

print(f"✅ MLflow run: {run_id}")
print(f"✅ Model registered: {CATALOG}.models.FLP_H1_usage_quotidien")
print(f"   Status: {'✅ VALIDATED' if all(thresholds_met.values()) else '⚠️ WARNING'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Results Summary

# COMMAND ----------

print("=" * 55)
print("  H1 — Multilingual Repertoire → Daily Mobilization")
print("=" * 55)
print(f"  F1-macro : {f1:.4f}  (threshold ≥ 0.70) {'✅' if thresholds_met['f1_macro'] else '⚠️'}")
print(f"  ROC-AUC  : {auc:.4f}  (threshold ≥ 0.75) {'✅' if thresholds_met['roc_auc'] else '⚠️'}")
print(f"  CV F1    : {cv_res['test_f1_macro'].mean():.4f} ± {cv_res['test_f1_macro'].std():.4f}")
print(f"  SHAP top : {top5[0][0] if top5 else 'N/A'}")
print("=" * 55)
print("  Next: Run notebook 03_train_h2.py")
