# Databricks notebook source
# MAGIC %md
# MAGIC # 07 — Combine all 8 models into ONE serving entity (v3, ASCII-safe names)
# MAGIC
# MAGIC Databricks Model Serving mangles non-ASCII column names (accents).
# MAGIC We strip accents from the input contract; the sklearn pipelines are
# MAGIC positional so predictions are unaffected.

# COMMAND ----------

# MAGIC %pip install xgboost

# COMMAND ----------

import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
import numpy as np
import unicodedata
from mlflow.models import infer_signature
from mlflow.pyfunc import PythonModel

CATALOG = "flp_catalog"

MODEL_URIS = {
    "h1":  "flp_h1_usage_quotidien",
    "h2a": "flp_h2_motivation",
    "h2b": "flp_h2_difficultes",
    "h3r": "flp_h3_attitude_reg",
    "h3c": "flp_h3_attitude_clf",
    "h4a": "flp_h4_motivation",
    "h4b": "flp_h4_engagement",
    "h4c": "flp_h4_discipline",
}
REGISTRY = {}
for k, v in MODEL_URIS.items():
    from mlflow.tracking import MlflowClient
    vs = MlflowClient().search_model_versions(f"name='{CATALOG}.models.{v}'")
    latest = max(vs, key=lambda mv: int(mv.version))
    REGISTRY[k] = f"models:/{CATALOG}.models.{v}/{latest.version}"
print(REGISTRY)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Collect signatures + bundle pipelines

# COMMAND ----------

bundle = {}
for key, uri in REGISTRY.items():
    info = mlflow.models.get_model_info(uri)
    sig = info.signature
    in_cols = [str(c) for c in sig.inputs.input_names()] if sig and sig.inputs else []
    out_cols = [str(c) for c in sig.outputs.input_names()] if sig and sig.outputs else []
    pipe = mlflow.sklearn.load_model(uri)
    bundle[key] = {"model": pipe, "in": in_cols, "out": out_cols}
    print(f"{key}: {len(in_cols)} in / {len(out_cols)} out")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. ASCII-normalize column names (serving-safe contract)

# COMMAND ----------

def ascii_norm(s):
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode("ascii")

all_orig = sorted({c for m in bundle.values() for c in m["in"]})
seen = {}
clean_map = {}
for c in all_orig:
    base = ascii_norm(c)
    if base in seen:
        seen[base] += 1
        clean_map[c] = f"{base}_{seen[base]}"
    else:
        seen[base] = 0
        clean_map[c] = base

for key in bundle:
    bundle[key]["in"] = [clean_map[c] for c in bundle[key]["in"]]

UNION_COLS = sorted(clean_map.values())
print(f"union input columns (ASCII-safe): {len(UNION_COLS)}")

bundle_path = "/tmp/flp_bundle.joblib"
with open(bundle_path, "wb") as f:
    joblib.dump(bundle, f)
print("bundle written")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Combined PythonModel

# COMMAND ----------

FIXED_OUT = (
    ["h1_pred", "h1_proba", "h2a_pred", "h2a_proba_max",
     "h2b_0", "h2b_1", "h2b_2", "h2b_3", "h2b_4", "h2b_5", "h2b_6",
     "h3_reg", "h3_clf", "h4a_pred", "h4a_proba", "h4b_engagement",
     "h4c_0", "h4c_1", "h4c_2", "h4c_3", "h4c_4"]
)

class FLPCombinedModel(PythonModel):
    def load_context(self, context):
        with open(context.artifacts["bundle"], "rb") as f:
            self.bundle = joblib.load(f)

    def _predict_one(self, key, df):
        entry = self.bundle[key]
        model = entry["model"]
        X = df.reindex(columns=entry["in"]).fillna(0).to_numpy()
        pred = np.asarray(model.predict(X))
        proba = None
        if hasattr(model, "predict_proba") and key in ("h1", "h2a", "h3c", "h4a"):
            proba = np.asarray(model.predict_proba(X))
        return pred, proba

    def predict(self, context, model_input):
        df = model_input.copy()
        requested = "all"
        if "model_name" in df.columns and len(df) > 0:
            requested = str(df["model_name"].iloc[0]).lower()
        rows = []
        for i in range(len(df)):
            row = {c: 0.0 for c in FIXED_OUT}
            one = df.iloc[[i]].reset_index(drop=True)
            targets = [requested] if requested != "all" else list(self.bundle.keys())
            for key in targets:
                if key not in self.bundle:
                    continue
                pred, proba = self._predict_one(key, one)
                p = pred[0]
                if key == "h1":
                    row["h1_pred"] = float(p)
                    row["h1_proba"] = float(proba[0][1]) if proba is not None else -1.0
                elif key == "h2a":
                    row["h2a_pred"] = float(p)
                    row["h2a_proba_max"] = float(proba[0].max()) if proba is not None else -1.0
                elif key == "h2b":
                    for j, v in enumerate(p):
                        row[f"h2b_{j}"] = float(v)
                elif key == "h3r":
                    row["h3_reg"] = float(p)
                elif key == "h3c":
                    row["h3_clf"] = float(p)
                elif key == "h4a":
                    row["h4a_pred"] = float(p)
                    row["h4a_proba"] = float(proba[0][1]) if proba is not None else -1.0
                elif key == "h4b":
                    row["h4b_engagement"] = float(p + 1)
                elif key == "h4c":
                    for j, v in enumerate(p):
                        row[f"h4c_{j}"] = float(v)
            rows.append(row)
        return pd.DataFrame(rows, columns=FIXED_OUT)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Signature + log pyfunc with bundled artifact

# COMMAND ----------

example_in = pd.DataFrame(
    {c: ([0.0] if c != "model_name" else ["all"]) for c in UNION_COLS + ["model_name"]}
)
example_in["model_name"] = "all"
example_in["sexe_bin"] = [0.0]
example_in["age"] = [14.0]

combined = FLPCombinedModel()

class _Ctx:
    artifacts = {"bundle": bundle_path}

combined.load_context(_Ctx())
example_out = combined.predict(None, example_in)
sig = infer_signature(example_in, example_out)
print("signature ok — output cols:", list(example_out.columns[:6]))

# COMMAND ----------

mlflow.set_experiment("/Shared/FLP_Serving")

with mlflow.start_run(run_name="flp_combined_serving_model_v3"):
    mlflow.pyfunc.log_model(
        "flp_all_models",
        python_model=combined,
        artifacts={"bundle": bundle_path},
        signature=sig,
        input_example=example_in,
        extra_pip_requirements=["xgboost", "scikit-learn", "joblib"],
        registered_model_name=f"{CATALOG}.models.flp_all",
    )
    print("combined model v3 registered:", f"{CATALOG}.models.flp_all")

# COMMAND ----------

print("DONE — update endpoint 'flp-all-models' to flp_catalog.models.flp_all version 3")
