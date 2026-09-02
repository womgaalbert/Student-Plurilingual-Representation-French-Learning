# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — Environment Setup & Unity Catalog Configuration
# MAGIC
# MAGIC **French-Learning-Perceptions ML — Databricks Migration**
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Creates the Unity Catalog structure (catalog → schemas → tables)
# MAGIC 2. Uploads raw data to DBFS / Unity Catalog Volume
# MAGIC 3. Validates cluster libraries
# MAGIC 4. Sets up MLflow experiment tracking
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - Databricks workspace (Community Edition or paid)
# MAGIC - Cluster with ML runtime (includes MLflow, scikit-learn, XGBoost)
# MAGIC - `data/raw/data_FLP.csv` uploaded to DBFS

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Validate Cluster Environment

# COMMAND ----------

import sys
print(f"Python version : {sys.version}")
print(f"Spark version  : {spark.version}")

def _step_log(name, ok, detail=""):
    try:
        spark.createDataFrame([(name, str(ok), str(detail)[:2000])],
                              ["step", "ok", "detail"]).write.format("delta").mode("append").saveAsTable("flp_catalog.monitoring.run_log")
    except Exception:
        try:
            spark.createDataFrame([(name, str(ok), str(detail)[:2000])],
                                  ["step", "ok", "detail"]).write.format("delta").mode("append").saveAsTable("flp_catalog.monitoring.run_log")
        except Exception:
            pass

_step_log("00_start", True, "python " + sys.version + " spark " + spark.version)

# Check key libraries
libraries = {
    "mlflow": None, "sklearn": None, "xgboost": None,
    "pandas": None, "numpy": None, "imblearn": None,
    "shap": None, "yaml": None, "streamlit": None,
}

for lib in libraries:
    try:
        mod = __import__(lib)
        ver = getattr(mod, "__version__", "installed")
        print(f"  ✅ {lib:20s} {ver}")
    except ImportError:
        print(f"  ❌ {lib:20s} NOT INSTALLED")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Unity Catalog — Create Catalog & Schemas
# MAGIC
# MAGIC Structure:
# MAGIC ```
# MAGIC flp_catalog                    ← top-level catalog
# MAGIC ├── raw                      ← raw CSV data
# MAGIC ├── processed                ← cleaned + feature-engineered
# MAGIC ├── models                   ← MLflow registered models
# MAGIC └── monitoring               ← drift reports + alerts
# MAGIC ```

# COMMAND ----------

CATALOG = "flp_catalog"
SCHEMAS = ["raw", "processed", "models", "monitoring"]

# Create catalog (requires appropriate privileges)
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
print(f"✅ Catalog '{CATALOG}' ready")

for schema in SCHEMAS:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{schema}")
    print(f"✅ Schema '{CATALOG}.{schema}' ready")

# Set default catalog
spark.sql(f"USE CATALOG {CATALOG}")
_step_log("00_catalog", True, f"catalog={CATALOG} schemas={SCHEMAS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Upload Raw Data to DBFS / Volume

# COMMAND ----------

import os

# ── Unity Catalog Volume (production-grade governance) ───────────────────
# The CSV is uploaded to the managed volume via the Databricks Files API:
#   PUT /api/2.0/fs/files/Volumes/flp_catalog/raw/data_files/data_FLP.csv
#   spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.raw.data_files")

# For this migration, we reference the Volume path where the CSV is uploaded:
DBFS_RAW_PATH = "/Volumes/flp_catalog/raw/data_files/data_FLP.csv"
DBFS_PROCESSED_DIR = "/Volumes/flp_catalog/raw/data_files/processed"

# Create directories
dbutils.fs.mkdirs(DBFS_PROCESSED_DIR)
print(f"✅ DBFS directories ready: {DBFS_PROCESSED_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Create Delta Tables from Raw CSV

# COMMAND ----------

from pyspark.sql import functions as F

# Read the raw CSV into a Spark DataFrame
# NOTE: the survey CSV contains newlines inside quoted fields and doubled
# quotes ("") — enable multiLine + escape to parse it correctly.
df_raw = spark.read.csv(
    DBFS_RAW_PATH,
    header=True,
    inferSchema=True,
    encoding="utf-8",
    multiLine=True,
    escape='"',
)

print(f"Raw data: {df_raw.count()} rows, {len(df_raw.columns)} columns")

# Sanitize column names for Delta (no spaces/newlines/;{}()= allowed)
import re as _re
def _clean_col(name):
    return _re.sub(r'[\s,;{}()\n\t=]+', '_', str(name)).strip('_') or "col"

df_raw = df_raw.toDF(*[_clean_col(c) for c in df_raw.columns])
print(f"Sanitized columns: {df_raw.columns[:5]}...")
_step_log("00_csv_read", True, f"rows={df_raw.count()} cols={len(df_raw.columns)}")

# Write to Delta Lake — raw layer
(
    df_raw.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOG}.raw.survey_responses")
)

print(f"✅ Delta table created: {CATALOG}.raw.survey_responses")
_step_log("00_delta_raw", True, "raw.survey_responses written")

# Show sample
display(df_raw.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Validate Delta Table

# COMMAND ----------

df_delta = spark.read.table(f"{CATALOG}.raw.survey_responses")
print(f"Delta table: {df_delta.count()} rows")
df_delta.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Set Up MLflow Experiments

# COMMAND ----------

import mlflow

# MLflow on Databricks is automatically configured — no tracking_uri needed
# Experiments are stored in the workspace

EXPERIMENTS = {
    "preprocess": "/Shared/FLP_Preprocess",
    "h1":         "/Shared/FLP_H1_Multilingual_Repertoire",
    "h2":         "/Shared/FLP_H2_French_Representations",
    "h3":         "/Shared/FLP_H3_Plurilingual_Exposure",
    "h4":         "/Shared/FLP_H4_Local_Language_Integration",
}

for key, path in EXPERIMENTS.items():
    try:
        exp_id = mlflow.create_experiment(path)
        print(f"✅ Created experiment '{path}' (id={exp_id})")
    except Exception:
        exp = mlflow.get_experiment_by_name(path)
        print(f"ℹ️  Experiment '{path}' already exists (id={exp.experiment_id})")

print("\n✅ MLflow experiments ready — tracked automatically by Databricks")
_step_log("00_mlflow_exps", True, "experiments ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Summary

# COMMAND ----------

print("=" * 60)
print("  DATABRICKS ENVIRONMENT SETUP COMPLETE")
print("=" * 60)
print(f"  Catalog        : {CATALOG}")
print(f"  Schemas        : {', '.join(SCHEMAS)}")
print(f"  Raw table      : {CATALOG}.raw.survey_responses")
print(f"  DBFS raw path  : {DBFS_RAW_PATH}")
print(f"  DBFS processed : {DBFS_PROCESSED_DIR}")
print(f"  MLflow         : Databricks managed (auto)")
print("=" * 60)
print("  Next: Run notebook 01_preprocess.py")
