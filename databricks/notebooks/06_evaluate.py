# Databricks notebook source
# MAGIC %md
# MAGIC # 06 — Global Evaluation & Summary
# MAGIC
# MAGIC **French-Learning-Perceptions ML — Databricks Migration**
# MAGIC
# MAGIC Aggregates results from all 4 hypotheses, validates thresholds,
# MAGIC and produces a pedagogical report.
# MAGIC
# MAGIC This replaces `src/evaluate.py`.

# COMMAND ----------

import mlflow
import pandas as pd
import numpy as np
from datetime import datetime

CATALOG = "flp_catalog"
mlflow.set_experiment("/Shared/FLP_Evaluation")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Latest Runs from MLflow

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()

experiments = {
    "H1": "/Shared/FLP_H1_Multilingual_Repertoire",
    "H2": "/Shared/FLP_H2_French_Representations",
    "H3": "/Shared/FLP_H3_Plurilingual_Exposure",
    "H4": "/Shared/FLP_H4_Local_Language_Integration",
}

latest_runs = {}
for hyp, exp_name in experiments.items():
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        print(f"⚠️ Experiment not found: {exp_name}")
        continue
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if runs:
        run = runs[0]
        latest_runs[hyp] = {
            "run_id": run.info.run_id,
            "status": run.data.tags.get("status", "UNKNOWN"),
            "metrics": run.data.metrics,
            "tags": run.data.tags,
        }
        print(f"✅ {hyp}: run={run.info.run_id[:8]}... status={run.data.tags.get('status', 'N/A')}")
    else:
        print(f"⚠️ No runs found for {hyp}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Validate Thresholds

# COMMAND ----------

THRESHOLDS = {
    "H1": {"f1_macro": 0.70, "roc_auc": 0.75},
    "H2": {"f1_weighted_A": 0.65, "f1_micro_B": 0.72},
    "H3": {"mae": 0.50, "f1_weighted": 0.68, "pearson_significant": True},
    "H4": {"f1_A": 0.70, "spearman_B": 0.55, "subset_C": 0.45},
}

results = {}
for hyp, run_data in latest_runs.items():
    metrics = run_data["metrics"]
    thresholds = THRESHOLDS.get(hyp, {})
    checks = {}
    for key, threshold in thresholds.items():
        if isinstance(threshold, bool):
            checks[key] = metrics.get(key, False) == threshold
        elif key == "mae":
            checks[key] = metrics.get(key, 999) <= threshold
        else:
            checks[key] = metrics.get(key, 0) >= threshold
    results[hyp] = {
        "metrics": {k: round(v, 4) for k, v in metrics.items() if isinstance(v, (int, float))},
        "checks": checks,
        "all_met": all(checks.values()),
        "run_id": run_data["run_id"],
    }
    status = "✅" if all(checks.values()) else "⚠️"
    print(f"{status} {hyp}: {checks}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Summary Table

# COMMAND ----------

rows = []
for hyp in ["H1", "H2", "H3", "H4"]:
    if hyp not in results:
        continue
    r = results[hyp]
    row = {"Hypothesis": hyp}
    for metric, check in r["checks"].items():
        val = r["metrics"].get(metric, "N/A")
        row[metric] = val
        row[f"{metric}_pass"] = "✅" if check else "⚠️"
    row["Overall"] = "✅ VALIDATED" if r["all_met"] else "⚠️ WARNING"
    rows.append(row)

summary_df = pd.DataFrame(rows)
display(summary_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Log Global Evaluation to MLflow

# COMMAND ----------

n_validated = sum(1 for r in results.values() if r["all_met"])

with mlflow.start_run(run_name=f"global_eval_{datetime.now().strftime('%Y%m%d_%H%M')}"):
    mlflow.set_tag("step", "global_evaluation")
    mlflow.set_tag("platform", "databricks")
    mlflow.log_metric("hypotheses_validated", n_validated)
    mlflow.log_metric("hypotheses_total", len(results))
    for hyp, r in results.items():
        for metric, val in r["metrics"].items():
            if isinstance(val, (int, float)):
                mlflow.log_metric(f"{hyp.lower()}_{metric}", val)
        mlflow.log_metric(f"{hyp.lower()}_all_met", int(r["all_met"]))

print(f"✅ Global evaluation logged to MLflow: {n_validated}/{len(results)} validated")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Generate Pedagogical Report

# COMMAND ----------

report_lines = [
    "=" * 70,
    "  FRENCH-LEARNING-PERCEPTIONS ML — PEDAGOGICAL REPORT",
    f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    "=" * 70,
    "",
]

HYPOTHESES_DESC = {
    "H1": "Multilingual repertoire → daily language mobilization",
    "H2": "French representations → motivation & difficulties",
    "H3": "Plurilingual exposure → attitudes toward French",
    "H4": "Local language integration → engagement & motivation",
}

for hyp in ["H1", "H2", "H3", "H4"]:
    if hyp not in results:
        continue
    r = results[hyp]
    report_lines.append(f"  {hyp} — {HYPOTHESES_DESC[hyp]}")
    for metric, check in r["checks"].items():
        val = r["metrics"].get(metric, "N/A")
        icon = "✅" if check else "⚠️"
        report_lines.append(f"    {icon} {metric}: {val}")
    report_lines.append(f"    Status: {'VALIDATED' if r['all_met'] else 'WARNING'}")
    report_lines.append("")

report_lines.append(f"  Hypotheses validated: {n_validated}/{len(results)}")
report_lines.append("=" * 70)

report_text = "\n".join(report_lines)
print(report_text)

# Write to Delta
report_df = spark.createDataFrame([(report_text,)], ["report_text"])
report_df.write.format("delta").mode("overwrite").saveAsTable(
    f"{CATALOG}.monitoring.pedagogical_report"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Model Registry — Production Status

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()

MODEL_NAMES = [
    f"{CATALOG}.models.FLP_H1_usage_quotidien",
    f"{CATALOG}.models.FLP_H2_motivation",
    f"{CATALOG}.models.FLP_H2_difficultes",
    f"{CATALOG}.models.FLP_H3_attitude_reg",
    f"{CATALOG}.models.FLP_H3_attitude_clf",
    f"{CATALOG}.models.FLP_H4_motivation",
    f"{CATALOG}.models.FLP_H4_engagement",
    f"{CATALOG}.models.FLP_H4_discipline",
]

print("Registered Models:")
for name in MODEL_NAMES:
    try:
        versions = client.search_model_versions(f"name='{name}'")
        latest = max(versions, key=lambda v: int(v.version)) if versions else None
        stage = latest.current_stage if latest else "N/A"
        print(f"  {name.split('.')[-1]:35s} → stage={stage}")
    except Exception as e:
        print(f"  {name.split('.')[-1]:35s} → not found")

# COMMAND ----------

print("\n" + "=" * 55)
print(f"  GLOBAL EVALUATION COMPLETE — {n_validated}/{len(results)} VALIDATED")
print("=" * 55)
print("  Next: Set up Workflow (07_workflow.json) or Model Serving")
