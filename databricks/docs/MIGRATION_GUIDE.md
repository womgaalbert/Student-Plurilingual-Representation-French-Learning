# Databricks Migration Guide — French-Learning-Perceptions ML

## Overview

This document describes the migration of the French-Learning-Perceptions ML pipeline from a **local MLOps setup** (MLflow local + Docker + FastAPI + GitHub Actions) to the **Databricks Data Intelligence Platform**.

**Project**: *Les représentations des élèves du premier cycle de l'enseignement secondaire camerounais sur l'apprentissage du français dans leur contexte plurilingue*

**Researcher**: Chancelline Armelle Nongni Kendjio
**ML Engineering**: Albert Womga
**Databricks Endpoint**: [Streamlit Dashboard](https://student-plurilingual-representation-french-learning-htyukpvkoo.streamlit.app/)

---

## Before vs After Architecture

```
BEFORE (Local):
  ┌─────────────┐    ┌──────────────┐    ┌───────────┐    ┌────────────┐    ┌──────────┐
  │  data/raw/  │ →  │  preprocess  │ →  │  train    │ →  │  FastAPI   │ →  │ Docker   │
  │  *.csv      │    │  .py         │    │  _h1-h4   │    │  .py       │    │ Container│
  └─────────────┘    └──────────────┘    └───────────┘    └────────────┘    └──────────┘
                            ↓                   ↓                               ↓
                      MLflow (local)      MLflow Registry               Prometheus/Grafana
                      (SQLite)            (local)                       (monitoring/)

AFTER (Databricks):
  ┌──────────────┐    ┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
  │ Unity Catalog│ →  │ Databricks      │ →  │ Databricks   │ →  │ Databricks      │
  │ Volume/DBFS  │    │ Notebooks       │    │ MLflow       │    │ Model Serving   │
  │ (raw data)   │    │ (preprocess +   │    │ (managed)    │    │ (endpoints)     │
  └──────────────┘    │  train H1-H4)   │    └──────────────┘    └─────────────────┘
                      └─────────────────┘            ↓                      ↓
                               ↓              Unity Catalog          Lakehouse Monitoring
                      Delta Lake tables       Model Registry         (drift detection)
                      (ACID, versioning)      (governed)             + SQL Dashboards
```

---

## What Changed

### 1. Data Storage: CSV → Delta Lake

| Before | After | Benefit |
|--------|-------|---------|
| `data/raw/data_FLP.csv` (plain CSV) | `flp_catalog.raw.survey_responses` (Delta table) | ACID transactions, time travel, schema enforcement |
| `data/processed/*.csv` (plain CSV) | `flp_catalog.processed.h{1-4}_features` (Delta tables) | Versioned, queryable, auditable |

**Key Delta Lake features demonstrated:**
- `DESCRIBE HISTORY flp_catalog.processed.h1_features` — full audit trail
- `SELECT * FROM flp_catalog.processed.h1_features VERSION AS OF 1` — time travel
- Automatic schema enforcement on write

### 2. Experiment Tracking: Local MLflow → Databricks Managed MLflow

| Before | After | Benefit |
|--------|-------|---------|
| `mlflow.set_tracking_uri("sqlite:///mlflow.db")` | Auto-configured by Databricks | No setup, no database management |
| Experiments in local directory | Experiments in Databricks workspace | Centralized, searchable, shareable |
| Model registry in SQLite | Unity Catalog Model Registry | Governance, access control, lineage |

**Key MLflow features demonstrated:**
- Automatic experiment tracking per notebook cell
- Model versioning with stage transitions (Staging → Production)
- Model lineage: trace which data + code produced which model

### 3. Orchestration: GitHub Actions → Databricks Workflows

| Before | After | Benefit |
|--------|-------|---------|
| `.github/workflows/train.yml` | `databricks/workflows/flp_pipeline_job.json` | Native scheduling, retry, alerting |
| GitHub Actions runner (limited) | Databricks Job Cluster (scalable) | Auto-scaling compute |
| Sequential steps | Task DAG with dependencies | Parallel H1-H4 training, single preprocess |

**Workflow DAG:**
```
preprocess
    ├── train_h1
    ├── train_h2
    ├── train_h3
    ├── train_h4
    └── evaluate (depends on all)
```

### 4. Model Serving: Docker + FastAPI → Databricks Model Serving

| Before | After | Benefit |
|--------|-------|---------|
| `Dockerfile` + `docker-compose.yml` | `serving/endpoint_config.json` | Zero infrastructure management |
| Manual container builds | Automatic endpoint provisioning | One-click deploy |
| 10+ FastAPI endpoints | 7 served models behind one endpoint | Unified API, auto-scaling |

### 5. Monitoring: Custom Scripts → Lakehouse Monitoring

| Before | After | Benefit |
|--------|-------|---------|
| `monitoring/drift_detector.py` | Databricks Lakehouse Monitoring | Managed drift detection |
| `monitoring/alert.py` | Built-in alerting | Email/webhook notifications |
| Prometheus `/metrics` endpoint | Databricks SQL dashboards | Interactive visualization |
| Grafana dashboards | Databricks SQL dashboards | No separate BI tool needed |

---

## Databricks Concepts Demonstrated

### Core Platform

| Concept | Implementation | File |
|---------|---------------|------|
| **Delta Lake** | Raw + processed data as Delta tables with time travel | `notebooks/00_setup_environment.py`, `notebooks/01_preprocess.py` |
| **Unity Catalog** | Catalog `flp_catalog` with schemas (raw, processed, models, monitoring) | `notebooks/00_setup_environment.py` |
| **Managed MLflow** | Auto-tracked experiments, Unity Catalog model registry | `notebooks/02-05_*.py`, `notebooks/06_evaluate.py` |
| **Databricks Workflows** | Multi-task job with DAG dependencies and scheduling | `workflows/flp_pipeline_job.json` |
| **Model Serving** | 7 real-time endpoints with auto-scaling | `serving/endpoint_config.json` |
| **Databricks SQL** | Dashboards for metrics and demographics | SQL queries below |
| **Photon Engine** | Accelerated query performance on Delta tables | Automatic on Databricks Runtime |
| **Collaborative Notebooks** | Python notebooks with markdown documentation | All notebooks |

### Data Intelligence Platform Alignment

The migration aligns with the Databricks Data Intelligence Platform architecture:

```
┌──────────────────────────────────────────────────────────────┐
│                    Unity Catalog                             │
│            Unified Governance: Data, Analytics, AI           │
│                    assets, Lineage                           │
└──────────────────┬───────────────────────────────────────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
    ▼              ▼              ▼
┌────────┐  ┌───────────┐  ┌──────────────┐
│ Delta  │  │ Databricks│  │  Databricks  │
│ Lake   │  │ SQL &     │  │  Model       │
│(storage│  │ Mosaic AI │  │  Serving     │
│ layer) │  │(analytics)│  │  (inference) │
└────────┘  └───────────┘  └──────────────┘
    │              │              │
    ▼              ▼              ▼
┌──────────────────────────────────────────┐
│        Delta Live Tables (DLT)          │
│     Native ETL pipelines + scheduling   │
└──────────────────────────────────────────┘
```

---

## Databricks SQL Queries (for Dashboards)

### Demographic Overview
```sql
SELECT
  region,
  classe,
  sexe,
  COUNT(*) as n_respondents,
  AVG(age) as avg_age
FROM flp_catalog.processed.h1_features
GROUP BY region, classe, sexe
ORDER BY n_respondents DESC
```

### Model Metrics Summary
```sql
-- Run after evaluate notebook
SELECT
  'H1' as hypothesis, metrics['f1_macro'] as f1, metrics['roc_auc'] as auc
FROM flp_catalog.monitoring.pedagogical_report
```

### Drift Detection (Lakehouse Monitoring)
```sql
-- Compare recent predictions vs reference data
SELECT
  model_key,
  prediction_date,
  drift_ratio,
  CASE WHEN drift_ratio > 0.3 THEN 'ALERT' ELSE 'OK' END as status
FROM flp_catalog.monitoring.drift_reports
ORDER BY prediction_date DESC
```

---

## How to Run This Migration

### Prerequisites
1. Databricks workspace (Community Edition at [community.cloud.databricks.com](https://community.cloud.databricks.com) or Azure/AWS trial)
2. Databricks CLI configured: `databricks configure --token`
3. Data uploaded to DBFS: `databricks fs cp data_FLP.csv dbfs:/FileStore/flp/data_FLP.csv`

### Step-by-Step

```bash
# 1. Import notebooks to Databricks workspace
databricks workspace import_dir databricks/notebooks /Shared/FLP

# 2. Run setup notebook
# Open Databricks UI → /Shared/FLP/00_setup_environment → Run All

# 3. Run pipeline (notebooks in order)
# 01_preprocess → 02_train_h1 → 03_train_h2 → 04_train_h3 → 05_train_h4 → 06_evaluate

# 4. Create Workflow (one-time setup)
databricks jobs create --json-file databricks/workflows/flp_pipeline_job.json

# 5. Create Model Serving endpoints
databricks serving-endpoints create --json-file databricks/serving/endpoint_config.json
```

### Verification Checklist

- [ ] Delta tables visible in Unity Catalog explorer
- [ ] MLflow experiments show runs for each hypothesis
- [ ] Models registered in Unity Catalog Model Registry
- [ ] Workflow job runs successfully on schedule
- [ ] Model Serving endpoints respond to prediction requests
- [ ] SQL dashboards display metrics and demographics

---

## MLOps Maturity Level

| Level | Description | Before | After |
|-------|-------------|--------|-------|
| **Level 0** | Manual scripts, no tracking | ✅ | ✅ |
| **Level 1** | Automated pipeline, MLflow tracking | ✅ | ✅ |
| **Level 2** | CI/CD, containerized serving, model registry | ✅ (GitHub Actions + Docker) | ✅ (Databricks Workflows + Model Serving) |
| **Level 3** | Drift detection, auto-retraining, monitoring | 🔲 (custom scripts) | ✅ (Lakehouse Monitoring + Workflows) |

---

## Resume / LinkedIn Bullets

After completing this migration, you can add these to your profile:

- **Migrated end-to-end ML pipeline** (data preprocessing, 4 hypothesis-driven models, NLP feature engineering with CamemBERT) from local Docker/FastAPI to **Databricks Data Intelligence Platform**
- **Implemented Delta Lake** for ACID-compliant data storage with time travel and schema enforcement
- **Configured Unity Catalog** for unified governance of data assets, ML models, and monitoring artifacts
- **Deployed 7 real-time ML models** using **Databricks Model Serving** with auto-scaling and traffic splitting
- **Built Databricks Workflows** replacing GitHub Actions CI/CD with scheduled, dependency-aware job orchestration
- **Set up Lakehouse Monitoring** for automated drift detection and model performance tracking
- **Created Databricks SQL dashboards** for demographic analysis and model metrics visualization
- **Managed MLflow experiments** in Databricks with Unity Catalog Model Registry for model versioning and stage management
