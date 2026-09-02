# Student-Plurilingual-Representation-French-Learning — Databricks

[![Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://student-plurilingual-representation-french-learning-htyukpvkoo.streamlit.app/)
[![Databricks](https://img.shields.io/badge/Databricks-Serverless-orange)](https://www.databricks.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Managed-0194E2)](https://mlflow.org/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **ML Research Platform on Databricks Data Intelligence Platform**
> Delta Lake · Unity Catalog · Databricks Workflows · Managed MLflow · Model Serving

A machine learning system analyzing how Cameroonian secondary school students'
**representations of the French language** are shaped by their plurilingual context —
and how these representations influence motivation, learning difficulties, and engagement.
The entire ML lifecycle — data, features, training, registry and serving — runs on
**Databricks** (serverless compute), replacing the previous Docker/FastAPI setup.

🎛️ **Live dashboard → [student-plurilingual-representation-french-learning-htyukpvkoo.streamlit.app](https://student-plurilingual-representation-french-learning-htyukpvkoo.streamlit.app/)**
— predictions served live from Databricks Model Serving (token stays server-side; falls back to local pickles if unconfigured).

---

## 🎓 Research Context

**Thesis title:**
*Les représentations des élèves du premier cycle de l'enseignement secondaire
camerounais sur l'apprentissage du français dans leur contexte plurilingue*

**Principal Researcher:** [Chancelline Armelle Nongni Kendjio](https://www.linkedin.com/in/chancelline-armelle-nongni-kendjio-9840582b/)
&nbsp;|&nbsp; Université Marie & Louis Pasteur de Besançon (France)

**Technical Implementation:** [Albert Womga](https://www.linkedin.com/in/albert-womga-009a7931/) — ML Engineering

**Dataset:** 500 respondents (495 after consent filter) · 34 columns · Cameroon

---

## ☁️ Databricks Architecture

```
┌──────────────────────────── DATABRICKS (serverless-only trial) ───────────────────────────┐
│  Unity Catalog: flp_catalog                                                                │
│   ├── raw.survey_responses      ← 500 rows from UC Volume (Files API, not DBFS)            │
│   ├── processed.h1..h4_features ← feature engineering on Delta Lake (495 rows)             │
│   ├── models/                   ← 9 registered models (8 hypotheses + flp_all bundle)      │
│   └── monitoring.pedagogical_report ← thresholds validation (H1✅ H2✅ H3⚠️ H4⚠️)          │
│                                                                                            │
│  Databricks Workflows (job 562108964197707, serverless, weekly-paused)                     │
│   setup → preprocess → train_h1 ─┐                                                         │
│                        train_h2 ─┼→ evaluate → serving_bundle (rebuilds flp_all)           │
│                        train_h3 ─┤                                                         │
│                        train_h4 ─┘                                                         │
│                                                                                            │
│  Managed MLflow (experiments /Shared/FLP_H1..H4) → UC Model Registry                       │
│                                                                                            │
│  Model Serving — endpoint "flp-all-models" (Small, scale-to-zero)                          │
│   one pyfunc entity dispatches 8 models via a "model_name" input column                    │
│   (free tier limit: 2 endpoints / 2 provisioned concurrency → all models bundled)          │
└────────────────────────────────────────────────────────────────────────────────────────────┘
        ▲ REST (POST /serving-endpoints/flp-all-models/invocations)
        │  Authorization: Bearer <token> — held in Streamlit Cloud secrets, never in the repo
┌───────┴────────┐
│  Streamlit app │  deploy_cloud/app.py + databricks_backend.py (FR/EN, live predictions)
└────────────────┘
```

---

## 🔬 4 Hypotheses — 8 ML Models (Databricks run)

| ID | Hypothesis | ML Task | Model | Key Metric | Status |
|----|-----------|---------|-------|-------------|--------|
| H1 | Multilingual repertoire → daily language use | Binary clf | XGBoost + SMOTE | F1=0.79 · AUC=0.84 | ✅ |
| H2 | French representations → motivation & difficulties | Multi-output clf | XGBoost (3-cls + multi-label) | F1=0.95 · F1-micro=0.80 | ✅ |
| H3 | Plurilingual exposure → attitudes toward French | Regression + clf + causal | RandomForest + XGBoost | MAE=0.47 · F1=0.45 | ⚠️ |
| H4 | Local language integration → engagement | Multi-label + ordinal | Voting + XGBoost | F1=0.55 · ρ=1.00 | ⚠️ |

**2/4 hypotheses validated on Databricks** — full report in `flp_catalog.monitoring.pedagogical_report`.
(The ⚠️ flags are honest outcomes; H3/H4 thresholds were met on the local CamemBERT-PCA pipeline.)

---

## 🚀 Quick Start — Databricks

```powershell
# 1. Configure the CLI (token + host)
pip install databricks-cli
databricks configure --token --host https://dbc-9e268203-7090.cloud.databricks.com

# 2. Import notebooks + data + create the workflow
databricks workspace import_dir databricks/notebooks /Shared/FLP --overwrite
#    (upload data_FLP.csv to volume flp_catalog.raw.data_files via the Files API)

# 3. Create the job (Jobs API 2.2, serverless compute)
#    see databricks/workflows/flp_pipeline_job.json

# 4. Run the full pipeline
databricks jobs run-now --job-id 562108964197707

# 5. Serve predictions (single bundled endpoint)
python databricks/invoke_endpoint.py all     # token from DATABRICKS_TOKEN env or ~/.databrickscfg
python databricks/invoke_endpoint.py h1      # h1, h2a, h2b, h3r, h3c, h4a, h4b, h4c, all
```

Local development (original pipeline) still works:

```powershell
.\scripts\setup_windows.ps1
python src/pipeline.py --config params.yaml
mlflow ui --backend-store-uri mlruns --port 5000
```

---

## 📁 Project Structure

```
├── databricks/                       ← ★ Databricks migration
│   ├── notebooks/
│   │   ├── 00_setup_environment.py   ← UC catalog + Delta raw table (Volume CSV)
│   │   ├── 01_preprocess.py          ← Preprocessing + 4 feature tables (Delta)
│   │   ├── 02..05_train_h{1-4}.py    ← Training + MLflow + UC registration
│   │   ├── 06_evaluate.py            ← Thresholds + pedagogical report
│   │   └── 07_combine_models.py      ← 8 models → single pyfunc serving bundle
│   ├── workflows/flp_pipeline_job.json ← 8-task serverless workflow DAG
│   ├── serving/endpoint_config.json  ← Single endpoint (flp-all-models)
│   ├── docs/
│   │   ├── MIGRATION_GUIDE.md        ← Before/after architecture
│   │   ├── PORTFOLIO_PRESENTATION.md ← Interview / LinkedIn material
│   │   └── SHARING_GUIDE.md          ← Share with the jury — zero access
│   ├── invoke_endpoint.py            ← Live endpoint test script (no token in repo)
│   └── setup_databricks.ps1 / .sh
├── deploy_cloud/                     ← 🎛️ Streamlit Cloud app
│   ├── app.py                        ← Dashboard (FR/EN)
│   ├── databricks_backend.py         ← ★ live mode: REST → Model Serving (no MLflow)
│   └── .streamlit/secrets.toml.example ← DATABRICKS_HOST + DATABRICKS_TOKEN template
├── src/                              ← Local pipeline (preprocess, train, evaluate…)
├── api/main.py                       ← FastAPI (legacy local serving)
├── monitoring/                       ← MLOps Level 3 (drift, alerts, Grafana)
├── data/raw/                         ← READ-ONLY (NEVER versioned)
└── params.yaml                       ← Hyperparameters (single source of truth)
```

---

## 🎛️ Streamlit Live Mode (Databricks, no MLflow on the app side)

- `deploy_cloud/databricks_backend.py` POSTs JSON to
  `/serving-endpoints/flp-all-models/invocations` using only `requests` —
  **the app has no MLflow / sklearn / xgboost dependency in live mode**.
- Credentials live in Streamlit Cloud → Settings → Secrets (`DATABRICKS_HOST`,
  `DATABRICKS_TOKEN`) — never committed (`secrets.toml` is gitignored).
- Without secrets the app falls back to the local `.pkl` models; the sidebar
  badge shows the active mode (🟢 Databricks live / 🟡 local).
- Scale-to-zero note: the first call after idle takes 1–2 min (the UI shows a
  "waking the model" spinner).

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Cloud platform | Databricks Data Intelligence Platform (serverless compute) |
| Data lakehouse | Delta Lake + Unity Catalog (`flp_catalog`) |
| Orchestration | Databricks Workflows (8-task DAG, paused weekly schedule) |
| Experiment tracking | Databricks-managed MLflow + UC Model Registry (9 models) |
| Serving | Databricks Model Serving (1 endpoint, 8 models bundled, scale-to-zero) |
| French NLP | CamemBERT (camembert-base), spaCy fr_core_news_sm (local pipeline) |
| ML Models | XGBoost, RandomForest, VotingClassifier, MultiOutputClassifier |
| Dashboard | Streamlit (bilingual FR/EN, live from Databricks) |
| Legacy local stack | FastAPI, Docker Compose, GitHub Actions (3 workflows, 42 tests) |
| Monitoring | Evidently AI, Prometheus, Grafana (local Level 3) |
| Tests | pytest + pytest-cov (42 tests) |

---

## 🔗 Inference API (Databricks Model Serving)

Single endpoint — dispatch with the `model_name` input column:

| model_name | Returns |
|-----------|---------|
| `h1` | H1 — Daily language use (pred + proba) |
| `h2a` | H2 — Motivation level (0-2 + confidence) |
| `h2b` | H2 — Difficulty types (7 labels) |
| `h3r` | H3 — Attitude score (1.0–5.0) |
| `h3c` | H3 — Attitude class (0-2) |
| `h4a` | H4 — Local-language motivation (pred + proba) |
| `h4b` | H4 — Engagement score (1–4) |
| `h4c` | H4 — Preferred disciplines (5 labels) |
| `all` | Everything above in one call |

```bash
curl -X POST \
  -H "Authorization: Bearer $DATABRICKS_TOKEN" \
  -H "Content-Type: application/json" \
  "https://dbc-9e268203-7090.cloud.databricks.com/serving-endpoints/flp-all-models/invocations" \
  -d '{"dataframe_records":[{...60 features..., "model_name":"all"}]}'
```

The full 60-feature contract + labels are documented in `databricks/invoke_endpoint.py`
and `deploy_cloud/databricks_backend.py`.

---

## 🛡️ Data Privacy

- Raw data (`data/raw/`) is **excluded from version control** (`.gitignore`)
- All data is anonymized before processing (timestamps & identifiers removed)
- Only respondents with valid consent ("J'accepte") are included (495/500)
- The serving endpoint transmits only 60 engineered features per prediction —
  never raw responses
- Databricks token is never in the repo; `secrets.toml` is gitignored
- Model outputs must not be used to individually grade or rank students

---

## 📄 License

MIT License — open for academic use and research collaboration.

---

## 🤝 Contributing

This is an active doctoral research project. If you work on French NLP, multilingualism
in education, or Cameroonian linguistics, feel free to open an issue or pull request.

---

*ML Engineering by [Albert Womga](https://www.linkedin.com/in/albert-womga-009a7931/)
in support of doctoral thesis research by
[Chancelline Armelle Nongni Kendjio](https://www.linkedin.com/in/chancelline-armelle-nongni-kendjio-9840582b/)*
