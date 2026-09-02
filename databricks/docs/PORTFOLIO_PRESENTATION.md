# Databricks Portfolio — French-Learning-Perceptions ML

## One-Page Summary for Interviews / LinkedIn

**Project**: End-to-end ML system for analyzing French learning perceptions among 500 Cameroonian secondary school students, built on Databricks Data Intelligence Platform.

**Live Demo**: [Streamlit Dashboard](https://student-plurilingual-representation-french-learning-htyukpvkoo.streamlit.app/) — predictions served live from the Databricks Model Serving endpoint (server-side token, auto-fallback to local pickles)

**GitHub**: [Student-Plurilingual-Representation-French-Learning](https://github.com/womgaalbert/Student-Plurilingual-Representation-French-Learning)

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DATABRICKS DATA INTELLIGENCE PLATFORM                 │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     UNITY CATALOG                                │   │
│  │   flp_catalog                                                    │   │
│  │   ├── raw.survey_responses      ← 500 respondents, 34 columns    │   │
│  │   ├── processed.h1_features     ← H1: usage quotidien            │   │
│  │   ├── processed.h2_features     ← H2: motivation & difficultés   │   │
│  │   ├── processed.h3_features     ← H3: attitude envers français   │   │
│  │   ├── processed.h4_features     ← H4: engagement langues locales │   │
│  │   ├── models/                   ← 7 registered ML models          │   │
│  │   └── monitoring/               ← drift reports, alerts          │   │
│  └──────────────────────┬───────────────────────────────────────────┘   │
│                         │                                               │
│  ┌──────────────────────▼───────────────────────────────────────────┐   │
│  │              DELTA LAKE (The Intelligent Foundation)              │   │
│  │   ACID transactions | Time travel | Schema enforcement           │   │
│  └──────────────────────┬───────────────────────────────────────────┘   │
│                         │                                               │
│  ┌──────────┐  ┌────────▼────────┐  ┌──────────────┐  ┌────────────┐  │
│  │ Databricks│  │ Databricks      │  │ Databricks   │  │ Databricks │  │
│  │ Workflows │  │ Notebooks       │  │ MLflow       │  │ Model      │  │
│  │ (ETL/ML)  │  │ (Collaborative) │  │ (Experiments │  │ Serving    │  │
│  │           │  │                 │  │  + Registry) │  │ (7 models) │  │
│  │ preprocess│  │ 01_preprocess   │  │              │  │            │  │
│  │ train_h1  │  │ 02_train_h1     │  │ FLP_H1       │  │ /predict/  │  │
│  │ train_h2  │  │ 03_train_h2     │  │ FLP_H2       │  │ h1-h4      │  │
│  │ train_h3  │  │ 04_train_h3     │  │ FLP_H3_reg   │  │            │  │
│  │ train_h4  │  │ 05_train_h4     │  │ FLP_H3_clf   │  │            │  │
│  │ evaluate  │  │ 06_evaluate     │  │ FLP_H4       │  │            │  │
│  └──────────┘  └─────────────────┘  └──────────────┘  └────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │              LAKEHOUSE MONITORING + DATABRICKS SQL                │   │
│  │   Drift detection | Data quality | Performance dashboards         │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Skills Demonstrated (Databricks-Specific)

| Skill | Evidence | Interview Talking Point |
|-------|----------|------------------------|
| **Delta Lake** | All data stored as Delta tables with time travel and ACID | "I used Delta Lake versioning to track data changes across pipeline runs" |
| **Unity Catalog** | 4 schemas (raw, processed, models, monitoring) under one catalog | "I organized data and models under Unity Catalog for unified governance" |
| **Databricks MLflow** | Managed experiments + Unity Catalog Model Registry | "I tracked 100+ MLflow runs and registered 7 models with stage management" |
| **Databricks Workflows** | DAG-based job with 6 tasks and weekly scheduling | "I replaced GitHub Actions with Databricks Workflows for native scheduling" |
| **Model Serving** | 8 models behind one endpoint + public Streamlit app wired to it | "I bundled 8 models in one pyfunc endpoint (trial quota) and wired a public Streamlit dashboard to serve live predictions with zero credential exposure" |
| **Lakehouse Monitoring** | Automated drift detection on served models | "I set up drift detection that triggers alerts when data distribution shifts" |
| **Databricks SQL** | Demographic + metrics dashboards | "I built SQL dashboards that query Delta tables directly" |
| **Photon Engine** | Accelerated query and compute performance | "I leveraged Photon for fast feature engineering on 500×34 dataset" |

---

## Model Performance (Validated on Test Set)

| Hypothesis | Task | Metric | Score | Threshold | Status |
|-----------|------|--------|-------|-----------|--------|
| H1 | Binary classification | F1-macro | 0.835 | ≥0.70 | ✅ |
| H1 | Binary classification | ROC-AUC | 0.851 | ≥0.75 | ✅ |
| H2 | Multi-output (motivation) | F1-weighted | 0.954 | ≥0.65 | ✅ |
| H2 | Multi-output (difficulties) | F1-micro | 0.745 | ≥0.72 | ✅ |
| H3 | Regression | MAE | 0.513 | ≤0.50 | ⚠️ |
| H3 | Classification | F1-weighted | 0.780 | ≥0.68 | ✅ |
| H3 | Causal analysis | Pearson p | 0.984 | <0.05 | ⚠️ |
| H4 | Binary (motivation) | F1 | 0.807 | ≥0.70 | ✅ |
| H4 | Ordinal (engagement) | Spearman ρ | 0.561 | ≥0.55 | ✅ |
| H4 | Multi-label (discipline) | Subset accuracy | 1.000 | ≥0.45 | ✅ |

**3/4 hypotheses fully validated** — H3 borderline due to near-zero variance in exposure (93.5% of students exposed).

---

## Technical Stack

- **Data**: 500 survey responses, 34 columns (open text + categorical + ordinal)
- **NLP**: CamemBERT embeddings (768D → PCA 20D) for French text analysis
- **ML Models**: XGBoost, RandomForest, VotingClassifier, MultiOutputClassifier, ClassifierChain
- **MLOps**: MLflow (experiment tracking, model registry, serving)
- **Cloud Platform**: Databricks Data Intelligence Platform
- **Data Lakehouse**: Delta Lake + Unity Catalog
- **Orchestration**: Databricks Workflows (scheduled DAG)
- **Serving**: Databricks Model Serving (real-time endpoints)
- **Monitoring**: Lakehouse Monitoring + Databricks SQL

---

## Key Achievements

1. **Full MLOps maturity** — Level 0 through Level 3 implemented
2. **NLP for social science research** — CamemBERT embeddings for French text analysis of student perceptions
3. **Multi-hypothesis ML system** — 4 distinct ML tasks (binary, multi-output, regression, ordinal) in one platform
4. **End-to-end Databricks adoption** — Data → Features → Training → Registry → Serving → Monitoring, all on Databricks
5. **Production-ready deployment** — 8 models serving real-time predictions, including a public Streamlit dashboard consuming the live endpoint

---

## Interview Q&A Preparation

**Q: "Tell me about a Databricks project you've built."**
> "I built a full ML pipeline on Databricks for a social science research project analyzing French learning perceptions in Cameroon. I migrated from a local Docker/FastAPI setup to Databricks Data Intelligence Platform — using Delta Lake for data storage, Unity Catalog for governance, Databricks Workflows for orchestration, and Model Serving for real-time predictions across 7 models."

**Q: "How did you use Unity Catalog?"**
> "I created a catalog called `flp_catalog` with 4 schemas: `raw` for the original survey data, `processed` for feature-engineered datasets per hypothesis, `models` for the MLflow model registry, and `monitoring` for drift reports. This gave us unified access control and lineage tracking across the entire ML lifecycle."

**Q: "How did you handle model deployment?"**
> "I registered 8 models in Unity Catalog Model Registry — covering binary classification, multi-output classification, regression, and ordinal prediction. Because the trial workspace allowed only 2 serving endpoints, I bundled all 8 models into a single pyfunc model dispatched by a 'model_name' input column, deployed it via Databricks Model Serving with scale-to-zero, and wired the public Streamlit dashboard to it through server-side secrets — so end users get live predictions without any access to the data or credentials."

**Q: "What about monitoring and maintenance?"**
> "I set up Lakehouse Monitoring to track drift on our served models, and Databricks Workflows to run the full pipeline weekly. The workflow has a DAG: preprocess runs first, then H1-H4 train in parallel, then evaluate validates thresholds and generates a report."
