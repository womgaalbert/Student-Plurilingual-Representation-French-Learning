# French-Learning-Perceptions in Plurilingual Cameroon

[![CI — Tests](https://github.com/womgaalbert/Student-Plurilingual-Representation-French-Learning/actions/workflows/ci.yml/badge.svg)](https://github.com/womgaalbert/Student-Plurilingual-Representation-French-Learning/actions/workflows/ci.yml)
[![Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://student-plurilingual-representation-french-learning-htyukpvkoo.streamlit.app/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![MLOps Level](https://img.shields.io/badge/MLOps-Level%203-brightgreen)]()

> **ML Research Platform** | MLOps Level 2 | FastAPI · Docker · Streamlit Cloud

A machine learning system analyzing how Cameroonian secondary school students'
**representations of the French language** are shaped by their plurilingual context —
and how these representations influence motivation, learning difficulties, and engagement.

🎛️ **Live dashboard → [student-plurilingual-representation-french-learning-htyukpvkoo.streamlit.app](https://student-plurilingual-representation-french-learning-htyukpvkoo.streamlit.app/)**

---

## 🎓 Research Context

**Thesis title:**
*Les représentations des élèves du premier cycle de l'enseignement secondaire
camerounais sur l'apprentissage du français dans leur contexte plurilingue*

**Principal Researcher:** [Chancelline Armelle Nongni Kendjio](https://www.linkedin.com/in/chancelline-armelle-nongni-kendjio-9840582b/)
&nbsp;|&nbsp; Université Marie & Louis Pasteur de Besançon (France)
&nbsp;|&nbsp; Under supervision of Prof. Serge Borg

**Technical Implementation:** [Albert Womga](https://www.linkedin.com/in/albert-womga-009a7931/) — ML Engineering

**Dataset:** 500 respondents · 15 schools · Cameroon

---

## 🔬 4 Hypotheses — 7 ML Models

| ID | Hypothesis | ML Task | Model | Key Metric | Status |
|----|-----------|---------|-------|-------------|--------|
| H1 | Multilingual repertoire → daily language use | Binary clf | XGBoost | F1=0.835 · AUC=0.851 | ✅ |
| H2 | French representations → motivation & difficulties | Multi-output clf | ClassifierChain + XGBoost | F1=0.954 | ✅ |
| H3 | Plurilingual exposure → attitudes toward French | Regression + causal | VotingRegressor (ET+XGBoost) | MAE=0.513 | ⚠️ |
| H4 | Local language integration → engagement | Multi-label + ordinal | XGBoost multi-task | F1=0.807 · ρ=0.561 | ✅ |

**3/4 hypotheses validated** — Full evaluation report in `reports/`.

---

## 🚀 Quick Start (Windows)

```powershell
# 1. Clone
git clone https://github.com/womgaalbert/Student-Plurilingual-Representation-French-Learning.git
cd Student-Plurilingual-Representation-French-Learning

# 2. Setup venv + dependencies
.\scripts\setup_windows.ps1

# 3. Add your data → Copy data_FLP.csv to data/raw/ (NEVER committed)

# 4. Run the full pipeline
python src/pipeline.py --config params.yaml

# 5. Launch MLflow UI
mlflow ui --backend-store-uri mlruns --port 5000
# → http://localhost:5000

# 6. Launch API (optional)
uvicorn api.main:app --host 0.0.0.0 --port 8001
# → http://localhost:8001/docs
```

Or with Docker:
```bash
docker compose up --build -d
# API → http://localhost:8001
# MLflow → http://localhost:5000
```

---

## 📁 Project Structure

```
├── .github/workflows/           ← CI/CD (tests, train, deploy)
│   ├── ci.yml                   ← Windows tests on push/PR
│   ├── tests.yml                ← Ubuntu tests
│   └── train.yml                ← Train & evaluate pipeline
├── api/
│   └── main.py                  ← FastAPI — 8 /predict endpoints
├── src/
│   ├── preprocess.py            ← Cleaning, encoding, feature engineering
│   ├── descriptive_analysis.py  ← CamemBERT, UMAP, stereotypes, n-grams
│   ├── train.py                 ← Training + tuning H1-H4 + MLflow
│   ├── pipeline.py              ← Full automated pipeline (10 steps)
│   ├── evaluate.py              ← Global metrics + JSON report
│   ├── models/                  ← Train scripts per hypothesis (gitignored)
│   └── utils/
│       ├── constants.py         ← CSV columns, encodings, keywords
│       ├── config.py            ← params.yaml loader
│       ├── embeddings.py        ← CamemBERT → PCA features (lazy import)
│       └── mlflow_utils.py      ← MLflow helpers + experiment setup
├── app.py                       ← 🎛️ Streamlit dashboard
├── deploy_cloud/
│   ├── models/                  ← 7 production .pkl models (tracked)
│   ├── reports/                 ← Descriptive analysis PNGs (tracked)
│   └── requirements-full.txt   ← Full dev dependencies (torch, spaCy, etc.)
├── data/
│   ├── raw/                     ← READ-ONLY (NEVER versioned)
│   └── processed/               ← Pipeline outputs
├── models/h{1-4}/               ← Serialized models (gitignored)
├── monitoring/                  ← MLOps Level 3: drift, alerts, Grafana
├── mlruns/                      ← MLflow tracking store
├── reports/                     ← Evaluation + demographics reports
├── tests/
│   ├── conftest.py
│   └── test_flp.py              ← 42 unit tests
├── scripts/
│   └── setup_windows.ps1        ← Windows environment bootstrap
├── docker-compose.yml           ← API + MLflow orchestration
├── Dockerfile                   ← FastAPI serving image
├── params.yaml                  ← All hyperparameters (single source of truth)
└── requirements.txt             ← Lightweight deps (Streamlit Cloud + CI)
```

---

## 🏗️ MLOps Maturity

```
✅ Level 0  Manual scripts, params.yaml, data separation, 42 unit tests
✅ Level 1  Automated pipeline (preprocess→H1→H4→evaluate), MLflow tracking
✅ Level 2  CI/CD (GitHub Actions), FastAPI /predict, Docker Compose, Streamlit Cloud
✅ Level 3  Drift detection (Evidently + scipy), Prometheus/Grafana, auto-retrain workflow
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| French NLP | CamemBERT (camembert-base), spaCy fr_core_news_sm |
| ML Models | XGBoost, LightGBM, ExtraTrees, VotingRegressor, ClassifierChain |
| Experiment Tracking | MLflow (116 runs, 7 models in registry) |
| API | FastAPI + Uvicorn (8 endpoints) |
| Dashboard | Streamlit (bilingual FR/EN, live predictions) |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions (3 workflows, 42 tests) |
| Monitoring (Level 3) | Evidently AI, Prometheus, Grafana, alerting |
| Visualization | Matplotlib, Seaborn, UMAP, NetworkX, WordCloud |
| Tests | pytest + pytest-cov (42 tests, Windows + Linux) |
| Environment | Python 3.10+ · Windows · Ubuntu |

---

## 🔗 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check + models loaded |
| GET | `/models` | List available models |
| POST | `/predict/h1` | H1 — Daily language use |
| POST | `/predict/h2/motivation` | H2 — Motivation level |
| POST | `/predict/h2/difficultes` | H2 — Difficulty types |
| POST | `/predict/h3/attitude` | H3 — Attitude score |
| POST | `/predict/h3/classification` | H3 — Attitude class |
| POST | `/predict/h4/motivation` | H4 — Local language motivation |
| POST | `/predict/h4/engagement` | H4 — Engagement score |
| POST | `/predict/h4/disciplines` | H4 — Preferred disciplines |

Swagger UI: `http://localhost:8001/docs`

---

## 🛡️ Data Privacy

- Raw data (`data/raw/`) is **excluded from version control** (`.gitignore`)
- All data is anonymized before processing (timestamps & identifiers removed)
- Only respondents with valid consent ("J'accepte") are included
- Model outputs must not be used to individually grade or rank students
- `data_FLP.csv` is the private property of the researcher — never displayed on GitHub

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
