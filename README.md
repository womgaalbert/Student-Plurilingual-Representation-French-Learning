# French-Learning-Perceptions-in-Plurilingual-Cameroon

> **ML Research Platform** | MLOps Level 0 → Level 1

A machine learning system analyzing how Cameroonian secondary school students' **representations of the French language** are shaped by their multilingual context — and how these representations influence motivation, learning difficulties, and engagement.

---

## 🎓 Research Context

**Thesis title:**
*Les représentations des élèves du premier cycle de l'enseignement secondaire camerounais sur l'apprentissage du français dans leur contexte plurilingue*

**Principal Researcher:** [Chancelline Armelle Nongni Kendjio](https://www.linkedin.com/in/chancelline-armelle-nongni-kendjio-9840582b/)
**Technical Implementation:** AI/ML Engineering support for thesis research

**Dataset:** 500 student respondents (Google Forms survey, 34 questions)
**Languages:** French NLP pipeline (CamemBERT, spaCy fr_core_news_sm)

---

## 🔬 Research Hypotheses & ML Models

| ID | Hypothesis | ML Task | Min. Threshold |
|----|-----------|---------|----------------|
| H1 | Multilingual repertoire → daily language mobilization | Binary classification | F1 ≥ 0.70, AUC ≥ 0.75 |
| H2 | French representations → motivation & difficulties | Multi-output classification | F1-A ≥ 0.65, F1-B ≥ 0.72 |
| H3 | Plurilingual exposure → attitudes toward French | Regression + causal inference | MAE ≤ 0.50, p < 0.05 |
| H4 | Local language integration → engagement | Multi-label + ordinal | F1 ≥ 0.70, ρ ≥ 0.55 |

---

## 🚀 Quick Start (Windows)

```powershell
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/Student-Plurilingual-Representation-French-Learning.git
cd Student-Plurilingual-Representation-French-Learning

# 2. Run setup (creates venv, installs dependencies)
.\scripts\setup_windows.ps1

# 3. Add your data
# → Copy data_FLP.csv to data/raw/

# 4. Run preprocessing (Level 0)
python src/preprocess.py --config params.yaml

# 5. Train a single hypothesis (Level 0)
python src/train.py --hypothesis H1 --config params.yaml

# 6. Run the full automated pipeline (Level 1)
python src/pipeline.py --config params.yaml

# 7. View results in MLflow
mlflow ui --backend-store-uri mlruns --port 5000
# → Open http://localhost:5000
```

---

## 📁 Project Structure

```
Student-Plurilingual-Representation-French-Learning/
├── CLAUDE.md                    ← Claude Code session context
├── params.yaml                  ← All hyperparameters (never hardcoded)
├── requirements.txt
├── Makefile
├── .claude/                     ← Claude Code configuration
│   ├── settings.json            ← Permissions & hooks
│   ├── rules/                   ← Code style, testing, MLOps levels
│   ├── skills/                  ← H1–H4 Synoptique indicators
│   ├── agents/                  ← Data auditor, feature engineer
│   ├── commands/                ← /project:preprocess | train | pipeline
│   └── hooks/                   ← validate-bash.ps1 (Windows)
├── data/
│   ├── raw/                     ← READ-ONLY (not versioned)
│   └── processed/               ← Pipeline outputs
├── src/
│   ├── preprocess.py            ← Level 0 preprocessing
│   ├── train.py                 ← Level 0 training (H1–H4) + MLflow
│   ├── pipeline.py              ← Level 1 automated pipeline
│   ├── evaluate.py              ← Metrics + pedagogical report
│   └── utils/
│       ├── constants.py         ← Column mappings & encodings
│       ├── config.py            ← params.yaml loader
│       └── mlflow_utils.py      ← MLflow helpers
├── notebooks/
│   └── flp_pipeline.ipynb      ← Step-by-step ML walkthrough
├── models/h{1-4}/               ← Serialized models (.pkl)
├── mlruns/                      ← MLflow local tracking
├── reports/                     ← JSON metrics per run
├── tests/
│   ├── conftest.py
│   └── test_flp.py              ← 40+ unit tests
└── scripts/
    └── setup_windows.ps1        ← Windows environment setup
```

---

## 🏗️ MLOps Roadmap

```
✅ Level 0  Manual scripts, params.yaml, data separation, unit tests
✅ Level 1  Automated pipeline, MLflow tracking, nested runs
🔲 Level 2  GitHub Actions CI/CD, Model Registry, FastAPI /predict
🔲 Level 3  Data drift detection, auto-retraining, live monitoring
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| French NLP | CamemBERT (camembert-base), spaCy fr_core_news_sm |
| Tabular ML | XGBoost, scikit-learn, LightGBM |
| Deep Learning | PyTorch + HuggingFace Transformers |
| Causal inference (H3) | DoWhy, EconML |
| Explainability | SHAP, LIME |
| Experiment tracking | MLflow (local) |
| Tests | pytest (40+ unit tests) |
| Environment | Python 3.10+, Windows + VS Code |

---

## 📊 MLflow Experiments

| Experiment | Hypothesis | Key Metrics |
|------------|-----------|-------------|
| `FLP_H1_Multilingual_Repertoire` | H1 | F1-macro, ROC-AUC |
| `FLP_H2_French_Representations` | H2 | F1-weighted (A), F1-micro (B) |
| `FLP_H3_Plurilingual_Exposure` | H3 | MAE, Pearson r, ATE |
| `FLP_H4_Local_Language_Integration` | H4 | F1, Spearman ρ, Subset acc |

---

## 🛡️ Data Privacy

- Raw data (`data/raw/`) is **excluded from version control** (.gitignore)
- All data is anonymized before processing (timestamps & identifiers removed)
- Only respondents with valid consent ("J'accepte") are included
- Model outputs must not be used to individually grade or rank students

---

## 📄 License

MIT License — open for academic use and research collaboration.

---

## 🤝 Contributing

This is an active research project. If you are working on French NLP, multilingualism in education, or Cameroonian linguistics, feel free to open an issue or pull request.

---

*Technical implementation by an AI/ML Engineer in support of doctoral thesis research.*
