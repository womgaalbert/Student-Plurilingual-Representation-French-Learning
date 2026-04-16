# French-Learning-Perceptions-in-Plurilingual-Cameroon

## Research
Thesis: "Les représentations des élèves du premier cycle de l'enseignement
secondaire camerounais sur l'apprentissage du français dans leur contexte plurilingue"

Researcher : Chancelline Armelle Nongni Kendjio
            https://www.linkedin.com/in/chancelline-armelle-nongni-kendjio-9840582b/
Technical  : AI/ML Engineering support

## GitHub Repository
https://github.com/womgaalbert/Student-Plurilingual-Representation-French-Learning

## Environment
- OS        : Windows (PowerShell / VS Code terminal)
- Python    : 3.10+ (.venv)
- Git       : GitHub — main / dev / feature/* / exp/*
- Tracking  : MLflow local (mlruns/)
- MLOps     : Level 0 ✅ | Level 1 ✅ | Level 2 🔲 | Level 3 🔲

## Data
- File      : data/raw/data_FLP.csv
- N         : 500 respondents (consent "J'accepte" required)
- Columns   : 34 (open text + categorical + ordinal)

## 4 Hypotheses
| ID | Hypothesis                                         | ML Task              |
|----|----------------------------------------------------|----------------------|
| H1 | Multilingual repertoire → daily mobilization       | Binary clf           |
| H2 | French representations → motivation & difficulties | Multi-output clf     |
| H3 | Plurilingual exposure → attitudes toward French    | Regression + causal  |
| H4 | Local language integration → engagement            | Multi-label + ordinal|

## Claude Code Commands
- /project:setup        — Windows environment setup
- /project:preprocess   — clean + encode CSV
- /project:train-h1     — train H1 + MLflow log
- /project:train-h2     — train H2 + MLflow log
- /project:train-h3     — train H3 + MLflow log
- /project:train-h4     — train H4 + MLflow log
- /project:train-all    — full pipeline H1→H4
- /project:evaluate     — metrics + pedagogical report
- /project:mlflow-ui    — launch MLflow UI (port 5000)
- /project:test         — run pytest suite

## Absolute Rules
1. NEVER write to data/raw/ (read-only)
2. Verify consent column before any processing
3. Anonymize before any export
4. All hyperparameters live in params.yaml only
5. Every training run must be logged to MLflow
