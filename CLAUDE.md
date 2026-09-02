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

## Full ML Pipeline — Ordre obligatoire par Hypothese

> Chaque hypothese suit les 12 phases completes du ML process.
> Ne jamais sauter une phase ni inverser l'ordre.
> Toujours commencer par les donnees demographiques globales.

---

### ETAPE 0 — Prerequis & Setup
- [ ] Copier `data_FLP.csv` dans `data/raw/` (READ-ONLY ensuite)
- [ ] Executer `scripts/setup_windows.ps1` (venv + deps + dossiers)
- [ ] Filtrage consentement : garder uniquement "J'accepte"
- [ ] Anonymisation : suppression Horodateur + identifiants
```powershell
python src/preprocess.py --config params.yaml --stage global
```

---

### ETAPE 1 — Description demographique globale
> A faire UNE FOIS avant toute analyse par hypothese.
- Age : distribution, moyenne, ecart-type, visualisation histogramme
- Niveau scolaire : frequences par classe (6e, 5e, 4e, 3e)
- Region / etablissement : carte de repartition
- Genre : distribution (si disponible)
- Langue(s) parlees a la maison : frequences multi-reponses
- Rapport : `reports/demographics/demographic_report.html`
```powershell
python src/descriptive_analysis.py --stage demographics --config params.yaml
```

---

### CYCLE COMPLET PAR HYPOTHESE (repete pour H1, H2, H3, H4)

#### Phase 1 — Data Cleaning & Preprocessing
- Identification des variables concernees par l'hypothese
- Tableau des valeurs manquantes par variable :

| Variable | N manquants | % manquants | Approche retenue |
|----------|-------------|-------------|-----------------|
| var_1    | ?           | ?%          | mode / median / KNN / flag |
| ...      | ...         | ...         | ...             |

Regles d'imputation :
- Categorielle < 5% manquants → mode
- Categorielle >= 5% manquants → categorie "Non-reponse" explicite
- Ordinale < 5% manquants → mediane
- Ordinale >= 5% manquants → KNNImputer (k=5)
- Texte ouvert manquant → chaine vide "" (pas d'imputation)
- Si variable > 30% manquants → exclure + documenter

```powershell
python src/preprocess.py --hypothesis H{n} --config params.yaml
```

#### Phase 2 — Exploratory Data Analysis (EDA)
- Statistiques descriptives : mean, std, min, max, quartiles
- Distribution de chaque variable (histogramme + boxplot)
- Matrice de correlation (Spearman pour ordinales)
- Detection outliers : IQR method sur variables numeriques
- Rapport EDA → `reports/h{n}/eda_report.html`

#### Phase 3 — Analyse Descriptive Textuelle
- **Statistiques textuelles** : longueur moyenne des reponses, richesse lexicale
- **CamemBERT (camembert-base)** : embeddings + clustering UMAP + K-Means
- **Analyse A Priori** (Moscovici, Jodelet) — classification par similarite cosinus :

| Categorie a priori          | Exemples attendus                        |
|-----------------------------|------------------------------------------|
| Representation utilitaire   | "necessite", "reussir", "travail"        |
| Representation identitaire  | "langue des autres", "pas la mienne"     |
| Representation affective    | "j'aime", "j'ai peur", "difficile"       |
| Representation institutionn.| "ecole", "professeur", "obligatoire"     |
| Resistance / contrainte     | "force", "oblige", "pas le choix"        |

- **Stereotypes & preconstruits** (prioritaire H2) :

| Marqueur                    | Preconstruit detecte              |
|-----------------------------|-----------------------------------|
| "langue des colons/blancs"  | Distanciation identitaire         |
| "trop difficile"            | Preconstruit de difficulte        |
| "inutile dans ma vie"       | Preconstruit d'inutilite          |
| "je suis nul en francais"   | Auto-devalorisation               |
| "pas pour nous"             | Exclusion symbolique              |
| "obliges de l'apprendre"    | Contrainte / resistance           |

- **N-grammes** (CountVectorizer + TfidfVectorizer, stopwords spaCy fr_core_news_sm) :
  - Unigrammes → vocabulaire dominant, nuage de mots
  - Bigrammes  → collocations ("tres difficile", "langue officielle")
  - Trigrammes → formules figees ("c'est trop dur", "pas pour nous")
  - Top-20 par type → bar charts

- **Visualisations** → `reports/h{n}/descriptive/` :
  - Nuage de mots global et par categorie a priori
  - Carte UMAP coloree par stereotype detecte
  - Bar charts Top-20 bi/trigrammes
  - Graphe de reseau co-occurrences (NetworkX)
  - Stacked bar distribution categories a priori

```powershell
python src/descriptive_analysis.py --hypothesis H{n} --config params.yaml
```

#### Phase 4 — Feature Engineering
- Construction des features specifiques a l'hypothese
- Encodage ordinal via FREQ_MAP (Toujours/Souvent/Parfois/Rarement/Jamais)
- Embeddings CamemBERT comme features numeriques (dim 768 → reduction PCA/UMAP)
- Scores composites (H3: h3_score_attitude, H4: h4_engagement_score) — AVANT split
- Normalisation / standardisation des features numeriques
- Toutes les colonnes via constantes COL_* de utils/constants.py

#### Phase 5 — Data Splitting (Train / Validation / Test)
- Split stratifie sur la target : 70% train / 15% val / 15% test
- Seed fixe dans params.yaml (reproductibilite)
- Verification distribution target identique dans les 3 splits
- SMOTE applique uniquement sur X_train (jamais val/test)
- Scaler/Imputer : fit sur X_train uniquement → transform val et test

#### Phase 6 — Model Selection
- Criteres de selection : complexite des donnees, taille N, interpretabilite
- Modeles candidats par hypothese :

| Hypothese | Modeles candidats                              |
|-----------|------------------------------------------------|
| H1        | LogisticRegression, RandomForest, XGBoost      |
| H2        | MultiOutputClassifier, ClassifierChain         |
| H3        | Ridge, SVR, XGBoostRegressor + DoWhy causal    |
| H4        | MultiLabelBinarizer + OrdinalClassifier (mord) |

- Baseline systematique : DummyClassifier / DummyRegressor
- Cross-validation 5-fold sur train pour comparaison initiale

#### Phase 7 — Model Training
- Entrainement sur X_train uniquement
- MLflow run obligatoire : params + metriques + artefact .pkl
- Tags : status OK / WARNING selon seuils
```powershell
python src/train.py --hypothesis H{n} --config params.yaml
```

#### Phase 8 — Model Evaluation
- Evaluation sur X_test (jamais vu pendant l'entrainement)
- Metriques par hypothese :

| Hypothese | Metriques                                    | Seuils               |
|-----------|----------------------------------------------|----------------------|
| H1        | F1-macro, ROC-AUC, matrice confusion         | F1>=0.70, AUC>=0.75  |
| H2        | F1-weighted (A), F1-micro (B)                | >=0.65, >=0.72       |
| H3        | MAE, F1-weighted clf, Pearson p              | <=0.50, >=0.68, <0.05|
| H4        | F1 binaire, Spearman rho, Subset accuracy    | >=0.70, >=0.55, >=0.45|

- SHAP values pour interpretabilite (features les plus importantes)
- Rapport evaluation → `reports/h{n}/evaluation_report.json`

#### Phase 9 — Hyperparameter Tuning
- Outil : GridSearchCV ou Optuna (selon params.yaml)
- Espace de recherche defini dans params.yaml uniquement
- Optimisation sur validation set (pas test)
- Meilleur modele re-entraine sur train+val avant evaluation finale
- Nouveau MLflow run avec tag "tuned"

#### Phase 10 — Model Validation (Generalization Check)
- Evaluation finale sur X_test (une seule fois)
- Verification absence de data leakage (tests pytest)
- Comparaison baseline vs modele final vs modele tune
- Rapport de validation → `reports/h{n}/validation_report.md`

#### Phase 11 — Model Packaging & Deployment (Level 2 MLOps)
- Serialisation modele : `models/h{n}/model_final.pkl`
- MLflow Model Registry : staging → production
- FastAPI endpoint : `api/main.py` → `/predict/H{n}`
- Dockerfile + docker-compose.yml
- GitHub Actions : `.github/workflows/train.yml`

#### Phase 12 — Monitoring & Drift Detection + Maintenance (Level 3 MLOps)
- Evidently AI : `monitoring/drift_detector.py`
- Alertes si drift > seuil : `monitoring/alert.py`
- Auto-retraining : `pipelines/retrain.yml` (GitHub Actions)
- Health endpoint : `api/health.py`
- Grafana dashboard : visualisation drift en temps-reel

---

### Checklist par Hypothese

#### H1 — Repertoire Multilingue & Mobilisation
- [ ] Phase 1 : Cleaning + valeurs manquantes documentes
- [ ] Phase 2 : EDA → reports/h1/eda_report.html
- [ ] Phase 3 : Descriptive (.+ A Priori + N-grams) → reports/h1/descriptive/
- [ ] Phase 4 : Feature engineering (FREQ_MAP, target binaire h1_target)
- [ ] Phase 5 : Split 70/15/15 stratifie
- [ ] Phase 6 : Selection modele (baseline + 3 candidats)
- [ ] Phase 7 : Entrainement + MLflow log
- [ ] Phase 8 : Evaluation F1>=0.70, AUC>=0.75
- [ ] Phase 9 : Hyperparameter tuning
- [ ] Phase 10 : Validation generalisation
- [ ] Phase 11 : Packaging (apres validation these)
- [ ] Phase 12 : Monitoring (apres deployment)

#### H2 — Representations du Francais → Motivation & Difficultes
- [ ] Phase 1 : Cleaning + valeurs manquantes documentes
- [ ] Phase 2 : EDA → reports/h2/eda_report.html
- [ ] Phase 3 : Descriptive + stereotypes/preconstruits → reports/h2/descriptive/
- [ ] Phase 4 : Feature engineering (binarisation diff_*, multi-output targets)
- [ ] Phase 5 : Split 70/15/15 stratifie
- [ ] Phase 6 : Selection modele multi-output
- [ ] Phase 7 : Entrainement + MLflow log
- [ ] Phase 8 : Evaluation F1-weighted>=0.65, F1-micro>=0.72
- [ ] Phase 9 : Hyperparameter tuning
- [ ] Phase 10 : Validation generalisation
- [ ] Phase 11-12 : Packaging + Monitoring

#### H3 — Exposition Plurilingue → Attitudes envers le Francais
- [ ] Phase 1 : Cleaning + valeurs manquantes documentes
- [ ] Phase 2 : EDA → reports/h3/eda_report.html
- [ ] Phase 3 : Descriptive (distribution score attitude) → reports/h3/descriptive/
- [ ] Phase 4 : Score h3_score_attitude [1.0-5.0] construit AVANT split
- [ ] Phase 5 : Split 70/15/15 stratifie
- [ ] Phase 6 : Selection modele regression + causal (DoWhy)
- [ ] Phase 7 : Entrainement + MLflow log
- [ ] Phase 8 : Evaluation MAE<=0.50, Pearson p<0.05
- [ ] Phase 9 : Hyperparameter tuning
- [ ] Phase 10 : Validation generalisation
- [ ] Phase 11-12 : Packaging + Monitoring

#### H4 — Integration Langues Locales → Engagement & Motivation
- [ ] Phase 1 : Cleaning + valeurs manquantes documentes
- [ ] Phase 2 : EDA → reports/h4/eda_report.html
- [ ] Phase 3 : Descriptive (engagement par langue) → reports/h4/descriptive/
- [ ] Phase 4 : Score h4_engagement {1,2,3,4} construit AVANT split
- [ ] Phase 5 : Split 70/15/15 stratifie
- [ ] Phase 6 : Selection modele multi-label + ordinal (mord)
- [ ] Phase 7 : Entrainement + MLflow log
- [ ] Phase 8 : Evaluation F1>=0.70, Spearman>=0.55, Subset accuracy>=0.45
- [ ] Phase 9 : Hyperparameter tuning
- [ ] Phase 10 : Validation generalisation
- [ ] Phase 11-12 : Packaging + Monitoring

---

### ETAPE FINALE — Evaluation globale
- [ ] `python src/evaluate.py --config params.yaml`
- [ ] `mlflow ui --port 5000`
- [ ] Rapport pedagogique → `reports/pedagogical_report.md`

---

## Claude Code Commands
- /project:describe     — descriptive analysis CamemBERT + stereotypes (par hypothese)
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

## Databricks Migration

### Overview
The ML pipeline has been migrated to the **Databricks Data Intelligence Platform**.
Local Docker/FastAPI serving has been replaced with Databricks Model Serving.
The workspace is a serverless-only trial (no classic clusters) — the whole
pipeline runs on **serverless job compute** (Jobs API 2.2, `compute: {kind: SERVERLESS}`).

### Databricks Structure
```
databricks/
├── notebooks/
│   ├── 00_setup_environment.py   ← Unity Catalog + Delta raw table (from Volume CSV)
│   ├── 01_preprocess.py          ← Preprocessing on Delta Lake (4 feature tables)
│   ├── 02_train_h1.py            ← H1 training + MLflow + UC registry
│   ├── 03_train_h2.py            ← H2 training + MLflow + UC registry
│   ├── 04_train_h3.py            ← H3 training + MLflow + UC registry
│   ├── 05_train_h4.py            ← H4 training + MLflow + UC registry
│   ├── 06_evaluate.py            ← Global evaluation + pedagogical report
│   └── 07_combine_models.py      ← Bundle all 8 models into one pyfunc (flp_all)
├── workflows/
│   └── flp_pipeline_job.json     ← Databricks Workflow (8 tasks, serverless, paused schedule)
├── serving/
│   └── endpoint_config.json      ← Single serving endpoint (flp-all-models)
├── docs/
│   ├── MIGRATION_GUIDE.md
│   └── PORTFOLIO_PRESENTATION.md
├── setup_databricks.sh
└── setup_databricks.ps1
```

### Deployed State (workspace dbc-9e268203-7090.cloud.databricks.com)
- **Catalog**: `flp_catalog` (raw / processed / models / monitoring)
- **Data**: `flp_catalog.raw.data_files` volume → `raw.survey_responses` Delta (500 rows)
- **Processed**: `processed.h1_features` … `processed.h4_features` (495 rows after consent)
- **Models (8)**: `models.flp_h1_usage_quotidien`, `flp_h2_motivation`,
  `flp_h2_difficultes`, `flp_h3_attitude_reg`, `flp_h3_attitude_clf`,
  `flp_h4_motivation`, `flp_h4_engagement`, `flp_h4_discipline`,
  plus the combined serving bundle `models.flp_all`
- **Workflow job**: `562108964197707` — setup → preprocess → train_h1-h4 → evaluate → serving_bundle
- **Serving endpoint**: `flp-all-models` (READY, Small, scale-to-zero)
  - POST `/serving-endpoints/flp-all-models/invocations`
  - Input: 60 ASCII-safe feature columns + `model_name` (h1, h2a, h2b, h3r, h3c, h4a, h4b, h4c, all)
  - Output: 21 fixed columns (h1_pred/h1_proba, h2a_*, h2b_0..6, h3_reg/h3_clf, h4a_*, h4b_engagement, h4c_0..4)
  - ⚠️ Serving mangles non-ASCII column names → notebook 07 strips accents (pipelines are positional)
- **Results**: H1 ✅ (F1 0.79, AUC 0.84) · H2 ✅ (F1 0.95 / 0.80) ·
  H3 ⚠️ (MAE 0.47, F1-clf 0.45) · H4 ⚠️ (F1 0.55, ρ 1.0, subset 1.0) →
  report in `monitoring.pedagogical_report`

### Databricks Concepts Used
- **Delta Lake**: All data (raw + processed) as Delta tables
- **Unity Catalog**: `flp_catalog` with schemas (raw, processed, models, monitoring)
- **Managed MLflow**: Auto-tracked experiments + Unity Catalog Model Registry
- **Databricks Workflows**: 8-task serverless job replacing GitHub Actions
- **Model Serving**: Single real-time endpoint (trial limit: 2 endpoints / 2 concurrency)
- **Free-tier constraints**: no classic clusters (serverless only), DBFS API disabled (use UC Volumes + Files API)

### Deploy / Re-run
```powershell
# Full pipeline (preprocess → train → evaluate → rebuild serving bundle)
databricks jobs run-now --job-id 562108964197707

# After training, point the endpoint at the new flp_all version:
#   POST /api/2.0/serving-endpoints/flp-all-models/config
#   {"served_entities":[{"name":"flp-all","entity_name":"flp_catalog.models.flp_all",
#     "entity_version":"<n>","workload_size":"Small","scale_to_zero_enabled":true}],
#    "traffic_config":{"routes":[{"served_model_name":"flp-all","traffic_percentage":100}]}}
```

### Live Endpoint
- **Streamlit Dashboard**: https://student-plurilingual-representation-french-learning-htyukpvkoo.streamlit.app/
  - `deploy_cloud/databricks_backend.py` calls the live Databricks endpoint when
    `st.secrets` contains `DATABRICKS_HOST` + `DATABRICKS_TOKEN` (never in the repo);
    otherwise it falls back to local `.pkl` models. Sidebar shows the active mode.
  - Local dev: `deploy_cloud/.streamlit/secrets.toml` (gitignored, see `secrets.toml.example`).
- **Databricks Model Serving**: `flp-all-models` endpoint serving H1-H4 predictions
