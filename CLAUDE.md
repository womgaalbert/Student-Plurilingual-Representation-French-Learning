# French-Learning-Perceptions-in-Plurilingual-Cameroon

## Research
Thesis: "Les représentations des élèves du premier cycle de l'enseignement
secondaire camerounais sur l'apprentissage du français dans leur contexte plurilingue"

Researcher : Chancelline Armelle Nongni Kendjio
            Universite Marie et Louis Pasteur de Besancon (France)
            https://www.linkedin.com/in/chancelline-armelle-nongni-kendjio-9840582b/

            
Technical  : Albert Womga — AI/ML Engineering support
            https://www.linkedin.com/in/albert-womga-009a7931/

## GitHub Repository
https://github.com/womgaalbert/Student-Plurilingual-Representation-French-Learning

## Environment
- OS        : Windows 11 (PowerShell / VS Code terminal)
- Python    : 3.13 (.venv)
- Git       : GitHub — main / dev / feature/* / exp/*
- Tracking  : MLflow local (mlruns/) — ⚠️ migrer vers sqlite:///mlflow.db
- MLOps     : Level 0 ✅ | Level 1 ✅ | Level 2 ✅ (80%) | Level 3 🔲
- Tests     : 42/42 ✅ (pytest)
- Audit     : 47% global → cible 70% après P0+P1 (reports/audit_mlops_mai2026.docx)

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
- [x] Copier `data_FLP.csv` dans `data/raw/` (READ-ONLY ensuite)
- [x] Executer `scripts/setup_windows.ps1` (venv + deps + dossiers)
- [x] Filtrage consentement : garder uniquement "J'accepte"
- [x] Anonymisation : suppression Horodateur + identifiants
```powershell
python src/preprocess.py --config params.yaml
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
- Embeddings CamemBERT comme features numeriques (dim 768 → reduction PCA)
  → Implémenté dans `src/utils/embeddings.py` (build_camembert_features)
  → PCA 768 → 20 composantes (params.yaml: camembert_n_components)
  → Branché dans preprocess.py pour H3 (84.8% variance) et H4 (88.6% variance)
  → Colonnes `emb_pca_0..19` intégrées dans feat_cols de train_h3/h4 et tune_h3/h4
- Scores composites (H3: h3_score_attitude, H4: h4_engagement_score) — AVANT split
- Normalisation / standardisation des features numeriques
- Toutes les colonnes via constantes COL_* de utils/constants.py

#### Phase 5 — Data Splitting (Train / Validation / Test)
- Split stratifie sur la target : 70% train / 15% val / 15% test
- Seed fixe dans params.yaml (reproductibilite)
- Verification distribution target identique dans les 3 splits
- ROS (RandomOverSampler) applique uniquement sur X_train (jamais val/test)
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
- Outil : GridSearchCV (defini dans params.yaml)
- Espace de recherche defini dans params.yaml uniquement
- Optimisation sur validation set (pas test)
- Meilleur modele re-entraine sur train+val avant evaluation finale
- Nouveau MLflow run avec tag "tuned"
- ⚠️ H1: tuning non implemente (performances deja au-dessus des seuils)

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
- [x] Phase 1 : Cleaning + valeurs manquantes documentes
- [x] Phase 2 : EDA → reports/h1/eda_report.html
- [x] Phase 3 : Descriptive (CamemBERT + A Priori + N-grams) → reports/h1/descriptive/
- [x] Phase 4 : Feature engineering (FREQ_MAP, target binaire h1_target)
- [x] Phase 5 : Split 70/15/15 stratifie
- [x] Phase 6 : Selection modele (XGBoost retenu, F1=0.835, AUC=0.851)
- [x] Phase 7 : Entrainement + MLflow log
- [x] Phase 8 : Evaluation — F1-macro=0.835 ✅ | ROC-AUC=0.851 ✅ | CV=0.807±0.039
- [ ] Phase 9 : Hyperparameter tuning (non implemente — perf. deja OK)
- [ ] Phase 10 : Validation generalisation — rapport manquant
- [ ] Phase 11 : Packaging (apres validation these)
- [ ] Phase 12 : Monitoring (apres deployment)

#### H2 — Representations du Francais → Motivation & Difficultes
- [x] Phase 1 : Cleaning + valeurs manquantes documentes
- [x] Phase 2 : EDA → reports/h2/eda_report.html
- [x] Phase 3 : Descriptive + stereotypes/preconstruits → reports/h2/descriptive/
- [x] Phase 4 : Feature engineering (binarisation diff_*, perception labels, facile_*)
- [x] Phase 5 : Split 70/15/15 stratifie
- [x] Phase 6 : Selection modele — ClassifierChain(XGBoost) pour Target B
- [x] Phase 7 : Entrainement + MLflow log
- [x] Phase 8 : Evaluation — F1-weighted=0.954 ✅ | F1-micro=0.745 ✅
- [x] Phase 9 : Hyperparameter tuning — GridSearchCV (H2 Target A & B)
- [ ] Phase 10 : Validation generalisation — rapport manquant
- [ ] Phase 11-12 : Packaging + Monitoring

#### H3 — Exposition Plurilingue → Attitudes envers le Francais
- [x] Phase 1 : Cleaning + valeurs manquantes documentes
- [x] Phase 2 : EDA → reports/h3/eda_report.html
- [x] Phase 3 : Descriptive (score attitude, UMAP, CamemBERT) → reports/h3/descriptive/
- [x] Phase 4 : Score h3_score_attitude [1.0-5.0] + exposition_bin (corrige) + CamemBERT PCA 20D
- [x] Phase 5 : Split 70/15/15 stratifie
- [x] Phase 6 : Selection modele — 5 modeles testes (RF → LGBM → XGBoost → ExtraTrees → VotingRegressor ET+XGB)
- [x] Phase 7 : Entrainement + MLflow log
- [x] Phase 8 : Evaluation — MAE=0.513 ❌ (seuil 0.50) | F1=0.780 ✅ | Pearson p=0.984 ❌
- [x] Phase 9 : Hyperparameter tuning — GridSearchCV ExtraTrees + blend VotingRegressor
- [ ] Phase 10 : Validation generalisation — rapport manquant
- [ ] Phase 11-12 : Packaging + Monitoring

⚠️ **H3 non validee** (1/3 seuils) : MAE=0.513 proche du seuil (0.50).
   5 modeles testes (RF, LGBM, XGBoost, ExtraTrees, VotingRegressor) — MAE plafonne ~0.51.
   Erreur irréductible du proxy composite h3_score_attitude (échelle 1-5, ~10% d'erreur).
   CamemBERT PCA 20D (84.8% variance) : MAE 0.531→0.513 (-3.4%), F1 0.660→0.780 (+18%).
   93.5% des eleves ont exposition_bin=1 → test causal structurellement impossible.
   Modèle final : VotingRegressor(ExtraTrees + XGBoost) via pipe().

#### H4 — Integration Langues Locales → Engagement & Motivation
- [x] Phase 1 : Cleaning + valeurs manquantes documentes
- [x] Phase 2 : EDA → reports/h4/eda_report.html
- [x] Phase 3 : Descriptive (engagement par discipline) → reports/h4/descriptive/
- [x] Phase 4 : Score h4_engagement {1,2,3,4} + interet_camarades_ord (corrige) + CamemBERT PCA 20D
- [x] Phase 5 : Split 70/15/15 stratifie
- [x] Phase 6 : Selection modele — XGBoost multi-tache + ClassifierChain
- [x] Phase 7 : Entrainement + MLflow log
- [x] Phase 8 : Evaluation — F1=0.807 ✅ | Spearman rho=0.561 ✅ | Subset=1.0 ✅
- [x] Phase 9 : Hyperparameter tuning — GridSearchCV (Targets A, B, C)
- [ ] Phase 10 : Validation generalisation — rapport manquant
- [ ] Phase 11-12 : Packaging + Monitoring

✅ **H4 VALIDEE** (3/3 sous-objectifs OK) : Spearman rho=0.561 >= seuil 0.55
   CamemBERT PCA 20D (variance expliquee 88.6%) a contribue de maniere decisive
   (+0.085 sur le rho par rapport a 0.476 sans embeddings)

---

### ETAPE FINALE — Evaluation globale
- [x] `python src/evaluate.py --config params.yaml` → `reports/rapport_evaluation_*.json`
- [x] `mlflow ui --port 5000`
- [ ] Rapport pedagogique → `reports/pedagogical_report.md`

**Dernier rapport d'evaluation** : `reports/rapport_evaluation_20260610_1312.json`
- H1 : ✅ VALIDEE — F1=0.835, AUC=0.851, CV=0.807±0.039
- H2 : ✅ VALIDEE — F1-weighted=0.954, F1-micro=0.745, Val-F1=0.98
- H3 : ❌ NON VALIDEE — MAE=0.513 (seuil 0.50) | F1=0.780 ✅ | p=0.984
        5 modeles testes, MAE plafonne a ~0.51 (erreur irreductible du proxy composite)
- H4 : ✅ VALIDEE — F1=0.807, rho=0.561, Subset=1.0 (CamemBERT PCA 20D: +0.085 rho)
- Synthese : **3/4 hypotheses validees** 🎉
- Statut : "✅ Projet ML operationnel"

---

## BUGFIXES Majeurs (a conserver en memoire)

### 1. SMOTE → RandomOverSampler
SMOTE exige >=6 echantillons minoritaires par fold CV → erreur `n_neighbors > n_samples_fit`
→ Remplace par `RandomOverSampler(random_state=42)` dans `_imb_pipe()` (train.py:388)

### 2. exposition_bin (H3)
FREQ_MAP ne correspondait pas aux reponses Oui/Non → tout etait 0
→ Corrige : encodage binaire Oui→1, Non→0 avec tolerance "Oio" (preprocess.py:185)

### 3. h4_target_motivation (H4)
`extract_oui_non()` ne reconnaissait pas "Bien"/"Tres bien" → tout -1 → tout 0
→ Corrige : INTERET_MAP avec seuil >=2 → 1 (preprocess.py:218)

### 4. h3_score_attitude (H3)
`startswith("OUI")` sur "Bien/Tres bien" → s2 toujours 0.5 → 3 valeurs seulement
→ Corrige : dictionnaire _S2 explicite (Très bien→2.0, Bien→1.5, Un peu→1.0, Pas du tout→0.5)

### 5. ClassifierChain predict_proba shape
MultiOutputClassifier → liste de matrices (n_samples, 2)
ClassifierChain → matrice unique (n_samples, n_labels)
→ Code de ranking adapte : `chain_p.mean(axis=0)` au lieu de `[p[:,1].mean() for p in probas]`

### 6. INTERET_MAP pour echelles Likert
Echelle "Très bien"/"Bien"/"Un peu"/"Pas du tout" n'avait pas de mapping
→ Ajout dans constants.py:100-105 — valeurs 3/2/1/0

### 7. comprehension manquant dans DIFFICULTE_KEYWORDS
"Compréhension de texte" etait la difficulte la plus frequente mais absente des keywords
→ Ajout de la cle "comprehension" (constants.py:120)

---

## Project Files Overview

### Source Code
| Fichier | Role |
|---------|------|
| `src/preprocess.py`              | Nettoyage, encodage, construction features + cibles |
| `src/descriptive_analysis.py`    | Phase 3 : CamemBERT, A Priori, stereotypes, N-grammes, UMAP |
| `src/train.py`                   | Entrainement + tuning H1-H4, MLflow tracking |
| `src/evaluate.py`                | Evaluation globale, rapport JSON |
| `src/pipeline.py`                | Orchestration 10 etapes |
| `src/utils/config.py`            | Chargement params.yaml |
| `src/utils/constants.py`         | Mappings CSV, encodages ordinaux, keywords |
| `src/utils/embeddings.py`        | 🆕 CamemBERT → PCA features (build_camembert_features) |
| `src/utils/mlflow_utils.py`      | Helpers MLflow (setup, log_params, log_metrics) |

### Scripts
| Fichier | Role |
|---------|------|
| `scripts/setup_windows.ps1`           | Setup venv + dependances |
| `scripts/generate_word_report.py`     | Rapport Word synthetique |
| `scripts/generate_word_report_full.py`| Rapport Word complet illustre (toutes figures) |
| `scripts/generate_mlops_audit.py`     | 🆕 Audit MLOps 10 dimensions (score 47%) |
| `scripts/generate_roadmap_chart.py`   | Diagramme roadmap |
| `scripts/diag_columns.py`             | Diagnostic colonnes CSV |

### Rapports générés
| Fichier | Contenu |
|---------|---------|
| `reports/rapport_evaluation_20260610_1312.json` | Dernières métriques (3/4 validées) |
| `reports/rapport_complet_FLP_mai2026.docx` | Rapport illustré complet (toutes figures) |
| `reports/audit_mlops_mai2026.docx` | 🆕 Audit MLOps 10 dimensions |

### CI/CD
| Fichier | Role |
|---------|------|
| `.github/workflows/ci.yml`    | CI pipeline (pytest + validation params.yaml) |
| `.github/workflows/tests.yml` | Tests automatisés (ubuntu + windows) |

---

## Known Gaps & Pending Work

### Urgent — Amelioration Modeles
- [x] Brancher `src/utils/embeddings.py` dans `preprocess.py` → colonnes `emb_pca_*`
- [x] Ajouter `emb_pca_*` aux `feat_cols` de `train_h3`, `tune_h3`, `train_h4`, `tune_h4`
- [x] Relancer pipeline complet → H4 ✅ (rho 0.476→0.561), H3 ⚠️ (MAE 0.531→0.513)
- [x] 5 modeles testes pour H3 : RF, LGBM, XGBoost, ExtraTrees, VotingRegressor ET+XGB
- [x] Tests : 42/42 ✅ (3 corrections : perc_difficil, perc_importan, exposition_bin)
- [x] Audit MLOps : `reports/audit_mlops_mai2026.docx` (score global 47%)

### Important — Documentation
- [ ] Rapport pedagogique `reports/pedagogical_report.md`
- [ ] Rapport de generalisation H1 `reports/h1/validation_report.md`
- [ ] Rapport de generalisation H2 `reports/h2/validation_report.md`
- [ ] Analyse SHAP pour H1 et H2

### Level 2 MLOps (P0 — cf. audit)
- [x] FastAPI : `api/main.py` → 8 endpoints /predict/h1..h4 (/health, /models)
- [x] Dockerfile + docker-compose.yml (API + MLflow server)
- [x] GitHub Actions train workflow (`.github/workflows/train.yml` — push + cron + manual)
- [x] MLflow Model Registry : 7 modèles (H1 v2, H2 v1, H3_reg v1, H3_clf v1, H4_a/b/c v1)
- [x] Migrer backend MLflow : fichiers → `sqlite:///mlflow.db`
- [ ] Promouvoir modèles staging → production (gates conditionnelles)
- [ ] Déploiement cloud (Vercel / AWS / GCP)

### Level 3 MLOps (P1 — cf. audit)
- [ ] Evidently AI drift monitoring
- [ ] Health endpoint + dashboard Grafana
- [ ] Auto-retraining pipeline

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
