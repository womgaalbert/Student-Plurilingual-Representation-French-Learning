# SKILL — H3 : Exposition Plurilingue → Attitudes envers le Français
# Auto-chargé quand H3 ou "exposition" est mentionné

## Hypothèse exacte (table French-Learning-Perceptions)
L'exposition au plurilinguisme influence positivement
les attitudes envers le français.

---

## Variable Indépendante → features Python

| Ind. | Description                            | Question | Variable Python         | Encodage         |
|------|----------------------------------------|----------|-------------------------|------------------|
| 1    | Exposition en classe ou hors classe    | III.1    | exposition_freq         | FREQ_MAP 0-4     |
| 2    | Intérêt pour apprendre autres langues  | III.1    | interet_bin+interet_sent| binaire+sentiment|
| 3    | Attitude envers le plurilinguisme      | III.2    | perception_multi_ord    | LIKERT_MAP+sent. |

## Variable Dépendante → score composite

| Ind. | Description                          | Question | Composante score      | Poids |
|------|--------------------------------------|----------|-----------------------|-------|
| 1    | Perception français en contexte multi | II.3    | score_att_comp1       | ×3    |
| 2    | Motivation plurilinguisme scolaire   | III.2    | score_att_comp2       | ×2    |

**Score composite** = f(comp1/3 × 3 + comp2/2 × 2) normalisé [1, 5]
**Classe** : Positive (≥3.5) | Neutre (≥2.5) | Négative (<2.5)

---

## Tâches ML
- Régression   : score h3_score_attitude [1-5] | MAE ≤ 0.50
- Classification: h3_attitude_class (3 classes) | F1-weighted ≥ 0.68
- Causal        : Pearson r (exposition_freq → score) | p < 0.05
- ATE           : effet exposition haute (≥3) vs basse (<3) sur attitude

## Pipeline
```
Régression    : SimpleImputer → StandardScaler → RandomForestRegressor
Classification: SimpleImputer → StandardScaler → XGBClassifier (multi:softprob)
Causal        : pearsonr() + ATE groupby exposition_freq_bin
```

## Paramètres (params.yaml → section h3)
```yaml
n_estimators_reg: 300 | n_estimators_clf: 200 | max_depth: 6 | cv_folds: 5
causal: true
```

## MLflow
- Experiment : flp_H3_exposition
- Métriques  : mae, f1_weighted_clf, pearson_r, pearson_p, ate

## Règles spécifiques H3
- Score composite construit dans preprocess.py (pas dans train.py)
- ATE = mean(score | expo≥3) - mean(score | expo<3)
- DoWhy optionnel pour analyse causale formelle (Level 2)
- Fichier : src/train.py → fonction train_h3()
