# SKILL — H1 : Répertoire Multilingue & Mobilisation des Langues
# Auto-chargé quand H1 ou "mobilisation" est mentionné dans Claude Code

## Hypothèse exacte (table French-Learning-Perceptions)
Le contexte multilingue dans lequel évoluent les élèves, ainsi que la diversité
de leur répertoire linguistique, les amènent à mobiliser plusieurs langues
dans l'ensemble de leurs interactions quotidiennes.

---

## Variable Indépendante → features Python

| Ind. | Description             | Question | Variable Python       | Encodage       |
|------|-------------------------|----------|-----------------------|----------------|
| 1    | Nombre de langues       | I.1      | nb_langues            | count()        |
| 2    | Langue maternelle       | I.1      | lm_{langue}           | one-hot top-10 |
| 3    | Usage quotidien         | I.1      | h1_target             | → CIBLE 0/1   |
| 4    | Apprentissage antérieur | I.1      | apprent_anterieur_bin | binaire        |
| 5    | Relation avec LM        | I.2      | relation_lm_ord       | ordinal 0-2    |

## Variable Dépendante → features contextuelles

| Ind. | Description             | Question  | Variable Python     | Encodage       |
|------|-------------------------|-----------|---------------------|----------------|
| 1    | Domaine d'usage         | III.1     | domaine_usage_freq  | FREQ_MAP 0-4   |
| 2    | Alternance codique      | I.1+III.1 | (implicite cible)   | —              |
| 3    | Valorisation pluriling. | I.2       | valorisation_sent   | sentiment -1/+1|

---

## Tâche ML
- Type    : Classification binaire
- Cible   : h1_target (OUI=1 / NON=0)
- Seuils  : F1-macro ≥ 0.70 | ROC-AUC ≥ 0.75

## Pipeline
```
SMOTE (train only) → SimpleImputer → StandardScaler → XGBClassifier
```

## Paramètres (params.yaml → section h1)
```yaml
n_estimators: 300 | max_depth: 5 | learning_rate: 0.05
subsample: 0.8 | colsample_bytree: 0.8 | cv_folds: 5
```

## MLflow
- Experiment : flp_H1_mobilisation
- Tags loggés : f1_macro, roc_auc, threshold_f1_macro, threshold_roc_auc, shap_top_features

## Règles spécifiques H1
- SMOTE uniquement sur train set (jamais val/test)
- Stratifier split par ratio OUI/NON ET région
- SHAP top features attendues : nb_langues, domaine_usage_freq, relation_lm_ord
- Fichier : src/train.py → fonction train_h1()
