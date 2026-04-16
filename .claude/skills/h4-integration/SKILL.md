# SKILL — H4 : Intégration Langues Locales → Engagement & Motivation
# Auto-chargé quand H4 ou "intégration" est mentionné

## Hypothèse exacte (table French-Learning-Perceptions)
L'utilisation des langues locales pendant les activités d'enseignement
renforce l'intérêt pour les tâches au cours de français
et améliore l'engagement global dans le cours.

---

## Variable Indépendante → features Python

| Ind. | Description                            | Question  | Variable Python        | Encodage         |
|------|----------------------------------------|-----------|------------------------|------------------|
| 1    | Intérêt pour langues des camarades     | III.3     | interet_camarades_bin  | binaire+sentiment|
| 2    | Souhait d'intégrer les langues locales | III.4.2   | souhait_freq           | FREQ_MAP 0-4     |
| 3    | Discipline à associer aux langues      | III.4.2   | vi_disc_{discipline}   | multi-label 5    |

## Variable Dépendante → 3 cibles

| Ind. | Description                   | Question | Variable Python        | Cible |
|------|-------------------------------|----------|------------------------|-------|
| 1    | Motivation accrue             | III.2    | h4_target_motivation   | A     |
| 2    | Engagement dans l'apprentissage| III.4.2  | h4_engagement_score    | B     |

**Score engagement (B)** = f(motivation_accrue×2 + souhait_freq/4×2) → ordinal [1,4]
**Cible C** (discipline préférée) = multi-label depuis vi_disc_{discipline}

---

## Tâches ML
- Cible A : Binaire OUI/NON | F1 ≥ 0.70
- Cible B : Ordinal 1-4 | Spearman ρ ≥ 0.55
- Cible C : Multi-label 5 disciplines | Subset accuracy ≥ 0.45

## Pipeline
```
Cible A : Imputer → Scaler → VotingClassifier(XGB + LogReg soft)
Cible B : Imputer → Scaler → XGBClassifier(num_class=4, 0-indexed)
Cible C : Imputer → Scaler → MultiOutputClassifier(XGBClassifier)
```

## Disciplines (Cible C)
vocabulaire | grammaire | lecture | expression_orale | conjugaison

## Paramètres (params.yaml → section h4)
```yaml
n_estimators: 200 | max_depth: 4 | learning_rate: 0.08 | cv_folds: 5
```

## MLflow
- Experiment : flp_H4_integration
- Métriques  : f1_A, spearman_B, subset_acc_C
- Tag spécial : discipline_top1 (discipline recommandée en priorité)

## Règles spécifiques H4
- Cible B encodée 0-indexed pour XGBoost, +1 pour evaluation finale
- Classement disciplines par predict_proba moyen → recommandation pédagogique
- Fichier : src/train.py → fonction train_h4()
