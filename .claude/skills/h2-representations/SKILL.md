# SKILL — H2 : Représentations du Français → Motivation & Difficultés
# Auto-chargé quand H2 ou "représentations" est mentionné

## Hypothèse exacte (table French-Learning-Perceptions)
Les représentations du français influencent la motivation
et les difficultés d'apprentissage des élèves.

---

## Variable Indépendante → features Python

| Ind. | Description                                    | Question | Variable Python    | Encodage            |
|------|------------------------------------------------|----------|--------------------|---------------------|
| 1    | Perception du français (difficile/utile/belle) | II.1     | perc_{label}       | one-hot 4 classes   |
| 2    | 4 mots associés au français                    | II.1     | mots_assoc_sent    | CamemBERT 128-dim   |
| 3    | Importance du français dans le monde           | II.3     | importance_sent    | binaire + sentiment |
| 4    | Hiérarchisation vs autres langues              | II.3     | hierarchie_fr      | IMPORTANCE_MAP 1-3  |

## Variable Dépendante → 2 cibles

| Ind. | Description                  | Question | Variable Python      | Cible |
|------|------------------------------|----------|----------------------|-------|
| 1    | Motivation à apprendre       | II.2     | h2_target_motivation | A     |
| 2    | Difficultés perçues          | II.2     | diff_{domaine}       | B     |
| 3    | Nature des difficultés       | III.4    | diff_{domaine}       | B     |
| 4    | Origine des difficultés      | III.4    | diff_{domaine}       | B     |

---

## Tâches ML
- Cible A : Classification 3 classes (0=Faible, 1=Moyen, 2=Élevé) | F1-weighted ≥ 0.65
- Cible B : Multi-label 6 labels (grammaire/vocabulaire/orthographe/conjugaison/expression_orale/analyse) | F1-micro ≥ 0.72

## Pipeline
```
Cible A : SimpleImputer → StandardScaler → XGBClassifier (multi:softprob, num_class=3)
Cible B : SimpleImputer → StandardScaler → MultiOutputClassifier(XGBClassifier)
```

## Paramètres (params.yaml → section h2)
```yaml
n_estimators: 200 | max_depth: 4 | learning_rate: 0.08 | cv_folds: 5
```

## MLflow
- Experiment : flp_H2_representations
- Métriques  : f1_weighted_A, f1_micro_B, threshold_f1_weighted_A, threshold_f1_micro_B

## Règles spécifiques H2
- Les labels diff_* sont construits via extraction mots-clés sur III.4 + II.2
- En production : remplacer mots_assoc_sent par embedding CamemBERT réel
- Fichier : src/train.py → fonction train_h2()
