# Règles de Test — French-Learning-Perceptions ML

## Commande
```powershell
pytest tests/ -v --tb=short
pytest tests/ -v --cov=src --cov-report=term-missing
```

## Règle universelle : No Data Leakage
- Scaler, Imputer, SMOTE : fit UNIQUEMENT sur X_train
- Scores composites H3/H4 : construits dans preprocess.py avant tout split
- Embeddings CamemBERT : encoder train et test séparément

## Seuils de validation par hypothèse

| Hypothèse | Métrique         | Seuil  |
|-----------|------------------|--------|
| H1        | F1-macro         | ≥ 0.70 |
| H1        | ROC-AUC          | ≥ 0.75 |
| H2        | F1-weighted (A)  | ≥ 0.65 |
| H2        | F1-micro (B)     | ≥ 0.72 |
| H3        | MAE              | ≤ 0.50 |
| H3        | F1-weighted clf  | ≥ 0.68 |
| H3        | Pearson p        | < 0.05 |
| H4        | F1 binaire       | ≥ 0.70 |
| H4        | Spearman ρ       | ≥ 0.55 |
| H4        | Subset accuracy  | ≥ 0.45 |

## Tests obligatoires (tests/test_french-learning-perceptions.py)
- test_consent_filter()         — seules les lignes "J'accepte" passent
- test_anonymize()              — Horodateur supprimé
- test_age_normalization()      — "12ans" → 12.0
- test_no_leakage_h1()         — h1_target absent des features
- test_no_leakage_h4()         — h4_target_motivation absent des features
- test_freq_map_coverage()      — FREQ_MAP couvre Toujours/Souvent/Parfois/Rarement/Jamais
- test_h1_target_binary()       — h1_target ∈ {0, 1}
- test_h3_score_range()         — h3_score_attitude ∈ [1.0, 5.0]
- test_h4_engagement_range()    — h4_engagement_score ∈ {1, 2, 3, 4}
- test_h2_diff_labels_binary()  — diff_* ∈ {0, 1}
- test_mlflow_experiment_names()— experiments définis dans params.yaml
