# Règles de style — French-Learning-Perceptions ML

## Conventions Python
- Fonctions préfixées par hypothèse : train_h1(), build_h1(), etc.
- Type hints obligatoires sur toutes les fonctions publiques
- Docstrings en français
- Logging via `logging` uniquement (pas de print())
- Encodage fichiers : utf-8-sig (compatible Windows Excel)

## Hyperparamètres
- JAMAIS en dur dans le code → toujours dans params.yaml
- Chargement via `load_config(config_path)` de utils/config.py

## Nommage des colonnes
- Utiliser les constantes de utils/constants.py (COL_*, CSV_COLUMN_MAP)
- Jamais le nom CSV brut directement dans le code

## MLflow — obligatoire dans chaque train_h*()
```python
with mlflow.start_run(run_name=f"H{n}_{timestamp}"):
    log_params_from_config(cfg["h{n}"])  # params
    # ... entraînement ...
    log_metrics(metrics)                  # métriques
    log_thresholds_met(thresholds_met)    # seuils 0/1
    log_model_artifact(model_path)        # .pkl
    mlflow.set_tag("status", "✅ OK" | "⚠️")
```

## Git — convention de commits
- feat: nouvelle fonctionnalité
- fix: correction bug
- chore: maintenance (deps, config)
- exp: nouvelle expérience ML
- data: modification données
Exemple : `git commit -m "exp: H1 ajout SMOTE + cv_folds=5"`

## Windows — compatibilité
- Chemins : utiliser pathlib.Path (jamais os.path.join avec '/')
- Encodage CSV : utf-8-sig (BOM pour Excel)
- Séparateur : ',' (pas ';')
