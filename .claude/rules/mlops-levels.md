# MLOps Levels — French-Learning-Perceptions ML
# Feuille de route implémentation

## ✅ Level 0 — Scripts manuels propres (IMPLÉMENTÉ)
Ce qui est en place :
- src/preprocess.py     → pipeline nettoyage manuel
- src/train.py          → entraînement par hypothèse (--hypothesis H1…H4)
- src/evaluate.py       → rapport JSON métriques
- params.yaml           → hyperparamètres externalisés
- data/raw/ (read-only) → séparation données brutes / traitées
- .gitignore            → data/raw/ et mlruns/ exclus du repo
- tests/                → validation pipeline (pytest)

## ✅ Level 1 — Pipeline automatisé + Tracking (IMPLÉMENTÉ)
Ce qui est en place :
- src/pipeline.py           → orchestration preprocess→H1→H2→H3→H4→evaluate
- src/utils/mlflow_utils.py → logging centralisé params/métriques/artefacts
- mlruns/                   → tracking local MLflow
- Nested runs MLflow         → parent pipeline + 4 enfants H1-H4
- Experiments nommés         → french-learning-perceptions_H{n}_{nom}
- Tags status ✅/⚠️          → visible dans l'UI MLflow
- Commandes Claude Code      → /project:preprocess | train-h* | pipeline | mlflow-ui

## 🔲 Level 2 — CI/CD + Model Registry + Serving (À FAIRE)
Prochaines étapes :
- .github/workflows/train.yml   → GitHub Actions (train on push to main)
- .github/workflows/test.yml    → CI tests automatiques sur PR
- MLflow Model Registry          → staging → production workflow
- api/main.py (FastAPI)         → endpoint /predict/{hypothese}
- Dockerfile                    → conteneurisation API
- docker-compose.yml            → API + MLflow server

## 🔲 Level 3 — Monitoring + Retraining automatique (À FAIRE)
Prochaines étapes :
- monitoring/drift_detector.py  → Evidently AI data drift detection
- monitoring/alert.py           → alertes si drift > seuil
- pipelines/retrain.yml         → GitHub Actions retraining automatique
- api/health.py                 → /health endpoint avec métriques live
- Grafana dashboard             → visualisation drift en temps réel

## Commande de vérification du niveau courant
```powershell
python -c "
from src.utils.config import load_config
cfg = load_config('params.yaml')
print('Level 0:', '✅')
print('Level 1:', '✅ MLflow tracking actif')
print('Level 2:', '🔲 Non encore implémenté')
print('Level 3:', '🔲 Non encore implémenté')
"
```
