"""
scripts/register_models.py — Active le Model Registry MLflow
Enregistre les meilleurs modèles de chaque hypothèse avec métadonnées.

Prérequis : pipeline d'entraînement déjà exécuté (modèles dans models/)
Usage      : python scripts/register_models.py
"""
import pickle
import sys
from pathlib import Path
from datetime import datetime

import mlflow
import yaml
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils.config import load_config


def _log_and_register(model_path: Path, model_name: str, params: dict,
                      metrics: dict, tags: dict, client: MlflowClient):
    """Charge un .pkl, le loggue dans un run MLflow et l'enregistre au Registry."""
    if not model_path.exists():
        print(f"[WARN] Modele introuvable : {model_path}")
        return

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with mlflow.start_run(run_name=f"reg_{model_name}_{datetime.now():%Y%m%d_%H%M}"):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        for k, v in tags.items():
            mlflow.set_tag(k, v)

        # Déterminer la flavor MLflow appropriée
        try:
            # sklearn flavor (Pipeline, VotingRegressor, etc.)
            mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)
        except Exception:
            # PyFunc fallback
            mlflow.pyfunc.log_model("model", python_model=model, registered_model_name=model_name)

    # Tag staging
    versions = client.search_model_versions(f"name='{model_name}'")
    if versions:
        latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
        client.set_model_version_tag(model_name, latest.version, "registered_by", "register_models.py")
        print(f"[OK] {model_name} v{latest.version} -> {latest.current_stage}")
    else:
        print(f"[OK] {model_name} -> registered")


def register_models():
    cfg = load_config("params.yaml")
    mc  = cfg["mlflow"]
    model_map = mc["models"]

    mlflow.set_tracking_uri(mc["tracking_uri"])
    mlflow.set_registry_uri(mc["registry_uri"])
    client = MlflowClient()

    # ── H1 : XGBoost — classification binaire ─────────────────────────────
    h1 = sorted(Path("models/h1").glob("*.pkl"),
                key=lambda p: p.stat().st_mtime, reverse=True)
    if h1:
        _log_and_register(h1[0], model_map["h1"],
                          params={"n_estimators": 300, "max_depth": 5},
                          metrics={"f1_macro": 0.8347, "roc_auc": 0.8506, "cv_f1_macro_mean": 0.8072},
                          tags={"hypothese": "H1", "modele": "XGBoost", "status": "✅ VALIDÉE"}, client=client)

    # ── H2 : Tuned ClassifierChain ────────────────────────────────────────
    h2 = sorted(Path("models/h2").glob("*tuned*.pkl"),
                key=lambda p: p.stat().st_mtime, reverse=True)
    if h2:
        _log_and_register(h2[0], model_map["h2"],
                          params={"modele": "ClassifierChain+XGBoost", "tuned": True},
                          metrics={"f1_weighted_A": 0.9535, "f1_micro_B": 0.7452, "val_f1_A": 0.98},
                          tags={"hypothese": "H2", "status": "✅ VALIDÉE"}, client=client)

    # ── H3 : VotingRegressor (reg) + XGBoost (clf) ────────────────────────
    h3_reg = sorted(Path("models/h3").glob("*reg_tuned*.pkl"),
                    key=lambda p: p.stat().st_mtime, reverse=True)
    h3_clf = sorted(Path("models/h3").glob("*clf_tuned*.pkl"),
                    key=lambda p: p.stat().st_mtime, reverse=True)
    if h3_reg:
        _log_and_register(h3_reg[0], f"{model_map['h3']}_reg",
                          params={"modele": "VotingRegressor(ET+XGB)", "tuned": True},
                          metrics={"mae": 0.5126, "f1_weighted_clf": 0.7797},
                          tags={"hypothese": "H3", "tache": "regression", "status": "⚠️ MAE=0.513/0.50"}, client=client)
    if h3_clf:
        _log_and_register(h3_clf[0], f"{model_map['h3']}_clf",
                          params={"modele": "XGBoostClassifier", "tuned": True},
                          metrics={"f1_weighted": 0.7797},
                          tags={"hypothese": "H3", "tache": "classification", "status": "✅ F1=0.780"}, client=client)

    # ── H4 : Tuned (A: motivation, B: engagement, C: disciplines) ─────────
    for suffix, task, m_name, m_val in [
        ("A_motivation_tuned", "binaire", "f1_A", 0.8065),
        ("B_engagement_tuned", "ordinal", "spearman_B", 0.5608),
        ("C_discipline_tuned", "multi-label", "subset_acc_C", 1.0),
    ]:
        h4_m = sorted(Path("models/h4").glob(f"*{suffix}*.pkl"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if h4_m:
            _log_and_register(h4_m[0], f"{model_map['h4']}_{suffix[0].lower()}",
                              params={"modele": "XGBoost", "tuned": True, "tache": task},
                              metrics={m_name: m_val},
                              tags={"hypothese": "H4", "tache": task, "status": "✅ VALIDÉE"}, client=client)

    # ── Résumé ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL REGISTRY — Modèles enregistrés")
    for rm in client.search_registered_models():
        versions = client.search_model_versions(f"name='{rm.name}'")
        v_latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0] if versions else None
        if v_latest:
            print(f"  - {rm.name}  v{v_latest.version}  [{v_latest.current_stage}]")
    print(f"\nTotal : {len(client.search_registered_models())} modèles enregistrés")
    print("=" * 60)


if __name__ == "__main__":
    # Créer l'expérience si elle n'existe pas
    try:
        cfg = load_config("params.yaml")
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        mlflow.set_experiment("FLP_Model_Registry")
    except Exception:
        pass
    register_models()
