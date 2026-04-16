"""
mlflow_utils.py — Helpers MLflow (Level 1)
Centralise la création d'experiments et le logging des runs.
"""
import mlflow
import json
import yaml
from pathlib import Path
from typing import Any, Union


def load_params(config_path: str = "params.yaml") -> dict:
    """Charge params.yaml et retourne le dict complet."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_mlflow(tracking_uri_or_params: Union[str, dict], experiment_name: str = "") -> str:
    """Configure MLflow et retourne l'experiment_id.
    Accepte soit (tracking_uri, experiment_name) soit un dict params complet.
    """
    if isinstance(tracking_uri_or_params, dict):
        params = tracking_uri_or_params
        tracking_uri = params.get("mlflow", {}).get("tracking_uri", "mlruns")
        experiment_name = params.get("mlflow", {}).get("experiments", {}).get("preprocess", "FLP_Preprocess")
    else:
        tracking_uri = tracking_uri_or_params
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    exp = mlflow.get_experiment_by_name(experiment_name)
    return exp.experiment_id if exp else "0"


def log_params_from_config(config_section: dict) -> None:
    """Log tous les hyperparamètres d'une section params.yaml."""
    flat = _flatten(config_section)
    mlflow.log_params(flat)


def log_metrics(metrics: dict) -> None:
    """Log un dict de métriques dans MLflow."""
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(k, v)


def log_thresholds_met(thresholds_met: dict) -> None:
    """Log les seuils atteints comme métriques binaires (0/1)."""
    for k, v in thresholds_met.items():
        mlflow.log_metric(f"threshold_{k}", int(v))


def log_artifact_json(data: dict, filename: str) -> None:
    """Sauvegarde un dict en JSON et le log comme artefact MLflow."""
    import numpy as np

    class _Encoder(json.JSONEncoder):
        def default(self, obj: Any) -> Any:
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, (np.bool_,)): return bool(obj)
            if isinstance(obj, (np.ndarray,)): return obj.tolist()
            return super().default(obj)

    path = Path("reports") / filename
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, cls=_Encoder)
    mlflow.log_artifact(str(path))


def log_model_artifact(model_path: str) -> None:
    """Log un fichier modèle .pkl comme artefact MLflow."""
    mlflow.log_artifact(model_path)


def _flatten(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Aplatit un dict imbriqué pour MLflow log_params."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten(v, new_key, sep))
        else:
            items[new_key] = str(v)
    return items
