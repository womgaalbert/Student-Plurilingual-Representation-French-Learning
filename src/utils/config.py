"""
config.py — Chargement de params.yaml
Utilisé par tous les scripts pour éviter les hyperparamètres en dur.
"""
import yaml
from pathlib import Path


def load_config(config_path: str = "params.yaml") -> dict:
    """Charge params.yaml et retourne un dict."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"params.yaml introuvable : {path.absolute()}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_section(config: dict, section: str) -> dict:
    """Retourne une section du config (data, h1, h2…)."""
    if section not in config:
        raise KeyError(f"Section '{section}' absente de params.yaml")
    return config[section]
