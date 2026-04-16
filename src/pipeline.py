"""
pipeline.py — French-Learning-Perceptions ML (Level 1)
Pipeline automatisé : preprocess → train H1→H4 → evaluate.
Un seul MLflow parent run qui encapsule tous les runs enfants.

Usage : python src/pipeline.py --config params.yaml
"""
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

import mlflow

sys.path.insert(0, str(Path(__file__).parent))
from utils.config import load_config
from utils.mlflow_utils import setup_mlflow, log_artifact_json
from preprocess import run as run_preprocess
from train import train_h1, train_h2, train_h3, train_h4
from evaluate import run as run_evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/pipeline.log", encoding="utf-8"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

Path("logs").mkdir(exist_ok=True)


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    mc  = cfg["mlflow"]
    ts  = datetime.now().strftime('%Y%m%d_%H%M')

    mlflow.set_tracking_uri(mc["tracking_uri"])
    mlflow.set_experiment("flp_PIPELINE")

    log.info("=" * 55)
    log.info(f"PIPELINE FRENCH-LEARNING-PERCEPTIONS ML — {ts}")
    log.info("=" * 55)

    with mlflow.start_run(run_name=f"pipeline_{ts}") as parent_run:
        mlflow.log_param("config", config_path)
        mlflow.log_param("timestamp", ts)

        # ── Step 1 : Prétraitement ───────────────────────────────────────
        log.info("\n[1/6] Prétraitement…")
        run_preprocess(config_path)
        mlflow.set_tag("step_preprocess", "done")

        # ── Step 2–5 : Entraînement H1→H4 (nested runs) ─────────────────
        all_results = {}
        for name, trainer in [("H1", train_h1), ("H2", train_h2),
                               ("H3", train_h3), ("H4", train_h4)]:
            log.info(f"\n[{['H1','H2','H3','H4'].index(name)+2}/6] Entraînement {name}…")
            with mlflow.start_run(run_name=f"{name}_{ts}", nested=True):
                result = trainer(cfg)
            all_results[name] = result
            status = "✅" if all(result.get("thresholds_met", {}).values()) else "⚠️"
            mlflow.set_tag(f"step_{name}", status)
            log.info(f"  {name} terminé {status}")

        # ── Step 6 : Évaluation globale ──────────────────────────────────
        log.info("\n[6/6] Évaluation globale…")
        report = run_evaluate(config_path, all_results)
        log_artifact_json(report, f"pipeline_report_{ts}.json")
        mlflow.set_tag("step_evaluate", "done")

        # Résumé global
        n_ok = sum(1 for r in all_results.values()
                   if all(r.get("thresholds_met", {}).values()))
        mlflow.log_metric("hypotheses_validated", n_ok)
        mlflow.set_tag("pipeline_status",
                       "✅ SUCCÈS" if n_ok == 4 else f"⚠️ {n_ok}/4 hypothèses validées")

    log.info(f"\n{'='*55}")
    log.info(f"PIPELINE TERMINÉ — {n_ok}/4 hypothèses validées")
    log.info(f"MLflow UI : mlflow ui --port 5000")
    log.info(f"{'='*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    run(args.config)
