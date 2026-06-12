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
from descriptive_analysis import run_full as run_descriptive
from train import train_h1, train_h2, train_h3, train_h4, tune_h2, tune_h3, tune_h4
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
        log.info("\n[1/7] Prétraitement…")
        run_preprocess(config_path)
        mlflow.set_tag("step_preprocess", "done")

        # ── Step 2 : Analyse descriptive (demographics + H1-H4 phases 1-3)
        log.info("\n[2/7] Analyse descriptive (demographics + CamemBERT H1-H4)…")
        desc_metrics = run_descriptive(config_path)
        mlflow.set_tag("step_descriptive", "done")
        for hyp, m in desc_metrics.items():
            mlflow.log_metrics({f"desc_{hyp}_{k}": v for k, v in m.items()
                                 if isinstance(v, (int, float))})

        # ── Step 3 : Entraînement H1 (nested run) ────────────────────────
        log.info("\n[3/10] Entraînement H1…")
        result_h1 = train_h1(cfg)
        mlflow.set_tag("step_H1", "✅" if all(result_h1.get("thresholds_met", {}).values()) else "⚠️")

        # ── Step 4 : Entraînement H2 + Phase 9 tuning ────────────────────
        log.info("\n[4/10] Entraînement H2…")
        train_h2(cfg)
        log.info("\n[5/10] Tuning H2 (Phase 9)…")
        result_h2 = tune_h2(cfg)
        mlflow.set_tag("step_H2", "✅" if all(result_h2.get("thresholds_met", {}).values()) else "⚠️")

        # ── Step 6 : Entraînement H3 + Phase 9 tuning ────────────────────
        log.info("\n[6/10] Entraînement H3…")
        train_h3(cfg)
        log.info("\n[7/10] Tuning H3 (Phase 9)…")
        result_h3 = tune_h3(cfg)
        mlflow.set_tag("step_H3", "✅" if all(result_h3.get("thresholds_met", {}).values()) else "⚠️")

        # ── Step 8 : Entraînement H4 + Phase 9 tuning ────────────────────
        log.info("\n[8/10] Entraînement H4…")
        train_h4(cfg)
        log.info("\n[9/10] Tuning H4 (Phase 9)…")
        result_h4 = tune_h4(cfg)
        mlflow.set_tag("step_H4", "✅" if all(result_h4.get("thresholds_met", {}).values()) else "⚠️")

        all_results = {
            "H1": result_h1,
            "H2": result_h2,
            "H3": result_h3,
            "H4": result_h4,
        }

        # ── Step 10 : Évaluation globale ─────────────────────────────────
        log.info("\n[10/10] Évaluation globale…")
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
