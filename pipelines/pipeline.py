"""
pipeline.py — French-Learning-Perceptions ML (Level 1)
Orchestrateur complet : preprocess → train → evaluate.
Chaque étape est loggée dans MLflow avec statut, durée et métriques.
Lance : python pipelines/pipeline.py --hypothesis ALL --params params.yaml
"""

import sys
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import mlflow

# Ajouter la racine au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.preprocess import run as run_preprocess
from src.models.train_h1 import train as train_h1
from src.models.train_h2 import train as train_h2
from src.models.train_h3 import train as train_h3
from src.models.train_h4 import train as train_h4
from src.evaluation.evaluate_all import run as run_evaluate
from src.utils.mlflow_utils import load_params, setup_mlflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/pipeline.log", encoding="utf-8"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

TRAINERS = {
    "H1": train_h1,
    "H2": train_h2,
    "H3": train_h3,
    "H4": train_h4,
}

BANNER = """
╔══════════════════════════════════════════════════════╗
║         FRENCH-LEARNING-PERCEPTIONS ML — PIPELINE LEVEL 1             ║
║  preprocess → train → evaluate → MLflow              ║
╚══════════════════════════════════════════════════════╝
"""


def run_step(name: str, fn, *args, **kwargs):
    """Exécute une étape et mesure la durée."""
    log.info(f"\n── {name} {'─' * (48 - len(name))}")
    t0 = time.time()
    try:
        result = fn(*args, **kwargs)
        elapsed = round(time.time() - t0, 2)
        log.info(f"✅ {name} — {elapsed}s")
        return result, elapsed, True
    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        log.error(f"❌ {name} — ERREUR : {e}")
        return None, elapsed, False


def run_pipeline(hypothesis: str = "ALL", params_path: str = "params.yaml"):
    print(BANNER)
    params = load_params(params_path)
    setup_mlflow(params)

    targets   = list(TRAINERS.keys()) if hypothesis == "ALL" else [hypothesis]
    results   = {}
    durations = {}
    ts        = datetime.now().strftime("%Y%m%d_%H%M")

    # ── Étape 1 : Prétraitement ──────────────────────────────────────────────
    res, dur, ok = run_step(
        "PRÉTRAITEMENT",
        run_preprocess,
        params["data"]["raw_path"],
        params["data"]["processed_dir"],
        params_path,
    )
    durations["preprocess"] = dur
    if not ok:
        log.error("Arrêt : échec du prétraitement.")
        return

    # ── Étape 2 : Entraînement par hypothèse ────────────────────────────────
    for h in targets:
        res, dur, ok = run_step(
            f"TRAIN {h}",
            TRAINERS[h],
            params_path,
        )
        durations[h]  = dur
        results[h]    = res
        if ok and res:
            validated = all(res.get("thresholds_met", {}).values())
            log.info(f"  {'✅ Seuils atteints' if validated else '⚠️  Seuils non atteints'}")

    # ── Étape 3 : Évaluation globale ────────────────────────────────────────
    if hypothesis == "ALL":
        _, dur, _ = run_step("ÉVALUATION GLOBALE", run_evaluate)
        durations["evaluate"] = dur

    # ── Résumé final ────────────────────────────────────────────────────────
    total = sum(durations.values())
    log.info("\n" + "═" * 55)
    log.info("RÉSUMÉ DU PIPELINE")
    log.info("═" * 55)
    for step, dur in durations.items():
        log.info(f"  {step:20s} {dur:6.1f}s")
    log.info(f"  {'TOTAL':20s} {total:6.1f}s")
    log.info("═" * 55)

    # Log pipeline summary dans MLflow
    with mlflow.start_run(run_name=f"pipeline_{ts}"):
        mlflow.set_tag("step",       "pipeline_summary")
        mlflow.set_tag("hypotheses", hypothesis)
        mlflow.log_metric("total_duration_s", total)
        for h, res in results.items():
            if res and "metrics" in res:
                for k, v in res["metrics"].items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"{h.lower()}_{k}", v)

    # Rapport pipeline JSON
    report = {
        "pipeline_ts": ts,
        "hypotheses":  targets,
        "durations_s": durations,
        "results":     {h: r for h, r in results.items() if r},
    }
    out = Path("reports") / f"pipeline_{ts}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"\nRapport pipeline → {out}")
    log.info("MLflow UI → http://localhost:5000  (lance: make mlflow-ui)")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="French-Learning-Perceptions ML — Pipeline Level 1")
    parser.add_argument("--hypothesis", choices=["H1","H2","H3","H4","ALL"], default="ALL")
    parser.add_argument("--params",     default="params.yaml")
    args = parser.parse_args()
    run_pipeline(args.hypothesis, args.params)
