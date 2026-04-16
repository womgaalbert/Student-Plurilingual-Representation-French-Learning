"""
evaluate.py — French-Learning-Perceptions ML
Rapport d'évaluation global + recommandations pédagogiques.
Usage : python src/evaluate.py --config params.yaml
"""
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent))
from utils.config import load_config

log = logging.getLogger(__name__)
Path("reports").mkdir(exist_ok=True)

SEUILS = {
    "H1": {"f1_macro": 0.70, "roc_auc": 0.75},
    "H2": {"f1_weighted_A": 0.65, "f1_micro_B": 0.72},
    "H3": {"mae": 0.50, "f1_weighted_clf": 0.68},
    "H4": {"f1_A": 0.70, "spearman_B": 0.55},
}

RECOMMANDATIONS = {
    "H1": ("Profils à répertoire large (≥3 langues) + exposition fréquente (III.1) "
           "mobilisent davantage leurs langues. → Encourager les pratiques translangagières."),
    "H2": ("Les élèves percevant le français comme 'difficile' ont une motivation plus faible. "
           "Grammaire & conjugaison sont les difficultés dominantes. → Activités de remédiation ciblées."),
    "H3": ("Une exposition plus fréquente aux autres langues corrèle positivement avec "
           "les attitudes envers le français. → Intégrer des moments d'éveil aux langues en classe."),
    "H4": ("La majorité des élèves seraient plus motivés par l'intégration des langues locales. "
           "→ Piloter des activités comparatives en vocabulaire en priorité."),
}


def load_latest(hypothese: str) -> Dict:
    pattern = f"h{hypothese[-1].lower()}_run_*.json"
    files   = sorted(Path("reports").glob(pattern), reverse=True)
    if not files: return {}
    with open(files[0], encoding="utf-8") as f:
        return json.load(f)


def run(config_path: str = "params.yaml", results: Dict = None) -> Dict:
    if results is None:
        results = {h: load_latest(h) for h in ["H1","H2","H3","H4"]}

    report = {
        "titre":      "Rapport d'évaluation — French-Learning-Perceptions ML",
        "date":       datetime.now().strftime("%d/%m/%Y %H:%M"),
        "n_repondants": 500,
        "hypotheses": {},
        "synthese":   {},
    }

    log.info("\n┌─────┬────────────────────────────────────────────┬────────┐")
    log.info("│ HYP │ MÉTRIQUES                                   │ STATUT │")
    log.info("├─────┼────────────────────────────────────────────┼────────┤")

    for h, res in results.items():
        met     = res.get("metrics", {})
        thr_met = res.get("thresholds_met", {})
        ok      = all(thr_met.values()) if thr_met else False
        status  = "✅ OK" if ok else "⚠️ KO"

        vals = " | ".join(f"{k}={v}" for k, v in list(met.items())[:2])
        log.info(f"│ {h}  │ {vals[:44]:44s}│ {status:6s} │")

        report["hypotheses"][h] = {
            "metrics":        met,
            "thresholds_met": thr_met,
            "validated":      ok,
            "recommandation": RECOMMANDATIONS.get(h, ""),
        }

    log.info("└─────┴────────────────────────────────────────────┴────────┘")

    n_ok = sum(1 for v in report["hypotheses"].values() if v["validated"])
    report["synthese"] = {
        "hypotheses_validees": f"{n_ok}/4",
        "statut": "✅ Projet ML opérationnel" if n_ok >= 3 else "⚠️ Affiner les modèles",
        "prochaines_etapes": [
            "Level 2 : Ajouter CI/CD GitHub Actions (train on push)",
            "Level 2 : Model Registry MLflow (staging → production)",
            "Level 2 : Endpoint FastAPI /predict pour chaque hypothèse",
            "Level 3 : Détection data drift (Evidently AI)",
            "Level 3 : Retraining automatique sur nouveau batch",
        ],
    }

    ts   = datetime.now().strftime('%Y%m%d_%H%M')
    path = Path("reports") / f"rapport_evaluation_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log.info(f"\nRapport → {path}")
    log.info(f"Hypothèses validées : {n_ok}/4 | {report['synthese']['statut']}")
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    run(args.config)
