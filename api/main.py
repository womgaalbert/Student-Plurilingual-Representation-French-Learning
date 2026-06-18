"""
api/main.py — FastAPI serving pour les 4 hypothèses
French-Learning-Perceptions ML (Level 2 MLOps)

Usage:
  uvicorn api.main:app --host 0.0.0.0 --port <PORT>   (défaut: params.yaml → api.port)
  API_PORT=9000 python api/main.py
  docker compose up   (API_PORT=9000 docker compose up)

Endpoints:
  GET  /health                        — health check
  GET  /models                        — liste des modèles servis
  POST /predict/h1                    — usage quotidien des langues
  POST /predict/h2/motivation         — motivation à apprendre
  POST /predict/h2/difficultes        — types de difficultés
  POST /predict/h3/attitude           — score d'attitude
  POST /predict/h3/classification     — classe d'attitude (Positive/Neutre/Négative)
  POST /predict/h4/motivation         — motivation langues locales
  POST /predict/h4/engagement         — score d'engagement
  POST /predict/h4/disciplines        — disciplines préférées
"""
import os
import sys
import time
import logging
from pathlib import Path
import pickle
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

# ── Prometheus (MLOps Level 3) ──────────────────────────────────────────
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY, CONTENT_TYPE_LATEST
    _PROMETHEUS = True
except ImportError:
    _PROMETHEUS = False

# ── Setup ────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils.config import load_config

cfg = load_config("params.yaml")

MODEL_DIR = Path("models")
MODELS: dict = {}

app = FastAPI(
    title="French-Learning-Perceptions ML API",
    description="API de prédiction pour les 4 hypothèses de la thèse de Chancelline Armelle Nongni Kendjio",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Prometheus metrics (MLOps Level 3) ──────────────────────────────────
if _PROMETHEUS:
    PREDICTION_COUNTER = Counter(
        "flp_predictions_total", "Total predictions per model",
        ["model_key"],
    )
    PREDICTION_LATENCY = Histogram(
        "flp_prediction_latency_seconds", "Prediction latency in seconds",
        ["model_key"],
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )
    DRIFT_GAUGE = Gauge(
        "flp_drift_ratio", "Data drift ratio per model (0=no drift, 1=full drift)",
        ["model_key"],
    )
else:
    PREDICTION_COUNTER = None
    PREDICTION_LATENCY = None
    DRIFT_GAUGE = None


# ── Pydantic schemas ─────────────────────────────────────────────────────────

class H1Input(BaseModel):
    nb_langues:             int   = Field(..., ge=1, le=10, description="Nombre de langues parlées")
    apprent_anterieur_bin:  int   = Field(..., ge=0, le=1)
    relation_lm_ord:        int   = Field(..., ge=0, le=2)
    domaine_usage_freq:     int   = Field(..., ge=0, le=4)
    valorisation_sent:      float = Field(..., ge=-1.0, le=1.0)
    sexe_bin:               int   = Field(..., ge=0, le=1)
    age:                    float = Field(..., ge=5, le=100)

class H2MotivationInput(BaseModel):
    perc_utile:      int = Field(0, ge=0, le=1)
    perc_belle:      int = Field(0, ge=0, le=1)
    perc_difficil:   int = Field(0, ge=0, le=1)
    perc_importan:   int = Field(0, ge=0, le=1)
    mots_assoc_sent: float = Field(0.0, ge=-1.0, le=1.0)
    importance_bin:  int = Field(0, ge=0, le=1)
    importance_sent: float = Field(0.0, ge=-1.0, le=1.0)
    hierarchie_fr:   float = Field(2.0, ge=1.0, le=3.0)
    sexe_bin:        int = Field(0, ge=0, le=1)
    age:             float = Field(12.0, ge=5, le=100)

class H3AttitudeInput(BaseModel):
    exposition_bin:        int   = Field(1, ge=0, le=1)
    interet_bin:           int   = Field(1, ge=0, le=1)
    interet_sent:          float = Field(0.0, ge=-1.0, le=1.0)
    perception_multi_sent: float = Field(0.0, ge=-1.0, le=1.0)
    perception_multi_ord:  float = Field(2.5, ge=0.0, le=4.0)
    nb_langues:            int   = Field(2, ge=1, le=10)
    sexe_bin:              int   = Field(0, ge=0, le=1)
    age:                   float = Field(12.0, ge=5, le=100)

class H4MotivationInput(BaseModel):
    interet_camarades_ord:  int   = Field(1, ge=0, le=3)
    interet_camarades_sent: float = Field(0.0, ge=-1.0, le=1.0)
    souhait_freq:           int   = Field(2, ge=0, le=4)
    sexe_bin:               int   = Field(0, ge=0, le=1)
    age:                    float = Field(12.0, ge=5, le=100)

class H4EngagementInput(H4MotivationInput):
    pass

class H4DisciplinesInput(H4MotivationInput):
    pass

class PredictionResponse(BaseModel):
    prediction:       str | float | int
    probability:      Optional[float] = None
    model:            str
    model_version:    str
    timestamp:        str


# ── Model loader ──────────────────────────────────────────────────────────────

_FEATURE_COLS = {}  # stocke les colonnes attendues par chaque modèle

def load_models():
    """Charge tous les modèles disponibles au démarrage + leurs features."""
    global MODELS, _FEATURE_COLS
    import pandas as pd

    # H1 — nb_langues, apprent_anterieur_bin, relation_lm_ord, domaine_usage_freq,
    #       valorisation_sent, sexe_bin, age + lm_* + region_*
    h1_models = sorted(MODEL_DIR.glob("h1/*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if h1_models:
        with open(h1_models[0], "rb") as f:
            MODELS["h1"] = pickle.load(f)
        # Extraire les features du pipeline
        try:
            _FEATURE_COLS["h1"] = list(MODELS["h1"].feature_names_in_)
        except Exception:
            _FEATURE_COLS["h1"] = []

    # H2
    h2a = sorted(MODEL_DIR.glob("h2/*tuned*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if h2a:
        with open(h2a[0], "rb") as f:
            MODELS["h2"] = pickle.load(f)
        try:
            _FEATURE_COLS["h2"] = list(MODELS["h2"].feature_names_in_)
        except Exception:
            _FEATURE_COLS["h2"] = []

    # H3
    h3_reg = sorted(MODEL_DIR.glob("h3/*reg_tuned*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    h3_clf = sorted(MODEL_DIR.glob("h3/*clf_tuned*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if h3_reg:
        with open(h3_reg[0], "rb") as f:
            MODELS["h3_reg"] = pickle.load(f)
        try:
            _FEATURE_COLS["h3_reg"] = list(MODELS["h3_reg"].feature_names_in_)
        except Exception:
            _FEATURE_COLS["h3_reg"] = []
    if h3_clf:
        with open(h3_clf[0], "rb") as f:
            MODELS["h3_clf"] = pickle.load(f)
        try:
            _FEATURE_COLS["h3_clf"] = list(MODELS["h3_clf"].feature_names_in_)
        except Exception:
            _FEATURE_COLS["h3_clf"] = []

    # H4
    for suffix, key in [("A_motivation_tuned", "h4_a"), ("B_engagement_tuned", "h4_b"), ("C_discipline_tuned", "h4_c")]:
        h4m = sorted(MODEL_DIR.glob(f"h4/*{suffix}*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if h4m:
            with open(h4m[0], "rb") as f:
                MODELS[key] = pickle.load(f)
            try:
                _FEATURE_COLS[key] = list(MODELS[key].feature_names_in_)
            except Exception:
                _FEATURE_COLS[key] = []

    return list(MODELS.keys())


def _df_from_input(data: BaseModel, model_key: str, extra: dict = None) -> pd.DataFrame:
    """Convertit un modèle Pydantic en DataFrame, puis complète les colonnes
    manquantes (régions, langues, embeddings) avec des zéros pour matcher
    les features d'entraînement."""
    d = data.model_dump()
    if extra:
        d.update(extra)
    df = pd.DataFrame([d])

    # Pad with zeros for columns the model expects but the user didn't provide
    expected = _FEATURE_COLS.get(model_key, [])
    for col in expected:
        if col not in df.columns:
            df[col] = 0

    # Ensure column order matches training
    if expected:
        df = df[[c for c in expected if c in df.columns]]

    return df


def _fmt_resp(pred, proba=None, model_key="") -> PredictionResponse:
    return PredictionResponse(
        prediction=str(pred) if not isinstance(pred, (int, float)) else pred,
        probability=round(float(proba), 4) if proba is not None else None,
        model=cfg["mlflow"]["models"].get(model_key, model_key),
        model_version="latest",
        timestamp=datetime.now().isoformat(),
    )


# ── Prediction logging (MLOps Level 3) ──────────────────────────────────
_log = logging.getLogger("api.monitoring")
_PRED_DB_PATH = None


def _init_monitoring():
    """Initialize prediction logging DB from config."""
    global _PRED_DB_PATH
    db = cfg.get("monitoring", {}).get("prediction_db", "monitoring/predictions.db")
    _PRED_DB_PATH = Path(db)
    _PRED_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        from monitoring.prediction_logger import init_predictions_db
        init_predictions_db(_PRED_DB_PATH)
    except Exception as e:
        _log.warning("Monitoring DB init skipped: %s", e)


def _log_and_return(
    model_key: str,
    input_data: BaseModel,
    prediction,
    probability: Optional[float],
    response: PredictionResponse,
    start_time: float,
) -> PredictionResponse:
    """Log prediction to DB + Prometheus, then return response. Never fails."""
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    # Prometheus
    if PREDICTION_COUNTER:
        try:
            PREDICTION_COUNTER.labels(model_key=model_key).inc()
            PREDICTION_LATENCY.labels(model_key=model_key).observe(latency_ms / 1000.0)
        except Exception:
            pass
    # SQLite
    if _PRED_DB_PATH:
        try:
            from monitoring.prediction_logger import log_prediction
            log_prediction(
                db_path=_PRED_DB_PATH,
                model_key=model_key,
                model_version="latest",
                input_data=input_data.model_dump(),
                prediction=str(prediction),
                probability=probability,
                latency_ms=latency_ms,
            )
        except Exception as e:
            _log.warning("Prediction log skip: %s", e)
    return response


# ── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    loaded = load_models()
    if not loaded:
        print("[WARN] Aucun modele charge — executez d'abord le pipeline d'entrainement")
    else:
        print(f"[OK] {len(loaded)} modeles charges : {loaded}")
    _init_monitoring()


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": len(MODELS), "models": list(MODELS.keys())}


@app.get("/models")
def list_models():
    return {"models": list(MODELS.keys()), "details": cfg["mlflow"]["models"]}


# ── Prometheus /metrics (MLOps Level 3) ──────────────────────────────────

@app.get("/metrics")
def metrics():
    """Prometheus scraping endpoint for Grafana dashboards."""
    if not _PROMETHEUS:
        raise HTTPException(501, "prometheus_client not installed")
    return Response(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )


# ── Drift check (MLOps Level 3) ──────────────────────────────────────────

@app.post("/drift/check/{model_key}")
def run_drift_check(model_key: str):
    """Trigger drift detection for a model against reference data."""
    from monitoring.drift_detector import detect_drift, build_reference_data
    from monitoring.prediction_logger import get_recent_predictions, get_db_path
    from monitoring.alert import check_and_alert

    mc = cfg.get("monitoring", {}).get("drift", {})
    window = mc.get("window_size", 200)
    threshold = mc.get("drift_threshold", 0.3)
    ref_dir = Path(mc.get("reference_data_path", "data/processed/"))
    report_dir = Path(mc.get("report_dir", "monitoring/reports/"))
    alert_file = Path(cfg.get("monitoring", {}).get("alert", {}).get(
        "alerts_dir", "monitoring/alerts/")) / "alerts.ndjson"

    # Reference data
    hyp = "".join(c for c in model_key if not c.isdigit() and c != "_")
    ref_csv = ref_dir / f"{hyp}_features.csv"
    if not ref_csv.exists():
        ref_csv = ref_dir / "h1_features.csv"  # fallback
    if not ref_csv.exists():
        raise HTTPException(503, f"Reference data not found: {ref_csv}")

    reference = build_reference_data(ref_csv, model_key)
    current = get_recent_predictions(_PRED_DB_PATH or get_db_path(), model_key, limit=window)

    if len(current) < 10:
        return {"status": "insufficient_data", "n_predictions": len(current)}

    result = detect_drift(reference, current, model_key, threshold, report_dir)

    # Check alerts
    check_and_alert(result, alert_file)

    # Update Prometheus gauge
    if DRIFT_GAUGE:
        try:
            DRIFT_GAUGE.labels(model_key=model_key).set(result["drift_ratio"])
        except Exception:
            pass

    return result


# ── H1 — Usage quotidien des langues ─────────────────────────────────────────

@app.post("/predict/h1", response_model=PredictionResponse)
def predict_h1(data: H1Input):
    if "h1" not in MODELS:
        raise HTTPException(503, "Modèle H1 non disponible")
    _t0 = time.perf_counter()
    df = _df_from_input(data, "h1")
    proba = MODELS["h1"].predict_proba(df)[0, 1]
    pred  = "Oui" if proba >= 0.5 else "Non"
    resp = _fmt_resp(pred, proba, "h1")
    return _log_and_return("h1", data, pred, proba, resp, _t0)


# ── H2 — Motivation & Difficultés ────────────────────────────────────────────

@app.post("/predict/h2/motivation", response_model=PredictionResponse)
def predict_h2_motivation(data: H2MotivationInput):
    if "h2" not in MODELS:
        raise HTTPException(503, "Modèle H2 non disponible")
    _t0 = time.perf_counter()
    df = _df_from_input(data, "h2")
    pred = MODELS["h2"].predict(df)[0]
    labels = {0: "Faible", 1: "Moyenne", 2: "Élevée"}
    label = labels.get(pred, str(pred))
    resp = _fmt_resp(label, model_key="h2")
    return _log_and_return("h2", data, label, None, resp, _t0)


DIFF_LABELS = ["grammaire", "vocabulaire", "orthographe", "conjugaison",
               "expression_orale", "comprehension", "analyse"]

@app.post("/predict/h2/difficultes", response_model=PredictionResponse)
def predict_h2_difficultes(data: H2MotivationInput):
    if "h2" not in MODELS:
        raise HTTPException(503, "Modèle H2 non disponible")
    _t0 = time.perf_counter()
    df = _df_from_input(data, "h2")
    preds = MODELS["h2"].predict(df)[0]
    detected = [DIFF_LABELS[i] for i, v in enumerate(preds) if v == 1]
    label = ", ".join(detected) if detected else "Aucune"
    resp = _fmt_resp(label, model_key="h2")
    return _log_and_return("h2", data, label, None, resp, _t0)



# ── H3 — Attitude envers le français ─────────────────────────────────────────

@app.post("/predict/h3/attitude", response_model=PredictionResponse)
def predict_h3_attitude(data: H3AttitudeInput):
    if "h3_reg" not in MODELS:
        raise HTTPException(503, "Modèle H3 regression non disponible")
    _t0 = time.perf_counter()
    df = _df_from_input(data, "h3_reg")
    score = round(float(MODELS["h3_reg"].predict(df)[0]), 2)
    resp = _fmt_resp(score, model_key="h3")
    return _log_and_return("h3_reg", data, score, None, resp, _t0)


@app.post("/predict/h3/classification", response_model=PredictionResponse)
def predict_h3_classification(data: H3AttitudeInput):
    if "h3_clf" not in MODELS:
        raise HTTPException(503, "Modèle H3 classification non disponible")
    _t0 = time.perf_counter()
    df = _df_from_input(data, "h3_clf")
    pred = int(MODELS["h3_clf"].predict(df)[0])
    labels = {0: "Négative", 1: "Neutre", 2: "Positive"}
    label = labels.get(pred, str(pred))
    resp = _fmt_resp(label, model_key="h3")
    return _log_and_return("h3_clf", data, label, None, resp, _t0)


# ── H4 — Intégration langues locales ─────────────────────────────────────────

@app.post("/predict/h4/motivation", response_model=PredictionResponse)
def predict_h4_motivation(data: H4MotivationInput):
    if "h4_a" not in MODELS:
        raise HTTPException(503, "Modèle H4 motivation non disponible")
    _t0 = time.perf_counter()
    df = _df_from_input(data, "h4_a")
    proba = MODELS["h4_a"].predict_proba(df)[0, 1]
    pred  = "Motivé" if proba >= 0.5 else "Peu motivé"
    resp = _fmt_resp(pred, proba, "h4")
    return _log_and_return("h4_a", data, pred, proba, resp, _t0)


@app.post("/predict/h4/engagement", response_model=PredictionResponse)
def predict_h4_engagement(data: H4EngagementInput):
    if "h4_b" not in MODELS:
        raise HTTPException(503, "Modèle H4 engagement non disponible")
    _t0 = time.perf_counter()
    df = _df_from_input(data, "h4_b")
    score = int(MODELS["h4_b"].predict(df)[0] + 1)
    resp = _fmt_resp(score, model_key="h4")
    return _log_and_return("h4_b", data, score, None, resp, _t0)


DISC_LABELS = ["vocabulaire", "grammaire", "lecture", "expression_orale", "conjugaison"]

@app.post("/predict/h4/disciplines", response_model=PredictionResponse)
def predict_h4_disciplines(data: H4DisciplinesInput):
    if "h4_c" not in MODELS:
        raise HTTPException(503, "Modèle H4 disciplines non disponible")
    _t0 = time.perf_counter()
    df = _df_from_input(data, "h4_c")
    preds = MODELS["h4_c"].predict(df)[0]
    detected = [DISC_LABELS[i] for i, v in enumerate(preds) if v == 1]
    label = ", ".join(detected) if detected else "Aucune"
    resp = _fmt_resp(label, model_key="h4")
    return _log_and_return("h4_c", data, label, None, resp, _t0)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", cfg.get("api", {}).get("port", 8001)))
    uvicorn.run(app, host="0.0.0.0", port=port)
