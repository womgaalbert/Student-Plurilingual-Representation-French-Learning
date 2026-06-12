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
from pathlib import Path
import pickle
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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


# ── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    loaded = load_models()
    if not loaded:
        print("[WARN] Aucun modele charge — executez d'abord le pipeline d'entrainement")
    else:
        print(f"[OK] {len(loaded)} modeles charges : {loaded}")


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": len(MODELS), "models": list(MODELS.keys())}


@app.get("/models")
def list_models():
    return {"models": list(MODELS.keys()), "details": cfg["mlflow"]["models"]}


# ── H1 — Usage quotidien des langues ─────────────────────────────────────────

@app.post("/predict/h1", response_model=PredictionResponse)
def predict_h1(data: H1Input):
    if "h1" not in MODELS:
        raise HTTPException(503, "Modèle H1 non disponible")
    df = _df_from_input(data, "h1")
    proba = MODELS["h1"].predict_proba(df)[0, 1]
    pred  = "Oui" if proba >= 0.5 else "Non"
    return _fmt_resp(pred, proba, "h1")


# ── H2 — Motivation & Difficultés ────────────────────────────────────────────

@app.post("/predict/h2/motivation", response_model=PredictionResponse)
def predict_h2_motivation(data: H2MotivationInput):
    if "h2" not in MODELS:
        raise HTTPException(503, "Modèle H2 non disponible")
    df = _df_from_input(data, "h2")
    pred = MODELS["h2"].predict(df)[0]
    labels = {0: "Faible", 1: "Moyenne", 2: "Élevée"}
    return _fmt_resp(labels.get(pred, str(pred)), model_key="h2")


DIFF_LABELS = ["grammaire", "vocabulaire", "orthographe", "conjugaison",
               "expression_orale", "comprehension", "analyse"]

@app.post("/predict/h2/difficultes", response_model=PredictionResponse)
def predict_h2_difficultes(data: H2MotivationInput):
    if "h2" not in MODELS:
        raise HTTPException(503, "Modèle H2 non disponible")
    df = _df_from_input(data, "h2")
    preds = MODELS["h2"].predict(df)[0]
    detected = [DIFF_LABELS[i] for i, v in enumerate(preds) if v == 1]
    return _fmt_resp(", ".join(detected) if detected else "Aucune", model_key="h2")



# ── H3 — Attitude envers le français ─────────────────────────────────────────

@app.post("/predict/h3/attitude", response_model=PredictionResponse)
def predict_h3_attitude(data: H3AttitudeInput):
    if "h3_reg" not in MODELS:
        raise HTTPException(503, "Modèle H3 regression non disponible")
    df = _df_from_input(data, "h3_reg")
    score = float(MODELS["h3_reg"].predict(df)[0])
    return _fmt_resp(round(score, 2), model_key="h3")


@app.post("/predict/h3/classification", response_model=PredictionResponse)
def predict_h3_classification(data: H3AttitudeInput):
    if "h3_clf" not in MODELS:
        raise HTTPException(503, "Modèle H3 classification non disponible")
    df = _df_from_input(data, "h3_clf")
    pred = int(MODELS["h3_clf"].predict(df)[0])
    labels = {0: "Négative", 1: "Neutre", 2: "Positive"}
    return _fmt_resp(labels.get(pred, str(pred)), model_key="h3")


# ── H4 — Intégration langues locales ─────────────────────────────────────────

@app.post("/predict/h4/motivation", response_model=PredictionResponse)
def predict_h4_motivation(data: H4MotivationInput):
    if "h4_a" not in MODELS:
        raise HTTPException(503, "Modèle H4 motivation non disponible")
    df = _df_from_input(data, "h4_a")
    proba = MODELS["h4_a"].predict_proba(df)[0, 1]
    pred  = "Motivé" if proba >= 0.5 else "Peu motivé"
    return _fmt_resp(pred, proba, "h4")


@app.post("/predict/h4/engagement", response_model=PredictionResponse)
def predict_h4_engagement(data: H4EngagementInput):
    if "h4_b" not in MODELS:
        raise HTTPException(503, "Modèle H4 engagement non disponible")
    df = _df_from_input(data, "h4_b")
    score = int(MODELS["h4_b"].predict(df)[0] + 1)  # 0-indexed
    return _fmt_resp(score, model_key="h4")


DISC_LABELS = ["vocabulaire", "grammaire", "lecture", "expression_orale", "conjugaison"]

@app.post("/predict/h4/disciplines", response_model=PredictionResponse)
def predict_h4_disciplines(data: H4DisciplinesInput):
    if "h4_c" not in MODELS:
        raise HTTPException(503, "Modèle H4 disciplines non disponible")
    df = _df_from_input(data, "h4_c")
    preds = MODELS["h4_c"].predict(df)[0]
    detected = [DISC_LABELS[i] for i, v in enumerate(preds) if v == 1]
    return _fmt_resp(", ".join(detected) if detected else "Aucune", model_key="h4")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", cfg.get("api", {}).get("port", 8001)))
    uvicorn.run(app, host="0.0.0.0", port=port)
