# databricks_backend.py — Live inference via Databricks Model Serving
#
# When Streamlit Cloud secrets contain DATABRICKS_HOST + DATABRICKS_TOKEN,
# predictions are served by the live Databricks endpoint (flp-all-models).
# Otherwise the app falls back to local .pkl models (see app.py).
#
# Secrets (set in Streamlit Cloud → Settings → Secrets, never in the repo):
#   DATABRICKS_HOST  = "https://dbc-9e268203-7090.cloud.databricks.com"
#   DATABRICKS_TOKEN = "dapi..."
import numpy as np
import pandas as pd
import streamlit as st

ENDPOINT_PATH = "/serving-endpoints/flp-all-models/invocations"

# Input contract of flp_catalog.models.flp_all (60 ASCII-safe feature columns)
FEATURES = [
    "age", "apprent_anterieur_bin", "domaine_usage_freq", "exposition_freq",
    "hierarchie_fr", "importance_bin", "importance_sent", "interet_bin",
    "interet_camarades_bin", "interet_camarades_sent", "interet_sent",
    "lm_bassa", "lm_bassa'a", "lm_bulu", "lm_eton", "lm_ewondo", "lm_foufoulde",
    "lm_fufulde", "lm_ghomala", "lm_haoussa", "lm_mafa", "mots_assoc_sent",
    "nb_langues", "perc_belle", "perc_difficile", "perc_importante", "perc_utile",
    "perception_multi_ord", "perception_multi_sent", "region_Adamaoua",
    "region_Adamy", "region_Centrafrique", "region_Centre", "region_Centre_Afrique",
    "region_Centre_Est", "region_Cet", "region_Est", "region_Extreme_Nord",
    "region_Extreme-Nord", "region_Francais", "region_Gabon", "region_L'Est",
    "region_Littoral", "region_Niger", "region_Nigeria", "region_Nord",
    "region_Ouest", "region_Republique_Centrafricaine", "region_Sud",
    "region_Sud_Ouest", "region_Tchad", "relation_lm_ord", "sexe_bin",
    "souhait_freq", "valorisation_sent", "vi_disc_conjugaison",
    "vi_disc_expression_orale", "vi_disc_grammaire", "vi_disc_lecture",
    "vi_disc_vocabulaire",
]

# App model keys → served model_name dispatch values
MODEL_MAP = {
    "h1": "h1",
    "h2": "h2a",
    "h3_reg": "h3r",
    "h3_clf": "h3c",
    "h4_a": "h4a",
    "h4_b": "h4b",
    "h4_c": "h4c",
}

# Local-pickle column names → endpoint contract names
COLUMN_ALIASES = {
    "perc_dificil": "perc_difficile",
    "perc_importan": "perc_importante",
}


def available() -> bool:
    try:
        return bool(st.secrets.get("DATABRICKS_HOST")) and bool(
            st.secrets.get("DATABRICKS_TOKEN")
        )
    except Exception:
        return False


def _endpoint_url() -> str:
    host = str(st.secrets["DATABRICKS_HOST"]).rstrip("/")
    return f"{host}{ENDPOINT_PATH}"


def _build_record(model_key: str, df: pd.DataFrame) -> dict:
    row = df.iloc[0]
    record = {c: 0.0 for c in FEATURES}
    record["model_name"] = MODEL_MAP[model_key]

    for c in FEATURES:
        if c in df.columns:
            record[c] = float(row[c])

    for src, dst in COLUMN_ALIASES.items():
        if src in df.columns and dst in FEATURES:
            record[dst] = float(row[src])

    # App (local pickles) uses binary flags where the Databricks models use
    # richer ordinal features — map them explicitly.
    if "exposition_bin" in df.columns:
        record["exposition_freq"] = 3.0 if float(row["exposition_bin"]) == 1 else 0.0
    if "interet_camarades_ord" in df.columns:
        record["interet_camarades_bin"] = 1.0 if float(row["interet_camarades_ord"]) >= 2 else 0.0

    return record


def _call(model_key: str, df: pd.DataFrame) -> dict:
    import requests

    resp = requests.post(
        _endpoint_url(),
        headers={"Authorization": f"Bearer {st.secrets['DATABRICKS_TOKEN']}"},
        json={"dataframe_records": [_build_record(model_key, df)]},
        timeout=300,  # scale-to-zero cold start can take 1-2 min
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"Databricks endpoint error {resp.status_code}: {resp.text[:300]}"
        )
    return resp.json()["predictions"][0]


def predict(model_key: str, df: pd.DataFrame) -> np.ndarray:
    """Same return contract as app.predict() for the local pickles."""
    p = _call(model_key, df)
    if model_key == "h1":
        return np.array([p["h1_pred"]])
    if model_key == "h2":
        return np.array([p["h2a_pred"]])
    if model_key == "h3_reg":
        return np.array([p["h3_reg"]])
    if model_key == "h3_clf":
        return np.array([p["h3_clf"]])
    if model_key == "h4_a":
        return np.array([p["h4a_pred"]])
    if model_key == "h4_b":
        # app.py adds +1 after predict — endpoint already returns 1..4
        return np.array([p["h4b_engagement"] - 1.0])
    if model_key == "h4_c":
        return np.array([p[f"h4c_{i}"] for i in range(5)])
    raise ValueError(f"Unknown model key: {model_key}")


def predict_proba(model_key: str, df: pd.DataFrame) -> np.ndarray:
    """Same return contract as app.predict_proba() for the local pickles."""
    p = _call(model_key, df)
    if model_key == "h1":
        return np.array([[1.0 - p["h1_proba"], p["h1_proba"]]])
    if model_key == "h4_a":
        return np.array([[1.0 - p["h4a_proba"], p["h4a_proba"]]])
    raise ValueError(f"No predict_proba for model key: {model_key}")
