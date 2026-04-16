"""
preprocess.py — French-Learning-Perceptions ML (Level 0 + Level 1)
Pipeline complet : chargement → filtre → anonymisation → encodage → features H1-H4.
Loggue les statistiques du dataset dans MLflow.
"""

import re
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow

from src.utils.constants import (
    FREQ_MAP, LIKERT_MAP, IMPORTANCE_MAP, RELATION_LM_MAP,
    DIFFICULTE_KEYWORDS, DISCIPLINE_KEYWORDS, PERCEPTION_LABELS,
    MOTIVATION_HIGH_KW, MOTIVATION_LOW_KW,
    COL_CONSENTEMENT, COL_AGE, COL_SEXE, COL_REGION,
    COL_LANGUES_PARLEES, COL_LANGUE_MATERNELLE, COL_USAGE_QUOTIDIEN,
    COL_APPRENTISSAGE_ANT, COL_RELATION_LM, COL_AVANTAGE_MULTI,
    COL_PERCEPTION_FR, COL_MOTS_ASSOCIES, COL_MOTIVATION,
    COL_FACILE, COL_DIFFICILE, COL_IMPORTANCE_FR, COL_COMPARAISON_FR,
    COL_EXPOSITION, COL_INTERET_APPRENDRE, COL_PERCEPTION_MULTI,
    COL_MOTIVATION_CAMERO, COL_PARLE_COURS, COL_INTERET_CAMARADES,
    COL_DIFFICULTES, COL_ORIGINE_DIFF, COL_INCLURE_LANGUES, COL_DISCIPLINE_ASSOC,
)
from src.utils.mlflow_utils import load_params, setup_mlflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/preprocess.log", encoding="utf-8"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

# ── Mapping colonnes CSV longues → noms courts ──────────────────────────────
CSV_MAP = {
    "Demande de consentement":                          COL_CONSENTEMENT,
    "Informations personnelles":                         COL_AGE,
    "sexe":                                             COL_SEXE,
    "Région d'origine":                                 COL_REGION,
    "Etablissement":                                    "etablissement",
    "Classe":                                           "classe",
    "Quelles sont les langues que vous parlez":         COL_LANGUES_PARLEES,
    "Quelle est votre langue maternelle":               COL_LANGUE_MATERNELLE,
    "Utilisez-vous d'autres langues dans votre quotidien ?": COL_USAGE_QUOTIDIEN,
    "Utilisez-vous d'autres langues dans votre quotidien ? Si oui": "usage_quotidien_details",
    "Avez-vous appris d'autres langues avant le français": COL_APPRENTISSAGE_ANT,
    "Comment décririez-vous votre relation":            COL_RELATION_LM,
    "Pensez-vous que parler plusieurs langues est un avantage": COL_AVANTAGE_MULTI,
    "Comment percevez-vous la langue française":        COL_PERCEPTION_FR,
    "citez quatre mots":                                COL_MOTS_ASSOCIES,
    "motive à apprendre le français":                   COL_MOTIVATION,
    "[Facile]":                                         COL_FACILE,
    "[Difficile]":                                      COL_DIFFICILE,
    "le français est une langue importante":            COL_IMPORTANCE_FR,
    "le français se compare":                           COL_COMPARAISON_FR,
    "l'occasion d'entendre ou de parler d'autres langues": COL_EXPOSITION,
    "apprendre davantage sur d'autres langues":         COL_INTERET_APPRENDRE,
    "percevez-vous le fait de parler plusieurs langues": COL_PERCEPTION_MULTI,
    "langues camerounaises vous motiverait":            COL_MOTIVATION_CAMERO,
    "langues des élèves":                               COL_PARLE_COURS,
    "langues parlées par vos camarades":                COL_INTERET_CAMARADES,
    "principales difficultés":                          COL_DIFFICULTES,
    "liées à la langue elle-même":                      COL_ORIGINE_DIFF,
    "cours de français incluent davantage":             COL_INCLURE_LANGUES,
    "discipline du français":                           COL_DISCIPLINE_ASSOC,
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _normalize(s: str) -> str:
    """Réduit une chaîne à ses lettres et chiffres ASCII pour la comparaison.
    Utilise NFD pour convertir les caractères accentués en leur base ASCII.
    """
    import re, unicodedata
    s = unicodedata.normalize('NFD', s).encode('ascii', 'ignore').decode('ascii')
    return re.sub(r'[^a-z0-9]', '', s.lower())


def _rename(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        col_norm = _normalize(col)
        best_key, best_short, best_len = None, None, 0
        for key, short in CSV_MAP.items():
            key_norm = _normalize(key)
            if key_norm and key_norm in col_norm and len(key_norm) > best_len:
                best_key, best_short, best_len = key, short, len(key_norm)
        if best_short:
            rename[col] = best_short
    return df.rename(columns=rename)


def _parse_age(val) -> float:
    if pd.isna(val): return np.nan
    m = re.search(r"\d+", str(val))
    return float(m.group()) if m else np.nan


def extract_oui_non(text) -> int:
    if pd.isna(text): return -1
    t = str(text).upper().strip()
    if t.startswith("OUI"): return 1
    if t.startswith("NON"): return 0
    return -1


def simple_sentiment(text) -> float:
    if pd.isna(text): return 0.0
    t = str(text).lower()
    pos = ["bien", "bon", "avantage", "utile", "important", "réussir",
           "aider", "faciliter", "intéressant", "communiquer", "meilleur"]
    neg = ["difficile", "dur", "compliqué", "problème", "faute", "mauvais"]
    score = sum(1 for w in pos if w in t) - sum(1 for w in neg if w in t)
    return float(np.clip(score / 3.0, -1.0, 1.0))


def count_languages(text) -> int:
    if pd.isna(text): return 0
    parts = re.split(r"[,;/\n]| et | and ", str(text), flags=re.IGNORECASE)
    return len([p.strip() for p in parts if p.strip()])


def extract_multilabels(text, kw_map: dict) -> list:
    if pd.isna(text): return []
    t = str(text).lower()
    return [label for label, kws in kw_map.items() if any(k in t for k in kws)]


# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPES PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def step_load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = _rename(df)
    log.info(f"Chargé : {len(df)} lignes, {len(df.columns)} colonnes")
    return df


def step_consent(df: pd.DataFrame) -> pd.DataFrame:
    col = COL_CONSENTEMENT
    if col not in df.columns:
        cols = [c for c in df.columns if "consentement" in c.lower() or "accepte" in c.lower()]
        if cols: df = df.rename(columns={cols[0]: col})
    mask = df[col].str.lower().str.contains("accepte", na=False)
    df   = df[mask].copy().reset_index(drop=True)
    log.info(f"Consentement : {len(df)} répondants valides")
    return df


def step_anonymize(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in df.columns if any(
        kw in c.lower() for kw in ["horodateur", "timestamp", "email"]
    )]
    df = df.drop(columns=drop, errors="ignore")
    return df


def step_demographics(df: pd.DataFrame) -> pd.DataFrame:
    if COL_AGE in df.columns:
        df[COL_AGE] = df[COL_AGE].apply(_parse_age)
        df[COL_AGE] = df[COL_AGE].fillna(df[COL_AGE].median())
    if COL_SEXE in df.columns:
        df["sexe_bin"] = df[COL_SEXE].str.upper().str.strip().map({"M": 1, "F": 0})
    if COL_REGION in df.columns:
        df[COL_REGION] = df[COL_REGION].str.strip().str.title()
        dummies = pd.get_dummies(df[COL_REGION], prefix="region")
        df = pd.concat([df, dummies], axis=1)
    return df


# ── Features par hypothèse ──────────────────────────────────────────────────

def build_h1(df: pd.DataFrame) -> pd.DataFrame:
    h = df.copy()
    h["nb_langues"]            = h[COL_LANGUES_PARLEES].apply(count_languages)
    h["apprent_anterieur_bin"] = h[COL_APPRENTISSAGE_ANT].apply(extract_oui_non)
    h["relation_lm_ord"]       = h[COL_RELATION_LM].map(RELATION_LM_MAP).fillna(0)
    h["domaine_usage_freq"]    = h[COL_EXPOSITION].map(FREQ_MAP).fillna(0)
    h["valorisation_sent"]     = h[COL_AVANTAGE_MULTI].apply(simple_sentiment)
    # one-hot langue maternelle
    if COL_LANGUE_MATERNELLE in h.columns:
        top_lm = h[COL_LANGUE_MATERNELLE].value_counts().head(10).index
        for lm in top_lm:
            h[f"lm_{lm.lower().replace(' ','_')}"] = (
                h[COL_LANGUE_MATERNELLE].str.lower() == lm.lower()
            ).astype(int)
    # CIBLE
    h["h1_target"] = h[COL_USAGE_QUOTIDIEN].apply(extract_oui_non)
    h = h[h["h1_target"].isin([0, 1])].copy()
    log.info(f"H1 : {len(h)} lignes | balance={h['h1_target'].value_counts().to_dict()}")
    return h


def build_h2(df: pd.DataFrame) -> pd.DataFrame:
    h = df.copy()
    for label in PERCEPTION_LABELS:
        h[f"perc_{label}"] = h[COL_PERCEPTION_FR].str.lower().str.contains(
            label, na=False).astype(int)
    h["mots_assoc_sent"] = h[COL_MOTS_ASSOCIES].apply(simple_sentiment)
    h["importance_bin"]  = h[COL_IMPORTANCE_FR].apply(extract_oui_non)
    h["importance_sent"] = h[COL_IMPORTANCE_FR].apply(simple_sentiment)
    h["hierarchie_fr"]   = h[COL_COMPARAISON_FR].str.lower().map(
        {k.lower(): v for k, v in IMPORTANCE_MAP.items()}).fillna(2.0)
    # Cible A — motivation
    def _motiv(text):
        if pd.isna(text): return 1
        t = str(text).lower()
        if sum(1 for w in MOTIVATION_HIGH_KW if w in t) >= 2: return 2
        if sum(1 for w in MOTIVATION_LOW_KW  if w in t) >= 1: return 0
        return 1
    h["h2_target_motivation"] = h[COL_MOTIVATION].apply(_motiv)
    # Cible B — difficultés multi-label
    def _diff(row):
        txt = " ".join([str(row.get(COL_DIFFICILE, "")),
                        str(row.get(COL_DIFFICULTES, "")),
                        str(row.get(COL_ORIGINE_DIFF, ""))])
        lbls = extract_multilabels(txt, DIFFICULTE_KEYWORDS)
        return {f"diff_{k}": int(k in lbls) for k in DIFFICULTE_KEYWORDS}
    h = pd.concat([h, h.apply(_diff, axis=1).apply(pd.Series)], axis=1)
    log.info(f"H2 : {len(h)} lignes | motiv={h['h2_target_motivation'].value_counts().to_dict()}")
    return h


def build_h3(df: pd.DataFrame) -> pd.DataFrame:
    h = df.copy()
    h["exposition_freq"]       = h[COL_EXPOSITION].map(FREQ_MAP).fillna(0)
    h["interet_bin"]           = h[COL_INTERET_APPRENDRE].apply(extract_oui_non)
    h["interet_sent"]          = h[COL_INTERET_APPRENDRE].apply(simple_sentiment)
    h["perception_multi_sent"] = h[COL_PERCEPTION_MULTI].apply(simple_sentiment)
    h["perception_multi_ord"]  = h[COL_PERCEPTION_MULTI].map(LIKERT_MAP).fillna(2.5)

    def _score(row):
        comp = str(row.get(COL_COMPARAISON_FR, "")).lower()
        s1   = 3.0 if "plus" in comp else (2.0 if "autant" in comp else 1.0)
        s2   = 2.0 if str(row.get(COL_MOTIVATION_CAMERO, "")).upper().startswith("OUI") else 0.5
        return float(np.clip(1 + ((s1/3)*3 + (s2/2)*2) / 5 * 4, 1.0, 5.0))

    h["h3_score_attitude"] = h.apply(_score, axis=1)
    h["h3_attitude_class"] = h["h3_score_attitude"].apply(
        lambda s: "Positive" if s >= 3.5 else ("Neutre" if s >= 2.5 else "Négative"))
    log.info(f"H3 : {len(h)} | score moy={h['h3_score_attitude'].mean():.2f} | "
             f"classes={h['h3_attitude_class'].value_counts().to_dict()}")
    return h


def build_h4(df: pd.DataFrame) -> pd.DataFrame:
    h = df.copy()
    h["interet_camarades_bin"]  = h[COL_INTERET_CAMARADES].apply(extract_oui_non)
    h["interet_camarades_sent"] = h[COL_INTERET_CAMARADES].apply(simple_sentiment)
    h["souhait_freq"]           = h[COL_INCLURE_LANGUES].map(FREQ_MAP).fillna(0)
    disc_list = h[COL_DISCIPLINE_ASSOC].apply(
        lambda x: extract_multilabels(x, DISCIPLINE_KEYWORDS))
    for disc in DISCIPLINE_KEYWORDS:
        h[f"vi_disc_{disc}"]  = disc_list.apply(lambda l: int(disc in l))
        h[f"vd_disc_{disc}"]  = h[f"vi_disc_{disc}"]
    h["h4_target_motivation"] = h[COL_MOTIVATION_CAMERO].apply(
        lambda x: 1 if LIKERT_MAP.get(str(x).strip(), 0) >= 3 else 0)

    def _eng(row):
        mp  = 2 if str(row.get(COL_MOTIVATION_CAMERO,"")).upper().startswith("OUI") else 0
        fp  = FREQ_MAP.get(str(row.get(COL_INCLURE_LANGUES,"")), 0)
        raw = mp + (fp / 4.0) * 2
        return 4 if raw >= 3.5 else (3 if raw >= 2.5 else (2 if raw >= 1.0 else 1))

    h["h4_engagement_score"] = h.apply(_eng, axis=1)
    log.info(f"H4 : {len(h)} | motiv={h['h4_target_motivation'].value_counts().to_dict()}")
    return h


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run(input_path: str, output_dir: str, params_path: str = "params.yaml") -> dict:
    params = load_params(params_path)
    setup_mlflow(params)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    log.info("═" * 55)
    log.info("FRENCH-LEARNING-PERCEPTIONS ML — PRÉTRAITEMENT")
    log.info("═" * 55)

    df = step_load(input_path)
    df = step_consent(df)
    df = step_anonymize(df)
    df = step_demographics(df)
    df.to_csv(out / "french-learning-perceptions_clean.csv", index=False)

    datasets = {
        "H1": build_h1(df),
        "H2": build_h2(df),
        "H3": build_h3(df),
        "H4": build_h4(df),
    }
    for h, d in datasets.items():
        d.to_csv(out / f"{h.lower()}_features.csv", index=False)

    # Log dataset stats dans MLflow
    with mlflow.start_run(run_name="preprocess"):
        mlflow.set_tag("step", "preprocess")
        mlflow.log_params({
            "n_total":     len(df),
            "n_columns":   len(df.columns),
            "input_path":  input_path,
        })
        for h, d in datasets.items():
            mlflow.log_metric(f"n_{h.lower()}", len(d))

    log.info("═" * 55)
    log.info(f"Terminé — {len(df)} répondants | 4 datasets générés")
    log.info("═" * 55)
    return {h: len(d) for h, d in datasets.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/raw/data_FLP.csv")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    run(args.input, args.output, args.params)
