"""
preprocess.py — French-Learning-Perceptions ML (Level 0)
Pipeline de nettoyage et encodage pour les 4 hypothèses.
Usage : python src/preprocess.py --config params.yaml
"""
import re
import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from utils.config import load_config
from utils.constants import (
    CSV_COLUMN_MAP, FREQ_MAP, LIKERT_MAP, IMPORTANCE_MAP, RELATION_LM_MAP,
    DIFFICULTE_KEYWORDS, DISCIPLINE_KEYWORDS,
    MOTIVATION_HIGH_KW, MOTIVATION_LOW_KW,
    COL_CONSENTEMENT, COL_AGE, COL_SEXE, COL_REGION,
    COL_LANGUES_PARLEES, COL_LANGUE_MATERNELLE, COL_USAGE_QUOTIDIEN,
    COL_APPRENTISSAGE_ANT, COL_RELATION_LM, COL_AVANTAGE_MULTI,
    COL_PERCEPTION_FR, COL_MOTS_ASSOCIES, COL_MOTIVATION,
    COL_FACILE, COL_DIFFICILE, COL_IMPORTANCE_FR, COL_COMPARAISON_FR,
    COL_EXPOSITION, COL_INTERET_APPRENDRE, COL_PERCEPTION_MULTI,
    COL_MOTIVATION_CAMERO, COL_INTERET_CAMARADES,
    COL_DIFFICULTES, COL_ORIGINE_DIFF, COL_INCLURE_LANGUES, COL_DISCIPLINE_ASSOC,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/preprocess.log", encoding="utf-8"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_age(val) -> float:
    if pd.isna(val): return np.nan
    m = re.search(r"\d+", str(val))
    return float(m.group()) if m else np.nan

def extract_oui_non(text) -> int:
    if pd.isna(text): return -1
    return 1 if str(text).upper().strip().startswith("OUI") else (
           0 if str(text).upper().strip().startswith("NON") else -1)

def simple_sentiment(text) -> float:
    if pd.isna(text): return 0.0
    t = str(text).lower()
    pos = sum(1 for w in ["bien", "bon", "utile", "avantage", "réussir",
                           "aider", "communiquer", "important"] if w in t)
    neg = sum(1 for w in ["difficile", "dur", "faute", "mauvais"] if w in t)
    return float(np.clip((pos - neg) / 3.0, -1.0, 1.0))

def count_languages(text) -> int:
    if pd.isna(text): return 0
    parts = re.split(r"[,;/\n]| et | and ", str(text), flags=re.IGNORECASE)
    return len([p.strip() for p in parts if p.strip()])

def extract_multilabels(text, kw_map: dict) -> list:
    if pd.isna(text): return []
    t = str(text).lower()
    return [label for label, kws in kw_map.items() if any(k in t for k in kws)]

def motivation_label(text) -> int:
    if pd.isna(text): return 1
    t = str(text).lower()
    high = sum(1 for w in MOTIVATION_HIGH_KW if w in t)
    low  = sum(1 for w in MOTIVATION_LOW_KW  if w in t)
    return 2 if high >= 2 else (0 if low >= 1 else 1)


# ── Pipeline ─────────────────────────────────────────────────────────────────

def load_and_rename(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {}
    for col in df.columns:
        for csv_name, short in CSV_COLUMN_MAP.items():
            if csv_name.strip() in col.strip() or col.strip() in csv_name.strip():
                rename[col] = short
                break
    return df.rename(columns=rename)

def filter_consent(df: pd.DataFrame, consent_value: str) -> pd.DataFrame:
    cols = [c for c in df.columns if "consentement" in c.lower() or "accepte" in c.lower()]
    if cols and COL_CONSENTEMENT not in df.columns:
        df = df.rename(columns={cols[0]: COL_CONSENTEMENT})
    mask = df[COL_CONSENTEMENT].str.lower().str.contains(consent_value, na=False)
    n = len(df)
    df = df[mask].reset_index(drop=True)
    log.info(f"Consentement : {len(df)}/{n} lignes valides")
    return df

def anonymize(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in df.columns if any(k in c.lower()
            for k in ["horodateur", "timestamp", "email", "nom", "prenom"])]
    return df.drop(columns=drop, errors="ignore")

def clean_demographics(df: pd.DataFrame) -> pd.DataFrame:
    if COL_AGE in df.columns:
        df[COL_AGE] = df[COL_AGE].apply(parse_age).fillna(df[COL_AGE].apply(parse_age).median())
    if COL_SEXE in df.columns:
        df["sexe_bin"] = df[COL_SEXE].str.upper().str.strip().map({"M": 1, "F": 0})
    if COL_REGION in df.columns:
        df[COL_REGION] = df[COL_REGION].str.strip().str.title()
        df = pd.concat([df, pd.get_dummies(df[COL_REGION], prefix="region")], axis=1)
    return df

def build_h1(df: pd.DataFrame) -> pd.DataFrame:
    h = df.copy()
    h["nb_langues"]            = h[COL_LANGUES_PARLEES].apply(count_languages)
    h["apprent_anterieur_bin"] = h[COL_APPRENTISSAGE_ANT].apply(extract_oui_non)
    h["relation_lm_ord"]       = h[COL_RELATION_LM].map(RELATION_LM_MAP).fillna(0)
    h["domaine_usage_freq"]    = h[COL_EXPOSITION].map(FREQ_MAP).fillna(0)
    h["valorisation_sent"]     = h[COL_AVANTAGE_MULTI].apply(simple_sentiment)
    h["h1_target"]             = h[COL_USAGE_QUOTIDIEN].apply(extract_oui_non)
    top_lm = h[COL_LANGUE_MATERNELLE].value_counts().head(10).index
    for lm in top_lm:
        h[f"lm_{lm.lower().replace(' ','_')}"] = (
            h[COL_LANGUE_MATERNELLE].str.lower() == lm.lower()).astype(int)
    return h[h["h1_target"].isin([0, 1])].reset_index(drop=True)

def build_h2(df: pd.DataFrame) -> pd.DataFrame:
    h = df.copy()
    for lbl in ["utile", "belle", "difficile", "importante"]:
        h[f"perc_{lbl}"] = h[COL_PERCEPTION_FR].str.lower().str.contains(lbl, na=False).astype(int)
    h["mots_assoc_sent"]    = h[COL_MOTS_ASSOCIES].apply(simple_sentiment)
    h["importance_bin"]     = h[COL_IMPORTANCE_FR].apply(extract_oui_non)
    h["importance_sent"]    = h[COL_IMPORTANCE_FR].apply(simple_sentiment)
    h["hierarchie_fr"]      = h[COL_COMPARAISON_FR].str.lower().map(
        {k.lower(): v for k, v in IMPORTANCE_MAP.items()}).fillna(2.0)
    h["h2_target_motivation"] = h[COL_MOTIVATION].apply(motivation_label)
    txt_diff = (h[COL_DIFFICILE].fillna("") + " " +
                h[COL_DIFFICULTES].fillna("") + " " +
                h[COL_ORIGINE_DIFF].fillna(""))
    for label, kws in DIFFICULTE_KEYWORDS.items():
        h[f"diff_{label}"] = txt_diff.str.lower().apply(
            lambda t: int(any(k in t for k in kws)))
    return h

def build_h3(df: pd.DataFrame) -> pd.DataFrame:
    h = df.copy()
    h["exposition_freq"]        = h[COL_EXPOSITION].map(FREQ_MAP).fillna(0)
    h["interet_bin"]            = h[COL_INTERET_APPRENDRE].apply(extract_oui_non)
    h["interet_sent"]           = h[COL_INTERET_APPRENDRE].apply(simple_sentiment)
    h["perception_multi_sent"]  = h[COL_PERCEPTION_MULTI].apply(simple_sentiment)
    h["perception_multi_ord"]   = h[COL_PERCEPTION_MULTI].map(LIKERT_MAP).fillna(2.5)

    def attitude_score(row) -> float:
        comp = str(row.get(COL_COMPARAISON_FR, "")).lower()
        s1 = 3.0 if "plus" in comp else (2.0 if "autant" in comp else 1.0)
        s2 = 2.0 if str(row.get(COL_MOTIVATION_CAMERO, "")).upper().startswith("OUI") else 0.5
        return float(np.clip(1 + ((s1 / 3.0 * 3 + s2 / 2.0 * 2) / 5.0) * 4, 1.0, 5.0))

    h["h3_score_attitude"] = h.apply(attitude_score, axis=1)
    h["h3_attitude_class"] = h["h3_score_attitude"].apply(
        lambda s: "Positive" if s >= 3.5 else ("Neutre" if s >= 2.5 else "Négative"))
    return h

def build_h4(df: pd.DataFrame) -> pd.DataFrame:
    h = df.copy()
    h["interet_camarades_bin"]  = h[COL_INTERET_CAMARADES].apply(extract_oui_non)
    h["interet_camarades_sent"] = h[COL_INTERET_CAMARADES].apply(simple_sentiment)
    h["souhait_freq"]           = h[COL_INCLURE_LANGUES].map(FREQ_MAP).fillna(0)
    disc_list = h[COL_DISCIPLINE_ASSOC].apply(
        lambda x: extract_multilabels(x, DISCIPLINE_KEYWORDS))
    for disc in DISCIPLINE_KEYWORDS:
        h[f"vi_disc_{disc}"] = disc_list.apply(lambda l: int(disc in l))
        h[f"vd_disc_{disc}"] = h[f"vi_disc_{disc}"]
    h["h4_target_motivation"] = h[COL_MOTIVATION_CAMERO].apply(extract_oui_non).replace(-1, 0)

    def engagement(row) -> int:
        m = 2 if str(row.get(COL_MOTIVATION_CAMERO, "")).upper().startswith("OUI") else 0
        f = FREQ_MAP.get(str(row.get(COL_INCLURE_LANGUES, "")), 0)
        raw = m + (f / 4.0) * 2
        return 4 if raw >= 3.5 else (3 if raw >= 2.5 else (2 if raw >= 1.0 else 1))

    h["h4_engagement_score"] = h.apply(engagement, axis=1)
    return h


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    dcfg = cfg["data"]
    out  = Path(dcfg["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    log.info("=" * 55)
    log.info("FRENCH-LEARNING-PERCEPTIONS — PRÉTRAITEMENT")
    log.info("=" * 55)

    df = load_and_rename(dcfg["raw_path"])
    df = filter_consent(df, dcfg["consent_value"])
    df = anonymize(df)
    df = clean_demographics(df)
    df.to_csv(out / "french-learning-perceptions_clean.csv", index=False, encoding="utf-8-sig")
    log.info(f"Dataset nettoyé : {len(df)} répondants")

    for name, builder in [("h1", build_h1), ("h2", build_h2),
                           ("h3", build_h3), ("h4", build_h4)]:
        df_h = builder(df)
        df_h.to_csv(out / f"{name}_features.csv", index=False, encoding="utf-8-sig")
        log.info(f"{name.upper()} : {len(df_h)} lignes sauvegardées")

    log.info("Prétraitement terminé ✅")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    run(args.config)
