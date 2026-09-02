# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Preprocess & Feature Engineering on Delta Lake
# MAGIC
# MAGIC **French-Learning-Perceptions ML — Databricks Migration**
# MAGIC
# MAGIC This notebook replaces `src/preprocessing/preprocess.py` and runs on Databricks:
# MAGIC 1. Load raw data from Delta table
# MAGIC 2. Filter consent + anonymize
# MAGIC 3. Build features for H1, H2, H3, H4
# MAGIC 4. Write to Delta Lake (processed layer)
# MAGIC 5. Log statistics to Databricks MLflow

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import re
import unicodedata
import logging
import numpy as np
import pandas as pd
import mlflow
from pyspark.sql import functions as F
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

CATALOG = "flp_catalog"
log.info("✅ Imports OK")

def _step_log(name, ok, detail=""):
    try:
        spark.createDataFrame([(name, str(ok), str(detail)[:2000])],
                              ["step", "ok", "detail"]).write.format("delta").mode("append").saveAsTable(f"{CATALOG}.monitoring.run_log")
    except Exception:
        pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## Constants (from `src/utils/constants.py`)

# COMMAND ----------

FREQ_MAP = {
    "Toujours": 4, "toujours": 4, "Souvent": 3, "souvent": 3,
    "Parfois": 2, "parfois": 2, "Rarement": 1, "rarement": 1,
    "Jamais": 0, "jamais": 0,
}
LIKERT_MAP = {
    "Très bien": 4, "très bien": 4, "Plutôt bien": 3, "plutôt bien": 3,
    "Un peu bien": 2, "un peu bien": 2, "Pas du tout bien": 1, "pas du tout bien": 1,
}
IMPORTANCE_MAP = {
    "plus important": 3, "Plus important": 3,
    "autant important": 2, "Autant important": 2,
    "moins important": 1, "Moins important": 1,
}
RELATION_LM_MAP = {"Aisé": 2, "aisé": 2, "Complexe": 1, "complexe": 1}
PERCEPTION_LABELS = [
    "belle", "difficile", "importante", "utile", "compliquée",
    "intéressante", "ennuyeuse", "nécessaire", "riche", "obligatoire",
]
DIFFICULTE_KEYWORDS = {
    "grammaire": ["grammaire", "grammatical", "accord"],
    "vocabulaire": ["vocabulaire", "mots", "lexique"],
    "orthographe": ["orthographe", "fautes"],
    "conjugaison": ["conjugaison", "conjuguer", "verbes"],
    "expression_orale": ["expression orale", "oral", "parler"],
    "comprehension": ["compréhension", "comprehension", "comprendre", "compris", "lecture"],
    "analyse": ["analyse", "analyser"],
}
DISCIPLINE_KEYWORDS = {
    "vocabulaire": ["vocabulaire", "mots", "lexique"],
    "grammaire": ["grammaire"],
    "lecture": ["lecture", "lire", "textes", "récit"],
    "expression_orale": ["expression orale", "oral", "parler"],
    "conjugaison": ["conjugaison", "conjuguer"],
}
MOTIVATION_HIGH_KW = [
    "communication", "communiquer", "réussir", "avenir",
    "s'exprimer", "aider", "voyager", "études", "comprendre",
]
MOTIVATION_LOW_KW = ["obligé", "forcé", "difficile", "rien"]

COL_CONSENTEMENT      = "consentement"
COL_AGE               = "age"
COL_SEXE              = "sexe"
COL_REGION            = "region"
COL_LANGUES_PARLEES   = "langues_parlees"
COL_LANGUE_MATERNELLE = "langue_maternelle"
COL_USAGE_QUOTIDIEN   = "usage_quotidien"
COL_APPRENTISSAGE_ANT = "apprentissage_anterieur"
COL_RELATION_LM       = "relation_langue_mat"
COL_AVANTAGE_MULTI    = "avantage_plurilingue"
COL_PERCEPTION_FR     = "perception_francais"
COL_MOTS_ASSOCIES     = "mots_associes"
COL_MOTIVATION        = "motivation_apprendre"
COL_FACILE            = "aspects_faciles"
COL_DIFFICILE         = "aspects_difficiles"
COL_IMPORTANCE_FR     = "importance_francais"
COL_COMPARAISON_FR    = "comparaison_francais"
COL_EXPOSITION        = "exposition_autres_langues"
COL_INTERET_APPRENDRE = "interet_autres_langues"
COL_PERCEPTION_MULTI  = "perception_plurilinguisme"
COL_MOTIVATION_CAMERO = "motivation_camerounaises"
COL_PARLE_COURS       = "langues_parlees_cours"
COL_INTERET_CAMARADES = "interet_langues_camarades"
COL_DIFFICULTES       = "difficultes_principales"
COL_ORIGINE_DIFF      = "origine_difficultes"
COL_INCLURE_LANGUES   = "souhait_inclure_langues"
COL_DISCIPLINE_ASSOC  = "discipline_associee"

log.info("✅ Constants loaded")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Raw Data from Delta

# COMMAND ----------

df_raw = spark.read.table(f"{CATALOG}.raw.survey_responses").toPandas()
print(f"Loaded: {len(df_raw)} rows, {len(df_raw.columns)} columns")
df_raw.head(2)
_step_log("01_load", True, f"rows={len(df_raw)} cols={len(df_raw.columns)} cols={list(df_raw.columns)[:8]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Rename Columns, Filter Consent, Anonymize

# COMMAND ----------

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


def _normalize(s: str) -> str:
    s = unicodedata.normalize('NFD', s).encode('ascii', 'ignore').decode('ascii')
    return re.sub(r'[^a-z0-9]', '', s.lower())


def _rename(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    used_short = set()
    for col in df.columns:
        col_norm = _normalize(col)
        best_key, best_short, best_len = None, None, 0
        for key, short in CSV_MAP.items():
            key_norm = _normalize(key)
            if key_norm and key_norm in col_norm and len(key_norm) > best_len:
                best_key, best_short, best_len = key, short, len(key_norm)
        if best_short:
            if best_short in used_short:
                log.info(f"Skipping duplicate rename: '{col}' -> {best_short} already mapped")
                continue
            used_short.add(best_short)
            rename[col] = best_short
    log.info(f"Renamed columns: {list(rename.values())}")
    return df.rename(columns=rename)


df = _rename(df_raw)

# Filter consent
mask = df[COL_CONSENTEMENT].str.lower().str.contains("accepte", na=False)
df = df[mask].copy().reset_index(drop=True)
log.info(f"After consent filter: {len(df)} respondents")
_step_log("01_consent", True, f"rows_after_consent={len(df)}")

# Anonymize
drop_cols = [c for c in df.columns if any(
    kw in c.lower() for kw in ["horodateur", "timestamp", "email"]
)]
df = df.drop(columns=drop_cols, errors="ignore")
log.info(f"Anonymized: dropped {drop_cols}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Demographics

# COMMAND ----------

def _parse_age(val) -> float:
    if pd.isna(val): return np.nan
    m = re.search(r"\d+", str(val))
    return float(m.group()) if m else np.nan

if COL_AGE in df.columns:
    df[COL_AGE] = df[COL_AGE].apply(_parse_age)
    df[COL_AGE] = df[COL_AGE].fillna(df[COL_AGE].median())
_step_log("01_demo_age", True, str(df[COL_AGE].head(2).tolist()) if COL_AGE in df.columns else "no age col")

if COL_SEXE in df.columns:
    df["sexe_bin"] = df[COL_SEXE].str.upper().str.strip().map({"M": 1, "F": 0})
_step_log("01_demo_sexe", True, str(df[COL_SEXE].head(2).tolist()) if COL_SEXE in df.columns else "no sexe col")

if COL_REGION in df.columns:
    df[COL_REGION] = df[COL_REGION].str.strip().str.title()
    dummies = pd.get_dummies(df[COL_REGION], prefix="region")
    df = pd.concat([df, dummies], axis=1)
_step_log("01_demo_region", True, str(df[COL_REGION].head(2).tolist()) if COL_REGION in df.columns else "no region col")

log.info("✅ Demographics cleaned")
df[[COL_AGE, COL_SEXE, "sexe_bin", COL_REGION]].head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Helper Functions

# COMMAND ----------

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

log.info("✅ Helper functions ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Build Features per Hypothesis

# COMMAND ----------

# MAGIC %md
# MAGIC ### H1 — Multilingual Repertoire → Daily Mobilization

# COMMAND ----------

h1 = df.copy()
try:
    h1["nb_langues"]            = h1[COL_LANGUES_PARLEES].apply(count_languages)
    h1["apprent_anterieur_bin"] = h1[COL_APPRENTISSAGE_ANT].apply(extract_oui_non)
    h1["relation_lm_ord"]       = h1[COL_RELATION_LM].map(RELATION_LM_MAP).fillna(0)
    h1["domaine_usage_freq"]    = h1[COL_EXPOSITION].map(FREQ_MAP).fillna(0)
    h1["valorisation_sent"]     = h1[COL_AVANTAGE_MULTI].apply(simple_sentiment)
except Exception as e:
    _step_log("01_h1_ERR1", False, repr(e))
    raise

# One-hot langue maternelle (top 10)
if COL_LANGUE_MATERNELLE in h1.columns:
    top_lm = h1[COL_LANGUE_MATERNELLE].value_counts().head(10).index
    for lm in top_lm:
        h1[f"lm_{lm.lower().replace(' ', '_')}"] = (
            h1[COL_LANGUE_MATERNELLE].str.lower() == lm.lower()
        ).astype(int)
_step_log("01_h1_lm", True, str(list(top_lm)[:5]))

h1["h1_target"] = h1[COL_USAGE_QUOTIDIEN].apply(extract_oui_non)
h1 = h1[h1["h1_target"].isin([0, 1])].copy()
log.info(f"H1: {len(h1)} rows | balance={h1['h1_target'].value_counts().to_dict()}")
_step_log("01_h1", True, f"rows={len(h1)} balance={h1['h1_target'].value_counts().to_dict()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### H2 — French Representations → Motivation & Difficulties

# COMMAND ----------

h2 = df.copy()
for label in PERCEPTION_LABELS:
    h2[f"perc_{label}"] = h2[COL_PERCEPTION_FR].str.lower().str.contains(label, na=False).astype(int)
h2["mots_assoc_sent"] = h2[COL_MOTS_ASSOCIES].apply(simple_sentiment)
h2["importance_bin"]  = h2[COL_IMPORTANCE_FR].apply(extract_oui_non)
h2["importance_sent"] = h2[COL_IMPORTANCE_FR].apply(simple_sentiment)
h2["hierarchie_fr"]   = h2[COL_COMPARAISON_FR].str.lower().map(
    {k.lower(): v for k, v in IMPORTANCE_MAP.items()}).fillna(2.0)

def _motiv(text):
    if pd.isna(text): return 1
    t = str(text).lower()
    if sum(1 for w in MOTIVATION_HIGH_KW if w in t) >= 2: return 2
    if sum(1 for w in MOTIVATION_LOW_KW if w in t) >= 1: return 0
    return 1

h2["h2_target_motivation"] = h2[COL_MOTIVATION].apply(_motiv)

def _diff(row):
    txt = " ".join([str(row.get(COL_DIFFICILE, "")),
                    str(row.get(COL_DIFFICULTES, "")),
                    str(row.get(COL_ORIGINE_DIFF, ""))])
    lbls = extract_multilabels(txt, DIFFICULTE_KEYWORDS)
    return {f"diff_{k}": int(k in lbls) for k in DIFFICULTE_KEYWORDS}

h2 = pd.concat([h2, h2.apply(_diff, axis=1).apply(pd.Series)], axis=1)
log.info(f"H2: {len(h2)} rows | motivation={h2['h2_target_motivation'].value_counts().to_dict()}")
_step_log("01_h2", True, f"rows={len(h2)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### H3 — Plurilingual Exposure → Attitudes

# COMMAND ----------

h3 = df.copy()
h3["exposition_freq"]       = h3[COL_EXPOSITION].map(FREQ_MAP).fillna(0)
h3["interet_bin"]           = h3[COL_INTERET_APPRENDRE].apply(extract_oui_non)
h3["interet_sent"]          = h3[COL_INTERET_APPRENDRE].apply(simple_sentiment)
h3["perception_multi_sent"] = h3[COL_PERCEPTION_MULTI].apply(simple_sentiment)
h3["perception_multi_ord"]  = h3[COL_PERCEPTION_MULTI].map(LIKERT_MAP).fillna(2.5)

def _score(row):
    comp = str(row.get(COL_COMPARAISON_FR, "")).lower()
    s1   = 3.0 if "plus" in comp else (2.0 if "autant" in comp else 1.0)
    s2   = 2.0 if str(row.get(COL_MOTIVATION_CAMERO, "")).upper().startswith("OUI") else 0.5
    return float(np.clip(1 + ((s1/3)*3 + (s2/2)*2) / 5 * 4, 1.0, 5.0))

h3["h3_score_attitude"] = h3.apply(_score, axis=1)
h3["h3_attitude_class"] = h3["h3_score_attitude"].apply(
    lambda s: "Positive" if s >= 3.5 else ("Neutre" if s >= 2.5 else "Négative"))
log.info(f"H3: {len(h3)} | score_mean={h3['h3_score_attitude'].mean():.2f}")
_step_log("01_h3", True, f"rows={len(h3)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### H4 — Local Language Integration → Engagement

# COMMAND ----------

h4 = df.copy()
h4["interet_camarades_bin"]  = h4[COL_INTERET_CAMARADES].apply(extract_oui_non)
h4["interet_camarades_sent"] = h4[COL_INTERET_CAMARADES].apply(simple_sentiment)
h4["souhait_freq"]           = h4[COL_INCLURE_LANGUES].map(FREQ_MAP).fillna(0)

disc_list = h4[COL_DISCIPLINE_ASSOC].apply(
    lambda x: extract_multilabels(x, DISCIPLINE_KEYWORDS))
for disc in DISCIPLINE_KEYWORDS:
    h4[f"vi_disc_{disc}"] = disc_list.apply(lambda l: int(disc in l))
    h4[f"vd_disc_{disc}"] = h4[f"vi_disc_{disc}"]

h4["h4_target_motivation"] = h4[COL_MOTIVATION_CAMERO].apply(
    lambda x: 1 if LIKERT_MAP.get(str(x).strip(), 0) >= 3 else 0)

def _eng(row):
    mp  = 2 if str(row.get(COL_MOTIVATION_CAMERO, "")).upper().startswith("OUI") else 0
    fp  = FREQ_MAP.get(str(row.get(COL_INCLURE_LANGUES, "")), 0)
    raw = mp + (fp / 4.0) * 2
    return 4 if raw >= 3.5 else (3 if raw >= 2.5 else (2 if raw >= 1.0 else 1))

h4["h4_engagement_score"] = h4.apply(_eng, axis=1)
log.info(f"H4: {len(h4)} | motivation={h4['h4_target_motivation'].value_counts().to_dict()}")
_step_log("01_h4", True, f"rows={len(h4)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Write to Delta Lake (Processed Layer)

# COMMAND ----------

def _sanitize_col(name):
    return re.sub(r'[\s,;{}()\n\t=/]+', '_', str(name)).strip('_') or "col"

datasets = {"H1": h1, "H2": h2, "H3": h3, "H4": h4}

for name, data in datasets.items():
    data = data.rename(columns=_sanitize_col)
    spark_df = spark.createDataFrame(data)
    table_name = f"{CATALOG}.processed.{name.lower()}_features"
    (
        spark_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(table_name)
    )
    log.info(f"✅ Delta table: {table_name} — {len(data)} rows")
    _step_log(f"01_write_{name.lower()}", True, f"rows={len(data)} cols={len(data.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Log to MLflow

# COMMAND ----------

with mlflow.start_run(run_name="databricks_preprocess"):
    mlflow.set_tag("step", "preprocess")
    mlflow.set_tag("platform", "databricks")
    mlflow.log_params({
        "n_total": len(df),
        "n_columns": len(df.columns),
        "source": f"{CATALOG}.raw.survey_responses",
    })
    for name, data in datasets.items():
        mlflow.log_metric(f"n_{name.lower()}", len(data))
    mlflow.log_metric("n_raw_respondents", len(df))

print("✅ MLflow run logged — check Databricks Experiments UI")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Validate Delta Tables

# COMMAND ----------

for name in ["h1", "h2", "h3", "h4"]:
    tbl = spark.read.table(f"{CATALOG}.processed.{name}_features")
    print(f"{name.upper()}_features: {tbl.count()} rows, {len(tbl.columns)} columns")

# COMMAND ----------

print("=" * 60)
print("  PREPROCESSING COMPLETE — Delta Lake (processed layer)")
print("=" * 60)
print("  Next: Run notebook 02_train_h1.py")
