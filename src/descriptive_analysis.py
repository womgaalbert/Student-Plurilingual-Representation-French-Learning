"""
descriptive_analysis.py -- French-Learning-Perceptions ML
Pipeline descriptif : Phase 1 (Cleaning/Missing), Phase 2 (EDA), Phase 3 (CamemBERT/A Priori/N-grams)
+ Stage global demographics.

Usage:
  python src/descriptive_analysis.py --stage demographics --config params.yaml
  python src/descriptive_analysis.py --hypothesis H1 --config params.yaml
  python src/descriptive_analysis.py --hypothesis all --config params.yaml
"""
import argparse
import logging
import re
import unicodedata
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import spacy
import torch
import mlflow
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from umap import UMAP
from wordcloud import WordCloud

from utils.config import load_config
from utils.constants import (
    COL_CONSENTEMENT, COL_AGE, COL_SEXE, COL_REGION, COL_CLASSE, COL_ETABLISSEMENT,
    COL_LANGUES_PARLEES, COL_LANGUE_MATERNELLE,
    COL_AVANTAGE_MULTI, COL_RELATION_LM,
    COL_PERCEPTION_FR, COL_MOTS_ASSOCIES, COL_MOTIVATION,
    COL_FACILE, COL_DIFFICILE, COL_IMPORTANCE_FR, COL_COMPARAISON_FR,
    COL_EXPOSITION, COL_INTERET_APPRENDRE, COL_PERCEPTION_MULTI,
    COL_MOTIVATION_CAMERO, COL_PARLE_COURS,
    COL_INTERET_CAMARADES, COL_DIFFICULTES, COL_ORIGINE_DIFF,
    COL_INCLURE_LANGUES, COL_DISCIPLINE_ASSOC,
    COL_USAGE_QUOTIDIEN, COL_APPRENTISSAGE_ANT,
    CSV_COLUMN_MAP,
)
from utils.mlflow_utils import setup_mlflow, log_metrics

warnings.filterwarnings("ignore")

Path("logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/descriptive_analysis.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ── Hypothesis column registry ────────────────────────────────────────────────

HYP_COLS: dict[str, dict] = {
    "H1": {
        "label":       "Repertoire Multilingue & Mobilisation",
        "text":        [COL_AVANTAGE_MULTI, COL_APPRENTISSAGE_ANT],
        "ordinal":     [COL_EXPOSITION, COL_RELATION_LM],
        "categorical": [COL_LANGUE_MATERNELLE, COL_LANGUES_PARLEES, COL_USAGE_QUOTIDIEN],
        "all":         [COL_AVANTAGE_MULTI, COL_APPRENTISSAGE_ANT, COL_EXPOSITION,
                        COL_RELATION_LM, COL_LANGUE_MATERNELLE, COL_LANGUES_PARLEES,
                        COL_USAGE_QUOTIDIEN],
        "target":      COL_USAGE_QUOTIDIEN,
    },
    "H2": {
        "label":       "Representations du Francais → Motivation & Difficultes",
        "text":        [COL_PERCEPTION_FR, COL_MOTS_ASSOCIES, COL_MOTIVATION,
                        COL_FACILE, COL_DIFFICILE, COL_IMPORTANCE_FR,
                        COL_DIFFICULTES, COL_ORIGINE_DIFF],
        "ordinal":     [COL_COMPARAISON_FR],
        "categorical": [COL_IMPORTANCE_FR, COL_PERCEPTION_FR, COL_MOTIVATION],
        "all":         [COL_PERCEPTION_FR, COL_MOTS_ASSOCIES, COL_MOTIVATION,
                        COL_FACILE, COL_DIFFICILE, COL_IMPORTANCE_FR, COL_COMPARAISON_FR,
                        COL_DIFFICULTES, COL_ORIGINE_DIFF],
        "target":      COL_MOTIVATION,
    },
    "H3": {
        "label":       "Exposition Plurilingue → Attitudes envers le Francais",
        "text":        [COL_INTERET_APPRENDRE, COL_PERCEPTION_MULTI, COL_PARLE_COURS],
        "ordinal":     [COL_EXPOSITION, COL_PERCEPTION_MULTI, COL_COMPARAISON_FR],
        "categorical": [COL_INTERET_APPRENDRE, COL_MOTIVATION_CAMERO],
        "all":         [COL_INTERET_APPRENDRE, COL_PERCEPTION_MULTI, COL_PARLE_COURS,
                        COL_EXPOSITION, COL_COMPARAISON_FR, COL_MOTIVATION_CAMERO],
        "target":      COL_INTERET_APPRENDRE,
    },
    "H4": {
        "label":       "Integration Langues Locales → Engagement & Motivation",
        "text":        [COL_DIFFICULTES, COL_ORIGINE_DIFF, COL_INTERET_CAMARADES],
        "ordinal":     [COL_INCLURE_LANGUES],
        "categorical": [COL_INTERET_CAMARADES, COL_MOTIVATION_CAMERO, COL_DISCIPLINE_ASSOC],
        "all":         [COL_DIFFICULTES, COL_ORIGINE_DIFF, COL_INTERET_CAMARADES,
                        COL_INCLURE_LANGUES, COL_MOTIVATION_CAMERO, COL_DISCIPLINE_ASSOC],
        "target":      COL_MOTIVATION_CAMERO,
    },
}

STEREOTYPE_MARKERS: dict[str, list[str]] = {
    "distanciation_identitaire": ["langue des colons", "langue des blancs", "langue etrangere"],
    "preconstruit_difficulte":   ["trop difficile", "trop dur", "tres difficile"],
    "preconstruit_inutilite":    ["inutile dans ma vie", "pas utile", "sert a rien"],
    "auto_devalorisation":       ["nul en francais", "mauvais en francais", "pas bon"],
    "exclusion_symbolique":      ["pas pour nous", "pas pour moi", "pas notre langue"],
    "contrainte_resistance":     ["obliges d apprendre", "pas le choix", "forces a"],
}

APRIORI_CATEGORIES: dict[str, str] = {
    "representation_utilitaire":       "necessite reussir travail utile important avenir",
    "representation_identitaire":      "langue des autres pas la mienne etrangers colons",
    "representation_affective":        "j aime j ai peur difficile peur passion belle",
    "representation_institutionnelle": "ecole professeur obligatoire cours classe regle",
    "resistance_contrainte":           "force oblige pas le choix contraint impose",
}


# ── HTML template ─────────────────────────────────────────────────────────────

_CSS = """<style>
body{font-family:Arial,sans-serif;margin:32px;color:#222;max-width:1100px}
h1{color:#2c5282;border-bottom:2px solid #2c5282;padding-bottom:8px}
h2{color:#2d3748;margin-top:32px;border-left:4px solid #4299e1;padding-left:10px}
h3{color:#4a5568;margin-top:20px}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13px}
th{background:#2c5282;color:#fff;padding:8px 12px;text-align:left}
td{padding:6px 12px;border-bottom:1px solid #e2e8f0}
tr:nth-child(even){background:#f7fafc}
.warn{color:#c05621;font-weight:bold}
.ok{color:#276749;font-weight:bold}
img{max-width:750px;display:block;margin:14px 0;border:1px solid #cbd5e0;border-radius:4px}
.note{background:#ebf8ff;border-left:4px solid #4299e1;padding:10px 14px;margin:12px 0;font-size:13px}
</style>"""


def _html_page(title: str, body: str) -> str:
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{title}</title>{_CSS}</head><body>{body}</body></html>"


# ── Data loading ──────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\x00-\x7f]", "", s)
    for ch in ("\x27", "\x22", "\x60", "\x2d"):
        s = s.replace(ch, " " if ch == "\x2d" else "")
    return re.sub(r"\s+", " ", s)


def load_raw(raw_path: str, consent_value: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    sorted_map = sorted(CSV_COLUMN_MAP.items(), key=lambda kv: len(kv[0]), reverse=True)
    rename = {}
    used_shorts: set = set()
    for col in df.columns:
        col_n = _norm(col)
        for csv_name, short in sorted_map:
            if short in used_shorts:
                continue
            csv_n = _norm(csv_name)
            if not col_n or not csv_n:
                continue
            ratio = min(len(col_n), len(csv_n)) / max(len(col_n), len(csv_n))
            if ratio >= 0.50 and (csv_n in col_n or col_n in csv_n):
                rename[col] = short
                used_shorts.add(short)
                break
    df = df.rename(columns=rename)
    if COL_CONSENTEMENT in df.columns:
        mask = df[COL_CONSENTEMENT].str.lower().str.contains(consent_value, na=False)
        before = len(df)
        df = df[mask].reset_index(drop=True)
        log.info(f"Consentement : {len(df)}/{before} repondants valides")
    drop_cols = [c for c in df.columns
                 if any(k in c.lower() for k in ["horodateur", "timestamp", "email", "nom", "prenom"])]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")
    return df


# ── Phase 1 — Missing values & imputation ────────────────────────────────────

def _col_type(col: str, hyp_cfg: dict) -> str:
    if col in hyp_cfg.get("text", []):    return "texte"
    if col in hyp_cfg.get("ordinal", []): return "ordinale"
    return "categorielle"


def analyze_missing(df: pd.DataFrame, cols: list[str], hyp_cfg: dict) -> pd.DataFrame:
    rows = []
    n = len(df)
    for col in cols:
        if col not in df.columns:
            continue
        n_miss = df[col].isna().sum()
        pct    = n_miss / n * 100
        typ    = _col_type(col, hyp_cfg)
        if typ == "texte":
            approach = "chaine vide"
        elif pct > 30:
            approach = "EXCLURE (>30%)"
        elif typ == "ordinale":
            approach = "mediane" if pct < 5 else "KNNImputer (k=5)"
        else:
            approach = "mode" if pct < 5 else "Non-reponse"
        rows.append({"variable": col, "type": typ, "n_manquants": n_miss,
                     "pct_manquants": round(pct, 1), "approche": approach})
    return pd.DataFrame(rows)


def apply_imputation(df: pd.DataFrame, missing_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    exclude = missing_df.loc[missing_df["approche"] == "EXCLURE (>30%)", "variable"].tolist()
    if exclude:
        log.warning(f"Colonnes exclues (>30% manquants) : {exclude}")
        df.drop(columns=exclude, inplace=True, errors="ignore")

    for _, row in missing_df.iterrows():
        col = row["variable"]
        if col not in df.columns or row["n_manquants"] == 0:
            continue
        approach = row["approche"]
        if approach == "chaine vide":
            df[col] = df[col].fillna("")
        elif approach == "mode":
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val.iloc[0] if len(mode_val) else "Inconnu")
        elif approach == "mediane":
            num = pd.to_numeric(df[col], errors="coerce")
            df[col] = num.fillna(num.median())
        elif approach == "KNNImputer (k=5)":
            num = pd.to_numeric(df[col], errors="coerce").values.reshape(-1, 1)
            imputed = KNNImputer(n_neighbors=5).fit_transform(num)
            df[col] = imputed.ravel()
        elif approach == "Non-reponse":
            df[col] = df[col].fillna("Non-reponse")
    return df


# ── Phase 2 — EDA ─────────────────────────────────────────────────────────────

def compute_stats(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    numeric_cols = [c for c in cols if c in df.columns and
                    pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce"))]
    if not numeric_cols:
        return pd.DataFrame()
    num_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return num_df.describe(percentiles=[0.25, 0.5, 0.75]).T.round(3)


def detect_outliers_iqr(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((series < low) | (series > high)).sum()
        rows.append({"variable": col, "Q1": round(q1, 3), "Q3": round(q3, 3),
                     "IQR": round(iqr, 3), "borne_basse": round(low, 3),
                     "borne_haute": round(high, 3), "n_outliers": int(n_out),
                     "pct_outliers": round(n_out / len(series) * 100, 1)})
    return pd.DataFrame(rows)


def plot_distributions(df: pd.DataFrame, cols: list[str], out_path: str) -> str:
    plot_cols = [c for c in cols if c in df.columns][:12]
    if not plot_cols:
        return out_path
    n = len(plot_cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).ravel() if n > 1 else [axes]
    for i, col in enumerate(plot_cols):
        ax = axes[i]
        series = pd.to_numeric(df[col], errors="coerce")
        counts = df[col].value_counts().head(15)
        if series.notna().sum() > 5 and series.nunique() > 2:
            series.dropna().plot(kind="hist", ax=ax, bins=20, color="steelblue", edgecolor="white")
        elif not counts.empty:
            counts.plot(kind="bar", ax=ax, color="steelblue")
        else:
            ax.text(0.5, 0.5, "Aucune donnee", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(col[:30], fontsize=9)
        ax.tick_params(axis="x", rotation=35, labelsize=7)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    log.info(f"Saved: {out_path}")
    return out_path


def plot_correlation(df: pd.DataFrame, cols: list[str], out_path: str) -> str:
    num_cols = [c for c in cols if c in df.columns and
                pd.to_numeric(df[c], errors="coerce").notna().sum() > 10]
    if len(num_cols) < 2:
        return out_path
    num_df = df[num_cols].apply(pd.to_numeric, errors="coerce")
    # Spearman for ordinal/non-normal
    corr = num_df.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(max(6, len(num_cols)), max(5, len(num_cols) - 1)))
    sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5)
    ax.set_title("Matrice de correlation (Spearman)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    log.info(f"Saved: {out_path}")
    return out_path


def plot_target_distribution(df: pd.DataFrame, target_col: str, out_path: str) -> str:
    if target_col not in df.columns:
        return out_path
    fig, ax = plt.subplots(figsize=(7, 4))
    df[target_col].value_counts().plot(kind="bar", ax=ax, color="coral", edgecolor="white")
    ax.set_title(f"Distribution de la target : {target_col}", fontsize=11)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    log.info(f"Saved: {out_path}")
    return out_path


def generate_eda_html(
    hyp: str, label: str, missing_df: pd.DataFrame, stats_df: pd.DataFrame,
    outlier_df: pd.DataFrame, n_total: int, plot_dir: Path,
) -> str:
    def _table(df: pd.DataFrame) -> str:
        if df.empty:
            return "<p><em>Aucune donnee numerique disponible.</em></p>"
        return df.to_html(classes="", border=0, index=True)

    def _img(name: str) -> str:
        p = plot_dir / name
        if p.exists():
            return f"<img src='{name}' alt='{name}'>"
        return ""

    miss_rows = ""
    for _, r in missing_df.iterrows():
        warn = ' class="warn"' if r["pct_manquants"] > 30 else (
               ' class="warn"' if r["pct_manquants"] >= 5 else "")
        miss_rows += (
            f"<tr><td>{r['variable']}</td><td>{r['type']}</td>"
            f"<td{warn}>{r['n_manquants']}</td><td{warn}>{r['pct_manquants']}%</td>"
            f"<td>{r['approche']}</td></tr>"
        )

    body = f"""
<h1>{hyp} — {label}</h1>
<div class='note'>N repondants valides (apres consentement) : <strong>{n_total}</strong></div>

<h2>Phase 1 — Valeurs manquantes & imputation</h2>
<table>
  <tr><th>Variable</th><th>Type</th><th>N manquants</th><th>% manquants</th><th>Approche retenue</th></tr>
  {miss_rows}
</table>

<h2>Phase 2 — Statistiques descriptives</h2>
{_table(stats_df)}

<h2>Distributions des variables</h2>
{_img("eda_distributions.png")}

<h2>Matrice de correlation (Spearman)</h2>
{_img("eda_correlation.png")}

<h2>Distribution de la target</h2>
{_img("eda_target.png")}

<h2>Detection outliers (methode IQR)</h2>
{_table(outlier_df)}
"""
    return _html_page(f"EDA — {hyp}", body)


# ── CamemBERT ─────────────────────────────────────────────────────────────────

def load_camembert(model_name: str, device: str) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def encode_texts(
    texts: list[str], tokenizer, model,
    batch_size: int, max_length: int, device: str,
) -> np.ndarray:
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        all_embs.append(emb.cpu().numpy())
    return np.vstack(all_embs)


# ── Phase 3 — Text analysis helpers ──────────────────────────────────────────

def _lemmatize(texts: pd.Series, nlp) -> list[str]:
    return [
        " ".join(
            tok.lemma_.lower() for tok in nlp(str(t))
            if not tok.is_stop and not tok.is_punct and len(tok.text) >= 3
        )
        for t in texts.dropna()
    ]


def detect_stereotypes_keyword(df: pd.DataFrame, text_cols: list[str]) -> pd.DataFrame:
    combined = df[text_cols].fillna("").apply(
        lambda r: " ".join(r.values), axis=1
    ).str.lower()
    for label, markers in STEREOTYPE_MARKERS.items():
        df[f"stereo_{label}"] = combined.apply(lambda t: int(any(m in t for m in markers)))
    return df


def detect_stereotypes_embedding(
    corpus_embs: np.ndarray, marker_embs: np.ndarray,
    marker_labels: list[str], threshold: float,
) -> pd.DataFrame:
    sims = cosine_similarity(corpus_embs, marker_embs)
    return pd.DataFrame(
        (sims >= threshold).astype(int),
        columns=[f"emb_stereo_{l}" for l in marker_labels],
    )


def classify_apriori(
    corpus_embs: np.ndarray, cat_embs: np.ndarray,
    cat_labels: list[str], threshold: float,
) -> pd.DataFrame:
    sims = cosine_similarity(corpus_embs, cat_embs)
    dominant = np.argmax(sims, axis=1)
    rows = [{f"apriori_{l}": float(sims[i, j]) for j, l in enumerate(cat_labels)} |
            {"apriori_dominant": cat_labels[dominant[i]]
             if sims[i, dominant[i]] >= threshold else "non_classe"}
            for i in range(len(sims))]
    return pd.DataFrame(rows)


def text_stats(df: pd.DataFrame, text_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in text_cols:
        if col not in df.columns:
            continue
        series = df[col].dropna().astype(str)
        lengths = series.str.split().str.len()
        tokens  = series.str.lower().str.split().explode()
        richness = tokens.nunique() / max(len(tokens), 1)
        rows.append({
            "colonne": col,
            "n_reponses": len(series),
            "longueur_moy": round(lengths.mean(), 1),
            "longueur_med": round(lengths.median(), 1),
            "longueur_max": int(lengths.max()),
            "richesse_lexicale": round(richness, 3),
        })
    return pd.DataFrame(rows)


def build_ngram_table(texts: pd.Series, nlp, n: int, top_k: int, min_df: int) -> pd.DataFrame:
    cleaned = _lemmatize(texts, nlp)
    if not cleaned:
        return pd.DataFrame(columns=["ngram", "count", "tfidf"])
    try:
        cv  = CountVectorizer(ngram_range=(n, n), min_df=min_df)
        tv  = TfidfVectorizer(ngram_range=(n, n), min_df=min_df)
        Xc  = cv.fit_transform(cleaned)
        Xt  = tv.fit_transform(cleaned)
    except ValueError:
        return pd.DataFrame(columns=["ngram", "count", "tfidf"])
    return (pd.DataFrame({
        "ngram": cv.get_feature_names_out(),
        "count": np.asarray(Xc.sum(0)).ravel(),
        "tfidf": np.asarray(Xt.mean(0)).ravel(),
    }).nlargest(top_k, "count").reset_index(drop=True))


# ── Phase 3 — Visualizations ──────────────────────────────────────────────────

def plot_wordcloud(text: str, title: str, out_path: str, nlp) -> str:
    tokens = " ".join(
        tok.lemma_.lower() for tok in nlp(text)
        if not tok.is_stop and not tok.is_punct and len(tok.text) >= 3
    )
    wc = WordCloud(width=900, height=460, background_color="white",
                   collocations=False, max_words=100).generate(tokens or "vide")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    log.info(f"Saved: {out_path}")
    return out_path


def plot_umap_scatter(
    embs: np.ndarray, labels: pd.Series, title: str, out_path: str, umap_cfg: dict,
) -> str:
    reducer = UMAP(n_components=2, n_neighbors=umap_cfg["n_neighbors"],
                   min_dist=umap_cfg["min_dist"], random_state=umap_cfg["random_state"])
    coords = reducer.fit_transform(embs)
    df_p = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "lbl": labels.values})
    fig, ax = plt.subplots(figsize=(9, 6))
    for lbl, grp in df_p.groupby("lbl"):
        ax.scatter(grp["x"], grp["y"], label=str(lbl), alpha=0.55, s=18)
    ax.legend(fontsize=8, markerscale=2, loc="best")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    log.info(f"Saved: {out_path}")
    return out_path


def plot_ngrams(table: pd.DataFrame, title: str, out_path: str) -> str:
    if table.empty:
        return out_path
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(table["ngram"], table["count"], color="steelblue")
    ax.invert_yaxis()
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Frequence")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    log.info(f"Saved: {out_path}")
    return out_path


def plot_cooccurrence_heatmap(
    sentences: list[str], out_path: str, min_df: int,
) -> str:
    if not sentences:
        return out_path
    try:
        vec = CountVectorizer(max_features=30, min_df=min_df)
        X   = vec.fit_transform(sentences)
    except ValueError:
        return out_path
    cooc = (X.T @ X).toarray().astype(float)
    np.fill_diagonal(cooc, 0)
    terms = vec.get_feature_names_out()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(pd.DataFrame(cooc, index=terms, columns=terms),
                ax=ax, cmap="YlOrRd", xticklabels=True, yticklabels=True)
    ax.set_title("Heatmap de co-occurrence des termes cles")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    log.info(f"Saved: {out_path}")
    return out_path


def plot_cooccurrence_network(
    sentences: list[str], out_path: str, min_df: int,
) -> str:
    if not sentences:
        return out_path
    try:
        vec = CountVectorizer(max_features=25, min_df=min_df)
        X   = vec.fit_transform(sentences)
    except ValueError:
        return out_path
    cooc = (X.T @ X).toarray()
    np.fill_diagonal(cooc, 0)
    terms = vec.get_feature_names_out()
    G = nx.Graph()
    for i, t1 in enumerate(terms):
        for j, t2 in enumerate(terms):
            if i < j and cooc[i, j] > 0:
                G.add_edge(t1, t2, weight=float(cooc[i, j]))
    top_edges = sorted(G.edges(data=True), key=lambda e: e[2]["weight"], reverse=True)[:40]
    G2 = nx.Graph()
    for u, v, d in top_edges:
        G2.add_edge(u, v, weight=d["weight"])
    fig, ax = plt.subplots(figsize=(12, 9))
    pos = nx.spring_layout(G2, seed=42)
    ws  = [G2[u][v]["weight"] for u, v in G2.edges()]
    mw  = max(ws) if ws else 1.0
    nx.draw_networkx(G2, pos=pos, ax=ax, node_size=550, node_color="lightblue",
                     font_size=8, alpha=0.85, width=[1 + 4 * (w / mw) for w in ws])
    ax.set_title("Reseau de co-occurrences des termes cles")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    log.info(f"Saved: {out_path}")
    return out_path


def plot_apriori_distribution(apriori_df: pd.DataFrame, out_path: str) -> str:
    counts = apriori_df["apriori_dominant"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    counts.plot(kind="bar", ax=ax, color="coral", edgecolor="white")
    ax.set_title("Distribution categories A Priori (Moscovici & Jodelet)", fontsize=11)
    ax.set_ylabel("N repondants")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    log.info(f"Saved: {out_path}")
    return out_path


def plot_stereotype_distribution(df: pd.DataFrame, out_path: str) -> str:
    stereo_cols = [c for c in df.columns if c.startswith("stereo_")]
    if not stereo_cols:
        return out_path
    totals = df[stereo_cols].sum().sort_values(ascending=False)
    totals.index = [c.replace("stereo_", "").replace("_", " ") for c in totals.index]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    totals.plot(kind="bar", ax=ax, color="tomato", edgecolor="white")
    ax.set_title("Detection stereotypes & preconstruits", fontsize=11)
    ax.set_ylabel("N repondants")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    log.info(f"Saved: {out_path}")
    return out_path


def generate_phase3_html(
    hyp: str, label: str, text_stats_df: pd.DataFrame,
    apriori_df: pd.DataFrame, desc_dir: Path,
) -> str:
    def _img(name: str) -> str:
        p = desc_dir / name
        return f"<img src='{name}' alt='{name}'>" if p.exists() else ""

    cat_imgs = "".join(
        f"<h3>{cat}</h3>{_img(f'wordcloud_{cat}.png')}"
        for cat in APRIORI_CATEGORIES
    )
    dom_counts = apriori_df["apriori_dominant"].value_counts().to_frame("n").to_html(border=0)

    body = f"""
<h1>{hyp} — Phase 3 : Analyse Descriptive Textuelle</h1>
<p><em>{label}</em></p>

<h2>Statistiques textuelles</h2>
{text_stats_df.to_html(border=0, index=False) if not text_stats_df.empty else "<p>N/A</p>"}

<h2>Nuage de mots global</h2>
{_img("wordcloud_global.png")}

<h2>Nuages de mots par categorie A Priori</h2>
{cat_imgs}

<h2>UMAP — Stereotypes detectes</h2>
{_img("umap_stereotypes.png")}

<h2>UMAP — Categories A Priori</h2>
{_img("umap_apriori.png")}

<h2>UMAP — Clusters K-Means</h2>
{_img("umap_clusters.png")}

<h2>Distribution categories A Priori</h2>
{dom_counts}
{_img("apriori_distribution.png")}

<h2>Stereotypes & preconstruits</h2>
{_img("stereotype_distribution.png")}

<h2>N-grammes — Unigrammes</h2>
{_img("ngrams_unigrammes.png")}
<h2>N-grammes — Bigrammes</h2>
{_img("ngrams_bigrammes.png")}
<h2>N-grammes — Trigrammes</h2>
{_img("ngrams_trigrammes.png")}

<h2>Heatmap co-occurrence</h2>
{_img("heatmap_cooccurrence.png")}

<h2>Reseau co-occurrence</h2>
{_img("network_cooccurrence.png")}
"""
    return _html_page(f"Phase 3 — {hyp}", body)


# ── Demographics stage ────────────────────────────────────────────────────────

def run_demographics(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("--- Demographics globale ---")
    saved = []

    demo_pairs = [
        (COL_AGE,           "Distribution Age"),
        (COL_CLASSE,        "Niveau scolaire"),
        (COL_REGION,        "Region d'origine"),
        (COL_SEXE,          "Genre"),
        (COL_ETABLISSEMENT, "Etablissement"),
        (COL_LANGUE_MATERNELLE, "Langue maternelle (Top 15)"),
    ]
    present = [(c, t) for c, t in demo_pairs if c in df.columns]
    ncols = 2
    nrows = (len(present) + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = np.array(axes).ravel()
    for i, (col, title) in enumerate(present):
        ax = axes[i]
        if col == COL_AGE:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            series.plot(kind="hist", ax=ax, bins=20, color="steelblue", edgecolor="white")
            ax.set_title(f"{title}\n(moy={series.mean():.1f}, std={series.std():.1f})", fontsize=9)
        else:
            df[col].value_counts().head(15).plot(kind="bar", ax=ax, color="steelblue")
            ax.set_title(title, fontsize=9)
            ax.tick_params(axis="x", rotation=40, labelsize=7)
    for j in range(len(present), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"Description demographique globale (N={len(df)})", fontsize=12)
    fig.tight_layout()
    demo_png = str(out_dir / "demographics_overview.png")
    fig.savefig(demo_png, dpi=130)
    plt.close(fig)
    saved.append(demo_png)
    log.info(f"Saved: {demo_png}")

    # Langues parlees (multi-reponse)
    if COL_LANGUES_PARLEES in df.columns:
        lang_exploded = (
            df[COL_LANGUES_PARLEES].dropna().str.split(r"[,;/\n]| et | and ", regex=True)
            .explode().str.strip().str.title()
        )
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        lang_exploded.value_counts().head(20).plot(kind="bar", ax=ax2, color="teal")
        ax2.set_title("Langues parlees (multi-reponses, Top 20)", fontsize=11)
        ax2.tick_params(axis="x", rotation=45, labelsize=8)
        fig2.tight_layout()
        lang_png = str(out_dir / "demographics_langues.png")
        fig2.savefig(lang_png, dpi=130)
        plt.close(fig2)
        saved.append(lang_png)
        log.info(f"Saved: {lang_png}")

    # Stats numeriques
    stats = {}
    if COL_AGE in df.columns:
        age = pd.to_numeric(df[COL_AGE], errors="coerce")
        stats["age_mean"] = round(float(age.mean()), 1)
        stats["age_std"]  = round(float(age.std()), 1)
        stats["age_min"]  = int(age.min()) if not pd.isna(age.min()) else 0
        stats["age_max"]  = int(age.max()) if not pd.isna(age.max()) else 0

    # HTML report
    def _img(name: str) -> str:
        p = out_dir / name
        return f"<img src='{name}' alt='{name}'>" if p.exists() else ""

    stat_rows = "".join(f"<tr><td>{k}</td><td><strong>{v}</strong></td></tr>"
                        for k, v in stats.items())

    body = f"""
<h1>Description demographique globale</h1>
<div class='note'>N repondants valides (apres consentement) : <strong>{len(df)}</strong></div>
<h2>Statistiques age</h2>
<table><tr><th>Indicateur</th><th>Valeur</th></tr>{stat_rows}</table>
<h2>Vue d'ensemble</h2>
{_img("demographics_overview.png")}
<h2>Langues parlees (multi-reponses)</h2>
{_img("demographics_langues.png")}
"""
    html_path = out_dir / "demographic_report.html"
    html_path.write_text(_html_page("Demographics", body), encoding="utf-8")
    log.info(f"HTML : {html_path}")


# ── Per-hypothesis runner ─────────────────────────────────────────────────────

def run_hypothesis(
    hyp: str, df: pd.DataFrame, cfg: dict,
    tokenizer, model, nlp, device: str,
) -> dict:
    hyp_cfg  = HYP_COLS[hyp]
    dcfg     = cfg["data"]
    descfg   = cfg.get("descriptive", {})
    label    = hyp_cfg["label"]
    hyp_key  = hyp.lower()

    # output dirs
    hyp_dir  = Path("reports") / hyp_key
    desc_dir = hyp_dir / "descriptive"
    hyp_dir.mkdir(parents=True, exist_ok=True)
    desc_dir.mkdir(parents=True, exist_ok=True)

    avail = lambda lst: [c for c in lst if c in df.columns]
    all_cols  = avail(hyp_cfg["all"])
    text_cols = avail(hyp_cfg["text"])

    log.info(f"=== {hyp} — {label} ===")

    # ── Phase 1: Missing values ───────────────────────────────
    log.info(f"{hyp} | Phase 1 — Valeurs manquantes")
    missing_df = analyze_missing(df, all_cols, hyp_cfg)
    df_h = apply_imputation(df[all_cols].copy(), missing_df)
    missing_df.to_csv(hyp_dir / "phase1_missing_values.csv", index=False, encoding="utf-8-sig")
    log.info(f"Saved: {hyp_dir / 'phase1_missing_values.csv'}")

    # ── Phase 2: EDA ──────────────────────────────────────────
    log.info(f"{hyp} | Phase 2 — EDA")
    stats_df   = compute_stats(df_h, all_cols)
    outlier_df = detect_outliers_iqr(df_h, all_cols)
    plot_distributions(df_h, all_cols, str(hyp_dir / "eda_distributions.png"))
    plot_correlation(df_h, all_cols,   str(hyp_dir / "eda_correlation.png"))
    plot_target_distribution(df, hyp_cfg["target"], str(hyp_dir / "eda_target.png"))

    eda_html = generate_eda_html(hyp, label, missing_df, stats_df, outlier_df, len(df), hyp_dir)
    (hyp_dir / "eda_report.html").write_text(eda_html, encoding="utf-8")
    log.info(f"HTML : {hyp_dir / 'eda_report.html'}")

    # ── Phase 3: CamemBERT + text ─────────────────────────────
    log.info(f"{hyp} | Phase 3 — Analyse descriptive textuelle (CamemBERT)")
    batch_size = int(descfg.get("batch_size", 16))
    max_length = int(descfg.get("max_length", 128))
    n_clusters = int(descfg.get("n_clusters", 5))
    cos_thresh = float(descfg.get("cosine_threshold", 0.30))
    top_k      = int(descfg.get("ngram_top_k", 20))
    min_df     = int(descfg.get("ngram_min_df", 2))
    umap_cfg   = {
        "n_neighbors": int(descfg.get("umap_n_neighbors", 15)),
        "min_dist":    float(descfg.get("umap_min_dist", 0.1)),
        "random_state":int(descfg.get("umap_random_state", 42)),
    }

    # text stats
    txt_stats_df = text_stats(df, text_cols)

    if not text_cols:
        log.warning(f"{hyp} | Aucune colonne texte disponible — Phase 3 ignoree")
        apriori_df = pd.DataFrame({"apriori_dominant": ["non_classe"] * len(df)})
        stereo_rate = 0.0
    else:
        corpus_texts = (
            df[text_cols].fillna("").apply(lambda r: " ".join(r.values), axis=1).tolist()
        )

        # CamemBERT corpus embeddings
        log.info(f"{hyp} | Encodage CamemBERT ({len(corpus_texts)} textes)...")
        corpus_embs = encode_texts(corpus_texts, tokenizer, model, batch_size, max_length, device)

        # Stereotypes
        marker_labels  = list(STEREOTYPE_MARKERS.keys())
        marker_phrases = [" ".join(p) for p in STEREOTYPE_MARKERS.values()]
        marker_embs    = encode_texts(marker_phrases, tokenizer, model, batch_size, max_length, device)
        stereo_emb_df  = detect_stereotypes_embedding(corpus_embs, marker_embs, marker_labels, cos_thresh)
        df = detect_stereotypes_keyword(df, text_cols)
        stereo_rate = stereo_emb_df.any(axis=1).mean()

        # A Priori
        cat_labels  = list(APRIORI_CATEGORIES.keys())
        cat_phrases = list(APRIORI_CATEGORIES.values())
        cat_embs    = encode_texts(cat_phrases, tokenizer, model, batch_size, max_length, device)
        apriori_df  = classify_apriori(corpus_embs, cat_embs, cat_labels, cos_thresh)

        # UMAP + K-Means
        log.info(f"{hyp} | UMAP + K-Means...")
        reducer = UMAP(n_components=2, n_neighbors=umap_cfg["n_neighbors"],
                       min_dist=umap_cfg["min_dist"], random_state=umap_cfg["random_state"])
        coords = reducer.fit_transform(corpus_embs)
        clusters = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit_predict(corpus_embs)
        series_stereo = stereo_emb_df.apply(
            lambda r: marker_labels[int(r.values.argmax())]
            if float(r.max()) >= cos_thresh else "aucun", axis=1,
        )

        # UMAP plots
        plot_umap_scatter(corpus_embs, series_stereo,
                          f"UMAP — Stereotypes ({hyp})", str(desc_dir / "umap_stereotypes.png"), umap_cfg)
        plot_umap_scatter(corpus_embs, apriori_df["apriori_dominant"],
                          f"UMAP — A Priori ({hyp})", str(desc_dir / "umap_apriori.png"), umap_cfg)
        plot_umap_scatter(corpus_embs, pd.Series(clusters.astype(str)),
                          f"UMAP — Clusters ({hyp})", str(desc_dir / "umap_clusters.png"), umap_cfg)

        # Word clouds
        global_text = " ".join(corpus_texts)
        plot_wordcloud(global_text, f"Nuage de mots global — {hyp}",
                       str(desc_dir / "wordcloud_global.png"), nlp)
        for cat in cat_labels:
            mask     = apriori_df["apriori_dominant"] == cat
            cat_text = " ".join(df.loc[mask, text_cols].fillna("").values.flatten())
            if cat_text.strip():
                plot_wordcloud(cat_text, f"Nuage : {cat}",
                               str(desc_dir / f"wordcloud_{cat}.png"), nlp)

        # N-grams
        hyp_texts = df[text_cols].fillna("").apply(lambda r: " ".join(r.values), axis=1)
        for n, label_ng in [(1, "unigrammes"), (2, "bigrammes"), (3, "trigrammes")]:
            table = build_ngram_table(hyp_texts, nlp, n, top_k, min_df)
            plot_ngrams(table, f"Top-{top_k} {label_ng} — {hyp}",
                        str(desc_dir / f"ngrams_{label_ng}.png"))
            table.to_csv(str(desc_dir / f"ngrams_{label_ng}.csv"), index=False, encoding="utf-8-sig")

        # Co-occurrence
        lemmatized = _lemmatize(hyp_texts, nlp)
        plot_cooccurrence_heatmap(lemmatized, str(desc_dir / "heatmap_cooccurrence.png"), min_df)
        plot_cooccurrence_network(lemmatized, str(desc_dir / "network_cooccurrence.png"), min_df)

        # Distributions
        plot_apriori_distribution(apriori_df, str(desc_dir / "apriori_distribution.png"))
        plot_stereotype_distribution(df, str(desc_dir / "stereotype_distribution.png"))

    # Phase 3 HTML
    p3_html = generate_phase3_html(hyp, label, txt_stats_df, apriori_df, desc_dir)
    (desc_dir / "descriptive_report.html").write_text(p3_html, encoding="utf-8")
    log.info(f"HTML : {desc_dir / 'descriptive_report.html'}")

    metrics = {
        "n_respondants":  len(df),
        "n_missing_cols": int((missing_df["n_manquants"] > 0).sum()),
        "pct_excluded":   float(missing_df["pct_manquants"].gt(30).mean()),
        "stereo_rate":    float(stereo_rate),
    }
    dominant_cnt = apriori_df["apriori_dominant"].value_counts()
    for cat, cnt in dominant_cnt.items():
        metrics[f"apriori_{cat}"] = int(cnt)

    log.info(f"{hyp} | [OK] Phases 1-3 terminees | stereo_rate={stereo_rate:.1%} | "
             f"categorie dominante={dominant_cnt.index[0] if len(dominant_cnt) else 'N/A'}")
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def run(config_path: str, stage: str, hypothesis: str) -> None:
    cfg = load_config(config_path)
    Path("logs").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    raw_path = Path(cfg["data"]["raw_path"])
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {raw_path.absolute()}\n"
            "=> Copier data_FLP.csv dans data/raw/"
        )

    df = load_raw(str(raw_path), cfg["data"]["consent_value"])

    # ── Stage: demographics ────────────────────────────────────
    if stage == "demographics":
        run_demographics(df, Path("reports") / "demographics")
        log.info("[OK] Rapport demographique -> reports/demographics/demographic_report.html")
        return

    # ── Stage: hypothesis ─────────────────────────────────────
    hyps = ["H1", "H2", "H3", "H4"] if hypothesis.lower() == "all" else [hypothesis.upper()]
    for h in hyps:
        if h not in HYP_COLS:
            log.error(f"Hypothese inconnue : {h}. Valeurs valides : H1 H2 H3 H4 all")
            continue

    # CamemBERT (charge une seule fois pour toutes les hypotheses)
    descfg = cfg.get("descriptive", {})
    model_name = descfg.get("camembert_model", "camembert-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Chargement CamemBERT ({model_name}) sur {device}...")
    tokenizer, camembert = load_camembert(model_name, device)

    try:
        nlp = spacy.load("fr_core_news_sm", disable=["ner", "parser"])
    except OSError:
        raise OSError("Modele spaCy manquant. Lancer : python -m spacy download fr_core_news_sm")

    # MLflow
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])

    for hyp in hyps:
        mlflow.set_experiment(f"FLP_Descriptive_{hyp}")
        _nested = mlflow.active_run() is not None
        with mlflow.start_run(run_name=f"descriptive_phases1-3_{hyp}", nested=_nested):
            mlflow.log_param("camembert_model", model_name)
            mlflow.log_param("cosine_threshold", descfg.get("cosine_threshold", 0.30))
            mlflow.log_param("n_clusters", descfg.get("n_clusters", 5))

            metrics = run_hypothesis(hyp, df.copy(), cfg, tokenizer, camembert, nlp, device)
            log_metrics(metrics)

            # Log all artifacts
            hyp_dir  = Path("reports") / hyp.lower()
            desc_dir = hyp_dir / "descriptive"
            for p in list(hyp_dir.glob("*.png")) + list(hyp_dir.glob("*.html")) + \
                     list(hyp_dir.glob("*.csv")) + list(desc_dir.glob("*.png")) + \
                     list(desc_dir.glob("*.html")) + list(desc_dir.glob("*.csv")):
                mlflow.log_artifact(str(p), artifact_path=f"reports/{hyp.lower()}")

    log.info("[OK] Analyse descriptive terminee.")


def run_full(config_path: str) -> dict:
    """Entry point for pipeline.py — demographics + H1-H4 phases 1-3 in one shot."""
    cfg = load_config(config_path)
    Path("logs").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    raw_path = Path(cfg["data"]["raw_path"])
    if not raw_path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {raw_path.absolute()}")

    df = load_raw(str(raw_path), cfg["data"]["consent_value"])

    run_demographics(df, Path("reports") / "demographics")
    log.info("[OK] Demographics terminee")

    descfg     = cfg.get("descriptive", {})
    model_name = descfg.get("camembert_model", "camembert-base")
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Chargement CamemBERT ({model_name}) sur {device}...")
    tokenizer, camembert = load_camembert(model_name, device)

    try:
        nlp = spacy.load("fr_core_news_sm", disable=["ner", "parser"])
    except OSError:
        raise OSError("Modele spaCy manquant. Lancer : python -m spacy download fr_core_news_sm")

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    all_metrics: dict = {}

    for hyp in ["H1", "H2", "H3", "H4"]:
        mlflow.set_experiment(f"FLP_Descriptive_{hyp}")
        _nested = mlflow.active_run() is not None
        with mlflow.start_run(run_name=f"descriptive_phases1-3_{hyp}", nested=_nested):
            mlflow.log_param("camembert_model", model_name)
            mlflow.log_param("cosine_threshold", descfg.get("cosine_threshold", 0.30))
            mlflow.log_param("n_clusters", descfg.get("n_clusters", 5))

            metrics = run_hypothesis(hyp, df.copy(), cfg, tokenizer, camembert, nlp, device)
            log_metrics(metrics)
            all_metrics[hyp] = metrics

            hyp_dir  = Path("reports") / hyp.lower()
            desc_dir = hyp_dir / "descriptive"
            for p in list(hyp_dir.glob("*.png")) + list(hyp_dir.glob("*.html")) + \
                     list(hyp_dir.glob("*.csv")) + list(desc_dir.glob("*.png")) + \
                     list(desc_dir.glob("*.html")) + list(desc_dir.glob("*.csv")):
                mlflow.log_artifact(str(p), artifact_path=f"reports/{hyp.lower()}")

    log.info("[OK] Analyse descriptive complete (demographics + H1-H4)")
    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse descriptive FLP — Phases 1-3 par hypothese + demographics"
    )
    parser.add_argument("--config",     default="params.yaml")
    parser.add_argument("--stage",      default="",    help="demographics")
    parser.add_argument("--hypothesis", default="",    help="H1 | H2 | H3 | H4 | all")
    args = parser.parse_args()

    if not args.stage and not args.hypothesis:
        parser.error("Specifier --stage demographics OU --hypothesis H1/H2/H3/H4/all")

    run(args.config, args.stage, args.hypothesis)
