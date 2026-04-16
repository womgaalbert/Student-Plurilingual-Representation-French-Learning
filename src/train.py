"""
train.py — French-Learning-Perceptions ML (Level 0 + Level 1)
Point d'entrée unique pour entraîner H1, H2, H3 ou H4.
Chaque run est loggé automatiquement dans MLflow.

Usage:
  python src/train.py --hypothesis H1 --config params.yaml
  python src/train.py --hypothesis ALL --config params.yaml
"""
import sys
import argparse
import logging
import pickle
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import mlflow
import yaml
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (f1_score, roc_auc_score, classification_report,
                              mean_absolute_error, accuracy_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from imblearn.over_sampling import SMOTE
from scipy.stats import pearsonr, spearmanr
import xgboost as xgb
import shap

sys.path.insert(0, str(Path(__file__).parent))
from utils.config import load_config
from utils.mlflow_utils import (setup_mlflow, log_params_from_config,
                                  log_metrics, log_thresholds_met,
                                  log_artifact_json, log_model_artifact)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/train.log", encoding="utf-8"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

Path("logs").mkdir(exist_ok=True)
Path("models/h1").mkdir(parents=True, exist_ok=True)
Path("models/h2").mkdir(parents=True, exist_ok=True)
Path("models/h3").mkdir(parents=True, exist_ok=True)
Path("models/h4").mkdir(parents=True, exist_ok=True)
Path("reports").mkdir(exist_ok=True)


def pipe(model) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   model),
    ])


# ── H1 ───────────────────────────────────────────────────────────────────────
def train_h1(cfg: dict) -> dict:
    c = cfg["h1"]; dc = cfg["data"]; mc = cfg["mlflow"]
    setup_mlflow(mc["tracking_uri"], mc["experiments"]["h1"])

    df = pd.read_csv(f"{dc['processed_dir']}/h1_features.csv")
    feat_cols = (
        [c2 for c2 in ["nb_langues","apprent_anterieur_bin","relation_lm_ord",
                        "domaine_usage_freq","valorisation_sent","sexe_bin","age"] if c2 in df.columns]
        + [c2 for c2 in df.columns if c2.startswith(("lm_","region_"))]
    )
    X, y = df[feat_cols].fillna(0), df["h1_target"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=dc["test_size"], stratify=y, random_state=dc["random_state"])
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=dc["val_size"]/(1-dc["test_size"]),
        stratify=y_tr, random_state=dc["random_state"])

    if c["smote"]:
        X_tr, y_tr = SMOTE(random_state=dc["random_state"]).fit_resample(X_tr, y_tr)

    model = pipe(xgb.XGBClassifier(
        n_estimators=c["n_estimators"], max_depth=c["max_depth"],
        learning_rate=c["learning_rate"], subsample=c["subsample"],
        colsample_bytree=c["colsample_bytree"],
        use_label_encoder=False, eval_metric="logloss",
        random_state=dc["random_state"]))

    with mlflow.start_run(run_name=f"H1_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        log_params_from_config(c)
        model.fit(X_tr, y_tr)

        y_pred  = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1]
        f1      = f1_score(y_te, y_pred, average="macro")
        auc     = roc_auc_score(y_te, y_proba)

        cv = cross_validate(pipe(xgb.XGBClassifier(
                use_label_encoder=False, eval_metric="logloss",
                random_state=dc["random_state"])),
            X_tr, y_tr, cv=StratifiedKFold(c["cv_folds"], shuffle=True,
                                             random_state=dc["random_state"]),
            scoring=["f1_macro","roc_auc"])

        metrics = {"f1_macro": round(f1,4), "roc_auc": round(auc,4),
                   "cv_f1_macro_mean": round(cv["test_f1_macro"].mean(),4),
                   "cv_f1_macro_std":  round(cv["test_f1_macro"].std(),4)}
        thresholds_met = {
            "f1_macro": f1  >= c["thresholds"]["f1_macro"],
            "roc_auc":  auc >= c["thresholds"]["roc_auc"],
        }

        log_metrics(metrics)
        log_thresholds_met(thresholds_met)

        # SHAP
        top5 = []
        try:
            explainer = shap.TreeExplainer(model.named_steps["model"])
            X_sc = model[:-1].transform(X_te)
            sv = explainer.shap_values(X_sc)
            top5 = sorted(zip(feat_cols, np.abs(sv).mean(axis=0)),
                           key=lambda x: x[1], reverse=True)[:5]
            mlflow.log_param("shap_top_features", str([f for f, _ in top5]))
        except Exception as e:
            log.warning(f"SHAP non disponible : {e}")

        # Save model
        ts = datetime.now().strftime('%Y%m%d_%H%M')
        mp = f"models/h1/h1_xgb_{ts}.pkl"
        with open(mp, "wb") as f: pickle.dump(model, f)
        log_model_artifact(mp)

        result = {"hypothese": "H1", "metrics": metrics,
                  "thresholds_met": thresholds_met, "model_path": mp,
                  "shap_top5": [f for f, _ in top5]}
        log_artifact_json(result, f"h1_run_{ts}.json")
        mlflow.set_tag("status", "✅ OK" if all(thresholds_met.values()) else "⚠️ Seuils non atteints")

    log.info(f"H1 — F1={f1:.3f} | AUC={auc:.3f} | {'✅' if all(thresholds_met.values()) else '⚠️'}")
    return result


# ── H2 ───────────────────────────────────────────────────────────────────────
def train_h2(cfg: dict) -> dict:
    c = cfg["h2"]; dc = cfg["data"]; mc = cfg["mlflow"]
    setup_mlflow(mc["tracking_uri"], mc["experiments"]["h2"])

    df = pd.read_csv(f"{dc['processed_dir']}/h2_features.csv")
    feat_cols = (
        [c2 for c2 in ["perc_utile","perc_belle","perc_difficile","perc_importante",
                        "mots_assoc_sent","importance_bin","importance_sent",
                        "hierarchie_fr","age","sexe_bin"] if c2 in df.columns]
        + [c2 for c2 in df.columns if c2.startswith("region_")]
    )
    X   = df[feat_cols].fillna(0)
    y_A = df["h2_target_motivation"]
    diff_cols = [c2 for c2 in df.columns if c2.startswith("diff_")]
    y_B = df[diff_cols].fillna(0).astype(int)

    idx_tr, idx_te = train_test_split(
        np.arange(len(X)), test_size=dc["test_size"],
        stratify=y_A, random_state=dc["random_state"])

    with mlflow.start_run(run_name=f"H2_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        log_params_from_config(c)

        # Cible A — motivation (3 classes)
        m_A = pipe(xgb.XGBClassifier(
            n_estimators=c["n_estimators"], max_depth=c["max_depth"],
            learning_rate=c["learning_rate"],
            use_label_encoder=False, eval_metric="mlogloss",
            num_class=3, objective="multi:softprob",
            random_state=dc["random_state"]))
        m_A.fit(X.iloc[idx_tr], y_A.iloc[idx_tr])
        f1_A = f1_score(y_A.iloc[idx_te], m_A.predict(X.iloc[idx_te]), average="weighted")

        # Cible B — difficultés (multi-label)
        m_B = pipe(MultiOutputClassifier(xgb.XGBClassifier(
            n_estimators=c["n_estimators"], max_depth=c["max_depth"],
            use_label_encoder=False, eval_metric="logloss",
            random_state=dc["random_state"])))
        m_B.fit(X.iloc[idx_tr], y_B.iloc[idx_tr])
        f1_B = f1_score(y_B.iloc[idx_te], m_B.predict(X.iloc[idx_te]),
                         average="micro", zero_division=0)

        metrics = {"f1_weighted_A": round(f1_A,4), "f1_micro_B": round(f1_B,4)}
        thresholds_met = {
            "f1_weighted_A": f1_A >= c["thresholds"]["f1_weighted_A"],
            "f1_micro_B":    f1_B >= c["thresholds"]["f1_micro_B"],
        }
        log_metrics(metrics); log_thresholds_met(thresholds_met)

        ts = datetime.now().strftime('%Y%m%d_%H%M')
        for name, m in [("A_motivation", m_A), ("B_difficultes", m_B)]:
            mp = f"models/h2/h2_{name}_{ts}.pkl"
            with open(mp, "wb") as f: pickle.dump(m, f)
            log_model_artifact(mp)

        result = {"hypothese": "H2", "metrics": metrics, "thresholds_met": thresholds_met}
        log_artifact_json(result, f"h2_run_{ts}.json")
        mlflow.set_tag("status", "✅ OK" if all(thresholds_met.values()) else "⚠️")

    log.info(f"H2 — F1-A={f1_A:.3f} | F1-B={f1_B:.3f} | {'✅' if all(thresholds_met.values()) else '⚠️'}")
    return result


# ── H3 ───────────────────────────────────────────────────────────────────────
def train_h3(cfg: dict) -> dict:
    c = cfg["h3"]; dc = cfg["data"]; mc = cfg["mlflow"]
    setup_mlflow(mc["tracking_uri"], mc["experiments"]["h3"])

    df = pd.read_csv(f"{dc['processed_dir']}/h3_features.csv")
    feat_cols = (
        [c2 for c2 in ["exposition_freq","interet_bin","interet_sent",
                        "perception_multi_sent","perception_multi_ord",
                        "nb_langues","age","sexe_bin"] if c2 in df.columns]
        + [c2 for c2 in df.columns if c2.startswith("region_")]
    )
    X      = df[feat_cols].fillna(0)
    y_reg  = df["h3_score_attitude"]
    le     = LabelEncoder().fit(["Négative","Neutre","Positive"])
    y_clf  = le.transform(df["h3_attitude_class"].fillna("Neutre"))

    X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(
        X, y_reg, y_clf, test_size=dc["test_size"], random_state=dc["random_state"])

    with mlflow.start_run(run_name=f"H3_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        log_params_from_config(c)

        m_reg = pipe(RandomForestRegressor(
            n_estimators=c["n_estimators_reg"], max_depth=c["max_depth"],
            random_state=dc["random_state"]))
        m_reg.fit(X_tr, yr_tr)
        mae = mean_absolute_error(yr_te, m_reg.predict(X_te))

        m_clf = pipe(xgb.XGBClassifier(
            n_estimators=c["n_estimators_clf"], max_depth=c["max_depth"],
            use_label_encoder=False, eval_metric="mlogloss",
            num_class=3, objective="multi:softprob",
            random_state=dc["random_state"]))
        m_clf.fit(X_tr, yc_tr)
        f1_w = f1_score(yc_te, m_clf.predict(X_te), average="weighted")

        r, p = pearsonr(df["exposition_freq"].fillna(0),
                        df["h3_score_attitude"].fillna(df["h3_score_attitude"].mean()))
        ate_high = df.loc[df["exposition_freq"] >= 3, "h3_score_attitude"].mean()
        ate_low  = df.loc[df["exposition_freq"] <  3, "h3_score_attitude"].mean()
        ate = round(ate_high - ate_low, 3)

        metrics = {"mae": round(mae,4), "f1_weighted_clf": round(f1_w,4),
                   "pearson_r": round(r,4), "pearson_p": round(p,4), "ate": ate}
        thresholds_met = {
            "mae":         mae  <= c["thresholds"]["mae"],
            "f1_weighted": f1_w >= c["thresholds"]["f1_weighted"],
            "causal_sig":  p    <  c["thresholds"]["pearson_p"] and r > 0,
        }
        log_metrics(metrics); log_thresholds_met(thresholds_met)

        ts = datetime.now().strftime('%Y%m%d_%H%M')
        for name, m in [("reg", m_reg), ("clf", m_clf)]:
            mp = f"models/h3/h3_{name}_{ts}.pkl"
            with open(mp, "wb") as f: pickle.dump(m, f)
            log_model_artifact(mp)

        result = {"hypothese": "H3", "metrics": metrics, "thresholds_met": thresholds_met}
        log_artifact_json(result, f"h3_run_{ts}.json")
        mlflow.set_tag("status", "✅ OK" if all(thresholds_met.values()) else "⚠️")

    log.info(f"H3 — MAE={mae:.3f} | F1={f1_w:.3f} | r={r:.3f}(p={p:.3f}) | {'✅' if all(thresholds_met.values()) else '⚠️'}")
    return result


# ── H4 ───────────────────────────────────────────────────────────────────────
def train_h4(cfg: dict) -> dict:
    c = cfg["h4"]; dc = cfg["data"]; mc = cfg["mlflow"]
    setup_mlflow(mc["tracking_uri"], mc["experiments"]["h4"])

    df = pd.read_csv(f"{dc['processed_dir']}/h4_features.csv")
    feat_cols = (
        [c2 for c2 in ["interet_camarades_bin","interet_camarades_sent",
                        "souhait_freq","age","sexe_bin"] if c2 in df.columns]
        + [c2 for c2 in df.columns if c2.startswith(("vi_disc_","region_"))]
    )
    X   = df[feat_cols].fillna(0)
    y_A = df["h4_target_motivation"].fillna(0).astype(int)
    y_B = df["h4_engagement_score"].fillna(1).astype(int)
    disc_cols = [c2 for c2 in df.columns if c2.startswith("vd_disc_")]
    y_C = df[disc_cols].fillna(0).astype(int)

    X_tr, X_te, yA_tr, yA_te, yB_tr, yB_te, yC_tr, yC_te = train_test_split(
        X, y_A, y_B, y_C, test_size=dc["test_size"],
        stratify=y_A, random_state=dc["random_state"])

    with mlflow.start_run(run_name=f"H4_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        log_params_from_config(c)

        # Cible A
        m_A = pipe(xgb.XGBClassifier(
            n_estimators=c["n_estimators"],
            use_label_encoder=False, eval_metric="logloss",
            scale_pos_weight=(yA_tr == 0).sum() / max((yA_tr == 1).sum(), 1),
            random_state=dc["random_state"]))
        m_A.fit(X_tr, yA_tr)
        f1_A = f1_score(yA_te, m_A.predict(X_te))

        # Cible B — ordinal via XGBoost multi-class
        m_B = pipe(xgb.XGBClassifier(
            n_estimators=c["n_estimators"], max_depth=c["max_depth"],
            learning_rate=c["learning_rate"],
            use_label_encoder=False, eval_metric="mlogloss",
            num_class=4, objective="multi:softprob",
            random_state=dc["random_state"]))
        m_B.fit(X_tr, yB_tr - 1)
        yB_pred = m_B.predict(X_te) + 1
        _sr = spearmanr(yB_te, yB_pred)
        rho = float(np.asarray(_sr.statistic).flat[0])

        # Cible C — multi-label
        m_C = pipe(MultiOutputClassifier(xgb.XGBClassifier(
            n_estimators=c["n_estimators"], max_depth=c["max_depth"],
            use_label_encoder=False, eval_metric="logloss",
            random_state=dc["random_state"])))
        m_C.fit(X_tr, yC_tr)
        sub_acc = accuracy_score(yC_te, m_C.predict(X_te))

        # Classement disciplines
        disc_labels = [d.replace("vd_disc_","") for d in disc_cols]
        probas = m_C.predict_proba(X_te)
        ranking = sorted(zip(disc_labels, [p[:,1].mean() for p in probas]),
                          key=lambda x: x[1], reverse=True)

        metrics = {"f1_A": round(f1_A,4), "spearman_B": round(rho,4),
                   "subset_acc_C": round(sub_acc,4)}
        thresholds_met = {
            "f1_A":       f1_A    >= c["thresholds"]["f1_A"],
            "spearman_B": rho     >= c["thresholds"]["spearman_B"],
            "subset_C":   sub_acc >= c["thresholds"]["subset_C"],
        }
        log_metrics(metrics); log_thresholds_met(thresholds_met)
        mlflow.log_param("discipline_top1", ranking[0][0] if ranking else "N/A")

        ts = datetime.now().strftime('%Y%m%d_%H%M')
        for name, m in [("A_motivation",m_A),("B_engagement",m_B),("C_discipline",m_C)]:
            mp = f"models/h4/h4_{name}_{ts}.pkl"
            with open(mp,"wb") as f: pickle.dump(m, f)
            log_model_artifact(mp)

        result = {"hypothese": "H4", "metrics": metrics,
                  "thresholds_met": thresholds_met,
                  "discipline_ranking": [{"disc": d, "proba": round(p,4)} for d,p in ranking]}
        log_artifact_json(result, f"h4_run_{ts}.json")
        mlflow.set_tag("status", "✅ OK" if all(thresholds_met.values()) else "⚠️")

    log.info(f"H4 — F1-A={f1_A:.3f} | rho={rho:.3f} | SubAcc={sub_acc:.3f} | "
             f"{'✅' if all(thresholds_met.values()) else '⚠️'}")
    return result


# ── Entry point ──────────────────────────────────────────────────────────────
TRAINERS = {"H1": train_h1, "H2": train_h2, "H3": train_h3, "H4": train_h4}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypothesis", choices=["H1","H2","H3","H4","ALL"], default="ALL")
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()

    cfg     = load_config(args.config)
    targets = list(TRAINERS.keys()) if args.hypothesis == "ALL" else [args.hypothesis]

    for h in targets:
        log.info(f"\n{'='*50}\nENTRAÎNEMENT {h}\n{'='*50}")
        TRAINERS[h](cfg)
