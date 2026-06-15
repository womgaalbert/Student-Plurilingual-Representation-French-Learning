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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import (f1_score, roc_auc_score, classification_report,
                              mean_absolute_error, accuracy_score, make_scorer)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import pearsonr, spearmanr
import xgboost as xgb
import lightgbm as lgb
import shap

sys.path.insert(0, str(Path(__file__).parent))
from utils.config import load_config
from utils.mlflow_utils import (setup_mlflow, log_params_from_config,
                                  log_metrics, log_thresholds_met,
                                  log_artifact_json, log_model_artifact)

Path("logs").mkdir(parents=True, exist_ok=True)
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

    _nested = mlflow.active_run() is not None
    with mlflow.start_run(run_name=f"H1_{datetime.now().strftime('%Y%m%d_%H%M')}", nested=_nested):
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
        [c2 for c2 in ["perc_utile","perc_belle","perc_diffi","perc_impor",
                        "perc_compli","perc_intére","perc_ennuye","perc_nécess",
                        "perc_riche","perc_obliga",
                        "mots_assoc_sent","importance_bin","importance_sent",
                        "hierarchie_fr","age","sexe_bin"] if c2 in df.columns]
        + [c2 for c2 in df.columns if c2.startswith(("region_","facile_"))]
    )
    X   = df[feat_cols].fillna(0)
    y_A = df["h2_target_motivation"]
    diff_cols = [c2 for c2 in df.columns if c2.startswith("diff_")]
    y_B = df[diff_cols].fillna(0).astype(int)

    idx_tr, idx_te = train_test_split(
        np.arange(len(X)), test_size=dc["test_size"],
        stratify=y_A, random_state=dc["random_state"])

    _nested = mlflow.active_run() is not None
    with mlflow.start_run(run_name=f"H2_{datetime.now().strftime('%Y%m%d_%H%M')}", nested=_nested):
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

        # Cible B — difficultés (ClassifierChain captures label dependencies)
        m_B = pipe(ClassifierChain(xgb.XGBClassifier(
            n_estimators=c["n_estimators"], max_depth=c["max_depth"],
            use_label_encoder=False, eval_metric="logloss",
            random_state=dc["random_state"]), random_state=dc["random_state"]))
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
        [c2 for c2 in ["exposition_bin","interet_bin","interet_sent",
                        "perception_multi_sent","perception_multi_ord",
                        "nb_langues","age","sexe_bin"] if c2 in df.columns]
        + [c2 for c2 in df.columns if c2.startswith(("region_","emb_pca_"))]
    )
    X      = df[feat_cols].fillna(0)
    y_reg  = df["h3_score_attitude"]
    le     = LabelEncoder().fit(["Négative","Neutre","Positive"])
    y_clf  = le.transform(df["h3_attitude_class"].fillna("Neutre"))

    X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(
        X, y_reg, y_clf, test_size=dc["test_size"], random_state=dc["random_state"])

    _nested = mlflow.active_run() is not None
    with mlflow.start_run(run_name=f"H3_{datetime.now().strftime('%Y%m%d_%H%M')}", nested=_nested):
        log_params_from_config(c)

        m_reg = pipe(VotingRegressor([
            ("et", ExtraTreesRegressor(n_estimators=500, max_depth=8, min_samples_leaf=2, random_state=42)),
            ("xgb", xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.03,
                                     subsample=0.85, objective="reg:squarederror", random_state=42)),
        ]))
        m_reg.fit(X_tr, yr_tr)
        mae = mean_absolute_error(yr_te, m_reg.predict(X_te))

        m_clf = pipe(xgb.XGBClassifier(
            n_estimators=c["n_estimators_clf"], max_depth=c["max_depth"],
            use_label_encoder=False, eval_metric="mlogloss",
            num_class=3, objective="multi:softprob",
            random_state=dc["random_state"]))
        m_clf.fit(X_tr, yc_tr)
        f1_w = f1_score(yc_te, m_clf.predict(X_te), average="weighted")

        r, p = pearsonr(df["exposition_bin"].fillna(0),
                        df["h3_score_attitude"].fillna(df["h3_score_attitude"].mean()))
        ate_high = df.loc[df["exposition_bin"] == 1, "h3_score_attitude"].mean()
        ate_low  = df.loc[df["exposition_bin"] == 0, "h3_score_attitude"].mean()
        ate = round(float(ate_high - ate_low) if not pd.isna(ate_high - ate_low) else 0.0, 3)

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
        [c2 for c2 in ["interet_camarades_ord","interet_camarades_sent",
                        "souhait_freq","age","sexe_bin"] if c2 in df.columns]
        + [c2 for c2 in df.columns if c2.startswith(("vi_disc_","region_","emb_pca_"))]
    )
    X   = df[feat_cols].fillna(0)
    y_A = df["h4_target_motivation"].fillna(0).astype(int)
    y_B = df["h4_engagement_score"].fillna(1).astype(int)
    disc_cols = [c2 for c2 in df.columns if c2.startswith("vd_disc_")]
    y_C = df[disc_cols].fillna(0).astype(int)

    X_tr, X_te, yA_tr, yA_te, yB_tr, yB_te, yC_tr, yC_te = train_test_split(
        X, y_A, y_B, y_C, test_size=dc["test_size"],
        stratify=y_A, random_state=dc["random_state"])

    _nested = mlflow.active_run() is not None
    with mlflow.start_run(run_name=f"H4_{datetime.now().strftime('%Y%m%d_%H%M')}", nested=_nested):
        log_params_from_config(c)

        # Cible A — binary with class weighting
        spw = (yA_tr == 0).sum() / max((yA_tr == 1).sum(), 1)
        m_A = pipe(xgb.XGBClassifier(
            n_estimators=c["n_estimators"],
            use_label_encoder=False, eval_metric="logloss",
            scale_pos_weight=spw,
            random_state=dc["random_state"]))
        m_A.fit(X_tr, yA_tr)
        f1_A = f1_score(yA_te, m_A.predict(X_te), zero_division=0)

        # Cible B — ordinal via XGBoost multi-class (4 levels → 0-indexed)
        n_cls_B = int(y_B.nunique())
        m_B = pipe(xgb.XGBClassifier(
            n_estimators=c["n_estimators"], max_depth=c["max_depth"],
            learning_rate=c["learning_rate"],
            use_label_encoder=False, eval_metric="mlogloss",
            num_class=n_cls_B, objective="multi:softprob",
            random_state=dc["random_state"]))
        m_B.fit(X_tr, yB_tr - 1)
        yB_pred = m_B.predict(X_te) + 1
        _sr = spearmanr(yB_te, yB_pred)
        rho = float(np.asarray(_sr.statistic).flat[0])

        # Cible C — multi-label (ClassifierChain captures label dependencies)
        m_C = pipe(ClassifierChain(xgb.XGBClassifier(
            n_estimators=c["n_estimators"], max_depth=c["max_depth"],
            use_label_encoder=False, eval_metric="logloss",
            random_state=dc["random_state"]), random_state=dc["random_state"]))
        m_C.fit(X_tr, yC_tr)
        sub_acc = accuracy_score(yC_te, m_C.predict(X_te))

        # Classement disciplines (ClassifierChain.predict_proba → shape n_samples × n_labels)
        disc_labels = [d.replace("vd_disc_","") for d in disc_cols]
        chain_p = m_C.predict_proba(X_te)  # (n_samples, n_labels)
        ranking = sorted(zip(disc_labels, chain_p.mean(axis=0).tolist()),
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


# ── Shared tuning helpers ─────────────────────────────────────────────────────

def _val_frac(dc: dict) -> float:
    return dc["val_size"] / (1.0 - dc["test_size"])

def _safe_pearsonr(x, y):
    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return 0.0, 1.0
    r, p = pearsonr(x, y)
    return (float(r), float(p)) if not (np.isnan(r) or np.isnan(p)) else (0.0, 1.0)

def _imb_pipe(model) -> ImbPipeline:
    # RandomOverSampler instead of SMOTE: works for any minority class size in CV folds
    return ImbPipeline([
        ("ros",     RandomOverSampler(random_state=42)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("model",   model),
    ])


# ── Phase 9 — Hyperparameter Tuning ──────────────────────────────────────────

def tune_h2(cfg: dict) -> dict:
    c = cfg["h2"]; dc = cfg["data"]; mc = cfg["mlflow"]
    tun = c.get("tuning", {})
    setup_mlflow(mc["tracking_uri"], mc["experiments"]["h2"])

    df = pd.read_csv(f"{dc['processed_dir']}/h2_features.csv")
    feat_cols = (
        [c2 for c2 in ["perc_utile","perc_belle","perc_diffi","perc_impor",
                        "perc_compli","perc_intére","perc_ennuye","perc_nécess",
                        "perc_riche","perc_obliga",
                        "mots_assoc_sent","importance_bin","importance_sent",
                        "hierarchie_fr","age","sexe_bin"] if c2 in df.columns]
        + [c2 for c2 in df.columns if c2.startswith(("region_","facile_"))]
    )
    X   = df[feat_cols].fillna(0)
    y_A = df["h2_target_motivation"]
    diff_cols = [c2 for c2 in df.columns if c2.startswith("diff_")]
    y_B = df[diff_cols].fillna(0).astype(int)

    # 3-way split 70/15/15
    X_tr, X_te, yA_tr, yA_te, yB_tr, yB_te = train_test_split(
        X, y_A, y_B, test_size=dc["test_size"],
        stratify=y_A, random_state=dc["random_state"])
    X_tr, X_val, yA_tr, yA_val, yB_tr, yB_val = train_test_split(
        X_tr, yA_tr, yB_tr, test_size=_val_frac(dc),
        stratify=yA_tr, random_state=dc["random_state"])

    cv = tun.get("cv_folds", 3)
    n_est = tun.get("n_estimators", [100, 200, 300])
    depths = tun.get("max_depth", [3, 4, 5])
    lrs    = tun.get("learning_rate", [0.05, 0.08, 0.10])

    # Target A — motivation (3-class) with ROS
    grid_A = {"model__n_estimators": n_est, "model__max_depth": depths,
               "model__learning_rate": lrs}
    gs_A = GridSearchCV(
        _imb_pipe(xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss",
                                     num_class=3, objective="multi:softprob",
                                     random_state=dc["random_state"])),
        grid_A, cv=StratifiedKFold(cv, shuffle=True, random_state=dc["random_state"]),
        scoring="f1_weighted", n_jobs=-1, refit=True, error_score=0.0)
    gs_A.fit(X_tr, yA_tr)
    f1_A_val = f1_score(yA_val, gs_A.best_estimator_.predict(X_val), average="weighted")

    # Retrain best A on train+val (re-clone to avoid leakage from CV best_estimator_)
    X_tv = pd.concat([X_tr, X_val]); yA_tv = pd.concat([yA_tr, yA_val])
    params_A2 = {k.replace("model__", ""): v for k, v in gs_A.best_params_.items()}
    xgb_A2 = xgb.XGBClassifier(**params_A2, use_label_encoder=False,
                                eval_metric="mlogloss", num_class=3,
                                objective="multi:softprob", random_state=dc["random_state"])
    best_A = _imb_pipe(xgb_A2) if yA_tv.nunique() > 1 else pipe(xgb_A2)
    best_A.fit(X_tv, yA_tv)
    f1_A = f1_score(yA_te, best_A.predict(X_te), average="weighted")

    # Target B — difficultés (ClassifierChain, captures label dependencies)
    grid_B = {"model__base_estimator__n_estimators": n_est,
               "model__base_estimator__max_depth": depths}
    gs_B = GridSearchCV(
        pipe(ClassifierChain(xgb.XGBClassifier(use_label_encoder=False,
             eval_metric="logloss", random_state=dc["random_state"]),
             random_state=dc["random_state"])),
        grid_B, cv=cv, scoring="f1_micro", n_jobs=-1, refit=True, error_score=0.0)
    gs_B.fit(X_tr, yB_tr)
    f1_B_val = f1_score(yB_val, gs_B.best_estimator_.predict(X_val),
                         average="micro", zero_division=0)

    yB_tv = pd.concat([yB_tr, yB_val])
    best_B = gs_B.best_estimator_
    best_B.fit(X_tv, yB_tv)
    f1_B = f1_score(yB_te, best_B.predict(X_te), average="micro", zero_division=0)

    metrics = {"f1_weighted_A": round(f1_A,4), "f1_micro_B": round(f1_B,4),
               "val_f1_A": round(f1_A_val,4), "val_f1_B": round(f1_B_val,4)}
    thresholds_met = {
        "f1_weighted_A": f1_A >= c["thresholds"]["f1_weighted_A"],
        "f1_micro_B":    f1_B >= c["thresholds"]["f1_micro_B"],
    }

    _nested = mlflow.active_run() is not None
    with mlflow.start_run(run_name=f"H2_tuned_{datetime.now().strftime('%Y%m%d_%H%M')}",
                          nested=_nested):
        mlflow.set_tag("phase", "9_tuning")
        mlflow.log_param("best_params_A", str(gs_A.best_params_))
        mlflow.log_param("best_params_B", str(gs_B.best_params_))
        log_metrics(metrics); log_thresholds_met(thresholds_met)

        ts = datetime.now().strftime('%Y%m%d_%H%M')
        for name, m in [("A_motivation_tuned", best_A), ("B_difficultes_tuned", best_B)]:
            mp = f"models/h2/h2_{name}_{ts}.pkl"
            with open(mp, "wb") as fh: pickle.dump(m, fh)
            log_model_artifact(mp)

        result = {"hypothese": "H2", "metrics": metrics, "thresholds_met": thresholds_met,
                  "best_params": {"A": gs_A.best_params_, "B": gs_B.best_params_}}
        log_artifact_json(result, f"h2_tuned_{ts}.json")
        mlflow.set_tag("status", "✅ OK" if all(thresholds_met.values()) else "⚠️ Seuils non atteints")

    log.info(f"H2 TUNED — F1-A={f1_A:.3f} | F1-B={f1_B:.3f} | "
             f"{'✅' if all(thresholds_met.values()) else '⚠️'}")
    return result


def tune_h3(cfg: dict) -> dict:
    c = cfg["h3"]; dc = cfg["data"]; mc = cfg["mlflow"]
    tun = c.get("tuning", {})
    setup_mlflow(mc["tracking_uri"], mc["experiments"]["h3"])

    df = pd.read_csv(f"{dc['processed_dir']}/h3_features.csv")
    feat_cols = (
        [c2 for c2 in ["exposition_bin","interet_bin","interet_sent",
                        "perception_multi_sent","perception_multi_ord",
                        "nb_langues","age","sexe_bin"] if c2 in df.columns]
        + [c2 for c2 in df.columns if c2.startswith(("region_","emb_pca_"))]
    )
    X      = df[feat_cols].fillna(0)
    y_reg  = df["h3_score_attitude"]
    le     = LabelEncoder().fit(["Négative","Neutre","Positive"])
    y_clf  = le.transform(df["h3_attitude_class"].fillna("Neutre"))

    # 3-way split 70/15/15
    X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(
        X, y_reg, y_clf, test_size=dc["test_size"], random_state=dc["random_state"])
    X_tr, X_val, yr_tr, yr_val, yc_tr, yc_val = train_test_split(
        X_tr, yr_tr, yc_tr, test_size=_val_frac(dc), random_state=dc["random_state"])

    cv = tun.get("cv_folds", 3)

    # Regression grid — ExtraTreesRegressor
    grid_reg = {
        "model__n_estimators":     tun.get("n_estimators_reg", [300, 500, 700]),
        "model__max_depth":        tun.get("max_depth_reg", [6, 8, 10]),
        "model__min_samples_leaf": tun.get("min_samples_leaf", [1, 2, 4]),
    }
    gs_reg = GridSearchCV(
        pipe(ExtraTreesRegressor(random_state=dc["random_state"])),
        grid_reg, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1, refit=True)

    # Fit ExtraTrees grid → then blend best ET with XGBoost via VotingRegressor
    gs_reg.fit(X_tr, yr_tr)
    mae_val = mean_absolute_error(yr_val, gs_reg.best_estimator_.predict(X_val))

    best_et_params = {k.replace("model__", ""): v for k, v in gs_reg.best_params_.items()}
    et_tuned = ExtraTreesRegressor(**best_et_params, random_state=dc["random_state"])
    xg_fixed = xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.03,
                                 subsample=0.85, objective="reg:squarederror", random_state=42)
    best_reg = pipe(VotingRegressor([("et", et_tuned), ("xgb", xg_fixed)]))

    X_tv = pd.concat([X_tr, X_val])
    yr_tv = pd.concat([yr_tr, yr_val])
    yc_tv = np.concatenate([yc_tr, yc_val])
    best_reg.fit(X_tv, yr_tv)
    mae = mean_absolute_error(yr_te, best_reg.predict(X_te))

    # Classifier grid
    grid_clf = {
        "model__n_estimators": tun.get("n_estimators_clf", [100, 200, 300]),
        "model__max_depth":    tun.get("max_depth_clf", [3, 4, 6]),
    }
    gs_clf = GridSearchCV(
        pipe(xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss",
                                num_class=3, objective="multi:softprob",
                                random_state=dc["random_state"])),
        grid_clf, cv=StratifiedKFold(cv, shuffle=True, random_state=dc["random_state"]),
        scoring="f1_weighted", n_jobs=-1, refit=True)
    gs_clf.fit(X_tr, yc_tr)
    f1_val = f1_score(yc_val, gs_clf.best_estimator_.predict(X_val), average="weighted")

    best_clf = gs_clf.best_estimator_
    best_clf.fit(X_tv, yc_tv)
    f1_w = f1_score(yc_te, best_clf.predict(X_te), average="weighted")

    # Causal analysis (safe NaN handling)
    r, p = _safe_pearsonr(df["exposition_bin"].fillna(0).values,
                           df["h3_score_attitude"].fillna(df["h3_score_attitude"].mean()).values)
    ate_high = df.loc[df["exposition_bin"] == 1, "h3_score_attitude"].mean()
    ate_low  = df.loc[df["exposition_bin"] == 0, "h3_score_attitude"].mean()
    ate = round(float(ate_high - ate_low) if not np.isnan(ate_high - ate_low) else 0.0, 3)

    metrics = {"mae": round(mae,4), "f1_weighted_clf": round(f1_w,4),
               "pearson_r": round(r,4), "pearson_p": round(p,4), "ate": ate,
               "val_mae": round(mae_val,4), "val_f1": round(f1_val,4)}
    thresholds_met = {
        "mae":         mae  <= c["thresholds"]["mae"],
        "f1_weighted": f1_w >= c["thresholds"]["f1_weighted"],
        "causal_sig":  p    <  c["thresholds"]["pearson_p"] and r > 0,
    }

    _nested = mlflow.active_run() is not None
    with mlflow.start_run(run_name=f"H3_tuned_{datetime.now().strftime('%Y%m%d_%H%M')}",
                          nested=_nested):
        mlflow.set_tag("phase", "9_tuning")
        mlflow.log_param("best_params_reg", str(gs_reg.best_params_))
        mlflow.log_param("best_params_clf", str(gs_clf.best_params_))
        log_metrics(metrics); log_thresholds_met(thresholds_met)

        ts = datetime.now().strftime('%Y%m%d_%H%M')
        for name, m in [("reg_tuned", best_reg), ("clf_tuned", best_clf)]:
            mp = f"models/h3/h3_{name}_{ts}.pkl"
            with open(mp, "wb") as fh: pickle.dump(m, fh)
            log_model_artifact(mp)

        result = {"hypothese": "H3", "metrics": metrics, "thresholds_met": thresholds_met,
                  "best_params": {"reg": gs_reg.best_params_, "clf": gs_clf.best_params_}}
        log_artifact_json(result, f"h3_tuned_{ts}.json")
        mlflow.set_tag("status", "✅ OK" if all(thresholds_met.values()) else "⚠️ Seuils non atteints")

    log.info(f"H3 TUNED — MAE={mae:.3f} | F1={f1_w:.3f} | r={r:.3f}(p={p:.4f}) | "
             f"{'✅' if all(thresholds_met.values()) else '⚠️'}")
    return result


def tune_h4(cfg: dict) -> dict:
    c = cfg["h4"]; dc = cfg["data"]; mc = cfg["mlflow"]
    tun = c.get("tuning", {})
    setup_mlflow(mc["tracking_uri"], mc["experiments"]["h4"])

    df = pd.read_csv(f"{dc['processed_dir']}/h4_features.csv")
    feat_cols = (
        [c2 for c2 in ["interet_camarades_ord","interet_camarades_sent",
                        "souhait_freq","age","sexe_bin"] if c2 in df.columns]
        + [c2 for c2 in df.columns if c2.startswith(("vi_disc_","region_","emb_pca_"))]
    )
    X   = df[feat_cols].fillna(0)
    y_A = df["h4_target_motivation"].fillna(0).astype(int)
    y_B = df["h4_engagement_score"].fillna(1).astype(int)
    disc_cols = [c2 for c2 in df.columns if c2.startswith("vd_disc_")]
    y_C = df[disc_cols].fillna(0).astype(int)

    # 3-way split 70/15/15
    X_tr, X_te, yA_tr, yA_te, yB_tr, yB_te, yC_tr, yC_te = train_test_split(
        X, y_A, y_B, y_C, test_size=dc["test_size"],
        stratify=y_A, random_state=dc["random_state"])
    X_tr, X_val, yA_tr, yA_val, yB_tr, yB_val, yC_tr, yC_val = train_test_split(
        X_tr, yA_tr, yB_tr, yC_tr, test_size=_val_frac(dc),
        stratify=yA_tr, random_state=dc["random_state"])

    cv = tun.get("cv_folds", 3)
    n_est   = tun.get("n_estimators", [100, 200, 300])
    depths  = tun.get("max_depth", [3, 4, 5])
    lrs     = tun.get("learning_rate", [0.05, 0.08, 0.10, 0.15])
    spw     = tun.get("scale_pos_weight", [1, 2, 5, 10])

    # Target A — binary motivation; use pipe() in CV (scale_pos_weight handles imbalance)
    # _imb_pipe() is NOT used in GridSearchCV because severe imbalance can give folds with 1 class
    grid_A = {"model__n_estimators": n_est, "model__max_depth": depths,
               "model__learning_rate": lrs, "model__scale_pos_weight": spw}
    gs_A = GridSearchCV(
        pipe(xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss",
                                random_state=dc["random_state"])),
        grid_A, cv=StratifiedKFold(cv, shuffle=True, random_state=dc["random_state"]),
        scoring="f1", n_jobs=-1, refit=True, error_score=0.0)
    gs_A.fit(X_tr, yA_tr)
    f1_A_val = f1_score(yA_val, gs_A.best_estimator_.predict(X_val), zero_division=0)

    X_tv = pd.concat([X_tr, X_val])
    yA_tv = pd.concat([yA_tr, yA_val])
    yB_tv = pd.concat([yB_tr, yB_val])
    yC_tv = pd.concat([yC_tr, yC_val])
    # Retrain on train+val (strip pipeline prefix from best_params)
    params_A = {k.replace("model__", ""): v for k, v in gs_A.best_params_.items()}
    xgb_A = xgb.XGBClassifier(**params_A, use_label_encoder=False,
                               eval_metric="logloss", random_state=dc["random_state"])
    # Use ROS only when both classes present; else plain pipeline
    best_A = _imb_pipe(xgb_A) if yA_tv.nunique() > 1 else pipe(xgb_A)
    best_A.fit(X_tv, yA_tv)
    f1_A = f1_score(yA_te, best_A.predict(X_te), zero_division=0)

    # Target B — ordinal (multi-class) grid
    grid_B = {"model__n_estimators": n_est, "model__max_depth": depths,
               "model__learning_rate": lrs}
    gs_B = GridSearchCV(
        pipe(xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss",
                                num_class=4, objective="multi:softprob",
                                random_state=dc["random_state"])),
        grid_B, cv=cv, scoring="f1_weighted", n_jobs=-1, refit=True)
    gs_B.fit(X_tr, yB_tr - 1)
    best_B = gs_B.best_estimator_
    best_B.fit(X_tv, yB_tv - 1)
    yB_pred = best_B.predict(X_te) + 1
    _sr = spearmanr(yB_te, yB_pred)
    rho = float(np.asarray(_sr.statistic).flat[0])

    # Target C — multi-label disciplines grid (ClassifierChain)
    grid_C = {"model__base_estimator__n_estimators": n_est[:2],
               "model__base_estimator__max_depth":   depths[:2]}
    gs_C = GridSearchCV(
        pipe(ClassifierChain(xgb.XGBClassifier(use_label_encoder=False,
             eval_metric="logloss", random_state=dc["random_state"]),
             random_state=dc["random_state"])),
        grid_C, cv=cv, scoring="f1_micro", n_jobs=-1, refit=True, error_score=0.0)
    gs_C.fit(X_tr, yC_tr)
    best_C = gs_C.best_estimator_
    best_C.fit(X_tv, yC_tv)
    sub_acc = accuracy_score(yC_te, best_C.predict(X_te))

    disc_labels = [d.replace("vd_disc_","") for d in disc_cols]
    chain_p = best_C.predict_proba(X_te)  # (n_samples, n_labels) for ClassifierChain
    ranking = sorted(zip(disc_labels, chain_p.mean(axis=0).tolist()),
                     key=lambda x: x[1], reverse=True)

    metrics = {"f1_A": round(f1_A,4), "spearman_B": round(rho,4),
               "subset_acc_C": round(sub_acc,4), "val_f1_A": round(f1_A_val,4)}
    thresholds_met = {
        "f1_A":       f1_A    >= c["thresholds"]["f1_A"],
        "spearman_B": rho     >= c["thresholds"]["spearman_B"],
        "subset_C":   sub_acc >= c["thresholds"]["subset_C"],
    }

    _nested = mlflow.active_run() is not None
    with mlflow.start_run(run_name=f"H4_tuned_{datetime.now().strftime('%Y%m%d_%H%M')}",
                          nested=_nested):
        mlflow.set_tag("phase", "9_tuning")
        mlflow.log_param("best_params_A", str(gs_A.best_params_))
        mlflow.log_param("best_params_B", str(gs_B.best_params_))
        mlflow.log_param("best_params_C", str(gs_C.best_params_))
        mlflow.log_param("discipline_top1", ranking[0][0] if ranking else "N/A")
        log_metrics(metrics); log_thresholds_met(thresholds_met)

        ts = datetime.now().strftime('%Y%m%d_%H%M')
        for name, m in [("A_motivation_tuned", best_A),
                        ("B_engagement_tuned", best_B),
                        ("C_discipline_tuned", best_C)]:
            mp = f"models/h4/h4_{name}_{ts}.pkl"
            with open(mp,"wb") as fh: pickle.dump(m, fh)
            log_model_artifact(mp)

        result = {"hypothese": "H4", "metrics": metrics, "thresholds_met": thresholds_met,
                  "discipline_ranking": [{"disc": d, "proba": round(p,4)} for d,p in ranking],
                  "best_params": {"A": gs_A.best_params_, "B": gs_B.best_params_,
                                  "C": gs_C.best_params_}}
        log_artifact_json(result, f"h4_tuned_{ts}.json")
        mlflow.set_tag("status", "✅ OK" if all(thresholds_met.values()) else "⚠️ Seuils non atteints")

    log.info(f"H4 TUNED — F1-A={f1_A:.3f} | rho={rho:.3f} | SubAcc={sub_acc:.3f} | "
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
