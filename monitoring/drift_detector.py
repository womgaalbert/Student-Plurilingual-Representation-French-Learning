"""
drift_detector.py — Data drift detection (Evidently AI + scipy fallback)
French-Learning-Perceptions ML — MLOps Level 3

Usage:
    from monitoring.drift_detector import detect_drift, build_reference_data

    ref = build_reference_data("data/processed/h1_features.csv", "h1")
    result = detect_drift(ref, current_df, "h1", drift_threshold=0.3)
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def build_reference_data(
    processed_csv: Path,
    model_key: str,
    feature_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load training data from data/processed/*.csv as drift reference.

    Args:
        processed_csv: Path to e.g. data/processed/h1_features.csv.
        model_key: Model identifier (h1, h2, h3_reg, etc.).
        feature_cols: Columns to keep. If None, keeps all.

    Returns:
        pd.DataFrame with reference feature values.
    """
    if not processed_csv.exists():
        raise FileNotFoundError(f"Reference data not found: {processed_csv}")

    df = pd.read_csv(processed_csv)
    if feature_cols:
        present = [c for c in feature_cols if c in df.columns]
        df = df[present]
    log.info("Reference data loaded for %s: %d rows × %d cols",
             model_key, len(df), len(df.columns))
    return df


def compute_statistical_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    drift_threshold: float = 0.3,
) -> dict:
    """Lightweight drift detection using scipy.stats (no Evidently required).

    Kolmogorov-Smirnov for numerical, chi-squared for categorical columns.
    Returns the same structure as detect_drift().
    """
    from scipy.stats import ks_2samp, chi2_contingency

    scores = {}
    common_cols = [c for c in reference.columns if c in current.columns]
    if not common_cols:
        return _empty_result("no_common_columns")

    for col in common_cols:
        try:
            ref_vals = reference[col].dropna().astype(float).values
            cur_vals = current[col].dropna().astype(float).values
            if len(ref_vals) < 5 or len(cur_vals) < 5:
                continue
            stat, pvalue = ks_2samp(ref_vals, cur_vals)
            scores[col] = float(pvalue)
        except (ValueError, TypeError):
            # Categorical fallback via value counts
            try:
                ref_counts = reference[col].value_counts()
                cur_counts = current[col].value_counts()
                all_cats = sorted(set(ref_counts.index) | set(cur_counts.index))
                tbl = []
                for cat in all_cats:
                    tbl.append([
                        ref_counts.get(cat, 0),
                        cur_counts.get(cat, 0),
                    ])
                if len(tbl) >= 2 and all(sum(r) > 0 for r in tbl):
                    _, pvalue, _, _ = chi2_contingency(tbl)
                    scores[col] = float(pvalue)
            except Exception:
                continue

    n_features = len(common_cols)
    n_drifted = sum(1 for p in scores.values() if p < 0.05)
    drift_ratio = n_drifted / n_features if n_features > 0 else 0.0

    return {
        "model_key": "unknown",
        "timestamp": datetime.now().isoformat(),
        "n_features": n_features,
        "n_drifted_features": n_drifted,
        "drift_ratio": round(drift_ratio, 4),
        "drift_detected": drift_ratio >= drift_threshold,
        "feature_drift_scores": scores,
        "report_path": None,
        "method": "scipy",
    }


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    model_key: str,
    drift_threshold: float = 0.3,
    output_dir: Optional[Path] = None,
) -> dict:
    """Compare current vs reference distribution using Evidently AI.

    Falls back to compute_statistical_drift() if Evidently is not installed.
    """
    if reference.empty or current.empty:
        return _empty_result("empty_data")

    common_cols = [c for c in reference.columns if c in current.columns]
    if not common_cols:
        return _empty_result("no_common_columns")

    # Try Evidently — fall back to scipy if not available
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        from evidently import ColumnMapping

        reference = reference[common_cols]
        current = current[common_cols]

        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=reference,
            current_data=current,
            column_mapping=None,
        )
        result_dict = report.as_dict()

        # Extract drift scores
        drift_metrics = result_dict.get("metrics", [])
        n_drifted = 0
        n_features = 0
        scores = {}
        for metric in drift_metrics:
            if metric.get("metric") == "DataDriftTable":
                table = metric.get("result", {}).get("drift_by_columns", {})
                for col, info in table.items():
                    n_features += 1
                    if info.get("drift_detected"):
                        n_drifted += 1
                    scores[col] = info.get("drift_score", 1.0)

        drift_ratio = n_drifted / n_features if n_features > 0 else 0.0
        detected = drift_ratio >= drift_threshold

        # Save HTML report if output_dir specified
        report_path = None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            html_path = output_dir / f"drift_{model_key}_{ts}.html"
            report.save_html(str(html_path))
            json_path = output_dir / f"drift_{model_key}_{ts}.json"
            json_path.write_text(json.dumps(result_dict, indent=2, default=str))
            report_path = str(html_path)
            log.info("Drift report saved: %s", report_path)

        return {
            "model_key": model_key,
            "timestamp": datetime.now().isoformat(),
            "n_features": n_features,
            "n_drifted_features": n_drifted,
            "drift_ratio": round(drift_ratio, 4),
            "drift_detected": detected,
            "feature_drift_scores": scores,
            "report_path": report_path,
            "method": "evidently",
        }

    except ImportError:
        log.info("Evidently not installed — using scipy fallback")
        result = compute_statistical_drift(reference, current, drift_threshold)
        result["model_key"] = model_key
        return result


def _empty_result(reason: str) -> dict:
    return {
        "model_key": "unknown",
        "timestamp": datetime.now().isoformat(),
        "n_features": 0,
        "n_drifted_features": 0,
        "drift_ratio": 0.0,
        "drift_detected": False,
        "feature_drift_scores": {},
        "report_path": None,
        "reason": reason,
    }
