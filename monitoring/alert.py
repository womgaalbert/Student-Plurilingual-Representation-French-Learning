"""
alert.py — Alerting system for drift detection
French-Learning-Perceptions ML — MLOps Level 3

Usage:
    from monitoring.alert import write_alert, check_and_alert

    check_and_alert(drift_result, Path("monitoring/alerts/alerts.ndjson"))
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)

INFO = "INFO"
WARNING = "WARNING"
CRITICAL = "CRITICAL"

_SEVERITY_ORDER = {INFO: 0, WARNING: 1, CRITICAL: 2}


def write_alert(
    alert_file: Path,
    model_key: str,
    severity: str,
    message: str,
    details: Optional[dict] = None,
) -> None:
    """Append a structured JSON alert to the .ndjson log file."""
    alert_file.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "severity": severity,
        "model_key": model_key,
        "message": message,
        "details": details or {},
    }
    try:
        with open(alert_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        log.info("Alert [%s] %s: %s", severity, model_key, message)
    except OSError as e:
        log.error("Failed to write alert: %s", e)


def check_and_alert(
    drift_result: dict,
    alert_file: Path,
    severity_map: Optional[dict] = None,
) -> Optional[str]:
    """Evaluate drift result and write alert if thresholds exceeded.

    Default severity map: drift_ratio < 0.3 → no alert,
                          drift_ratio 0.3-0.5 → WARNING,
                          drift_ratio > 0.5 → CRITICAL.
    """
    if severity_map is None:
        severity_map = {
            (0.0, 0.3): None,       # pas d'alerte
            (0.3, 0.5): WARNING,
            (0.5, 1.0): CRITICAL,
        }

    ratio = drift_result.get("drift_ratio", 0.0)
    model_key = drift_result.get("model_key", "unknown")
    n_drifted = drift_result.get("n_drifted_features", 0)
    n_total = drift_result.get("n_features", 0)

    severity = None
    for (lo, hi), sev in severity_map.items():
        if lo <= ratio < hi:
            severity = sev
            break

    if severity is None:
        return None

    message = (
        f"Drift detected: {n_drifted}/{n_total} features drifted "
        f"({ratio:.1%}) — threshold exceeded"
    )
    write_alert(alert_file, model_key, severity, message, drift_result)
    return severity


def get_recent_alerts(
    alert_file: Path,
    min_severity: str = WARNING,
    limit: int = 20,
) -> list[dict]:
    """Read recent alerts from .ndjson file, filtered by severity."""
    if not alert_file.exists():
        return []
    min_level = _SEVERITY_ORDER.get(min_severity, 0)
    alerts = []
    try:
        for line in reversed(list(open(alert_file, encoding="utf-8"))):
            if len(alerts) >= limit:
                break
            entry = json.loads(line.strip())
            if _SEVERITY_ORDER.get(entry.get("severity", INFO), 0) >= min_level:
                alerts.append(entry)
    except (OSError, json.JSONDecodeError) as e:
        log.warning("Error reading alerts: %s", e)
    return alerts


def send_github_dispatch(
    drift_result: dict,
    repo: str = "womgaalbert/Student-Plurilingual-Representation-French-Learning",
) -> bool:
    """Trigger retrain workflow via GitHub API repository_dispatch.

    Requires GITHUB_TOKEN env var or gh CLI authenticated.
    Returns True if dispatch was sent, False otherwise.
    """
    import os
    import subprocess
    import sys

    token = os.getenv("GITHUB_TOKEN", "")
    if not token:
        log.info("GITHUB_TOKEN not set — skipping GitHub dispatch")
        return False

    import urllib.request
    url = f"https://api.github.com/repos/{repo}/dispatches"
    body = json.dumps({
        "event_type": "drift_detected",
        "client_payload": {
            "model_key": drift_result.get("model_key", ""),
            "drift_ratio": drift_result.get("drift_ratio", 0.0),
            "timestamp": drift_result.get("timestamp", ""),
        },
    }).encode()

    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("Content-Type", "application/json")

    try:
        urllib.request.urlopen(req, timeout=10)
        log.info("GitHub dispatch sent for drift on %s", drift_result.get("model_key"))
        return True
    except Exception as e:
        log.warning("GitHub dispatch failed: %s", e)
        return False
