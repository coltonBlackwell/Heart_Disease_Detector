import json
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_model_and_metadata(project_root: Path):
    models_dir = project_root / "models"
    model_path = models_dir / "model.joblib"
    meta_path = models_dir / "metadata.json"

    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Trained model or metadata not found. Run training via the Streamlit app first.")

    model = joblib.load(model_path)
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    return model, metadata


def load_eval_holdout(project_root: Path):
    eval_path = project_root / "models" / "eval" / "holdout.joblib"
    if not eval_path.exists():
        raise FileNotFoundError("Missing eval artifacts. Train once to generate holdout.joblib.")
    art = joblib.load(eval_path)
    return art["y_test"], art["y_proba"]


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray):
    y_pred = (y_proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }


def compare_to_baseline(current: dict, baseline: dict, tolerances: dict):
    deltas = {}
    alerts = []
    for k, v in current.items():
        if k not in baseline:
            continue
        delta = v - baseline[k]
        deltas[k] = delta
        tol = tolerances.get(k, 0.0)
        if abs(delta) > tol:
            alerts.append({
                "metric": k,
                "baseline": baseline[k],
                "current": v,
                "delta": delta,
                "tolerance": tol,
            })
    return deltas, alerts


def main():
    project_root = Path(__file__).resolve().parents[2]
    # Load trained model and metadata
    _, metadata = load_model_and_metadata(project_root)
    y_test, y_proba = load_eval_holdout(project_root)

    current_metrics = compute_metrics(np.asarray(y_test), np.asarray(y_proba))

    baseline = metadata.get("metrics", {})
    tolerances = {
        "accuracy": 0.03,
        "precision": 0.05,
        "recall": 0.05,
        "roc_auc": 0.02,
        "pr_auc": 0.03,
    }

    deltas, alerts = compare_to_baseline(current_metrics, baseline, tolerances)

    monitor_dir = project_root / "data" / "logs" / "monitor"
    _ensure_dir(monitor_dir)

    report = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "model_trained_at": metadata.get("trained_at"),
        "data_path": metadata.get("data_path"),
        "data_sha256": metadata.get("data_sha256"),
        "baseline_metrics": baseline,
        "current_metrics": current_metrics,
        "deltas": deltas,
        "alerts": alerts,
        "status": "alert" if alerts else "ok",
    }

    # Write daily file and latest pointer
    day_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    day_file = monitor_dir / f"report_{day_key}.json"
    with open(day_file, "w") as f:
        json.dump(report, f, indent=2)

    with open(monitor_dir / "latest.json", "w") as f:
        json.dump(report, f, indent=2)

    # Print status for CI logs
    print(json.dumps({"status": report["status"], "alerts": alerts}))

    # Non-zero exit on alert to fail the scheduled job if drift detected
    if alerts:
        raise SystemExit(2)


if __name__ == "__main__":
    main()


