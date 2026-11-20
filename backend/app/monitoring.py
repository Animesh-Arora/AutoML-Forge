"""Model-quality monitoring helpers: validate a new batch, evaluate the
champion, retrain a candidate and decide on promotion with a quality gate."""
from __future__ import annotations

import pandas as pd
from sklearn.base import clone

from .metrics import calculate_metrics


def validate_batch(frame: pd.DataFrame, required_columns: list[str], target: str,
                   task_type: str = "classification") -> dict:
    """Check schema, missing values and duplicates of an incoming batch.

    Missing feature values are only reported because the pipeline imputes them.
    Missing columns or an unusable target stop the workflow.
    """
    missing_columns = [c for c in required_columns if c not in frame.columns]
    if missing_columns:
        raise ValueError(f"New batch is missing required columns: {missing_columns}")

    report = {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "required_columns": list(required_columns),
        "extra_columns": [c for c in frame.columns if c not in required_columns],
        "missing_values": {c: int(n) for c, n in frame[required_columns].isna().sum().items()},
        "missing_target_rows": int(frame[target].isna().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
    }

    clean = frame[required_columns].drop_duplicates().dropna(subset=[target])
    report["rows_after_cleaning"] = int(len(clean))
    if task_type == "classification":
        report["target_distribution"] = {str(k): int(v) for k, v in clean[target].value_counts().items()}
    if clean[target].nunique() < 2:
        raise ValueError("New batch needs at least two target classes after cleaning.")
    return report


def clean_batch(frame: pd.DataFrame, required_columns: list[str], target: str) -> pd.DataFrame:
    return frame[required_columns].drop_duplicates().dropna(subset=[target]).reset_index(drop=True)


def evaluate_model(model, X: pd.DataFrame, y: pd.Series, task_type: str = "classification") -> dict:
    metrics = calculate_metrics(y, model.predict(X), task_type)
    metrics["rows"] = int(len(y))
    return metrics


def retrain_candidate(champion, X: pd.DataFrame, y: pd.Series):
    """Refit an unfitted copy of the champion pipeline (same preprocessing
    and algorithm settings) on the combined data."""
    return clone(champion).fit(X, y)


def quality_gate(champion_metrics: dict, candidate_metrics: dict, metric: str = "f1",
                 min_score: float = 0.75, min_improvement: float = 0.01) -> dict:
    """Candidate is eligible only if it clears an absolute floor and beats the
    champion by at least ``min_improvement`` on the same holdout data."""
    champ = float(champion_metrics[metric])
    cand = float(candidate_metrics[metric])
    checks = {
        f"candidate_{metric} >= {min_score}": cand >= min_score,
        f"candidate_{metric} - champion_{metric} >= {min_improvement}": round(cand - champ, 6) >= min_improvement,
    }
    eligible = all(checks.values())
    return {
        "metric": metric,
        "champion_score": champ,
        "candidate_score": cand,
        "improvement": round(cand - champ, 6),
        "checks": checks,
        "decision": "candidate_eligible" if eligible else "keep_champion",
        "eligible_for_promotion": eligible,
    }
