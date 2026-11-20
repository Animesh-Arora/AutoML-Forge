"""Monitoring + retraining workflow for the deployed champion model.

Steps: validate new batch -> evaluate champion -> retrain candidate ->
quality gate. The champion is only replaced when --promote is passed AND
the gate passed, and every run is appended to reports/model_decisions.jsonl.

    python scripts/monitor_and_retrain.py
    python scripts/monitor_and_retrain.py --promote
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import joblib
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from app.data_loader import load_csv
from app.mlflow_tracker import MLflowTracker
from app.monitoring import clean_batch, evaluate_model, quality_gate, retrain_candidate, validate_batch

REPORTS = ROOT / "reports"


def write_json(name: str, data: dict) -> Path:
    path = REPORTS / name
    path.write_text(json.dumps(data, indent=2))
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--promote", action="store_true",
                        help="replace the champion if the candidate passed the quality gate")
    args = parser.parse_args()

    params = yaml.safe_load((ROOT / "params.yaml").read_text())
    cfg, seed = params["monitoring"], params["training"]["random_state"]
    target = cfg["target"]
    run_time = datetime.now(timezone.utc).isoformat(timespec="seconds")
    REPORTS.mkdir(exist_ok=True)

    # Task A - prepare the new batch
    champion = joblib.load(ROOT / cfg["champion_model"])
    features = list(champion.feature_names_in_)
    required = features + [target]
    raw_batch = load_csv(ROOT / cfg["new_batch"])
    validation = validate_batch(raw_batch, required, target)
    write_json("new_batch_validation.json", validation)
    batch = clean_batch(raw_batch, required, target)
    print(f"[A] new batch: {validation['rows']} rows -> {len(batch)} after removing "
          f"{validation['duplicate_rows']} duplicates")
    print(f"    missing values: {validation['missing_values']}")
    print(f"    target distribution: {validation['target_distribution']}")

    # Task B - how does the current champion do on the new labelled data?
    champion_eval = evaluate_model(champion, batch[features], batch[target])
    write_json("champion_evaluation.json", {
        "evaluated_at": run_time,
        "model_path": cfg["champion_model"],
        "algorithm": type(champion.named_steps["model"]).__name__,
        "dataset": cfg["new_batch"],
        "metrics": champion_eval,
    })
    print("[B] champion on new batch: " + ", ".join(
        f"{k}={v:.3f}" for k, v in champion_eval.items() if k != "rows"))

    # Task C - retrain on old data + part of the new batch.
    # A slice of the new batch is held out so both models are compared on
    # rows neither of them was trained on.
    new_train, holdout = train_test_split(batch, test_size=cfg["holdout_size"],
                                          random_state=seed, stratify=batch[target])
    reference = pd.read_csv(ROOT / cfg["reference_data"])[required]
    combined = pd.concat([reference, new_train], ignore_index=True).drop_duplicates()
    combined = combined.dropna(subset=[target])
    candidate = retrain_candidate(champion, combined[features], combined[target])
    candidate_path = ROOT / cfg["candidate_model"]
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(candidate, candidate_path)
    print(f"[C] candidate trained on {len(combined)} rows "
          f"({len(reference)} old + {len(new_train)} new), saved to {cfg['candidate_model']}")

    # Task D - quality gate on the holdout
    champ_hold = evaluate_model(champion, holdout[features], holdout[target])
    cand_hold = evaluate_model(candidate, holdout[features], holdout[target])
    gate = quality_gate(champ_hold, cand_hold, cfg["gate_metric"],
                        cfg["min_score"], cfg["min_improvement"])

    # falls back to a local sqlite store when no tracking server is configured
    tracker = MLflowTracker(os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"),
                            os.getenv("MLFLOW_EXPERIMENT_NAME", "AutoML Forge - retraining"))
    mlflow_run = tracker.log_run(
        "retrain-candidate",
        {"algorithm": type(candidate.named_steps["model"]).__name__,
         "new_batch": cfg["new_batch"], "train_rows": len(combined),
         "holdout_rows": len(holdout), "gate_metric": cfg["gate_metric"],
         "decision": gate["decision"]},
        {**{f"candidate_{k}": v for k, v in cand_hold.items() if k != "rows"},
         **{f"champion_{k}": v for k, v in champ_hold.items() if k != "rows"}},
        str(candidate_path),
    )

    decision = {
        "run_at": run_time,
        "champion_model": cfg["champion_model"],
        "candidate_model": cfg["candidate_model"],
        "holdout_rows": len(holdout),
        "champion_holdout_metrics": champ_hold,
        "candidate_holdout_metrics": cand_hold,
        **gate,
        "promoted": False,
        "mlflow_run_id": mlflow_run["run_id"],
    }

    if args.promote and gate["eligible_for_promotion"]:
        archive = ROOT / "models" / "archive"
        archive.mkdir(parents=True, exist_ok=True)
        stamp = run_time.replace(":", "").replace("-", "")[:15]
        shutil.copy2(ROOT / cfg["champion_model"], archive / f"champion_{stamp}.joblib")
        shutil.copy2(candidate_path, ROOT / cfg["champion_model"])
        decision["promoted"] = True
        decision["previous_champion_archived_to"] = f"models/archive/champion_{stamp}.joblib"
    elif args.promote:
        print("    --promote ignored: candidate did not pass the quality gate")

    write_json("quality_gate.json", decision)
    with open(REPORTS / "model_decisions.jsonl", "a") as log:
        log.write(json.dumps({k: decision[k] for k in (
            "run_at", "metric", "champion_score", "candidate_score",
            "decision", "promoted", "mlflow_run_id")}) + "\n")

    m = cfg["gate_metric"]
    print(f"[D] holdout {m}: champion={gate['champion_score']:.3f} "
          f"candidate={gate['candidate_score']:.3f} -> {gate['decision']}"
          + (" (promoted)" if decision["promoted"] else ""))
    print(f"    MLflow logged: {mlflow_run['logged']}  run_id={mlflow_run['run_id']}")


if __name__ == "__main__":
    main()
