import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from app.automl import make_pipeline
from app.monitoring import clean_batch, evaluate_model, quality_gate, retrain_candidate, validate_batch

COLUMNS = ["age", "income", "city", "purchased"]


def make_batch():
    return pd.DataFrame({
        "age": [22, 35, 41, 28, 35, 50],
        "income": [28000, 52000, None, 39000, 52000, 80000],
        "city": ["Agra", "Delhi", "Agra", None, "Delhi", "Jaipur"],
        "purchased": ["no", "yes", "yes", "no", "yes", "yes"],
    })


def test_validate_batch_reports_missing_and_duplicates():
    report = validate_batch(make_batch(), COLUMNS, "purchased")
    assert report["rows"] == 6
    assert report["duplicate_rows"] == 1
    assert report["missing_values"]["income"] == 1
    assert report["missing_values"]["city"] == 1
    assert report["rows_after_cleaning"] == 5
    assert report["target_distribution"] == {"yes": 3, "no": 2}


def test_validate_batch_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        validate_batch(make_batch().drop(columns=["city"]), COLUMNS, "purchased")


def test_validate_batch_rejects_single_class():
    batch = make_batch().assign(purchased="yes")
    with pytest.raises(ValueError, match="two target classes"):
        validate_batch(batch, COLUMNS, "purchased")


def test_retrain_keeps_champion_untouched():
    batch = clean_batch(make_batch(), COLUMNS, "purchased")
    X, y = batch.drop(columns="purchased"), batch["purchased"]
    champion = make_pipeline(X, LogisticRegression()).fit(X.iloc[:4], y.iloc[:4])
    before = champion.named_steps["model"].coef_.copy()
    candidate = retrain_candidate(champion, X, y)
    assert candidate is not champion
    assert (champion.named_steps["model"].coef_ == before).all()
    assert set(evaluate_model(candidate, X, y)) == {"accuracy", "precision", "recall", "f1", "rows"}


def test_quality_gate_promotes_better_candidate():
    gate = quality_gate({"f1": 0.80}, {"f1": 0.86}, "f1", min_score=0.75, min_improvement=0.01)
    assert gate["eligible_for_promotion"] is True
    assert gate["decision"] == "candidate_eligible"


@pytest.mark.parametrize("champion, candidate", [
    (0.80, 0.805),  # not enough improvement
    (0.60, 0.70),   # better, but below the minimum score
    (0.90, 0.85),   # worse
])
def test_quality_gate_keeps_champion(champion, candidate):
    gate = quality_gate({"f1": champion}, {"f1": candidate}, "f1", min_score=0.75, min_improvement=0.01)
    assert gate["eligible_for_promotion"] is False
    assert gate["decision"] == "keep_champion"
