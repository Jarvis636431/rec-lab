import numpy as np
import pytest

from models.lr.train import evaluate, evaluate_always_negative, experiment_name


def test_evaluate_reports_ranking_probability_and_classification_metrics() -> None:
    labels = np.array([0, 0, 1, 1], dtype=np.float32)
    probabilities = np.array([0.1, 0.4, 0.6, 0.9], dtype=np.float32)

    metrics = evaluate(labels, probabilities)

    assert metrics["auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0


def test_always_negative_has_high_accuracy_but_zero_recall() -> None:
    labels = np.array([0] * 99 + [1], dtype=np.float32)

    metrics = evaluate_always_negative(labels)

    assert metrics["accuracy"] == 0.99
    assert metrics["recall"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["auc"] == 0.5


def test_configured_experiments_do_not_overwrite_legacy_outputs() -> None:
    assert experiment_name(include_cross=False, base_ctr=None) == "baseline"
    assert experiment_name(include_cross=True, base_ctr=None) == "with_cross"
    assert (
        experiment_name(include_cross=False, base_ctr=0.005)
        == "baseline_base_ctr_0p005"
    )


@pytest.mark.parametrize("label", [0, 1])
def test_single_class_metrics_are_explicit_and_probability_metrics_still_work(label) -> None:
    labels = np.full(15, label, dtype=np.float32)
    probabilities = np.full(15, 0.01 if label == 0 else 0.99, dtype=np.float32)

    metrics = evaluate(labels, probabilities)

    assert metrics["auc"] is None
    assert metrics["pr_auc"] is None
    assert metrics["log_loss"] == pytest.approx(-np.log(0.99), abs=1e-6)
    assert metrics["ece"] == pytest.approx(0.01, abs=1e-6)
    assert metrics["accuracy"] == 1.0


@pytest.mark.parametrize("probabilities", [
    np.array([]), np.array([0.1]), np.array([np.nan, 0.1]),
    np.array([0.1, 1.1]), np.array([0.1, -0.1]),
])
def test_invalid_predictions_are_rejected(probabilities) -> None:
    with pytest.raises(ValueError):
        evaluate(np.array([0, 1]), probabilities)
