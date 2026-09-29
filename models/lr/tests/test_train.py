import importlib
import json

import numpy as np
import pytest
import torch

from models.lr.data import generate_exposures, prepare_data
from models.lr.model import LogisticRegression

training = importlib.import_module("models.lr.train")


def read_json(path):
    # Reject Python's nonstandard NaN/Infinity JSON extensions as well.
    def reject_constant(value):
        raise AssertionError(f"invalid JSON constant: {value}")

    return json.loads(path.read_text(), parse_constant=reject_constant)


def test_early_stopping_restores_best_state_before_test_and_checkpoint(tmp_path, monkeypatch):
    args = training.parse_args([
        "--samples", "1000", "--epochs", "8", "--patience", "2",
        "--output-dir", str(tmp_path),
    ])
    original_predict = training.predict
    validation_losses = iter([0.4, 0.3, 0.35, 0.36])
    snapshots = []
    test_calls = []

    def controlled_predict(model, features, labels, batch_size):
        if len(snapshots) < 4:
            snapshots.append({key: value.clone() for key, value in model.state_dict().items()})
            return np.zeros(len(labels)), next(validation_losses)
        # A fifth call must be the final test prediction, using epoch 2's weights.
        test_calls.append(True)
        for key, value in model.state_dict().items():
            assert torch.equal(value, snapshots[1][key])
        return original_predict(model, features, labels, batch_size)

    monkeypatch.setattr(training, "predict", controlled_predict)
    run = training.train(args)
    assert len(test_calls) == 1
    history = read_json(run / "history.json")
    metrics = read_json(run / "metrics.json")
    config = read_json(run / "config.json")
    assert [row["val_loss"] for row in history] == [0.4, 0.3, 0.35, 0.36]
    assert metrics["best_epoch"] == 2
    assert metrics["epochs_completed"] == 4
    assert metrics["stopped_early"] is True
    assert config["arguments"]["patience"] == 2
    assert config["data"]["splits"]["train"]["samples"] == 700

    checkpoint = torch.load(run / "model.pt", weights_only=True, map_location="cpu")
    reloaded = LogisticRegression(len(checkpoint["feature_names"]))
    reloaded.load_state_dict(checkpoint["state_dict"])
    data = prepare_data(samples=1000, seed=42)
    probabilities, _ = original_predict(reloaded, data.test_x, data.test_y, 256)
    assert training.evaluate(data.test_y, probabilities)["log_loss"] == pytest.approx(
        metrics["log_loss"], abs=1e-10,
    )
    assert torch.equal(checkpoint["mean"], torch.from_numpy(data.mean))
    assert torch.equal(checkpoint["std"], torch.from_numpy(data.std))
    exposures = generate_exposures(samples=1000, seed=42)
    raw_features = torch.from_numpy(np.column_stack([
        exposures[name][850:] for name in checkpoint["feature_names"]
    ]))
    with torch.no_grad():
        restored_features = (raw_features - checkpoint["mean"]) / checkpoint["std"]
        restored_probabilities = torch.sigmoid(reloaded(restored_features)).numpy()
    np.testing.assert_allclose(restored_probabilities, probabilities, atol=1e-7)
    weights = read_json(run / "weights.json")
    assert weights["bias"] == reloaded.linear.bias.item()
    for key, value in checkpoint["state_dict"].items():
        assert torch.equal(value, snapshots[1][key])


def test_repeated_low_ctr_runs_are_complete_reproducible_and_do_not_overwrite(tmp_path):
    args = training.parse_args([
        "--samples", "100", "--epochs", "2", "--base-ctr", "0.005",
        "--output-dir", str(tmp_path),
    ])
    first = training.train(args)
    first_metrics_text = (first / "metrics.json").read_text()
    second = training.train(args)
    assert first != second
    assert first.parent == second.parent
    assert (first / "metrics.json").read_text() == first_metrics_text
    for run in (first, second):
        assert {path.name for path in run.iterdir()} == {
            "config.json", "history.json", "metrics.json", "weights.json", "model.pt",
        }
        metrics = read_json(run / "metrics.json")
        assert metrics["auc"] is None
        assert metrics["pr_auc"] is None
        assert metrics["splits"]["test"]["positives"] == 0
        assert metrics["epochs_completed"] == 2
        assert metrics["stopped_early"] is False
        assert metrics["best_val_loss"] == min(
            row["val_loss"] for row in read_json(run / "history.json")
        )
    assert read_json(first / "metrics.json") == read_json(second / "metrics.json")
    assert read_json(first / "weights.json") == read_json(second / "weights.json")


@pytest.mark.parametrize("argument,value", [
    ("--epochs", "0"), ("--batch-size", "0"), ("--learning-rate", "nan"),
    ("--learning-rate", "inf"), ("--patience", "-1"),
])
def test_invalid_training_config_does_not_create_output(tmp_path, argument, value):
    args = training.parse_args([argument, value, "--output-dir", str(tmp_path)])
    with pytest.raises(ValueError):
        training.train(args)
    assert list(tmp_path.iterdir()) == []
