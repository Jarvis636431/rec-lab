import json

import numpy as np
import torch

from models.fm.data import prepare_data
from models.fm.model import FactorizationMachine
from models.fm.train import parse_args, train
from models.lr.data import prepare_data as prepare_lr_data
from models.lr.train import evaluate, predict


def test_fm_uses_same_data_without_manual_cross():
    fm_data = prepare_data(samples=1000, seed=7)
    lr_data = prepare_lr_data(samples=1000, seed=7, include_cross=True)
    assert fm_data.train_x.shape == (700, 5)
    assert "sports_match" not in fm_data.feature_names
    np.testing.assert_array_equal(fm_data.train_y, lr_data.train_y)
    np.testing.assert_array_equal(fm_data.val_y, lr_data.val_y)
    np.testing.assert_array_equal(fm_data.test_y, lr_data.test_y)
    np.testing.assert_array_equal(fm_data.train_x, lr_data.train_x[:, :5])


def test_checkpoint_restores_best_model_and_metrics(tmp_path):
    run = train(parse_args(["--samples", "1000", "--epochs", "3",
                            "--embedding-dim", "3", "--output-dir", str(tmp_path)]))
    metrics = json.loads((run / "metrics.json").read_text())
    history = json.loads((run / "history.json").read_text())
    checkpoint = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    model = FactorizationMachine(5, 3)
    model.load_state_dict(checkpoint["state_dict"])
    data = prepare_data(samples=1000, seed=42)
    probabilities, _ = predict(model, data.test_x, data.test_y, 256)

    assert metrics["best_val_loss"] == min(row["val_loss"] for row in history)
    assert metrics["best_epoch"] == min(history, key=lambda row: row["val_loss"])["epoch"]
    assert np.isclose(evaluate(data.test_y, probabilities)["log_loss"], metrics["log_loss"])
    assert checkpoint["feature_names"] == data.feature_names
    torch.testing.assert_close(checkpoint["mean"], torch.from_numpy(data.mean))
