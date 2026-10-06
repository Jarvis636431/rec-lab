import json

import pytest
import torch

from models.deepfm.data import prepare_data
from models.deepfm.model import DeepFM
from models.deepfm.train import parse_args, train
from models.lr.train import evaluate
from models.sparse_ctr.train import predict


@pytest.mark.parametrize("variant", ["fm_only", "deep_only", "deepfm"])
def test_checkpoint_restores_best_and_reproduces_metric(variant, tmp_path):
    run = train(parse_args([
        "--variant", variant, "--samples", "1000", "--epochs", "3",
        "--output-dir", str(tmp_path),
    ]))
    config = json.loads((run / "config.json").read_text())
    metrics = json.loads((run / "metrics.json").read_text())
    history = json.loads((run / "history.json").read_text())
    checkpoint = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    data = prepare_data(samples=1000, seed=42)
    model = DeepFM(sum(checkpoint["vocab_sizes"]), variant=variant)
    model.load_state_dict(checkpoint["state_dict"])
    probabilities, _ = predict(model, data.test_cat, data.test_num, data.test_y, 512)

    assert metrics["best_val_loss"] == min(row["val_loss"] for row in history)
    assert evaluate(data.test_y, probabilities)["log_loss"] == pytest.approx(
        metrics["log_loss"]
    )
    assert checkpoint["vocab_sizes"] == data.vocab_sizes
    assert config["data"]["vocabulary_fit"] == "train_only"
