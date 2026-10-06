import json

import pytest
import torch

from models.sparse_ctr.data import prepare_data
from models.sparse_ctr.train import make_model, parse_args, predict, train
from models.lr.train import evaluate


@pytest.mark.parametrize("name", ["lr", "fm", "mlp"])
def test_checkpoint_reproduces_test_metrics(name, tmp_path):
    run = train(parse_args([
        "--model", name, "--samples", "1000", "--epochs", "3",
        "--output-dir", str(tmp_path),
    ]))
    metrics = json.loads((run / "metrics.json").read_text())
    history = json.loads((run / "history.json").read_text())
    checkpoint = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    model = make_model(name, sum(checkpoint["vocab_sizes"]), 8)
    model.load_state_dict(checkpoint["state_dict"])
    data = prepare_data(samples=1000, seed=42)
    probabilities, _ = predict(model, data.test_cat, data.test_num, data.test_y, 512)

    assert metrics["best_val_loss"] == min(row["val_loss"] for row in history)
    assert evaluate(data.test_y, probabilities)["log_loss"] == pytest.approx(metrics["log_loss"])
    assert checkpoint["vocab_sizes"] == data.vocab_sizes
