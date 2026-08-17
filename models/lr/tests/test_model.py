import torch

from models.lr.model import LogisticRegression


def test_model_returns_one_logit_per_sample() -> None:
    model = LogisticRegression(num_features=5)
    features = torch.randn(8, 5)

    logits = model(features)

    assert logits.shape == (8,)
    assert torch.isfinite(logits).all()
