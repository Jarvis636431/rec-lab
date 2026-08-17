"""The smallest useful CTR model: one linear layer."""

import torch
from torch import nn


class LogisticRegression(nn.Module):
    """Return logits for binary click prediction."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be positive")
        self.linear = nn.Linear(num_features, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(-1)
