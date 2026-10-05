"""Second-order Factorization Machine for dense float features."""

import torch
from torch import nn


class FactorizationMachine(nn.Module):
    """Return CTR logits with a linear term and learned pair interactions."""

    def __init__(self, num_features: int, embedding_dim: int = 8) -> None:
        super().__init__()
        if num_features < 2:
            raise ValueError("num_features must be at least 2")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.linear = nn.Linear(num_features, 1)
        self.factors = nn.Parameter(torch.empty(num_features, embedding_dim))
        nn.init.normal_(self.factors, mean=0.0, std=0.1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        linear = self.linear(features).squeeze(-1)
        projected = features @ self.factors
        interactions = 0.5 * (
            projected.square() - features.square() @ self.factors.square()
        ).sum(dim=-1)
        return linear + interactions
