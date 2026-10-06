"""Three baselines over identical categorical and numeric input fields."""

import torch
from torch import nn


class SparseLR(nn.Module):
    def __init__(self, total_tokens: int, num_numeric: int = 2) -> None:
        super().__init__()
        self.category_weight = nn.Embedding(total_tokens, 1)
        self.numeric_weight = nn.Linear(num_numeric, 1, bias=False)
        self.bias = nn.Parameter(torch.tensor(-2.0))
        nn.init.zeros_(self.category_weight.weight)
        nn.init.zeros_(self.numeric_weight.weight)

    def forward(self, category: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        return (
            self.category_weight(category).sum(dim=1).squeeze(-1)
            + self.numeric_weight(numeric).squeeze(-1) + self.bias
        )


class SparseFM(SparseLR):
    def __init__(self, total_tokens: int, num_numeric: int = 2, embedding_dim: int = 8) -> None:
        super().__init__(total_tokens, num_numeric)
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.category_factors = nn.Embedding(total_tokens, embedding_dim)
        self.numeric_factors = nn.Parameter(torch.empty(num_numeric, embedding_dim))
        nn.init.normal_(self.category_factors.weight, std=0.05)
        nn.init.normal_(self.numeric_factors, std=0.05)

    def forward(self, category: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        first_order = super().forward(category, numeric)
        vectors = torch.cat((
            self.category_factors(category),
            numeric.unsqueeze(-1) * self.numeric_factors.unsqueeze(0),
        ), dim=1)
        interactions = 0.5 * (
            vectors.sum(dim=1).square() - vectors.square().sum(dim=1)
        ).sum(dim=1)
        return first_order + interactions


class SparseMLP(nn.Module):
    def __init__(self, total_tokens: int, fields: int = 6,
                 num_numeric: int = 2, embedding_dim: int = 8) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.embeddings = nn.Embedding(total_tokens, embedding_dim)
        nn.init.normal_(self.embeddings.weight, std=0.05)
        self.network = nn.Sequential(
            nn.Linear(fields * embedding_dim + num_numeric, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1),
        )
        nn.init.constant_(self.network[-1].bias, -2.0)

    def forward(self, category: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        embedded = self.embeddings(category).flatten(start_dim=1)
        return self.network(torch.cat((embedded, numeric), dim=1)).squeeze(-1)
