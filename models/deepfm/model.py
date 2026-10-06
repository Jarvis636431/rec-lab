"""First-order, FM, and deep branches with shared feature embeddings."""

from typing import Dict

import torch
from torch import nn


VARIANTS = ("fm_only", "deep_only", "deepfm")


class DeepFM(nn.Module):
    """Return logits; the FM and deep branches consume the same vectors."""

    def __init__(
        self,
        total_tokens: int,
        categorical_fields: int = 6,
        numeric_fields: int = 2,
        embedding_dim: int = 8,
        variant: str = "deepfm",
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if total_tokens <= 0 or categorical_fields <= 0 or numeric_fields <= 0:
            raise ValueError("feature dimensions must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        self.variant = variant
        self.categorical_fields = categorical_fields
        self.numeric_fields = numeric_fields
        self.category_weight = nn.Embedding(total_tokens, 1)
        self.numeric_weight = nn.Linear(numeric_fields, 1, bias=False)
        self.bias = nn.Parameter(torch.tensor(-2.0))
        self.feature_embeddings = nn.Embedding(total_tokens, embedding_dim)
        self.numeric_embeddings = nn.Parameter(torch.empty(numeric_fields, embedding_dim))
        nn.init.zeros_(self.category_weight.weight)
        nn.init.zeros_(self.numeric_weight.weight)
        nn.init.normal_(self.feature_embeddings.weight, std=0.05)
        nn.init.normal_(self.numeric_embeddings, std=0.05)

        self.deep = None
        if variant in ("deep_only", "deepfm"):
            self.deep = nn.Sequential(
                nn.Linear((categorical_fields + numeric_fields) * embedding_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
            nn.init.zeros_(self.deep[-1].bias)

    def components(self, category: torch.Tensor, numeric: torch.Tensor) -> Dict[str, torch.Tensor]:
        if category.ndim != 2 or category.shape[1] != self.categorical_fields:
            raise ValueError("category has wrong shape")
        if numeric.ndim != 2 or numeric.shape != (len(category), self.numeric_fields):
            raise ValueError("numeric has wrong shape")
        first_order = (
            self.category_weight(category).sum(dim=1).squeeze(-1)
            + self.numeric_weight(numeric).squeeze(-1) + self.bias
        )
        # These same vectors feed both the FM and deep branches.
        vectors = torch.cat((
            self.feature_embeddings(category),
            numeric.unsqueeze(-1) * self.numeric_embeddings.unsqueeze(0),
        ), dim=1)
        parts = {"first_order": first_order}
        if self.variant in ("fm_only", "deepfm"):
            parts["fm"] = 0.5 * (
                vectors.sum(dim=1).square() - vectors.square().sum(dim=1)
            ).sum(dim=1)
        if self.deep is not None:
            parts["deep"] = self.deep(vectors.flatten(start_dim=1)).squeeze(-1)
        return parts

    def forward(self, category: torch.Tensor, numeric: torch.Tensor) -> torch.Tensor:
        parts = self.components(category, numeric)
        return sum(parts.values())
