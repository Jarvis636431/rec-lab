import pytest
import torch

from models.deepfm.model import DeepFM


@pytest.mark.parametrize("variant,keys", [
    ("fm_only", {"first_order", "fm"}),
    ("deep_only", {"first_order", "deep"}),
    ("deepfm", {"first_order", "fm", "deep"}),
])
def test_branch_outputs_sum_to_logits(variant, keys):
    model = DeepFM(total_tokens=30, variant=variant)
    category = torch.tensor([[1, 5, 10, 15, 20, 25]])
    numeric = torch.tensor([[0.5, -0.2]])
    parts = model.components(category, numeric)

    assert set(parts) == keys
    assert all(value.shape == (1,) for value in parts.values())
    torch.testing.assert_close(model(category, numeric), sum(parts.values()))


def test_fm_branch_matches_explicit_pairs():
    torch.manual_seed(3)
    model = DeepFM(total_tokens=30, embedding_dim=4, variant="fm_only").double()
    category = torch.tensor([[1, 5, 10, 15, 20, 25], [2, 6, 11, 16, 21, 26]])
    numeric = torch.tensor([[0.5, -0.2], [-0.7, 1.1]], dtype=torch.float64)
    vectors = torch.cat((
        model.feature_embeddings(category),
        numeric.unsqueeze(-1) * model.numeric_embeddings.unsqueeze(0),
    ), dim=1)
    explicit = torch.zeros(2, dtype=torch.float64)
    for left in range(vectors.shape[1]):
        for right in range(left + 1, vectors.shape[1]):
            explicit = explicit + (vectors[:, left] * vectors[:, right]).sum(dim=1)
    torch.testing.assert_close(model.components(category, numeric)["fm"], explicit,
                               rtol=1e-12, atol=1e-12)


def test_both_branches_backpropagate_to_one_shared_embedding_table():
    model = DeepFM(total_tokens=30, variant="deepfm")
    category = torch.tensor([[1, 5, 10, 15, 20, 25]])
    numeric = torch.tensor([[0.5, -0.2]])
    assert [name for name, _ in model.named_parameters() if "feature_embeddings" in name] == [
        "feature_embeddings.weight"
    ]

    model.components(category, numeric)["fm"].sum().backward()
    fm_gradient = model.feature_embeddings.weight.grad.clone()
    model.zero_grad()
    model.components(category, numeric)["deep"].sum().backward()
    deep_gradient = model.feature_embeddings.weight.grad
    assert fm_gradient[1].abs().sum() > 0
    assert deep_gradient[1].abs().sum() > 0
