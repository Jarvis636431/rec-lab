import pytest
import torch

from models.fm.model import FactorizationMachine


@pytest.mark.parametrize("batch_size,num_features,embedding_dim", [(1, 2, 1), (8, 5, 4)])
def test_fast_interaction_equals_explicit_pairs(batch_size, num_features, embedding_dim):
    torch.manual_seed(7)
    model = FactorizationMachine(num_features, embedding_dim).double()
    features = torch.randn(batch_size, num_features, dtype=torch.float64)

    actual = model(features)
    explicit = model.linear(features).squeeze(-1)
    for left in range(num_features):
        for right in range(left + 1, num_features):
            explicit = explicit + (
                model.factors[left] * model.factors[right]
            ).sum() * features[:, left] * features[:, right]

    assert actual.shape == (batch_size,)
    torch.testing.assert_close(actual, explicit, rtol=1e-12, atol=1e-12)


def test_pair_factors_receive_gradient():
    model = FactorizationMachine(5, 4)
    logits = model(torch.tensor([[1., 1., 0., 0., 0.]]))
    logits.sum().backward()

    assert model.factors.grad is not None
    assert model.factors.grad[0].abs().sum() > 0
    assert model.factors.grad[1].abs().sum() > 0
    assert torch.count_nonzero(model.factors.grad[2:]) == 0
