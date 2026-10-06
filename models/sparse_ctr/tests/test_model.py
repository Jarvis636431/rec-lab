import torch

from models.sparse_ctr.model import SparseFM, SparseLR, SparseMLP


def test_three_models_return_one_logit_per_example():
    category = torch.tensor([[1, 4, 8, 12, 17, 20]])
    numeric = torch.tensor([[0.3, -0.1]])
    for model in (SparseLR(25), SparseFM(25), SparseMLP(25)):
        output = model(category, numeric)
        assert output.shape == (1,)
        assert torch.isfinite(output).all()


def test_sparse_fm_fast_pairs_match_explicit_sum():
    torch.manual_seed(3)
    model = SparseFM(30, embedding_dim=4).double()
    category = torch.tensor([[1, 5, 10, 15, 20, 25], [2, 6, 11, 16, 21, 26]])
    numeric = torch.tensor([[0.5, -0.2], [-0.7, 1.1]], dtype=torch.float64)
    vectors = torch.cat((
        model.category_factors(category),
        numeric.unsqueeze(-1) * model.numeric_factors.unsqueeze(0),
    ), dim=1)
    explicit = SparseLR.forward(model, category, numeric)
    for left in range(vectors.shape[1]):
        for right in range(left + 1, vectors.shape[1]):
            explicit = explicit + (vectors[:, left] * vectors[:, right]).sum(dim=1)
    torch.testing.assert_close(model(category, numeric), explicit, rtol=1e-12, atol=1e-12)
