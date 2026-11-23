import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.features import shuffle_candidates  # noqa: E402
from src.models.fusion import CandidateConditionedFusion, build_candidate_sequences  # noqa: E402


@pytest.mark.parametrize("m", [0, 1, 2, 3, 4])
def test_candidate_placed_at_missing_position_and_context_order_kept(m):
    B, K, D = 2, 6, 7
    ctx = torch.arange(B * 4 * D, dtype=torch.float).view(B, 4, D)
    cand = -1 - torch.arange(B * K * D, dtype=torch.float).view(B, K, D)
    seq = build_candidate_sequences(ctx, cand, torch.full((B,), m))
    assert seq.shape == (B, K, 5, D)
    for b in range(B):
        for j in range(K):
            assert torch.equal(seq[b, j, m], cand[b, j])
            others = [p for p in range(5) if p != m]
            assert torch.equal(seq[b, j, others], ctx[b])  # same relative order


def test_mixed_missing_positions_in_one_batch():
    ctx = torch.randn(5, 4, 3); cand = torch.randn(5, 6, 3)
    mp = torch.tensor([0, 1, 2, 3, 4])
    seq = build_candidate_sequences(ctx, cand, mp)
    for b in range(5):
        assert torch.equal(seq[b, 2, mp[b]], cand[b, 2])


def test_forward_shape_and_grad():
    model = CandidateConditionedFusion(hidden=32, heads=4, ff=64, layers=1)
    v, t = torch.randn(3, 10, 768), torch.randn(3, 10, 384)
    h = torch.randint(0, 2, (3, 10)).float()
    logits = model(v, t, h, torch.tensor([0, 2, 4]))
    assert logits.shape == (3, 6)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2])).backward()


def test_candidate_permutation_equivariance():
    """Permuting candidates must permute the logits identically (eval mode)."""
    torch.manual_seed(0)
    model = CandidateConditionedFusion(hidden=32, heads=4, ff=64, layers=2).eval()
    v, t = torch.randn(2, 10, 768), torch.randn(2, 10, 384)
    h = torch.ones(2, 10); mp = torch.tensor([1, 3])
    perm = torch.randperm(6)
    v2, t2, h2 = v.clone(), t.clone(), h.clone()
    v2[:, 4:], t2[:, 4:], h2[:, 4:] = v[:, 4:][:, perm], t[:, 4:][:, perm], h[:, 4:][:, perm]
    with torch.no_grad():
        a, b = model(v, t, h, mp), model(v2, t2, h2, mp)
    assert torch.allclose(a[:, perm], b, atol=1e-5)


def test_shuffle_candidates_remaps_label():
    B = 8
    vis = torch.zeros(B, 10, 2)
    vis[:, 4:, 0] = torch.arange(6).float()          # candidate identity stored in the feature
    txt, has = torch.zeros(B, 10, 3), torch.zeros(B, 10)
    label = torch.randint(0, 6, (B,))
    v2, _, _, l2 = shuffle_candidates(vis, txt, has, label)
    assert torch.equal(v2[:, :4], vis[:, :4])         # context untouched
    for b in range(B):
        assert int(v2[b, 4 + l2[b], 0]) == int(label[b])  # new label points at the same candidate
