"""Tests for the toggleable transformer DecoderBlock.

The block is the apparatus for the gradient-flow ablation, so the tests focus
on the two things that must be airtight:

  1. **Correctness** across all six configs (3 norm placements x 2 residual
     settings): shape preservation and finite-difference gradcheck.
  2. **The toggles do what they claim** — pre/post/none and residual on/off
     produce the structurally correct computation (verified against a manual
     recomputation, and via a torch-parity anchor for one config).

Single-head, causal — matching the ablation's scope.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from jacobigrad.autograd import Tensor, numgrad_check
from jacobigrad.nn.block import NORM_PLACEMENTS, DecoderBlock

ALL_CONFIGS = list(itertools.product(NORM_PLACEMENTS, [True, False]))


def _make_block(norm, residual, *, B=2, T=4, d=6, d_k=5, d_ff=12, seed=0):
    rng = np.random.default_rng(seed)
    block = DecoderBlock(d, d_k, d_ff, rng, norm=norm, residual=residual, causal=True)
    X = rng.normal(size=(B, T, d))
    return block, X


# ----------------------------------------------------------------------
# 1. Shape + gradcheck across all six configs.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("norm,residual", ALL_CONFIGS)
def test_forward_preserves_shape(norm, residual):
    block, X = _make_block(norm, residual)
    Y = block.forward(X)
    assert Y.shape == X.shape


@pytest.mark.parametrize("norm,residual", ALL_CONFIGS)
def test_gradcheck_all_configs(norm, residual):
    block, X = _make_block(norm, residual, seed=1)
    Xt = Tensor(X)
    params = block.parameters()

    def forward(ps):
        return block.forward(Xt).sum()

    # Check input grad and all block params together.
    numgrad_check(forward, [Xt, *params])


# ----------------------------------------------------------------------
# 2. The toggles produce the structurally correct computation.
# ----------------------------------------------------------------------


def test_norm_none_uses_no_layernorm_params():
    block, _ = _make_block("none", True)
    # With norm == "none", LN params are excluded from the parameter set.
    ln_params = {id(p) for p in (*block.ln1.parameters(), *block.ln2.parameters())}
    assert all(id(p) not in ln_params for p in block.parameters())


def test_norm_pre_post_include_layernorm_params():
    for norm in ("pre", "post"):
        block, _ = _make_block(norm, True)
        # 3 attn (Wq,Wk,Wv) + W_O + (W1,b1,W2,b2) + 2 LN * (gamma,beta) = 12
        assert block.num_parameters() == sum(
            p.size for p in block.parameters()
        )
        n_tensors = len(block.parameters())
        assert n_tensors == 8 + 4  # 8 core tensors + 4 LN tensors


def test_residual_off_changes_output():
    # Same seed/weights, residual on vs off must differ (the skip is real).
    on, X = _make_block("pre", True, seed=7)
    off, _ = _make_block("pre", False, seed=7)
    y_on = on.forward(X).data
    y_off = off.forward(X).data
    assert not np.allclose(y_on, y_off)


def test_pre_norm_matches_manual_recomputation():
    # Verify the pre-norm residual policy x + f(LN(x)) for the attention
    # sublayer specifically, by recomputing the first sublayer by hand.
    block, X = _make_block("pre", True, seed=3)
    Xt = Tensor(X)

    # Manual: x + (attn(LN(x)) @ W_O)
    normed = block.ln1.forward(Xt)
    sub = block.attn.forward(normed) @ block.W_O
    expected_after_sublayer1 = (Xt + sub).data

    # Reach into the block's first-sublayer output via _apply.
    got = block._apply(Xt, block._attn_sublayer, block.ln1).data
    np.testing.assert_allclose(got, expected_after_sublayer1, atol=1e-12)


def test_post_norm_normalizes_after_residual():
    # For post-norm with residual, the sublayer output should be normalized:
    # each feature row of LN(x + f(x)) has ~zero mean (gamma=1, beta=0 init).
    block, X = _make_block("post", True, seed=5)
    Xt = Tensor(X)
    out = block._apply(Xt, block._attn_sublayer, block.ln1).data
    np.testing.assert_allclose(out.mean(axis=-1), 0.0, atol=1e-12)


# ----------------------------------------------------------------------
# torch-parity anchor for the canonical pre-norm + residual config.
# ----------------------------------------------------------------------


def test_prenorm_block_matches_torch():
    torch = pytest.importorskip("torch")
    block, X = _make_block("pre", True, seed=2, B=2, T=4, d=6, d_k=6, d_ff=12)

    # Ours.
    Xt = Tensor(X)
    Y = block.forward(Xt)
    gY = np.random.default_rng(0).normal(size=Y.shape)
    Y.backward(grad=gY)

    # Torch reconstruction of the exact same graph.
    import math

    def t(a):
        return torch.tensor(a, dtype=torch.float64, requires_grad=True)

    Xtt = t(X)
    Wq, Wk, Wv = t(block.attn.W_Q.data), t(block.attn.W_K.data), t(block.attn.W_V.data)
    Wo = t(block.W_O.data)
    W1, b1 = t(block.W1.data), t(block.b1.data)
    W2, b2 = t(block.W2.data), t(block.b2.data)
    g1, be1 = t(block.ln1.gamma.data), t(block.ln1.beta.data)
    g2, be2 = t(block.ln2.gamma.data), t(block.ln2.beta.data)

    def torch_ln(x, g, b, eps=1e-5):
        mu = x.mean(-1, keepdim=True)
        var = ((x - mu) ** 2).mean(-1, keepdim=True)
        return (x - mu) / torch.sqrt(var + eps) * g + b

    def torch_attn(x):
        T = x.shape[-2]
        Q, K, V = x @ Wq, x @ Wk, x @ Wv
        S = Q @ K.transpose(-2, -1) / math.sqrt(block.attn.d_k)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(mask, float("-inf"))
        A = torch.softmax(S, dim=-1)
        return (A @ V) @ Wo

    # Pre-norm policy for both sublayers: h = x + attn(LN1(x)); out = h + ffn(LN2(h)).
    h = Xtt + torch_attn(torch_ln(Xtt, g1, be1))
    ffn = (torch_ln(h, g2, be2) @ W1 + b1).relu() @ W2 + b2
    out = h + ffn
    out.backward(torch.tensor(gY))

    np.testing.assert_allclose(Y.data, out.detach().numpy(), atol=1e-12)
    np.testing.assert_allclose(Xt.grad, Xtt.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(block.W_O.grad, Wo.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(block.attn.W_Q.grad, Wq.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(block.ln1.gamma.grad, g1.grad.numpy(), atol=1e-10)
