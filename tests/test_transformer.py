"""Tests for the stacked CharTransformer LM.

Covers the things the ablation depends on:
  - forward produces (B, T, V) logits;
  - the loss reshape path ((B,T,V)->(B*T,V)) is correct and differentiable
    (end-to-end finite-difference gradcheck on tiny dims);
  - forward_with_taps returns one tap per block, and every tap's .grad is
    populated after backward (the instrumentation contract);
  - all 6 (norm, residual) configs build and train one step without NaN.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from jacobigrad.autograd import numgrad_check
from jacobigrad.nn.block import NORM_PLACEMENTS
from jacobigrad.nn.transformer import CharTransformer
from jacobigrad.optim import AdamW

ALL_CONFIGS = list(itertools.product(NORM_PLACEMENTS, [True, False]))


def _tiny_model(norm="pre", residual=True, *, V=7, block=5, d=6, d_k=6, d_ff=12, L=2, seed=0):
    rng = np.random.default_rng(seed)
    return CharTransformer(V, block, d, d_k, d_ff, L, rng, norm=norm, residual=residual)


def test_forward_shape():
    m = _tiny_model()
    rng = np.random.default_rng(1)
    idx = rng.integers(0, m.V, size=(3, 4))
    logits = m.forward(idx)
    assert logits.shape == (3, 4, m.V)


def test_sequence_longer_than_block_raises():
    m = _tiny_model(block=5)
    idx = np.zeros((2, 6), dtype=np.int64)
    with pytest.raises(ValueError):
        m.forward(idx)


def test_loss_is_scalar_and_finite():
    m = _tiny_model()
    rng = np.random.default_rng(2)
    idx = rng.integers(0, m.V, size=(3, 4))
    targets = rng.integers(0, m.V, size=(3, 4))
    loss = m.loss(idx, targets)
    assert loss.data.ndim == 0
    assert np.isfinite(loss.data)


def test_batch_ce_matches_loss():
    # The numpy-only batch_ce should equal the autograd loss value.
    m = _tiny_model()
    rng = np.random.default_rng(3)
    idx = rng.integers(0, m.V, size=(4, 5))
    targets = rng.integers(0, m.V, size=(4, 5))
    np.testing.assert_allclose(m.batch_ce(idx, targets), float(m.loss(idx, targets).data), atol=1e-12)


def test_end_to_end_gradcheck():
    # Finite-difference gradcheck through the whole model on tiny dims.
    m = _tiny_model(d=4, d_k=4, d_ff=8, L=2, seed=4)
    rng = np.random.default_rng(5)
    idx = rng.integers(0, m.V, size=(2, 3))
    targets = rng.integers(0, m.V, size=(2, 3))
    params = m.parameters()

    def forward(ps):
        return m.loss(idx, targets)

    numgrad_check(forward, params)


def test_forward_with_taps_returns_one_per_block():
    m = _tiny_model(L=3)
    rng = np.random.default_rng(6)
    idx = rng.integers(0, m.V, size=(2, 4))
    logits, taps = m.forward_with_taps(idx)
    assert len(taps) == m.n_layers
    assert all(t.shape == (2, 4, m.d) for t in taps)


def test_taps_grad_populated_after_backward():
    # The instrumentation contract: every block-output tap has a gradient
    # after backward, so per-depth grad norms are readable.
    m = _tiny_model(L=3)
    rng = np.random.default_rng(7)
    idx = rng.integers(0, m.V, size=(2, 4))
    targets = rng.integers(0, m.V, size=(2, 4))
    B, T = idx.shape

    from jacobigrad.nn.losses import cross_entropy

    logits, taps = m.forward_with_taps(idx)
    loss = cross_entropy(logits.reshape(B * T, m.V), targets.reshape(B * T))
    loss.backward()

    for t in taps:
        assert t.grad is not None
        assert t.grad.shape == (2, 4, m.d)
        assert np.all(np.isfinite(t.grad))


@pytest.mark.parametrize("norm,residual", ALL_CONFIGS)
def test_all_configs_train_one_step_without_nan(norm, residual):
    m = _tiny_model(norm=norm, residual=residual, seed=8)
    rng = np.random.default_rng(9)
    idx = rng.integers(0, m.V, size=(4, 5))
    targets = rng.integers(0, m.V, size=(4, 5))

    opt = AdamW(m.parameters(), lr=1e-3)
    opt.zero_grad()
    loss = m.loss(idx, targets)
    loss.backward()
    opt.step()

    assert np.isfinite(loss.data)
    assert all(np.all(np.isfinite(p.data)) for p in m.parameters())
