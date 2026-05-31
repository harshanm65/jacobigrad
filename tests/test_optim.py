"""Tests for the from-scratch AdamW optimizer.

Validated two ways:
  1. **Hand-stepped reference** — one Adam update computed explicitly from the
     formula, matching the optimizer's parameter trajectory.
  2. **torch.optim.AdamW parity** — full multi-step parameter trajectory on a
     toy quadratic agrees with PyTorch to deep precision, including bias
     correction and decoupled weight decay.
"""

from __future__ import annotations

import numpy as np
import pytest

from jacobigrad.autograd import Tensor
from jacobigrad.optim import AdamW


def test_zero_grad_clears_grads():
    p = Tensor(np.ones(3))
    p._add_grad(np.ones(3))
    opt = AdamW([p])
    opt.zero_grad()
    assert p.grad is None


def test_single_step_matches_hand_computed():
    # One AdamW step from the closed formula, weight_decay=0.
    lr, b1, b2, eps = 0.1, 0.9, 0.999, 1e-8
    p = Tensor(np.array([1.0, -2.0, 3.0]))
    g = np.array([0.5, 0.25, -1.0])
    p._add_grad(g.copy())

    opt = AdamW([p], lr=lr, betas=(b1, b2), eps=eps, weight_decay=0.0)
    opt.step()

    # Hand reference (t=1): m_hat = g, v_hat = g^2  (since 1-b^1 cancels the (1-b) factor)
    m = (1 - b1) * g
    v = (1 - b2) * (g * g)
    m_hat = m / (1 - b1 ** 1)
    v_hat = v / (1 - b2 ** 1)
    expected = np.array([1.0, -2.0, 3.0]) - lr * (m_hat / (np.sqrt(v_hat) + eps))

    np.testing.assert_allclose(p.data, expected, atol=1e-12)


def test_weight_decay_is_decoupled():
    # With grad = 0, AdamW should still shrink the parameter by lr*wd*p
    # (decoupled decay), whereas plain Adam would leave it unchanged.
    lr, wd = 0.1, 0.5
    p = Tensor(np.array([2.0, -4.0]))
    p._add_grad(np.zeros(2))
    opt = AdamW([p], lr=lr, weight_decay=wd)
    opt.step()
    # m_hat = v_hat = 0 -> adaptive term is 0; only decay remains.
    expected = np.array([2.0, -4.0]) * (1 - lr * wd)
    np.testing.assert_allclose(p.data, expected, atol=1e-12)


@pytest.mark.parametrize("weight_decay", [0.0, 0.01])
def test_trajectory_matches_torch_adamw(weight_decay):
    torch = pytest.importorskip("torch")

    rng = np.random.default_rng(0)
    x0 = rng.normal(size=(5, 4))
    # Toy objective: 0.5 * sum((W - target)^2) -> grad = (W - target).
    target = rng.normal(size=(5, 4))

    # Ours.
    W = Tensor(x0.copy())
    opt = AdamW([W], lr=0.05, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    for _ in range(25):
        opt.zero_grad()
        # grad of 0.5*||W-target||^2 is (W - target); set it directly.
        W._add_grad(W.data - target)
        opt.step()

    # Torch.
    Wt = torch.tensor(x0.copy(), requires_grad=True)
    tgt = torch.tensor(target)
    opt_t = torch.optim.AdamW([Wt], lr=0.05, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    for _ in range(25):
        opt_t.zero_grad()
        loss = 0.5 * ((Wt - tgt) ** 2).sum()
        loss.backward()
        opt_t.step()

    np.testing.assert_allclose(W.data, Wt.detach().numpy(), atol=1e-10)


def test_converges_on_quadratic():
    # Sanity: AdamW should drive params toward the minimum of a convex bowl.
    target = np.array([3.0, -1.0, 2.5])
    W = Tensor(np.zeros(3))
    opt = AdamW([W], lr=0.1)
    for _ in range(500):
        opt.zero_grad()
        W._add_grad(W.data - target)
        opt.step()
    np.testing.assert_allclose(W.data, target, atol=1e-3)
