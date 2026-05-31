"""Tests for LayerNorm in the jacobigrad engine.

LayerNorm is implemented by composing autograd primitives (mean, subtract,
power, multiply), so its backward is produced by the engine rather than
hand-written. We validate it three ways, the same ladder used elsewhere:

  1. **Forward parity** with ``torch.nn.LayerNorm`` (incl. learned affine).
  2. **Numerical gradcheck** of the composed forward.
  3. **Closed-form parity** — the input gradient matches the hand-derived
     formula in ``writeups/cs229/layernorm_backward.md`` to machine epsilon,
     and the full backward matches ``torch.autograd`` for x, gamma, beta.
"""

from __future__ import annotations

import numpy as np
import pytest

from jacobigrad.autograd import Tensor, numgrad_check
from jacobigrad.nn.layernorm import LayerNorm

torch = pytest.importorskip("torch")


# ----------------------------------------------------------------------
# 1. Forward parity with torch.nn.LayerNorm.
# ----------------------------------------------------------------------


def test_forward_normalizes_to_zero_mean_unit_var():
    rng = np.random.default_rng(0)
    ln = LayerNorm(8)
    x = rng.normal(size=(4, 5, 8)) * 3.0 + 7.0  # off-center, scaled
    y = ln.forward(x).data
    # With gamma=1, beta=0, output rows are standardized.
    np.testing.assert_allclose(y.mean(axis=-1), 0.0, atol=1e-12)
    np.testing.assert_allclose(y.std(axis=-1), 1.0, atol=1e-4)


def test_forward_matches_torch_layernorm():
    rng = np.random.default_rng(1)
    H = 8
    ln = LayerNorm(H, eps=1e-5)
    # Give gamma/beta non-trivial values so the affine path is exercised.
    ln.gamma.data = rng.normal(size=H)
    ln.beta.data = rng.normal(size=H)
    x = rng.normal(size=(4, 5, H))

    y_ours = ln.forward(x).data

    ln_t = torch.nn.LayerNorm(H, eps=1e-5).double()
    with torch.no_grad():
        ln_t.weight.copy_(torch.tensor(ln.gamma.data))
        ln_t.bias.copy_(torch.tensor(ln.beta.data))
    y_torch = ln_t(torch.tensor(x)).detach().numpy()

    np.testing.assert_allclose(y_ours, y_torch, atol=1e-12)


# ----------------------------------------------------------------------
# 2. Numerical gradient check.
# ----------------------------------------------------------------------


def test_gradcheck_layernorm():
    rng = np.random.default_rng(2)
    H = 6
    ln = LayerNorm(H)
    ln.gamma.data = rng.normal(size=H)
    ln.beta.data = rng.normal(size=H)
    x = Tensor(rng.normal(size=(3, 4, H)))

    def forward(ps):
        ln.gamma, ln.beta = ps[1], ps[2]
        return ln.forward(ps[0]).sum()

    numgrad_check(forward, [x, ln.gamma, ln.beta])


# ----------------------------------------------------------------------
# 3a. Full backward parity with torch.autograd.
# ----------------------------------------------------------------------


def test_backward_matches_torch_autograd():
    rng = np.random.default_rng(3)
    H = 8
    ln = LayerNorm(H, eps=1e-5)
    ln.gamma.data = rng.normal(size=H)
    ln.beta.data = rng.normal(size=H)
    x = rng.normal(size=(2, 5, H))
    grad_y = rng.normal(size=(2, 5, H))

    # Ours.
    xt = Tensor(x)
    y = ln.forward(xt)
    y.backward(grad=grad_y)

    # Torch.
    x_t = torch.tensor(x, requires_grad=True)
    ln_t = torch.nn.LayerNorm(H, eps=1e-5).double()
    with torch.no_grad():
        ln_t.weight.copy_(torch.tensor(ln.gamma.data))
        ln_t.bias.copy_(torch.tensor(ln.beta.data))
    y_t = ln_t(x_t)
    y_t.backward(torch.tensor(grad_y))

    np.testing.assert_allclose(xt.grad, x_t.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(ln.gamma.grad, ln_t.weight.grad.numpy(), atol=1e-10)
    np.testing.assert_allclose(ln.beta.grad, ln_t.bias.grad.numpy(), atol=1e-10)


# ----------------------------------------------------------------------
# 3b. Input gradient parity with the hand-derived closed form.
# ----------------------------------------------------------------------


def _closed_form_dx(x, grad_y, gamma, eps):
    """dL/dx per writeups/cs229/layernorm_backward.md (numpy, no autograd)."""
    mu = x.mean(axis=-1, keepdims=True)
    centered = x - mu
    var = (centered ** 2).mean(axis=-1, keepdims=True)
    rstd = (var + eps) ** -0.5
    xhat = centered * rstd

    dxhat = grad_y * gamma
    mean1 = dxhat.mean(axis=-1, keepdims=True)
    mean2 = (dxhat * xhat).mean(axis=-1, keepdims=True)
    return rstd * (dxhat - mean1 - xhat * mean2)


def test_layernorm_dx_matches_closed_form():
    rng = np.random.default_rng(4)
    H = 7
    eps = 1e-5
    ln = LayerNorm(H, eps=eps)
    ln.gamma.data = rng.normal(size=H)
    ln.beta.data = rng.normal(size=H)
    x = rng.normal(size=(3, 4, H))
    grad_y = rng.normal(size=(3, 4, H))

    xt = Tensor(x)
    y = ln.forward(xt)
    y.backward(grad=grad_y)

    dx_closed = _closed_form_dx(x, grad_y, ln.gamma.data, eps)
    np.testing.assert_allclose(xt.grad, dx_closed, atol=1e-12)
