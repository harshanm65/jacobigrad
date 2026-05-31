"""Tests for single-head scaled dot-product attention in the jacobigrad engine.

Three layers of validation, mirroring the engine-validation ladder used for
the rest of the project:

  1. **Forward parity** — our forward matches Luis's PyTorch reference
     (``reference_attention.torch_attention``) to float64 precision.
  2. **Numerical gradcheck** — our backward matches central differences
     (the non-negotiable per PLAN.md).
  3. **Closed-form / torch oracle** — our gradients for X, W_Q, W_K, W_V
     match Luis's hand-derived ``closed_form_backward`` (which is itself
     checked against ``torch.autograd``). This is the payoff of his
     ``attention_backward.md`` contribution: it is the independent spec our
     implementation is verified against.

The reference module lives in ``tests/`` and is imported as a top-level
module (same as ``test_reference_attention.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

from jacobigrad.autograd import Tensor, numgrad_check
from jacobigrad.nn.attention import SingleHeadAttention, causal_mask

from reference_attention import causal_mask as torch_causal_mask
from reference_attention import closed_form_backward, torch_attention

torch = pytest.importorskip("torch")


# ----------------------------------------------------------------------
# Helpers.
# ----------------------------------------------------------------------


def _make_layer(B, T, d, d_k, *, causal=False, seed=0):
    """Build an attention layer plus a random input, sharing one rng."""
    rng = np.random.default_rng(seed)
    layer = SingleHeadAttention(d, d_k, rng, causal=causal)
    X = rng.normal(size=(B, T, d))
    return layer, X


# ----------------------------------------------------------------------
# 1. Forward parity with the torch reference.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("causal", [False, True])
def test_forward_matches_torch_reference(causal):
    layer, X = _make_layer(2, 4, 5, 3, causal=causal, seed=0)

    Y_ours = layer.forward(X).data

    mask = torch_causal_mask(4).double() if causal else None
    Y_torch = torch_attention(
        torch.tensor(X),
        torch.tensor(layer.W_Q.data),
        torch.tensor(layer.W_K.data),
        torch.tensor(layer.W_V.data),
        mask=mask,
    ).numpy()

    np.testing.assert_allclose(Y_ours, Y_torch, atol=1e-12)


def test_causal_mask_matches_reference():
    # Our numpy mask and Luis's torch mask should agree (0 below/on diag, -inf above).
    ours = causal_mask(5)
    theirs = torch_causal_mask(5).numpy()
    # -inf == -inf is False under subtraction; compare finiteness + values separately.
    np.testing.assert_array_equal(np.isneginf(ours), np.isneginf(theirs))
    finite = ~np.isneginf(ours)
    np.testing.assert_array_equal(ours[finite], theirs[finite])


# ----------------------------------------------------------------------
# 2. Numerical gradient check (finite differences).
# ----------------------------------------------------------------------


@pytest.mark.parametrize("causal", [False, True])
def test_gradcheck_attention(causal):
    layer, X = _make_layer(2, 4, 5, 3, causal=causal, seed=1)
    Xt = Tensor(X)

    # Scalar reduction so numgrad_check has a scalar loss; check all params + X.
    def forward(ps):
        # ps = [X, W_Q, W_K, W_V]; rebuild with the layer's weights swapped in.
        layer.W_Q, layer.W_K, layer.W_V = ps[1], ps[2], ps[3]
        return layer.forward(ps[0]).sum()

    numgrad_check(forward, [Xt, layer.W_Q, layer.W_K, layer.W_V])


# ----------------------------------------------------------------------
# 3. Closed-form / torch oracle parity (Luis's derivation as the spec).
# ----------------------------------------------------------------------


@pytest.mark.parametrize("causal", [False, True])
def test_backward_matches_closed_form_oracle(causal):
    layer, X = _make_layer(2, 4, 5, 3, causal=causal, seed=2)
    Xt = Tensor(X)

    # Our forward + backward with a random upstream grad.
    Y = layer.forward(Xt)
    rng = np.random.default_rng(99)
    grad_Y = rng.normal(size=Y.shape)
    Y.backward(grad=grad_Y)

    # Luis's closed-form backward as the oracle.
    mask = torch_causal_mask(X.shape[1]).double() if causal else None
    dX, dW_Q, dW_K, dW_V = closed_form_backward(
        torch.tensor(grad_Y),
        torch.tensor(X),
        torch.tensor(layer.W_Q.data),
        torch.tensor(layer.W_K.data),
        torch.tensor(layer.W_V.data),
        mask=mask,
    )

    np.testing.assert_allclose(Xt.grad, dX.numpy(), atol=1e-10)
    np.testing.assert_allclose(layer.W_Q.grad, dW_Q.numpy(), atol=1e-10)
    np.testing.assert_allclose(layer.W_K.grad, dW_K.numpy(), atol=1e-10)
    np.testing.assert_allclose(layer.W_V.grad, dW_V.numpy(), atol=1e-10)


def test_causal_mask_blocks_future_positions():
    # With causal masking, position 0's output must depend only on V[0],
    # i.e. attention weights form a lower-triangular matrix.
    layer, X = _make_layer(1, 5, 4, 3, causal=True, seed=3)
    Xt = Tensor(X)

    # Reconstruct the attention matrix to assert it's lower-triangular.
    Q = Xt @ layer.W_Q
    K = Xt @ layer.W_K
    scores = (Q @ K.transpose(-2, -1)) * (1.0 / np.sqrt(layer.d_k))
    scores = scores + causal_mask(5)
    A = scores.softmax(axis=-1).data[0]

    upper = np.triu(np.ones((5, 5), dtype=bool), k=1)
    np.testing.assert_allclose(A[upper], 0.0, atol=1e-15)
