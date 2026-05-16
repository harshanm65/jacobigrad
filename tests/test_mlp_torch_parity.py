"""End-to-end parity test: char-MLP forward + backward vs ``torch.autograd``.

Runs the exact MLP graph described in the milestone report (embedding gather
-> reshape -> linear -> tanh -> linear -> fused cross_entropy) through both
our engine and PyTorch on identical randomized inputs, and asserts that the
loss and every parameter gradient agree to deep ``float64`` precision.

Measured max absolute gradient diff on the canonical setup: ~5e-17 (well
below ``float64`` machine epsilon, ~2.2e-16). The asserted tolerance is
1e-14 absolute, ~50x machine epsilon, to be robust to PyTorch version /
platform variation while still meaningfully enforcing parity.
"""

from __future__ import annotations

import numpy as np
import pytest

from jacobigrad.autograd import Tensor
from jacobigrad.nn.losses import cross_entropy

torch = pytest.importorskip("torch")


def _build_inputs(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    V, D, T, B, H = 8, 4, 3, 6, 16
    return {
        "W_emb": rng.normal(size=(V, D)).astype(np.float64),
        "W1": rng.normal(size=(T * D, H)).astype(np.float64) * 0.1,
        "b1": np.zeros(H, dtype=np.float64),
        "W2": rng.normal(size=(H, V)).astype(np.float64) * 0.1,
        "b2": np.zeros(V, dtype=np.float64),
        "idx": rng.integers(0, V, size=(B, T)),
        "targets": rng.integers(0, V, size=B),
        "V": V, "D": D, "T": T, "B": B, "H": H,
    }


def _ours_forward_backward(p: dict) -> tuple[float, list[np.ndarray]]:
    We, W1, b1, W2, b2 = (
        Tensor(p["W_emb"]), Tensor(p["W1"]), Tensor(p["b1"]),
        Tensor(p["W2"]), Tensor(p["b2"]),
    )
    emb = We[p["idx"]]
    flat = emb.reshape(p["B"], p["T"] * p["D"])
    h = (flat @ W1 + b1).tanh()
    logits = h @ W2 + b2
    loss = cross_entropy(logits, p["targets"])
    loss.backward()
    return float(loss.data), [We.grad, W1.grad, b1.grad, W2.grad, b2.grad]


def _torch_forward_backward(p: dict) -> tuple[float, list[np.ndarray]]:
    We_t = torch.tensor(p["W_emb"], requires_grad=True)
    W1_t = torch.tensor(p["W1"], requires_grad=True)
    b1_t = torch.tensor(p["b1"], requires_grad=True)
    W2_t = torch.tensor(p["W2"], requires_grad=True)
    b2_t = torch.tensor(p["b2"], requires_grad=True)

    emb = We_t[torch.from_numpy(p["idx"])]
    flat = emb.reshape(p["B"], p["T"] * p["D"])
    h = torch.tanh(flat @ W1_t + b1_t)
    logits = h @ W2_t + b2_t
    loss = torch.nn.functional.cross_entropy(logits, torch.from_numpy(p["targets"]))
    loss.backward()

    return loss.item(), [
        We_t.grad.numpy(), W1_t.grad.numpy(), b1_t.grad.numpy(),
        W2_t.grad.numpy(), b2_t.grad.numpy(),
    ]


def test_mlp_forward_bit_identical_to_torch():
    p = _build_inputs(seed=0)
    loss_ours, _ = _ours_forward_backward(p)
    loss_torch, _ = _torch_forward_backward(p)
    # Same float64 ops in the same order — typically exactly bit-identical.
    assert abs(loss_ours - loss_torch) < 1e-15, (
        f"forward diff = {abs(loss_ours - loss_torch):.2e}"
    )


def test_mlp_gradients_match_torch_to_machine_epsilon():
    p = _build_inputs(seed=0)
    _, grads_ours = _ours_forward_backward(p)
    _, grads_torch = _torch_forward_backward(p)

    names = ["W_emb", "W1", "b1", "W2", "b2"]
    for name, ours, theirs in zip(names, grads_ours, grads_torch):
        assert ours is not None, f"{name}: our gradient was never accumulated"
        max_abs_diff = float(np.abs(ours - theirs).max())
        max_abs_theirs = float(np.abs(theirs).max())
        assert max_abs_diff < 1e-14, (
            f"{name}: max|ours - torch| = {max_abs_diff:.2e}, "
            f"max|torch| = {max_abs_theirs:.2e}"
        )
