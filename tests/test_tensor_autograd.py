"""Tests for jacobigrad.autograd.Tensor.

The strategy is the same as for the scalar engine: build a small graph with
randomized inputs, then call ``numgrad_check`` to verify backward against
central differences. Every op gets at least one shape with broadcasting and
one without, so the unbroadcast logic is exercised.
"""

from __future__ import annotations

import numpy as np
import pytest

from jacobigrad.autograd import Tensor, numgrad_check
from jacobigrad.autograd.tensor import _unbroadcast


# ----------------------------------------------------------------------
# Helpers.
# ----------------------------------------------------------------------


def _check_op_with_explicit_grad(op_fn, params, eps=1e-6, atol=1e-7, rtol=1e-5):
    """Manual gradcheck variant for ops that return a non-scalar.

    Picks a random upstream gradient, computes JVP analytically via backward,
    and compares each parameter's grad against central differences of the
    contraction ``loss = (op_fn(params).data * grad).sum()``.
    """
    rng = np.random.default_rng(123)

    # Forward to get out shape, then pick a random upstream grad.
    for p in params:
        p.zero_grad()
    out = op_fn(params)
    upstream = rng.normal(size=out.shape)

    # Analytic.
    out.backward(grad=upstream)
    analytic = [
        p.grad.copy() if p.grad is not None else np.zeros_like(p.data) for p in params
    ]

    # Numerical: scalar loss = sum(out.data * upstream)
    def loss_value() -> float:
        return float(np.sum(op_fn(params).data * upstream))

    for p_idx, p in enumerate(params):
        numerical = np.zeros_like(p.data)
        for idx in np.ndindex(p.data.shape):
            orig = float(p.data[idx])
            p.data[idx] = orig + eps
            l_plus = loss_value()
            p.data[idx] = orig - eps
            l_minus = loss_value()
            p.data[idx] = orig
            numerical[idx] = (l_plus - l_minus) / (2 * eps)

        diff = np.abs(analytic[p_idx] - numerical)
        tol = atol + rtol * np.abs(numerical)
        assert np.all(diff <= tol), (
            f"param {p_idx} failed: max_diff={diff.max():.3e}, "
            f"max_tol={tol.max():.3e}"
        )


# ----------------------------------------------------------------------
# _unbroadcast helper.
# ----------------------------------------------------------------------


def test_unbroadcast_drops_leading_axes():
    g = np.ones((4, 3, 5))
    out = _unbroadcast(g, (3, 5))
    assert out.shape == (3, 5)
    np.testing.assert_array_equal(out, np.full((3, 5), 4.0))


def test_unbroadcast_sums_size_one_axes():
    g = np.ones((2, 4, 3))
    out = _unbroadcast(g, (1, 4, 3))
    assert out.shape == (1, 4, 3)
    np.testing.assert_array_equal(out, np.full((1, 4, 3), 2.0))


def test_unbroadcast_handles_both():
    # target (1, 3): grad is (5, 4, 3), broadcast collapsed both to leading
    # axis (5, 4) AND size-1 axis (1).
    g = np.ones((5, 4, 3))
    out = _unbroadcast(g, (1, 3))
    assert out.shape == (1, 3)
    np.testing.assert_array_equal(out, np.full((1, 3), 20.0))


def test_unbroadcast_identity_when_shapes_match():
    g = np.arange(6.0).reshape(2, 3)
    out = _unbroadcast(g, (2, 3))
    np.testing.assert_array_equal(out, g)


# ----------------------------------------------------------------------
# Forward sanity.
# ----------------------------------------------------------------------


def test_tensor_construction_from_list():
    t = Tensor([[1, 2], [3, 4]])
    assert t.shape == (2, 2)
    assert t.data.dtype == np.float64
    assert t.grad is None


def test_tensor_construction_from_scalar():
    t = Tensor(3.0)
    assert t.shape == ()
    assert t.data.dtype == np.float64


def test_forward_arithmetic_no_broadcast():
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    np.testing.assert_array_equal((a + b).data, [5, 7, 9])
    np.testing.assert_array_equal((a - b).data, [-3, -3, -3])
    np.testing.assert_array_equal((a * b).data, [4, 10, 18])
    np.testing.assert_allclose((a / b).data, [0.25, 0.4, 0.5])
    np.testing.assert_array_equal((a ** 2).data, [1, 4, 9])
    np.testing.assert_array_equal((-a).data, [-1, -2, -3])


def test_forward_arithmetic_with_broadcast():
    a = Tensor([[1, 2], [3, 4]])  # (2, 2)
    b = Tensor([10, 20])  # (2,) broadcasts to (2, 2)
    np.testing.assert_array_equal((a + b).data, [[11, 22], [13, 24]])
    np.testing.assert_array_equal((a * b).data, [[10, 40], [30, 80]])


# ----------------------------------------------------------------------
# Backward correctness via gradcheck. Because we don't have ``sum`` yet,
# tests use the explicit-grad helper that contracts with a fixed upstream.
# ----------------------------------------------------------------------


def _rand_tensor(rng, *shape):
    return Tensor(rng.normal(size=shape))


def test_grad_add_no_broadcast():
    rng = np.random.default_rng(0)
    a, b = _rand_tensor(rng, 3, 4), _rand_tensor(rng, 3, 4)
    _check_op_with_explicit_grad(lambda ps: ps[0] + ps[1], [a, b])


def test_grad_add_broadcast_bias():
    rng = np.random.default_rng(0)
    a, b = _rand_tensor(rng, 5, 7), _rand_tensor(rng, 7)
    _check_op_with_explicit_grad(lambda ps: ps[0] + ps[1], [a, b])


def test_grad_add_broadcast_size1_dim():
    rng = np.random.default_rng(0)
    a, b = _rand_tensor(rng, 5, 7), Tensor(rng.normal(size=(1, 7)))
    _check_op_with_explicit_grad(lambda ps: ps[0] + ps[1], [a, b])


def test_grad_mul_no_broadcast():
    rng = np.random.default_rng(0)
    a, b = _rand_tensor(rng, 4, 3), _rand_tensor(rng, 4, 3)
    _check_op_with_explicit_grad(lambda ps: ps[0] * ps[1], [a, b])


def test_grad_mul_broadcast():
    rng = np.random.default_rng(0)
    a, b = _rand_tensor(rng, 4, 3), _rand_tensor(rng, 3)
    _check_op_with_explicit_grad(lambda ps: ps[0] * ps[1], [a, b])


def test_grad_sub():
    rng = np.random.default_rng(0)
    a, b = _rand_tensor(rng, 3, 5), _rand_tensor(rng, 3, 5)
    _check_op_with_explicit_grad(lambda ps: ps[0] - ps[1], [a, b])


def test_grad_div_no_broadcast():
    rng = np.random.default_rng(0)
    a = _rand_tensor(rng, 3, 5)
    # b away from zero to keep the gradient stable.
    b = Tensor(rng.normal(size=(3, 5)) + 2.0)
    _check_op_with_explicit_grad(lambda ps: ps[0] / ps[1], [a, b])


def test_grad_pow():
    rng = np.random.default_rng(0)
    a = Tensor(rng.uniform(0.5, 2.0, size=(3, 4)))  # positive, away from 0
    _check_op_with_explicit_grad(lambda ps: ps[0] ** 3, [a])
    _check_op_with_explicit_grad(lambda ps: ps[0] ** -2, [a])


def test_grad_neg():
    rng = np.random.default_rng(0)
    a = _rand_tensor(rng, 3, 5)
    _check_op_with_explicit_grad(lambda ps: -ps[0], [a])


def test_grad_composed_expression():
    # f(a, b) = (a + b) * (a - b) = a^2 - b^2
    rng = np.random.default_rng(0)
    a, b = _rand_tensor(rng, 4, 3), _rand_tensor(rng, 4, 3)
    _check_op_with_explicit_grad(
        lambda ps: (ps[0] + ps[1]) * (ps[0] - ps[1]),
        [a, b],
    )


def test_grad_aliased_input():
    # f(a) = a * a + a — same Tensor object referenced multiple times.
    rng = np.random.default_rng(0)
    a = _rand_tensor(rng, 3, 4)
    _check_op_with_explicit_grad(lambda ps: ps[0] * ps[0] + ps[0], [a])


# ----------------------------------------------------------------------
# Reverse-arith with Python numbers on the left.
# ----------------------------------------------------------------------


def test_grad_radd_rsub_rmul_rdiv():
    rng = np.random.default_rng(0)
    a = Tensor(rng.uniform(0.5, 2.0, size=(2, 3)))
    _check_op_with_explicit_grad(lambda ps: 2.0 + ps[0], [a])
    _check_op_with_explicit_grad(lambda ps: 5.0 - ps[0], [a])
    _check_op_with_explicit_grad(lambda ps: 3.0 * ps[0], [a])
    _check_op_with_explicit_grad(lambda ps: 1.0 / ps[0], [a])


# ----------------------------------------------------------------------
# Backward semantics: scalar output requires no grad arg; non-scalar requires
# explicit seed.
# ----------------------------------------------------------------------


def test_backward_non_scalar_without_grad_raises():
    a = Tensor([1.0, 2.0])
    with pytest.raises(RuntimeError, match="scalar output"):
        a.backward()


def test_backward_seed_propagates_correctly():
    # With a constant 2x grad seed on a*b, da should be 2*b and db should be 2*a.
    a, b = Tensor([1.0, 2.0]), Tensor([3.0, 4.0])
    out = a * b
    out.backward(grad=np.full_like(out.data, 2.0))
    np.testing.assert_array_equal(a.grad, [6.0, 8.0])
    np.testing.assert_array_equal(b.grad, [2.0, 4.0])


# ----------------------------------------------------------------------
# Reductions: sum, mean.
# ----------------------------------------------------------------------


def test_sum_forward_full():
    a = Tensor([[1, 2], [3, 4]])
    assert a.sum().data == 10.0
    assert a.sum().shape == ()


def test_sum_forward_axis():
    a = Tensor([[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(a.sum(axis=0).data, [5, 7, 9])
    np.testing.assert_array_equal(a.sum(axis=1).data, [6, 15])
    np.testing.assert_array_equal(a.sum(axis=1, keepdims=True).data, [[6], [15]])


def test_grad_sum_full():
    # dL/da = 1 for every element if loss = a.sum()
    rng = np.random.default_rng(0)
    a = _rand_tensor(rng, 3, 4)
    numgrad_check(lambda ps: ps[0].sum(), [a])


def test_grad_sum_axis():
    rng = np.random.default_rng(0)
    a = _rand_tensor(rng, 4, 5)
    numgrad_check(lambda ps: ps[0].sum(axis=0).sum(), [a])
    numgrad_check(lambda ps: ps[0].sum(axis=1).sum(), [a])


def test_grad_sum_keepdims():
    rng = np.random.default_rng(0)
    a = _rand_tensor(rng, 3, 4, 5)
    # axis=1, keepdims preserves the size-1 dim — backward should still
    # broadcast correctly back to the original shape.
    numgrad_check(lambda ps: ps[0].sum(axis=1, keepdims=True).sum(), [a])


def test_grad_sum_multi_axis():
    rng = np.random.default_rng(0)
    a = _rand_tensor(rng, 3, 4, 5)
    numgrad_check(lambda ps: ps[0].sum(axis=(0, 2)).sum(), [a])


def test_mean_forward():
    a = Tensor([[1, 2], [3, 4]])
    assert a.mean().data == 2.5
    np.testing.assert_array_equal(a.mean(axis=0).data, [2, 3])


def test_grad_mean_full():
    # dL/da = 1/n for every element
    rng = np.random.default_rng(0)
    a = _rand_tensor(rng, 3, 4)
    numgrad_check(lambda ps: ps[0].mean(), [a])


def test_grad_mean_axis():
    rng = np.random.default_rng(0)
    a = _rand_tensor(rng, 4, 5)
    numgrad_check(lambda ps: ps[0].mean(axis=0).sum(), [a])
    numgrad_check(lambda ps: ps[0].mean(axis=1).sum(), [a])


def test_grad_composed_with_reduction():
    # f(a, b) = ((a + b) * a).mean() — a appears in two paths
    rng = np.random.default_rng(0)
    a, b = _rand_tensor(rng, 3, 4), _rand_tensor(rng, 3, 4)
    numgrad_check(lambda ps: ((ps[0] + ps[1]) * ps[0]).mean(), [a, b])


def test_grad_composed_with_broadcast_and_reduction():
    # f(a, b) = (a * b).sum(), with b broadcast across leading dim of a
    rng = np.random.default_rng(0)
    a, b = _rand_tensor(rng, 5, 7), _rand_tensor(rng, 7)
    numgrad_check(lambda ps: (ps[0] * ps[1]).sum(), [a, b])


# ----------------------------------------------------------------------
# Matrix multiply.
# ----------------------------------------------------------------------


def test_matmul_forward_2d():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])
    np.testing.assert_array_equal((a @ b).data, [[19, 22], [43, 50]])


def test_matmul_forward_batched_left():
    a = Tensor(np.random.default_rng(0).normal(size=(3, 4, 5)))
    b = Tensor(np.random.default_rng(1).normal(size=(5, 7)))
    out = a @ b
    assert out.shape == (3, 4, 7)
    np.testing.assert_allclose(out.data, a.data @ b.data)


def test_grad_matmul_2d():
    rng = np.random.default_rng(0)
    a = _rand_tensor(rng, 4, 5)
    b = _rand_tensor(rng, 5, 3)
    numgrad_check(lambda ps: (ps[0] @ ps[1]).sum(), [a, b])


def test_grad_matmul_batched_left_only():
    # Common MLP shape: X (B, D_in) @ W (D_in, D_out)  — here we test the
    # 3D-on-the-left version (B, T, D_in) @ W (D_in, D_out), which is what
    # we'll need for transformer-style activations later.
    rng = np.random.default_rng(0)
    X = _rand_tensor(rng, 2, 3, 5)
    W = _rand_tensor(rng, 5, 4)
    numgrad_check(lambda ps: (ps[0] @ ps[1]).sum(), [X, W])


def test_grad_matmul_both_batched_same_batch():
    # (B, M, K) @ (B, K, N) — what scaled-dot-product attention will use.
    rng = np.random.default_rng(0)
    A = _rand_tensor(rng, 2, 4, 5)
    B = _rand_tensor(rng, 2, 5, 3)
    numgrad_check(lambda ps: (ps[0] @ ps[1]).sum(), [A, B])


def test_grad_matmul_with_bias_and_activation_chain():
    # Mini linear-layer pattern: (X @ W + b).sum() with W shared and b broadcast.
    rng = np.random.default_rng(0)
    X = _rand_tensor(rng, 3, 4)  # batch_size=3, in_dim=4
    W = _rand_tensor(rng, 4, 5)  # out_dim=5
    b = _rand_tensor(rng, 5)  # broadcast across batch
    numgrad_check(lambda ps: ((ps[0] @ ps[1]) + ps[2]).sum(), [X, W, b])


# ----------------------------------------------------------------------
# Gather (embedding lookup) and reshape.
# ----------------------------------------------------------------------


def test_gather_forward_1d_index():
    W = Tensor([[1, 2], [3, 4], [5, 6]])  # (3, 2)
    idx = np.array([0, 2, 1, 0])
    out = W[idx]
    assert out.shape == (4, 2)
    np.testing.assert_array_equal(out.data, [[1, 2], [5, 6], [3, 4], [1, 2]])


def test_gather_forward_2d_index():
    W = Tensor([[1, 2], [3, 4], [5, 6]])  # (3, 2)
    idx = np.array([[0, 1], [2, 0]])  # (2, 2)
    out = W[idx]
    assert out.shape == (2, 2, 2)
    np.testing.assert_array_equal(out.data, [[[1, 2], [3, 4]], [[5, 6], [1, 2]]])


def test_grad_gather_unique_indices():
    rng = np.random.default_rng(0)
    W = _rand_tensor(rng, 5, 3)
    idx = np.array([0, 2, 4])
    numgrad_check(lambda ps: ps[0][idx].sum(), [W])


def test_grad_gather_repeated_indices_accumulate():
    # The same row is indexed twice — its grad must accumulate, not overwrite.
    # This is exactly the bug `np.add.at` exists to prevent (the buffered
    # `arr[idx] += val` would only count one of the two contributions).
    rng = np.random.default_rng(0)
    W = _rand_tensor(rng, 4, 3)
    idx = np.array([1, 1, 2, 0, 1])
    numgrad_check(lambda ps: ps[0][idx].sum(), [W])


def test_grad_gather_2d_index():
    rng = np.random.default_rng(0)
    W = _rand_tensor(rng, 6, 4)
    idx = np.array([[0, 3, 5], [1, 1, 2]])
    numgrad_check(lambda ps: ps[0][idx].sum(), [W])


def test_gather_rejects_non_int_index():
    W = Tensor([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(TypeError, match="integer ndarray"):
        W[np.array([0.0, 1.0])]
    with pytest.raises(TypeError, match="integer ndarray"):
        W[[0, 1]]  # plain list, not ndarray


def test_reshape_forward():
    a = Tensor(np.arange(12.0).reshape(3, 4))
    out = a.reshape(2, 6)
    assert out.shape == (2, 6)
    np.testing.assert_array_equal(out.data, np.arange(12.0).reshape(2, 6))


def test_grad_reshape():
    rng = np.random.default_rng(0)
    a = _rand_tensor(rng, 3, 4)
    numgrad_check(lambda ps: ps[0].reshape(2, 6).sum(), [a])


def test_grad_gather_then_reshape_then_matmul():
    # The exact pattern the MLP forward will use:
    #   W_emb (V, D) gathered by idx (B, T) -> (B, T, D)
    #   reshape -> (B, T*D)
    #   matmul with W (T*D, H) -> (B, H)
    rng = np.random.default_rng(0)
    V, D, T, B, H = 6, 3, 2, 4, 5
    W_emb = _rand_tensor(rng, V, D)
    W_lin = _rand_tensor(rng, T * D, H)
    idx = rng.integers(0, V, size=(B, T))

    def forward(ps):
        emb = ps[0][idx]  # (B, T, D)
        flat = emb.reshape(B, T * D)
        h = flat @ ps[1]  # (B, H)
        return h.sum()

    numgrad_check(forward, [W_emb, W_lin])


# ----------------------------------------------------------------------
# Activations and log_softmax.
# ----------------------------------------------------------------------


def test_relu_forward():
    a = Tensor([-2.0, -0.1, 0.0, 0.5, 3.0])
    np.testing.assert_array_equal(a.relu().data, [0, 0, 0, 0.5, 3.0])


def test_grad_relu():
    rng = np.random.default_rng(0)
    a = _rand_tensor(rng, 4, 5)
    numgrad_check(lambda ps: ps[0].relu().sum(), [a])


def test_tanh_forward():
    a = Tensor([0.0, 1.0, -1.0])
    np.testing.assert_allclose(a.tanh().data, [0.0, np.tanh(1.0), -np.tanh(1.0)])


def test_grad_tanh():
    rng = np.random.default_rng(0)
    a = _rand_tensor(rng, 3, 4)
    numgrad_check(lambda ps: ps[0].tanh().sum(), [a])


def test_log_softmax_forward_sums_to_one():
    rng = np.random.default_rng(0)
    a = Tensor(rng.normal(size=(5, 7)))
    log_p = a.log_softmax(axis=-1)
    np.testing.assert_allclose(np.exp(log_p.data).sum(axis=-1), np.ones(5), atol=1e-12)


def test_log_softmax_numerically_stable():
    # Without the log-sum-exp shift, exp(1000) overflows to inf.
    a = Tensor([[1000.0, 1001.0, 1002.0]])
    log_p = a.log_softmax(axis=-1)
    assert np.all(np.isfinite(log_p.data))
    np.testing.assert_allclose(np.exp(log_p.data).sum(), 1.0, atol=1e-12)


def test_grad_log_softmax_axis_minus_one():
    rng = np.random.default_rng(0)
    a = _rand_tensor(rng, 4, 6)
    numgrad_check(lambda ps: ps[0].log_softmax(axis=-1).sum(), [a])


def test_grad_log_softmax_axis_zero():
    rng = np.random.default_rng(0)
    a = _rand_tensor(rng, 4, 6)
    numgrad_check(lambda ps: ps[0].log_softmax(axis=0).sum(), [a])


# ----------------------------------------------------------------------
# Cross-entropy (the fused softmax + NLL loss).
# ----------------------------------------------------------------------


def test_cross_entropy_forward_matches_manual():
    from jacobigrad.nn.losses import cross_entropy

    rng = np.random.default_rng(0)
    logits = Tensor(rng.normal(size=(4, 5)))
    targets = np.array([0, 2, 4, 1])

    out = cross_entropy(logits, targets)
    # Manual: stable log_softmax + NLL.
    m = logits.data.max(axis=-1, keepdims=True)
    log_p = (logits.data - m) - np.log(np.exp(logits.data - m).sum(axis=-1, keepdims=True))
    expected = -log_p[np.arange(4), targets].mean()
    np.testing.assert_allclose(out.data, expected, atol=1e-12)


def test_grad_cross_entropy():
    from jacobigrad.nn.losses import cross_entropy

    rng = np.random.default_rng(0)
    logits = _rand_tensor(rng, 6, 8)
    targets = rng.integers(0, 8, size=6)
    numgrad_check(lambda ps: cross_entropy(ps[0], targets), [logits])


def test_grad_full_mlp_pattern():
    # End-to-end: embedding -> reshape -> linear -> tanh -> linear -> CE
    from jacobigrad.nn.losses import cross_entropy

    rng = np.random.default_rng(0)
    V, D, T, B, H = 6, 3, 2, 5, 7
    W_emb = _rand_tensor(rng, V, D)
    W1 = _rand_tensor(rng, T * D, H)
    b1 = _rand_tensor(rng, H)
    W2 = _rand_tensor(rng, H, V)
    b2 = _rand_tensor(rng, V)
    idx = rng.integers(0, V, size=(B, T))
    targets = rng.integers(0, V, size=B)

    def forward(ps):
        emb = ps[0][idx]
        flat = emb.reshape(B, T * D)
        h = (flat @ ps[1] + ps[2]).tanh()
        logits = h @ ps[3] + ps[4]
        return cross_entropy(logits, targets)

    numgrad_check(forward, [W_emb, W1, b1, W2, b2])
