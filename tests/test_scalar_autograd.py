"""Tests for jacobigrad.autograd.scalar.Value.

Strategy: for each op (and a few non-trivial composed expressions), build
the graph with `Value`, call backward, and compare the analytic gradients
against central-difference estimates computed from a plain-Python `math`
implementation of the same function. This is the same gradient-check
discipline we will lift to tensor autograd in Week 2.
"""

from __future__ import annotations

import math

import pytest

from jacobigrad.autograd.scalar import Value


# ----------------------------------------------------------------------
# Gradient-check helper.
# ----------------------------------------------------------------------


def grad_check(value_fn, float_fn, xs, eps: float = 1e-5, atol: float = 1e-5):
    """Compare backward(value_fn(*xs)) vs central-diff of float_fn at xs."""
    vs = [Value(x) for x in xs]
    out = value_fn(*vs)
    assert isinstance(out, Value), "value_fn must return a Value"
    out.backward()
    analytic = [v.grad for v in vs]

    numerical = []
    for i in range(len(xs)):
        plus = list(xs); plus[i] += eps
        minus = list(xs); minus[i] -= eps
        numerical.append((float_fn(*plus) - float_fn(*minus)) / (2 * eps))

    for i, (a, n) in enumerate(zip(analytic, numerical)):
        assert abs(a - n) < atol, f"grad arg {i}: analytic={a:.6e}, numerical={n:.6e}"
    return out, analytic, numerical


# ----------------------------------------------------------------------
# Forward correctness.
# ----------------------------------------------------------------------


def test_forward_arithmetic():
    a, b = Value(2.0), Value(-3.0)
    assert (a + b).data == pytest.approx(-1.0)
    assert (a - b).data == pytest.approx(5.0)
    assert (a * b).data == pytest.approx(-6.0)
    assert (a / b).data == pytest.approx(2.0 / -3.0)
    assert (a**3).data == pytest.approx(8.0)
    assert (-a).data == pytest.approx(-2.0)


def test_forward_unary():
    assert Value(0.5).tanh().data == pytest.approx(math.tanh(0.5))
    assert Value(1.5).exp().data == pytest.approx(math.exp(1.5))
    assert Value(2.0).log().data == pytest.approx(math.log(2.0))
    assert Value(-1.0).relu().data == 0.0
    assert Value(2.5).relu().data == pytest.approx(2.5)


def test_log_of_nonpositive_raises():
    with pytest.raises(ValueError, match="non-positive"):
        Value(0.0).log()
    with pytest.raises(ValueError, match="non-positive"):
        Value(-1.0).log()


# ----------------------------------------------------------------------
# Backward for each op individually.
# ----------------------------------------------------------------------


def test_backward_add():
    grad_check(lambda a, b: a + b, lambda a, b: a + b, [0.7, -1.3])


def test_backward_mul():
    grad_check(lambda a, b: a * b, lambda a, b: a * b, [0.7, -1.3])


def test_backward_sub():
    grad_check(lambda a, b: a - b, lambda a, b: a - b, [0.7, -1.3])


def test_backward_div():
    grad_check(lambda a, b: a / b, lambda a, b: a / b, [0.7, -1.3])


def test_backward_pow():
    grad_check(lambda a: a**3, lambda a: a**3, [1.4])
    grad_check(lambda a: a**-2, lambda a: a**-2, [1.4])  # negative exponent
    grad_check(lambda a: a**0.5, lambda a: a**0.5, [2.0])  # fractional


def test_backward_relu_positive_branch():
    grad_check(lambda a: a.relu(), lambda a: max(0.0, a), [0.7])


def test_backward_relu_negative_branch():
    grad_check(lambda a: a.relu(), lambda a: max(0.0, a), [-0.7])


def test_backward_tanh():
    grad_check(lambda a: a.tanh(), math.tanh, [0.5])
    grad_check(lambda a: a.tanh(), math.tanh, [-2.0])


def test_backward_exp():
    grad_check(lambda a: a.exp(), math.exp, [0.5])


def test_backward_log():
    grad_check(lambda a: a.log(), math.log, [2.5])


# ----------------------------------------------------------------------
# Backward for composed expressions.
# ----------------------------------------------------------------------


def test_backward_polynomial():
    # f(a, b, c) = (a*b + c)^2
    grad_check(
        lambda a, b, c: (a * b + c) ** 2,
        lambda a, b, c: (a * b + c) ** 2,
        [0.3, -0.7, 1.1],
    )


def test_backward_logistic_neuron():
    # f(x, w, b) = tanh(w*x + b) — a one-neuron MLP forward pass
    grad_check(
        lambda x, w, b: (w * x + b).tanh(),
        lambda x, w, b: math.tanh(w * x + b),
        [1.5, 0.5, -0.3],
    )


def test_backward_two_layer_neuron():
    # f = tanh(w2 * relu(w1 * x + b1) + b2)
    def vfn(x, w1, b1, w2, b2):
        return (w2 * (w1 * x + b1).relu() + b2).tanh()

    def ffn(x, w1, b1, w2, b2):
        return math.tanh(w2 * max(0.0, w1 * x + b1) + b2)

    grad_check(vfn, ffn, [0.8, 0.5, -0.2, 1.3, 0.1])


def test_backward_log_softmax_two_class():
    # log p(class_0) = a - log(exp(a) + exp(b))
    # Built with Value primitives only.
    def vfn(a, b):
        return a - (a.exp() + b.exp()).log()

    def ffn(a, b):
        return a - math.log(math.exp(a) + math.exp(b))

    grad_check(vfn, ffn, [0.7, -0.4])


# ----------------------------------------------------------------------
# DAG semantics: aliased and shared subexpressions must accumulate.
# ----------------------------------------------------------------------


def test_aliased_input_doubles_gradient():
    # f(a) = a + a → df/da = 2. Two references to the same Value object.
    a = Value(1.7)
    out = a + a
    out.backward()
    assert a.grad == pytest.approx(2.0)


def test_aliased_input_in_product_uses_chain_rule():
    # f(a) = a * a = a^2 → df/da = 2a
    a = Value(1.7)
    out = a * a
    out.backward()
    assert a.grad == pytest.approx(2 * 1.7)


def test_shared_intermediate_accumulates():
    # h = a * b
    # f = h + h * 3   (h appears twice as a child of two parents)
    # df/da = b + 3b = 4b ; df/db = a + 3a = 4a
    a, b = Value(0.7), Value(-1.3)
    h = a * b
    out = h + h * 3
    out.backward()
    assert a.grad == pytest.approx(4 * -1.3)
    assert b.grad == pytest.approx(4 * 0.7)


def test_topological_order_correctness():
    # Diamond DAG:
    #     a
    #    / \
    #   x   y
    #    \ /
    #     z
    # x = a * 2,  y = a + 5,  z = x * y
    # dz/da = y * dx/da + x * dy/da = (a+5)*2 + (a*2)*1 = 2a+10 + 2a = 4a + 10
    a = Value(2.5)
    x = a * 2
    y = a + 5
    z = x * y
    z.backward()
    assert a.grad == pytest.approx(4 * 2.5 + 10)


# ----------------------------------------------------------------------
# Reverse-arithmetic (when the left operand is a Python number).
# ----------------------------------------------------------------------


def test_reverse_ops_forward():
    a = Value(2.0)
    assert (3 + a).data == 5.0
    assert (3 - a).data == 1.0
    assert (3 * a).data == 6.0
    assert (3 / a).data == 1.5


def test_reverse_ops_backward():
    grad_check(lambda a: 3 + a, lambda a: 3 + a, [2.0])
    grad_check(lambda a: 3 - a, lambda a: 3 - a, [2.0])
    grad_check(lambda a: 3 * a, lambda a: 3 * a, [2.0])
    grad_check(lambda a: 3 / a, lambda a: 3 / a, [2.0])
