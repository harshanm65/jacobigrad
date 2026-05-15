"""Scalar reverse-mode autograd, in the spirit of Karpathy's micrograd.

This is the smallest possible reverse-mode autograd engine: each Value wraps
a Python float, and each elementary op records (i) the children it depends
on and (ii) a backward closure that distributes the upstream gradient
through the op's local Jacobian.

The pattern is identical to what we will lift to tensors in Week 2:

  forward op:   out = f(a, b, ...)            # build a new node
                out._prev = (a, b, ...)        # remember inputs
                out._backward = closure        # how to push grads upstream

  backward():   topo_sort(start_from=output)
                output.grad = 1
                for node in reverse(topo):
                    node._backward()           # accumulates grads onto _prev

Gradients ACCUMULATE (`+=`) so that a node feeding into multiple downstream
ops collects contributions from all of them. The topological order ensures
every downstream gradient is fully accumulated before we propagate further
upstream.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from numbers import Real


class Value:
    """A scalar with autograd. Wraps a float and remembers how it was made."""

    __slots__ = ("data", "grad", "_prev", "_op", "_backward")

    def __init__(
        self,
        data: float,
        _children: Iterable["Value"] = (),
        _op: str = "",
    ):
        self.data: float = float(data)
        self.grad: float = 0.0
        # Use a tuple, not a set — set membership relies on hash(Value) which
        # is identity-based by default. Tuple is enough; the topo-sort dedups
        # via a `visited` set keyed by id.
        self._prev: tuple[Value, ...] = tuple(_children)
        self._op: str = _op
        self._backward: Callable[[], None] = _no_op_backward

    # ------------------------------------------------------------------
    # Elementary ops. Each op:
    #   1. Computes the forward value.
    #   2. Builds a new Value with the right children + op name.
    #   3. Defines a closure capturing `self`, `other`, `out` to distribute
    #      d(out) backward through the local Jacobian of the op.
    # ------------------------------------------------------------------

    def __add__(self, other: "Value | Real") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            # d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: "Value | Real") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            # d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent: Real) -> "Value":
        # Constant exponent only; raising to a Value exponent is rare in NN
        # math and the closed form (x^y * log x) needs both legs of the chain.
        if not isinstance(exponent, Real) or isinstance(exponent, bool):
            raise TypeError(f"exponent must be a real number, got {type(exponent).__name__}")
        n = float(exponent)
        out = Value(self.data**n, (self,), f"**{n}")

        def _backward() -> None:
            # d(x**n)/dx = n * x**(n-1)
            self.grad += n * (self.data ** (n - 1)) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> "Value":
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward() -> None:
            # d(relu)/dx = 1 if x > 0 else 0. (Convention: 0 at x = 0.)
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Value":
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            # d(tanh)/dx = 1 - tanh(x)^2
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> "Value":
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward() -> None:
            # d(exp)/dx = exp(x); reuse the cached forward value.
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self) -> "Value":
        if self.data <= 0:
            raise ValueError(f"log of non-positive value: {self.data}")
        out = Value(math.log(self.data), (self,), "log")

        def _backward() -> None:
            # d(log)/dx = 1/x
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Convenience ops — defined in terms of the elementary ones so we
    # don't need extra backward closures.
    # ------------------------------------------------------------------

    def __neg__(self) -> "Value":
        return self * -1

    def __sub__(self, other: "Value | Real") -> "Value":
        return self + (-other if isinstance(other, Value) else -float(other))

    def __truediv__(self, other: "Value | Real") -> "Value":
        if not isinstance(other, Value):
            other = Value(other)
        return self * (other**-1)

    def __radd__(self, other: "Value | Real") -> "Value":
        return self + other

    def __rsub__(self, other: "Value | Real") -> "Value":
        return (-self) + other

    def __rmul__(self, other: "Value | Real") -> "Value":
        return self * other

    def __rtruediv__(self, other: "Value | Real") -> "Value":
        return (self**-1) * other

    # ------------------------------------------------------------------
    # The reverse pass.
    # ------------------------------------------------------------------

    def backward(self) -> None:
        """Run reverse-mode autograd from this node back to all ancestors.

        Sets self.grad = 1 (treating self as the scalar output) and
        accumulates d(self)/d(ancestor) into ancestor.grad for every
        ancestor reachable through _prev. Does not zero existing grads
        on ancestors — caller must zero between independent backward
        passes if it intends to reuse the graph.
        """
        topo: list[Value] = []
        visited: set[int] = set()

        def build(node: Value) -> None:
            # id-keyed visited set: Value has no custom __hash__, but using
            # id is more explicit about "we mean object identity here."
            if id(node) in visited:
                return
            visited.add(id(node))
            for child in node._prev:
                build(child)
            topo.append(node)

        build(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g}, op={self._op!r})"


def _no_op_backward() -> None:
    pass
