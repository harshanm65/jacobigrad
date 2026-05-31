"""Reverse-mode autograd over batched tensors.

Same closure-based local-Jacobian pattern as ``scalar.py``, lifted to numpy
arrays. The two genuine complications relative to the scalar engine:

  1. **Broadcasting in backward.** When a forward op broadcasts (e.g. adding
     a (D,) bias to a (B, D) batch), the upstream gradient comes back with
     the broadcasted shape and must be **summed along broadcast axes** to
     match each input's actual shape. ``_unbroadcast`` is the helper.

  2. **Lazy grad allocation.** A parameter's ``.grad`` is ``None`` until the
     first contribution arrives — then it's allocated once and accumulated
     into. Saves memory for graphs that don't touch every leaf.

The graph is otherwise identical: each forward op records its inputs in
``_prev`` and installs a ``_backward`` closure that pushes upstream grad
through the local Jacobian. ``backward()`` topologically sorts from the
output and walks in reverse.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from numbers import Real

import numpy as np


ArrayLike = np.ndarray | float | int | list


def _unbroadcast(grad: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Reduce ``grad`` to ``target_shape`` by summing along broadcast axes.

    Numpy broadcasting rules: missing leading dims are treated as size 1, and
    size-1 dims broadcast against any size. Backward reverses this:

      - Sum over leading dims that don't exist in ``target_shape``.
      - Sum over axes where the target has size 1 (keepdims).
    """
    # Drop leading dims that were broadcast in.
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    # Reduce size-1 axes.
    for axis, size in enumerate(target_shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    """A numpy array with autograd. Wraps ``data`` and remembers how it was made."""

    __slots__ = ("data", "grad", "_prev", "_op", "_backward")

    def __init__(
        self,
        data: ArrayLike,
        _children: Iterable["Tensor"] = (),
        _op: str = "",
    ):
        if isinstance(data, np.ndarray):
            self.data: np.ndarray = data.astype(np.float64, copy=False)
        else:
            self.data = np.asarray(data, dtype=np.float64)
        self.grad: np.ndarray | None = None
        self._prev: tuple[Tensor, ...] = tuple(_children)
        self._op: str = _op
        self._backward: Callable[[], None] = _no_op_backward

    # ------------------------------------------------------------------
    # Properties + housekeeping
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    def zero_grad(self) -> None:
        self.grad = None

    def _add_grad(self, g: np.ndarray) -> None:
        """Lazy-allocate ``self.grad`` on first contribution, then accumulate."""
        if self.grad is None:
            self.grad = g.astype(np.float64, copy=True)
        else:
            self.grad += g

    # ------------------------------------------------------------------
    # Elementwise binary ops (with broadcasting).
    #
    # Each closure captures (self, other, out) and uses _unbroadcast to
    # collapse the broadcasted upstream grad back to each input's shape.
    # ------------------------------------------------------------------

    def __add__(self, other: "Tensor | Real | np.ndarray") -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            # d(a+b)/da = 1, d(a+b)/db = 1
            self._add_grad(_unbroadcast(out.grad, self.shape))
            other._add_grad(_unbroadcast(out.grad, other.shape))

        out._backward = _backward
        return out

    def __mul__(self, other: "Tensor | Real | np.ndarray") -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            # d(a*b)/da = b, d(a*b)/db = a; multiply by upstream then unbroadcast.
            self._add_grad(_unbroadcast(out.grad * other.data, self.shape))
            other._add_grad(_unbroadcast(out.grad * self.data, other.shape))

        out._backward = _backward
        return out

    def __sub__(self, other: "Tensor | Real | np.ndarray") -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, (self, other), "-")

        def _backward() -> None:
            self._add_grad(_unbroadcast(out.grad, self.shape))
            other._add_grad(_unbroadcast(-out.grad, other.shape))

        out._backward = _backward
        return out

    def __truediv__(self, other: "Tensor | Real | np.ndarray") -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, (self, other), "/")

        def _backward() -> None:
            # d(a/b)/da = 1/b
            self._add_grad(_unbroadcast(out.grad / other.data, self.shape))
            # d(a/b)/db = -a/b^2
            other._add_grad(
                _unbroadcast(-out.grad * self.data / (other.data ** 2), other.shape)
            )

        out._backward = _backward
        return out

    def __pow__(self, exponent: Real) -> "Tensor":
        if not isinstance(exponent, Real) or isinstance(exponent, bool):
            raise TypeError(f"exponent must be a real number, got {type(exponent).__name__}")
        n = float(exponent)
        out = Tensor(self.data ** n, (self,), f"**{n}")

        def _backward() -> None:
            # d(x**n)/dx = n * x**(n-1)
            self._add_grad(out.grad * n * (self.data ** (n - 1)))

        out._backward = _backward
        return out

    def __neg__(self) -> "Tensor":
        out = Tensor(-self.data, (self,), "neg")

        def _backward() -> None:
            self._add_grad(-out.grad)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Matrix multiply.
    #
    # For C = A @ B:
    #   dA = dC @ B^T    (B^T means swap the last two axes)
    #   dB = A^T @ dC
    #
    # When one operand has fewer batch dims than the other (e.g. a shared
    # weight matrix W (D, K) used across a batched activation X (B, T, D)),
    # numpy broadcasts on the forward and we must sum over the broadcast
    # leading axes on the backward — that's just _unbroadcast.
    # ------------------------------------------------------------------

    def __matmul__(self, other: "Tensor | np.ndarray") -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(other)
        if self.ndim < 2 or other.ndim < 2:
            raise NotImplementedError(
                "Tensor matmul currently requires both operands to have ndim >= 2; "
                f"got self.ndim={self.ndim}, other.ndim={other.ndim}"
            )
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward() -> None:
            grad_self = out.grad @ np.swapaxes(other.data, -1, -2)
            grad_other = np.swapaxes(self.data, -1, -2) @ out.grad
            self._add_grad(_unbroadcast(grad_self, self.shape))
            other._add_grad(_unbroadcast(grad_other, other.shape))

        out._backward = _backward
        return out

    def __rmatmul__(self, other: "np.ndarray") -> "Tensor":
        # For ``np.ndarray @ Tensor`` — wrap and delegate.
        return Tensor(other) @ self

    # ------------------------------------------------------------------
    # Indexing (embedding lookup) and reshape.
    # ------------------------------------------------------------------

    def __getitem__(self, idx: np.ndarray) -> "Tensor":
        """Integer-array gather along axis 0 — i.e. an embedding lookup.

        For ``W`` of shape ``(V, D)`` and integer ``idx`` of any shape ``S``,
        returns a Tensor of shape ``S + (D,)``. The backward is a scatter-add:
        every gradient slice is added back to the row of ``W`` it came from
        (``np.add.at`` handles repeated indices correctly).
        """
        if not isinstance(idx, np.ndarray) or idx.dtype.kind not in "iu":
            raise TypeError(
                f"Tensor indexing requires an integer ndarray; got {type(idx).__name__}"
            )
        out = Tensor(self.data[idx], (self,), "gather")

        def _backward() -> None:
            grad = np.zeros_like(self.data)
            # np.add.at is the unbuffered scatter — repeated indices in idx
            # accumulate correctly, unlike the buffered grad[idx] += out.grad.
            np.add.at(grad, idx, out.grad)
            self._add_grad(grad)

        out._backward = _backward
        return out

    def reshape(self, *shape: int) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        out = Tensor(self.data.reshape(shape), (self,), f"reshape({shape})")

        def _backward() -> None:
            self._add_grad(out.grad.reshape(self.shape))

        out._backward = _backward
        return out

    def transpose(self, axis1: int = -2, axis2: int = -1) -> "Tensor":
        """Swap two axes (default: the last two — a batched matrix transpose).

        Needed for attention's ``Q @ K^T``: the matmul op multiplies the last
        two axes, so transposing ``K`` is how we contract over the feature dim.
        ``swapaxes`` is its own inverse for a given axis pair, so the backward
        just swaps the upstream grad back.
        """
        out = Tensor(np.swapaxes(self.data, axis1, axis2), (self,), f"transpose({axis1},{axis2})")

        def _backward() -> None:
            self._add_grad(np.swapaxes(out.grad, axis1, axis2))

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Activations and the (numerically stable) log-softmax.
    # ------------------------------------------------------------------

    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(self.data, 0.0), (self,), "relu")

        def _backward() -> None:
            # d(relu)/dx = 1 where x > 0, else 0. Use the post-activation value
            # to determine the mask — equivalent and one fewer comparison.
            self._add_grad(out.grad * (out.data > 0))

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")

        def _backward() -> None:
            # d(tanh)/dx = 1 - tanh(x)^2; reuse the cached forward value.
            self._add_grad(out.grad * (1.0 - t * t))

        out._backward = _backward
        return out

    def log_softmax(self, axis: int = -1) -> "Tensor":
        """Numerically stable log-softmax along ``axis``.

        Forward uses the log-sum-exp shift to keep ``exp`` inputs <= 0.

        Backward: with ``y_i = x_i - log Z``, ``dy_i/dx_k = delta_ik - p_k``
        where ``p_k = softmax(x)_k``. Chain rule gives
            ``dL/dx_k = dL/dy_k - p_k * sum_i(dL/dy_i)``
        — the upstream grad with the row-weighted softmax mean subtracted.
        """
        m = self.data.max(axis=axis, keepdims=True)
        shifted = self.data - m
        log_z = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
        log_p = shifted - log_z
        out = Tensor(log_p, (self,), f"log_softmax(axis={axis})")

        def _backward() -> None:
            p = np.exp(log_p)
            self._add_grad(out.grad - p * out.grad.sum(axis=axis, keepdims=True))

        out._backward = _backward
        return out

    def softmax(self, axis: int = -1) -> "Tensor":
        """Numerically stable softmax along ``axis``, returning probabilities.

        ``log_softmax`` already covers cross-entropy, but attention needs the
        probabilities themselves (the attention weights ``A`` that multiply
        ``V``). The backward is the full softmax-row Jacobian collapse — the
        same identity Luis derived for attention (``attention_backward.md``):

            with ``a = softmax(x)``,   ``da_j/dx_i = a_j (delta_ij - a_i)``
            ==> ``dL/dx_i = a_i * (dL/da_i - <a, dL/da>)``

        The dot product ``<a, dL/da>`` is one scalar shared across the row, so
        the same softmax-weighted correction is subtracted at every position.
        """
        m = self.data.max(axis=axis, keepdims=True)
        e = np.exp(self.data - m)
        a = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(a, (self,), f"softmax(axis={axis})")

        def _backward() -> None:
            # row-weighted correction: a * (g - sum(a * g))
            dot = (a * out.grad).sum(axis=axis, keepdims=True)
            self._add_grad(a * (out.grad - dot))

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Reductions: sum, mean.
    #
    # Backward broadcasts the upstream grad back to ``self.shape``. For
    # axis-wise reductions with ``keepdims=False`` we re-insert the
    # collapsed size-1 axes before broadcasting.
    # ------------------------------------------------------------------

    def sum(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            (self,),
            f"sum(axis={axis}, keepdims={keepdims})",
        )

        def _backward() -> None:
            g = _broadcast_grad_to(out.grad, self.shape, axis, keepdims)
            self._add_grad(g)

        out._backward = _backward
        return out

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> "Tensor":
        out = Tensor(
            self.data.mean(axis=axis, keepdims=keepdims),
            (self,),
            f"mean(axis={axis}, keepdims={keepdims})",
        )

        # The denominator is the number of elements averaged over.
        if axis is None:
            n = self.size
        else:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            n = int(np.prod([self.shape[a] for a in axes]))

        def _backward() -> None:
            # Mean = sum / n, so its gradient is sum's gradient / n.
            g = _broadcast_grad_to(out.grad / n, self.shape, axis, keepdims)
            self._add_grad(g)

        out._backward = _backward
        return out

    # Reverse-arith for the case where a Python number / ndarray is on the left.
    def __radd__(self, other: "Real | np.ndarray") -> "Tensor":
        return self + other

    def __rsub__(self, other: "Real | np.ndarray") -> "Tensor":
        return (-self) + other

    def __rmul__(self, other: "Real | np.ndarray") -> "Tensor":
        return self * other

    def __rtruediv__(self, other: "Real | np.ndarray") -> "Tensor":
        return Tensor(other) / self

    # ------------------------------------------------------------------
    # The reverse pass.
    # ------------------------------------------------------------------

    def backward(self, grad: np.ndarray | None = None) -> None:
        """Run reverse-mode autograd from this node to every reachable ancestor.

        If ``grad`` is None, ``self`` must be scalar and ``out.grad`` is
        seeded with 1.0. For non-scalar outputs, pass an explicit ``grad``
        of matching shape (the upstream Jacobian-vector seed).
        """
        if grad is None:
            if self.data.ndim != 0:
                raise RuntimeError(
                    "backward() with no argument requires a scalar output; "
                    f"got shape {self.shape}. Pass grad=... explicitly for non-scalars."
                )
            grad = np.array(1.0, dtype=np.float64)

        topo: list[Tensor] = []
        visited: set[int] = set()

        def build(node: Tensor) -> None:
            if id(node) in visited:
                return
            visited.add(id(node))
            for child in node._prev:
                build(child)
            topo.append(node)

        build(self)
        # Seed the output gradient (overwrite, don't accumulate, so backward
        # is a function of the seed alone).
        self.grad = grad.astype(np.float64, copy=True)
        for node in reversed(topo):
            node._backward()

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        grad_repr = "None" if self.grad is None else f"<{self.grad.shape}>"
        return f"Tensor(shape={self.shape}, op={self._op!r}, grad={grad_repr})"


def _no_op_backward() -> None:
    pass


def _broadcast_grad_to(
    g: np.ndarray,
    target_shape: tuple[int, ...],
    axis: int | tuple[int, ...] | None,
    keepdims: bool,
) -> np.ndarray:
    """Inverse of np.sum/np.mean shape-wise: broadcast ``g`` back to ``target_shape``.

    If the reduction kept dims (or reduced everything), ``g`` is already
    broadcastable. Otherwise we re-insert size-1 axes at each reduced
    position before broadcasting. Insertions are processed in increasing
    axis order so each insert grows the array's ndim by 1 and the next
    insert position remains valid.
    """
    if not keepdims and axis is not None:
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        axes = tuple(a % len(target_shape) for a in axes)
        for a in sorted(axes):
            g = np.expand_dims(g, axis=a)
    return np.broadcast_to(g, target_shape)
