"""Single-head scaled dot-product attention, in the jacobigrad engine.

This is the central new layer for the gradient-flow ablation. The forward
mirrors the standard scaled dot-product attention and matches Luis's
PyTorch reference (``tests/reference_attention.py``) op-for-op, so his
closed-form backward serves as an independent oracle for ours:

    Q = X @ W_Q          (B, T, d_k)
    K = X @ W_K
    V = X @ W_V
    S = Q @ K^T / sqrt(d_k)        (B, T, T)
    S = S + mask                   (optional additive causal mask)
    A = softmax(S, axis=-1)        attention weights
    Y = A @ V                      (B, T, d_k)

Every operation here is a ``Tensor`` op with its own ``_backward`` closure
(see ``autograd/tensor.py``) — there is no torch in this module, and no
hand-written backward. The derivation Luis wrote (``attention_backward.md``)
is realized implicitly by composing ``@``, ``transpose``, ``softmax``, and
the scale: in particular the softmax-row Jacobian collapse lives in
``Tensor.softmax`` and the projection fan-in lives in ``Tensor.__matmul__``.

Scope: single head, additive causal masking. Multi-head is explicitly out
of scope per PLAN.md.
"""

from __future__ import annotations

import numpy as np

from jacobigrad.autograd.tensor import Tensor


def causal_mask(T: int) -> np.ndarray:
    """Additive (T, T) mask: 0 on/below the diagonal, -inf strictly above.

    Added to the pre-softmax scores so future positions get probability 0
    after softmax (and, since their attention weight is exactly 0, no
    gradient flows through them). Same construction as the torch reference.
    """
    mask = np.zeros((T, T), dtype=np.float64)
    upper = np.triu(np.ones((T, T), dtype=bool), k=1)
    mask[upper] = -np.inf
    return mask


class SingleHeadAttention:
    """Single-head scaled dot-product self-attention trained via jacobigrad."""

    def __init__(
        self,
        embed_dim: int,
        head_dim: int,
        rng: np.random.Generator,
        *,
        causal: bool = False,
        init_scale: float | None = None,
    ):
        self.d = embed_dim
        self.d_k = head_dim
        self.causal = causal
        # Xavier-style init (sqrt(1/fan_in)) so the projected Q/K/V keep
        # roughly unit variance — same rationale as the MLP's linear layers.
        scale = init_scale if init_scale is not None else np.sqrt(1.0 / embed_dim)
        self.W_Q = Tensor(rng.normal(size=(embed_dim, head_dim)) * scale)
        self.W_K = Tensor(rng.normal(size=(embed_dim, head_dim)) * scale)
        self.W_V = Tensor(rng.normal(size=(embed_dim, head_dim)) * scale)

    def parameters(self) -> list[Tensor]:
        return [self.W_Q, self.W_K, self.W_V]

    def num_parameters(self) -> int:
        return sum(p.size for p in self.parameters())

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def forward(self, X: Tensor | np.ndarray) -> Tensor:
        """``X`` shape ``(B, T, d)``. Returns attention output ``(B, T, d_k)``."""
        X = X if isinstance(X, Tensor) else Tensor(X)
        T = X.shape[-2]

        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V

        # Scaled scores. transpose() swaps the last two axes -> K^T per batch.
        scores = (Q @ K.transpose(-2, -1)) * (1.0 / np.sqrt(self.d_k))
        if self.causal:
            scores = scores + causal_mask(T)

        attn = scores.softmax(axis=-1)
        return attn @ V

    __call__ = forward
