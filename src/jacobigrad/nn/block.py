"""Transformer decoder block with toggleable normalization and residuals.

This is the experimental apparatus for the gradient-flow ablation. A single
block has two sublayers:

    1. causal single-head self-attention, projected back to ``embed_dim``;
    2. a position-wise feed-forward network (Linear -> ReLU -> Linear).

Each sublayer ``f`` is wrapped by a normalization/residual *policy* governed
by two switches, applied identically to both sublayers so that the six
configurations differ ONLY in these toggles:

    norm placement in {"pre", "post", "none"}:
        pre :   x + f(LN(x))            (norm before sublayer; "pre-LN")
        post:   LN(x + f(x))           (norm after residual add; "post-LN")
        none:   x + f(x)               (no normalization at all)

    residual in {True, False}:
        True  : keep the ``x +`` skip connection
        False : drop it (so "pre"/"none" reduce to f(LN(x)) / f(x), and
                "post" becomes LN(f(x)))

The whole point of the project is to measure how these choices govern
gradient flow through depth, so the block keeps everything else fixed and
exposes exactly these two knobs.
"""

from __future__ import annotations

import numpy as np

from jacobigrad.autograd.tensor import Tensor
from jacobigrad.nn.attention import SingleHeadAttention
from jacobigrad.nn.layernorm import LayerNorm

NORM_PLACEMENTS = ("pre", "post", "none")


class DecoderBlock:
    """A single decoder block with norm-placement and residual toggles."""

    def __init__(
        self,
        embed_dim: int,
        head_dim: int,
        ff_dim: int,
        rng: np.random.Generator,
        *,
        norm: str = "pre",
        residual: bool = True,
        causal: bool = True,
        eps: float = 1e-5,
    ):
        if norm not in NORM_PLACEMENTS:
            raise ValueError(f"norm must be one of {NORM_PLACEMENTS}; got {norm!r}")
        self.d = embed_dim
        self.norm = norm
        self.residual = residual

        # --- Sublayer 1: self-attention + output projection back to embed_dim.
        self.attn = SingleHeadAttention(embed_dim, head_dim, rng, causal=causal)
        self.W_O = Tensor(rng.normal(size=(head_dim, embed_dim)) * np.sqrt(1.0 / head_dim))

        # --- Sublayer 2: position-wise FFN (Linear -> ReLU -> Linear).
        self.W1 = Tensor(rng.normal(size=(embed_dim, ff_dim)) * np.sqrt(1.0 / embed_dim))
        self.b1 = Tensor(np.zeros(ff_dim))
        self.W2 = Tensor(rng.normal(size=(ff_dim, embed_dim)) * np.sqrt(1.0 / ff_dim))
        self.b2 = Tensor(np.zeros(embed_dim))

        # --- One LayerNorm per sublayer (unused weights are harmless when
        #     norm == "none"; we simply never call them).
        self.ln1 = LayerNorm(embed_dim, eps=eps)
        self.ln2 = LayerNorm(embed_dim, eps=eps)

    # ------------------------------------------------------------------
    # Parameters.
    # ------------------------------------------------------------------

    def parameters(self) -> list[Tensor]:
        params = [*self.attn.parameters(), self.W_O, self.W1, self.b1, self.W2, self.b2]
        if self.norm != "none":
            params += [*self.ln1.parameters(), *self.ln2.parameters()]
        return params

    def num_parameters(self) -> int:
        return sum(p.size for p in self.parameters())

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    # ------------------------------------------------------------------
    # The two sublayer transforms (before any norm/residual wrapping).
    # ------------------------------------------------------------------

    def _attn_sublayer(self, x: Tensor) -> Tensor:
        return self.attn.forward(x) @ self.W_O

    def _ffn_sublayer(self, x: Tensor) -> Tensor:
        return (x @ self.W1 + self.b1).relu() @ self.W2 + self.b2

    # ------------------------------------------------------------------
    # The norm/residual policy, applied identically to both sublayers.
    # ------------------------------------------------------------------

    def _apply(self, x: Tensor, sublayer, ln: LayerNorm) -> Tensor:
        """Wrap ``sublayer`` around ``x`` per the (norm, residual) policy."""
        if self.norm == "pre":
            out = sublayer(ln.forward(x))
            return x + out if self.residual else out
        if self.norm == "post":
            out = sublayer(x)
            return ln.forward(x + out) if self.residual else ln.forward(out)
        # norm == "none"
        out = sublayer(x)
        return x + out if self.residual else out

    def forward(self, x: Tensor | np.ndarray) -> Tensor:
        """``x`` shape ``(B, T, embed_dim)``; returns the same shape."""
        x = x if isinstance(x, Tensor) else Tensor(x)
        x = self._apply(x, self._attn_sublayer, self.ln1)
        x = self._apply(x, self._ffn_sublayer, self.ln2)
        return x

    __call__ = forward
