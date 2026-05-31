"""Stacked decoder-only transformer for character-level language modeling.

Wraps N ``DecoderBlock``s into a full LM. All blocks share the same
norm-placement and residual policy, so the whole model is one point in the
gradient-flow ablation's 6-config grid.

    idx (B, T)                      integer context
    tok = E[idx]      (B, T, d)     token embedding (gather)
    pos = P[:T]       (T, d)        learned positional embedding (broadcast add)
    h   = tok + pos   (B, T, d)
    h   = block_i(h)  (B, T, d)     x N, causal self-attention + FFN
    h   = LN_f(h)     (B, T, d)     final norm (only if norm != "none")
    logits = h @ W_head + b_head    (B, T, V)

Positional embeddings are required because self-attention is permutation
-equivariant -- without them the model cannot tell position 0 from position 5.

The ``forward_with_taps`` variant returns the per-block output Tensors so the
instrumentation can read their ``.grad`` after backward: that per-depth
gradient norm is the central diagnostic of the whole project.
"""

from __future__ import annotations

import numpy as np

from jacobigrad.autograd.tensor import Tensor
from jacobigrad.nn.block import DecoderBlock
from jacobigrad.nn.layernorm import LayerNorm
from jacobigrad.nn.losses import cross_entropy


class CharTransformer:
    """Decoder-only char-level transformer LM trained via jacobigrad."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        embed_dim: int,
        head_dim: int,
        ff_dim: int,
        n_layers: int,
        rng: np.random.Generator,
        *,
        norm: str = "pre",
        residual: bool = True,
        eps: float = 1e-5,
        embed_init_scale: float = 0.02,
    ):
        self.V = vocab_size
        self.block_size = block_size
        self.d = embed_dim
        self.n_layers = n_layers
        self.norm = norm
        self.residual = residual

        # Token + learned positional embeddings (small Gaussian init).
        self.E = Tensor(rng.normal(size=(vocab_size, embed_dim)) * embed_init_scale)
        self.P = Tensor(rng.normal(size=(block_size, embed_dim)) * embed_init_scale)

        # Stack of decoder blocks, all sharing the config's norm/residual.
        self.blocks = [
            DecoderBlock(
                embed_dim, head_dim, ff_dim, rng,
                norm=norm, residual=residual, causal=True, eps=eps,
            )
            for _ in range(n_layers)
        ]

        # Final norm before the head (pre-norm convention). Skipped entirely
        # for the "none" config so it stays a genuine no-normalization model.
        self.ln_f = LayerNorm(embed_dim, eps=eps) if norm != "none" else None

        # LM head.
        self.W_head = Tensor(rng.normal(size=(embed_dim, vocab_size)) * np.sqrt(1.0 / embed_dim))
        self.b_head = Tensor(np.zeros(vocab_size))

    # ------------------------------------------------------------------
    # Parameters.
    # ------------------------------------------------------------------

    def parameters(self) -> list[Tensor]:
        params = [self.E, self.P]
        for blk in self.blocks:
            params += blk.parameters()
        if self.ln_f is not None:
            params += self.ln_f.parameters()
        params += [self.W_head, self.b_head]
        return params

    def num_parameters(self) -> int:
        return sum(p.size for p in self.parameters())

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    # ------------------------------------------------------------------
    # Forward.
    # ------------------------------------------------------------------

    def _embed(self, idx: np.ndarray) -> Tensor:
        T = idx.shape[1]
        if T > self.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.block_size}")
        tok = self.E[idx]                       # (B, T, d)
        pos = self.P[np.arange(T)]              # (T, d) -> broadcasts over batch
        return tok + pos

    def forward(self, idx: np.ndarray) -> Tensor:
        """``idx`` shape ``(B, T)``. Returns logits ``(B, T, V)``."""
        h = self._embed(idx)
        for blk in self.blocks:
            h = blk.forward(h)
        if self.ln_f is not None:
            h = self.ln_f.forward(h)
        return h @ self.W_head + self.b_head

    def forward_with_taps(self, idx: np.ndarray) -> tuple[Tensor, list[Tensor]]:
        """Like ``forward`` but also returns each block's output Tensor.

        The taps are the inputs to the gradient-flow instrumentation: after
        ``loss.backward()`` each tap's ``.grad`` holds dL/dh at that depth.
        """
        h = self._embed(idx)
        taps: list[Tensor] = []
        for blk in self.blocks:
            h = blk.forward(h)
            taps.append(h)
        h_final = h if self.ln_f is None else self.ln_f.forward(h)
        logits = h_final @ self.W_head + self.b_head
        return logits, taps

    # ------------------------------------------------------------------
    # Loss.
    # ------------------------------------------------------------------

    def loss(self, idx: np.ndarray, targets: np.ndarray) -> Tensor:
        """Mean next-token cross-entropy. ``idx``/``targets`` shape ``(B, T)``.

        cross_entropy is 2-D only, so flatten (B, T, V) -> (B*T, V) via the
        differentiable ``Tensor.reshape`` and targets (B, T) -> (B*T,).
        """
        B, T = idx.shape
        logits = self.forward(idx)
        return cross_entropy(logits.reshape(B * T, self.V), targets.reshape(B * T))

    # ------------------------------------------------------------------
    # Numpy-only evaluation (no autograd) over a set of batches.
    # ------------------------------------------------------------------

    def batch_ce(self, idx: np.ndarray, targets: np.ndarray) -> float:
        """Mean cross-entropy (nats) on one (idx, targets) batch, no autograd."""
        B, T = idx.shape
        logits = self.forward(idx).data.reshape(B * T, self.V)
        tgt = targets.reshape(B * T)
        m = logits.max(axis=-1, keepdims=True)
        log_p = (logits - m) - np.log(np.exp(logits - m).sum(axis=-1, keepdims=True))
        return float(-log_p[np.arange(B * T), tgt].mean())
