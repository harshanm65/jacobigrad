"""Bengio-style character-level MLP language model.

Architecture (Bengio et al. 2003 "A Neural Probabilistic Language Model",
adapted to char-level):

    idx  (B, T)              integer context — last T characters
    E[idx]  (B, T, D)        embedding lookup
    flat  (B, T*D)           concatenate the T context embeddings
    h = tanh(flat @ W1 + b1) (B, H)
    logits = h @ W2 + b2     (B, V)

Loss is fused cross-entropy on the next-char target. Trained end-to-end via
``jacobigrad`` reverse-mode autograd — no torch in the training loop.
"""

from __future__ import annotations

import numpy as np

from jacobigrad.autograd.tensor import Tensor
from jacobigrad.nn.losses import cross_entropy


class CharMLP:
    """Char-level MLP language model trained via jacobigrad."""

    def __init__(
        self,
        vocab_size: int,
        context_size: int,
        embed_dim: int,
        hidden_dim: int,
        rng: np.random.Generator,
        *,
        embed_init_scale: float = 0.01,
    ):
        self.V = vocab_size
        self.T = context_size
        self.D = embed_dim
        self.H = hidden_dim

        # Initialization:
        #   - Embedding: small Gaussian (we want the model to learn meaningful
        #     directions; large init dominates the linear layers).
        #   - Linear weights: Xavier-style sqrt(1/fan_in) — keeps the variance
        #     of pre-activations roughly constant through the tanh layer.
        self.E = Tensor(rng.normal(size=(vocab_size, embed_dim)) * embed_init_scale)
        fan_in_1 = context_size * embed_dim
        self.W1 = Tensor(rng.normal(size=(fan_in_1, hidden_dim)) * np.sqrt(1.0 / fan_in_1))
        self.b1 = Tensor(np.zeros(hidden_dim))
        self.W2 = Tensor(rng.normal(size=(hidden_dim, vocab_size)) * np.sqrt(1.0 / hidden_dim))
        self.b2 = Tensor(np.zeros(vocab_size))

    def parameters(self) -> list[Tensor]:
        return [self.E, self.W1, self.b1, self.W2, self.b2]

    def num_parameters(self) -> int:
        return sum(p.size for p in self.parameters())

    def forward(self, idx: np.ndarray) -> Tensor:
        """``idx`` shape ``(B, T)`` integer array. Returns logits ``(B, V)``."""
        B = idx.shape[0]
        emb = self.E[idx]
        flat = emb.reshape(B, self.T * self.D)
        h = (flat @ self.W1 + self.b1).tanh()
        return h @ self.W2 + self.b2

    def loss(self, idx: np.ndarray, targets: np.ndarray) -> Tensor:
        return cross_entropy(self.forward(idx), targets)

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def sgd_step(self, lr: float) -> None:
        for p in self.parameters():
            if p.grad is not None:
                p.data -= lr * p.grad

    # ------------------------------------------------------------------
    # Numpy-only evaluation. No autograd graph — much faster for the
    # full-corpus CE pass that runs every eval interval.
    # ------------------------------------------------------------------

    def evaluate_corpus_ce(self, ids: np.ndarray, *, chunk: int = 8192) -> float:
        """Mean cross-entropy (nats) over every (context, target) in ``ids``.

        Uses raw numpy for the forward — autograd is unnecessary for eval.
        """
        E, W1, b1, W2, b2 = (p.data for p in self.parameters())
        n_pairs = len(ids) - self.T
        if n_pairs <= 0:
            raise ValueError(f"ids too short for context_size={self.T}")

        offsets = np.arange(self.T)
        total = 0.0
        for s in range(0, n_pairs, chunk):
            e = min(s + chunk, n_pairs)
            # Vectorized context construction: for each position s..e-1, take
            # the T chars starting at that position.
            positions = np.arange(s, e)[:, None] + offsets[None, :]
            contexts = ids[positions]
            targets = ids[s + self.T : e + self.T]

            emb = E[contexts]
            flat = emb.reshape(e - s, -1)
            h = np.tanh(flat @ W1 + b1)
            logits = h @ W2 + b2

            m = logits.max(axis=-1, keepdims=True)
            log_p = (logits - m) - np.log(np.exp(logits - m).sum(axis=-1, keepdims=True))
            total += float(-log_p[np.arange(e - s), targets].sum())
        return total / n_pairs
