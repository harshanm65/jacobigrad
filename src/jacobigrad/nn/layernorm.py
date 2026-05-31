"""Layer normalization, in the jacobigrad engine.

LayerNorm normalizes each token's feature vector to zero mean and unit
variance, then applies a learned per-feature affine (gamma, beta):

    mu    = mean(x)                 over the feature axis
    var   = mean((x - mu)^2)        biased (population) variance
    xhat  = (x - mu) / sqrt(var + eps)
    y     = gamma * xhat + beta

Design choice (see writeups/cs229/layernorm_backward.md): rather than
hand-write the backward, we **compose it from Tensor primitives** (mean,
subtract, power, multiply, divide) and let reverse-mode autograd produce the
gradient. The point of the engine is exactly this — once mean/var/affine are
expressed as differentiable ops, a non-trivial normalization layer needs no
bespoke backward. The closed-form gradient is still derived in the writeup
and pinned down by a dedicated parity test, so we get both the convenience
and the understanding.

The variance is the *biased* estimator (divide by N, not N-1) to match
``torch.nn.LayerNorm``.
"""

from __future__ import annotations

import numpy as np

from jacobigrad.autograd.tensor import Tensor


class LayerNorm:
    """Layer normalization over the last (feature) axis, with affine params."""

    def __init__(self, normalized_dim: int, *, eps: float = 1e-5):
        self.dim = normalized_dim
        self.eps = eps
        # Standard init: gamma = 1, beta = 0 -> the layer starts as a pure
        # normalizer and learns to scale/shift away from it.
        self.gamma = Tensor(np.ones(normalized_dim))
        self.beta = Tensor(np.zeros(normalized_dim))

    def parameters(self) -> list[Tensor]:
        return [self.gamma, self.beta]

    def num_parameters(self) -> int:
        return sum(p.size for p in self.parameters())

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def forward(self, x: Tensor | np.ndarray) -> Tensor:
        """Normalize over the last axis. ``x`` shape ``(..., dim)``."""
        x = x if isinstance(x, Tensor) else Tensor(x)

        mu = x.mean(axis=-1, keepdims=True)
        centered = x - mu
        var = (centered ** 2).mean(axis=-1, keepdims=True)
        # rstd = (var + eps) ** -0.5. Composed entirely from autograd ops, so
        # the backward through the mean/variance path is built automatically.
        rstd = (var + self.eps) ** -0.5
        xhat = centered * rstd
        return xhat * self.gamma + self.beta

    __call__ = forward
