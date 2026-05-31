"""Optimizers over ``Tensor`` parameters.

The engine's parameter convention is a plain ``list[Tensor]`` where each
parameter exposes ``.data`` (the numpy array) and ``.grad`` (its gradient,
``None`` until backward populates it) -- see ``CharMLP.sgd_step`` for the
minimal in-place SGD pattern this generalizes.

AdamW is the relevant optimizer for the gradient-flow ablation: post-LN and
no-norm configs are exactly the unstable training regimes, and Adam's
per-parameter adaptive step keeps the comparison across configs fair where
plain SGD would simply fail to train the badly-conditioned ones.
"""

from __future__ import annotations

import numpy as np

from jacobigrad.autograd.tensor import Tensor


class AdamW:
    """AdamW (Adam with decoupled weight decay), from scratch.

    Update per step ``t`` (Loshchilov & Hutter 2019):

        m = beta1*m + (1-beta1)*g
        v = beta2*v + (1-beta2)*g^2
        m_hat = m / (1 - beta1^t)          # bias correction
        v_hat = v / (1 - beta2^t)
        p -= lr * ( m_hat / (sqrt(v_hat) + eps) + weight_decay * p )

    The weight decay is *decoupled* -- applied directly to the parameter
    rather than folded into the gradient (the "W" in AdamW). Moment buffers
    are keyed by ``id(param)`` so the optimizer is stateless w.r.t. parameter
    ordering and safe if the same list is passed back each step.
    """

    def __init__(
        self,
        params: list[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        # Per-parameter first/second moment buffers, lazily zero-initialized.
        self._m: dict[int, np.ndarray] = {}
        self._v: dict[int, np.ndarray] = {}

    def zero_grad(self) -> None:
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        self.t += 1
        bias1 = 1.0 - self.beta1 ** self.t
        bias2 = 1.0 - self.beta2 ** self.t

        for p in self.params:
            if p.grad is None:
                continue
            key = id(p)
            if key not in self._m:
                self._m[key] = np.zeros_like(p.data)
                self._v[key] = np.zeros_like(p.data)

            g = p.grad
            m = self._m[key]
            v = self._v[key]
            # In-place moment updates: m = b1*m + (1-b1)*g ; v = b2*v + (1-b2)*g^2
            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * (g * g)

            m_hat = m / bias1
            v_hat = v / bias2

            # Decoupled weight decay: applied to the parameter, not the grad.
            p.data -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * p.data)
