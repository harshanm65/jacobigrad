"""Loss functions over ``Tensor``s."""

from __future__ import annotations

import numpy as np

from jacobigrad.autograd.tensor import Tensor


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Mean cross-entropy of ``logits`` against integer class ``targets``.

    Shapes:
        logits  : (B, V)
        targets : (B,) integer class IDs in [0, V)

    Implemented as a fused op rather than ``softmax + log + gather``: the
    closed-form gradient of softmax + cross-entropy is

        dL/dlogits = (softmax(logits) - onehot(targets)) / B

    which is much cleaner numerically and skips building an intermediate
    one-hot tensor in the graph.

    Forward also uses the log-sum-exp shift for numerical stability, so the
    computation is well-defined for logits at arbitrary scale.
    """
    if logits.ndim != 2:
        raise ValueError(f"cross_entropy expects 2-D logits (B, V); got {logits.shape}")
    if not isinstance(targets, np.ndarray) or targets.dtype.kind not in "iu":
        raise TypeError("targets must be an integer ndarray of class IDs")
    B, V = logits.shape
    if targets.shape != (B,):
        raise ValueError(f"targets shape {targets.shape} does not match logits batch {B}")

    # Forward: numerically stable log-softmax, then NLL on the targets.
    m = logits.data.max(axis=-1, keepdims=True)
    shifted = logits.data - m
    log_z = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    log_p = shifted - log_z
    nll = -log_p[np.arange(B), targets]
    loss_val = nll.mean()

    out = Tensor(np.array(loss_val), (logits,), "cross_entropy")

    def _backward() -> None:
        # dL/dlogits = (p - onehot(targets)) / B, scaled by upstream out.grad.
        p = np.exp(log_p)
        p[np.arange(B), targets] -= 1.0
        p /= B
        logits._add_grad(p * out.grad)

    out._backward = _backward
    return out
