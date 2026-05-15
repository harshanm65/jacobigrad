"""Softmax-regression baseline for character-level next-char prediction.

The model is structurally a bigram: parameters W in R^(V x V) and b in R^V;
for an input character ID i, the predicted next-char distribution is
softmax(W[i, :] + b). With cross-entropy loss the optimum is the empirical
bigram conditional, which we exploit as a closed-form sanity check on the
gradient-descent trajectory.

All math is hand-derived; this module deliberately does not depend on any
autograd. The two key gradient identities used here:

  d L / d logits_t = softmax(logits_t) - onehot(y_t)        (per example)
  d L / d W[i, :] = sum_{t : x_t = i} d L / d logits_t      (input is one-hot)

Both fall out of softmax + cross-entropy in closed form; see e.g.
Bishop PRML Ch. 4.3.4 or any softmax-regression derivation.
"""

from __future__ import annotations

import numpy as np


def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable log-softmax via the log-sum-exp trick.

    Subtracting the per-row max keeps exp() inputs <= 0, avoiding overflow.
    """
    m = logits.max(axis=axis, keepdims=True)
    shifted = logits - m
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def closed_form_bigram_logprobs(
    train_ids: np.ndarray, vocab_size: int, alpha: float = 1.0
) -> np.ndarray:
    """Empirical conditional log P(j | i) on `train_ids`, with add-alpha smoothing.

    Returns a (V, V) array of log probabilities. alpha must be > 0 so that
    pairs absent from train still receive positive mass (otherwise val CE
    blows up to +inf on any unseen bigram).
    """
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0 (else log 0 on unseen pairs); got {alpha}")
    counts = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    np.add.at(counts, (train_ids[:-1], train_ids[1:]), 1.0)
    counts += alpha
    return np.log(counts) - np.log(counts.sum(axis=1, keepdims=True))


def cross_entropy_under_logprobs(
    log_probs: np.ndarray, x_ids: np.ndarray, y_ids: np.ndarray
) -> float:
    """Mean CE (nats) of a model with precomputed log P(j|i) on (x, y) pairs."""
    return float(-np.mean(log_probs[x_ids, y_ids]))


def loss_and_grads(
    W: np.ndarray, b: np.ndarray, x_ids: np.ndarray, y_ids: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Mean cross-entropy loss + analytic gradients (dW, db).

    Shapes: W (V, V), b (V,), x_ids (B,), y_ids (B,). Returns scalar loss
    plus arrays matching W and b.
    """
    B = x_ids.shape[0]

    logits = W[x_ids] + b
    log_p = log_softmax(logits, axis=-1)
    loss = float(-log_p[np.arange(B), y_ids].mean())

    # dL/dlogits_t = (softmax_t - onehot(y_t)) / B  (the /B comes from the mean)
    dlogits = np.exp(log_p)
    dlogits[np.arange(B), y_ids] -= 1.0
    dlogits /= B

    # Input is one-hot, so each example's contribution lands on a single row of W.
    # Multiple examples may share the same input row → np.add.at scatters them.
    dW = np.zeros_like(W)
    np.add.at(dW, x_ids, dlogits)

    db = dlogits.sum(axis=0)
    return loss, dW, db


def full_corpus_ce(
    W: np.ndarray, b: np.ndarray, ids: np.ndarray, *, chunk: int = 8192
) -> float:
    """Mean CE (nats) over every consecutive (ids[t], ids[t+1]) bigram pair.

    Chunks the (V,) embedding lookups so memory stays bounded for long ids.
    """
    n_pairs = len(ids) - 1
    total = 0.0
    for s in range(0, n_pairs, chunk):
        e = min(s + chunk, n_pairs)
        log_p = log_softmax(W[ids[s:e]] + b, axis=-1)
        total += float(-log_p[np.arange(e - s), ids[s + 1 : e + 1]].sum())
    return total / n_pairs
