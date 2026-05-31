import numpy as np
import pytest

from jacobigrad.baselines.softmax_regression import (
    closed_form_bigram_logprobs,
    cross_entropy_under_logprobs,
    full_corpus_ce,
    log_softmax,
    loss_and_grads,
)


def test_log_softmax_rows_sum_to_one():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(5, 7))
    p = np.exp(log_softmax(logits, axis=-1))
    np.testing.assert_allclose(p.sum(axis=-1), np.ones(5), atol=1e-12)


def test_log_softmax_stable_for_large_logits():
    # exp(1000) overflows; the log-sum-exp shift must keep this finite.
    log_p = log_softmax(np.array([[1000.0, 1001.0, 1002.0]]))
    assert np.all(np.isfinite(log_p))
    np.testing.assert_allclose(np.exp(log_p).sum(), 1.0, atol=1e-12)


def test_closed_form_bigram_normalized():
    ids = np.array([0, 1, 0, 1, 0, 1, 0], dtype=np.int64)  # "ababab" + 'a'
    log_p = closed_form_bigram_logprobs(ids, vocab_size=2, alpha=1e-6)
    p = np.exp(log_p)
    np.testing.assert_allclose(p.sum(axis=1), np.ones(2), atol=1e-12)
    # In the alpha -> 0 limit, P(b|a) -> 1 and P(a|b) -> 1 on this corpus.
    assert log_p[0, 1] > log_p[0, 0]
    assert log_p[1, 0] > log_p[1, 1]


def test_closed_form_bigram_smoothing_handles_unseen_pairs():
    ids = np.array([0, 1], dtype=np.int64)  # only (0, 1) appears
    log_p = closed_form_bigram_logprobs(ids, vocab_size=2, alpha=1.0)
    assert np.isfinite(log_p[0, 0])  # would be -inf without smoothing
    assert log_p[0, 1] > log_p[0, 0]


def test_closed_form_bigram_alpha_must_be_positive():
    ids = np.array([0, 1, 0, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="alpha must be > 0"):
        closed_form_bigram_logprobs(ids, vocab_size=2, alpha=0.0)


def test_cross_entropy_matches_manual():
    log_p = np.log(np.array([[0.7, 0.3], [0.2, 0.8]]))
    x = np.array([0, 1])
    y = np.array([1, 0])
    # -mean(log 0.3, log 0.2)
    expected = -float(np.mean([np.log(0.3), np.log(0.2)]))
    assert abs(cross_entropy_under_logprobs(log_p, x, y) - expected) < 1e-12


def test_loss_and_grads_match_finite_differences():
    """Hand-derived dW, db must agree with central-difference estimate."""
    rng = np.random.default_rng(0)
    V, B = 4, 16
    W = rng.normal(size=(V, V))
    b = rng.normal(size=V)
    x_ids = rng.integers(0, V, size=B)
    y_ids = rng.integers(0, V, size=B)

    _, dW, db = loss_and_grads(W, b, x_ids, y_ids)

    eps = 1e-5
    for i in range(V):
        for j in range(V):
            Wp = W.copy()
            Wp[i, j] += eps
            Wm = W.copy()
            Wm[i, j] -= eps
            lp, _, _ = loss_and_grads(Wp, b, x_ids, y_ids)
            lm, _, _ = loss_and_grads(Wm, b, x_ids, y_ids)
            num = (lp - lm) / (2 * eps)
            assert abs(num - dW[i, j]) < 1e-7, (
                f"dW[{i},{j}]: analytic={dW[i, j]:.3e}, finite-diff={num:.3e}"
            )

    for j in range(V):
        bp = b.copy()
        bp[j] += eps
        bm = b.copy()
        bm[j] -= eps
        lp, _, _ = loss_and_grads(W, bp, x_ids, y_ids)
        lm, _, _ = loss_and_grads(W, bm, x_ids, y_ids)
        num = (lp - lm) / (2 * eps)
        assert abs(num - db[j]) < 1e-7


def test_full_corpus_ce_matches_per_pair_average():
    rng = np.random.default_rng(0)
    V = 5
    ids = rng.integers(0, V, size=100).astype(np.int64)
    W = rng.normal(size=(V, V))
    b = rng.normal(size=V)

    fast = full_corpus_ce(W, b, ids)

    log_p = log_softmax(W[ids[:-1]] + b, axis=-1)
    manual = float(-log_p[np.arange(len(ids) - 1), ids[1:]].mean())
    assert abs(fast - manual) < 1e-12


def test_one_gd_step_decreases_loss():
    rng = np.random.default_rng(0)
    V, B = 4, 32
    W = rng.normal(size=(V, V)) * 0.1
    b = rng.normal(size=V) * 0.1
    x = rng.integers(0, V, size=B)
    y = rng.integers(0, V, size=B)

    loss_before, dW, db = loss_and_grads(W, b, x, y)
    loss_after, _, _ = loss_and_grads(W - 0.1 * dW, b - 0.1 * db, x, y)
    assert loss_after < loss_before
