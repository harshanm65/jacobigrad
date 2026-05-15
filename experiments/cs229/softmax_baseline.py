"""Train softmax regression on Tiny Shakespeare and compare to the closed-form
bigram optimum.

Run from the repo root:  uv run python experiments/cs229/softmax_baseline.py

This is the gating Week 1 deliverable per PLAN.md: it produces the project's
first val cross-entropy / perplexity number on Tiny Shakespeare and validates
the data + eval pipeline before any autograd risk lands.

Three numbers are reported:
  1. Uniform baseline      — log V (the floor: random next-char predictor).
  2. Closed-form bigram    — empirical conditional with Laplace smoothing.
  3. Softmax-regression GD — the same model class fit by mini-batch SGD.

The model (W in R^(V x V), b in R^V; logits = W[i] + b for input i) is
exactly the bigram model in another guise, so (3) should converge close to
(2). The gap, if any, comes from finite training time and the absence of the
explicit Laplace smoothing in pure GD on the empirical loss.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jacobigrad.baselines.softmax_regression import (
    closed_form_bigram_logprobs,
    cross_entropy_under_logprobs,
    full_corpus_ce,
    loss_and_grads,
)
from jacobigrad.data import load_tinyshakespeare


def train_sgd(
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    vocab_size: int,
    *,
    lr: float = 0.5,
    batch_size: int = 4096,
    steps: int = 3000,
    eval_every: int = 200,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rng = np.random.default_rng(seed)
    W = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    b = np.zeros(vocab_size, dtype=np.float64)
    history: list[dict] = []

    for step in range(1, steps + 1):
        idx = rng.integers(0, len(train_ids) - 1, size=batch_size)
        x = train_ids[idx]
        y = train_ids[idx + 1]
        loss, dW, db = loss_and_grads(W, b, x, y)
        W -= lr * dW
        b -= lr * db
        if step == 1 or step % eval_every == 0:
            train_ce = full_corpus_ce(W, b, train_ids)
            val_ce = full_corpus_ce(W, b, val_ids)
            history.append(
                {"step": step, "batch_loss": loss, "train_ce": train_ce, "val_ce": val_ce}
            )
            print(
                f"  step {step:5d} | batch_loss {loss:.4f} "
                f"| train_ce {train_ce:.4f} | val_ce {val_ce:.4f}"
            )
    return W, b, history


def main() -> None:
    print("Loading Tiny Shakespeare ...")
    train_ids, val_ids, tok = load_tinyshakespeare()
    V = tok.vocab_size
    print(
        f"  vocab_size={V}, train_chars={len(train_ids):,}, val_chars={len(val_ids):,}"
    )

    print("\n[1] Uniform baseline (random next-char prediction)")
    uniform_ce = float(np.log(V))
    print(
        f"  CE={uniform_ce:.4f} nats  |  bpc={uniform_ce / np.log(2):.4f}  "
        f"|  perplexity={np.exp(uniform_ce):.2f}"
    )

    print("\n[2] Closed-form bigram (empirical conditional, Laplace alpha=1.0)")
    alpha = 1.0
    log_probs = closed_form_bigram_logprobs(train_ids, V, alpha=alpha)
    bigram_train_ce = cross_entropy_under_logprobs(log_probs, train_ids[:-1], train_ids[1:])
    bigram_val_ce = cross_entropy_under_logprobs(log_probs, val_ids[:-1], val_ids[1:])
    print(
        f"  train_CE={bigram_train_ce:.4f} nats  |  val_CE={bigram_val_ce:.4f} nats"
    )
    print(
        f"  val bpc={bigram_val_ce / np.log(2):.4f}  "
        f"|  val perplexity={np.exp(bigram_val_ce):.2f}"
    )

    print("\n[3] Softmax regression by mini-batch SGD")
    lr, batch_size, steps, seed = 0.5, 4096, 3000, 0
    print(f"  lr={lr}, batch_size={batch_size}, steps={steps}, seed={seed}")
    W, b, history = train_sgd(
        train_ids, val_ids, V,
        lr=lr, batch_size=batch_size, steps=steps, eval_every=200, seed=seed,
    )
    sr_train_ce = full_corpus_ce(W, b, train_ids)
    sr_val_ce = full_corpus_ce(W, b, val_ids)
    print(
        f"  final train_CE={sr_train_ce:.4f}  |  val_CE={sr_val_ce:.4f}  "
        f"|  val bpc={sr_val_ce / np.log(2):.4f}  |  val ppl={np.exp(sr_val_ce):.2f}"
    )
    print(f"  gap to closed-form bigram (val): {sr_val_ce - bigram_val_ce:+.4f} nats")

    results = {
        "vocab_size": V,
        "train_chars": int(len(train_ids)),
        "val_chars": int(len(val_ids)),
        "uniform": {
            "ce_nats": uniform_ce,
            "bits_per_char": uniform_ce / float(np.log(2)),
            "perplexity": float(np.exp(uniform_ce)),
        },
        "bigram_closed_form": {
            "alpha": alpha,
            "train_ce_nats": bigram_train_ce,
            "val_ce_nats": bigram_val_ce,
            "val_bits_per_char": bigram_val_ce / float(np.log(2)),
            "val_perplexity": float(np.exp(bigram_val_ce)),
        },
        "softmax_regression_sgd": {
            "lr": lr,
            "batch_size": batch_size,
            "steps": steps,
            "seed": seed,
            "train_ce_nats": sr_train_ce,
            "val_ce_nats": sr_val_ce,
            "val_bits_per_char": sr_val_ce / float(np.log(2)),
            "val_perplexity": float(np.exp(sr_val_ce)),
            "gap_to_bigram_val_nats": sr_val_ce - bigram_val_ce,
            "history": history,
        },
    }

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "softmax_baseline.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults -> {out_path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
