"""Train a Bengio-style char-MLP on Tiny Shakespeare via our own autograd.

Run from the repo root:  uv run python experiments/cs229/mlp_baseline.py

This is the headline result for the milestone report. The model is trained
end-to-end with ``jacobigrad`` (no torch in the loop) and reports val CE
against the bigram closed-form floor. The story we want to tell:

    uniform           4.17 nats   (random next-char baseline)
    bigram (closed)   2.48 nats   (best possible 2-gram given train counts)
    softmax-reg SGD   2.67 nats   (the same 2-gram class fit by GD)
    MLP via ours      < 2.48 nats (broader context — must beat bigram)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from jacobigrad.data import get_batch, load_tinyshakespeare
from jacobigrad.nn import CharMLP


def train(
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    vocab_size: int,
    *,
    context_size: int = 3,
    embed_dim: int = 16,
    hidden_dim: int = 64,
    lr: float = 0.1,
    batch_size: int = 256,
    steps: int = 4000,
    eval_every: int = 200,
    seed: int = 0,
) -> tuple[CharMLP, list[dict]]:
    rng = np.random.default_rng(seed)
    mlp = CharMLP(vocab_size, context_size, embed_dim, hidden_dim, rng)
    print(f"  parameters: {mlp.num_parameters():,}")

    history: list[dict] = []
    t0 = time.time()
    for step in range(1, steps + 1):
        # get_batch returns x, y of shape (B, T) where y is x shifted by 1.
        # For an MLP that predicts a single next-char from a context, take
        # the full window as context and y[:, -1] as the single target.
        x, y_full = get_batch(train_ids, batch_size, context_size, rng)
        y = y_full[:, -1]

        mlp.zero_grad()
        loss = mlp.loss(x, y)
        loss.backward()
        mlp.sgd_step(lr)

        if step == 1 or step % eval_every == 0:
            train_ce = mlp.evaluate_corpus_ce(train_ids)
            val_ce = mlp.evaluate_corpus_ce(val_ids)
            history.append(
                {
                    "step": step,
                    "batch_loss": float(loss.data),
                    "train_ce": train_ce,
                    "val_ce": val_ce,
                }
            )
            elapsed = time.time() - t0
            print(
                f"  step {step:5d} | batch_loss {float(loss.data):.4f} "
                f"| train_ce {train_ce:.4f} | val_ce {val_ce:.4f} "
                f"| elapsed {elapsed:.1f}s"
            )
    return mlp, history


def main() -> None:
    print("Loading Tiny Shakespeare ...")
    train_ids, val_ids, tok = load_tinyshakespeare()
    V = tok.vocab_size
    print(f"  vocab_size={V}, train_chars={len(train_ids):,}, val_chars={len(val_ids):,}")

    bigram_val_ce_nats = 2.4819  # from softmax_baseline.json (alpha=1.0)

    print("\nTraining char-MLP via jacobigrad autograd ...")
    config = dict(
        context_size=3, embed_dim=16, hidden_dim=64,
        lr=0.1, batch_size=256, steps=4000, eval_every=200, seed=0,
    )
    print(f"  config: {config}")
    mlp, history = train(train_ids, val_ids, V, **config)

    final_train_ce = mlp.evaluate_corpus_ce(train_ids)
    final_val_ce = mlp.evaluate_corpus_ce(val_ids)
    gap = final_val_ce - bigram_val_ce_nats
    print(
        f"\nFinal:  train_CE={final_train_ce:.4f}  val_CE={final_val_ce:.4f}  "
        f"val bpc={final_val_ce / np.log(2):.4f}  "
        f"val ppl={np.exp(final_val_ce):.2f}"
    )
    print(
        f"Gap to closed-form bigram floor (val): {gap:+.4f} nats  "
        f"(< 0 means MLP beats bigram)"
    )

    results = {
        "model": "char_mlp_via_jacobigrad",
        "vocab_size": V,
        "train_chars": int(len(train_ids)),
        "val_chars": int(len(val_ids)),
        "config": config,
        "num_parameters": mlp.num_parameters(),
        "final": {
            "train_ce_nats": final_train_ce,
            "val_ce_nats": final_val_ce,
            "val_bits_per_char": final_val_ce / float(np.log(2)),
            "val_perplexity": float(np.exp(final_val_ce)),
        },
        "bigram_floor_val_ce_nats": bigram_val_ce_nats,
        "gap_to_bigram_val_nats": gap,
        "history": history,
    }

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mlp_baseline.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults -> {out_path.relative_to(Path.cwd())}")


if __name__ == "__main__":
    main()
