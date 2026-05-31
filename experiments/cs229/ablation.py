"""Gradient-flow ablation: LayerNorm placement x residual, on Tiny Shakespeare.

Run from the repo root:  uv run python experiments/cs229/ablation.py

The project's central experiment. Trains the same small decoder-only
transformer under all six configs (norm in {pre, post, none} x residual in
{on, off}) with everything else held fixed -- same seed, same data draws, same
AdamW schedule, same step budget -- so the ONLY variable is the architectural
toggle. At each eval interval we record:

  - a sampled validation cross-entropy (over a FIXED set of val batches, the
    same batches for every config -> a fair cross-config comparison), and
  - a gradient-flow snapshot (per-layer dL/dh norms etc., see instrument.py).

Note on the val-CE metric: this is a *sampled* estimate over fixed val
batches, NOT the exact full-corpus CE that the baseline scripts report. It is
fair for comparing configs to each other, but is not directly comparable to
the bigram/MLP baseline table, and is flagged as such in the output JSON.

Outputs one JSON per config to results/ablation/{norm}_{res}.json plus a
combined results/ablation/summary.json.
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

from jacobigrad.data import get_batch, load_tinyshakespeare
from jacobigrad.nn import NORM_PLACEMENTS, CharTransformer, cross_entropy
from jacobigrad.optim import AdamW

# instrument.py is a sibling experiment module, not part of the package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from instrument import (  # noqa: E402
    grad_norm_decay_rate,
    gradient_flow_snapshot,
    param_group_grad_norms,
)


def _fixed_val_batches(val_ids, n_batches, batch_size, block_size, seed):
    """A fixed list of (x, y) val batches, identical across all configs."""
    rng = np.random.default_rng(seed)
    return [get_batch(val_ids, batch_size, block_size, rng) for _ in range(n_batches)]


def _sampled_val_ce(model, val_batches):
    """Mean cross-entropy (nats) over the fixed val batches, no autograd."""
    return float(np.mean([model.batch_ce(x, y) for x, y in val_batches]))


def train_config(
    train_ids,
    val_batches,
    vocab_size,
    norm,
    residual,
    *,
    block_size,
    embed_dim,
    head_dim,
    ff_dim,
    n_layers,
    lr,
    weight_decay,
    batch_size,
    steps,
    eval_every,
    seed,
):
    # Same seed for every config: identical init draws and batch sequence, so
    # the toggle is the only difference between runs.
    rng = np.random.default_rng(seed)
    model = CharTransformer(
        vocab_size, block_size, embed_dim, head_dim, ff_dim, n_layers, rng,
        norm=norm, residual=residual,
    )
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: list[dict] = []
    gradflow: list[dict] = []
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = get_batch(train_ids, batch_size, block_size, rng)

        opt.zero_grad()
        logits, taps = model.forward_with_taps(x)
        B, T = x.shape
        loss = cross_entropy(logits.reshape(B * T, vocab_size), y.reshape(B * T))
        loss.backward()
        opt.step()

        if step == 1 or step % eval_every == 0:
            val_ce = _sampled_val_ce(model, val_batches)
            snap = gradient_flow_snapshot(taps)
            history.append({"step": step, "batch_loss": float(loss.data), "val_ce": val_ce})
            gradflow.append(
                {
                    "step": step,
                    "per_layer": snap,
                    "decay_rate": grad_norm_decay_rate(snap),
                    "param_groups": param_group_grad_norms(model),
                }
            )
            elapsed = time.time() - t0
            print(
                f"    step {step:5d} | batch_loss {float(loss.data):.4f} "
                f"| val_ce {val_ce:.4f} | decay {grad_norm_decay_rate(snap):.3f} "
                f"| {elapsed:.1f}s"
            )
    return model, history, gradflow


def main() -> None:
    print("Loading Tiny Shakespeare ...")
    train_ids, val_ids, tok = load_tinyshakespeare()
    V = tok.vocab_size
    print(f"  vocab_size={V}, train_chars={len(train_ids):,}, val_chars={len(val_ids):,}")

    config = dict(
        block_size=64,
        embed_dim=64,
        head_dim=64,
        ff_dim=256,
        n_layers=4,
        lr=3e-3,
        weight_decay=0.01,
        batch_size=32,
        steps=2000,
        eval_every=100,
        seed=0,
    )
    # Fixed val batches shared across every config (fair comparison).
    val_batches = _fixed_val_batches(
        val_ids, n_batches=20, batch_size=config["batch_size"],
        block_size=config["block_size"], seed=1234,
    )
    print(f"  config: {config}")
    print(f"  val metric: sampled over {len(val_batches)} fixed batches (NOT full-corpus CE)\n")

    out_dir = Path(__file__).resolve().parent / "results" / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for norm, residual in itertools.product(NORM_PLACEMENTS, [True, False]):
        tag = f"{norm}_{'res' if residual else 'nores'}"
        print(f"=== config: norm={norm}  residual={residual}  ({tag}) ===")
        model, history, gradflow = train_config(
            train_ids, val_batches, V, norm, residual, **config
        )
        final_val_ce = history[-1]["val_ce"]
        result = {
            "model": "char_transformer_via_jacobigrad",
            "norm": norm,
            "residual": residual,
            "tag": tag,
            "vocab_size": V,
            "config": config,
            "num_parameters": model.num_parameters(),
            "val_metric": "sampled_ce_over_fixed_val_batches",
            "final": {
                "val_ce_nats": final_val_ce,
                "val_bits_per_char": final_val_ce / float(np.log(2)),
                "val_perplexity": float(np.exp(final_val_ce)),
                "final_decay_rate": gradflow[-1]["decay_rate"],
            },
            "history": history,
            "gradflow": gradflow,
        }
        (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=2))
        summary.append(
            {
                "tag": tag, "norm": norm, "residual": residual,
                "num_parameters": model.num_parameters(),
                "final_val_ce_nats": final_val_ce,
                "final_val_perplexity": float(np.exp(final_val_ce)),
                "final_decay_rate": gradflow[-1]["decay_rate"],
            }
        )
        print(
            f"    -> final val_CE={final_val_ce:.4f} "
            f"ppl={np.exp(final_val_ce):.2f} decay={gradflow[-1]['decay_rate']:.3f}\n"
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Summary (sorted by val CE):")
    for row in sorted(summary, key=lambda r: r["final_val_ce_nats"]):
        print(
            f"  {row['tag']:12s} val_CE={row['final_val_ce_nats']:.4f} "
            f"ppl={row['final_val_perplexity']:.2f} decay={row['final_decay_rate']:.3f}"
        )
    print(f"\nResults -> {out_dir.relative_to(Path.cwd())}/")


if __name__ == "__main__":
    main()
