"""Generate the milestone-report figures from the saved experiment results.

Inputs:
    experiments/cs229/results/softmax_baseline.json
    experiments/cs229/results/mlp_baseline.json

Outputs:
    writeups/cs229/figures/training_curves.pdf
    writeups/cs229/figures/baselines_comparison.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIG_DIR = REPO_ROOT / "writeups" / "cs229" / "figures"


def load_results():
    sm = json.loads((RESULTS_DIR / "softmax_baseline.json").read_text())
    ml = json.loads((RESULTS_DIR / "mlp_baseline.json").read_text())
    return sm, ml


def plot_training_curves(sm: dict, ml: dict) -> None:
    """MLP training curve with bigram floor + uniform ceiling reference lines."""
    fig, ax = plt.subplots(figsize=(5.5, 3.6))

    steps = [h["step"] for h in ml["history"]]
    train_ce = [h["train_ce"] for h in ml["history"]]
    val_ce = [h["val_ce"] for h in ml["history"]]

    ax.plot(steps, train_ce, label="MLP train", color="#1f77b4", linewidth=1.5)
    ax.plot(steps, val_ce, label="MLP val", color="#ff7f0e", linewidth=1.5)

    bigram_floor = sm["bigram_closed_form"]["val_ce_nats"]
    uniform = sm["uniform"]["ce_nats"]
    ax.axhline(
        bigram_floor, color="#2ca02c", linestyle="--", linewidth=1.2,
        label=f"Bigram floor (val, {bigram_floor:.3f})",
    )
    ax.axhline(
        uniform, color="#7f7f7f", linestyle=":", linewidth=1.0,
        label=f"Uniform ceiling ({uniform:.3f})",
    )

    ax.set_xlabel("SGD step")
    ax.set_ylabel("Cross-entropy (nats / char)")
    ax.set_title("Char-MLP training curve, Tiny Shakespeare")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(steps))

    out = FIG_DIR / "training_curves.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def plot_baselines_comparison(sm: dict, ml: dict) -> None:
    """Bar chart: val CE in nats for each model."""
    labels = ["Uniform", "Softmax-reg\n(SGD)", "Bigram\n(closed form)", "Char-MLP\n(ours)"]
    vals_nats = [
        sm["uniform"]["ce_nats"],
        sm["softmax_regression_sgd"]["val_ce_nats"],
        sm["bigram_closed_form"]["val_ce_nats"],
        ml["final"]["val_ce_nats"],
    ]
    vals_ppl = [float(np.exp(v)) for v in vals_nats]
    colors = ["#7f7f7f", "#d62728", "#2ca02c", "#1f77b4"]

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    bars = ax.bar(labels, vals_nats, color=colors, edgecolor="black", linewidth=0.5)

    # Annotate each bar with nats and (ppl).
    for bar, n, p in zip(bars, vals_nats, vals_ppl):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{n:.3f}\n(ppl {p:.2f})",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_ylabel("Val cross-entropy (nats / char)")
    ax.set_title("Val CE on Tiny Shakespeare across baselines")
    ax.set_ylim(0, max(vals_nats) * 1.18)
    ax.grid(True, axis="y", alpha=0.3)

    out = FIG_DIR / "baselines_comparison.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sm, ml = load_results()
    plot_training_curves(sm, ml)
    plot_baselines_comparison(sm, ml)


if __name__ == "__main__":
    main()
