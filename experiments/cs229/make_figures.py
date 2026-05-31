"""Generate the report figures from the saved experiment results.

Inputs:
    experiments/cs229/results/softmax_baseline.json
    experiments/cs229/results/mlp_baseline.json
    experiments/cs229/results/ablation/*.json   (if present)

Outputs:
    writeups/cs229/figures/training_curves.pdf
    writeups/cs229/figures/baselines_comparison.pdf
    writeups/cs229/figures/gradflow_heatmap.pdf   (ablation, if results present)
    writeups/cs229/figures/ablation_curves.pdf    (ablation, if results present)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
ABLATION_DIR = RESULTS_DIR / "ablation"
FIG_DIR = REPO_ROOT / "writeups" / "cs229" / "figures"

# Stable display order + labels for the six configs.
CONFIG_ORDER = [
    ("pre_res", "pre + res"),
    ("pre_nores", "pre, no res"),
    ("post_res", "post + res"),
    ("post_nores", "post, no res"),
    ("none_res", "none + res"),
    ("none_nores", "none, no res"),
]


def load_results():
    sm = json.loads((RESULTS_DIR / "softmax_baseline.json").read_text())
    ml = json.loads((RESULTS_DIR / "mlp_baseline.json").read_text())
    return sm, ml


def load_ablation() -> dict:
    """Load per-config ablation JSONs keyed by tag; empty dict if absent."""
    if not ABLATION_DIR.exists():
        return {}
    out = {}
    for tag, _ in CONFIG_ORDER:
        path = ABLATION_DIR / f"{tag}.json"
        if path.exists():
            out[tag] = json.loads(path.read_text())
    return out


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


def plot_gradflow_heatmap(ablation: dict) -> None:
    """Hero figure: per-layer gradient norm (log scale) vs depth, all configs.

    Uses the LAST recorded gradflow snapshot per config (end of training).
    Columns = configs, rows = layers (depth 0 nearest the input). The color is
    log10 grad-norm, so vanishing (dark, small) vs exploding (bright, large) is
    immediately legible across the six configs.
    """
    tags = [(t, lbl) for t, lbl in CONFIG_ORDER if t in ablation]
    if not tags:
        return
    n_layers = len(ablation[tags[0][0]]["gradflow"][-1]["per_layer"])

    matrix = np.zeros((n_layers, len(tags)))
    for j, (tag, _) in enumerate(tags):
        per_layer = ablation[tag]["gradflow"][-1]["per_layer"]
        for i, row in enumerate(per_layer):
            matrix[i, j] = row["grad_norm"]

    log_matrix = np.log10(np.maximum(matrix, 1e-30))

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    im = ax.imshow(log_matrix, aspect="auto", cmap="viridis", origin="lower")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\log_{10}\,\|\partial L/\partial h_\ell\|_2$", fontsize=9)

    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels([lbl for _, lbl in tags], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"layer {i}" for i in range(n_layers)], fontsize=8)
    ax.set_ylabel("Depth (0 = nearest input)")
    ax.set_title("Per-layer gradient norm across norm/residual configs")

    # Annotate each cell with the raw value for precision.
    for i in range(n_layers):
        for j in range(len(tags)):
            ax.text(
                j, i, f"{matrix[i, j]:.1e}",
                ha="center", va="center", fontsize=6,
                color="white" if log_matrix[i, j] < log_matrix.max() - 0.5 else "black",
            )

    out = FIG_DIR / "gradflow_heatmap.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def plot_ablation_curves(ablation: dict) -> None:
    """Val CE vs step, one line per config."""
    tags = [(t, lbl) for t, lbl in CONFIG_ORDER if t in ablation]
    if not tags:
        return

    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    cmap = plt.get_cmap("tab10")
    for k, (tag, lbl) in enumerate(tags):
        hist = ablation[tag]["history"]
        steps = [h["step"] for h in hist]
        val = [h["val_ce"] for h in hist]
        ax.plot(steps, val, label=lbl, color=cmap(k), linewidth=1.4)

    ax.set_xlabel("AdamW step")
    ax.set_ylabel("Sampled val cross-entropy (nats / char)")
    ax.set_title("Training stability by norm placement x residual")
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)
    ax.grid(True, alpha=0.3)

    out = FIG_DIR / "ablation_curves.pdf"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sm, ml = load_results()
    plot_training_curves(sm, ml)
    plot_baselines_comparison(sm, ml)

    ablation = load_ablation()
    if ablation:
        plot_gradflow_heatmap(ablation)
        plot_ablation_curves(ablation)
    else:
        print("  (no ablation results yet -- skipping gradflow/ablation figures)")


if __name__ == "__main__":
    main()
