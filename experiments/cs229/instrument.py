"""Gradient-flow instrumentation for the ablation.

The project's central measurement: how does the gradient signal propagate
through depth under each (norm, residual) config? After a normal
``loss.backward()`` on a model built with ``forward_with_taps``, every
block-output Tensor exposes ``.grad`` = dL/dh at that depth. We summarize the
flow with a few scalars per layer.

These metrics are the standard signal-propagation diagnostics:

  - **grad norm** ‖dL/dh_ℓ‖₂ per layer ℓ. The headline number. In a healthy
    network it stays O(1) across depth; vanishing means it shrinks toward the
    input, exploding means it grows.
  - **activation norm** ‖h_ℓ‖₂ (RMS per element) — the forward-side companion.
  - **grad-norm ratio** between adjacent layers — the per-layer
    amplification/decay factor; its geometric mean is the decay rate.

A secondary view groups *parameter* gradients by role (attention vs FFN vs
LayerNorm) to see which sublayers receive signal.

Everything here is pure read-out of numpy arrays already on the graph — no
autograd, no torch.
"""

from __future__ import annotations

import numpy as np


def _l2(a: np.ndarray) -> float:
    return float(np.sqrt(np.sum(a * a)))


def _rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a * a)))


def gradient_flow_snapshot(taps: list, *, eps: float = 1e-12) -> list[dict]:
    """Per-layer gradient/activation summary from block-output taps.

    ``taps`` is the list returned by ``CharTransformer.forward_with_taps``,
    after ``loss.backward()`` has run. Index 0 is the first (closest to input)
    block; the last entry is closest to the output.

    Returns a list of dicts, one per layer, each with:
        depth, grad_norm, grad_rms, act_norm, act_rms, grad_ratio_to_next
    where ``grad_ratio_to_next`` is ‖g_{ℓ+1}‖ / ‖g_ℓ‖ (None for the last layer).
    """
    grad_norms = [_l2(t.grad) if t.grad is not None else 0.0 for t in taps]
    rows: list[dict] = []
    for ell, t in enumerate(taps):
        g = t.grad if t.grad is not None else np.zeros_like(t.data)
        nxt = grad_norms[ell + 1] if ell + 1 < len(taps) else None
        ratio = (nxt / (grad_norms[ell] + eps)) if nxt is not None else None
        rows.append(
            {
                "depth": ell,
                "grad_norm": grad_norms[ell],
                "grad_rms": _rms(g),
                "act_norm": _l2(t.data),
                "act_rms": _rms(t.data),
                "grad_ratio_to_next": ratio,
            }
        )
    return rows


def grad_norm_decay_rate(snapshot: list[dict], *, eps: float = 1e-12) -> float:
    """Geometric-mean per-layer grad-norm ratio (input side -> output side).

    > 1 means the gradient grows toward the output (so it *decays* toward the
    input — vanishing); ~1 means well-preserved; >> 1 means exploding. Computed
    as (‖g_last‖ / ‖g_first‖) ** (1/(L-1)).
    """
    if len(snapshot) < 2:
        return 1.0
    first = snapshot[0]["grad_norm"] + eps
    last = snapshot[-1]["grad_norm"] + eps
    return float((last / first) ** (1.0 / (len(snapshot) - 1)))


def param_group_grad_norms(model) -> dict[str, float]:
    """Total grad L2 norm per parameter role: attention, ffn, layernorm, embed, head.

    A coarse secondary view of where gradient signal lands. Uses the known
    structure of CharTransformer / DecoderBlock rather than fragile name
    parsing.
    """
    groups = {"attn": [], "ffn": [], "ln": [], "embed": [], "head": []}

    groups["embed"] += [model.E, model.P]
    groups["head"] += [model.W_head, model.b_head]
    if model.ln_f is not None:
        groups["ln"] += list(model.ln_f.parameters())

    for blk in model.blocks:
        groups["attn"] += [*blk.attn.parameters(), blk.W_O]
        groups["ffn"] += [blk.W1, blk.b1, blk.W2, blk.b2]
        if blk.norm != "none":
            groups["ln"] += [*blk.ln1.parameters(), *blk.ln2.parameters()]

    out: dict[str, float] = {}
    for name, params in groups.items():
        sq = sum(float(np.sum(p.grad * p.grad)) for p in params if p.grad is not None)
        out[name] = float(np.sqrt(sq))
    return out
