# CS 229 Project Plan

**Project:** Micro-Transformer — A From-Scratch Autograd and Attention Engine for Character-Level Language Modeling
**Team:** Manikanta Harsha Nadimpalli (`nmharsha`), Luis Blanco Hoyos (`blancol`)
**Window:** Apr 21 – Jun 7, 2026 (~6.7 weeks)
**Effective workforce:** 1 (minimal partner support assumed)

> Course deadlines (milestone, poster, final) below are placeholders. Confirm against the CS 229 Logistics doc and update.

## Feasibility

Tight but realistic, conditional on three non-negotiables:

1. **Numerical gradient checking is written before any layer is "done."** Without it, autograd debugging will eat 1–2 weeks we don't have.
2. **Hold the scope.** Single-head attention (not multi-head). One seed per config. Tiny Shakespeare only. The thesis is the *gradient-flow ablation*; everything else is supporting infrastructure.
3. **Working softmax-regression numbers by end of Week 1.** Proves data pipeline + eval + writeup loop work end-to-end before any autograd risk lands.

Biggest risk: attention backward pass with closed-form Q/K/V Jacobians takes 4 days instead of 2. Mitigation: the milestone (Week 4) only needs the autograd + an MLP working; transformer can be in-progress.

## Weekly milestones

### Week 1 — Apr 21 – Apr 27: Foundations
- Submit project proposal.
- Tiny Shakespeare data loader + character tokenizer (~65-char vocab).
- **Softmax-regression baseline** in plain NumPy (no autograd) → first val cross-entropy / perplexity number.
- Karpathy scalar `micrograd` port as autograd warm-up.

### Week 2 — Apr 28 – May 4: Tensor Autograd Core
- Reverse-mode DAG over batched tensors with broadcasting.
- Ops: `+`, `-`, `*`, `/`, `sum`, `mean`, parameter updates.
- **Numerical gradient checker** — gates everything downstream.
- Toy MLP trains on a synthetic task using only our engine.

### Week 3 — May 5 – May 11: Linear-Algebra Primitives
- Batched matmul (forward + backward, grad-checked).
- Numerically stable softmax (log-sum-exp).
- Cross-entropy loss + backward.
- LayerNorm forward + backward.
- Custom MLP trains on Tiny Shakespeare next-char and matches a PyTorch MLP within tolerance.

### Week 4 — May 12 – May 18: Attention + Milestone
- Scaled dot-product attention forward.
- Attention backward with closed-form Jacobians for Q, K, V projections.
- Decoder block with toggleable LayerNorm placement (pre/post/none) and residual on/off.
- PyTorch reference implementation, parity test against ours.
- **Milestone writeup (3 pages) submitted ~Sun May 17.**

### Week 5 — May 19 – May 25: Ablation + Benchmarks
- Ablation grid: 3 norm-placements × 2 residual settings = 6 configs.
- Instrumentation: per-layer gradient norms, activation norms, Jacobian singular-value statistics logged across training.
- Benchmark suite vs PyTorch: validation cross-entropy, held-out perplexity, steps-to-threshold, peak memory, wallclock per step.

### Week 6 — May 26 – Jun 1: Analysis + Poster
- Plots: gradient-flow heatmaps, training curves, perplexity-vs-steps, stability comparisons.
- Qualitative generation samples per configuration.
- Poster build (figures-first per CS 229 guidelines).
- Poster session prep.

### Week 7 — Jun 2 – Jun 7: Final Report
- 5-page final writeup.
- Code cleanup, README, reproducibility instructions.
- Submit final report.

## Scope cut order (if behind schedule)

In priority order — drop from the bottom first:

1. ~~Multi-seed runs~~ → single seed per config.
2. ~~Multi-head attention~~ → single-head.
3. ~~Generation samples~~ → drop, keep quantitative metrics only.
4. ~~Full benchmark suite~~ → keep only val-perplexity vs PyTorch parity.

## Standing risks

- **Hardware:** numpy autograd is CPU-only; verify Tiny Shakespeare configs train in minutes, not hours, on the dev machine before Week 3.
- **Numerical instability** in softmax / attention: budget half a day per primitive for stability fixes.
- **Partner contribution gap:** plan assumes solo execution; any contribution from Luis is upside, not a dependency.
