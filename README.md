# jacobigrad

A from-scratch reverse-mode autograd engine over batched tensors, with a small set of linear-algebra primitives and a decoder-only transformer block built on top. Named for Carl Gustav Jacob Jacobi, whose Jacobians the engine surfaces directly.

## Layout

```
src/jacobigrad/    # library: autograd engine, primitives, attention, layers
experiments/       # research and coursework
  cs229/           # CS 229 final project: gradient-flow ablation
writeups/          # LaTeX sources for proposals, milestones, reports
notebooks/         # exploration and one-off analyses
tests/             # unit and gradient-check tests
```

## Status

Early scaffolding. The first concrete deliverable is the CS 229 course project (Spring 2025–26): a gradient-flow ablation over LayerNorm placement and residual connections, trained on character-level Tiny Shakespeare, benchmarked against a PyTorch transformer and a softmax-regression baseline.
