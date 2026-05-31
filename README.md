# jacobigrad

A from-scratch reverse-mode autograd engine over batched NumPy tensors, and a
decoder-only transformer built entirely on top of it. Named for Carl Gustav
Jacob Jacobi, whose Jacobians the engine surfaces directly.

Built for the CS 229 final project (Spring 2025–26). The engine is the
instrument; the result is a **gradient-flow ablation**: how do LayerNorm
placement (pre / post / none) and residual connections shape the gradient that
reaches the early layers of a transformer during training?

**Headline finding.** At depth 4 on character-level Tiny Shakespeare, the
**residual connection — not normalization placement — is the decisive factor.**
All three normalized variants *with* residuals train to ≈1.8 nats/char (beating
a closed-form bigram floor of 2.48); all three *without* residuals collapse to
≈3.34 as the gradient vanishes toward the input (the no-norm/no-residual model
has exactly zero gradient in its first three layers).

No deep-learning framework is used in the model or its gradients — only NumPy.
PyTorch appears solely as a test oracle to validate our backward passes.

## Layout

```
src/jacobigrad/        # the library
  autograd/            # scalar + tensor reverse-mode engines, gradient checker
  nn/                  # attention, layernorm, decoder block, transformer, MLP, losses
  baselines/           # closed-form bigram, softmax regression
  data/                # Tiny Shakespeare loader, tokenizer, batching
  optim.py             # from-scratch AdamW
experiments/cs229/     # baselines, the ablation, instrumentation, figures
  results/             # committed JSON outputs (baselines + 6-config ablation)
writeups/cs229/        # LaTeX: proposal, milestone, final report, poster, derivations
tests/                 # per-op gradient checks + torch/closed-form parity (164 tests)
```

## Setup

This project uses [uv](https://docs.astral.sh/uv/). From the repo root:

```bash
uv sync --extra dev --extra torch   # torch is only needed for the parity tests
```

## Reproduce the results

```bash
# Baselines (uniform / bigram / softmax-regression / char-MLP)
uv run python experiments/cs229/softmax_baseline.py
uv run python experiments/cs229/mlp_baseline.py

# The gradient-flow ablation — trains all 6 configs (a few minutes, CPU)
uv run python experiments/cs229/ablation.py

# Regenerate the report/poster figures from the saved JSON
uv run python experiments/cs229/make_figures.py
```

## Tests

```bash
uv run pytest -q        # 164 tests: per-op gradchecks + torch/closed-form parity
uv run ruff check       # lint
```

Every engine layer is validated to machine precision (≈1e-10 to 1e-16) against
finite differences and a PyTorch / closed-form oracle before it is used.

## Report & poster

- Final report: `writeups/cs229/final.pdf`
- Poster: `writeups/cs229/poster/poster.pdf`
- Derivations: `writeups/cs229/attention_backward.md`, `layernorm_backward.md`

## Authors

Manikanta Harsha Nadimpalli (`nmharsha`) · Luis Blanco Hoyos (`blancol`).
See the report's Contributions section for the work split.

## License

MIT — see [LICENSE](LICENSE).
