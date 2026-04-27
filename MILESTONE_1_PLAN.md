# Milestone 1 Plan — Week 1: Foundations

**Parent plan:** [PLAN.md](PLAN.md) → Week 1 (Apr 21 – Apr 27, 2026, flexible).
**Goal:** Stand up the data, eval, and writeup loop end-to-end *before* any autograd risk lands. Get a first val cross-entropy / perplexity number on Tiny Shakespeare from a baseline whose optimum we can solve in closed form, so we know the loop works.

> Timeline is a guideline. Depth and learning take priority over hitting the Apr 27 boundary.

## Status of Week 1 deliverables

| # | Deliverable | Status |
|---|---|---|
| 0 | Project proposal | ✅ submitted |
| 1 | Tiny Shakespeare data loader + char tokenizer | ⬜ this milestone |
| 2 | Softmax-regression baseline (plain numpy) | ⬜ this milestone |
| 3 | Karpathy `micrograd` scalar port | ⬜ this milestone |

## Step 1 — Data layer

**Module:** `src/jacobigrad/data/tinyshakespeare.py`

- **Tokenizer (char-level):** vocab = sorted unique chars from the corpus (~65). `encode: str → np.ndarray[int64]`, `decode: np.ndarray[int64] → str`. Vocab order is deterministic so train/val IDs are stable across runs.
- **Split:** 90/10 train/val by *position*, not random. Random shuffling at the char level would leak local context across the boundary and inflate val performance in a way that doesn't reflect generalization.
- **Sampling:** `get_batch(split, batch_size, block_size, rng)` returns `(x, y)` where `x[i]` is a length-`block_size` window starting at a random position in `split`, and `y[i] = x[i]` shifted by 1 (next-char target). Caller passes an `np.random.Generator` so runs are reproducible.
- **Download:** fetch Karpathy's `input.txt` into `data/tiny_shakespeare.txt` (gitignored) on first use; idempotent cache. URL pinned in code with a comment for provenance.

**Tests:** vocab size sanity, encode/decode round-trip, `get_batch` shapes + shift invariant (`y[:, :-1] == x[:, 1:]`).

## Step 2 — Softmax-regression baseline

**Module:** `experiments/cs229/softmax_baseline.py`

- **Model:** `P(x_{t+1} | x_t) = softmax(W · onehot(x_t) + b)`, with `W ∈ ℝ^(V×V)`, `b ∈ ℝ^V`. Structurally a bigram model.
- **Why this baseline:** the optimum has a closed form — the empirical bigram distribution `P̂(j | i) = count(i, j) / count(i)` (with optional add-ε smoothing). We can compute this directly and compare gradient-descent's converged solution against it. This validates that:
  - the data loader produces correct (x, y) pairs,
  - the cross-entropy + softmax math is correct,
  - the eval pipeline (val CE, perplexity) returns numbers we trust.
- **Math (derived in code comments / writeup):**
  - `loss = -mean_t log p(y_t | x_t)`
  - With softmax + cross-entropy, `∂L/∂logits_t = p_t − onehot(y_t)` (the classic identity — derive it once, use everywhere).
  - For one-hot inputs, `∂L/∂W[:, x_t] = (p_t − onehot(y_t))`, all other columns get zero contribution from sample t.
- **Training:** plain numpy, full-batch or mini-batch GD. Single seed.
- **Reporting:** train + val cross-entropy (nats *and* bits per char), perplexity, gap to the closed-form bigram optimum. Save metrics to `experiments/cs229/results/softmax_baseline.json`.

**Acceptance:** GD converges to within ~1e-3 nats of the closed-form bigram CE on the val split.

## Step 3 — Scalar `micrograd` port (autograd warm-up)

**Module:** `src/jacobigrad/autograd/scalar.py`

- `Value` class with `data`, `grad`, `_prev: tuple[Value, ...]`, `_op: str`, `_backward: Callable[[], None]`.
- Ops: `+`, `*`, `-`, `/`, `**` (constant exponent), `tanh`, `exp`, `log`, `relu`. Each forward op installs a closure that distributes the upstream gradient through the local Jacobian.
- `backward()`: build a topological order of the DAG from the output node, zero grads, set output grad to 1, walk in reverse calling each `_backward`.
- Treat this as *rehearsal* for Week 2's tensor autograd — same pattern (closure-based local Jacobian, topo-sort reverse pass), just scalar.

**Tests:** numerical gradient check against central-difference finite differences on a non-trivial expression (e.g., a small MLP-shaped scalar computation). Tolerance: `1e-6` absolute or `1e-4` relative.

## Layout

```
src/jacobigrad/
  data/
    __init__.py
    tinyshakespeare.py
  autograd/
    __init__.py
    scalar.py
experiments/cs229/
  softmax_baseline.py
  results/
    softmax_baseline.json
tests/
  test_tinyshakespeare.py
  test_scalar_autograd.py
```

## Order of operations

1. **Data layer** (Step 1) — unblocks everything.
2. **Softmax-regression baseline** (Step 2) — the de-risking deliverable; gates downstream work per PLAN.md.
3. **Scalar autograd port** (Step 3) — independent of 1 and 2; safe last because it's prep for Week 2 rather than a Week 1 dependency.

## Acceptance for the milestone

- `pytest` green for `tests/test_tinyshakespeare.py` and `tests/test_scalar_autograd.py`.
- Softmax baseline reports val CE / perplexity within tolerance of the closed-form bigram optimum.
- Scalar autograd matches finite differences within tolerance on the gradient-check expression.
- All work merged into `main` via the `milestone-1-foundations` branch.
