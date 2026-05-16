# CS 229 Milestone Report Plan

**Deadline:** Sun May 17, 2026.
**Goal:** A 3-page milestone report whose Section 3 ("Preliminary Experiments and Results") is grounded in real numbers from a model **trained with our own autograd engine** — not just hand-derived baselines.

This compresses the original PLAN.md Weeks 2–3 into a 2-day push. Matmul/LayerNorm/attention engine ports are deferred to post-milestone (Luis's branch already has the attention spec we'll port against).

## Why this scope

The milestone template (`.claude_docs/CS_229_Project_Milestone/cs229-milestone.tex`) demands real measurements in Section 3. Our current pipeline gives:

  - Uniform / closed-form bigram / softmax-regression baselines (✅ committed).
  - Scalar reverse-mode autograd, gradient-checked + bit-identical with PyTorch (✅ committed).

That's defensible but thin: Section 3 reads as "we built scaffolding." Adding **a tiny MLP trained on Tiny Shakespeare via our own tensor autograd, beating the bigram floor**, turns the section into "we built and validated the central engine of the project."

## Phase 2 — Tensor autograd core (Day 1, May 15)

Minimum viable surface for an MLP:

| Op / utility | Shape semantics | Backward notes |
|---|---|---|
| `Tensor` class | wraps `np.ndarray`, `data` + `grad` + `_prev` + `_op` + `_backward` | identical pattern to scalar engine; broadcasting-aware grad accumulation |
| `+`, `-`, `*`, `/`, `neg`, `pow` (const exp) | elementwise w/ broadcasting | `unbroadcast(grad, target_shape)` reduces broadcast axes |
| `sum`, `mean` | axis + keepdims | broadcast `out.grad` back to input shape |
| `@` (matmul) | (..., M, K) @ (..., K, N) | `dA = dC @ Bᵀ`, `dB = Aᵀ @ dC`; sum over leading batch dims if input shape lacks them (Luis's "fan-out → fan-in") |
| `__getitem__(int_array)` | embedding lookup, multi-d index OK | `np.add.at(dW, idx, dout)` scatter-add |
| `reshape` | view-like | reshape grad back |
| `tanh` | elementwise | `1 − tanh²` |
| `log_softmax(axis=-1)` | row-wise stable | `dS = dA − A · sum(dA)` row-wise (Luis's softmax identity) |
| `cross_entropy(logits, targets)` | scalar | fused: `dlogits = (softmax − onehot)/B` |
| `numgrad_check(fn, params)` | utility | central-difference vs analytic; the **non-negotiable** Week 2 deliverable from PLAN.md |

Layout:

```
src/jacobigrad/autograd/
  scalar.py          # ✅ done
  tensor.py          # NEW: Tensor class + ops
  gradcheck.py       # NEW: numerical gradient checker
src/jacobigrad/nn/
  __init__.py        # NEW
  losses.py          # NEW: cross_entropy
  functional.py      # NEW: log_softmax, tanh — anything that's stateless
tests/
  test_tensor_autograd.py    # NEW: gradcheck on every op
```

## Phase 3 — Train an MLP (Day 2 morning, May 16)

**Architecture (Bengio 2003 char-MLP):**

  - Embedding: `E ∈ R^(V × d_emb)`, `d_emb = 16`.
  - Context: previous `block_size = 3` characters.
  - Hidden: `tanh(x_emb_concat @ W₁ + b₁)`, `d_hidden = 64`.
  - Output: `... @ W₂ + b₂`, shape `(B, V)`.
  - Loss: cross-entropy on next-char target.
  - Total params: 65·16 + (3·16)·64 + 64 + 64·65 + 65 ≈ 8.5K.

**Training:** SGD (or simple AdamW if needed), `batch_size = 256`, `lr ∈ {0.05, 0.1}`, 2k–5k steps, log train + val CE every 200 steps.

**Acceptance:** val CE strictly below the bigram floor (2.4819 nats). Anything < 2.3 is a strong result; even ~2.4 is fine for a context-3 shallow MLP.

## Phase 4 — Writeup (Day 2 afternoon → Day 3, May 16–17)

Sections of `cs229-milestone.tex` (already templated):

  1. **Motivation** — gradient-flow ablation thesis (LayerNorm placement × residual). Why this experiment is interesting; what we expect to learn.
  2. **Methods** — data pipeline → handcrafted softmax-regression → scalar autograd → tensor autograd → MLP. Mention Luis's attention-backward derivation as part of the autograd-engine roadmap.
  3. **Preliminary Results** — baseline table (uniform / bigram / softmax-regression / **MLP-via-our-autograd**) + training-curve figure + autograd correctness validation (gradcheck + torch parity).
  4. **Next Steps** — port Luis's closed-form attention into the Tensor engine; build the decoder block; run the 6-config ablation grid.
  5. **Team Contributions** — Harsha: data pipeline, baselines, scalar + tensor autograd, MLP, writeup. Luis: closed-form attention backward derivation + PyTorch reference + parity tests.

Two figures planned in `writeups/cs229/figures/`:
  - `training_curves.pdf` — train + val CE vs steps with bigram floor + uniform ceiling.
  - `gradcheck_errors.pdf` — log-scale histogram of analytic-vs-finite-diff gradient errors across the engine's ops.

## Acceptance for the milestone

- Tensor autograd has all ops above, every op gradient-checked.
- MLP val CE < bigram floor (2.4819 nats).
- 3-page PDF with all 5 sections filled, figures embedded, real numbers in Section 3.
- Submitted by May 17.

## Deferred (post-milestone)

Matmul-based deep nets beyond the MLP, LayerNorm forward+backward, scaled-dot-product attention forward/backward in our engine, decoder block with toggleable LayerNorm placement, ablation grid (6 configs), final report.
