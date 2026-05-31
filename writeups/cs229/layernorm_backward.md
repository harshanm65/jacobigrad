# LayerNorm backward (closed form)

We implement LayerNorm by composing autograd primitives and let the engine
differentiate it — but the closed form is worth deriving once, both to
understand what the engine is computing and to pin it down with a parity test
(`tests/test_layernorm.py::test_layernorm_dx_matches_closed_form`).

## Forward

Normalize over the feature axis (length `H`). For one token's feature vector
`x` (length `H`):

```
mu   = (1/H) sum_i x_i
var  = (1/H) sum_i (x_i - mu)^2          (biased: divide by H, matches torch)
xhat = (x - mu) / sqrt(var + eps)
y    = gamma * xhat + beta               (elementwise, gamma/beta length H)
```

## Backward

Given `dL/dy` (same shape as `x`). Want `dL/dx`, `dL/dgamma`, `dL/dbeta`.

### Affine params (easy — they broadcast across the batch)

```
dL/dbeta  = sum over all non-feature positions of  dL/dy
dL/dgamma = sum over all non-feature positions of  dL/dy * xhat
```

(In our engine this summation is the broadcasting fan-in: gamma/beta are
shape `(H,)`, the activations are `(B, T, H)`, so `_unbroadcast` sums the
upstream grad over the leading `(B, T)` axes automatically.)

### Input gradient (the interesting part)

First pull the upstream grad through the affine: let `dxhat = dL/dy * gamma`.

Now we need `dL/dx` given `dxhat`, where `xhat = (x - mu)/sqrt(var+eps)` and
both `mu` and `var` depend on every `x_i`. Working per row (length `H`), with
`rstd = (var + eps)^{-1/2}`:

The standard result (collecting the direct, mean-path, and variance-path
contributions) is:

```
dL/dx_i = rstd * ( dxhat_i
                   - (1/H) sum_j dxhat_j
                   - xhat_i * (1/H) sum_j (dxhat_j * xhat_j) )
```

Read the three terms as:
- **direct**: `rstd * dxhat_i` — the normalized passthrough.
- **mean path**: subtract the row-mean of `dxhat` (because shifting any `x_i`
  moves `mu`, which moves every `xhat_j`).
- **variance path**: subtract `xhat_i` times the row-mean of `dxhat * xhat`
  (because shifting `x_i` moves `var`, scaling the whole row).

Both correction terms are row-shared scalars, exactly analogous to the
softmax backward's shared `<a, dL/da>` correction.

## Vectorized (over a batch, feature axis = -1)

```
dxhat = dL/dy * gamma
mean1 = mean(dxhat,          axis=-1, keepdims=True)
mean2 = mean(dxhat * xhat,   axis=-1, keepdims=True)
dL/dx = rstd * (dxhat - mean1 - xhat * mean2)
```

This is what `tests/test_layernorm.py` checks our composed-autograd gradient
against (alongside finite differences and a `torch.nn.LayerNorm` parity test).
The engine reproduces it to machine precision without any of this being
written down in code — the derivation is for us, not the computer.
```
