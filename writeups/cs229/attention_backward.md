# Attention backward (closed form)


## Forward

```
X  (B, T, d)               embeddings
W_Q, W_K, W_V  (d, d_k)    projection weights (single head)

Q = X @ W_Q                (B, T, d_k)
K = X @ W_K
V = X @ W_V

S = Q @ K.T / sqrt(d_k)    (B, T, T)
S = S + mask               (optional; additive, -inf = masked)
A = softmax(S, axis=-1)
Y = A @ V                  (B, T, d_k)
```

Causal mask: 0 on/below the diagonal, -inf strictly above. Additive form makes
masked positions get softmax-prob 0 naturally without breaking the row normalization.

## Backward

Given dL/dY shape (B, T, d_k). Want dL/dX, dL/dW_Q, dL/dW_K, dL/dW_V.

Reused throughout: if `Y = M1 @ M2`, then
```
dL/dM1 = dL/dY @ M2.T
dL/dM2 = M1.T  @ dL/dY
```
Easy way to remember: the *other* operand, transposed, on the opposite side.
(Derivation is just expanding the matmul as a sum and applying the chain rule
component-wise. I worked it out once on paper to convince myself.)

### Y = A @ V

```
dL/dA = dL/dY @ V.T          (B, T, T)
dL/dV = A.T   @ dL/dY        (B, T, d_k)
```

dL/dV is one of the four final answers. Done with it.

### A = softmax(S)

This is the only step that isn't just matmul backward, so worth being careful.

Softmax acts row by row, so I can think about a single row at a time. Let `a`
and `s` be one row each (length T). The annoying property: every output `a_j`
depends on every input `s_i`, because of the normalizer. So the per-row Jacobian
is a full T-by-T matrix, not diagonal.

Per-element partial (quotient rule on `a_j = exp(s_j) / sum_k exp(s_k)`):

```
∂a_j / ∂s_i = a_j * (δ_ij - a_i)
```

Chain rule, then collect terms:

```
dL/ds_i = sum_j (dL/da_j) * a_j * (δ_ij - a_i)
        = a_i (dL/da_i) - a_i sum_j a_j (dL/da_j)
        = a_i * (dL/da_i - <a, dL/da>)
```

The dot product `<a, dL/da>` is one scalar shared across the whole row, so
the same correction is subtracted from every position. Conceptually it's the "average gradient over this row,
weighted by softmax probs," and it's there because of probability conservation
(if you push one s_i up, all the other a_j get smaller by an a_j-weighted
amount).

Vectorized over the (T, T) batch:

```
dL/dS = A * (dL/dA - rowsum(A * dL/dA))
```

with rowsum keeping a trailing 1-axis so it broadcasts back across the row.
In torch: `A * (dA - (A * dA).sum(dim=-1, keepdim=True))`.

Note: mask doesn't appear here. Additive masks have identity Jacobian, and
masked positions have A = 0 exactly, so no gradient flows through them.

### S = Q @ K.T / sqrt(d_k)

Two operations stacked. The scale just divides through (if S = M/c then dL/dM
= dL/dS / c). Real work is the matmul, with the wrinkle that K is transposed
inside it.

For dL/dQ, Q is the left operand. Pattern gives `dL/dM @ M2.T`, where M2 = K.T,
so M2.T = K. That's almost the whole derivation:

```
dL/dQ = (dL/dS @ K) / sqrt(d_k)
```

For dL/dK it's fussier. The pattern gives me `dL/d(K.T)`, not what I want. So
I compute `dL/d(K.T) = Q.T @ dL/dM` and then transpose. Using (AB).T = B.T A.T:

```
dL/dK = (dL/dS.T @ Q) / sqrt(d_k)
```

Nice symmetry with dL/dQ in the end. Same shape, just dL/dS gets transposed
and Q swaps in for K.

### Projections: Q, K, V = X @ W_*

Three applications of the same matmul-backward pattern. X is always the left
operand, the weight is always on the right.

Weights:

```
dL/dW_P = X.T @ dL/dP
```

X contributions:

```
dL/dX|_P = dL/dP @ W_P.T
```

Two things to be careful about, both about the batch dimension:

X has a batch dim (B, T, d), the weights do not (d, d_k) because they're shared
across batch elements. So when I write `X.T @ dL/dP` in batched torch, the
result is shape (B, d, d_k) and I need to **sum over the batch axis** to
collapse to (d, d_k). Fan-out in forward (W appears in B different forward
computations) means fan-in in backward (sum over those B contributions).

dL/dX is the sum of all three path contributions, because X feeds into Q, K,
and V in the forward:

```
dL/dX = dL/dQ @ W_Q.T + dL/dK @ W_K.T + dL/dV @ W_V.T
```

## Final formulas

```
dL/dV   = A.T @ dL/dY
dL/dA   = dL/dY @ V.T
dL/dS   = A * (dL/dA - rowsum(A * dL/dA))
dL/dQ   = (dL/dS   @ K) / sqrt(d_k)
dL/dK   = (dL/dS.T @ Q) / sqrt(d_k)

dL/dW_Q = sum over batch of  X[b].T @ dL/dQ[b]    -> (d, d_k)
dL/dW_K = sum over batch of  X[b].T @ dL/dK[b]
dL/dW_V = sum over batch of  X[b].T @ dL/dV[b]

dL/dX   = dL/dQ @ W_Q.T  +  dL/dK @ W_K.T  +  dL/dV @ W_V.T
