"""PyTorch reference for single-head scaled dot-product attention.

Used to check our jacobigrad implementation matches autograd.
"""

import math

import torch


def causal_mask(T):
    # (T, T) additive mask. 0 below/on the diagonal, -inf above
    # (so future tokens get zero probability after softmax).
    mask = torch.zeros(T, T)
    upper = torch.triu(torch.ones(T, T), diagonal=1).bool()
    return mask.masked_fill(upper, float("-inf"))


def torch_attention(X, W_Q, W_K, W_V, mask=None):
    # X: (B, T, d), W_*: (d, d_k)
    d_k = W_Q.shape[1]

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    # S: (B, T, T)
    S = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        S = S + mask

    A = torch.softmax(S, dim=-1)
    Y = A @ V
    return Y


def closed_form_backward(grad_Y, X, W_Q, W_K, W_V, mask=None):
    """Returns dX, dW_Q, dW_K, dW_V via the closed-form Jacobians.

    See writeups/cs229/attention_backward.md for the derivation.
    No autograd anywhere -- it's the spec.
    """
    d_k = W_Q.shape[1]
    scale = 1.0 / math.sqrt(d_k)

    # Recompute forward intermediates we need (A, V, Q, K).
    # A real autograd engine would save these in the forward instead.
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    S = Q @ K.transpose(-2, -1) * scale
    if mask is not None:
        S = S + mask
    A = torch.softmax(S, dim=-1)

    # Y = A V
    dA = grad_Y @ V.transpose(-2, -1)
    dV = A.transpose(-2, -1) @ grad_Y

    # Softmax backward, per row:
    #   dS_i = a_i * (dA_i - sum_j a_j * dA_j)
    dS = A * (dA - (A * dA).sum(dim=-1, keepdim=True))

    # S = Q K^T / sqrt(d_k)
    dQ = dS @ K * scale
    dK = dS.transpose(-2, -1) @ Q * scale

    # Projections {Q, K, V} = X @ W_*. Weight grads sum over the batch
    # (W is shared across batch elements, X is not).
    dW_Q = (X.transpose(-2, -1) @ dQ).sum(dim=0)
    dW_K = (X.transpose(-2, -1) @ dK).sum(dim=0)
    dW_V = (X.transpose(-2, -1) @ dV).sum(dim=0)

    # X feeds into all three projections, so dL/dX is the sum of contributions.
    dX = dQ @ W_Q.transpose(-2, -1) + dK @ W_K.transpose(-2, -1) + dV @ W_V.transpose(-2, -1)

    return dX, dW_Q, dW_K, dW_V
