"""Check that closed_form_backward matches torch.autograd."""

import torch

from reference_attention import causal_mask, closed_form_backward, torch_attention


def _random_inputs(B, T, d, d_k, seed=0):
    torch.manual_seed(seed)
    X = torch.randn(B, T, d, dtype=torch.float64, requires_grad=True)
    W_Q = torch.randn(d, d_k, dtype=torch.float64, requires_grad=True)
    W_K = torch.randn(d, d_k, dtype=torch.float64, requires_grad=True)
    W_V = torch.randn(d, d_k, dtype=torch.float64, requires_grad=True)
    return X, W_Q, W_K, W_V


def test_gradcheck_no_mask():
    # gradcheck = autograd's analytical grad vs. finite-difference numerical grad.
    # If they agree, our forward has correct gradients.
    X, W_Q, W_K, W_V = _random_inputs(2, 4, 5, 3)
    assert torch.autograd.gradcheck(torch_attention, (X, W_Q, W_K, W_V))


def test_gradcheck_with_mask():
    X, W_Q, W_K, W_V = _random_inputs(2, 4, 5, 3)
    mask = causal_mask(4).double()

    def fn(X, W_Q, W_K, W_V):
        return torch_attention(X, W_Q, W_K, W_V, mask=mask)

    assert torch.autograd.gradcheck(fn, (X, W_Q, W_K, W_V))


def test_closed_form_matches_autograd_no_mask():
    X, W_Q, W_K, W_V = _random_inputs(2, 4, 5, 3)

    Y = torch_attention(X, W_Q, W_K, W_V)
    grad_Y = torch.randn_like(Y)

    # autograd's answer
    dX, dW_Q, dW_K, dW_V = torch.autograd.grad(Y, [X, W_Q, W_K, W_V], grad_outputs=grad_Y)

    # our answer
    dX_mine, dW_Q_mine, dW_K_mine, dW_V_mine = closed_form_backward(
        grad_Y, X.detach(), W_Q.detach(), W_K.detach(), W_V.detach(),
    )

    assert torch.allclose(dX_mine, dX, atol=1e-9)
    assert torch.allclose(dW_Q_mine, dW_Q, atol=1e-9)
    assert torch.allclose(dW_K_mine, dW_K, atol=1e-9)
    assert torch.allclose(dW_V_mine, dW_V, atol=1e-9)


def test_closed_form_matches_autograd_with_mask():
    X, W_Q, W_K, W_V = _random_inputs(2, 4, 5, 3, seed=1)
    mask = causal_mask(4).double()

    Y = torch_attention(X, W_Q, W_K, W_V, mask=mask)
    grad_Y = torch.randn_like(Y)

    dX, dW_Q, dW_K, dW_V = torch.autograd.grad(Y, [X, W_Q, W_K, W_V], grad_outputs=grad_Y)

    dX_mine, dW_Q_mine, dW_K_mine, dW_V_mine = closed_form_backward(
        grad_Y, X.detach(), W_Q.detach(), W_K.detach(), W_V.detach(), mask=mask,
    )

    assert torch.allclose(dX_mine, dX, atol=1e-9)
    assert torch.allclose(dW_Q_mine, dW_Q, atol=1e-9)
    assert torch.allclose(dW_K_mine, dW_K, atol=1e-9)
    assert torch.allclose(dW_V_mine, dW_V, atol=1e-9)
