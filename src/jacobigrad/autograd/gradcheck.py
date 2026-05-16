"""Numerical gradient checker for ``Tensor`` autograd.

The non-negotiable Week 2 deliverable per PLAN.md: nothing downstream
(matmul, attention, decoder) should be considered "done" until its backward
matches central differences on a randomized test.

This is a generic harness — pass a closure that takes a list of Tensor
parameters and returns a scalar Tensor loss. The checker:

  1. Computes analytic grads via ``loss.backward()``.
  2. For each scalar element of each parameter, perturbs by ``+/- eps``,
     re-evaluates the closure, and computes a central-difference estimate.
  3. Compares the two element-wise.

Cost is O(sum(p.size) * 2 * forward_cost), so it's only practical on small
randomized inputs. That's the point — it catches bugs in the *math*, not
performance issues.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from jacobigrad.autograd.tensor import Tensor


def numgrad_check(
    fn: Callable[[list[Tensor]], Tensor],
    params: list[Tensor],
    *,
    eps: float = 1e-6,
    atol: float = 1e-6,
    rtol: float = 1e-4,
) -> None:
    """Verify ``fn``'s analytic gradients against central differences.

    Parameters
    ----------
    fn : closure
        Takes the params list and returns a scalar Tensor (the "loss").
        Must rebuild the graph from ``param.data`` each call — it will be
        invoked many times with mutated ``param.data`` values.
    params : list[Tensor]
        Parameters whose gradients are checked. Their ``.grad`` is reset
        before the analytic pass and restored after.

    Raises
    ------
    AssertionError if any element's analytic grad differs from central-diff
    by more than ``max(atol, rtol * |numerical|)``.
    """
    # Analytic pass.
    for p in params:
        p.zero_grad()
    loss = fn(params)
    if loss.data.ndim != 0:
        raise ValueError(f"fn must return a scalar Tensor; got shape {loss.shape}")
    loss.backward()
    analytic = [
        p.grad.copy() if p.grad is not None else np.zeros_like(p.data) for p in params
    ]

    # Numerical pass.
    for p_idx, p in enumerate(params):
        numerical = np.zeros_like(p.data)
        for idx in np.ndindex(p.data.shape):
            orig = float(p.data[idx])
            p.data[idx] = orig + eps
            l_plus = float(fn(params).data)
            p.data[idx] = orig - eps
            l_minus = float(fn(params).data)
            p.data[idx] = orig
            numerical[idx] = (l_plus - l_minus) / (2.0 * eps)

        a = analytic[p_idx]
        diff = np.abs(a - numerical)
        tol = atol + rtol * np.abs(numerical)
        if not np.all(diff <= tol):
            worst = np.unravel_index(np.argmax(diff - tol), diff.shape)
            raise AssertionError(
                f"gradcheck failed for param {p_idx} at index {worst}: "
                f"analytic={a[worst]:.6e}, numerical={numerical[worst]:.6e}, "
                f"diff={diff[worst]:.3e}, tol={tol[worst]:.3e}"
            )
