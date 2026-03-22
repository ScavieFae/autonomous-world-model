"""Muon optimizer — Newton-Schulz orthogonalized SGD for weight matrices.

Reference: https://github.com/KellerJordan/Muon

Uses Muon (orthogonalized momentum SGD) for 2D+ weight parameters,
falls back to an internal AdamW instance for embeddings, biases, and
1D parameters. The Newton-Schulz iteration approximates the matrix
square root inverse of G@G^T in 3 steps, giving an orthogonalized
gradient update that improves training stability and convergence.
"""

import torch
from torch.optim import Optimizer


class Muon(Optimizer):
    """Muon optimizer — Newton-Schulz orthogonalized SGD for weight matrices.

    Uses Muon for 2D+ weight parameters, falls back to internal AdamW
    for embeddings, biases, and 1D parameters.

    Args:
        muon_params: Parameters to optimize with Muon (should be 2D+ weight matrices).
        adamw_params: Parameters to optimize with AdamW (embeddings, biases, 1D).
        lr: Learning rate for Muon params.
        momentum: Momentum coefficient for Muon (default 0.95).
        adamw_lr: Learning rate for AdamW params.
        adamw_betas: Beta coefficients for AdamW.
        adamw_wd: Weight decay for AdamW.
    """

    def __init__(
        self,
        muon_params,
        adamw_params,
        lr: float = 0.02,
        momentum: float = 0.95,
        adamw_lr: float = 3e-4,
        adamw_betas: tuple[float, float] = (0.9, 0.999),
        adamw_wd: float = 1e-5,
    ):
        # Materialize iterators
        muon_params = list(muon_params)
        adamw_params = list(adamw_params)

        defaults = dict(lr=lr, momentum=momentum)
        all_params = muon_params + adamw_params
        super().__init__(all_params, defaults)

        # Track which params get Muon updates
        self._muon_params = set(id(p) for p in muon_params)

        # Internal AdamW for non-Muon params
        self._adamw = torch.optim.AdamW(
            adamw_params, lr=adamw_lr, betas=adamw_betas, weight_decay=adamw_wd,
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]

            for p in group["params"]:
                if p.grad is None or id(p) not in self._muon_params:
                    continue

                g = p.grad
                if g.ndim < 2:
                    continue

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                # Newton-Schulz orthogonalization (3 iterations)
                g_orth = buf
                if g_orth.ndim > 2:
                    # Reshape to 2D for NS iteration
                    shape = g_orth.shape
                    g_orth = g_orth.reshape(g_orth.shape[0], -1)

                X = g_orth / (g_orth.norm() + 1e-7)
                for _ in range(3):
                    A = X @ X.T
                    B = (3.0 * torch.eye(A.shape[0], device=A.device, dtype=A.dtype) - A) / 2.0
                    X = B @ X

                if buf.ndim > 2:
                    X = X.reshape(shape)

                p.add_(X, alpha=-lr)

        # Step AdamW for non-muon params
        self._adamw.step()

        return loss
