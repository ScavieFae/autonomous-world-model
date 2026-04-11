"""Sketch Isotropic Gaussian Regularizer (SIGReg).

Ported from lucas-maes/le-wm (MIT license).
Paper: LeJEPA, arXiv 2511.08544 (Balestriero & LeCun).

Enforces that the embedding distribution matches an isotropic Gaussian
via the Epps-Pulley characteristic function test on random projections.
Uses Cramer-Wold theorem: matching all 1D marginals matches the full joint.
"""

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer.

    Ported directly from le-wm/module.py (MIT license).
    Single-GPU implementation — no cross-GPU gather needed.

    Args:
        knots: Number of quadrature points for the CF test (default 17).
        num_proj: Number of random projection directions (default 1024).
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """Compute SIGReg loss.

        Args:
            proj: (T, B, D) — embeddings. T=time, B=batch, D=embed_dim.
                  Note: time-first ordering. Transpose from (B, T, D) before calling.

        Returns:
            Scalar loss — lower means closer to isotropic Gaussian.
        """
        # Sample random projections (unit norm)
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))

        # Compute the Epps-Pulley statistic
        # Project embeddings onto random directions, evaluate CF at knot points
        x_t = (proj @ A).unsqueeze(-1) * self.t  # (..., num_proj, knots)

        # Compare empirical CF to standard Gaussian CF
        # E[cos(tX)] should equal exp(-t^2/2), E[sin(tX)] should equal 0
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()

        # Weighted integration (trapezoidal rule with Gaussian window)
        statistic = (err @ self.weights) * proj.size(-2)

        return statistic.mean()  # average over projections and time
