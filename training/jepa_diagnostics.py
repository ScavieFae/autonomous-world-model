"""Identity and quality diagnostics for the JEPA world model.

Pure functions, no external dependencies (no sklearn), GPU-resident.
Designed to run on a fixed held-out batch per epoch during training,
or post-hoc on a trained checkpoint via scripts/run_jepa_diagnostics.py.

The central concern: given that the encoder concatenates P0 and P1 features
into a dedicated-slot input and runs them through an MLP trunk, does the
trunk preserve player identity or collapse to swap-symmetric features?
See docs/jepa-data-flow.md for the full trace and motivation.

Diagnostics:
    swap_test              — encoder collapse detector (P0↔P1 swap cosine sim)
    per_player_probes      — linear decodability of P0/P1 physical quantities
    relational_probe       — cross-player binding (e.g., P0.x - P1.x)
    temporal_straightness  — LeWM's emergent diagnostic on latent trajectories

All functions take a JEPAWorldModel (or its encoder) and a batch of
(float_frames, int_frames, ctrl_inputs). They do not mutate state.

Expected reading:
    swap_test.mean_cosine_sim:  LOW (< ~0.5) = healthy identity preservation
                                HIGH (> ~0.9) = identity collapse
    swap_test.ditto_cosine_sim: sharpest signal — dittos have no character
                                asymmetry, so position/velocity/action are
                                the only identity signal
    per_player probes R²:        HIGH (> 0.8) for position, percent = healthy
                                LOW (< 0.3) = latent doesn't encode basic physics
    relational probe R²:         HIGH for (P0.x - P1.x) = cross-player binding OK
    temporal_straightness:       should INCREASE during training per LeWM
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from models.encoding import EncodingConfig


# ============================================================================
# Swap test — P0↔P1 identity preservation
# ============================================================================

@torch.no_grad()
def swap_p0_p1(
    float_frames: torch.Tensor,
    int_frames: torch.Tensor,
    ctrl_inputs: Optional[torch.Tensor],
    cfg: EncodingConfig,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Swap P0 and P1 in the raw input tensors.

    The float and int layouts are [P0 features || P1 features || (stage)], so
    swapping is a straightforward re-concat of halves. The resulting tensors
    represent "the same game situation with port assignments reversed."

    A healthy, identity-aware encoder should produce a *different* latent for
    swapped input, because slot-position has semantic meaning (P0 is always
    the training-time "self" perspective).

    Args:
        float_frames: (..., F) where F = 2 * cfg.float_per_player
        int_frames:   (..., I) where I = 2 * cfg.int_per_player + 1
        ctrl_inputs:  (..., C) with C = cfg.ctrl_conditioning_dim, or None
        cfg:          EncodingConfig for slot sizing

    Returns:
        (swapped_floats, swapped_ints, swapped_ctrls)
        swapped_ctrls is None if ctrl_inputs was None.
    """
    fp = cfg.float_per_player
    ipp = cfg.int_per_player

    # Floats: [P0 (fp) || P1 (fp)]  →  [P1 (fp) || P0 (fp)]
    swapped_floats = torch.cat(
        [float_frames[..., fp : 2 * fp], float_frames[..., 0:fp]],
        dim=-1,
    )

    # Ints: [P0 (ipp) || P1 (ipp) || stage (1)]  →  [P1 || P0 || stage]
    swapped_ints = torch.cat(
        [
            int_frames[..., ipp : 2 * ipp],
            int_frames[..., 0:ipp],
            int_frames[..., 2 * ipp : 2 * ipp + 1],
        ],
        dim=-1,
    )

    # Controls: [p0_ctrl (13) || p1_ctrl (13) || (p0_thresh (5) || p1_thresh (5))]
    # Matches JEPAFrameDataset._extract_ctrl packing.
    swapped_ctrls = None
    if ctrl_inputs is not None:
        cd = cfg.controller_dim  # 13
        p0_ctrl = ctrl_inputs[..., 0:cd]
        p1_ctrl = ctrl_inputs[..., cd : 2 * cd]
        parts = [p1_ctrl, p0_ctrl]
        if cfg.ctrl_threshold_features:
            # 5 analog threshold bits per player, in the same order
            p0_thresh = ctrl_inputs[..., 2 * cd : 2 * cd + 5]
            p1_thresh = ctrl_inputs[..., 2 * cd + 5 : 2 * cd + 10]
            parts.extend([p1_thresh, p0_thresh])
        swapped_ctrls = torch.cat(parts, dim=-1)

    return swapped_floats, swapped_ints, swapped_ctrls


@dataclass
class SwapTestResult:
    mean_cosine_sim: float
    ditto_cosine_sim: float  # nan if no dittos in batch
    nonditto_cosine_sim: float  # nan if no non-dittos in batch
    n_ditto: int
    n_nonditto: int

    def to_dict(self) -> dict[str, float]:
        return {
            "swap/mean_cosine_sim": self.mean_cosine_sim,
            "swap/ditto_cosine_sim": self.ditto_cosine_sim,
            "swap/nonditto_cosine_sim": self.nonditto_cosine_sim,
            "swap/n_ditto": float(self.n_ditto),
            "swap/n_nonditto": float(self.n_nonditto),
        }


@torch.no_grad()
def swap_test(
    encoder,
    float_frames: torch.Tensor,
    int_frames: torch.Tensor,
    cfg: EncodingConfig,
) -> SwapTestResult:
    """Run P0↔P1 swap test on the encoder.

    Encodes both the normal input and the P0/P1-swapped input, then measures
    cosine similarity between corresponding latents.

    Healthy: similarity is LOW (< ~0.5). Swap produces a genuinely different
    latent because port position has meaning.

    Collapsed: similarity is HIGH (> ~0.9). Encoder has learned swap-symmetric
    features and lost player identity. Dittos will show the highest collapse.

    Args:
        encoder:      GameStateEncoder (or model.encoder) in eval mode
        float_frames: (B, T, F) input
        int_frames:   (B, T, I) input
        cfg:          EncodingConfig for slot sizing and ditto detection

    Returns:
        SwapTestResult with mean, ditto, and non-ditto cosine similarities.
    """
    assert not encoder.training, (
        "swap_test requires encoder in eval mode (BatchNorm in projector "
        "would otherwise use batch stats). Call encoder.eval() first."
    )

    # Normal encoding
    normal_embs = encoder(float_frames, int_frames)  # (B, T, D)

    # Swapped encoding
    swapped_floats, swapped_ints, _ = swap_p0_p1(
        float_frames, int_frames, None, cfg
    )
    swapped_embs = encoder(swapped_floats, swapped_ints)  # (B, T, D)

    # Flatten (B, T, D) → (B*T, D) for per-frame comparisons
    n = normal_embs.flatten(0, 1)  # (B*T, D)
    s = swapped_embs.flatten(0, 1)  # (B*T, D)
    cos_sim = F.cosine_similarity(n, s, dim=-1)  # (B*T,)

    # Ditto detection: P0 and P1 have the same character ID
    # int layout per player: [action(0), jumps(1), character(2), ...]
    ipp = cfg.int_per_player
    p0_char = int_frames[..., 2]  # (B, T)
    p1_char = int_frames[..., ipp + 2]  # (B, T)
    is_ditto = (p0_char == p1_char).flatten()  # (B*T,)

    mean = cos_sim.mean().item()
    if is_ditto.any():
        ditto = cos_sim[is_ditto].mean().item()
    else:
        ditto = float("nan")
    if (~is_ditto).any():
        nonditto = cos_sim[~is_ditto].mean().item()
    else:
        nonditto = float("nan")

    return SwapTestResult(
        mean_cosine_sim=mean,
        ditto_cosine_sim=ditto,
        nonditto_cosine_sim=nonditto,
        n_ditto=int(is_ditto.sum().item()),
        n_nonditto=int((~is_ditto).sum().item()),
    )


# ============================================================================
# Linear probes — decodability of game state from the latent
# ============================================================================

def linear_probe_r2(
    embeddings: torch.Tensor,
    targets: torch.Tensor,
    val_frac: float = 0.2,
) -> float:
    """Fit a linear regression from embeddings to targets with a holdout split.

    Pure torch, closed-form least squares — no sklearn dependency.

    Args:
        embeddings: (N, D) — flattened latents
        targets:    (N,)   — continuous target (e.g., P0.x)
        val_frac:   fraction of samples held out for R² evaluation

    Returns:
        R² score on the validation split (can be negative for very bad fits).
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got {embeddings.shape}")
    if targets.ndim != 1 or targets.shape[0] != embeddings.shape[0]:
        raise ValueError(
            f"targets must be 1D matching embeddings[0]: "
            f"emb {embeddings.shape}, targets {targets.shape}"
        )

    N, D = embeddings.shape
    device = embeddings.device
    dtype = torch.float32

    emb = embeddings.to(dtype)
    y = targets.to(dtype)

    val_n = max(1, int(N * val_frac))
    perm = torch.randperm(N, device=device)
    val_idx = perm[:val_n]
    train_idx = perm[val_n:]
    if train_idx.numel() < D + 1:
        # Not enough training samples for a unique solution — fall back to
        # a ridge-regularized fit so we still return a finite R².
        return _ridge_r2(emb, y, train_idx, val_idx, ridge=1e-3)

    ones_train = torch.ones(train_idx.numel(), 1, device=device, dtype=dtype)
    ones_val = torch.ones(val_n, 1, device=device, dtype=dtype)
    X_train = torch.cat([emb[train_idx], ones_train], dim=-1)
    X_val = torch.cat([emb[val_idx], ones_val], dim=-1)
    y_train = y[train_idx]
    y_val = y[val_idx]

    # Closed-form least squares
    try:
        w = torch.linalg.lstsq(X_train, y_train.unsqueeze(-1)).solution.squeeze(-1)
    except RuntimeError:
        return _ridge_r2(emb, y, train_idx, val_idx, ridge=1e-3)

    y_pred = X_val @ w
    ss_res = ((y_val - y_pred) ** 2).sum()
    ss_tot = ((y_val - y_val.mean()) ** 2).sum().clamp_min(1e-12)
    r2 = (1.0 - ss_res / ss_tot).item()
    return r2


def _ridge_r2(
    emb: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    ridge: float,
) -> float:
    """Ridge-regularized least squares fallback when samples < features."""
    device = emb.device
    dtype = emb.dtype
    X_train = torch.cat(
        [emb[train_idx], torch.ones(train_idx.numel(), 1, device=device, dtype=dtype)],
        dim=-1,
    )
    X_val = torch.cat(
        [emb[val_idx], torch.ones(val_idx.numel(), 1, device=device, dtype=dtype)],
        dim=-1,
    )
    y_train = y[train_idx]
    y_val = y[val_idx]
    D = X_train.shape[-1]
    A = X_train.T @ X_train + ridge * torch.eye(D, device=device, dtype=dtype)
    b = X_train.T @ y_train
    w = torch.linalg.solve(A, b)
    y_pred = X_val @ w
    ss_res = ((y_val - y_pred) ** 2).sum()
    ss_tot = ((y_val - y_val.mean()) ** 2).sum().clamp_min(1e-12)
    return (1.0 - ss_res / ss_tot).item()


def extract_probe_targets(
    float_frames: torch.Tensor,
    int_frames: torch.Tensor,
    cfg: EncodingConfig,
) -> dict[str, torch.Tensor]:
    """Extract interpretable per-player ground truth for linear probing.

    The float layout per player (from EncodingConfig):
        [percent, x, y, shield, (5 velocity dims), ...]

    So P0.percent is at index 0, P0.x at 1, P0.y at 2, etc.
    P1 is offset by float_per_player.

    Args:
        float_frames: (B, T, 2*fp)
        int_frames:   (B, T, 2*ipp+1)
        cfg:          EncodingConfig

    Returns:
        Flat dict of named (B*T,) tensors ready for linear_probe_r2.
    """
    fp = cfg.float_per_player
    ipp = cfg.int_per_player

    # Core continuous layout: percent=0, x=1, y=2, shield=3
    targets = {
        "p0_percent": float_frames[..., 0].flatten(),
        "p0_x": float_frames[..., 1].flatten(),
        "p0_y": float_frames[..., 2].flatten(),
        "p0_shield": float_frames[..., 3].flatten(),
        "p1_percent": float_frames[..., fp + 0].flatten(),
        "p1_x": float_frames[..., fp + 1].flatten(),
        "p1_y": float_frames[..., fp + 2].flatten(),
        "p1_shield": float_frames[..., fp + 3].flatten(),
        # Relational targets — require cross-player binding in the latent
        "rel_x": (float_frames[..., 1] - float_frames[..., fp + 1]).flatten(),
        "rel_y": (float_frames[..., 2] - float_frames[..., fp + 2]).flatten(),
        "rel_percent": (float_frames[..., 0] - float_frames[..., fp + 0]).flatten(),
    }
    return targets


@dataclass
class ProbeResults:
    per_player: dict[str, float]
    relational: dict[str, float]

    def to_dict(self) -> dict[str, float]:
        out = {}
        for k, v in self.per_player.items():
            out[f"probe/{k}_r2"] = v
        for k, v in self.relational.items():
            out[f"probe/{k}_r2"] = v
        return out


@torch.no_grad()
def run_linear_probes(
    encoder,
    float_frames: torch.Tensor,
    int_frames: torch.Tensor,
    cfg: EncodingConfig,
    val_frac: float = 0.2,
) -> ProbeResults:
    """Encode the batch and fit linear probes for per-player and relational targets.

    Per-player probes (separate P0 and P1):
        percent, x, y, shield

    Relational probes (cross-player binding):
        rel_x   = P0.x - P1.x
        rel_y   = P0.y - P1.y
        rel_percent = P0.percent - P1.percent

    A healthy representation fits all per-player probes with high R² (> 0.8
    expected for positions and percent, which are directly encoded inputs).
    Relational probes test whether the latent encodes cross-player structure
    rather than just stacking independent per-player slots.

    Args:
        encoder:      GameStateEncoder in eval mode
        float_frames: (B, T, F)
        int_frames:   (B, T, I)
        cfg:          EncodingConfig
        val_frac:     fraction held out for R²

    Returns:
        ProbeResults with per_player and relational R² dicts.
    """
    assert not encoder.training, "run_linear_probes requires encoder.eval()"

    embs = encoder(float_frames, int_frames)  # (B, T, D)
    flat_embs = embs.flatten(0, 1)  # (B*T, D)

    targets = extract_probe_targets(float_frames, int_frames, cfg)

    per_player_keys = ["p0_percent", "p0_x", "p0_y", "p0_shield",
                       "p1_percent", "p1_x", "p1_y", "p1_shield"]
    relational_keys = ["rel_x", "rel_y", "rel_percent"]

    per_player = {k: linear_probe_r2(flat_embs, targets[k], val_frac)
                  for k in per_player_keys}
    relational = {k: linear_probe_r2(flat_embs, targets[k], val_frac)
                  for k in relational_keys}

    return ProbeResults(per_player=per_player, relational=relational)


# ============================================================================
# Temporal straightness — LeWM's emergent diagnostic
# ============================================================================

@torch.no_grad()
def temporal_straightness(embeddings: torch.Tensor) -> float:
    """Cosine similarity of consecutive latent velocity vectors.

    LeWM (arXiv 2603.19312) reports that latent trajectories become
    increasingly "straight" during training — consecutive velocity vectors
    point in similar directions — as an emergent property with no explicit
    regularization. Increasing straightness over training epochs is a
    qualitative health signal for the predictor.

    Args:
        embeddings: (B, T, D) with T >= 3

    Returns:
        Mean cosine similarity of consecutive velocity vectors.
        NaN if T < 3.
    """
    if embeddings.shape[1] < 3:
        return float("nan")

    # Velocity: first difference along time
    vel = embeddings[:, 1:] - embeddings[:, :-1]  # (B, T-1, D)
    # Consecutive velocity pairs
    v1 = vel[:, :-1]  # (B, T-2, D)
    v2 = vel[:, 1:]   # (B, T-2, D)
    sim = F.cosine_similarity(v1, v2, dim=-1)  # (B, T-2)
    return sim.mean().item()


# ============================================================================
# Top-level: run the whole diagnostic suite
# ============================================================================

@torch.no_grad()
def run_diagnostic_suite(
    model,
    float_frames: torch.Tensor,
    int_frames: torch.Tensor,
    ctrl_inputs: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    """Run the full JEPA diagnostic suite on a batch.

    Intended to be called once per epoch on a fixed held-out batch.
    Returns a flat dict of metrics suitable for wandb logging.

    Runs:
        - swap_test       (swap/mean, swap/ditto, swap/nonditto)
        - run_linear_probes (probe/p0_x_r2, probe/rel_x_r2, ...)
        - temporal_straightness (emergent/straightness)

    Args:
        model: JEPAWorldModel — must have .encoder and .cfg
        float_frames: (B, T, F)
        int_frames:   (B, T, I)
        ctrl_inputs:  (B, T, C) or None — currently unused by suite,
                      reserved for future predictor-level diagnostics

    Returns:
        Flat dict of metric_name → float, ready for wandb.log().
    """
    was_training = model.training
    model.eval()
    try:
        cfg = model.cfg
        encoder = model.encoder

        metrics: dict[str, float] = {}

        # Swap test (encoder-level)
        swap_result = swap_test(encoder, float_frames, int_frames, cfg)
        metrics.update(swap_result.to_dict())

        # Linear probes (encoder-level)
        probe_result = run_linear_probes(encoder, float_frames, int_frames, cfg)
        metrics.update(probe_result.to_dict())

        # Temporal straightness on the encoded sequence
        embs = encoder(float_frames, int_frames)
        metrics["emergent/straightness"] = temporal_straightness(embs)

        return metrics
    finally:
        if was_training:
            model.train()
