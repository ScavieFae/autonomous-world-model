"""Identity and quality diagnostics for the JEPA world model.

Pure functions, no external dependencies (no sklearn), GPU-resident.
Designed to run on a fixed held-out batch per epoch during training,
or post-hoc on a trained checkpoint via scripts/run_jepa_diagnostics.py.

The central concern: given that the encoder concatenates P0 and P1 features
into a dedicated-slot input and runs them through an MLP trunk, does the
trunk preserve player identity or collapse to swap-symmetric features?
See docs/jepa-data-flow.md for the full trace and motivation.

Diagnostics:
    swap_test                  — encoder collapse detector (P0↔P1 swap cosine sim)
    linear_probe_regression    — continuous probe with R² and game-units MAE
    linear_probe_classification — discrete probe (action_state) with holdout acc
    run_linear_probes          — per-player + relational + action probes
    temporal_straightness      — LeWM's emergent diagnostic on latent trajectories

All functions take a JEPAWorldModel (or its encoder) and a batch of
(float_frames, int_frames, ctrl_inputs). They do not mutate state.

Expected reading:
    swap_test.mean_cosine_sim:   LOW (< ~0.5) = healthy identity preservation
                                 HIGH (> ~0.9) = identity collapse
    swap_test.ditto_cosine_sim:  sharpest signal — dittos have no character
                                 asymmetry, so position/velocity/action are
                                 the only identity signal
    per_player probes R²:        HIGH (> 0.8) for position, percent = healthy
                                 LOW (< 0.3) = latent doesn't encode basic physics
    per_player probes mae_units: MAE in game units (pixels for x/y, pct for
                                 percent/shield). The run card success
                                 criteria are written in game-unit terms.
    relational probe R²:         HIGH for (P0.x - P1.x) = cross-player binding OK
    action probe accuracy:       random = 1/action_vocab (~0.25% on 400 classes);
                                 healthy is well above random (tens of percent)
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
#
# Three probe types:
#   1. linear_probe_regression — closed-form least squares, returns R² and
#      MAE in game units (via a per-target scale factor from EncodingConfig).
#      R² is scale-invariant so the raw fit signal is the same either way,
#      but MAE-in-game-units is what the run card's success criteria are
#      written against ("recovers position to <N game units error").
#   2. linear_probe_classification — ported from the deleted training/jepa_eval.py
#      (commit 01aaa99). 200 SGD steps on a linear head, returns holdout
#      accuracy. Used for discrete targets like action_state (400 classes).
#   3. run_linear_probes — wraps everything, computes per-player + relational +
#      action probes in a single pass on one batch.


def linear_probe_regression(
    embeddings: torch.Tensor,
    targets: torch.Tensor,
    val_frac: float = 0.2,
    scale: float = 1.0,
) -> dict[str, float]:
    """Fit a linear regression from embeddings to targets with a holdout split.

    Pure torch, closed-form least squares — no sklearn dependency. Returns
    both the scale-invariant R² on held-out samples and the MAE denormalized
    to game units via the `scale` factor.

    Args:
        embeddings: (N, D) — flattened latents
        targets:    (N,)   — continuous target in normalized encoding units
                             (e.g., `float_frames[..., 1]` for P0.x × 0.05)
        val_frac:   fraction of samples held out for R²/MAE evaluation
        scale:      EncodingConfig scale factor (e.g., `cfg.xy_scale = 0.05`).
                    MAE in game units = MAE_normalized / scale.
                    Pass 1.0 if the target is already in game units.

    Returns:
        dict with:
            "r2":        R² on the held-out split (can be negative for bad fits)
            "mae_units": MAE on the held-out split, converted to game units
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
        # a ridge-regularized fit so we still return finite metrics.
        return _ridge_regression(emb, y, train_idx, val_idx, ridge=1e-3, scale=scale)

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
        return _ridge_regression(emb, y, train_idx, val_idx, ridge=1e-3, scale=scale)

    y_pred = X_val @ w
    return _r2_and_mae(y_val, y_pred, scale)


def _ridge_regression(
    emb: torch.Tensor,
    y: torch.Tensor,
    train_idx: torch.Tensor,
    val_idx: torch.Tensor,
    ridge: float,
    scale: float,
) -> dict[str, float]:
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
    return _r2_and_mae(y_val, y_pred, scale)


def _r2_and_mae(
    y_val: torch.Tensor, y_pred: torch.Tensor, scale: float
) -> dict[str, float]:
    """Compute R² (scale-invariant) and MAE in game units from a val split."""
    ss_res = ((y_val - y_pred) ** 2).sum()
    ss_tot = ((y_val - y_val.mean()) ** 2).sum().clamp_min(1e-12)
    r2 = (1.0 - ss_res / ss_tot).item()
    mae_normalized = (y_val - y_pred).abs().mean().item()
    # scale is the EncodingConfig multiplier (x_normalized = x_game * scale).
    # To recover game units: divide normalized by scale.
    mae_units = mae_normalized / scale if scale > 0 else mae_normalized
    return {"r2": r2, "mae_units": mae_units}


def linear_probe_classification(
    embeddings: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    val_frac: float = 0.2,
    lr: float = 0.1,
    steps: int = 200,
) -> float:
    """Fit a linear classifier with internal holdout, return validation accuracy.

    Closed-form multinomial logistic isn't cheap; a few hundred SGD steps on
    a linear head is enough for a probe — we just need to know whether the
    signal is linearly decodable. Ported from training/jepa_eval.py (deleted
    in the PR #22 merge) with the split changed to internal-holdout to match
    the rest of this module.

    Args:
        embeddings: (N, D) — flattened latents
        targets:    (N,)   — long tensor of class indices
        num_classes: size of the classification head
        val_frac:   fraction of samples held out for accuracy evaluation
        lr:         SGD learning rate (matches jepa_eval.py default)
        steps:      number of SGD steps (matches jepa_eval.py default)

    Returns:
        Held-out classification accuracy in [0, 1].
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

    val_n = max(1, int(N * val_frac))
    perm = torch.randperm(N, device=device)
    val_idx = perm[:val_n]
    train_idx = perm[val_n:]

    X_tr = embeddings[train_idx].float()
    y_tr = targets[train_idx].long()
    X_va = embeddings[val_idx].float()
    y_va = targets[val_idx].long()

    W = torch.zeros(D, num_classes, device=device, requires_grad=True)
    bias = torch.zeros(num_classes, device=device, requires_grad=True)
    optim = torch.optim.SGD([W, bias], lr=lr, momentum=0.9)

    # Training loop needs grads even though this function is called under no_grad;
    # guard with enable_grad for safety.
    with torch.enable_grad():
        for _ in range(steps):
            logits = X_tr @ W + bias
            loss = F.cross_entropy(logits, y_tr)
            optim.zero_grad()
            loss.backward()
            optim.step()

    with torch.no_grad():
        val_logits = X_va @ W + bias
        pred = val_logits.argmax(dim=-1)
        acc = (pred == y_va).float().mean().item()

    return acc


def extract_probe_targets(
    float_frames: torch.Tensor,
    int_frames: torch.Tensor,
    cfg: EncodingConfig,
) -> dict[str, torch.Tensor]:
    """Extract interpretable per-player ground truth for linear probing.

    The float layout per player (from EncodingConfig):
        [percent, x, y, shield, (5 velocity dims), ...]
    So P0.percent is at index 0, P0.x at 1, P0.y at 2, shield at 3.
    P1 is offset by float_per_player.

    The int layout per player (from EncodingConfig):
        [action, jumps, character, l_cancel, hurtbox, ground, last_attack, (state_age)]
    So P0.action is at int index 0, P1.action at int index int_per_player.

    Args:
        float_frames: (B, T, 2*fp)
        int_frames:   (B, T, 2*ipp+1)
        cfg:          EncodingConfig

    Returns:
        Flat dict of named (B*T,) tensors. Continuous targets are normalized
        (use `_probe_scale(name, cfg)` to convert MAE to game units).
        Categorical targets are long tensors.
    """
    fp = cfg.float_per_player
    ipp = cfg.int_per_player

    targets = {
        # Continuous, normalized (scale via _probe_scale)
        "p0_percent": float_frames[..., 0].flatten(),
        "p0_x": float_frames[..., 1].flatten(),
        "p0_y": float_frames[..., 2].flatten(),
        "p0_shield": float_frames[..., 3].flatten(),
        "p1_percent": float_frames[..., fp + 0].flatten(),
        "p1_x": float_frames[..., fp + 1].flatten(),
        "p1_y": float_frames[..., fp + 2].flatten(),
        "p1_shield": float_frames[..., fp + 3].flatten(),
        # Relational — require cross-player binding in the latent
        "rel_x": (float_frames[..., 1] - float_frames[..., fp + 1]).flatten(),
        "rel_y": (float_frames[..., 2] - float_frames[..., fp + 2]).flatten(),
        "rel_percent": (float_frames[..., 0] - float_frames[..., fp + 0]).flatten(),
        # Categorical — long tensors for classification probes
        "p0_action": int_frames[..., 0].flatten().long(),
        "p1_action": int_frames[..., ipp].flatten().long(),
    }
    return targets


def _probe_scale(name: str, cfg: EncodingConfig) -> float:
    """Map a continuous probe target name to its EncodingConfig scale factor.

    The scale factor is the multiplier that takes game units → normalized
    encoding (e.g., `x_normalized = x_game * cfg.xy_scale`), so dividing a
    normalized MAE by the scale recovers game-unit error. Targets absent
    from this map get scale=1.0 (returned MAE stays in normalized units).
    """
    if name in ("p0_x", "p0_y", "p1_x", "p1_y", "rel_x", "rel_y"):
        return cfg.xy_scale
    if name in ("p0_percent", "p1_percent", "rel_percent"):
        return cfg.percent_scale
    if name in ("p0_shield", "p1_shield"):
        return cfg.shield_scale
    return 1.0


@dataclass
class ProbeResults:
    # Continuous probes: each value is {"r2": float, "mae_units": float}
    per_player: dict[str, dict[str, float]]
    relational: dict[str, dict[str, float]]
    # Classification probes: each value is accuracy in [0, 1]
    action: dict[str, float]

    def to_dict(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, v in self.per_player.items():
            out[f"probe/{k}_r2"] = v["r2"]
            out[f"probe/{k}_mae_units"] = v["mae_units"]
        for k, v in self.relational.items():
            out[f"probe/{k}_r2"] = v["r2"]
            out[f"probe/{k}_mae_units"] = v["mae_units"]
        for k, v in self.action.items():
            out[f"probe/{k}_acc"] = v
        return out


def linear_probe_regression_holdout(
    train_embs: torch.Tensor,
    train_y: torch.Tensor,
    val_embs: torch.Tensor,
    val_y: torch.Tensor,
    scale: float = 1.0,
    ridge: float = 1e-3,
) -> dict[str, float]:
    """Fit a linear regression on train, evaluate R² and MAE on val.

    Unlike `linear_probe_regression`, does not split internally — the caller
    provides already-disjoint train and val sets. This is the right shape
    for a real held-out evaluation: fit on one batch of val-game frames,
    evaluate on a stride-disjoint second batch from the same val distribution.

    Always uses ridge regression — avoids the lstsq-or-fallback fork from
    `linear_probe_regression` and keeps numerics stable when train_embs is
    rank-deficient (common at small batch sizes).

    Args:
        train_embs: (N_train, D) — fit embeddings
        train_y:    (N_train,)   — fit targets (normalized encoding units)
        val_embs:   (N_val, D)   — eval embeddings
        val_y:      (N_val,)     — eval targets (normalized encoding units)
        scale:      EncodingConfig scale factor for the target (see
                    `_probe_scale`); MAE is reported in game units.
        ridge:      L2 regularization — low enough to be a near-OLS fit
                    when well-conditioned, high enough to keep the solve
                    stable when not.

    Returns:
        dict with "r2" and "mae_units" on the val split.
    """
    if train_embs.ndim != 2 or val_embs.ndim != 2:
        raise ValueError(
            f"Embeddings must be 2D; got train {train_embs.shape}, val {val_embs.shape}"
        )
    if train_embs.shape[1] != val_embs.shape[1]:
        raise ValueError(
            f"Embedding dim mismatch: train {train_embs.shape[1]}, val {val_embs.shape[1]}"
        )
    if train_y.shape[0] != train_embs.shape[0] or val_y.shape[0] != val_embs.shape[0]:
        raise ValueError("Target shape must match embedding leading dim")

    device = train_embs.device
    dtype = torch.float32

    Xt = torch.cat(
        [train_embs.to(dtype), torch.ones(train_embs.shape[0], 1, device=device, dtype=dtype)],
        dim=-1,
    )
    Xv = torch.cat(
        [val_embs.to(dtype), torch.ones(val_embs.shape[0], 1, device=device, dtype=dtype)],
        dim=-1,
    )
    yt = train_y.to(dtype)
    yv = val_y.to(dtype)

    D = Xt.shape[-1]
    A = Xt.T @ Xt + ridge * torch.eye(D, device=device, dtype=dtype)
    b = Xt.T @ yt
    w = torch.linalg.solve(A, b)
    y_pred = Xv @ w
    return _r2_and_mae(yv, y_pred, scale)


def linear_probe_classification_holdout(
    train_embs: torch.Tensor,
    train_y: torch.Tensor,
    val_embs: torch.Tensor,
    val_y: torch.Tensor,
    num_classes: int,
    lr: float = 0.1,
    steps: int = 200,
) -> float:
    """Fit a linear classifier on train, return accuracy on val.

    Unlike `linear_probe_classification`, takes separate train and val
    batches — no internal split. See `linear_probe_regression_holdout` for
    the motivation.

    Returns:
        Held-out classification accuracy in [0, 1].
    """
    if train_embs.ndim != 2 or val_embs.ndim != 2:
        raise ValueError("Embeddings must be 2D")
    device = train_embs.device

    Xt = train_embs.float()
    yt = train_y.long()
    Xv = val_embs.float()
    yv = val_y.long()

    D = Xt.shape[-1]
    W = torch.zeros(D, num_classes, device=device, requires_grad=True)
    bias = torch.zeros(num_classes, device=device, requires_grad=True)
    optim = torch.optim.SGD([W, bias], lr=lr, momentum=0.9)

    with torch.enable_grad():
        for _ in range(steps):
            logits = Xt @ W + bias
            loss = F.cross_entropy(logits, yt)
            optim.zero_grad()
            loss.backward()
            optim.step()

    with torch.no_grad():
        val_logits = Xv @ W + bias
        pred = val_logits.argmax(dim=-1)
        return (pred == yv).float().mean().item()


@torch.no_grad()
def run_linear_probes_holdout(
    encoder,
    fit_batch: tuple[torch.Tensor, torch.Tensor],
    eval_batch: tuple[torch.Tensor, torch.Tensor],
    cfg: EncodingConfig,
) -> ProbeResults:
    """Run all linear probes with held-out fit and eval batches.

    This is the methodology-correct version of `run_linear_probes`. Fit
    batch and eval batch must come from disjoint samples of the val set
    (e.g., interleaved stride sampling — see `JEPATrainer.__init__`).

    Args:
        encoder:    GameStateEncoder in eval mode
        fit_batch:  (float_frames, int_frames) for probe fitting
        eval_batch: (float_frames, int_frames) for probe evaluation
                    — must be disjoint from fit_batch
        cfg:        EncodingConfig for slot sizing and probe scales

    Returns:
        ProbeResults with per_player, relational, and action dicts.
    """
    assert not encoder.training, "run_linear_probes_holdout requires encoder.eval()"

    fit_float, fit_int = fit_batch
    eval_float, eval_int = eval_batch

    fit_embs = encoder(fit_float, fit_int).flatten(0, 1)   # (N_fit, D)
    eval_embs = encoder(eval_float, eval_int).flatten(0, 1)  # (N_eval, D)

    fit_targets = extract_probe_targets(fit_float, fit_int, cfg)
    eval_targets = extract_probe_targets(eval_float, eval_int, cfg)

    per_player_keys = [
        "p0_percent", "p0_x", "p0_y", "p0_shield",
        "p1_percent", "p1_x", "p1_y", "p1_shield",
    ]
    relational_keys = ["rel_x", "rel_y", "rel_percent"]
    action_keys = ["p0_action", "p1_action"]

    per_player = {
        k: linear_probe_regression_holdout(
            fit_embs, fit_targets[k],
            eval_embs, eval_targets[k],
            scale=_probe_scale(k, cfg),
        )
        for k in per_player_keys
    }
    relational = {
        k: linear_probe_regression_holdout(
            fit_embs, fit_targets[k],
            eval_embs, eval_targets[k],
            scale=_probe_scale(k, cfg),
        )
        for k in relational_keys
    }
    action = {
        k: linear_probe_classification_holdout(
            fit_embs, fit_targets[k],
            eval_embs, eval_targets[k],
            num_classes=cfg.action_vocab,
        )
        for k in action_keys
    }

    return ProbeResults(per_player=per_player, relational=relational, action=action)


@torch.no_grad()
def run_linear_probes(
    encoder,
    float_frames: torch.Tensor,
    int_frames: torch.Tensor,
    cfg: EncodingConfig,
    val_frac: float = 0.2,
) -> ProbeResults:
    """Encode the batch and fit all linear probes.

    Per-player continuous probes (regression):
        p0_percent, p0_x, p0_y, p0_shield, p1_percent, p1_x, p1_y, p1_shield
        Each returns R² and MAE in game units.

    Relational continuous probes (regression):
        rel_x = P0.x - P1.x
        rel_y = P0.y - P1.y
        rel_percent = P0.percent - P1.percent
        Test cross-player binding — a latent that just stacks independent
        per-player slots can't fit these well.

    Action classification probes:
        p0_action, p1_action — 400-class linear classification, holdout accuracy.

    Healthy fit (expected on a working representation):
        - per-player r2 > 0.8, mae_units small (e.g., x mae_units < ~10 pixels)
        - relational r2 > 0.8
        - action accuracy > random (random = 1/400 = 0.25%); good ≳ 20-40%
          depending on how much the first frame determines action_state.

    Args:
        encoder:      GameStateEncoder in eval mode
        float_frames: (B, T, F)
        int_frames:   (B, T, I)
        cfg:          EncodingConfig (for slot sizing and probe scales)
        val_frac:     fraction held out within the batch

    Returns:
        ProbeResults dataclass with per_player, relational, and action dicts.
    """
    assert not encoder.training, "run_linear_probes requires encoder.eval()"

    embs = encoder(float_frames, int_frames)  # (B, T, D)
    flat_embs = embs.flatten(0, 1)  # (B*T, D)

    targets = extract_probe_targets(float_frames, int_frames, cfg)

    per_player_keys = [
        "p0_percent", "p0_x", "p0_y", "p0_shield",
        "p1_percent", "p1_x", "p1_y", "p1_shield",
    ]
    relational_keys = ["rel_x", "rel_y", "rel_percent"]
    action_keys = ["p0_action", "p1_action"]

    per_player = {
        k: linear_probe_regression(
            flat_embs, targets[k], val_frac, scale=_probe_scale(k, cfg),
        )
        for k in per_player_keys
    }
    relational = {
        k: linear_probe_regression(
            flat_embs, targets[k], val_frac, scale=_probe_scale(k, cfg),
        )
        for k in relational_keys
    }
    action = {
        k: linear_probe_classification(
            flat_embs, targets[k], num_classes=cfg.action_vocab, val_frac=val_frac,
        )
        for k in action_keys
    }

    return ProbeResults(per_player=per_player, relational=relational, action=action)


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


@torch.no_grad()
def run_diagnostic_suite_holdout(
    model,
    fit_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    eval_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, float]:
    """Run the full JEPA diagnostic suite with held-out probe methodology.

    This is the methodology-correct variant of `run_diagnostic_suite`:
      - Linear probes fit on `fit_batch`, evaluate on `eval_batch`
      - Swap test runs on `fit_batch` (could also run on both concatenated,
        but the swap test is cheap and doesn't benefit meaningfully from
        more samples beyond ~1K)
      - Temporal straightness runs on `fit_batch`
      - Both batches must be from the same population (e.g., stride-sampled
        slices of val_dataset) but must be disjoint sample indices

    Use this when training; `run_diagnostic_suite` still exists for the
    `scripts/run_jepa_diagnostics.py --smoke` path where only a single
    synthetic batch is available.

    Args:
        model:      JEPAWorldModel — must have .encoder and .cfg
        fit_batch:  (float_frames, int_frames, ctrl_inputs) for probe
                    fitting, swap test, and straightness
        eval_batch: (float_frames, int_frames, ctrl_inputs) for probe
                    evaluation — must be disjoint from fit_batch

    Returns:
        Flat dict of metric_name → float, ready for wandb.log().
    """
    was_training = model.training
    model.eval()
    try:
        cfg = model.cfg
        encoder = model.encoder

        metrics: dict[str, float] = {}

        fit_float, fit_int, _ = fit_batch
        eval_float, eval_int, _ = eval_batch

        # Swap test on the fit batch (cheap, no train/eval distinction needed)
        swap_result = swap_test(encoder, fit_float, fit_int, cfg)
        metrics.update(swap_result.to_dict())

        # Linear probes — the actually-held-out version
        probe_result = run_linear_probes_holdout(
            encoder, (fit_float, fit_int), (eval_float, eval_int), cfg,
        )
        metrics.update(probe_result.to_dict())

        # Temporal straightness on the fit batch
        embs = encoder(fit_float, fit_int)
        metrics["emergent/straightness"] = temporal_straightness(embs)

        return metrics
    finally:
        if was_training:
            model.train()
