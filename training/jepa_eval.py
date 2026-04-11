"""Per-epoch diagnostics for the JEPA world model.

Three cheap, held-out signals that give us real go/no-go information
before a decoder exists:

1. **Linear probe** — fit a closed-form linear layer from frozen encoder
   embeddings to game state targets (position, percent, action_state).
   If a linear probe can't recover basic physics from the latent, JEPA
   has not learned a usable representation regardless of pred_loss.

2. **Temporal straightness** — cosine similarity of consecutive latent
   velocity vectors. LeWM reports this as an emergent diagnostic: a
   well-trained JEPA latent space shows increasingly straight temporal
   trajectories over training, without any explicit smoothness loss.

3. **Embedding statistics** — mean/std/rank of the embedding distribution.
   Cheap sanity check that SIGReg is doing its job.

All three run on held-out games, cheap enough to log per epoch.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F

from data.jepa_dataset import JEPAFrameDataset
from models.encoding import EncodingConfig
from models.jepa.model import JEPAWorldModel

logger = logging.getLogger(__name__)


@torch.no_grad()
def collect_embeddings(
    model: JEPAWorldModel,
    dataset: JEPAFrameDataset,
    cfg: EncodingConfig,
    num_samples: int,
    device: str,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Encode a batch of held-out frames and return embeddings + targets.

    Returns a dict with:
        embeddings: (N, D) — first-frame encoder output per sample
        pos_target: (N, 4) — [p0_x, p0_y, p1_x, p1_y] in game units
        percent_target: (N, 2) — [p0_percent, p1_percent] in game units
        action_target: (N, 2) — [p0_action, p1_action] as long tensors
    """
    model.eval()
    rng = np.random.RandomState(seed)
    n_total = len(dataset)
    n = min(num_samples, n_total)
    indices = rng.choice(n_total, size=n, replace=False)

    float_frames, int_frames, ctrl_inputs = dataset.get_batch(indices)
    float_frames = float_frames.to(device)
    int_frames = int_frames.to(device)

    # Encode the first frame of each subsequence (B, F) → (B, D)
    embs = model.encoder(float_frames[:, :1], int_frames[:, :1])[:, 0]

    # Targets from the first frame (denormalized)
    fp = cfg.float_per_player
    ipp = cfg.int_per_player
    first_float = float_frames[:, 0].cpu()   # (B, F)
    first_int = int_frames[:, 0].cpu()       # (B, I)

    pos_target = torch.stack([
        first_float[:, 1] / cfg.xy_scale,        # p0_x
        first_float[:, 2] / cfg.xy_scale,        # p0_y
        first_float[:, fp + 1] / cfg.xy_scale,   # p1_x
        first_float[:, fp + 2] / cfg.xy_scale,   # p1_y
    ], dim=1)

    percent_target = torch.stack([
        first_float[:, 0] / cfg.percent_scale,        # p0_percent
        first_float[:, fp] / cfg.percent_scale,       # p1_percent
    ], dim=1)

    action_target = torch.stack([
        first_int[:, 0],          # p0_action
        first_int[:, ipp],        # p1_action
    ], dim=1)

    return {
        "embeddings": embs.cpu(),
        "pos_target": pos_target,
        "percent_target": percent_target,
        "action_target": action_target,
    }


def linear_probe_regression(
    embeddings_train: torch.Tensor,
    targets_train: torch.Tensor,
    embeddings_val: torch.Tensor,
    targets_val: torch.Tensor,
    ridge: float = 1e-3,
) -> dict[str, float]:
    """Closed-form ridge regression linear probe.

    Fits w = (X^T X + λI)^-1 X^T y on train, evaluates R^2 and MAE on val.

    Args:
        embeddings_*: (N, D) — latent vectors
        targets_*: (N, T) — continuous target values (denormalized)
        ridge: L2 regularization strength

    Returns:
        dict with 'r2' (averaged over target dims) and 'mae' (per-dim mean)
    """
    X_tr = embeddings_train.float()
    y_tr = targets_train.float()
    X_va = embeddings_val.float()
    y_va = targets_val.float()

    # Add bias column
    X_tr_b = torch.cat([X_tr, torch.ones(X_tr.shape[0], 1)], dim=1)
    X_va_b = torch.cat([X_va, torch.ones(X_va.shape[0], 1)], dim=1)

    D = X_tr_b.shape[1]
    A = X_tr_b.T @ X_tr_b + ridge * torch.eye(D)
    b = X_tr_b.T @ y_tr
    w = torch.linalg.solve(A, b)  # (D+1, T)

    y_pred = X_va_b @ w  # (N, T)

    # MAE per target dim, averaged
    mae = (y_pred - y_va).abs().mean().item()

    # R^2: 1 - SS_res / SS_tot
    ss_res = ((y_pred - y_va) ** 2).sum(dim=0)
    ss_tot = ((y_va - y_va.mean(dim=0, keepdim=True)) ** 2).sum(dim=0).clamp(min=1e-8)
    r2 = (1 - ss_res / ss_tot).mean().item()

    return {"r2": r2, "mae": mae}


def linear_probe_classification(
    embeddings_train: torch.Tensor,
    targets_train: torch.Tensor,
    embeddings_val: torch.Tensor,
    targets_val: torch.Tensor,
    num_classes: int,
    lr: float = 0.1,
    steps: int = 200,
) -> dict[str, float]:
    """Linear classifier trained with a few SGD steps.

    Closed-form multinomial logistic isn't cheap; a few hundred SGD steps
    on a linear layer is fine for a probe — we just need to know if the
    signal is linearly decodable.

    Targets are (N, T) — we fit one head per column and average accuracy.

    Returns:
        dict with 'acc' averaged over target dims
    """
    X_tr = embeddings_train.float()
    y_tr = targets_train.long()
    X_va = embeddings_val.float()
    y_va = targets_val.long()

    D = X_tr.shape[1]
    T = y_tr.shape[1]

    accs = []
    for col in range(T):
        W = torch.zeros(D, num_classes, requires_grad=True)
        b = torch.zeros(num_classes, requires_grad=True)
        optim = torch.optim.SGD([W, b], lr=lr, momentum=0.9)
        y_col = y_tr[:, col]
        for _ in range(steps):
            logits = X_tr @ W + b
            loss = F.cross_entropy(logits, y_col)
            optim.zero_grad()
            loss.backward()
            optim.step()
        with torch.no_grad():
            val_logits = X_va @ W + b
            pred = val_logits.argmax(dim=-1)
            acc = (pred == y_va[:, col]).float().mean().item()
            accs.append(acc)

    return {"acc": float(np.mean(accs))}


@torch.no_grad()
def temporal_straightness(
    model: JEPAWorldModel,
    dataset: JEPAFrameDataset,
    cfg: EncodingConfig,
    num_trajectories: int,
    traj_length: int,
    device: str,
    seed: int = 42,
) -> dict[str, float]:
    """Cosine similarity of consecutive latent velocity vectors.

    Emergent diagnostic from LeWM — a well-trained JEPA latent space
    shows increasingly straight temporal trajectories over training,
    without any explicit smoothness loss. This is free to compute.

    Samples random starts from the dataset, extends each into a
    traj_length contiguous frame window, encodes them, and measures
    avg(cos(z_{t+1} - z_t, z_{t+2} - z_{t+1})).

    Returns:
        dict with 'cosine' (mean straightness)
    """
    model.eval()
    rng = np.random.RandomState(seed)

    raw = dataset.data  # MeleeDataset with contiguous floats/ints tensors
    offsets = raw.game_offsets
    # Pick games in the val range and sample start frames
    valid_starts = []
    for gi in range(len(offsets) - 1):
        gs = offsets[gi]
        ge = offsets[gi + 1]
        if ge - gs > traj_length + 1:
            valid_starts.extend(range(gs, ge - traj_length - 1))
    if not valid_starts:
        return {"cosine": float("nan")}

    n = min(num_trajectories, len(valid_starts))
    starts = rng.choice(valid_starts, size=n, replace=False)

    # Build (B, traj_length, ...) batch
    offsets_arr = np.arange(traj_length)
    frame_indices = starts[:, None] + offsets_arr[None, :]
    float_frames = raw.floats[frame_indices].to(device)
    int_frames = raw.ints[frame_indices].to(device)

    # Encode all frames in all trajectories
    embs = model.encoder(float_frames, int_frames)  # (B, T, D)

    # Velocity vectors: v_t = z_{t+1} - z_t
    velocities = embs[:, 1:] - embs[:, :-1]  # (B, T-1, D)

    # Cosine similarity between consecutive velocities
    v_curr = velocities[:, :-1]  # (B, T-2, D)
    v_next = velocities[:, 1:]   # (B, T-2, D)
    cos = F.cosine_similarity(v_curr, v_next, dim=-1)  # (B, T-2)

    return {"cosine": cos.mean().item()}


@torch.no_grad()
def embedding_stats(
    model: JEPAWorldModel,
    dataset: JEPAFrameDataset,
    num_samples: int,
    device: str,
    seed: int = 42,
) -> dict[str, float]:
    """Cheap sanity check on the embedding distribution.

    If SIGReg is doing its job, embeddings should be approximately
    isotropic Gaussian: mean≈0, std≈1 per dim, effective rank ≈ D.

    Returns:
        dict with 'mean_abs', 'std', 'rank_frac' (singular values above 0.01)
    """
    model.eval()
    rng = np.random.RandomState(seed)
    n_total = len(dataset)
    n = min(num_samples, n_total)
    indices = rng.choice(n_total, size=n, replace=False)

    float_frames, int_frames, _ = dataset.get_batch(indices)
    float_frames = float_frames.to(device)
    int_frames = int_frames.to(device)

    embs = model.encoder(float_frames, int_frames).reshape(-1, model.embed_dim)
    mean_abs = embs.mean(dim=0).abs().mean().item()
    std = embs.std(dim=0).mean().item()

    # Effective rank via singular values (fraction above 1% of max)
    try:
        s = torch.linalg.svdvals(embs.float().cpu())
        rank_frac = float((s > 0.01 * s.max()).float().mean().item())
    except Exception:
        rank_frac = float("nan")

    return {
        "mean_abs": mean_abs,
        "std": std,
        "rank_frac": rank_frac,
    }


def run_jepa_eval(
    model: JEPAWorldModel,
    train_dataset: JEPAFrameDataset,
    val_dataset: JEPAFrameDataset,
    cfg: EncodingConfig,
    device: str,
    probe_samples: int = 1024,
    straightness_trajectories: int = 128,
    straightness_length: int = 20,
    seed: int = 42,
) -> dict[str, float]:
    """Run all JEPA per-epoch diagnostics. Returns a flat metrics dict.

    Designed to be called from JEPATrainer._val_epoch. Everything runs on
    held-out val data except the linear probe, which needs a train split
    for fitting and a val split for evaluation.
    """
    was_training = model.training
    model.eval()
    try:
        # Collect embeddings for linear probes
        train_data = collect_embeddings(
            model, train_dataset, cfg, probe_samples, device, seed=seed,
        )
        val_data = collect_embeddings(
            model, val_dataset, cfg, probe_samples, device, seed=seed + 1,
        )

        pos_metrics = linear_probe_regression(
            train_data["embeddings"], train_data["pos_target"],
            val_data["embeddings"], val_data["pos_target"],
        )
        pct_metrics = linear_probe_regression(
            train_data["embeddings"], train_data["percent_target"],
            val_data["embeddings"], val_data["percent_target"],
        )
        action_metrics = linear_probe_classification(
            train_data["embeddings"], train_data["action_target"],
            val_data["embeddings"], val_data["action_target"],
            num_classes=cfg.action_vocab,
        )

        straightness = temporal_straightness(
            model, val_dataset, cfg,
            num_trajectories=straightness_trajectories,
            traj_length=straightness_length,
            device=device, seed=seed,
        )

        stats = embedding_stats(
            model, val_dataset, probe_samples, device, seed=seed,
        )
    finally:
        if was_training:
            model.train()

    return {
        "probe/pos_r2": pos_metrics["r2"],
        "probe/pos_mae": pos_metrics["mae"],
        "probe/percent_r2": pct_metrics["r2"],
        "probe/percent_mae": pct_metrics["mae"],
        "probe/action_acc": action_metrics["acc"],
        "straightness/cosine": straightness["cosine"],
        "emb/mean_abs": stats["mean_abs"],
        "emb/std": stats["std"],
        "emb/rank_frac": stats["rank_frac"],
    }
